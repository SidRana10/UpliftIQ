"""Criteo C1: is treatment independent of the pre-treatment features?

All statistics use TRAIN rows only. These are diagnostics of the assumptions behind sample-level
propensity-based evaluation (constant propensity, no confounding by f0-f11, positivity); they
cannot prove causal identifiability, and they cannot see selection created by the hidden,
non-uniform subsampling. A failed check is reported, never "fixed" by estimating propensities.

The constant 0.85 propensity is a publisher-supported assumption: Diemert et al. (arXiv:2111.10106,
Section 3, "Considerations to concatenate data from different tests") state that CRITEO-UPLIFTv2
subsamples all source experiments to one global treatment ratio of 85%, and Section 5.1 reports
treatment-independence validation. The released file has no test identifier, so the assumption
cannot be verified independently here. The row-order structure (A1d) is therefore an ordering
warning, not evidence of heterogeneous propensities, and does not by itself fail the verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from upliftiq.data.criteo import FEATURES  # noqa: E402
from upliftiq.data.criteo_prepare import (  # noqa: E402
    PROCESSED_DIR,
    REPORT_DIR,
    SEED,
    fit_scaler,
    load_arrays,
    load_split,
)
from upliftiq.evaluation.stats import wilson_interval  # noqa: E402

# Pre-specified thresholds: (pass_if_at_most, warn_if_at_most); above warn => fail.
THRESHOLDS = {
    "max_abs_smd": (0.05, 0.10),
    "auc_gap": (0.01, 0.05),  # held-out AUC minus 0.5
    "max_bin_deviation": (0.01, 0.03),  # |treated share in a bin - overall share|, bins n >= 1000
    "max_block_deviation": (
        0.01,
        0.03,
    ),  # same, across contiguous row-order blocks (ordering warning)
}
POSITIVITY_RANGE = (0.05, 0.95)
MIN_BIN_N = 1000
N_BINS = 20
N_BLOCKS = 20
DIAG_ROWS = 1_000_000  # rows per diagnostic fit / eval set (disjoint, both from train)


def _grade(value: float, key: str) -> str:
    ok, warn = THRESHOLDS[key]
    return "pass" if value <= ok else "warn" if value <= warn else "fail"


def smd_table(X: np.ndarray, treatment: np.ndarray, idx: np.ndarray) -> pd.DataFrame:
    """Standardized mean differences (treated - control) per feature, on the given rows."""
    t = treatment[idx] == 1
    rows = []
    for j, f in enumerate(FEATURES):
        x = np.asarray(X[:, j])[idx].astype(np.float64)
        a, b = x[t], x[~t]
        sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        rows.append(
            {
                "feature": f,
                "mean_treated": a.mean(),
                "mean_control": b.mean(),
                "smd": (a.mean() - b.mean()) / sd if sd > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def ks_table(X, treatment, idx, n: int = 400_000, seed: int = SEED) -> pd.DataFrame:
    """Two-sample KS distance treated vs control on a seeded subsample (whole-distribution check)."""
    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False))
    t = treatment[sub] == 1
    rows = []
    for j, f in enumerate(FEATURES):
        x = np.asarray(X[:, j])[sub]
        rows.append({"feature": f, "ks": float(stats.ks_2samp(x[t], x[~t]).statistic)})
    return pd.DataFrame(rows)


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)) / (
        n_pos * n_neg
    )
    return float(np.sqrt(max(var, 0.0)))


def diagnostic_split(treatment, train_idx, n: int = DIAG_ROWS, seed: int = SEED):
    """Two disjoint, treatment-stratified row sets drawn from TRAIN (fit / evaluate)."""
    rng = np.random.default_rng(seed)
    fit, ev = [], []
    for t in (0, 1):
        rows = rng.permutation(train_idx[np.asarray(treatment)[train_idx] == t])
        k = round(n * len(rows) / len(train_idx))
        k = min(k, len(rows) // 2)
        fit.append(rows[:k])
        ev.append(rows[k : 2 * k])
    return np.sort(np.concatenate(fit)), np.sort(np.concatenate(ev))


def treatment_classifier_diagnostic(X, treatment, train_idx, scaler, seed: int = SEED) -> dict:
    """Predict treatment from f0-f11: fit on one train-derived set, AUC on a disjoint one."""
    fit_idx, ev_idx = diagnostic_split(treatment, train_idx, seed=seed)
    assert not set(fit_idx) & set(ev_idx) and set(fit_idx) | set(ev_idx) <= set(train_idx)
    yf, ye = np.asarray(treatment)[fit_idx], np.asarray(treatment)[ev_idx]
    Xf, Xe = np.asarray(X[fit_idx]), np.asarray(X[ev_idx])
    gbm = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        num_leaves=31,
        min_child_samples=200,
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    gbm.fit(Xf, yf)
    p_gbm = gbm.predict_proba(Xe)[:, 1]
    logit = LogisticRegression(max_iter=500).fit(scaler.transform(X, fit_idx), yf)
    p_log = logit.predict_proba(scaler.transform(X, ev_idx))[:, 1]
    n_pos, n_neg = int(ye.sum()), int((1 - ye).sum())
    out = {"n_fit": len(fit_idx), "n_eval": len(ev_idx), "base_rate_eval": float(ye.mean())}
    rng = np.random.default_rng(seed)
    for name, p in (("lightgbm", p_gbm), ("logistic", p_log)):
        auc = float(roc_auc_score(ye, p))
        out[name] = {
            "auc": auc,
            "auc_se": hanley_mcneil_se(auc, n_pos, n_neg),
            "grade": _grade(auc - 0.5, "auc_gap"),
        }
    out["permuted_label_auc"] = float(roc_auc_score(rng.permutation(ye), p_gbm))
    imp = pd.Series(gbm.booster_.feature_importance("gain"), index=FEATURES)
    out["feature_importance_gain_share"] = (imp / imp.sum()).round(4).to_dict()
    lo, hi = POSITIVITY_RANGE
    out["predicted_propensity"] = {
        "min": float(p_gbm.min()),
        "q01": float(np.quantile(p_gbm, 0.01)),
        "median": float(np.median(p_gbm)),
        "q99": float(np.quantile(p_gbm, 0.99)),
        "max": float(p_gbm.max()),
        "share_outside_0.05_0.95": float(np.mean((p_gbm < lo) | (p_gbm > hi))),
    }
    return out


def bin_proportions(X, treatment, idx, n_bins: int = N_BINS) -> pd.DataFrame:
    """Treated share across quantile bins of each feature, with Wilson 95% intervals."""
    overall = float(np.asarray(treatment)[idx].mean())
    t_all = np.asarray(treatment)[idx]
    rows = []
    for j, f in enumerate(FEATURES):
        x = np.asarray(X[:, j])[idx]
        edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
        b = np.digitize(x, edges[1:-1], right=True)
        for k in np.unique(b):
            m = b == k
            n = int(m.sum())
            share = float(t_all[m].mean())
            lo, hi = wilson_interval(int(t_all[m].sum()), n)
            rows.append(
                {
                    "feature": f,
                    "bin": int(k),
                    "n": n,
                    "treated_share": share,
                    "ci_low": lo,
                    "ci_high": hi,
                    "deviation": share - overall,
                }
            )
    return pd.DataFrame(rows)


def block_proportions(treatment, idx, n_blocks: int = N_BLOCKS) -> pd.DataFrame:
    """Treated share in contiguous blocks of the original row order (hidden-structure probe)."""
    t = np.asarray(treatment)[idx]
    overall = float(t.mean())
    rows = []
    for k, part in enumerate(np.array_split(np.arange(len(idx)), n_blocks)):
        share = float(t[part].mean())
        rows.append(
            {"block": k, "n": len(part), "treated_share": share, "deviation": share - overall}
        )
    return pd.DataFrame(rows)


def run_structure(X, treatment, idx, max_runs: int = 50) -> dict:
    """Contiguous single-treatment runs in row order, with per-run feature means.

    A randomized, shuffled file has ~n/4 short runs. A handful of long runs means the file is sorted
    or concatenated from sub-datasets (hidden tests/advertisers) with different treated shares.
    """
    t = np.asarray(treatment)[idx].astype(int)
    starts = np.r_[0, np.flatnonzero(np.diff(t)) + 1]
    ends = np.r_[starts[1:], len(t)]
    out = {"n_runs": int(len(starts)), "n_transitions": int(len(starts) - 1)}
    if len(starts) > max_runs:
        out["runs"] = None  # shuffled-looking: no contiguous structure to tabulate
        return out
    Xi = np.asarray(X[idx])
    runs = []
    for a, b in zip(starts, ends, strict=True):
        runs.append(
            {
                "start": int(a),
                "n": int(b - a),
                "treatment": int(t[a]),
                **{f"mean_{f}": float(Xi[a:b, j].mean()) for j, f in enumerate(FEATURES)},
            }
        )
    # Hypothesis only: consecutive treated + control runs are one sub-dataset
    groups, cur = [], None
    for r in runs:
        if cur is None or (r["treatment"] == 1 and cur["has_control"]):
            cur = {"n_treated": 0, "n_control": 0, "has_control": False}
            groups.append(cur)
        cur["n_treated" if r["treatment"] == 1 else "n_control"] += r["n"]
        cur["has_control"] |= r["treatment"] == 0
    for g in groups:
        g["treated_share"] = g["n_treated"] / (g["n_treated"] + g["n_control"])
        del g["has_control"]
    out["runs"] = runs
    out["inferred_groups"] = groups
    return out


ASSUMPTION_NOTE = (
    "The constant 0.85 treatment propensity is supported by the publisher (Diemert et al., "
    "arXiv:2111.10106: all source tests were subsampled to a global 85% treatment ratio, with "
    "treatment-independence validation reported), but it cannot be independently verified from the "
    "released file, which has no test identifier. Criteo's outcome-dependent privacy subsampling "
    "limits causal interpretation and prevents claims about original-population incrementality or ROI."
)


def verdict(checks: dict) -> dict:
    """Feature-based failures block; an A1d ordering warning and borderline grades do not."""
    grades = {k: c["grade"] for k, c in checks.items()}
    if "fail" in grades.values():
        failed = ", ".join(k for k, g in grades.items() if g == "fail")
        return {
            "status": "does_not_support",
            "statement": (
                f"Failed check(s): {failed}. Treatment is not independent of the observed features, or "
                "positivity is violated, within this file. Policy-value estimates under a "
                "constant-propensity assumption would be unreliable; report the issue and stop."
            ),
        }
    caveats = [k for k, g in grades.items() if g in ("warn", "ordering_warning")]
    extra = (
        f" Warnings (not failures): {', '.join(caveats)}; the row-order finding is an ordering "
        "warning, not proof of heterogeneous propensities."
        if caveats
        else ""
    )
    return {
        "status": "benchmark_modeling_allowed",
        "statement": (
            "No feature-based check failed: treatment looks independent of f0-f11 with adequate "
            "positivity. Benchmark modeling and sample-level, comparative evaluation may proceed under "
            "the publisher-supported constant-0.85 propensity assumption."
            + extra
            + " "
            + ASSUMPTION_NOTE
            + " This is a diagnostic judgment, not proof of identifiability."
        ),
    }


def run_diagnostics(X, treatment, train_idx, scaler, seed: int = SEED) -> dict:
    smd = smd_table(X, treatment, train_idx)
    ks = ks_table(X, treatment, train_idx, seed=seed)
    clf = treatment_classifier_diagnostic(X, treatment, train_idx, scaler, seed)
    bins = bin_proportions(X, treatment, train_idx)
    blocks = block_proportions(treatment, train_idx)
    runs = run_structure(X, treatment, train_idx)
    big = bins[bins["n"] >= MIN_BIN_N]
    max_smd = float(smd["smd"].abs().max())
    max_bin = float(big["deviation"].abs().max())
    max_block = float(blocks["deviation"].abs().max())
    auc_gap = max(clf["lightgbm"]["auc"], clf["logistic"]["auc"]) - 0.5
    lo, hi = POSITIVITY_RANGE
    pos_ok = (big["treated_share"].min() >= lo) and (big["treated_share"].max() <= hi)
    floor = min(big["treated_share"].min(), 1 - big["treated_share"].max())
    max_weight = float(1 / floor) if floor > 0 else None  # None = unbounded weight
    checks = {
        "A1a_standardized_mean_differences": {
            "value": max_smd,
            "grade": _grade(max_smd, "max_abs_smd"),
            "rule": "max |SMD| <= 0.05 pass, <= 0.10 warn",
        },
        "A1b_treatment_from_features_auc": {
            "value": auc_gap,
            "grade": _grade(auc_gap, "auc_gap"),
            "rule": "best held-out AUC - 0.5 <= 0.01 pass, <= 0.05 warn",
        },
        "A1c_treatment_share_by_feature_bin": {
            "value": max_bin,
            "grade": _grade(max_bin, "max_bin_deviation"),
            "rule": f"max |bin share - overall| (bins n >= {MIN_BIN_N}) <= 0.01 pass, <= 0.03 warn",
        },
        "A1d_stability_across_row_blocks": {
            "value": max_block,
            "grade": "pass"
            if max_block <= THRESHOLDS["max_block_deviation"][0]
            else "ordering_warning",
            "rule": "max |block share - overall| <= 0.01 pass, else ordering_warning (does not fail the "
            "verdict: row order is not the randomization unit)",
        },
        "A2_positivity": {
            "value": {
                "min_bin_share": float(big["treated_share"].min()),
                "max_bin_share": float(big["treated_share"].max()),
                "max_ipw_weight": max_weight,
            },
            "grade": "pass"
            if pos_ok
            and clf["predicted_propensity"]["q01"] >= lo
            and clf["predicted_propensity"]["q99"] <= hi
            else "fail",
            "rule": "all bin shares, and the 1st-99th percentile of predicted propensity, "
            "within [0.05, 0.95]",
        },
    }
    return {
        "n_train_rows": int(len(train_idx)),
        "overall_treated_share": float(np.asarray(treatment)[train_idx].mean()),
        "smd": smd.to_dict("records"),
        "ks": ks.to_dict("records"),
        "classifier": clf,
        "bins": bins.to_dict("records"),
        "blocks": blocks.to_dict("records"),
        "run_structure": runs,
        "checks": checks,
        "verdict": verdict(checks),
        "thresholds": THRESHOLDS,
    }


def _figure(result: dict, path: Path) -> None:
    bins = pd.DataFrame(result["bins"])
    fig, axes = plt.subplots(3, 4, figsize=(14, 8), sharey=True)
    for ax, f in zip(axes.flat, FEATURES, strict=True):
        b = bins[bins["feature"] == f]
        ax.errorbar(
            b["bin"],
            b["treated_share"],
            yerr=[b["treated_share"] - b["ci_low"], b["ci_high"] - b["treated_share"]],
            fmt="o-",
            ms=3,
            color="#1f77b4",
        )
        ax.axhline(result["overall_treated_share"], color="k", lw=0.8, ls="--")
        ax.set_title(
            f"{f} (smd {next(r['smd'] for r in result['smd'] if r['feature'] == f):+.3f})",
            fontsize=9,
        )
        ax.set_ylim(0.8, 0.9)
    fig.supxlabel("feature quantile bin")
    fig.supylabel("treated share (train)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def build_markdown(result: dict) -> str:
    c, clf = result["checks"], result["classifier"]
    smd = pd.DataFrame(result["smd"]).merge(pd.DataFrame(result["ks"]), on="feature")
    rows = "\n".join(
        f"| {k} | {v['value'] if not isinstance(v['value'], dict) else v['value']} | {v['grade']} | {v['rule']} |"
        for k, v in c.items()
    )
    rs = result["run_structure"]
    if rs.get("runs"):
        tab = pd.DataFrame(rs["runs"])[["start", "n", "treatment", "mean_f0", "mean_f6", "mean_f9"]]
        run_text = (
            f"The train rows consist of only **{rs['n_runs']} contiguous single-treatment runs** "
            f"({rs['n_transitions']} transitions); a shuffled file would have millions. The file is "
            "ordered, which is why the row-block check raises an **ordering warning**. Feature means "
            "shift between runs (below), so the file is probably a concatenation of sources.\n\n"
            f"{tab.round(2).to_markdown(index=False)}\n\n"
            "This is a warning about row order, not proof of heterogeneous treatment propensities: the "
            "publishers state that all source tests were subsampled to a common 85% treatment ratio. "
            "Run boundaries are not used as experiment IDs, and treated shares of adjacent runs are "
            "not interpreted as test-level propensities. Do not rely on row order in any split or "
            "evaluation (the C0 split is a random, treatment-stratified permutation)."
        )
    else:
        run_text = f"{rs['n_runs']} treatment runs in row order: no contiguous group structure."
    return f"""# Criteo C1: treatment diagnostics (train split only)

**Verdict: {result["verdict"]["status"]}.** {result["verdict"]["statement"]}

Train rows: {result["n_train_rows"]:,}; overall treated share {result["overall_treated_share"]:.4f}.
Thresholds were fixed before looking at the results.

| Check | Value | Grade | Rule |
|---|---|---|---|
{rows}

## Treatment-from-features classifier
Fitted on {clf["n_fit"]:,} train rows, evaluated on a disjoint {clf["n_eval"]:,}-row train-derived set.
AUC: LightGBM {clf["lightgbm"]["auc"]:.4f} (SE {clf["lightgbm"]["auc_se"]:.4f}), logistic {clf["logistic"]["auc"]:.4f}
(SE {clf["logistic"]["auc_se"]:.4f}); permuted-label control {clf["permuted_label_auc"]:.4f}.
Predicted propensity range {clf["predicted_propensity"]["min"]:.3f} to {clf["predicted_propensity"]["max"]:.3f}
(median {clf["predicted_propensity"]["median"]:.3f}).

## Balance by feature
{smd.round(4).to_markdown(index=False)}

Treated share by feature bin: `treatment_balance.png`; per-bin and per-block tables: `treatment_diagnostics.json`.

## Row-order structure
{run_text}

## Limitations (read before using any result)
- **Non-uniform subsampling.** Criteo sub-sampled the data non-uniformly for privacy; the sampling
  mechanism is unknown. Balance on f0-f11 does not rule out selection that depends on unobserved
  variables (advertiser, campaign, time) or on outcomes. These diagnostics see only the observed features.
- **Publisher-supported, unverifiable propensity.** The paper (Section 3, "Considerations to concatenate
  data from different tests") says CRITEO-UPLIFTv2 was subsampled to a global 85% treatment ratio, and
  Section 5.1 reports treatment-independence validation. The released file has no test identifier, so the
  constant 0.85 propensity is an assumption we use, not one we can verify here.
- **Outcome-dependent privacy subsampling.** It limits causal interpretation: estimates describe this
  released sample, and original-population incrementality or ROI claims are not supported.
- **Large n.** With millions of rows even trivial imbalances are statistically detectable, so the
  thresholds are practical (effect-size) rules, not significance tests.
- A pass supports benchmark modeling and sample-level, comparative evaluation within this file. It does
  not establish population-level incrementality or advertising ROI. A fail is reported as-is: no
  propensity model or alternative architecture is introduced automatically.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Criteo C1 treatment diagnostics.")
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--out", type=Path, default=REPORT_DIR)
    args = parser.parse_args(argv)
    X, labels = load_arrays(args.data)
    split = load_split(args.data)
    treatment = np.asarray(labels["treatment"])
    scaler = fit_scaler(X, split.train_idx)
    result = run_diagnostics(X, treatment, split.train_idx, scaler)
    from upliftiq.data.criteo_prepare import peak_rss_gb

    result["peak_rss_gb"] = round(peak_rss_gb(), 2)
    result["split_fingerprint"] = split.fingerprint()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "treatment_diagnostics.json").write_text(
        json.dumps(result, indent=2, default=float) + "\n"
    )
    (args.out / "treatment_diagnostics.md").write_text(build_markdown(result))
    _figure(result, args.out / "treatment_balance.png")
    print(result["verdict"]["status"], {k: (v["grade"]) for k, v in result["checks"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
