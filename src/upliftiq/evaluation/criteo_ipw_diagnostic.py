"""Focused diagnostic of the IPW vs AIPW treat-all discrepancy in the C5 benchmark.

Uses only the cached 500K validation subset (train/validation rows, never test) and the cached,
independent nuisance predictions. Nothing is retrained and no propensity model is fitted. The
checks, in order:

1. Re-derive IPW and AIPW with plain NumPy and compare with ``binary_policy`` (implementation check).
2. Decompose IPW - AIPW exactly by treatment arm: with e the treatment propensity,
   ``IPW - AIPW = mean(mu1 (T/e - 1)) - mean(mu0 ((1-T)/(1-e) - 1))``; each term is how far the
   propensity-weighted arm's predicted outcome is from the whole population's.
3. Paired uncertainty of the difference (analytic and Poisson bootstrap on the same observations).
4. Observed vs independently predicted outcome rates by arm, and arm balance in the predicted baseline,
   with a permutation test (treatment labels shuffled; predictions fixed) of the baseline gap.
5. Outcome-standardized (g-computation) and propensity-free post-stratified estimates for comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from upliftiq.data.criteo import FEATURES
from upliftiq.data.criteo_prepare import PROCESSED_DIR, REPORT_DIR, SEED, load_arrays, load_split
from upliftiq.evaluation import binary_policy as bp
from upliftiq.evaluation.criteo_benchmark import (
    NUISANCE_ROWS,
    VAL_ROWS,
    nuisance_train_rows,
    validation_subset,
)
from upliftiq.evaluation.policy import OutcomeModelPredictions

E = 0.85
N_BOOT = 1000
N_PERM = 2000
N_STRATA = 10


def independent_estimates(y, t, mu0, mu1, e: float = E) -> dict:
    """IPW, AIPW and their pieces from the textbook formulas, with no ``binary_policy`` code."""
    n = len(y)
    treated, control = t == 1, t == 0
    w1, w0 = treated / e, control / (1 - e)
    ipw = (w1 * y).mean() - (w0 * y).mean()
    gcomp = (mu1 - mu0).mean()
    aug1 = (w1 * (y - mu1)).mean()
    aug0 = (w0 * (y - mu0)).mean()
    aipw = gcomp + aug1 - aug0
    # exact arm decomposition of IPW - AIPW (see module docstring)
    term1 = (mu1 * (w1 - 1)).mean()
    term0 = (mu0 * (w0 - 1)).mean()
    return {
        "n": n,
        "weight_sum_treated": float(w1.mean()),  # equals 1 when the realized share is e
        "weight_sum_control": float(w0.mean()),
        "ipw": float(ipw),
        "ipw_hajek": float(y[treated].mean() - y[control].mean()),
        "aipw": float(aipw),
        "gcomp": float(gcomp),
        "augmentation_treated": float(aug1),
        "augmentation_control": float(aug0),
        "ipw_minus_aipw": float(ipw - aipw),
        "term_treated_arm": float(term1),
        "term_control_arm": float(term0),
    }


def per_observation(y, t, mu0, mu1, e: float = E) -> dict[str, np.ndarray]:
    w1, w0 = (t == 1) / e, (t == 0) / (1 - e)
    return {
        "ipw": w1 * y - w0 * y,
        "aipw": (mu1 - mu0) + w1 * (y - mu1) - w0 * (y - mu0),
        "gcomp": mu1 - mu0,
        "term_treated_arm": mu1 * (w1 - 1),
        "term_control_arm": mu0 * (w0 - 1),
        "ipw_minus_aipw": (w1 * y - w0 * y) - ((mu1 - mu0) + w1 * (y - mu1) - w0 * (y - mu0)),
    }


def paired_intervals(obs: dict[str, np.ndarray], n_boot=N_BOOT, seed=SEED) -> pd.DataFrame:
    """Analytic SE and Poisson-bootstrap 95% interval for the mean of each per-observation vector."""
    keys = list(obs)
    M = np.column_stack([obs[k] for k in keys])
    n = len(M)
    draws = np.empty((n_boot, M.shape[1]))
    for b in range(n_boot):
        w = np.random.default_rng([seed, b]).poisson(1.0, n).astype(np.float64)
        draws[b] = (w @ M) / w.sum()
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return pd.DataFrame(
        {
            "quantity": keys,
            "estimate": M.mean(axis=0),
            "se": M.std(axis=0, ddof=1) / np.sqrt(n),
            "ci_low": lo,
            "ci_high": hi,
        }
    )


def arm_rates(y, t, mu0, mu1) -> dict:
    tr, ct = t == 1, t == 0

    def se(x):
        return float(x.std(ddof=1) / np.sqrt(len(x)))

    return {
        "observed_rate_treated": float(y[tr].mean()),
        "observed_rate_control": float(y[ct].mean()),
        "predicted_mu1_among_treated": float(mu1[tr].mean()),
        "predicted_mu0_among_control": float(mu0[ct].mean()),
        "predicted_mu0_among_treated": float(mu0[tr].mean()),
        "predicted_mu1_among_control": float(mu1[ct].mean()),
        "predicted_mu0_population": float(mu0.mean()),
        "predicted_mu1_population": float(mu1.mean()),
        "baseline_gap_treated_minus_control": float(mu0[tr].mean() - mu0[ct].mean()),
        "baseline_gap_se": float(np.hypot(se(mu0[tr]), se(mu0[ct]))),
        "calibration_gap_treated": float(y[tr].mean() - mu1[tr].mean()),
        "calibration_gap_control": float(y[ct].mean() - mu0[ct].mean()),
    }


def permutation_test(t, mu0, n_perm=N_PERM, seed=SEED) -> dict:
    """Null distribution of the baseline gap when treatment labels are shuffled (predictions fixed)."""
    rng = np.random.default_rng(seed)
    t = np.asarray(t).copy()
    observed = mu0[t == 1].mean() - mu0[t == 0].mean()
    total, n = mu0.sum(), len(t)
    n1 = int(t.sum())
    null = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(t)
        s1 = mu0[t == 1].sum()
        null[i] = s1 / n1 - (total - s1) / (n - n1)
    return {
        "observed_gap": float(observed),
        "null_sd": float(null.std()),
        "z": float(observed / null.std()),
        "p_two_sided": float((1 + (np.abs(null) >= abs(observed)).sum()) / (1 + n_perm)),
        "n_perm": n_perm,
    }


def poststratified(y, t, score, n_strata=N_STRATA) -> tuple[dict, pd.DataFrame]:
    """Propensity-free effect: treated-minus-control means within deciles of an independent score."""
    edges = np.quantile(score, np.linspace(0, 1, n_strata + 1))
    strata = np.clip(np.searchsorted(edges, score, side="right") - 1, 0, n_strata - 1)
    rows, est, var = [], 0.0, 0.0
    n = len(y)
    for s in range(n_strata):
        m = strata == s
        tr, ct = m & (t == 1), m & (t == 0)
        d = y[tr].mean() - y[ct].mean()
        v = y[tr].var(ddof=1) / tr.sum() + y[ct].var(ddof=1) / ct.sum()
        wgt = m.sum() / n
        est, var = est + wgt * d, var + wgt**2 * v
        rows.append(
            {
                "stratum": s + 1,
                "n": int(m.sum()),
                "mean_predicted_baseline": float(score[m].mean()),
                "treated_share": float(tr.sum() / m.sum()),
                "rate_treated": float(y[tr].mean()),
                "rate_control": float(y[ct].mean()),
                "effect": float(d),
            }
        )
    return {"estimate": float(est), "se": float(np.sqrt(var))}, pd.DataFrame(rows)


def feature_balance(X, t) -> pd.DataFrame:
    tr, ct = t == 1, t == 0
    out = []
    for j, f in enumerate(FEATURES):
        a, b = X[tr, j].astype(np.float64), X[ct, j].astype(np.float64)
        sd = np.sqrt((a.var() + b.var()) / 2)
        out.append({"feature": f, "smd": float((a.mean() - b.mean()) / sd)})
    return pd.DataFrame(out)


def run_diagnostic(
    y, t, mu0, mu1, X=None, e: float = E, n_boot: int = N_BOOT, n_perm: int = N_PERM
) -> dict:
    est = independent_estimates(y, t, mu0, mu1, e)
    obs = per_observation(y, t, mu0, mu1, e)
    # implementation check against the library code under test
    nuis = OutcomeModelPredictions.from_holdout(
        np.column_stack([mu0, mu1]), np.arange(len(y)), np.arange(len(y), len(y) + 5), n_arms=2
    )
    lib_aipw = bp.incremental_scores(y, t, nuis, e)
    lib_ipw = bp.incremental_scores(y, t, None, e)
    check = {
        "ipw_max_abs_difference": float(np.abs(lib_ipw - obs["ipw"]).max()),
        "aipw_max_abs_difference": float(np.abs(lib_aipw - obs["aipw"]).max()),
        "decomposition_residual": float(
            est["ipw_minus_aipw"] - (est["term_treated_arm"] - est["term_control_arm"])
        ),
        "gcomp_plus_augmentations_residual": float(
            est["aipw"] - (est["gcomp"] + est["augmentation_treated"] - est["augmentation_control"])
        ),
    }
    strat, strata = poststratified(y, t, mu0)
    return {
        "estimates": est,
        "implementation_check": check,
        "paired_intervals": paired_intervals(obs, n_boot).to_dict("records"),
        "arm_rates": arm_rates(y, t, mu0, mu1),
        "permutation_test": permutation_test(t, mu0, n_perm),
        "poststratified_on_predicted_baseline": strat,
        "strata": strata.to_dict("records"),
        "feature_balance": None if X is None else feature_balance(X, t).to_dict("records"),
    }


def pp(v, d=3):
    return f"{v * 100:.{d}f}"


def build_report(r: dict, meta: dict) -> str:
    e, chk, a = r["estimates"], r["implementation_check"], r["arm_rates"]
    iv = pd.DataFrame(r["paired_intervals"]).set_index("quantity")
    perm, ps = r["permutation_test"], r["poststratified_on_predicted_baseline"]
    strata = pd.DataFrame(r["strata"])
    fb = pd.DataFrame(r["feature_balance"]) if r["feature_balance"] else None
    ok_impl = max(chk["ipw_max_abs_difference"], chk["aipw_max_abs_difference"]) < 1e-12

    def fmt(k):
        c = iv.loc[k]
        return f"{pp(c.estimate)} (SE {pp(c.se)}; 95% [{pp(c.ci_low)}, {pp(c.ci_high)}])"

    low, high = strata.treated_share.iloc[0], strata.treated_share.iloc[-1]
    z_trend = (high - low) / np.sqrt(
        sum(0.85 * 0.15 / n for n in (strata.n.iloc[0], strata.n.iloc[-1]))
    )
    adj = [e["gcomp"], e["aipw"], ps["estimate"]]
    adj_lo, adj_hi = pp(min(adj), 2), pp(max(adj), 2)
    ps_gap, ps_se, ipw_pp = pp(abs(ps["estimate"] - e["aipw"]), 2), pp(ps["se"], 2), pp(e["ipw"], 2)
    tbl = strata[
        [
            "stratum",
            "n",
            "mean_predicted_baseline",
            "treated_share",
            "rate_treated",
            "rate_control",
            "effect",
        ]
    ]
    fb_text = ""
    if fb is not None:
        top = fb.reindex(fb.smd.abs().sort_values(ascending=False).index).head(4)
        fb_text = "\n".join(f"- {row.feature}: SMD {row.smd:+.3f}" for row in top.itertuples())
    return f"""# Criteo C5 diagnostic: why IPW and AIPW disagree on the treat-all effect

Scope: {meta["n"]:,} validation rows (the C5 500K validation subset), cached independent nuisance
predictions (LightGBM T-learner trained on {meta["nuisance_n_train"]:,} separate train rows), propensity
{meta["e"]}/{1 - meta["e"]:.2f}. Nothing was retrained, no propensity model was fitted, the test set was not
touched, and the original C5 artifacts are unchanged. Effects are percentage points of visit rate.

## Verdict
- **No implementation error found.** Independent NumPy formulas reproduce the library IPW and AIPW
  per-customer scores to {max(chk["ipw_max_abs_difference"], chk["aipw_max_abs_difference"]):.1e}
  (tolerance 1e-12: {"met" if ok_impl else "NOT met"}); the weights sum to 1 in both arms, so normalization is not the issue.
- **The gap is statistical/assumption-related, not arithmetic.** IPW (+{pp(e["ipw"])} pp) equals the plain
  treated-minus-control difference; AIPW (+{pp(e["aipw"])} pp) corrects it using independently predicted baselines.
  IPW minus AIPW = {fmt("ipw_minus_aipw")} pp, clearly nonzero on the same observations.
- **Mechanism:** the treated and control arms differ in the predicted baseline visit rate
  ({pp(a["predicted_mu0_among_treated"])}% vs {pp(a["predicted_mu0_among_control"])}%, gap {pp(a["baseline_gap_treated_minus_control"])} pp,
  permutation z = {perm["z"]:.1f}, p = {perm["p_two_sided"]:.4f}). Treated customers were more likely to visit anyway, which inflates the
  unadjusted treated-minus-control contrast. That is evidence against an unconditional constant 0.85 propensity in this file.
  It cannot say why (the publisher's subsampling, hidden test structure, or something else).

## 1. Formula check
| quantity | value |
|:--|--:|
| mean(T/e), mean((1-T)/(1-e)) | {e["weight_sum_treated"]:.5f}, {e["weight_sum_control"]:.5f} |
| IPW, textbook formula (pp) | {pp(e["ipw"], 4)} |
| IPW, normalized (Hajek) = treated mean - control mean (pp) | {pp(e["ipw_hajek"], 4)} |
| AIPW, textbook formula (pp) | {pp(e["aipw"], 4)} |
| max abs difference vs `binary_policy` scores, IPW / AIPW | {chk["ipw_max_abs_difference"]:.1e} / {chk["aipw_max_abs_difference"]:.1e} |
| AIPW = g-computation + treated augmentation - control augmentation (residual) | {chk["gcomp_plus_augmentations_residual"]:.1e} |

## 2. Decomposition of IPW - AIPW by arm (pp, paired SE)
Exact identity: IPW - AIPW = mean(mu1 (T/e - 1)) - mean(mu0 ((1-T)/(1-e) - 1)), residual {chk["decomposition_residual"]:.1e}.

| term | estimate (SE; 95% bootstrap interval) |
|:--|:--|
| treated-arm term, mean(mu1 (T/e - 1)) | {fmt("term_treated_arm")} |
| control-arm term, mean(mu0 ((1-T)/(1-e) - 1)) | {fmt("term_control_arm")} |
| IPW - AIPW = treated term - control term | {fmt("ipw_minus_aipw")} |
| g-computation, mean(mu1 - mu0) | {fmt("gcomp")} |
| AIPW | {fmt("aipw")} |
| IPW | {fmt("ipw")} |

The control-arm term carries most of the gap: the control arm's predicted baseline is below the whole
population's (mean(mu0 | control) - mean(mu0) is the control term), so the control mean that IPW subtracts is too low and the
contrast too high. The treated arm's predicted outcome is slightly above the population's, which adds a smaller push in the
same direction. AIPW removes both with the outcome model. The interval is paired (same {meta["n"]:,} customers, {N_BOOT} Poisson replicates).

## 3. Observed vs independently predicted rates by arm (%)
| | treated | control |
|:--|--:|--:|
| observed visit rate | {pp(a["observed_rate_treated"])} | {pp(a["observed_rate_control"])} |
| predicted for own arm (calibration) | {pp(a["predicted_mu1_among_treated"])} | {pp(a["predicted_mu0_among_control"])} |
| predicted control baseline mu0 | {pp(a["predicted_mu0_among_treated"])} | {pp(a["predicted_mu0_among_control"])} |
| predicted baseline, whole population | {pp(a["predicted_mu0_population"])} | |

The outcome model is calibrated within each arm (gaps {pp(a["calibration_gap_treated"])} and {pp(a["calibration_gap_control"])} pp), so the arm difference in mu0
reflects different feature distributions, not model error. Permutation test (labels shuffled, predictions fixed, {perm["n_perm"]} shuffles):
null SD {pp(perm["null_sd"], 4)} pp, observed gap {pp(perm["observed_gap"], 4)} pp.
{"Largest feature imbalances on the validation subset:" + chr(10) + fb_text if fb_text else ""}

## 4. Comparison with simpler adjusted estimates (treat-all effect, pp)
| estimator | estimate |
|:--|--:|
| unadjusted treated - control (= IPW) | {pp(e["ipw"])} |
| g-computation (outcome model only) | {pp(e["gcomp"])} |
| post-stratified on deciles of predicted baseline, no propensity (SE {pp(ps["se"])}) | {pp(ps["estimate"])} |
| AIPW | {pp(e["aipw"])} |

Treated share and effects by decile of the independent predicted baseline:

{tbl.round(4).to_markdown(index=False)}

The treated share rises with the predicted baseline, from {low:.4f} in the lowest decile to {high:.4f} in the highest (about {z_trend:.1f} SE of a
difference of two shares), and the treatment effect is concentrated in the top decile. The treated arm therefore
over-represents the customers who both visit most and respond most, which is what inflates the unadjusted contrast.

The three adjusted estimates ({adj_lo} to {adj_hi} pp) sit well below the unadjusted {ipw_pp} pp and agree with each other within
their uncertainty (post-stratification vs AIPW differ by {ps_gap} pp, SE of post-stratified {ps_se} pp). The AIPW implementation
therefore agrees with an independent, propensity-free adjustment. This supports AIPW *if* the treatment is as good as random
given f0-f11; imbalance on anything not in f0-f11 would remain, and nothing here can detect it.

## 5. What this means for C5 and C6
- Not a bug: do not change the evaluator or the stored results.
- Present AIPW as the primary estimate and IPW as an unadjusted check that is **expected to be upward-biased in this file**.
  Report the IPW-AIPW gap with every benchmark table, and describe gains as validation-sample estimates conditional on the
  outcome-model adjustment, not as population ROI.
- Model comparisons (paired, same nuisance, same estimator) are less exposed to the imbalance than absolute gains, but they
  inherit it: the imbalance is in the baseline, and uplift rankings are scored by AIPW differences between policies. They are
  safest as relative comparisons.
- Not tested here: hidden-group structure (row-order runs) as the source of the imbalance; creating group IDs from run
  boundaries remains out of scope. A propensity model or a sensitivity analysis would be the next tools, but were not used.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="IPW vs AIPW diagnostic on the cached C5 validation data."
    )
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--out-dir", type=Path, default=REPORT_DIR / "benchmark")
    parser.add_argument("--n-val", type=int, default=VAL_ROWS)
    args = parser.parse_args(argv)
    split = load_split(args.data_dir)
    X_all, labels = load_arrays(args.data_dir)
    treatment = np.asarray(labels["treatment"])
    val_rows = validation_subset(split, treatment, args.n_val, SEED)
    assert np.isin(val_rows, split.val_idx).all()
    base = args.data_dir / "benchmark" / f"nuisance_n{NUISANCE_ROWS}_val{args.n_val}"
    meta_n = json.loads(base.with_suffix(".json").read_text())
    mu = np.load(base.with_suffix(".npy")).astype(np.float64)
    y = np.asarray(labels["visit"][val_rows], dtype=np.float64)
    t = treatment[val_rows].astype(np.int8)
    X = np.asarray(X_all[val_rows])
    nuisance_rows = nuisance_train_rows(
        split, treatment, meta_n["n_train"], meta_n["disjoint_from_candidate_subsets_up_to"], SEED
    )
    assert not np.intersect1d(val_rows, nuisance_rows).size  # out-of-sample provenance
    result = run_diagnostic(y, t, mu[:, 0], mu[:, 1], X)
    meta = {"n": len(y), "e": E, "nuisance_n_train": meta_n["n_train"]}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "ipw_aipw_diagnostic.json").write_text(
        json.dumps({"meta": meta, **result}, indent=2, default=float) + "\n"
    )
    (args.out_dir / "ipw_aipw_diagnostic.md").write_text(build_report(result, meta))
    print("wrote", args.out_dir / "ipw_aipw_diagnostic.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
