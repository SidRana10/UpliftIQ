"""C6 final: full-scale validation benchmark of the three finalists, selection, and freeze.

Reuses the C5/C6 pipeline unchanged: isolated worker processes, the fixed 500K validation subset,
the shared out-of-sample AIPW nuisance, paired Poisson bootstrap. The only new work is

1. training the finalists on the *whole* training split (no tuning; same fixed hyperparameters),
2. applying the predefined selection rule to the validation results,
3. comparing against the cached 2M rung (paired, same validation customers), and
4. writing a frozen configuration *before* any test evaluation.

The held-out test set is never read: every scoring call passes the test ids as protected ids.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from upliftiq.data.criteo_prepare import PROCESSED_DIR, REPORT_DIR, SEED, load_split
from upliftiq.evaluation import binary_policy as bp
from upliftiq.evaluation import criteo_benchmark as cb
from upliftiq.evaluation.criteo_scaling import _validation_inputs

FINALISTS = ("logistic", "s_learner", "tarnet_cpu")  # simplest first (plan: logistic < S < TARNet)
BASE_SIZE = 2_000_000
PRIMARY_METRIC, PRIMARY_BUDGET = "value@20%", 0.20
Z95 = 1.959964
OUT_DIR = REPORT_DIR / "benchmark_full"

SELECTION_RULE = (
    "Primary metric: AIPW policy value at a 20% contact budget on the fixed validation subset "
    "(Qini is a supporting diagnostic only). 1) A model is eligible only if its paired difference "
    "from random targeting at 20% has a 95% interval above 0. 2) The best eligible model by point "
    "value is found. 3) One-SE simplicity rule: walking from the simplest model "
    "(logistic < S-learner < TARNet), the first eligible model whose value is within one paired "
    "standard error of the best wins. The paired SE is the bootstrap SE of the difference, taken "
    "as the 95% percentile-interval width / 3.92. Hyperparameters are never tuned."
)


# ---------------------------------------------------------------- selection


def paired_difference(bundle: dict, a: str, b: str, metric: str) -> dict:
    """AIPW difference ``a - b`` with its paired 95% interval and SE, from the joint bootstrap."""
    for ref, res in bundle["results"].items():
        pw = res.pairwise[(res.pairwise.estimator == "AIPW") & (res.pairwise.metric == metric)]
        if ref == b and (pw.model == a).any():
            r, sign = pw[pw.model == a].iloc[0], 1.0
        elif ref == a and (pw.model == b).any():
            r, sign = pw[pw.model == b].iloc[0], -1.0
        else:
            continue
        lo, hi = sorted((sign * r.ci_low, sign * r.ci_high))
        return {
            "diff": sign * float(r["diff"]),
            "ci_low": lo,
            "ci_high": hi,
            "se": (hi - lo) / (2 * Z95),
        }
    raise KeyError(f"no paired comparison between {a} and {b} for {metric}")


def select(bundle: dict, order=FINALISTS, budget: float = PRIMARY_BUDGET) -> dict:
    """Apply the predefined rule (see ``SELECTION_RULE``) to a scored validation bundle."""
    pol = next(iter(bundle["results"].values())).policies
    pol = pol[
        (pol.estimator == "AIPW") & (pol.policy == "budget") & (pol.budget_requested == budget)
    ]
    rows = {}
    for n in order:
        p = pol[pol.model == n].iloc[0]
        rows[n] = {
            "value": float(p.value),
            "ci_low": float(p.ci_low),
            "ci_high": float(p.ci_high),
            "vs_random": float(p.diff_vs_random),
            "vs_random_ci_low": float(p.diff_vs_random_ci_low),
            "vs_random_ci_high": float(p.diff_vs_random_ci_high),
            "beats_random": bool(p.diff_vs_random_ci_low > 0),
        }
    eligible = [n for n in order if rows[n]["beats_random"]]
    out = {"order": list(order), "rows": rows, "eligible": eligible, "steps": [], "chosen": None}
    if not eligible:
        out["reason"] = "no finalist beats random targeting at the 20% budget"
        return out
    best = max(eligible, key=lambda n: (rows[n]["value"], -order.index(n)))
    out["best_by_value"] = best
    chosen = best
    for n in eligible:
        if n == best:
            break
        d = paired_difference(bundle, best, n, PRIMARY_METRIC)
        within = bool(d["diff"] <= d["se"])
        out["steps"].append({"simpler": n, "best": best, **d, "within_one_se": within})
        if within:
            chosen = n
            break
    out["chosen"] = chosen
    q = {
        r.model: r.qini
        for r in next(iter(bundle["results"].values())).qini.itertuples()
        if r.estimator == "AIPW"
    }
    out["best_by_qini"] = max(order, key=lambda n: q[n])
    out["qini_supports_choice"] = bool(out["best_by_qini"] == chosen)
    return out


# ---------------------------------------------------------------- comparison with the 2M rung


def size_changes(
    names, data_dir, n_val, seed, n_boot, base_size, full_size, nuisance_rows=cb.NUISANCE_ROWS
):
    """Paired change from ``base_size`` to ``full_size`` training rows for each model."""
    data_dir = Path(data_dir)
    split, y, t, val_rows, nuisance = _validation_inputs(data_dir, n_val, seed, nuisance_rows)
    rows = []
    for name in names:
        scores = {
            "base": np.load(cb.rung_dir(data_dir, base_size, n_val) / f"{name}_uplift.npy"),
            "full": np.load(cb.rung_dir(data_dir, full_size, n_val) / f"{name}_uplift.npy"),
        }
        res = bp.evaluate_targeting(
            y, t, scores, nuisance, budgets=cb.BUDGETS, n_boot=n_boot, seed=seed,
            row_ids=val_rows, protected_ids=split.test_idx,
        )  # fmt: skip
        pol = res.policies[(res.policies.estimator == "AIPW") & (res.policies.policy == "budget")]
        q = res.qini[res.qini.estimator == "AIPW"].set_index("model")
        pw = res.pairwise[res.pairwise.estimator == "AIPW"].set_index("metric")
        for metric in [f"value@{b:.0%}" for b in cb.BUDGETS] + ["qini"]:
            if metric == "qini":
                lvl = {
                    k: (q.loc[k, "qini"], q.loc[k, "ci_low"], q.loc[k, "ci_high"]) for k in scores
                }
            else:
                b = float(metric[6:-1]) / 100
                lvl = {}
                for k in scores:
                    p = pol[(pol.model == k) & (pol.budget_requested == b)].iloc[0]
                    lvl[k] = (p.value, p.ci_low, p.ci_high)
            d = pw.loc[metric]  # "full" minus the first model ("base")
            rows.append(
                {
                    "model": name,
                    "metric": metric,
                    "base": lvl["base"],
                    "full": lvl["full"],
                    "change": (float(d["diff"]), float(d.ci_low), float(d.ci_high)),
                }
            )
    return rows


# ---------------------------------------------------------------- frozen configuration


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_frozen_config(bundle, selection, size, data_dir, n_val, seed, n_boot, nuisance_rows):
    data_dir = Path(data_dir)
    run_dir = cb.rung_dir(data_dir, size, n_val)
    name = selection["chosen"]
    manifest = bundle["manifests"][name]
    first = next(iter(bundle["results"].values()))
    nuis = bundle["nuisance"]
    return {
        "status": "frozen before any test-set evaluation; test set not accessed",
        "test_set_accessed": False,
        "selected": {
            "name": name,
            "family": manifest["family"],
            "constructor_kwargs": manifest["kwargs"],
            "hyperparameters": manifest["hyperparameters"],
            "seed": manifest["seed"],
            "training_rows": manifest["n_fit"],
            "training_data": "entire training split of the frozen split (no subsampling)",
            "validation_uplift_sha256": _sha256(run_dir / f"{name}_uplift.npy"),
            "features": "f0-f11 only; exposure/visit/conversion/treatment never used as features",
        },
        "selection": {
            "rule": SELECTION_RULE,
            "primary_metric": PRIMARY_METRIC,
            "outcome": {k: v for k, v in selection.items() if k != "rows"},
            "candidates": selection["rows"],
            "finalists": list(selection["order"]),
        },
        "dataset": {
            "source_sha256": manifest["provenance"]["data_sha256"],
            "rows": None,
        },
        "split": {
            "fingerprint": manifest["provenance"]["split_fingerprint"],
            "seed": manifest["provenance"]["split_seed"],
            "fractions": "train 64% / validation 16% / test 20%, stratified on treatment",
        },
        "evaluation": {
            "estimator_primary": "AIPW (IPW reported as a diagnostic only)",
            "propensity": bp.P_TREAT,
            "budgets": list(cb.BUDGETS),
            "validation_subset": {
                "rows": n_val,
                "stratified_by": "treatment",
                "seed": seed,
            },
            "nuisance": {
                "model": nuis["model"],
                "train_rows": nuis["n_train"],
                "train_rows_fingerprint": nuis["train_rows_fingerprint"],
                "disjoint_from_train_subsets_up_to": nuis["disjoint_from_candidate_subsets_up_to"],
                "note": (
                    "trained on train rows only; the full-train finalists overlap with these rows "
                    "(about 5.6% of training rows), but validation rows are untouched by both"
                ),
            },
            "bootstrap": first.meta["bootstrap"],
            "n_boot": n_boot,
            "tie_break": first.meta["tie_break"],
            "qini": "100-point trapezoid, area between gain curve and random line",
        },
        "code": manifest["code"],
        "libraries": manifest["libraries"],
        "omp_num_threads": manifest["omp_num_threads"],
        "caveats": [
            "Conditional estimates for the released Criteo sample, not population-level effects or ROI.",
            "The constant 0.85 propensity is publisher-supported but unverifiable; IPW and AIPW disagree on this file.",
            "Validation results guided this choice; they are not evidence of final generalization.",
        ],
    }


# ---------------------------------------------------------------- report


def _pct(v, lo, hi):
    return cb._ci(v, lo, hi)


def build_report(bundle, tables, selection, changes, size, base_size) -> str:
    md = lambda df: df.to_markdown(index=False)  # noqa: E731
    m0 = next(iter(bundle["manifests"].values()))
    prov, ac = m0["provenance"], bundle["assumption_check"]
    libs = {k: v for m in bundle["manifests"].values() for k, v in m["libraries"].items()}
    nuis, meta = bundle["nuisance"], next(iter(bundle["results"].values())).meta
    z = ac["predicted_baseline_gap"] / ac["predicted_baseline_gap_se"]
    names = list(selection["order"])

    rows = []
    for n in names:
        r = selection["rows"][n]
        rows.append(
            {
                "model": n,
                "AIPW value@20% (pp)": _pct(r["value"], r["ci_low"], r["ci_high"]),
                "vs random@20% (pp)": _pct(
                    r["vs_random"], r["vs_random_ci_low"], r["vs_random_ci_high"]
                ),
                "beats random": "yes" if r["beats_random"] else "no",
            }
        )
    elig = md(pd.DataFrame(rows))
    step_lines = (
        "\n".join(
            f"- {s['best']} minus {s['simpler']}: {s['diff'] * 100:.3f} pp vs one paired SE "
            f"{s['se'] * 100:.3f} pp, {'within' if s['within_one_se'] else 'outside'} one SE "
            f"({'simpler model chosen' if s['within_one_se'] else 'continue'})"
            for s in selection["steps"]
        )
        or "- (the best model is the simplest eligible one)"
    )
    chosen = selection["chosen"]
    outcome = (
        f"**Selected finalist: `{chosen}`.** Best eligible by point value: `{selection['best_by_value']}`; "
        f"best by Qini: `{selection['best_by_qini']}` "
        f"({'agrees with' if selection['qini_supports_choice'] else 'differs from'} the selection)."
        if chosen
        else f"**No finalist selected:** {selection['reason']}."
    )

    pair_rows = []
    for ref, res in bundle["results"].items():
        if res.pairwise.empty:
            continue
        pw = res.pairwise[res.pairwise.estimator == "AIPW"]
        for r in pw.itertuples():
            pair_rows.append((ref, r.model, r.metric, r.diff, r.ci_low, r.ci_high))
    seen, pairs = set(), []
    for ref, model, metric, d, lo, hi in pair_rows:
        key = frozenset((ref, model)), metric
        if key in seen:
            continue
        seen.add(key)
        pairs.append(
            {
                "comparison": f"{model} minus {ref}",
                "metric": metric,
                "difference (pp)": _pct(d, lo, hi),
                "excludes 0": bool(lo > 0 or hi < 0),
            }
        )
    pairs = pd.DataFrame(pairs)

    ch = pd.DataFrame(
        [
            {
                "model": c["model"],
                "metric": c["metric"],
                f"{base_size:,} rows": _pct(*c["base"]),
                f"{size:,} rows": _pct(*c["full"]),
                "change (pp, paired)": _pct(*c["change"])
                + ("" if c["change"][1] <= 0 <= c["change"][2] else " *"),
            }
            for c in changes
        ]
    )
    eng = tables["engineering"]
    return f"""# Criteo C6 final: full-scale validation benchmark of three finalists

**Validation only.** The held-out test set was not accessed. Hyperparameters are the fixed
C3/C4 defaults (no tuning); the three finalists were trained on the entire training split
({size:,} rows). Values are **conditional estimates for the released Criteo sample** in percentage
points of visit rate per evaluated customer relative to treating nobody, estimated with AIPW
(IPW as a diagnostic). They are not population-level incrementality, guaranteed causal gains, or
advertising ROI, and because validation results drove this choice they are not evidence of final
generalization performance.

## Setup
- Finalists: {", ".join(names)} (isolated worker processes, `OMP_NUM_THREADS={m0["omp_num_threads"]}`).
- Training rows: {size:,} (whole training split). Validation subset: {bundle["n"]:,} rows (treatment-stratified, fixed, seed {m0["seed"]}). Bootstrap: {meta["n_boot"]} Poisson replicates, shared weights (paired).
- AIPW nuisance (shared by every model): {nuis["model"]} trained on {nuis["n_train"]:,} train rows, predicting the validation subset out of sample. Those rows were chosen to be disjoint from the 2M candidate subset, so they **overlap the full training split** (about {nuis["n_train"] / size:.1%} of it). Both models are fitted on train rows only and neither sees a validation outcome, so the out-of-sample property for scored rows holds; the earlier rungs' strict disjointness does not.
- Split fingerprint `{prov["split_fingerprint"][:16]}…`, data sha256 `{(prov["data_sha256"] or "n/a")[:16]}…`.
- Code commit `{(m0["code"]["commit"] or "n/a")[:10]}` (dirty: {m0["code"]["dirty"]}); python {libs["python"]}, lightgbm {libs["lightgbm"]}, torch {libs.get("torch", "n/a")}, scikit-learn {libs["scikit-learn"]}.

## Policy value at matched contact budgets (AIPW)
{md(tables["value"])}

References:

{md(tables["references"])}

## Qini (supporting diagnostic)
{md(tables["qini"].round({"Qini / treat-all effect": 3}))}

## Paired model differences (AIPW)
Differences come from joint bootstrap runs with shared weights (each pair appears once).

{md(pairs)}

## Selection (predefined rule)
{SELECTION_RULE}

Eligibility at the 20% budget:

{elig}

One-SE walk:

{step_lines}

{outcome}

Selection is a statement about this validation subset under fixed untuned hyperparameters. A model
that fails to separate from simpler ones is not evidence it is equivalent; it only means the data
cannot distinguish them at this size, which is why the simpler one wins ties.

## Change from the previous 2M benchmark (paired, same validation customers)
`*` marks a paired interval that excludes 0. Intervals reflect scoring noise on the validation
subset, not retraining variability.

{md(ch)}

## Engineering: training cost, latency, memory
Latency uses `predict_uplift` on validation rows (includes preprocessing). RSS peak in fit is
sampled every 50 ms; process peak includes data loading.

{md(eng.round({"train_s": 1, "predict 500k rows (s)": 2, "RSS before fit (GB)": 2, "RSS peak in fit (GB)": 2, "process peak RSS (GB)": 2}))}

## Assumption check: IPW vs AIPW (identification caveat preserved)
| quantity | value |
|:--|--:|
| treated share (validation subset) | {ac["treated_share"]:.4f} |
| treat-all effect, IPW, pp | {ac["ate_ipw"] * 100:.3f} |
| treat-all effect, AIPW, pp | {ac["ate_aipw"] * 100:.3f} |
| IPW minus AIPW, pp (SE) | {ac["ate_ipw_minus_aipw"] * 100:.3f} ({ac["ate_ipw_minus_aipw_se"] * 100:.3f}) |
| predicted baseline, treated vs control, pp | {ac["predicted_baseline_treated"] * 100:.3f} vs {ac["predicted_baseline_control"] * 100:.3f} (z = {z:.1f}) |

The discrepancy documented in `reports/criteo/benchmark/ipw_aipw_diagnostic.md` persists: the
constant 0.85 propensity does not fully hold in the released file, IPW is likely biased upward, and
AIPW is primary. AIPW is not guaranteed unbiased if the propensity assumption and the outcome
nuisance are both wrong, and one shared estimator makes models comparable without making their
relative rankings unbiased.

## Frozen configuration
`frozen_config.json` in this folder records the selected model, effective hyperparameters, seed,
dataset checksum, split fingerprint, validation/nuisance/bootstrap settings, the SHA-256 of the
selected model's validation scores and the library and code state. It was written before any test
evaluation, and `test_set_accessed` is `false`. The code state is marked dirty because these results
are intentionally uncommitted; commit them before the one-shot test evaluation so the freeze is
tied to a clean commit.
"""


# ---------------------------------------------------------------- driver


def run(
    data_dir=PROCESSED_DIR,
    out_dir=None,
    names=FINALISTS,
    n_val=cb.VAL_ROWS,
    seed=SEED,
    n_boot=cb.N_BOOT,
    base_size=BASE_SIZE,
    nuisance_rows=cb.NUISANCE_ROWS,
    nuisance_exclude=cb.NUISANCE_EXCLUDE,
    specs=None,
) -> Path:
    specs = specs or cb.MODEL_SPECS
    data_dir, names = Path(data_dir), list(names)
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    split = load_split(data_dir)
    size = len(split.train_idx)
    for n in names:  # earlier rung must already exist: no earlier benchmark is repeated
        base = cb.rung_dir(data_dir, base_size, n_val)
        if not (base / f"{n}_uplift.npy").exists():
            raise FileNotFoundError(f"missing {base_size:,}-row result for {n}; run C6 first")
    nuisance = cb.ensure_nuisance(
        data_dir, split, n_val, seed, False, nuisance_rows, nuisance_exclude
    )
    for n in names:
        cb.ensure_model(
            n, {"family": specs[n][0], "kwargs": specs[n][1]}, size, data_dir, split, n_val, seed
        )
    bundle = cb.evaluate_rung(data_dir, split, size, n_val, names, nuisance, seed, n_boot)
    tables = cb.build_tables(bundle)
    selection = select(bundle, names)
    changes = size_changes(names, data_dir, n_val, seed, n_boot, base_size, size, nuisance_rows)
    frozen = None
    if selection["chosen"]:
        frozen = build_frozen_config(
            bundle, selection, size, data_dir, n_val, seed, n_boot, nuisance_rows
        )
        frozen["dataset"]["rows"] = int(len(split.assignment))
        cb._json(out_dir / "frozen_config.json", frozen)
    report = out_dir / "final_report.md"
    report.write_text(build_report(bundle, tables, selection, changes, size, base_size))
    cb._json(
        out_dir / "final_results.json",
        {
            "status": "validation-only full-scale benchmark; test set untouched",
            "n_fit": size,
            "n_val": bundle["n"],
            "selection": selection,
            "assumption_check": bundle["assumption_check"],
            "meta": next(iter(bundle["results"].values())).meta,
            "tables": {k: v.to_dict("records") for k, v in tables.items()},
            "changes_vs_base": changes,
            "manifests": {
                n: {k: v for k, v in m.items() if k != "training"}
                | (
                    {"training": {k: v for k, v in m["training"].items() if k != "history"}}
                    if "training" in m
                    else {}
                )
                for n, m in bundle["manifests"].items()
            },
            "policies": {k: r.policies.to_dict("records") for k, r in bundle["results"].items()},
            "qini": {k: r.qini.to_dict("records") for k, r in bundle["results"].items()},
            "pairwise": {k: r.pairwise.to_dict("records") for k, r in bundle["results"].items()},
        },
    )
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Criteo C6 final full-scale benchmark and freeze.")
    ap.add_argument("--data-dir", default=str(PROCESSED_DIR))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-boot", type=int, default=cb.N_BOOT)
    args = ap.parse_args(argv)
    print("wrote", run(args.data_dir, args.out_dir, n_boot=args.n_boot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
