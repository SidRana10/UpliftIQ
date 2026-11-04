"""C6 learning curve: how much does more training data help each model? (validation only)

Reads the cached validation uplift scores of the 100K, 500K and 2M rungs (already trained by
``criteo_benchmark``; nothing is retrained) and scores the three sizes of one model together in a
single ``evaluate_targeting`` call, so the paired bootstrap gives a CI for the *change* between
training sizes. The same validation subset, shared out-of-sample nuisance and bootstrap weights as
the per-rung reports are used. The held-out test set is never read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from upliftiq.data.criteo_prepare import PROCESSED_DIR, REPORT_DIR, SEED, load_arrays, load_split  # noqa: E402
from upliftiq.evaluation import binary_policy as bp  # noqa: E402
from upliftiq.evaluation.criteo_benchmark import (  # noqa: E402
    BUDGETS,
    DEFAULT_MODELS,
    N_BOOT,
    NUISANCE_ROWS,
    VAL_ROWS,
    _ci,
    _json,
    nuisance_train_rows,
    rung_dir,
    validation_subset,
)
from upliftiq.evaluation.policy import OutcomeModelPredictions  # noqa: E402

SIZES = (100_000, 500_000, 2_000_000)


def label(size: int) -> str:
    return f"{size // 1_000_000}M" if size >= 1_000_000 else f"{size // 1_000}K"


METRICS = [f"value@{b:.0%}" for b in BUDGETS] + ["qini"]


def _validation_inputs(data_dir: Path, n_val: int, seed: int, nuisance_rows: int):
    split = load_split(data_dir)
    _, labels = load_arrays(data_dir)
    treatment = np.asarray(labels["treatment"])
    val_rows = validation_subset(split, treatment, n_val, seed)
    y = np.asarray(labels["visit"][val_rows], dtype=np.float64)
    base = Path(data_dir) / "benchmark" / f"nuisance_n{nuisance_rows}_val{n_val}"
    meta = json.loads(base.with_suffix(".json").read_text())
    rows = nuisance_train_rows(
        split, treatment, meta["n_train"], meta["disjoint_from_candidate_subsets_up_to"], seed
    )
    nuisance = OutcomeModelPredictions.from_holdout(
        np.load(base.with_suffix(".npy")).astype(np.float64), val_rows, rows, n_arms=2
    )
    return split, y, treatment[val_rows], val_rows, nuisance


def learning_curve(
    names,
    data_dir=PROCESSED_DIR,
    n_val=VAL_ROWS,
    seed=SEED,
    n_boot=N_BOOT,
    sizes=SIZES,
    nuisance_rows=NUISANCE_ROWS,
) -> dict:
    """Per model: AIPW metrics at each size and paired changes relative to the middle size."""
    data_dir = Path(data_dir)
    split, y, t, val_rows, nuisance = _validation_inputs(data_dir, n_val, seed, nuisance_rows)
    ref = sizes[1]
    out = {"levels": [], "changes": [], "engineering": []}
    for name in names:
        scores = {
            label(s): np.load(rung_dir(data_dir, s, n_val) / f"{name}_uplift.npy") for s in sizes
        }
        order = [label(ref), *[label(s) for s in sizes if s != ref]]
        res = bp.evaluate_targeting(
            y,
            t,
            {k: scores[k] for k in order},
            nuisance,
            budgets=BUDGETS,
            n_boot=n_boot,
            seed=seed,
            row_ids=val_rows,
            protected_ids=split.test_idx,
        )
        pol = res.policies[(res.policies.estimator == "AIPW") & (res.policies.policy == "budget")]
        qini = res.qini[res.qini.estimator == "AIPW"]
        for s in sizes:
            row = {"model": name, "size": s}
            for b in BUDGETS:
                p = pol[(pol.model == label(s)) & (pol.budget_requested == b)].iloc[0]
                row[f"value@{b:.0%}"] = (p.value, p.ci_low, p.ci_high)
            q = qini[qini.model == label(s)].iloc[0]
            row["qini"] = (q.qini, q.ci_low, q.ci_high)
            m = json.loads((rung_dir(data_dir, s, n_val) / f"{name}.json").read_text())
            out["levels"].append(row)
            out["engineering"].append(
                {
                    "model": name,
                    "size": s,
                    "train_s": m["fit_seconds"],
                    "rss_peak_fit_gb": m["rss_gb_peak_during_fit"],
                    "epochs": m.get("training", {}).get("epochs_run"),
                }
            )
        pw = res.pairwise[res.pairwise.estimator == "AIPW"]
        for _, r in pw.iterrows():
            # r.model is the larger/smaller size relative to the 500K reference
            small = r.model == label(sizes[0])
            sign = -1.0 if small else 1.0  # express every change as "bigger size minus smaller"
            lo, hi = sorted((sign * r.ci_low, sign * r.ci_high))
            out["changes"].append(
                {
                    "model": name,
                    "step": f"{label(sizes[0])} to {label(ref)}"
                    if small
                    else f"{label(ref)} to {label(sizes[2])}",
                    "metric": r.metric,
                    "diff": sign * r["diff"],
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    return out


def _tables(lc: dict, sizes=SIZES) -> dict[str, pd.DataFrame]:
    levels = pd.DataFrame(
        [
            {
                "model": r["model"],
                "train rows": label(r["size"]),
                **{f"AIPW {m} (pp)": _ci(*r[m]) for m in METRICS},
            }
            for r in lc["levels"]
        ]
    )
    ch = pd.DataFrame(lc["changes"])
    ch = ch[ch.metric.isin(["value@20%", "qini"])]
    steps = sorted(ch.step.unique(), key=lambda s: not s.startswith(label(sizes[0])))
    rows = []
    for name in dict.fromkeys(ch.model):
        row = {"model": name}
        for step in steps:
            for metric in ("value@20%", "qini"):
                c = ch[(ch.model == name) & (ch.step == step) & (ch.metric == metric)].iloc[0]
                flag = "" if c.ci_low <= 0 <= c.ci_high else " *"
                row[f"{step}: {metric} (pp)"] = _ci(c["diff"], c.ci_low, c.ci_high) + flag
        rows.append(row)
    changes = pd.DataFrame(rows)
    eng = pd.DataFrame(lc["engineering"])
    cost = eng.pivot(index="model", columns="size", values="train_s")
    cost.columns = [f"train s @{label(c)}" for c in cost.columns]
    big, mid = label(sizes[2]), label(sizes[1])
    cost[f"{big} / {mid}"] = cost[f"train s @{big}"] / cost[f"train s @{mid}"]
    rss = eng.pivot(index="model", columns="size", values="rss_peak_fit_gb")
    cost[f"fit RSS peak @{big} (GB)"] = rss[sizes[2]]
    return {"levels": levels, "changes": changes, "cost": cost.reset_index()}


def _figure(lc: dict, path: Path) -> None:
    df = pd.DataFrame(lc["levels"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, metric in zip(axes, ("value@20%", "qini"), strict=True):
        for name, g in df.groupby("model", sort=False):
            v = np.array([r[metric] for _, r in g.iterrows()]) * 100
            ax.plot(g["size"], v[:, 0], marker="o", lw=1.3, label=name)
        ax.set_xscale("log")
        ax.set_xlabel("training rows (log scale)")
        ax.set_ylabel(f"AIPW {metric} (pp)")
        ax.set_title(f"{metric} vs training size (validation subset)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def build_report(lc: dict, tables: dict, n_val: int, n_boot: int, sizes=SIZES) -> str:
    md = lambda df: df.to_markdown(index=False)  # noqa: E731
    return f"""# Criteo C6 learning curve: {", ".join(label(s) for s in sizes)} training rows

**Preliminary and validation-only.** Same validation subset ({n_val:,} rows), shared out-of-sample
AIPW nuisance, fixed C3/C4 hyperparameters, seed {SEED} and {n_boot} paired Poisson bootstrap
replicates as the per-rung reports. The held-out test set was not touched and **no model has been
selected or frozen**. Values are conditional estimates for this released sample (percentage points
of visit rate per evaluated customer relative to treating nobody), not population-level causal
effects or advertising ROI; they are not evidence of final generalization performance.

## AIPW policy value and Qini by training size
{md(tables["levels"])}

## Change from adding training data (AIPW, paired 95% interval)
Each change is the bigger training set minus the smaller one for the same model, scored on the same
validation customers with shared bootstrap weights. `*` marks a paired interval that excludes 0.
Only the 20% budget and Qini are shown here; all budgets are in the JSON. Each interval covers
scoring noise on the validation subset; it does not include the variability of re-training with
another seed, so a "significant" change is not guaranteed to reproduce.

{md(tables["changes"])}

## Training cost by size
{md(tables["cost"].round(2))}

Figure: `scaling_curves.png`.

## How to read this
- Diminishing returns are judged from the paired changes above, not from the point estimates: a step
  whose interval contains 0 is not distinguishable from no improvement on this validation subset.
- AIPW remains consistent only if the constant 0.85 propensity or the nuisance model is adequate;
  the IPW/AIPW discrepancy documented in `ipw_aipw_diagnostic.md` is still present and is shown in
  each rung's report. Using one shared estimator makes comparisons across models and sizes
  consistent but does not by itself make rankings unbiased.
- A single seed and fixed, untuned hyperparameters: changes may reflect the defaults' capacity
  (for example LightGBM's `min_child_samples`, TARNet's early stopping) as much as data
  limits.
"""


def run(
    names,
    data_dir=PROCESSED_DIR,
    out_dir=None,
    n_val=VAL_ROWS,
    seed=SEED,
    n_boot=N_BOOT,
    sizes=SIZES,
    nuisance_rows=NUISANCE_ROWS,
):
    out_dir = Path(out_dir) if out_dir else REPORT_DIR / "benchmark_2m"
    out_dir.mkdir(parents=True, exist_ok=True)
    lc = learning_curve(names, data_dir, n_val, seed, n_boot, sizes, nuisance_rows)
    tables = _tables(lc, sizes)
    _figure(lc, out_dir / "scaling_curves.png")
    (out_dir / "scaling_report.md").write_text(build_report(lc, tables, n_val, n_boot, sizes))
    _json(
        out_dir / "scaling_results.json",
        {
            "status": "preliminary validation learning curve; test untouched; no model selected",
            "n_val": n_val,
            "n_boot": n_boot,
            "seed": seed,
            **lc,
        },
    )
    return out_dir / "scaling_report.md"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Criteo C6 learning-curve report.")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=DEFAULT_MODELS)
    ap.add_argument("--data-dir", default=str(PROCESSED_DIR))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)
    print("wrote", run(args.models, args.data_dir, args.out_dir, n_boot=args.n_boot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
