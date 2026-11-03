"""Write the Stage 4 artifacts: tables, frozen config, figure and the plain-language report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from upliftiq.modeling.select import BASELINES, FAMILIES  # noqa: E402

LABELS = {
    "no_email": "No E-Mail for all",
    "mens_for_all": "Men's E-Mail for all",
    "womens_for_all": "Women's E-Mail for all",
    "random": "Uniform random",
    "logistic": "Logistic (interactions)",
    "s_learner": "LightGBM S-learner",
    "t_learner": "LightGBM T-learner",
    "dr_learner": "DR-learner",
}


def _pp(v: float) -> str:
    return f"{v * 100:+.2f} pp"


def _ci(lo: float, hi: float) -> str:
    return f"[{lo * 100:+.2f}, {hi * 100:+.2f}] pp"


def _figure(summary: pd.DataFrame, path: Path) -> None:
    s = summary.set_index("policy").loc[[*BASELINES, *FAMILIES]]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = np.arange(len(s))[::-1]
    colors = ["#8c8c8c"] * len(BASELINES) + ["#1f77b4"] * len(FAMILIES)
    ax.errorbar(
        s["aipw_value"] * 100,
        y,
        xerr=[(s["aipw_value"] - s["ci_low"]) * 100, (s["ci_high"] - s["aipw_value"]) * 100],
        fmt="none",
        ecolor="k",
        capsize=3,
    )
    ax.scatter(s["aipw_value"] * 100, y, c=colors, zorder=3)
    ax.set_yticks(y, [LABELS[p] for p in s.index])
    ax.set_xlabel("Cross-validated visit rate under the policy (%, AIPW, 95% bootstrap CI)")
    ax.set_title("Development-set policy values (out-of-fold)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def build_report(result: dict, cfg: dict) -> str:
    s = result["summary"].set_index("policy")
    d = result["decision"]
    chosen = d["chosen"]
    n = result["n_dev"]
    best_fam = max(FAMILIES, key=lambda f: s.loc[f, "aipw_value"])
    mens = s.loc[best_fam]
    vs_m = (
        mens["diff_vs_mens_for_all"],
        mens["diff_vs_mens_for_all_ci_low"],
        mens["diff_vs_mens_for_all_ci_high"],
    )
    supported = vs_m[1] > 0

    if chosen in BASELINES:
        headline = (
            f"**Selected policy: {LABELS[chosen]}.** No personalized model earned selection under the "
            "pre-specified rule."
        )
    else:
        headline = (
            f"**Selected policy: {LABELS[chosen]}** ({cfg['chosen_policy']['config']}) "
            "- a personalized policy that satisfied the baseline-beating requirement."
        )
    personalization = (
        f"The best-valued personalized model ({LABELS[best_fam]}) differs from Men's-for-all by "
        f"{_pp(vs_m[0])} {_ci(vs_m[1], vs_m[2])}. "
        + (
            "The interval excludes zero, so the development data support a gain from personalization."
            if supported
            else "The interval includes zero, so the development data do **not** show that "
            "personalization beats sending the Men's e-mail to everyone."
        )
    )

    cols = ["policy", "config", "aipw_value", "ci_low", "ci_high", "ipw_value"]
    tab = s.reset_index()[cols].copy()
    tab["policy"] = tab["policy"].map(LABELS)
    for c in ("aipw_value", "ci_low", "ci_high", "ipw_value"):
        tab[c] = (tab[c] * 100).round(2)
    tab.columns = ["Policy", "Config", "AIPW visit rate %", "CI low", "CI high", "IPW check %"]

    diff_rows = []
    for f in FAMILIES:
        r = s.loc[f]
        row = {"Model": LABELS[f]}
        for b in BASELINES:
            row[f"vs {LABELS[b]}"] = (
                f"{_pp(r[f'diff_vs_{b}'])} {_ci(r[f'diff_vs_{b}_ci_low'], r[f'diff_vs_{b}_ci_high'])}"
            )
        diff_rows.append(row)
    diffs = pd.DataFrame(diff_rows)

    cv = result["cv_results"]
    grid = cv[cv["family"] != "baseline"][
        [
            "policy",
            "aipw_value",
            "aipw_se",
            "ipw_value",
            "share_no_email",
            "share_mens",
            "share_womens",
            "family_best",
        ]
    ].copy()
    for c in grid.columns[1:7]:
        grid[c] = grid[c].round(4)

    return f"""# Hillstrom Model Selection (Stage 4, development data only)

{headline}

{personalization}

Scope: {n:,} development customers (80% of Hillstrom), 5-fold stratified cross-validation, primary outcome
= visit. **The held-out 12,800-customer test set was not accessed.** Numbers below are out-of-fold
estimates on development data; the one-time test evaluation is a later step.

## 1. Policy values (out-of-fold, AIPW; IPW as a check)

{tab.to_markdown(index=False)}

AIPW uses a fixed T-learner, cross-fitted so each customer's outcome prediction never comes from a model
trained on that customer. IPW needs no outcome model; agreement between the two columns is a sanity check.
Intervals are 95% percentile bootstrap intervals over customers ({cfg["evaluation"]["bootstrap"]["n_boot"]:,}
resamples, seed {cfg["seed"]}). Plot: `value_comparison.png`.

## 2. Paired differences (candidate minus baseline, 95% bootstrap CI)

Differences are computed on the same customers, so they are tighter than the overlap of the intervals above suggests.

{diffs.to_markdown(index=False)}

## 3. Selection

- Within each model family, the simplest hyperparameter setting within one paired standard error of the best
  was kept (`family_best` in `cv_results.csv`).
- Across options, the best by CV value was **{LABELS.get(d["best_by_value"], d["best_by_value"])}**; the rule
  then picks the simplest option within one paired SE of it (order: fixed baselines, logistic, S, T, DR).
- Personalized models below the best fixed baseline ({LABELS[d["best_baseline"]]}) are ineligible:
  {", ".join(LABELS[m] for m in d["ineligible_models"]) or "none"}.
- Result: **{LABELS.get(chosen, chosen)}**.

Full grid (policy = family:hyperparameters; shares are the fraction of customers recommended each arm):

{grid.to_markdown(index=False)}

## 4. Caveats

- Hyperparameters were chosen on the same out-of-fold data used to compare models, so the reported CV values
  of tuned models are slightly optimistic (22 settings were tried). The untouched test set is the unbiased check.
- Bootstrap intervals reflect sampling variability of customers, not the variability from refitting
  models; per-fold values are in `cv_results.csv`.
- Visit is a proxy: this stage does not evaluate revenue or conversion (planned later), and policies ignore
  e-mail cost.
- Qini curves and decile calibration from the protocol are not part of this stage's selection (the one-SE rule
  with a simplicity order makes them tie-breakers only); they are not reported here.
- A single two-week experiment from 2008 limits generalization.

## 5. Reproducibility

Frozen configuration: `frozen_config.json` (chosen policy, seed, split fingerprint
`{cfg["split"]["fingerprint"][:16]}...`, evaluation settings, data checksum, git commit). Regenerate with
`python -m upliftiq.modeling.select`.
"""


def write_artifacts(result: dict, cfg: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result["cv_results"].to_csv(out_dir / "cv_results.csv", index=False)
    result["summary"].to_csv(out_dir / "selection_summary.csv", index=False)
    (out_dir / "frozen_config.json").write_text(json.dumps(cfg, indent=2, default=float) + "\n")
    _figure(result["summary"], out_dir / "value_comparison.png")
    (out_dir / "report.md").write_text(build_report(result, cfg))
