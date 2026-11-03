"""Write Stage 5 artifacts: tables, figures, evaluation record and the final report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from upliftiq.data.registry import PROJECT_ROOT  # noqa: E402

LABELS = {
    "frozen_policy": "Frozen policy",
    "no_email": "No E-Mail for all",
    "mens_for_all": "Men's E-Mail for all",
    "womens_for_all": "Women's E-Mail for all",
    "random": "Uniform random",
}
OUTCOME_UNIT = {"visit": ("%", 100), "conversion": ("%", 100), "spend": ("$", 1)}
SELECTION_SUMMARY = PROJECT_ROOT / "reports" / "model_selection" / "selection_summary.csv"


def _fmt(outcome: str, v: float, signed: bool = False) -> str:
    if outcome == "spend":
        return f"{'-' if v < 0 else ('+' if signed else '')}${abs(v):.3f}"
    return f"{v * 100:+.2f} pp" if signed else f"{v * 100:.2f}%"


def _ci(outcome: str, lo: float, hi: float, signed: bool = False) -> str:
    return f"[{_fmt(outcome, lo, signed)}, {_fmt(outcome, hi, signed)}]"


def _p(p: float) -> str:
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def _values_table(values: pd.DataFrame, outcome: str) -> str:
    v = values[values["outcome"] == outcome]
    rows = []
    for pol in LABELS:
        row = {"Policy": LABELS[pol]}
        for est in ("AIPW", "IPW"):
            r = v[(v["estimator"] == est) & (v["policy"] == pol)].iloc[0]
            row[est] = f"{_fmt(outcome, r['value'])} {_ci(outcome, r['ci_low'], r['ci_high'])}"
        rows.append(row)
    return pd.DataFrame(rows).to_markdown(index=False)


def _comparison_table(comps: pd.DataFrame, outcome: str) -> str:
    c = comps[comps["outcome"] == outcome]
    rows = []
    for base in c["baseline"].unique():
        row = {
            "Frozen policy minus": LABELS[base]
            + (" (primary)" if c[c["baseline"] == base]["primary"].any() else "")
        }
        for est in ("AIPW", "IPW"):
            r = c[(c["estimator"] == est) & (c["baseline"] == base)].iloc[0]
            row[est] = (
                f"{_fmt(outcome, r['diff'], True)} {_ci(outcome, r['ci_low'], r['ci_high'], True)}, "
                f"p={_p(r['p_value'])}, Holm p={_p(r['p_value_holm'])}"
            )
        rows.append(row)
    return pd.DataFrame(rows).to_markdown(index=False)


def _dev_evidence() -> str:
    """One sentence on the development-stage evidence for personalization (from Stage 4)."""
    if not SELECTION_SUMMARY.exists():
        return ""
    s = pd.read_csv(SELECTION_SUMMARY).set_index("policy")
    fam = [f for f in ("logistic", "s_learner", "t_learner", "dr_learner") if f in s.index]
    best = max(fam, key=lambda f: s.loc[f, "aipw_value"])
    r = s.loc[best]
    return (
        f"In development cross-validation the best personalized model ({best}) differed from "
        f"Men's-for-all by {r['diff_vs_mens_for_all'] * 100:+.2f} pp "
        f"[{r['diff_vs_mens_for_all_ci_low'] * 100:+.2f}, {r['diff_vs_mens_for_all_ci_high'] * 100:+.2f}] pp, "
        "an interval that includes zero."
    )


def _figures(result: dict, out_dir: Path) -> None:
    values = result["policy_values"]
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def forest(ax, outcome, title):
        v = values[(values["outcome"] == outcome) & (values["estimator"] == "AIPW")]
        v = v[v["policy"] != "frozen_policy"].set_index("policy").loc[list(LABELS)[1:]]
        unit, k = OUTCOME_UNIT[outcome]
        y = np.arange(len(v))[::-1]
        ax.errorbar(
            v["value"] * k,
            y,
            xerr=[(v["value"] - v["ci_low"]) * k, (v["ci_high"] - v["value"]) * k],
            fmt="o",
            color="#1f77b4",
            ecolor="k",
            capsize=3,
        )
        ax.set_yticks(y, [LABELS[p] for p in v.index])
        ax.set_title(title)
        ax.set_xlabel("expected " + ("rate (%)" if unit == "%" else "spend per customer ($)"))

    fig, ax = plt.subplots(figsize=(7, 3.8))
    forest(ax, "visit", "Primary: visit rate by policy (held-out, AIPW, 95% CI)")
    fig.tight_layout()
    fig.savefig(fig_dir / "primary_visit_policy_values.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    forest(axes[0], "conversion", "Exploratory: conversion rate")
    forest(axes[1], "spend", "Exploratory: mean spend per customer")
    axes[1].set_yticklabels([])
    fig.tight_layout()
    fig.savefig(fig_dir / "secondary_policy_values.png", dpi=130)
    plt.close(fig)

    obs = result["observed_arms"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (col, lo, hi, title, k) in zip(
        axes,
        [
            ("visit_rate", "visit_ci_low", "visit_ci_high", "Observed visit rate (%)", 100),
            (
                "conversion_rate",
                "conversion_ci_low",
                "conversion_ci_high",
                "Observed conversion rate (%)",
                100,
            ),
            (
                "mean_spend_per_customer",
                "spend_ci_low",
                "spend_ci_high",
                "Observed mean spend ($)",
                1,
            ),
        ],
        strict=True,
    ):
        ax.bar(
            obs["arm"],
            obs[col] * k,
            color=["#8c8c8c", "#1f77b4", "#d95f02"],
            yerr=[(obs[col] - obs[lo]) * k, (obs[hi] - obs[col]) * k],
            capsize=4,
        )
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(fig_dir / "observed_outcomes.png", dpi=130)
    plt.close(fig)


def build_report(result: dict, cfg: dict, record: dict) -> str:
    values, comps = result["policy_values"], result["comparisons"]
    chosen = cfg["chosen_policy"]
    name = LABELS.get(chosen.get("name", ""), chosen.get("config", "personalized model"))
    primary = record["primary_comparator"]
    prim = comps[
        (comps["outcome"] == "visit")
        & (comps["estimator"] == "AIPW")
        & (comps["baseline"] == primary)
    ].iloc[0]
    fp = values[
        (values["outcome"] == "visit")
        & (values["estimator"] == "AIPW")
        & (values["policy"] == "frozen_policy")
    ].iloc[0]
    supported = prim["ci_low"] > 0
    rehearsal = record.get("rehearsal", False)
    scope = (
        "**REHEARSAL on development data only (fold 0 as pseudo-holdout); not the final result.**"
        if rehearsal
        else f"**{result['n_holdout']:,} held-out customers** that were not used for any fitting, tuning or selection."
    )
    obs_cols = [
        "arm",
        "n",
        "visit_rate",
        "conversion_rate",
        "mean_spend_per_customer",
        "share_zero_spend",
    ]
    obs = result["observed_arms"][obs_cols].copy()
    for c in ("visit_rate", "conversion_rate", "share_zero_spend"):
        obs[c] = (obs[c] * 100).round(2)
    obs["mean_spend_per_customer"] = obs["mean_spend_per_customer"].round(3)
    eff = result["observed_effects"][
        ["comparison", "outcome", "abs_effect", "abs_ci_low", "abs_ci_high", "p_value_holm"]
    ].copy()
    eff = eff.round(4)

    return f"""# Final Held-Out Evaluation (Stage 5)

**Frozen policy: {name}.** Evaluated once on {scope}

**Primary result (visit rate, AIPW):** {name} yields an expected visit rate of
{_fmt("visit", fp["value"])} {_ci("visit", fp["ci_low"], fp["ci_high"])}. Against the strongest alternative
fixed baseline chosen on development data ({LABELS[primary]}), the difference is
{_fmt("visit", prim["diff"], True)} {_ci("visit", prim["ci_low"], prim["ci_high"], True)}
(Holm p = {_p(prim["p_value_holm"])}), which {"excludes" if supported else "includes"} zero.

Outcome roles: **visit is the primary, pre-specified outcome. Conversion and spend are exploratory**;
conversions are rare, spend is 99% zeros, and no multiplicity control spans outcomes.

## 1. Primary outcome: visit rate (modeled policy estimates)

Expected visit rate if every held-out customer followed each policy. IPW needs no outcome model;
AIPW adds out-of-sample outcome predictions from a T-learner fitted on development data only. Intervals:
95% percentile bootstrap over customers ({record.get("n_boot", "n/a")} resamples).

{_values_table(values, "visit")}

Paired comparisons (same customers; Holm adjustment across the baselines compared):

{_comparison_table(comps, "visit")}

Plot: `figures/primary_visit_policy_values.png`.

## 2. Secondary outcomes (exploratory)

### Conversion rate

{_values_table(values, "conversion")}

{_comparison_table(comps, "conversion")}

### Mean spend per assigned customer (zeros included)

Outcome model for AIPW: per-arm LightGBM Tweedie regressors (fixed, untuned, fitted on development data).

{_values_table(values, "spend")}

{_comparison_table(comps, "spend")}

Plot: `figures/secondary_policy_values.png`. Spend is heavily right-skewed, so its intervals are wide;
treat these results as descriptive.

## 3. Observed randomized outcomes (no model)

Raw held-out outcomes by assigned arm, and the randomized contrasts versus No E-Mail with the
Milestone 1 methods (Newcombe/permutation/bootstrap; Holm across six tests). These are observed
experimental results, not model-based policy estimates; the Men's-for-all policy value above is
essentially the Men's arm's observed mean.

{obs.to_markdown(index=False)}

{eff.to_markdown(index=False)}

Plot: `figures/observed_outcomes.png`.

## 4. What was frozen and what was fitted

- The policy, seed, split and evaluation settings come from `reports/model_selection/frozen_config.json`,
  verified (checksum, split fingerprint, committed file) before scoring. Nothing was retuned.
- The test rows were used only to compute predictions and to score outcomes. Every fitted component
  (preprocessing, AIPW outcome models) was fitted on the {result["n_train"]:,} development rows.
- The held-out evaluation was run once; results were not revised afterwards.

## 5. What this does and does not show

- It estimates how well a fixed "Men's e-mail for everyone" rule performs on new customers from the same
  experiment, relative to sending nothing, Women's e-mail or random assignment.
- It does **not** show that personalized machine learning improves targeting. {_dev_evidence()} No
  personalized model was evaluated on the test set, and none was selected.

## 6. Limitations

- One 2008 experiment, two-week outcomes; no costs or margins. Visit is a proxy for revenue.
- Test-set intervals reflect customer sampling only. About {result["n_holdout"]:,} customers resolve policy
  differences of roughly 1 percentage point on visit; conversion and spend are much noisier.
- Policy comparisons here are for fixed rules and are not a Men's-vs-Women's causal heterogeneity analysis.
"""


def write_artifacts(result: dict, cfg: dict, record: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result["policy_values"].to_csv(out_dir / "policy_values.csv", index=False)
    result["comparisons"].to_csv(out_dir / "comparisons.csv", index=False)
    result["observed_arms"].to_csv(out_dir / "observed_arms.csv", index=False)
    result["observed_effects"].to_csv(out_dir / "observed_effects.csv", index=False)
    _figures(result, out_dir)
    (out_dir / "report.md").write_text(build_report(result, cfg, record))
    # The record is written last: its presence marks a completed one-shot evaluation.
    (out_dir / "evaluation_record.json").write_text(
        json.dumps(record, indent=2, default=float) + "\n"
    )
