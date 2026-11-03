"""Generate tables, charts and a business-facing report for the Hillstrom analysis.

Usage: python -m upliftiq.evaluation.hillstrom_report
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from upliftiq.data.registry import HILLSTROM, PROJECT_ROOT, RAW_DIR  # noqa: E402
from upliftiq.data.validate import validate_hillstrom  # noqa: E402
from upliftiq.evaluation.hillstrom_analysis import (  # noqa: E402
    ARMS,
    SEED,
    balance_table,
    covariate_profile,
    group_summary,
    spend_distribution,
    treatment_effects,
)

OUT_DIR = PROJECT_ROOT / "reports" / "hillstrom"
COLORS = {"No E-Mail": "#8c8c8c", "Mens E-Mail": "#1f77b4", "Womens E-Mail": "#d95f02"}


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_figures(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    balance: pd.DataFrame,
    effects: pd.DataFrame,
    fig_dir: Path,
) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # 1. Outcome rates and spend by arm (with 95% intervals from the effect table where possible)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, col, title, fmt in [
        (axes[0], "visit_rate", "Visit rate", "{:.1%}"),
        (axes[1], "conversion_rate", "Conversion rate", "{:.2%}"),
        (axes[2], "mean_spend_per_customer", "Mean spend per assigned customer ($)", "${:.2f}"),
    ]:
        vals = summary.set_index("segment").loc[ARMS, col]
        bars = ax.bar(ARMS, vals, color=[COLORS[a] for a in ARMS])
        for b, v in zip(bars, vals, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
    _save(fig, p := fig_dir / "outcomes_by_group.png")
    paths.append(p)

    # 2. Covariate balance (love plot)
    b = balance.sort_values("max_abs_smd_mens_vs_control")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    y = np.arange(len(b))
    ax.scatter(
        b["max_abs_smd_mens_vs_control"], y, label="Mens vs control", color=COLORS["Mens E-Mail"]
    )
    ax.scatter(
        b["max_abs_smd_womens_vs_control"],
        y,
        label="Womens vs control",
        color=COLORS["Womens E-Mail"],
        marker="s",
    )
    ax.axvline(0.1, ls="--", color="k", lw=1, label="0.1 threshold")
    ax.set_yticks(y, b["covariate"])
    ax.set_xlabel("Max absolute standardized mean difference")
    ax.set_title("Baseline balance (pre-treatment covariates)")
    ax.legend()
    _save(fig, p := fig_dir / "covariate_balance.png")
    paths.append(p)

    # 3. Covariate distributions by arm
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(
        axes.flat,
        ["recency", "history_segment", "zip_code", "channel", "newbie", "mens"],
        strict=True,
    ):
        shares = pd.crosstab(df[col], df["segment"], normalize="columns")[ARMS]
        shares.plot.bar(ax=ax, color=[COLORS[a] for a in ARMS], legend=False, width=0.8)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("share of arm")
        ax.tick_params(axis="x", rotation=30)
    axes[0, 0].legend(fontsize=8)
    _save(fig, p := fig_dir / "covariate_distributions.png")
    paths.append(p)

    # 4. Spend distribution among converters (skew)
    conv = df[df["conversion"] == 1]
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, conv["spend"].quantile(0.99), 40)
    for a in ARMS:
        ax.hist(
            conv.loc[conv["segment"] == a, "spend"], bins=bins, alpha=0.5, label=a, color=COLORS[a]
        )
    ax.set_xlabel("Spend among converters ($, clipped at 99th percentile)")
    ax.set_ylabel("customers")
    ax.set_title("Spend among converters is right-skewed")
    ax.legend()
    _save(fig, p := fig_dir / "spend_distribution.png")
    paths.append(p)

    # 5. Effect forest plot
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    for ax, outcome, scale, unit in [
        (axes[0], "visit", 100, "pp"),
        (axes[1], "conversion", 100, "pp"),
        (axes[2], "spend", 1, "$"),
    ]:
        e = effects[effects["outcome"] == outcome].reset_index(drop=True)
        for i, r in e.iterrows():
            arm = r["comparison"].split(" vs ")[0]
            ax.errorbar(
                r["abs_effect"] * scale,
                i,
                xerr=[
                    [(r["abs_effect"] - r["abs_ci_low"]) * scale],
                    [(r["abs_ci_high"] - r["abs_effect"]) * scale],
                ],
                fmt="o",
                color=COLORS[arm],
                capsize=4,
            )
        ax.axvline(0, color="k", lw=1)
        ax.set_yticks(range(len(e)), [c.split(" vs ")[0] for c in e["comparison"]])
        ax.set_ylim(-0.5, len(e) - 0.5)
        ax.set_title(f"{outcome}: absolute lift vs No E-Mail ({unit})")
    _save(fig, p := fig_dir / "treatment_effects.png")
    paths.append(p)
    return paths


def _usd(v: float) -> str:
    return f"{'-' if v < 0 else '+'}${abs(v):.3f}"


def _fmt_effect_table(effects: pd.DataFrame) -> str:
    lines = [
        "| Comparison | Outcome | Control | Treated | Abs. lift [95% CI] | Rel. lift [95% CI] "
        "| p | Holm p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in effects.iterrows():
        if r["outcome"] == "spend":
            f = lambda v: f"${v:.3f}"  # noqa: E731
            absf = lambda v: f"{'-' if v < 0 else '+'}${abs(v):.3f}"  # noqa: E731
        else:
            f = lambda v: f"{v:.2%}"  # noqa: E731
            absf = lambda v: f"{v * 100:+.2f} pp"  # noqa: E731
        lines.append(
            f"| {r['comparison']} | {r['outcome']} | {f(r['control'])} | {f(r['treated'])} | "
            f"{absf(r['abs_effect'])} [{absf(r['abs_ci_low'])}, {absf(r['abs_ci_high'])}] | "
            f"{r['rel_effect']:+.1%} [{r['rel_ci_low']:+.1%}, {r['rel_ci_high']:+.1%}] | "
            f"{r['p_value']:.2g} | {r['p_value_holm']:.2g} |"
        )
    return "\n".join(lines)


def build_report(df, summary, balance, effects, spend_dist) -> str:
    e = effects.set_index(["comparison", "outcome"])
    m, w = "Mens E-Mail vs No E-Mail", "Womens E-Mail vs No E-Mail"

    def line(cmp_, o):
        r = e.loc[(cmp_, o)]
        if o == "spend":
            est = f"{_usd(r['abs_effect'])} per customer ({r['rel_effect']:+.0%})"
            ci = f"[{_usd(r['abs_ci_low'])}, {_usd(r['abs_ci_high'])}]"
        else:
            est = f"{r['abs_effect'] * 100:+.2f} pp ({r['rel_effect']:+.0%})"
            ci = f"[{r['abs_ci_low'] * 100:+.2f}, {r['abs_ci_high'] * 100:+.2f}] pp"
        return f"{est}, 95% CI {ci}, Holm-adjusted p = {r['p_value_holm']:.2g}"

    bal_flag = balance[~balance["balanced_smd_lt_0.1"]]["covariate"].tolist()
    n_p05 = int((balance["p_value"] < 0.05).sum())
    arm_n = summary.set_index("segment")["n"]
    sig = effects[effects["significant_holm_0.05"]]
    nsig = effects[~effects["significant_holm_0.05"]]
    nonsig_txt = (
        ", ".join(f"{r.comparison.split(' vs ')[0]} {r.outcome}" for r in nsig.itertuples())
        or "none"
    )

    return f"""# Hillstrom E-Mail Experiment: Exploratory and Statistical Analysis

_Generated by `python -m upliftiq.evaluation.hillstrom_report` (seed {SEED}). Data: {HILLSTROM.name},
{len(df):,} customers; source and checksum in `docs/datasets.md` and `data/raw/hillstrom.source.json`._

## 1. Executive summary

Customers were randomly assigned to a Men's e-mail ({arm_n["Mens E-Mail"]:,}), a Women's e-mail
({arm_n["Womens E-Mail"]:,}) or no e-mail ({arm_n["No E-Mail"]:,}). Outcomes were measured over the
following two weeks.

- **Both e-mails raise site visits and conversions** relative to sending nothing. Significant effects after
  multiple-comparison correction (Holm, 6 tests): {len(sig)} of 6. Not significant: {nonsig_txt}.
- **Visits:** Men's {line(m, "visit")}; Women's {line(w, "visit")}.
- **Conversions:** Men's {line(m, "conversion")}; Women's {line(w, "conversion")}.
- **Revenue per customer mailed:** Men's {line(m, "spend")}; Women's {line(w, "spend")}.
- Spend per customer is noisy (about {summary["share_zero_spend"].mean():.1%} of customers spend nothing and
  the rest are heavily skewed), so revenue conclusions carry much more uncertainty than visit conclusions.
- **Randomization looks sound**: no pre-treatment covariate is meaningfully imbalanced
  (all max |SMD| < 0.1{"" if not bal_flag else "; exceptions: " + ", ".join(bal_flag)}), so the differences
  above can be read as causal effects of the e-mails.
- These are *average* effects. This analysis does not say *who* responds best; that is the job of the
  uplift modeling in later milestones.

## 2. Data and group composition

{len(df):,} customers, 12 columns, no missing values. Arms are close to one-third each.

{summary.to_markdown(index=False, floatfmt=".4f")}

Spend among converters is strongly right-skewed:

{spend_dist.to_markdown(index=False, floatfmt=".2f")}

Figures: `figures/outcomes_by_group.png`, `figures/spend_distribution.png`,
`figures/covariate_distributions.png`.

## 3. Baseline comparability

If assignment was truly random, pre-treatment features should look alike across arms. We compare all 8
covariates with an omnibus test across the three arms (Kruskal-Wallis for `recency`/`history`,
chi-square for the others) and standardized mean differences (SMD) of each e-mail arm against control.
Rule of thumb: |SMD| < 0.1 is good balance.

{balance.to_markdown(index=False, floatfmt=".4f")}

{n_p05} of 8 covariates have an omnibus p < 0.05; with 8 independent tests about 0.4 such results are
expected by chance, and every SMD is tiny, so this is not evidence of a broken randomization.
See `figures/covariate_balance.png`.

## 4. Treatment effects

Estimand: effect of being *assigned* to an e-mail (intention-to-treat) on each outcome per assigned customer.
Spend includes customers who spent $0, so it is revenue per customer mailed, not per buyer.

{_fmt_effect_table(effects)}

`pp` = percentage points. Relative lift = treated rate / control rate - 1. Plot: `figures/treatment_effects.png`.

## 5. Methodology

- **Binary outcomes (visit, conversion):** absolute difference with Newcombe hybrid-score 95% CI (accurate
  for rare outcomes such as the ~0.6-1.3% conversion rate); pooled two-sided z-test; relative lift CI from
  the log risk ratio (Katz).
- **Spend:** mean difference with a percentile bootstrap 95% CI (5,000 resamples per arm) and a two-sided
  permutation test (20,000 permutations, minimum attainable p = 5e-5, so a p of 5e-5 means "at or below the resolution limit"). These make no normality assumption,
  which matters for zero-heavy, skewed data. Welch's t-test p-values are kept in `effects.csv` as a
  cross-check. Relative-lift CIs for spend are bootstrap ratios.
- **Multiple comparisons:** two e-mail arms x three outcomes = 6 hypothesis tests, adjusted together with
  the Holm step-down procedure (controls family-wise error rate at 5%). The reported 95% CIs are *not*
  widened for multiplicity; use the Holm p-values to judge significance.
- **Reproducibility:** fixed seed ({SEED}); all tables and figures are regenerated from the raw CSV.

## 6. Limitations

- **Short horizon:** only two weeks of outcomes. Long-run value, fatigue and unsubscribe effects are unseen.
- **Spend estimates are imprecise:** conversions are rare ({summary["conversion_rate"].mean():.1%} overall), so spend
  CIs are wide. The relative revenue lifts have wide intervals (see table), so treat their size with caution.
- **Revenue vs. profit:** spend is revenue; e-mail costs, margins and discounts are not modeled.
- **Average effects only:** subgroup differences (e.g. by `mens`/`womens` purchase history, `channel`) are
  not tested here; doing so without correction would inflate false positives. Heterogeneity is deferred to
  uplift modeling with proper evaluation.
- **Single historical experiment:** one 2008 campaign from one retailer; generalization to other
  campaigns or periods is unproven. Conversion and spend are only observed after a visit.
- **Men's vs. Women's e-mail** were not compared head-to-head (out of scope); do not infer a winner from
  overlapping or non-overlapping CIs alone.
- Conditional-on-conversion spend (shown descriptively) must not be read causally, because conversion is
  itself affected by treatment.
"""


def run(out_dir: Path = OUT_DIR, data_path: Path = RAW_DIR / HILLSTROM.filename) -> dict:
    df = validate_hillstrom(data_path)
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    summary = group_summary(df)
    balance = balance_table(df)
    effects = treatment_effects(df)
    spend_dist = spend_distribution(df)

    summary.to_csv(tables / "group_summary.csv", index=False)
    covariate_profile(df).to_csv(tables / "covariate_profile.csv", index=False)
    balance.to_csv(tables / "balance.csv", index=False)
    effects.to_csv(tables / "effects.csv", index=False)
    spend_dist.to_csv(tables / "spend_distribution.csv", index=False)

    figs = make_figures(df, summary, balance, effects, out_dir / "figures")
    report = build_report(df, summary, balance, effects, spend_dist)
    (out_dir / "report.md").write_text(report)
    return {"effects": effects, "balance": balance, "summary": summary, "figures": figs}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Hillstrom statistical analysis.")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--data", type=Path, default=RAW_DIR / HILLSTROM.filename)
    args = parser.parse_args(argv)
    run(args.out, args.data)
    print(f"Report written to {args.out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
