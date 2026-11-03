"""Descriptive, balance and treatment-effect tables for the Hillstrom experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from upliftiq.evaluation.stats import (
    holm_adjust,
    mean_effect,
    standardized_mean_difference,
    two_proportion_effect,
)

SEED = 42
CONTROL = "No E-Mail"
ARMS = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]
TREATMENTS = ["Mens E-Mail", "Womens E-Mail"]
NUMERIC_COVARIATES = ["recency", "history"]
CATEGORICAL_COVARIATES = ["history_segment", "mens", "womens", "zip_code", "newbie", "channel"]
OUTCOMES = {"visit": "binary", "conversion": "binary", "spend": "mean"}


def group_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-arm size, outcome rates and spend distribution (zeros included)."""
    rows = []
    for arm in ARMS:
        g = df[df["segment"] == arm]
        conv = g[g["conversion"] == 1]
        rows.append(
            {
                "segment": arm,
                "n": len(g),
                "share": len(g) / len(df),
                "visit_rate": g["visit"].mean(),
                "conversion_rate": g["conversion"].mean(),
                "mean_spend_per_customer": g["spend"].mean(),
                "median_spend_per_customer": g["spend"].median(),
                "share_zero_spend": (g["spend"] == 0).mean(),
                "n_converters": len(conv),
                "mean_spend_per_converter": conv["spend"].mean() if len(conv) else np.nan,
                "max_spend": g["spend"].max(),
            }
        )
    return pd.DataFrame(rows)


def covariate_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Overall covariate distributions (numeric: describe; categorical: level shares)."""
    rows = []
    for c in NUMERIC_COVARIATES:
        d = df[c].describe()
        rows.append(
            {
                "covariate": c,
                "level": "mean / median / sd",
                "value": f"{d['mean']:.2f} / {df[c].median():.2f} / {d['std']:.2f}",
            }
        )
    for c in CATEGORICAL_COVARIATES:
        for level, share in df[c].value_counts(normalize=True).sort_index().items():
            rows.append({"covariate": c, "level": str(level), "value": f"{share:.1%}"})
    return pd.DataFrame(rows)


def balance_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-treatment covariate balance across the three arms.

    Omnibus test: Kruskal-Wallis (numeric) or chi-square (categorical) across all arms.
    SMD: max absolute standardized mean difference of each treated arm vs control
    (categorical covariates use one 0/1 indicator per level).
    """
    rows = []
    ctrl = df[df["segment"] == CONTROL]
    for c in NUMERIC_COVARIATES + CATEGORICAL_COVARIATES:
        if c in NUMERIC_COVARIATES:
            groups = [df.loc[df["segment"] == a, c] for a in ARMS]
            p = float(stats.kruskal(*groups).pvalue)
            test = "Kruskal-Wallis"
            smds = {
                t: abs(standardized_mean_difference(df.loc[df["segment"] == t, c], ctrl[c]))
                for t in TREATMENTS
            }
        else:
            p = float(stats.chi2_contingency(pd.crosstab(df["segment"], df[c]))[1])
            test = "chi-square"
            smds = {}
            for t in TREATMENTS:
                g = df[df["segment"] == t]
                smds[t] = max(
                    abs(
                        standardized_mean_difference(
                            (g[c] == lv).astype(float), (ctrl[c] == lv).astype(float)
                        )
                    )
                    for lv in df[c].unique()
                )
        rows.append(
            {
                "covariate": c,
                "test": test,
                "p_value": p,
                "max_abs_smd_mens_vs_control": smds["Mens E-Mail"],
                "max_abs_smd_womens_vs_control": smds["Womens E-Mail"],
                "balanced_smd_lt_0.1": max(smds.values()) < 0.1,
            }
        )
    return pd.DataFrame(rows)


def treatment_effects(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Each email arm vs No E-Mail for visit, conversion and spend per assigned customer.

    Holm adjustment is applied across all six tests (2 comparisons x 3 outcomes).
    """
    ctrl = df[df["segment"] == CONTROL]
    rows = []
    for t in TREATMENTS:
        g = df[df["segment"] == t]
        for outcome, kind in OUTCOMES.items():
            if kind == "binary":
                res = two_proportion_effect(
                    int(g[outcome].sum()), len(g), int(ctrl[outcome].sum()), len(ctrl)
                )
                welch_p = np.nan
            else:
                res = mean_effect(g[outcome].to_numpy(), ctrl[outcome].to_numpy(), seed=seed)
                welch_p = float(stats.ttest_ind(g[outcome], ctrl[outcome], equal_var=False).pvalue)
            rows.append(
                {
                    "comparison": f"{t} vs {CONTROL}",
                    "outcome": outcome,
                    "n_treated": len(g),
                    "n_control": len(ctrl),
                    "control": res.control_estimate,
                    "treated": res.treated_estimate,
                    "abs_effect": res.abs_effect,
                    "abs_ci_low": res.abs_ci[0],
                    "abs_ci_high": res.abs_ci[1],
                    "rel_effect": res.rel_effect,
                    "rel_ci_low": res.rel_ci[0],
                    "rel_ci_high": res.rel_ci[1],
                    "p_value": res.p_value,
                    "welch_p_value": welch_p,
                    "method": res.method,
                }
            )
    out = pd.DataFrame(rows)
    out["p_value_holm"] = holm_adjust(out["p_value"].to_numpy())
    out["significant_holm_0.05"] = out["p_value_holm"] < 0.05
    return out


def spend_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Quantiles of spend among converters, by arm (shows the skew)."""
    conv = df[df["conversion"] == 1]
    q = conv.groupby("segment")["spend"].quantile([0.25, 0.5, 0.75, 0.9, 0.99]).unstack()
    q.columns = [f"p{int(c * 100)}" for c in q.columns]
    q["skewness"] = conv.groupby("segment")["spend"].apply(lambda s: float(stats.skew(s)))
    return q.reset_index()
