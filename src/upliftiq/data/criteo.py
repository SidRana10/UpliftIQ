"""Chunked validation and summary of the Criteo uplift v2.1 dataset (compressed CSV).

The file is never loaded whole: it is streamed in chunks and only running aggregates are kept.
``treatment`` (random assignment) and ``exposure`` (whether the ad was actually shown, a
post-treatment outcome) are kept strictly apart; only ``f0``-``f11`` may be model features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from upliftiq.data.registry import CRITEO, PROJECT_ROOT, RAW_DIR

FEATURES = list(CRITEO.features)
TREATMENT = CRITEO.treatment
EXPOSURE = CRITEO.exposure
OUTCOMES = list(CRITEO.outcomes)
BINARY = [TREATMENT, EXPOSURE, *OUTCOMES]
FORBIDDEN_AS_FEATURES = set(BINARY)  # treatment, exposure (post-treatment), visit, conversion
OUT_DIR = PROJECT_ROOT / "reports" / "criteo"
QUANTILES = [0.01, 0.25, 0.5, 0.75, 0.99]


def assert_no_leakage(columns) -> None:
    """Raise unless ``columns`` are exactly the twelve anonymized features f0-f11."""
    cols = list(columns)
    leaked = FORBIDDEN_AS_FEATURES & set(cols)
    if leaked:
        raise ValueError(f"treatment/exposure/outcome columns used as features: {sorted(leaked)}")
    if sorted(cols) != sorted(FEATURES):
        raise ValueError(f"feature columns {cols} do not match {FEATURES}")


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    assert_no_leakage(X.columns)
    return X


def summarize_criteo(
    path: Path, chunksize: int = 500_000, sample_every: int = 100, seed: int = 42
) -> dict:
    """Stream the (optionally gzipped) CSV and return counts, rates and feature statistics."""
    header = list(pd.read_csv(path, nrows=0).columns)
    rng = np.random.default_rng(seed)
    n = 0
    nulls = dict.fromkeys(CRITEO.columns, 0)
    nonbinary = dict.fromkeys(BINARY, 0)
    s = dict.fromkeys(FEATURES, 0.0)
    ss = dict.fromkeys(FEATURES, 0.0)
    cnt = dict.fromkeys(FEATURES, 0)
    lo = dict.fromkeys(FEATURES, np.inf)
    hi = dict.fromkeys(FEATURES, -np.inf)
    nonfinite = dict.fromkeys(FEATURES, 0)
    groups = None
    exposed_untreated = conv_no_visit = 0
    samples = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        if list(chunk.columns) != header:
            raise ValueError("inconsistent columns across chunks")
        n += len(chunk)
        for c in CRITEO.columns:
            if c in chunk:
                nulls[c] += int(chunk[c].isna().sum())
        for c in BINARY:
            if c in chunk:
                nonbinary[c] += int((~chunk[c].isin([0, 1]) & chunk[c].notna()).sum())
        for f in FEATURES:
            if f not in chunk:
                continue
            x = chunk[f].to_numpy(dtype=float)
            fin = np.isfinite(x)
            nonfinite[f] += int((~np.isnan(x) & ~fin).sum())
            xf = x[fin]
            if len(xf):
                s[f] += xf.sum()
                ss[f] += (xf**2).sum()
                cnt[f] += len(xf)
                lo[f], hi[f] = min(lo[f], xf.min()), max(hi[f], xf.max())
        if {TREATMENT, EXPOSURE, "visit", "conversion"} <= set(chunk.columns):
            exposed_untreated += int(((chunk[EXPOSURE] == 1) & (chunk[TREATMENT] == 0)).sum())
            conv_no_visit += int(((chunk["conversion"] == 1) & (chunk["visit"] == 0)).sum())
            g = chunk.groupby([TREATMENT, EXPOSURE]).agg(
                n=("visit", "size"), visits=("visit", "sum"), conversions=("conversion", "sum")
            )
            groups = g if groups is None else groups.add(g, fill_value=0)
        samples.append(
            chunk[[f for f in FEATURES if f in chunk]][rng.random(len(chunk)) < 1 / sample_every]
        )
    sample = pd.concat(samples)

    features = {}
    for f in FEATURES:
        if cnt[f] == 0:
            continue
        mean = s[f] / cnt[f]
        features[f] = {
            "mean": mean,
            "std": float(np.sqrt(max(ss[f] / cnt[f] - mean**2, 0.0))),
            "min": lo[f],
            "max": hi[f],
            **{f"q{int(q * 100):02d}": float(sample[f].quantile(q)) for q in QUANTILES},
            "non_finite": nonfinite[f],
        }
    result = {
        "n_rows": n,
        "columns": header,
        "missing": nulls,
        "non_binary_values": nonbinary,
        "features": features,
        "quantiles_from_sample_rows": len(sample),
        "exposed_without_treatment": exposed_untreated,
        "conversion_without_visit": conv_no_visit,
    }
    if groups is not None:
        g = groups.reset_index()
        tot = lambda mask, col: float(g.loc[mask, col].sum())  # noqa: E731
        arms = {}
        for t, name in ((1, "treated"), (0, "control")):
            m = g[TREATMENT] == t
            nt = tot(m, "n")
            arms[name] = {
                "n": int(nt),
                "share": nt / n,
                "visit_rate": tot(m, "visits") / nt if nt else float("nan"),
                "conversion_rate": tot(m, "conversions") / nt if nt else float("nan"),
                "exposed_share": tot(m & (g[EXPOSURE] == 1), "n") / nt if nt else float("nan"),
            }
        result["arms"] = arms
        result["overall"] = {
            "visit_rate": float(g["visits"].sum() / n),
            "conversion_rate": float(g["conversions"].sum() / n),
        }
        result["treatment_by_exposure"] = [
            {
                "treatment": int(r[TREATMENT]),
                "exposure": int(r[EXPOSURE]),
                "n": int(r["n"]),
                "visit_rate": r["visits"] / r["n"],
                "conversion_rate": r["conversions"] / r["n"],
            }
            for _, r in g.iterrows()
        ]
    return result


def validation_errors(summary: dict, expected_rows: int = CRITEO.n_rows) -> list[str]:
    """Hard schema/integrity problems (empty list = valid). Descriptive findings are not errors."""
    errs = []
    if summary["columns"] != list(CRITEO.columns) and sorted(summary["columns"]) != sorted(
        CRITEO.columns
    ):
        errs.append(f"unexpected columns: {summary['columns']}")
    if summary["n_rows"] != expected_rows:
        errs.append(f"expected {expected_rows} rows, got {summary['n_rows']}")
    for c, k in summary["missing"].items():
        if k:
            errs.append(f"{k} missing values in {c}")
    for c, k in summary["non_binary_values"].items():
        if k:
            errs.append(f"{k} non-binary values in {c}")
    for f, st in summary["features"].items():
        if st["non_finite"]:
            errs.append(f"{st['non_finite']} non-finite values in {f}")
    if summary["exposed_without_treatment"]:
        errs.append(
            f"{summary['exposed_without_treatment']} control users flagged exposed (exposure "
            "should require treatment)"
        )
    return errs


def validate_criteo(path: Path, **kwargs) -> dict:
    summary = summarize_criteo(path, **kwargs)
    errs = validation_errors(summary)
    if errs:
        raise ValueError("Criteo validation failed: " + "; ".join(errs))
    return summary


def summary_markdown(summary: dict, errors: list[str]) -> str:
    a = summary["arms"]
    feat = pd.DataFrame(summary["features"]).T[["mean", "std", "min", "q01", "q50", "q99", "max"]]
    tx = pd.DataFrame(summary["treatment_by_exposure"])
    return f"""# Criteo Uplift v2.1: dataset summary

Validation: {"**PASSED**" if not errors else "**FAILED**: " + "; ".join(errors)}

- Rows: {summary["n_rows"]:,} (expected {CRITEO.n_rows:,}); missing values: {sum(summary["missing"].values())}
- Treatment allocation: treated {a["treated"]["share"]:.4f} ({a["treated"]["n"]:,}), control {a["control"]["share"]:.4f} ({a["control"]["n"]:,})
- Overall visit rate {summary["overall"]["visit_rate"]:.5f}, conversion rate {summary["overall"]["conversion_rate"]:.5f}
- Visit / conversion rate, treated: {a["treated"]["visit_rate"]:.5f} / {a["treated"]["conversion_rate"]:.5f}; control: {a["control"]["visit_rate"]:.5f} / {a["control"]["conversion_rate"]:.5f}
- Exposed share: treated {a["treated"]["exposed_share"]:.4f}, control {a["control"]["exposed_share"]:.4f}
- Control users flagged exposed: {summary["exposed_without_treatment"]:,}; conversions without a visit: {summary["conversion_without_visit"]:,}

These are observed descriptive rates, not causal estimates. The treated share is about 0.85, nothing like
Hillstrom's 1/3, and the file is a non-uniform subsample, so propensities must not be assumed.
`exposure` is post-treatment and must not be a feature.

## Features (quantiles from a {summary["quantiles_from_sample_rows"]:,}-row sample)

{feat.round(4).to_markdown()}

## Treatment x exposure

{tx.round({"visit_rate": 5, "conversion_rate": 5}).to_markdown(index=False)}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and summarize Criteo uplift v2.1.")
    parser.add_argument("--path", type=Path, default=RAW_DIR / CRITEO.filename)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args(argv)
    summary = summarize_criteo(args.path, chunksize=args.chunksize)
    errors = validation_errors(summary)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n"
    )
    (args.out / "dataset_summary.md").write_text(summary_markdown(summary, errors))
    print("PASSED" if not errors else "FAILED: " + "; ".join(errors))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
