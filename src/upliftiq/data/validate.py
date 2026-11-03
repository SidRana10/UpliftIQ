"""Schema and integrity validation for the Hillstrom dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from upliftiq.data.registry import HILLSTROM, RAW_DIR


def validate_hillstrom_frame(df: pd.DataFrame) -> list[str]:
    """Return a list of problems found (empty list means valid)."""
    errors: list[str] = []
    if tuple(df.columns) != HILLSTROM.columns:
        errors.append(f"columns mismatch: {list(df.columns)}")
        return errors
    if len(df) != HILLSTROM.n_rows:
        errors.append(f"expected {HILLSTROM.n_rows} rows, got {len(df)}")
    if df.isna().any().any():
        errors.append("contains missing values")
    if set(df["segment"].unique()) != HILLSTROM.segments:
        errors.append(f"unexpected segments: {sorted(df['segment'].unique())}")
    for col in ("mens", "womens", "newbie", "visit", "conversion"):
        if not set(df[col].unique()) <= {0, 1}:
            errors.append(f"{col} is not binary")
    if (df["spend"] < 0).any():
        errors.append("negative spend")
    if (df.loc[df["conversion"] == 0, "spend"] != 0).any():
        errors.append("spend > 0 without conversion")
    if ((df["conversion"] == 1) & (df["visit"] == 0)).any():
        errors.append("conversion without visit")
    return errors


def validate_hillstrom(path: Path = RAW_DIR / HILLSTROM.filename) -> pd.DataFrame:
    df = pd.read_csv(path)
    errors = validate_hillstrom_frame(df)
    if errors:
        raise ValueError("Hillstrom validation failed: " + "; ".join(errors))
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the Hillstrom dataset.")
    parser.add_argument("--path", type=Path, default=RAW_DIR / HILLSTROM.filename)
    args = parser.parse_args(argv)
    df = validate_hillstrom(args.path)
    print(f"OK: {len(df)} rows, {len(df.columns)} columns")
    print("Segment counts:")
    print(df["segment"].value_counts().to_string())
    print(f"Visit rate: {df['visit'].mean():.4f}  Conversion rate: {df['conversion'].mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
