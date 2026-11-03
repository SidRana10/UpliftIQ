"""Feature whitelist, treatment encoding and fold-safe preprocessing.

Only pre-treatment covariates may enter the feature matrix. Outcomes (visit, conversion, spend)
and the treatment (segment) are kept out of X; ``history_segment`` is a binned copy of
``history`` and is dropped as redundant (docs/causal_modeling_plan.md, section 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from upliftiq.modeling.split import Split

NUMERIC = ["recency", "history"]
BINARY = ["mens", "womens", "newbie"]
CATEGORICAL = ["zip_code", "channel"]
FEATURES = NUMERIC + BINARY + CATEGORICAL  # the seven approved pre-treatment features

TREATMENT_COL = "segment"
ARMS = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]  # code 0 = control
OUTCOMES = ["visit", "conversion", "spend"]
FORBIDDEN = set(OUTCOMES) | {TREATMENT_COL, "history_segment"}
PROPENSITY = 1 / 3  # known from randomization (design value, not estimated)


def encode_arm(segment: pd.Series) -> np.ndarray:
    """Map arm names to integer codes (0 = No E-Mail, 1 = Mens, 2 = Womens)."""
    codes = segment.map({a: i for i, a in enumerate(ARMS)})
    if codes.isna().any():
        raise ValueError(f"unknown treatment arm(s): {sorted(segment[codes.isna()].unique())}")
    return codes.to_numpy(dtype=int)


def assert_no_leakage(columns) -> None:
    """Raise if a column list is not exactly the approved pre-treatment features."""
    cols = list(columns)
    leaked = FORBIDDEN & set(cols)
    if leaked:
        raise ValueError(f"post-treatment / treatment columns in features: {sorted(leaked)}")
    if sorted(cols) != sorted(FEATURES):
        raise ValueError(f"feature columns {cols} do not match approved {FEATURES}")


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the approved feature columns (in canonical order)."""
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise KeyError(f"missing feature columns: {sorted(missing)}")
    X = df[FEATURES].copy()
    assert_no_leakage(X.columns)
    return X


def make_preprocessor(kind: str = "linear") -> ColumnTransformer:
    """Unfitted preprocessor. ``linear``: log1p+scale numerics, one-hot categoricals.

    ``tree``: numerics/binaries pass through, categoricals ordinal-encoded and placed last
    (see ``tree_categorical_indices``); unseen categories map to -1.
    """
    if kind == "linear":
        return ColumnTransformer(
            [
                (
                    "history",
                    Pipeline(
                        [
                            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    ["history"],
                ),
                ("recency", StandardScaler(), ["recency"]),
                ("binary", "passthrough", BINARY),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ]
        )
    if kind == "tree":
        return ColumnTransformer(
            [
                ("num", "passthrough", ["recency", "history", *BINARY]),
                (
                    "cat",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                    CATEGORICAL,
                ),
            ]
        )
    raise ValueError("kind must be 'linear' or 'tree'")


def tree_categorical_indices() -> list[int]:
    n_num = len(NUMERIC) + len(BINARY)
    return list(range(n_num, n_num + len(CATEGORICAL)))


@dataclass
class PreparedData:
    """Model-ready matrices. ``X_*`` come from the preprocessor fitted on ``train`` only."""

    X_train: np.ndarray
    X_eval: np.ndarray
    arm_train: np.ndarray
    arm_eval: np.ndarray
    y_train: pd.DataFrame  # outcomes, kept outside X
    y_eval: pd.DataFrame
    preprocessor: ColumnTransformer


def _prepare(df: pd.DataFrame, train_idx, eval_idx, kind: str) -> PreparedData:
    train, evl = df.iloc[train_idx], df.iloc[eval_idx]
    pre = make_preprocessor(kind)
    X_train = pre.fit_transform(select_features(train))  # fit on training rows only
    X_eval = pre.transform(select_features(evl))
    return PreparedData(
        X_train,
        X_eval,
        encode_arm(train[TREATMENT_COL]),
        encode_arm(evl[TREATMENT_COL]),
        train[OUTCOMES].reset_index(drop=True),
        evl[OUTCOMES].reset_index(drop=True),
        pre,
    )


def prepare_fold(df: pd.DataFrame, split: Split, k: int, kind: str = "linear") -> PreparedData:
    """CV fold ``k``: preprocessor fitted on the fold's training rows, applied to validation."""
    train_idx, val_idx = split.fold_indices(k)
    return _prepare(df, train_idx, val_idx, kind)


def prepare_dev_test(df: pd.DataFrame, split: Split, kind: str = "linear") -> PreparedData:
    """Final refit: preprocessor fitted on all dev rows, applied to test (Stage 5 only)."""
    return _prepare(df, split.dev_idx, split.test_idx, kind)
