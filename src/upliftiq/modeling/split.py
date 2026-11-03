"""Reproducible dev/test split with stratified CV folds inside dev.

Protocol (docs/causal_modeling_plan.md, section 2): 20% untouched test set, stratified on
``segment x visit``; 5-fold stratified CV inside the remaining dev data. The whole split is one
integer array ``assignment`` aligned with the rows of the dataset:
``-1`` = test, ``0..n_folds-1`` = CV fold (the validation fold for that index).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 42
TEST_SIZE = 0.2
N_FOLDS = 5
TEST = -1


@dataclass(frozen=True)
class Split:
    assignment: np.ndarray  # shape (n_rows,), -1 = test, k = validation fold k
    seed: int = SEED
    n_folds: int = N_FOLDS

    @property
    def n_rows(self) -> int:
        return len(self.assignment)

    @property
    def dev_idx(self) -> np.ndarray:
        return np.flatnonzero(self.assignment != TEST)

    @property
    def test_idx(self) -> np.ndarray:
        return np.flatnonzero(self.assignment == TEST)

    def fold_indices(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(train, validation) row positions for CV fold ``k``; never includes test rows."""
        if not 0 <= k < self.n_folds:
            raise ValueError(f"fold must be in [0, {self.n_folds})")
        return (
            np.flatnonzero((self.assignment != TEST) & (self.assignment != k)),
            np.flatnonzero(self.assignment == k),
        )

    def fingerprint(self) -> str:
        """Hash of the assignment; recorded in later frozen configs."""
        return hashlib.sha256(self.assignment.astype(np.int8).tobytes()).hexdigest()


def strata_labels(df: pd.DataFrame) -> pd.Series:
    return df["segment"].astype(str) + "|" + df["visit"].astype(str)


def make_split(
    df: pd.DataFrame, seed: int = SEED, test_size: float = TEST_SIZE, n_folds: int = N_FOLDS
) -> Split:
    """Build the stratified dev/test split and CV fold assignment."""
    strata = strata_labels(df).to_numpy()
    positions = np.arange(len(df))
    dev_pos, test_pos = train_test_split(
        positions, test_size=test_size, stratify=strata, random_state=seed
    )
    assignment = np.full(len(df), TEST, dtype=np.int8)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    dev_pos = np.sort(dev_pos)
    for k, (_, val) in enumerate(skf.split(dev_pos, strata[dev_pos])):
        assignment[dev_pos[val]] = k
    assert (assignment[test_pos] == TEST).all()
    return Split(assignment=assignment, seed=seed, n_folds=n_folds)


def save_split(split: Split, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": split.seed,
        "n_folds": split.n_folds,
        "n_rows": split.n_rows,
        "fingerprint": split.fingerprint(),
        "assignment": split.assignment.tolist(),
    }
    path.write_text(json.dumps(payload))


def load_split(path: Path, n_rows: int | None = None) -> Split:
    payload = json.loads(path.read_text())
    split = Split(
        assignment=np.asarray(payload["assignment"], dtype=np.int8),
        seed=payload["seed"],
        n_folds=payload["n_folds"],
    )
    if split.fingerprint() != payload["fingerprint"]:
        raise ValueError("split file is corrupted (fingerprint mismatch)")
    if n_rows is not None and split.n_rows != n_rows:
        raise ValueError(f"split has {split.n_rows} rows but data has {n_rows}")
    return split
