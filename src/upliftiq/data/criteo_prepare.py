"""Criteo C0: convert to memory-mapped arrays, make the 64/16/20 split, fit train-only scaling.

Layout in ``data/processed/criteo/`` (git-ignored): ``X.npy`` (float32, rows x 12 features) and
separate uint8 arrays ``treatment.npy``, ``exposure.npy``, ``visit.npy``, ``conversion.npy``.
Features and labels never share a file. The split depends on ``treatment`` only (no outcome).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from upliftiq.data.criteo import FEATURES, assert_no_leakage
from upliftiq.data.registry import CRITEO, PROJECT_ROOT, RAW_DIR

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "criteo"
REPORT_DIR = PROJECT_ROOT / "reports" / "criteo"
LABELS = ["treatment", "exposure", "visit", "conversion"]
SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.64, 0.16  # test gets the remaining 20%
TRAIN, VAL, TEST = 0, 1, 2
CLIP_PERCENTILES = (0.1, 99.9)


def peak_rss_gb() -> float:
    """Peak resident memory of this process (macOS reports bytes, Linux kilobytes)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024**3 if sys.platform == "darwin" else 1024**2)


# ---------------------------------------------------------------- conversion


def convert_to_npy(
    csv_path: Path,
    out_dir: Path = PROCESSED_DIR,
    expected_rows: int = CRITEO.n_rows,
    chunksize: int = 1_000_000,
) -> dict:
    """Stream the gzipped CSV into float32 / uint8 ``.npy`` files; return a provenance manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    X = np.lib.format.open_memmap(
        out_dir / "X.npy", mode="w+", dtype=np.float32, shape=(expected_rows, len(FEATURES))
    )
    labels = {
        k: np.lib.format.open_memmap(
            out_dir / f"{k}.npy", mode="w+", dtype=np.uint8, shape=(expected_rows,)
        )
        for k in LABELS
    }
    pos = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        if set(chunk.columns) != set(CRITEO.columns):
            raise ValueError(f"unexpected columns: {list(chunk.columns)}")
        end = pos + len(chunk)
        if end > expected_rows:
            raise ValueError(f"file has more than the expected {expected_rows} rows")
        feats = chunk[FEATURES].to_numpy(dtype=np.float64)
        if not np.isfinite(feats).all():
            raise ValueError("non-finite feature values")
        X[pos:end] = feats.astype(np.float32)
        for k in LABELS:
            v = chunk[k].to_numpy()
            if not np.isin(v, [0, 1]).all():
                raise ValueError(f"non-binary values in {k}")
            labels[k][pos:end] = v.astype(np.uint8)
        pos = end
    if pos != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {pos}")
    X.flush()
    for a in labels.values():
        a.flush()
    manifest = {
        "rows": pos,
        "features": FEATURES,
        "feature_dtype": "float32",
        "label_files": LABELS,
        "source_file": csv_path.name,
        "source_sha256": sha256_file(csv_path),
        "source_sha256_matches_registry": sha256_file(csv_path) == CRITEO.sha256,
        "dataset_version": CRITEO.version,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def load_arrays(out_dir: Path = PROCESSED_DIR) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Memory-mapped features and (separately) the label arrays."""
    X = np.load(out_dir / "X.npy", mmap_mode="r")
    labels = {k: np.load(out_dir / f"{k}.npy", mmap_mode="r") for k in LABELS}
    assert X.shape[1] == len(FEATURES)
    return X, labels


# ---------------------------------------------------------------- split


@dataclass(frozen=True)
class CriteoSplit:
    assignment: np.ndarray  # int8: 0 train, 1 validation, 2 test
    seed: int = SEED

    def idx(self, part: int) -> np.ndarray:
        return np.flatnonzero(self.assignment == part)

    @property
    def train_idx(self) -> np.ndarray:
        return self.idx(TRAIN)

    @property
    def val_idx(self) -> np.ndarray:
        return self.idx(VAL)

    @property
    def test_idx(self) -> np.ndarray:
        return self.idx(TEST)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.assignment.astype(np.int8).tobytes()).hexdigest()

    def sizes(self) -> dict[str, int]:
        return {
            n: int((self.assignment == p).sum())
            for n, p in (("train", TRAIN), ("validation", VAL), ("test", TEST))
        }


def make_split(treatment: np.ndarray, seed: int = SEED) -> CriteoSplit:
    """64/16/20 split stratified by treatment assignment only (no outcome is read)."""
    treatment = np.asarray(treatment)
    rng = np.random.default_rng(seed)
    assignment = np.full(len(treatment), TEST, dtype=np.int8)
    for t in (0, 1):
        rows = rng.permutation(np.flatnonzero(treatment == t))
        n_train, n_val = round(TRAIN_FRAC * len(rows)), round(VAL_FRAC * len(rows))
        assignment[rows[:n_train]] = TRAIN
        assignment[rows[n_train : n_train + n_val]] = VAL
    return CriteoSplit(assignment, seed)


def train_subset(split: CriteoSplit, treatment: np.ndarray, n: int, seed: int = SEED) -> np.ndarray:
    """Nested, treatment-stratified subset of TRAIN rows (for the development ladder)."""
    train = split.train_idx
    if n >= len(train):
        return train
    rng = np.random.default_rng(seed)
    chosen = []
    for t in (0, 1):
        rows = train[np.asarray(treatment)[train] == t]
        k = round(n * len(rows) / len(train))
        chosen.append(rng.permutation(rows)[:k])
    return np.sort(np.concatenate(chosen))


def save_split(split: CriteoSplit, out_dir: Path = PROCESSED_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "split.npy", split.assignment)
    (out_dir / "split.json").write_text(
        json.dumps(
            {
                "seed": split.seed,
                "fingerprint": split.fingerprint(),
                "sizes": split.sizes(),
                "stratified_by": "treatment",
            },
            indent=2,
        )
        + "\n"
    )


def load_split(out_dir: Path = PROCESSED_DIR) -> CriteoSplit:
    meta = json.loads((out_dir / "split.json").read_text())
    split = CriteoSplit(np.load(out_dir / "split.npy"), meta["seed"])
    if split.fingerprint() != meta["fingerprint"]:
        raise ValueError("split.npy does not match its recorded fingerprint")
    return split


# ---------------------------------------------------------------- preprocessing


@dataclass
class FeatureScaler:
    """Clip at train percentiles, then standardize. Every statistic comes from TRAIN rows only."""

    lo: np.ndarray
    hi: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    def transform(self, X: np.ndarray, idx: np.ndarray, chunk: int = 1_000_000) -> np.ndarray:
        out = np.empty((len(idx), X.shape[1]), dtype=np.float32)
        for s in range(0, len(idx), chunk):
            block = np.asarray(X[idx[s : s + chunk]], dtype=np.float32)
            block = np.clip(block, self.lo, self.hi)
            out[s : s + chunk] = (block - self.mean) / self.std
        return out

    def to_dict(self) -> dict:
        return {k: getattr(self, k).tolist() for k in ("lo", "hi", "mean", "std")}


def fit_scaler(X: np.ndarray, train_idx: np.ndarray, percentiles=CLIP_PERCENTILES) -> FeatureScaler:
    p_lo, p_hi = percentiles
    lo, hi, mean, std = [], [], [], []
    for j in range(X.shape[1]):
        col = np.asarray(X[:, j])[train_idx].astype(np.float64)
        a, b = np.percentile(col, [p_lo, p_hi])
        col = np.clip(col, a, b)
        lo.append(a), hi.append(b), mean.append(col.mean())
        std.append(col.std() if col.std() > 0 else 1.0)
    return FeatureScaler(*(np.asarray(v, dtype=np.float32) for v in (lo, hi, mean, std)))


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Criteo C0: npy conversion, split, scaler.")
    parser.add_argument("--raw", type=Path, default=RAW_DIR / CRITEO.filename)
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    manifest_path = args.out / "manifest.json"
    if args.force or not manifest_path.exists():
        convert_to_npy(args.raw, args.out)
    manifest = json.loads(manifest_path.read_text())
    if not manifest["source_sha256_matches_registry"]:
        raise RuntimeError("source file checksum does not match the pinned registry value")
    X, labels = load_arrays(args.out)
    assert_no_leakage(manifest["features"])
    split = make_split(np.asarray(labels["treatment"]))
    save_split(split, args.out)
    scaler = fit_scaler(X, split.train_idx)
    (args.out / "scaler.json").write_text(json.dumps(scaler.to_dict(), indent=2) + "\n")

    treat = np.asarray(labels["treatment"])
    summary = {
        "rows": int(len(treat)),
        "seed": split.seed,
        "stratified_by": "treatment (no outcome used)",
        "split_fingerprint": split.fingerprint(),
        "sizes": split.sizes(),
        "treated_share": {
            n: float(treat[split.idx(p)].mean())
            for n, p in (("train", TRAIN), ("validation", VAL), ("test", TEST))
        },
        "source": {
            "file": manifest["source_file"],
            "sha256": manifest["source_sha256"],
            "url": CRITEO.download_url,
            "version": CRITEO.version,
        },
        "arrays": {
            "X_dtype": "float32",
            "X_gb": round(X.nbytes / 1024**3, 3),
            "labels_dtype": "uint8",
        },
        "scaler_fitted_on": "train rows only",
        "peak_rss_gb": round(peak_rss_gb(), 2),
    }
    args.report.mkdir(parents=True, exist_ok=True)
    (args.report / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
