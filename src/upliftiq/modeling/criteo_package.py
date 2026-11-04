"""Package and serve the frozen Criteo S-learner (local inference; no raw dataset needed).

``package`` exports the fitted LightGBM booster of the approved S-learner (taken from the locally
saved C7 evaluation artifact, or refitted with the frozen training procedure on training rows only if
that artifact is missing) into a self-describing directory:

    models/criteo_s_learner/<version>/model.txt        LightGBM text model (no pickle at serve time)
    models/criteo_s_learner/<version>/manifest.json    schema, fingerprints, checksums, dependencies

It then verifies reproducibility by scoring the *validation* subset with the packaged model and
comparing the SHA-256 of the scores to the frozen validation-prediction fingerprint. The held-out
test set is never read.

``CriteoUpliftModel`` loads a package, checks the model checksum, and predicts for one record or a
batch. The S-learner takes raw f0-f11 (tree model: no scaling) plus an internal treatment indicator.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from upliftiq.data.registry import CRITEO, PROJECT_ROOT

FEATURES = list(CRITEO.features)  # f0 ... f11
MODEL_NAME = "criteo_s_learner"
MODEL_VERSION = "1.0.0"
MODEL_ROOT = PROJECT_ROOT / "models" / MODEL_NAME
MANIFEST_COPY = PROJECT_ROOT / "reports" / "criteo" / "model_package.json"
FROZEN_PATH = PROJECT_ROOT / "reports" / "criteo" / "benchmark_full" / "frozen_config.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "criteo"
MAX_BATCH = 200_000


class InputValidationError(ValueError):
    """The input does not match the model's schema."""


class PackageIntegrityError(RuntimeError):
    """A packaged file does not match its recorded checksum."""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


# ---------------------------------------------------------------- inference


def to_frame(data) -> tuple[pd.DataFrame, bool]:
    """Validate and normalise input to a float32 (n, 12) frame; returns (frame, was_single_record).

    Accepted: one dict, a list of dicts, a DataFrame, or an array of shape (n, 12) / (12,).
    """
    single = False
    if isinstance(data, dict):
        single, data = True, [data]
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, list | tuple) and all(isinstance(r, dict) for r in data):
        frame = pd.DataFrame(list(data))
    else:
        try:
            arr = np.asarray(data, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise InputValidationError(f"input is not numeric: {e}") from None
        if arr.ndim == 1:
            single, arr = True, arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != len(FEATURES):
            raise InputValidationError(
                f"array input must have shape (n, {len(FEATURES)}) with columns {FEATURES}, "
                f"got {arr.shape}"
            )
        frame = pd.DataFrame(arr, columns=FEATURES)
    if len(frame) == 0:
        raise InputValidationError("input is empty")
    if len(frame) > MAX_BATCH:
        raise InputValidationError(f"batch of {len(frame)} exceeds the limit of {MAX_BATCH}")
    missing = [c for c in FEATURES if c not in frame.columns]
    if missing:
        raise InputValidationError(f"missing required features: {missing}")
    extra = [c for c in frame.columns if c not in FEATURES]
    if extra:
        raise InputValidationError(
            f"unexpected fields {extra}: only {FEATURES} are accepted (treatment, exposure and "
            "outcome columns must not be supplied)"
        )
    frame = frame[FEATURES]
    for c in FEATURES:
        col = frame[c]
        if col.isna().any():
            raise InputValidationError(f"feature '{c}' has missing values")
        if pd.api.types.is_bool_dtype(col) or not pd.api.types.is_numeric_dtype(col):
            try:
                if pd.api.types.is_bool_dtype(col):
                    raise ValueError("boolean")
                frame[c] = pd.to_numeric(col)
            except (TypeError, ValueError):
                raise InputValidationError(f"feature '{c}' must be numeric") from None
    values = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        bad = [c for c, ok in zip(FEATURES, np.isfinite(values).all(axis=0), strict=True) if not ok]
        raise InputValidationError(f"non-finite values (inf/NaN) in features {bad}")
    return pd.DataFrame(values.astype(np.float32), columns=FEATURES), single


class CriteoUpliftModel:
    """The approved model: P(visit | x, control), P(visit | x, treatment), uplift, recommendation."""

    def __init__(self, booster, manifest: dict):
        self.booster, self.manifest = booster, manifest
        self.version = manifest["model_version"]

    @classmethod
    def load(cls, package_dir: Path | str | None = None, verify_checksum: bool = True):
        import lightgbm as lgb

        pdir = Path(package_dir) if package_dir else MODEL_ROOT / MODEL_VERSION
        manifest = json.loads((pdir / "manifest.json").read_text())
        model_path = pdir / manifest["artifact"]["file"]
        if verify_checksum and sha256_file(model_path) != manifest["artifact"]["sha256"]:
            raise PackageIntegrityError(f"{model_path} does not match the manifest checksum")
        return cls(lgb.Booster(model_file=str(model_path)), manifest)

    def _outcomes(self, X: np.ndarray) -> np.ndarray:
        # identical to SLearner.predict_outcomes: treatment indicator appended as the last column
        return np.column_stack(
            [self.booster.predict(np.column_stack([X, np.full(len(X), a)])) for a in (0, 1)]
        )

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        mu = self._outcomes(X)
        return mu[:, 1] - mu[:, 0]

    def predict(self, data) -> pd.DataFrame:
        """Batch inference. Columns: p_visit_control, p_visit_treatment, uplift, recommend_treatment."""
        frame, _ = to_frame(data)
        mu = self._outcomes(frame.to_numpy())
        uplift = mu[:, 1] - mu[:, 0]
        return pd.DataFrame(
            {
                "p_visit_control": mu[:, 0],
                "p_visit_treatment": mu[:, 1],
                "uplift": uplift,
                "recommend_treatment": (uplift > 0).astype(int),
            }
        )

    def predict_one(self, record) -> dict:
        """Single-record inference; returns plain Python numbers plus the model version."""
        frame, single = to_frame(record)
        if len(frame) != 1:
            raise InputValidationError(
                "predict_one needs exactly one record; use predict() for batches"
            )
        row = self.predict(frame).iloc[0]
        return {
            "p_visit_control": float(row.p_visit_control),
            "p_visit_treatment": float(row.p_visit_treatment),
            "uplift": float(row.uplift),
            "recommend_treatment": bool(row.recommend_treatment),
            "model_version": self.version,
        }


# ---------------------------------------------------------------- packaging


def _source_booster(data_dir: Path, frozen_path: Path, cfg_sha: str):
    """The fitted S-learner's booster from the trusted local C7 artifact (refit if missing)."""
    from upliftiq.evaluation import criteo_heldout as ch

    pkl = ch.work_dir(data_dir) / "visit.pkl"
    meta = ch.work_dir(data_dir) / "visit_fit.json"
    source = "local C7 evaluation artifact"
    if not (pkl.exists() and meta.exists()):
        source = "refit with the frozen training procedure on training rows only"
        ch.ensure_fitted(data_dir, frozen_path, cfg_sha)
    with pkl.open("rb") as f:  # trusted: produced locally by criteo_heldout; verified below
        fitted = pickle.load(f)
    return fitted["model"].model_.booster_, {"source": source, "source_file": str(pkl.name)}


def validation_fingerprint(model: CriteoUpliftModel, data_dir: Path, cfg: dict) -> str:
    """SHA-256 of the packaged model's float32 uplift scores on the frozen validation subset."""
    from upliftiq.data.criteo_prepare import load_arrays, load_split
    from upliftiq.evaluation import criteo_benchmark as cb

    split = load_split(data_dir)
    X, labels = load_arrays(data_dir)
    rows = cb.validation_subset(
        split,
        np.asarray(labels["treatment"]),
        cfg["evaluation"]["validation_subset"]["rows"],
        cfg["selected"]["seed"],
    )
    assert np.isin(rows, split.val_idx).all()  # validation rows only; never the test set
    uplift = np.asarray(model.predict_uplift(np.asarray(X[rows], dtype=np.float32)), np.float32)
    buf = io.BytesIO()
    np.save(buf, uplift)
    return sha256_bytes(buf.getvalue())


def build_manifest(cfg: dict, cfg_sha: str, model_sha: str, fingerprint: str, source: dict) -> dict:
    import lightgbm

    sel = cfg["selected"]
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "approved: frozen C6 selection, evaluated once in C7",
        "algorithm": "LightGBM S-learner (one classifier on [f0..f11, treatment]); uplift = P(visit|x,1) - P(visit|x,0)",
        "input_schema": {
            "features": FEATURES,
            "dtype": "float32",
            "constraints": "all 12 required, finite, numeric; extra fields rejected",
            "preprocessing": "none (raw anonymized floats; tree model is scale-invariant)",
            "treatment_indicator": "added internally (0 then 1); never supplied by callers",
            "max_batch": MAX_BATCH,
        },
        "outputs": {
            "p_visit_control": "P(visit | x, no treatment)",
            "p_visit_treatment": "P(visit | x, treatment)",
            "uplift": "p_visit_treatment - p_visit_control",
            "recommend_treatment": "uplift > 0 (default policy; budgeted ranking is an evaluation concept)",
        },
        "artifact": {"file": "model.txt", "sha256": model_sha, "format": "LightGBM text model"},
        "frozen": {
            "config_file": "reports/criteo/benchmark_full/frozen_config.json",
            "config_sha256": cfg_sha,
            "hyperparameters": sel["hyperparameters"],
            "seed": sel["seed"],
            "training_rows": sel["training_rows"],
            "dataset_sha256": cfg["dataset"]["source_sha256"],
            "dataset_version": "Criteo Uplift Prediction Dataset v2.1",
            "split_fingerprint": cfg["split"]["fingerprint"],
            "validation_uplift_sha256": sel["validation_uplift_sha256"],
        },
        "reproducibility": {
            "packaged_model_validation_uplift_sha256": fingerprint,
            "matches_frozen_validation_fingerprint": fingerprint == sel["validation_uplift_sha256"],
            "held_out_test_set_read": False,
        },
        "source": source,
        "dependencies": {
            "python": platform.python_version(),
            "lightgbm": lightgbm.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "evaluation_reports": {
            "validation": "reports/criteo/benchmark_full/final_report.md",
            "held_out": "reports/criteo/final/final_report.md",
            "evaluation_record": "reports/criteo/final/evaluation_record.json",
        },
        "limitations": [
            "Trained on the anonymized Criteo benchmark: features are not business-interpretable.",
            "Conditional estimates for the released sample; not population ROI or guaranteed causal gains.",
            "The constant 0.85 propensity is unverifiable and IPW/AIPW disagree on this file (AIPW primary).",
            "The model's uplift is a ranking signal; its calibration as an absolute effect was not validated.",
        ],
    }


def package(
    data_dir: Path = PROCESSED_DIR,
    frozen_path: Path = FROZEN_PATH,
    out_root: Path = MODEL_ROOT,
    manifest_copy: Path | None = MANIFEST_COPY,
) -> dict:
    """Export, verify against the frozen validation fingerprint, and write the package."""
    import hashlib as _h

    from upliftiq.evaluation import criteo_heldout as ch

    data_dir, frozen_path = Path(data_dir), Path(frozen_path)
    cfg, cfg_sha = ch.load_frozen(frozen_path)
    ch.verify_frozen(cfg, data_dir, None)  # config, checksum, split, hyperparameters, protocol
    booster, source = _source_booster(data_dir, frozen_path, cfg_sha)
    out = Path(out_root) / MODEL_VERSION
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out / "model.txt"))
    model_sha = sha256_file(out / "model.txt")
    import lightgbm as lgb

    model = CriteoUpliftModel(
        lgb.Booster(model_file=str(out / "model.txt")), {"model_version": MODEL_VERSION}
    )
    fingerprint = validation_fingerprint(model, data_dir, cfg)
    if fingerprint != cfg["selected"]["validation_uplift_sha256"]:
        raise ch.ReproducibilityError(
            "packaged model does not reproduce the frozen validation fingerprint; not approving"
        )
    source["source_artifact_sha256"] = _h.sha256(
        (ch.work_dir(data_dir) / "visit.pkl").read_bytes()
    ).hexdigest()
    manifest = build_manifest(cfg, cfg_sha, model_sha, fingerprint, source)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest_copy is not None:
        Path(manifest_copy).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Package the frozen Criteo S-learner.")
    ap.add_argument("--data-dir", default=str(PROCESSED_DIR))
    args = ap.parse_args(argv)
    m = package(Path(args.data_dir))
    print(json.dumps(m["reproducibility"] | {"artifact": m["artifact"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
