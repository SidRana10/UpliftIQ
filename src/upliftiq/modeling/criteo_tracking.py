"""Local MLflow tracking for the approved Criteo S-learner (no server; data in git-ignored mlruns/).

Two kinds of runs, told apart by the ``provenance`` tag:

``historical``     metrics and parameters *imported* from the already-generated, committed reports
                   (C6 validation benchmark, C7 held-out evaluation). Nothing is recomputed and no
                   model is trained; the run records which report files and checksums it came from.
``new_execution``  a run executed now: packaging the model and re-verifying it reproduces the frozen
                   validation fingerprint (the held-out set is not read).

Metrics are in percentage points (pp) of visit rate vs treating nobody (AIPW unless noted).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow  # noqa: E402

from upliftiq.data.registry import PROJECT_ROOT  # noqa: E402

TRACKING_DIR = PROJECT_ROOT / "mlruns"
EXPERIMENT = "UpliftIQ-Criteo"
VALIDATION_RESULTS = PROJECT_ROOT / "reports" / "criteo" / "benchmark_full" / "final_results.json"
HELD_OUT_RESULTS = PROJECT_ROOT / "reports" / "criteo" / "final" / "final_results.json"
HELD_OUT_RECORD = PROJECT_ROOT / "reports" / "criteo" / "final" / "evaluation_record.json"
MODEL_DIR = PROJECT_ROOT / "models" / "criteo_s_learner"
MODEL = "s_learner"


def tracking_uri(root: Path = TRACKING_DIR) -> str:
    """Local SQLite store inside the (git-ignored) tracking dir: no server process needed."""
    Path(root).mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{Path(root).resolve() / 'mlflow.db'}"


def _experiment(root: Path):
    mlflow.set_tracking_uri(tracking_uri(root))
    found = mlflow.get_experiment_by_name(EXPERIMENT)
    if found:
        return found
    artifacts = (Path(root).resolve() / "artifacts").as_uri()
    mlflow.create_experiment(EXPERIMENT, artifact_location=artifacts)
    return mlflow.get_experiment_by_name(EXPERIMENT)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policy_metrics(policies: list[dict], qini: list[dict], prefix: str, model: str = MODEL) -> dict:
    """AIPW value at each budget (with interval, and the paired difference from random), Qini, IPW."""
    out = {}
    for p in policies:
        if p["model"] != model or p["policy"] != "budget":
            continue
        tag = f"{p['budget_requested'] * 100:.0f}pct"
        if p["estimator"] == "AIPW":
            out[f"{prefix}_aipw_value_{tag}_pp"] = p["value"] * 100
            out[f"{prefix}_aipw_value_{tag}_ci_low_pp"] = p["ci_low"] * 100
            out[f"{prefix}_aipw_value_{tag}_ci_high_pp"] = p["ci_high"] * 100
            out[f"{prefix}_aipw_vs_random_{tag}_pp"] = p["diff_vs_random"] * 100
        else:
            out[f"{prefix}_ipw_value_{tag}_pp"] = p["value"] * 100
    for q in qini:
        if q["model"] == model:
            out[f"{prefix}_{q['estimator'].lower()}_qini_pp"] = q["qini"] * 100
            if q["estimator"] == "AIPW":
                out[f"{prefix}_treat_all_aipw_pp"] = q["ate_treat_all"] * 100
    return out


def _validation_policies(results: dict) -> tuple[list[dict], list[dict]]:
    ref = next(iter(results["policies"]))  # any reference run scores the same model identically
    return results["policies"][ref], results["qini"][ref]


def _find(client, experiment_id: str, kind: str, key: str):
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.run_kind = '{kind}' and tags.frozen_config_sha256 = '{key}'",
    )
    return runs[0].info.run_id if runs else None


def log_historical(manifest: dict, root: Path = TRACKING_DIR, force: bool = False) -> str:
    """Import the frozen validation and held-out results as one clearly-labelled historical run."""
    exp = _experiment(root)
    mlflow.set_experiment(EXPERIMENT)
    client = mlflow.MlflowClient()
    key = manifest["frozen"]["config_sha256"]
    existing = _find(client, exp.experiment_id, "historical_import", key)
    if existing and not force:
        return existing

    val = json.loads(VALIDATION_RESULTS.read_text())
    held = json.loads(HELD_OUT_RESULTS.read_text())
    record = json.loads(HELD_OUT_RECORD.read_text())
    metrics = {}
    metrics |= policy_metrics(*_validation_policies(val), "validation")
    metrics |= policy_metrics(held["policies"], held["qini"], "heldout")
    metrics["heldout_ipw_minus_aipw_treat_all_pp"] = (
        held["assumption_check"]["ate_ipw_minus_aipw"] * 100
    )
    metrics["validation_ipw_minus_aipw_treat_all_pp"] = (
        val["assumption_check"]["ate_ipw_minus_aipw"] * 100
    )

    fz = manifest["frozen"]
    with mlflow.start_run(run_name="criteo-s-learner-historical-results") as run:
        mlflow.set_tags(
            {
                "run_kind": "historical_import",
                "provenance": "historical",
                "executed_by_this_run": "false",
                "frozen_config_sha256": key,
                "model_name": manifest["model_name"],
                "model_version": manifest["model_version"],
                "validation_source": "reports/criteo/benchmark_full/final_results.json",
                "validation_source_sha256": _sha(VALIDATION_RESULTS),
                "heldout_source": "reports/criteo/final/final_results.json",
                "heldout_source_sha256": _sha(HELD_OUT_RESULTS),
                "heldout_record_status": record["status"],
                "heldout_code_dirty": str(record["code_dirty"]).lower(),
                "heldout_evaluated_once": "true",
                "estimand_note": "conditional estimates for the released Criteo sample; not ROI",
            }
        )
        mlflow.log_params(
            {
                "dataset": fz["dataset_version"],
                "dataset_sha256": fz["dataset_sha256"],
                "split_fingerprint": fz["split_fingerprint"],
                "training_rows": fz["training_rows"],
                "seed": fz["seed"],
                "primary_metric": "aipw_value_20pct_pp",
                "features": ",".join(manifest["input_schema"]["features"]),
                **{f"lgbm_{k}": v for k, v in fz["hyperparameters"]["lightgbm"].items()},
                "early_stopping_rounds": fz["hyperparameters"]["early_stopping_rounds"],
                "early_stopping_val_fraction": fz["hyperparameters"]["val_fraction"],
                "validation_subset_rows": val["n_val"],
                "bootstrap_replicates": held["meta"]["n_boot"],
            }
        )
        mlflow.log_metrics(metrics)
        return run.info.run_id


def log_packaging(manifest: dict, root: Path = TRACKING_DIR, model_dir: Path | None = None) -> str:
    """Record this packaging/verification execution (new run; no training, no test-set access)."""
    _experiment(root)
    mlflow.set_experiment(EXPERIMENT)
    model_dir = Path(model_dir or MODEL_DIR / manifest["model_version"])
    rep = manifest["reproducibility"]
    with mlflow.start_run(run_name=f"criteo-s-learner-package-{manifest['model_version']}") as run:
        mlflow.set_tags(
            {
                "run_kind": "package_verification",
                "provenance": "new_execution",
                "executed_by_this_run": "package + verify only (no training, no held-out access)",
                "frozen_config_sha256": manifest["frozen"]["config_sha256"],
                "model_name": manifest["model_name"],
                "model_version": manifest["model_version"],
                "model_source": manifest["source"]["source"],
            }
        )
        mlflow.log_params(
            {
                "artifact_sha256": manifest["artifact"]["sha256"],
                "frozen_validation_uplift_sha256": manifest["frozen"]["validation_uplift_sha256"],
                "packaged_validation_uplift_sha256": rep["packaged_model_validation_uplift_sha256"],
                **{f"dep_{k}": v for k, v in manifest["dependencies"].items()},
            }
        )
        mlflow.log_metric(
            "validation_fingerprint_match", float(rep["matches_frozen_validation_fingerprint"])
        )
        mlflow.log_metric("heldout_test_set_read", float(rep["held_out_test_set_read"]))
        mlflow.log_artifact(str(model_dir / "manifest.json"))
        mlflow.log_artifact(str(model_dir / manifest["artifact"]["file"]), artifact_path="model")
        return run.info.run_id


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Log the approved Criteo model to local MLflow.")
    ap.add_argument("--tracking-dir", default=str(TRACKING_DIR))
    ap.add_argument("--force", action="store_true", help="re-import the historical run")
    a = ap.parse_args(argv)
    manifest = json.loads((MODEL_DIR / "1.0.0" / "manifest.json").read_text())
    print("historical run:", log_historical(manifest, Path(a.tracking_dir), a.force))
    print("package run:   ", log_packaging(manifest, Path(a.tracking_dir)))
    print("browse with:    mlflow ui --backend-store-uri", tracking_uri(Path(a.tracking_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
