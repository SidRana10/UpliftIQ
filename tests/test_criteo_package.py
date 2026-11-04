"""Packaged Criteo model: schema validation, inference equivalence, integrity, MLflow tracking."""

from __future__ import annotations

import json
import shutil

import mlflow
import numpy as np
import pandas as pd
import pytest

from upliftiq.modeling import criteo_package as cp
from upliftiq.modeling import criteo_tracking as ct
from upliftiq.modeling.criteo_learners import SLearner

R = np.random.default_rng(0)


@pytest.fixture(scope="module")
def fitted():
    n = 6000
    X = R.normal(size=(n, 12)).astype(np.float32)
    t = (R.random(n) < 0.85).astype(np.int8)
    y = (R.random(n) < 0.3 + 0.1 * np.tanh(X[:, 0]) * t).astype(float)
    return SLearner(params={"n_estimators": 40}, n_jobs=2).fit(X, t, y), X


@pytest.fixture()
def package_dir(fitted, tmp_path):
    model, _ = fitted
    model.model_.booster_.save_model(str(tmp_path / "model.txt"))
    manifest = {
        "model_name": "test",
        "model_version": "9.9.9",
        "artifact": {"file": "model.txt", "sha256": cp.sha256_file(tmp_path / "model.txt")},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def record(X, i=0):
    return {f"f{j}": float(X[i, j]) for j in range(12)}


def test_packaged_model_matches_the_fitted_learner_exactly(fitted, package_dir):
    model, X = fitted
    served = cp.CriteoUpliftModel.load(package_dir)
    out = served.predict(X[:200])
    assert np.array_equal(out[["p_visit_control", "p_visit_treatment"]].to_numpy(),
                          model.predict_outcomes(X[:200]))  # fmt: skip
    assert np.array_equal(out.uplift.to_numpy(), model.predict_uplift(X[:200]))
    assert np.array_equal(out.recommend_treatment.to_numpy(), model.recommend(X[:200]))
    assert list(out.columns) == [
        "p_visit_control", "p_visit_treatment", "uplift", "recommend_treatment"
    ]  # fmt: skip
    assert ((out.p_visit_control > 0) & (out.p_visit_treatment < 1)).all()


def test_single_record_and_batch_agree_across_input_types(fitted, package_dir):
    _, X = fitted
    served = cp.CriteoUpliftModel.load(package_dir)
    one = served.predict_one(record(X))
    batch = served.predict([record(X, i) for i in range(5)])
    assert one["uplift"] == pytest.approx(batch.uplift[0]) and one["model_version"] == "9.9.9"
    assert one["uplift"] == pytest.approx(one["p_visit_treatment"] - one["p_visit_control"])
    assert isinstance(one["recommend_treatment"], bool)
    frame = pd.DataFrame([record(X, i) for i in range(5)]).iloc[
        :, ::-1
    ]  # column order is irrelevant
    assert np.allclose(served.predict(frame).uplift, batch.uplift)
    assert np.allclose(served.predict(X[:5]).uplift, batch.uplift)
    assert served.predict_one(X[0])["uplift"] == pytest.approx(one["uplift"])


@pytest.mark.parametrize(
    "bad, match",
    [
        ({f"f{j}": 0.0 for j in range(11)}, "missing required features"),
        ({**{f"f{j}": 0.0 for j in range(12)}, "treatment": 1}, "unexpected fields"),
        ({**{f"f{j}": 0.0 for j in range(12)}, "visit": 0}, "unexpected fields"),
        ({**{f"f{j}": 0.0 for j in range(12)}, "f3": float("nan")}, "missing values"),
        ({**{f"f{j}": 0.0 for j in range(12)}, "f3": float("inf")}, "non-finite"),
        ({**{f"f{j}": 0.0 for j in range(12)}, "f3": "abc"}, "must be numeric"),
        ({**{f"f{j}": 0.0 for j in range(12)}, "f3": None}, "missing values"),
        ({**{f"f{j}": 0.0 for j in range(12)}, "f3": True}, "must be numeric"),
        ([], "empty"),
        (np.zeros((3, 11)), "shape"),
        (np.zeros((2, 2, 12)), "shape"),
        ("not data", "not numeric|shape"),
    ],
)
def test_invalid_inputs_get_clear_errors(fitted, package_dir, bad, match):
    served = cp.CriteoUpliftModel.load(package_dir)
    with pytest.raises(cp.InputValidationError, match=match):
        served.predict(bad)


def test_batch_limit_and_predict_one_needs_a_single_record(fitted, package_dir, monkeypatch):
    _, X = fitted
    served = cp.CriteoUpliftModel.load(package_dir)
    with pytest.raises(cp.InputValidationError, match="exactly one record"):
        served.predict_one([record(X, 0), record(X, 1)])
    monkeypatch.setattr(cp, "MAX_BATCH", 3)
    with pytest.raises(cp.InputValidationError, match="exceeds the limit"):
        served.predict(X[:4])


def test_tampered_model_file_is_rejected_before_loading(package_dir):
    with (package_dir / "model.txt").open("a") as f:
        f.write("\n# tampered\n")
    with pytest.raises(cp.PackageIntegrityError):
        cp.CriteoUpliftModel.load(package_dir)


def test_inference_needs_no_dataset_or_pickle_and_loads_only_package_files(package_dir, tmp_path):
    only = tmp_path / "isolated"
    shutil.copytree(package_dir, only)
    assert {p.name for p in only.iterdir()} == {"model.txt", "manifest.json"}
    cp.CriteoUpliftModel.load(only)


# ------------------------------------------------------------------ real package (if present)

REAL = cp.MODEL_ROOT / cp.MODEL_VERSION


@pytest.mark.skipif(not (REAL / "manifest.json").exists(), reason="model not packaged locally")
def test_real_package_manifest_records_frozen_fingerprints_and_reproducibility():
    m = json.loads((REAL / "manifest.json").read_text())
    frozen = json.loads(cp.FROZEN_PATH.read_text())
    assert m["input_schema"]["features"] == [f"f{i}" for i in range(12)]
    assert m["frozen"]["validation_uplift_sha256"] == frozen["selected"]["validation_uplift_sha256"]
    assert m["frozen"]["split_fingerprint"] == frozen["split"]["fingerprint"]
    assert m["frozen"]["config_sha256"] == cp.sha256_file(cp.FROZEN_PATH)
    assert m["reproducibility"]["matches_frozen_validation_fingerprint"] is True
    assert m["reproducibility"]["held_out_test_set_read"] is False
    assert m["artifact"]["sha256"] == cp.sha256_file(REAL / "model.txt")
    served = cp.CriteoUpliftModel.load(REAL)
    out = served.predict(np.zeros((2, 12)))
    assert out.shape == (2, 4) and (out.p_visit_control.between(0, 1)).all()


# ------------------------------------------------------------------ MLflow


def test_policy_metrics_converts_to_pp_and_ignores_other_models():
    pol = [
        {"model": "s_learner", "policy": "budget", "estimator": "AIPW", "budget_requested": 0.2,
         "value": 0.0067, "ci_low": 0.006, "ci_high": 0.007, "diff_vs_random": 0.005},
        {"model": "s_learner", "policy": "budget", "estimator": "IPW", "budget_requested": 0.2,
         "value": 0.0097},
        {"model": "logistic", "policy": "budget", "estimator": "AIPW", "budget_requested": 0.2,
         "value": 0.5, "ci_low": 0, "ci_high": 1, "diff_vs_random": 0},
    ]  # fmt: skip
    q = [{"model": "s_learner", "estimator": "AIPW", "qini": 0.003, "ate_treat_all": 0.007}]
    m = ct.policy_metrics(pol, q, "heldout")
    assert m["heldout_aipw_value_20pct_pp"] == pytest.approx(0.67)
    assert m["heldout_ipw_value_20pct_pp"] == pytest.approx(0.97)
    assert m["heldout_aipw_qini_pp"] == pytest.approx(0.3)
    assert 50.0 not in m.values()  # the logistic row (0.5 -> 50 pp) is not included


@pytest.mark.skipif(not ct.HELD_OUT_RESULTS.exists(), reason="reports not present")
def test_mlflow_records_historical_and_new_runs_separately_and_idempotently(tmp_path):
    frozen_sha = cp.sha256_file(cp.FROZEN_PATH)
    frozen = json.loads(cp.FROZEN_PATH.read_text())
    sel = frozen["selected"]
    manifest = {
        "model_name": "criteo_s_learner",
        "model_version": "1.0.0",
        "input_schema": {"features": cp.FEATURES},
        "frozen": {
            "config_sha256": frozen_sha,
            "dataset_version": "v2.1",
            "dataset_sha256": frozen["dataset"]["source_sha256"],
            "split_fingerprint": frozen["split"]["fingerprint"],
            "training_rows": sel["training_rows"],
            "seed": sel["seed"],
            "hyperparameters": sel["hyperparameters"],
            "validation_uplift_sha256": sel["validation_uplift_sha256"],
        },
        "artifact": {"file": "model.txt", "sha256": "x"},
        "dependencies": {"python": "3"},
        "source": {"source": "test"},
        "reproducibility": {
            "packaged_model_validation_uplift_sha256": sel["validation_uplift_sha256"],
            "matches_frozen_validation_fingerprint": True,
            "held_out_test_set_read": False,
        },
    }
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "model.txt").write_text("m")
    (pkg / "manifest.json").write_text("{}")
    root = tmp_path / "mlruns"
    first = ct.log_historical(manifest, root)
    assert ct.log_historical(manifest, root) == first  # not duplicated
    new = ct.log_packaging(manifest, root, pkg)
    client = mlflow.MlflowClient(ct.tracking_uri(root))
    hist, run = client.get_run(first), client.get_run(new)
    assert hist.data.tags["provenance"] == "historical"
    assert hist.data.tags["executed_by_this_run"] == "false"
    assert run.data.tags["provenance"] == "new_execution"
    held = json.loads(ct.HELD_OUT_RESULTS.read_text())
    want = next(
        p for p in held["policies"]
        if p["estimator"] == "AIPW" and p["policy"] == "budget" and p["budget_requested"] == 0.2
    )["value"] * 100  # fmt: skip
    assert hist.data.metrics["heldout_aipw_value_20pct_pp"] == pytest.approx(want)
    assert "validation_aipw_value_20pct_pp" in hist.data.metrics
    assert hist.data.params["training_rows"] == str(sel["training_rows"])
    assert run.data.metrics["validation_fingerprint_match"] == 1.0
    assert run.data.metrics["heldout_test_set_read"] == 0.0
    assert (root / "mlflow.db").exists()
