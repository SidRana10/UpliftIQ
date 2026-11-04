"""API: response schemas, inference parity, targeting rules, errors, missing assets, isolation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from upliftiq.backend.app import create_app
from upliftiq.backend.settings import Settings
from upliftiq.evaluation.binary_policy import rank_order
from upliftiq.modeling.criteo_learners import SLearner
from upliftiq.modeling.criteo_package import FEATURES, CriteoUpliftModel, sha256_file

V1 = "/api/v1"
R = np.random.default_rng(1)


@pytest.fixture(scope="module")
def pkg(tmp_path_factory):
    d = tmp_path_factory.mktemp("pkg")
    n = 6000
    X = R.normal(size=(n, 12)).astype(np.float32)
    t = (R.random(n) < 0.85).astype(np.int8)
    y = (R.random(n) < 0.3 + 0.1 * np.tanh(X[:, 0]) * t).astype(float)
    m = SLearner(params={"n_estimators": 40}, n_jobs=2).fit(X, t, y)
    m.model_.booster_.save_model(str(d / "model.txt"))
    manifest = {
        "model_name": "criteo_s_learner",
        "model_version": "9.9.9",
        "frozen": {"seed": 42},
        "artifact": {"file": "model.txt", "sha256": sha256_file(d / "model.txt")},
    }
    (d / "manifest.json").write_text(json.dumps(manifest))
    return d


@pytest.fixture(scope="module")
def client(pkg):
    with TestClient(create_app(Settings(model_dir=pkg, max_batch=50))) as c:
        yield c


def rec(i=0, **extra):
    r = np.random.default_rng(100 + i)
    return {**{f: float(v) for f, v in zip(FEATURES, r.normal(size=12), strict=True)}, **extra}


# ---------------------------------------------------------------- schemas and content

GET_ENDPOINTS = [
    "/datasets", "/overview",
    "/hillstrom/groups", "/hillstrom/effects", "/hillstrom/balance", "/hillstrom/models",
    "/hillstrom/decision",
    "/criteo/diagnostics", "/criteo/ipw-aipw", "/criteo/benchmark", "/criteo/heldout",
    "/criteo/curves", "/models", "/models/criteo",
]  # fmt: skip


@pytest.mark.parametrize("path", GET_ENDPOINTS)
def test_report_endpoints_return_the_envelope_with_sources_and_provenance(client, path):
    r = client.get(V1 + path)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {
        "dataset",
        "source",
        "provenance",
        "frozen_config_sha256",
        "caveats",
        "data",
    }
    assert (
        body["dataset"] in {"hillstrom", "criteo", "platform"}
        and body["provenance"] == "historical"
    )
    assert body["data"] not in (None, [], {})


def test_health_reports_model_and_reports(client):
    h = client.get(V1 + "/health").json()
    assert h["status"] == "ok" and h["criteo_model_loaded"] and h["model_version"] == "9.9.9"
    assert all(h["reports_available"].values())


def test_headline_numbers_are_the_committed_report_values_not_recomputed(client):
    held = pd.read_csv(Settings().reports / "criteo/final/heldout_value.csv")
    ov = client.get(V1 + "/overview").json()["data"]
    assert ov["criteo"]["budgets"] == json.loads(held.to_json(orient="records"))
    pv = pd.read_csv(Settings().reports / "final_evaluation/policy_values.csv")
    row = pv[
        (pv.outcome == "visit") & (pv.estimator == "AIPW") & (pv.policy == "frozen_policy")
    ].iloc[0]
    assert ov["hillstrom"]["value"]["value"] == pytest.approx(row.value)
    assert ov["hillstrom"]["frozen_policy"] == "mens_for_all"
    assert "ROI" in " ".join(ov["criteo"]["caveats"])


def test_criteo_heldout_keeps_ipw_aipw_limitation_and_marks_conversion_exploratory(client):
    b = client.get(V1 + "/criteo/heldout").json()
    d = b["data"]
    assert d["assumption_check"]["ate_ipw_minus_aipw"] > 0
    assert "exploratory" in d["exploratory_conversion"]["note"]
    assert any("IPW and AIPW disagree" in c for c in b["caveats"])
    types = {r["policy"]: r["type"] for r in d["references"]}
    assert types["treat_all"] == "unconstrained reference"


def test_curves_expose_only_the_evaluated_budgets(client):
    d = client.get(V1 + "/criteo/curves").json()["data"]
    assert d["evaluated_budgets"] == [0.1, 0.2, 0.3] and len(d["points"]) == 3
    for url in d["figures"].values():
        assert client.get(url).headers["content-type"] == "image/png"
    assert client.get(V1 + "/criteo/figures/nope").status_code == 404
    assert client.get(V1 + "/hillstrom/figures/..%2f..%2fpyproject").status_code == 404


def test_datasets_stay_isolated_between_hillstrom_and_criteo(client):
    for path in [p for p in GET_ENDPOINTS if p.startswith("/hillstrom")]:
        b = client.get(V1 + path).json()
        assert b["dataset"] == "hillstrom"
        assert not any("criteo" in s.lower() for s in b["source"])
        assert "criteo" not in json.dumps(b).lower()
    for path in [p for p in GET_ENDPOINTS if p.startswith("/criteo")]:
        b = client.get(V1 + path).json()
        assert b["dataset"] == "criteo"
        assert all(s.startswith(("reports/criteo", "mlruns")) for s in b["source"])
        assert "hillstrom" not in json.dumps(b).lower()
    assert client.get(V1 + "/hillstrom/heldout").status_code == 404  # no cross-dataset routes
    assert client.get(V1 + "/criteo/groups").status_code == 404


def test_mlflow_runs_label_historical_and_new_executions(client):
    r = client.get(V1 + "/models/criteo/runs")
    if r.status_code == 404:
        pytest.skip("no local MLflow store")
    assert {x["provenance"] for x in r.json()["data"]} <= {"historical", "new_execution"}


# ---------------------------------------------------------------- inference parity


def test_single_prediction_matches_the_wrapper_exactly(client, pkg):
    model = CriteoUpliftModel.load(pkg)
    r = client.post(V1 + "/criteo/predict", json=rec(3))
    assert r.status_code == 200
    assert r.json() == model.predict_one(rec(3))


def test_batch_prediction_matches_the_wrapper_and_the_single_endpoint(client, pkg):
    model = CriteoUpliftModel.load(pkg)
    records = [rec(i) for i in range(20)]
    body = client.post(V1 + "/criteo/predict/batch", json={"records": records}).json()
    want = model.predict(records)
    assert body["n"] == 20 and body["model_version"] == "9.9.9"
    got = pd.DataFrame(body["predictions"])
    assert np.array_equal(got.uplift.to_numpy(), want.uplift.to_numpy())
    assert np.array_equal(
        got.recommend_treatment.to_numpy(), want.recommend_treatment.astype(bool).to_numpy()
    )
    single = client.post(V1 + "/criteo/predict", json=records[7]).json()
    assert single["uplift"] == body["predictions"][7]["uplift"]
    assert all(0 < p["p_visit_control"] < 1 for p in body["predictions"])


def test_model_is_loaded_once_and_reused(client):
    m1 = client.app.state.model
    client.post(V1 + "/criteo/predict", json=rec(1))
    assert client.app.state.model is m1


# ---------------------------------------------------------------- invalid inputs

BAD_BODIES = [
    {k: v for k, v in rec().items() if k != "f5"},
    rec(treatment=1),
    rec(visit=0),
    {**rec(), "f2": "abc"},
    {**rec(), "f2": None},
    {**rec(), "f2": [1.0]},
]


@pytest.mark.parametrize("bad", BAD_BODIES)
def test_invalid_single_records_are_rejected_with_422(client, bad):
    r = client.post(V1 + "/criteo/predict", json=bad)
    assert r.status_code == 422 and r.json()["detail"]


def test_non_finite_values_and_bad_batches_are_rejected(client):
    raw = json.dumps(rec()).replace(f'"f1": {rec()["f1"]}', '"f1": NaN')
    assert "NaN" in raw
    r = client.post(
        V1 + "/criteo/predict", content=raw, headers={"Content-Type": "application/json"}
    )
    assert r.status_code == 422
    assert client.post(V1 + "/criteo/predict/batch", json={"records": []}).status_code == 422
    assert client.post(V1 + "/criteo/predict/batch", json={"rows": [rec()]}).status_code == 422
    big = client.post(V1 + "/criteo/predict/batch", json={"records": [rec(i) for i in range(51)]})
    assert big.status_code == 422 and big.json()["code"] == "invalid_input"
    assert client.post(V1 + "/criteo/predict", content="not json").status_code == 422


# ---------------------------------------------------------------- targeting


def target(client, records, budget):
    return client.post(V1 + "/criteo/target", json={"records": records, "budget": budget})


@pytest.mark.parametrize("budget", [0.1, 0.2, 0.3, 0.5, 1.0, 0.07])
def test_budget_selects_exactly_the_top_share_by_uplift(client, budget):
    records = [rec(i, id=1000 + i) for i in range(40)]
    body = target(client, records, budget).json()
    k = int(np.floor(budget * 40 + 1e-9))
    items = body["items"]
    assert body["n"] == 40 and body["n_selected"] == k == sum(i["selected"] for i in items)
    assert [i["rank"] for i in items] == list(range(1, 41))
    uplift = [i["uplift"] for i in items]
    assert uplift == sorted(uplift, reverse=True)
    assert all(i["selected"] == (i["rank"] <= k) for i in items)
    assert min((i["uplift"] for i in items if i["selected"]), default=np.inf) >= max(
        (i["uplift"] for i in items if not i["selected"]), default=-np.inf
    )


def test_ranking_is_deterministic_order_independent_and_uses_the_validated_tie_break(client):
    base = [rec(i, id=i) for i in range(30)]
    dup = [rec(0, id=500 + j) for j in range(10)]  # identical customers -> exact ties
    records = base + dup
    a = target(client, records, 0.5).json()
    b = target(client, list(reversed(records)), 0.5).json()
    assert [i["id"] for i in a["items"]] == [i["id"] for i in b["items"]]
    assert target(client, records, 0.5).json() == a
    tied = [i for i in a["items"] if i["id"] >= 500]
    ids = np.array([r_["id"] for r_ in records])
    out = client.post(
        V1 + "/criteo/predict/batch",
        json={"records": [{k: v for k, v in r_.items() if k != "id"} for r_ in records]},
    ).json()
    scores = np.array([p["uplift"] for p in out["predictions"]])
    order = rank_order(scores, ids, 42)
    assert [i["id"] for i in a["items"]] == [int(ids[p]) for p in order]
    assert len(tied) == 10


def test_targeting_without_ids_uses_positions_and_validates_budget_and_ids(client):
    records = [rec(i) for i in range(10)]
    body = target(client, records, 0.3).json()
    assert sorted(i["id"] for i in body["items"]) == list(range(10)) and body["n_selected"] == 3
    for bad_budget in (0, -0.1, 1.01):
        assert target(client, records, bad_budget).status_code == 422
    assert target(client, [rec(0, id=1), rec(1)], 0.5).status_code == 422  # mixed ids
    assert target(client, [rec(0, id=1), rec(1, id=1)], 0.5).status_code == 422  # duplicates
    assert target(client, [rec(0, id=-1)], 0.5).status_code == 422
    assert target(client, [], 0.5).status_code == 422
    assert "ROI" in " ".join(body["caveats"])


# ---------------------------------------------------------------- missing assets and CORS


def test_missing_model_degrades_health_and_returns_503_for_inference(tmp_path):
    with TestClient(create_app(Settings(model_dir=tmp_path / "absent"))) as c:
        h = c.get(V1 + "/health").json()
        assert h["status"] == "degraded" and not h["criteo_model_loaded"] and h["model_error"]
        r = c.post(V1 + "/criteo/predict", json=rec())
        assert r.status_code == 503 and r.json()["code"] == "model_unavailable"
        assert (
            c.post(V1 + "/criteo/target", json={"records": [rec()], "budget": 0.5}).status_code
            == 503
        )
        assert c.get(V1 + "/hillstrom/groups").status_code == 200  # reports still served


def test_corrupted_model_file_is_refused_not_served(pkg, tmp_path):
    import shutil

    bad = tmp_path / "bad"
    shutil.copytree(pkg, bad)
    with (bad / "model.txt").open("a") as f:
        f.write("\n# tampered\n")
    with TestClient(create_app(Settings(model_dir=bad))) as c:
        assert c.get(V1 + "/health").json()["criteo_model_loaded"] is False
        assert c.post(V1 + "/criteo/predict", json=rec()).status_code == 503


def test_missing_reports_give_a_clear_404_with_the_file_name(tmp_path, pkg):
    with TestClient(create_app(Settings(root=tmp_path, model_dir=pkg))) as c:
        for path in ("/hillstrom/groups", "/overview", "/criteo/heldout", "/models/criteo/runs"):
            r = c.get(V1 + path)
            assert r.status_code == 404 and r.json()["code"] == "report_unavailable"
            assert "report not available" in r.json()["detail"]
        h = c.get(V1 + "/health").json()
        assert h["status"] == "degraded" and not any(h["reports_available"].values())
        assert c.get(V1 + "/criteo/figures/heldout_gain_curve").status_code == 404


def test_cors_allows_the_local_frontend_only(client):
    ok = client.options(
        V1 + "/criteo/predict",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST",
                 "Access-Control-Request-Headers": "content-type"},
    )  # fmt: skip
    assert ok.headers["access-control-allow-origin"] == "http://localhost:3000"
    evil = client.get(V1 + "/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in evil.headers


def test_cors_origins_are_configurable_by_environment(monkeypatch, pkg):
    monkeypatch.setenv("UPLIFTIQ_CORS_ORIGINS", "https://demo.example.com")
    with TestClient(create_app(Settings(model_dir=pkg))) as c:
        r = c.get(V1 + "/health", headers={"Origin": "https://demo.example.com"})
        assert r.headers["access-control-allow-origin"] == "https://demo.example.com"
