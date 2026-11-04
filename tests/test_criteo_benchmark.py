"""C5 benchmark pipeline on a small synthetic prepared dataset (isolated worker processes)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

from upliftiq.data.criteo_prepare import (
    load_split,
    make_split,
    save_split,
    train_subset,
)
from upliftiq.evaluation import criteo_benchmark as cb

N = 60_000
N_FIT, N_VAL, N_NUIS, N_EXCL = 8_000, 6_000, 10_000, 12_000
SPECS = {
    "logistic": ("logistic", {}),
    "t_learner": ("t_learner", {"params": {"n_estimators": 30}, "n_jobs": 2}),
    "tarnet_cpu": ("tarnet", {"device": "cpu", "max_epochs": 3}),
}
NAMES = list(SPECS)


def make_dataset(path, seed=0):
    """Prepared-dataset layout (X.npy + separate label arrays + split) with a planted effect."""
    r = np.random.default_rng(seed)
    X = r.normal(size=(N, 12)).astype(np.float32)
    p0 = 0.3 + 0.08 * np.tanh(X[:, 1])
    tau = 0.18 * np.tanh(1.5 * X[:, 0])
    t = (r.random(N) < 0.85).astype(np.uint8)
    y = (r.random(N) < p0 + t * tau).astype(np.uint8)
    path.mkdir(parents=True)
    np.save(path / "X.npy", X)
    np.save(path / "treatment.npy", t)
    np.save(path / "visit.npy", y)
    np.save(path / "exposure.npy", (t * (r.random(N) < 0.04)).astype(np.uint8))
    np.save(path / "conversion.npy", (y * (r.random(N) < 0.06)).astype(np.uint8))
    save_split(make_split(t), path)
    (path / "manifest.json").write_text(json.dumps({"source_sha256": "synthetic"}))
    return path


def run(data_dir, out_dir, names=NAMES, specs=SPECS):
    return cb.run_ladder(
        [N_FIT],
        names,
        data_dir,
        out_dir,
        n_val=N_VAL,
        n_boot=40,
        nuisance_rows=N_NUIS,
        nuisance_exclude=N_EXCL,
        specs=specs,
    )


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("c5")
    data = make_dataset(root / "data")
    mp = pytest.MonkeyPatch()
    mp.setenv("OMP_NUM_THREADS", "8")  # the driver must override this for its workers
    try:
        reports = run(data, root / "reports")
    finally:
        mp.undo()
    return {"data": data, "out": root / "reports", "root": root, "reports": reports}


def test_report_and_outputs_are_written_and_labelled_preliminary(pipeline):
    out = pipeline["out"]
    report = (out / f"fit{N_FIT}_report.md").read_text()
    assert "Preliminary" in report and "no model has been selected" in report
    assert "not final held-out results" in report and "ROI" in report
    for name in NAMES:
        assert name in report
    assert (out / f"fit{N_FIT}_gain_curves.png").exists()
    assert (out / f"fit{N_FIT}_cost_quality.png").exists()
    res = json.loads((out / f"fit{N_FIT}_results.json").read_text())
    assert res["status"].startswith("preliminary") and res["n_fit"] == N_FIT
    assert set(res["tables"]) >= {"value", "qini", "pairwise", "engineering", "references"}
    refs = {r["policy"] for r in res["tables"]["references"]}
    assert {"treat_nobody", "treat_all", "random@10%", "random@20%", "random@30%"} <= refs
    assert any("unconstrained" in r["feasibility"] for r in res["tables"]["references"])
    # pairwise differences are reported against both reference models
    assert set(res["pairwise"]) == {"logistic", "tarnet_cpu"}


def test_manifests_record_provenance_resources_and_isolated_environment(pipeline):
    run_dir = cb.rung_dir(pipeline["data"], N_FIT, N_VAL)
    split = load_split(pipeline["data"])
    for name in NAMES:
        m = json.loads((run_dir / f"{name}.json").read_text())
        assert m["provenance"]["split_fingerprint"] == split.fingerprint()
        assert m["provenance"]["data_sha256"] == "synthetic"
        assert m["seed"] == 42 and m["n_fit"] == N_FIT and m["n_val"] == N_VAL
        assert m["omp_num_threads"] == "1"  # set before Python started, despite parent's value 8
        assert m["fit_seconds"] > 0 and m["peak_rss_gb"] > 0
        assert {"lightgbm", "numpy", "python"} <= set(m["libraries"])
        assert [d["batch"] for d in m["latency"]] == [1, 1000]  # 100k batch skipped: 6k val rows
        assert all(d["p95_ms"] >= d["p50_ms"] > 0 and d["rows_per_s"] > 0 for d in m["latency"])
    tar = json.loads((run_dir / "tarnet_cpu.json").read_text())
    assert tar["libraries"]["torch"] and tar["training"]["device"] == "cpu"
    assert len(tar["training"]["history"]) >= 1


def test_models_share_identical_training_and_validation_subsets_and_informative_ones_beat_random(
    pipeline,
):
    res = json.loads((pipeline["out"] / f"fit{N_FIT}_results.json").read_text())
    q = {r["model"]: r for r in res["qini"]["logistic"] if r["estimator"] == "AIPW"}
    assert q["logistic"]["ci_low"] > 0 and q["tarnet_cpu"]["ci_low"] > 0
    assert {m["n_fit"] for m in res["manifests"].values()} == {N_FIT}
    assert res["n_val"] == N_VAL
    nuis = res["nuisance"]
    assert nuis["n_train"] == N_NUIS and nuis["disjoint_from_candidate_subsets_up_to"] == N_EXCL


def test_nuisance_rows_are_disjoint_from_candidates_and_inside_train(pipeline):
    split = load_split(pipeline["data"])
    t = np.load(pipeline["data"] / "treatment.npy")
    nuis = cb.nuisance_train_rows(split, t, N_NUIS, N_EXCL)
    cand = train_subset(split, t, N_EXCL)  # superset of every smaller nested subset
    assert not set(nuis) & set(cand)
    assert set(nuis) <= set(split.train_idx)
    val = cb.validation_subset(split, t, N_VAL)
    assert set(val) <= set(split.val_idx) and not set(val) & set(nuis)
    assert t[val].mean() == pytest.approx(t[split.val_idx].mean(), abs=0.01)


def test_rerun_reuses_workers_and_force_retrains(pipeline):
    run_dir = cb.rung_dir(pipeline["data"], N_FIT, N_VAL)
    before = {n: (run_dir / f"{n}.json").stat().st_mtime_ns for n in NAMES}
    run(pipeline["data"], pipeline["out"])
    assert before == {n: (run_dir / f"{n}.json").stat().st_mtime_ns for n in NAMES}
    # a changed configuration invalidates only that model's cache
    changed = {**SPECS, "logistic": ("logistic", {"C": 0.5})}
    run(pipeline["data"], pipeline["out"], specs=changed)
    after = {n: (run_dir / f"{n}.json").stat().st_mtime_ns for n in NAMES}
    assert after["logistic"] != before["logistic"] and after["t_learner"] == before["t_learner"]


def test_test_rows_never_influence_the_benchmark(tmp_path):
    a = make_dataset(tmp_path / "a")
    shutil.copytree(a, tmp_path / "b")
    b = tmp_path / "b"
    test_idx = load_split(b).test_idx
    X, y = np.load(b / "X.npy"), np.load(b / "visit.npy")
    X[test_idx] += 1000.0
    y[test_idx] = 1 - y[test_idx]
    np.save(b / "X.npy", X)
    np.save(b / "visit.npy", y)
    specs = {"logistic": SPECS["logistic"]}
    ra = run(a, tmp_path / "ra", ["logistic"], specs)[0].parent / f"fit{N_FIT}_results.json"
    rb = run(b, tmp_path / "rb", ["logistic"], specs)[0].parent / f"fit{N_FIT}_results.json"
    pa, pb = (json.loads(p.read_text()) for p in (ra, rb))
    assert pa["policies"] == pb["policies"] and pa["qini"] == pb["qini"]


def test_driver_process_does_not_load_lightgbm_or_torch():
    code = (
        "import sys, upliftiq.evaluation.criteo_benchmark as cb;"
        "assert 'torch' not in sys.modules and 'lightgbm' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def test_helpers_latency_subsets_and_config_key():
    calls = []
    X = np.zeros((300, 12), dtype=np.float32)
    stats = cb.latency_stats(
        lambda rows: calls.append(len(rows)), X, plan=((1, 5), (100, 3), (1000, 2))
    )
    assert [s["batch"] for s in stats] == [1, 100]  # 1000 > available rows is skipped
    assert all(s["p95_ms"] >= s["p50_ms"] for s in stats)
    idx = np.arange(10_000)
    t = (np.random.default_rng(0).random(10_000) < 0.85).astype(int)
    sub = cb.stratified_subset(idx, t, 1000)
    assert abs(len(sub) - 1000) <= 2 and t[sub].mean() == pytest.approx(0.85, abs=0.03)
    assert np.array_equal(sub, cb.stratified_subset(idx, t, 1000))
    assert cb.config_key({"a": 1}) == cb.config_key({"a": 1}) != cb.config_key({"a": 2})


def test_failed_worker_surfaces_its_error():
    with pytest.raises(RuntimeError, match="worker failed"):
        cb._spawn(
            [
                "model",
                "--data-dir",
                "/nonexistent",
                "--name",
                "x",
                "--spec",
                "{}",
                "--size",
                "1",
                "--n-val",
                "1",
                "--seed",
                "1",
                "--out-dir",
                "/tmp",
                "--config-key",
                "k",
            ]
        )
