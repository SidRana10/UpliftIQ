import gzip
import json

import numpy as np
import pandas as pd
import pytest

import upliftiq.data.criteo_diagnostics as dg
from upliftiq.data.criteo import FEATURES
from upliftiq.data.criteo_prepare import (
    PROCESSED_DIR,
    TEST,
    TRAIN,
    VAL,
    convert_to_npy,
    fit_scaler,
    load_arrays,
    load_split,
    make_split,
    save_split,
    train_subset,
)
from upliftiq.data.registry import CRITEO


def frame(n=20000, seed=0, confound=False, positivity_break=False):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 12)), columns=FEATURES)
    df["f9"] = np.exp(df["f9"])  # heavy tail
    p = np.full(n, 0.85)
    if confound:
        p = 1 / (1 + np.exp(-(1.7 + 1.5 * df["f0"].to_numpy())))
    if positivity_break:
        p = np.where(df["f1"].to_numpy() > 1.5, 0.0, 0.85)
    df["treatment"] = (rng.random(n) < p).astype(int)
    df["exposure"] = df["treatment"] * (rng.random(n) < 0.04).astype(int)
    df["visit"] = (rng.random(n) < 0.05).astype(int)
    df["conversion"] = df["visit"] * (rng.random(n) < 0.06).astype(int)
    return df[list(CRITEO.columns)]


@pytest.fixture
def prepared(tmp_path):
    df = frame()
    gz = tmp_path / "c.csv.gz"
    with gzip.open(gz, "wt") as f:
        df.to_csv(f, index=False)
    out = tmp_path / "proc"
    manifest = convert_to_npy(gz, out, expected_rows=len(df), chunksize=3000)
    return df, out, manifest, gz


# ---------- conversion ----------


def test_conversion_round_trips_and_keeps_labels_out_of_features(prepared):
    df, out, manifest, _ = prepared
    X, labels = load_arrays(out)
    assert X.dtype == np.float32 and X.shape == (len(df), 12)
    np.testing.assert_allclose(X, df[FEATURES].to_numpy(np.float32), rtol=1e-6)
    for k in ("treatment", "exposure", "visit", "conversion"):
        assert labels[k].dtype == np.uint8
        np.testing.assert_array_equal(labels[k], df[k].to_numpy())
    assert manifest["rows"] == len(df) and manifest["features"] == FEATURES
    assert sorted(p.name for p in out.glob("*.npy")) == sorted(
        ["X.npy", "treatment.npy", "exposure.npy", "visit.npy", "conversion.npy"]
    )
    assert (
        json.loads((out / "manifest.json").read_text())["source_sha256"]
        == manifest["source_sha256"]
    )


def test_conversion_rejects_wrong_row_count_and_bad_values(prepared, tmp_path):
    df, _, _, gz = prepared
    with pytest.raises(ValueError):
        convert_to_npy(gz, tmp_path / "a", expected_rows=len(df) + 5)
    with pytest.raises(ValueError):
        convert_to_npy(gz, tmp_path / "b", expected_rows=len(df) - 5)
    bad = df.copy()
    bad.loc[3, "visit"] = 2
    p = tmp_path / "bad.csv.gz"
    with gzip.open(p, "wt") as f:
        bad.to_csv(f, index=False)
    with pytest.raises(ValueError, match="non-binary"):
        convert_to_npy(p, tmp_path / "c", expected_rows=len(df))


# ---------- split ----------


def test_split_sizes_reproducibility_and_stratification():
    df = frame(30000)
    t = df["treatment"].to_numpy()
    s = make_split(t)
    sz = s.sizes()
    assert sum(sz.values()) == len(t)
    assert sz["train"] == pytest.approx(0.64 * len(t), abs=3)
    assert sz["validation"] == pytest.approx(0.16 * len(t), abs=3)
    assert sz["test"] == pytest.approx(0.20 * len(t), abs=3)
    assert not set(s.train_idx) & set(s.val_idx) | set(s.val_idx) & set(s.test_idx)
    for part in (TRAIN, VAL, TEST):
        assert t[s.idx(part)].mean() == pytest.approx(t.mean(), abs=2e-4)
    again = make_split(t)
    assert again.fingerprint() == s.fingerprint()
    assert make_split(t, seed=7).fingerprint() != s.fingerprint()


def test_split_uses_treatment_only_not_outcomes():
    df = frame(10000)
    t = df["treatment"].to_numpy()
    base = make_split(t).fingerprint()
    # nothing but treatment is an input, so outcomes cannot influence the assignment
    shuffled = df.assign(visit=1 - df["visit"], conversion=0, exposure=0)
    assert make_split(shuffled["treatment"].to_numpy()).fingerprint() == base
    import inspect

    src = inspect.getsource(make_split)
    assert "visit" not in src and "conversion" not in src and "exposure" not in src


def test_split_save_load_and_corruption(tmp_path):
    s = make_split(frame(4000)["treatment"].to_numpy())
    save_split(s, tmp_path)
    assert load_split(tmp_path).fingerprint() == s.fingerprint()
    bad = s.assignment.copy()
    bad[0] = (bad[0] + 1) % 3
    np.save(tmp_path / "split.npy", bad)
    with pytest.raises(ValueError):
        load_split(tmp_path)


def test_train_subsets_are_nested_stratified_and_train_only():
    t = frame(40000)["treatment"].to_numpy()
    s = make_split(t)
    small, large = train_subset(s, t, 2000), train_subset(s, t, 8000)
    assert set(small) <= set(s.train_idx) and set(large) <= set(s.train_idx)
    assert len(small) == pytest.approx(2000, abs=2)
    assert t[small].mean() == pytest.approx(t[s.train_idx].mean(), abs=2e-3)
    assert (train_subset(s, t, 10**9) == s.train_idx).all()
    assert np.array_equal(small, train_subset(s, t, 2000))


# ---------- preprocessing ----------


def test_scaler_is_fitted_on_train_rows_only(prepared):
    df, out, *_ = prepared
    X, labels = load_arrays(out)
    s = make_split(np.asarray(labels["treatment"]))
    sc = fit_scaler(X, s.train_idx)
    train = np.clip(X[s.train_idx], sc.lo, sc.hi)
    np.testing.assert_allclose(sc.mean, train.mean(axis=0), rtol=1e-4, atol=1e-5)
    # corrupting validation/test rows leaves the fitted statistics unchanged
    X2 = np.array(X)
    X2[np.concatenate([s.val_idx, s.test_idx])] += 1000
    sc2 = fit_scaler(X2, s.train_idx)
    for a, b in zip(
        (sc.lo, sc.hi, sc.mean, sc.std), (sc2.lo, sc2.hi, sc2.mean, sc2.std), strict=True
    ):
        np.testing.assert_array_equal(a, b)
    z = sc.transform(X, s.train_idx)
    assert z.dtype == np.float32 and z.shape == (len(s.train_idx), 12)
    assert np.abs(z.mean(axis=0)).max() < 1e-3
    assert np.abs(sc.transform(X2, s.test_idx)).max() > 0  # applied (clipped) to held-out rows


def test_scaler_clips_outliers_using_train_percentiles(prepared):
    _, out, *_ = prepared
    X, labels = load_arrays(out)
    s = make_split(np.asarray(labels["treatment"]))
    sc = fit_scaler(X, s.train_idx)
    wild = np.full((1, 12), 1e9, dtype=np.float32)
    np.testing.assert_allclose(
        sc.transform(wild, np.array([0])), ((sc.hi - sc.mean) / sc.std)[None], rtol=1e-4
    )


# ---------- diagnostics ----------


def run_diag(df):
    X = df[FEATURES].to_numpy(np.float32)
    t = df["treatment"].to_numpy()
    s = make_split(t)
    sc = fit_scaler(X, s.train_idx)
    return dg.run_diagnostics(X, t, s.train_idx, sc), X, t, s


def test_randomized_data_is_supported(monkeypatch):
    monkeypatch.setattr(dg, "DIAG_ROWS", 20000)
    res, *_ = run_diag(frame(300000, seed=1))
    assert res["verdict"]["status"] == "benchmark_modeling_allowed"
    assert "cannot be independently verified" in res["verdict"]["statement"]
    assert "ROI" in res["verdict"]["statement"]
    assert res["checks"]["A1a_standardized_mean_differences"]["grade"] == "pass"
    assert abs(res["classifier"]["lightgbm"]["auc"] - 0.5) < 0.03
    assert abs(res["classifier"]["permuted_label_auc"] - 0.5) < 0.03


def test_confounded_assignment_is_flagged_and_not_auto_corrected(monkeypatch):
    monkeypatch.setattr(dg, "DIAG_ROWS", 6000)
    monkeypatch.setattr(dg, "MIN_BIN_N", 200)
    res, *_ = run_diag(frame(60000, seed=2, confound=True))
    assert res["verdict"]["status"] == "does_not_support"
    assert res["checks"]["A1b_treatment_from_features_auc"]["grade"] == "fail"
    assert res["checks"]["A1a_standardized_mean_differences"]["grade"] in ("warn", "fail")
    assert res["classifier"]["feature_importance_gain_share"]["f0"] > 0.5
    assert "report the issue" in res["verdict"]["statement"]


def test_positivity_violation_is_detected(monkeypatch):
    monkeypatch.setattr(dg, "DIAG_ROWS", 6000)
    monkeypatch.setattr(dg, "MIN_BIN_N", 200)
    res, *_ = run_diag(frame(60000, seed=3, positivity_break=True))
    assert res["checks"]["A2_positivity"]["grade"] == "fail"
    assert res["verdict"]["status"] == "does_not_support"


def test_classifier_split_is_disjoint_stratified_and_inside_train():
    df = frame(30000, seed=4)
    t = df["treatment"].to_numpy()
    s = make_split(t)
    fit, ev = dg.diagnostic_split(t, s.train_idx, n=5000)
    assert not set(fit) & set(ev)
    assert set(fit) | set(ev) <= set(s.train_idx)
    assert not (set(fit) | set(ev)) & (set(s.val_idx) | set(s.test_idx))
    assert t[fit].mean() == pytest.approx(t[s.train_idx].mean(), abs=0.02)


def test_diagnostics_ignore_validation_and_test_rows(monkeypatch):
    monkeypatch.setattr(dg, "DIAG_ROWS", 4000)
    monkeypatch.setattr(dg, "MIN_BIN_N", 200)
    df = frame(30000, seed=5)
    res1, X, t, s = run_diag(df)
    t2 = t.copy()
    held = np.concatenate([s.val_idx, s.test_idx])
    t2[held] = 1 - t2[held]
    X2 = X.copy()
    X2[held] += 100
    sc = fit_scaler(X2, s.train_idx)
    res2 = dg.run_diagnostics(X2, t2, s.train_idx, sc)
    assert res1["smd"] == res2["smd"] and res1["bins"] == res2["bins"]
    assert res1["classifier"]["lightgbm"]["auc"] == res2["classifier"]["lightgbm"]["auc"]


def test_helpers_smd_bins_blocks_and_se():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50000, 12)).astype(np.float32)
    t = (rng.random(50000) < 0.5).astype(int)
    X[t == 1, 0] += 1.0
    idx = np.arange(len(t))
    smd = dg.smd_table(X, t, idx).set_index("feature")["smd"]
    assert smd["f0"] == pytest.approx(1.0, abs=0.05) and abs(smd["f5"]) < 0.05
    bins = dg.bin_proportions(X, t, idx, n_bins=10)
    assert bins.groupby("feature")["n"].sum().eq(len(t)).all()
    assert bins[bins["feature"] == "f0"]["deviation"].abs().max() > 0.3
    blocks = dg.block_proportions(t, idx, n_blocks=10)
    assert len(blocks) == 10 and blocks["n"].sum() == len(t)
    assert 0 < dg.hanley_mcneil_se(0.6, 1000, 1000) < 0.02
    assert dg._grade(0.0, "max_abs_smd") == "pass" and dg._grade(0.07, "max_abs_smd") == "warn"
    assert dg._grade(0.2, "max_abs_smd") == "fail"


@pytest.mark.network
def test_real_processed_data_if_present():
    if not (PROCESSED_DIR / "manifest.json").exists():
        pytest.skip("Criteo not prepared")
    X, labels = load_arrays()
    s = load_split()
    assert X.shape == (13_979_592, 12) and s.sizes()["test"] == 2_795_918
    assert s.sizes()["train"] + s.sizes()["validation"] + s.sizes()["test"] == 13_979_592


def test_run_structure_detects_sorted_subdatasets_but_not_shuffled_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6000, 12)).astype(np.float32)
    shuffled = (rng.random(6000) < 0.85).astype(int)
    assert dg.run_structure(X, shuffled, np.arange(6000))["runs"] is None
    sorted_t = np.r_[np.ones(2000), np.zeros(500), np.ones(2500), np.zeros(1000)].astype(int)
    rs = dg.run_structure(X, sorted_t, np.arange(6000))
    assert rs["n_runs"] == 4 and [r["n"] for r in rs["runs"]] == [2000, 500, 2500, 1000]
    shares = [g["treated_share"] for g in rs["inferred_groups"]]
    assert shares == pytest.approx([0.8, 2500 / 3500])


def test_row_ordered_groups_raise_an_ordering_warning_not_a_failure(monkeypatch):
    monkeypatch.setattr(dg, "DIAG_ROWS", 20000)
    df = frame(300000, seed=6).sort_values("treatment", kind="stable").reset_index(drop=True)
    res, *_ = run_diag(df)
    assert res["checks"]["A1d_stability_across_row_blocks"]["grade"] == "ordering_warning"
    assert res["verdict"]["status"] == "benchmark_modeling_allowed"
    assert "ordering warning" in res["verdict"]["statement"]
    assert "not proof of heterogeneous propensities" in res["verdict"]["statement"]
    assert res["run_structure"]["n_runs"] == 2
    md = dg.build_markdown(res)
    assert "ordering warning" in md and "experiment IDs" in md
