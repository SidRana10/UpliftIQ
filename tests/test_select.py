import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import upliftiq.modeling.select as sel
from upliftiq.evaluation.policy import OutcomeModelPredictions
from upliftiq.modeling.features import ARMS, FEATURES
from upliftiq.modeling.select import (
    BASELINES,
    Candidate,
    choose_in_family,
    frozen_config,
    get_split,
    load_dev_data,
    one_se_choice,
    oof_nuisance,
    oof_recommendations,
    run_model_selection,
    select_final,
)
from upliftiq.modeling.select_report import write_artifacts
from upliftiq.modeling.split import make_split


def synth_frame(n=6000, seed=0, mode="hetero"):
    """mode: 'hetero' (Men's helps mens==1, Women's helps womens==1), 'homog' (Men's +0.09 for
    everyone, Women's +0.04 for everyone), 'null' (no effects)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "recency": rng.integers(1, 13, n),
            "history": rng.exponential(200, n) + 30,
            "mens": rng.integers(0, 2, n),
            "womens": rng.integers(0, 2, n),
            "zip_code": rng.choice(["Urban", "Surburban", "Rural"], n),
            "newbie": rng.integers(0, 2, n),
            "channel": rng.choice(["Web", "Phone", "Multichannel"], n),
        }
    )
    arm = rng.integers(0, 3, n)
    if mode == "hetero":
        tau = np.column_stack([np.zeros(n), 0.18 * df["mens"], 0.18 * df["womens"]])
    elif mode == "homog":
        tau = np.column_stack([np.zeros(n), np.full(n, 0.09), np.full(n, 0.04)])
    else:
        tau = np.zeros((n, 3))
    p = 0.10 + tau
    df["segment"] = np.array(ARMS)[arm]
    df["visit"] = (rng.random(n) < p[np.arange(n), arm]).astype(int)
    df["conversion"] = 0
    df["spend"] = 0.0
    df["history_segment"] = "x"
    return df


@pytest.fixture
def small_grid(monkeypatch):
    monkeypatch.setattr(sel, "LOGISTIC_GRID", [{"C": 0.1}])
    monkeypatch.setattr(
        sel,
        "LGBM_GRID",
        [{"num_leaves": 4, "min_child_samples": 300}, {"num_leaves": 8, "min_child_samples": 100}],
    )


def dev_of(df):
    split = make_split(df)
    return (*load_dev_data(df, split), split)


# ---------- out-of-fold mechanics ----------


class _Recorder:
    """Stand-in model that checks no evaluated row was seen in training."""

    calls = []

    def fit(self, X, arm, y):
        self.train_index = set(X.index)
        return self

    def recommend(self, X):
        assert not set(X.index) & self.train_index, "validation row was in training data"
        _Recorder.calls.append((self.train_index, set(X.index)))
        return np.ones(len(X), dtype=int)


def test_oof_recommendations_never_train_on_evaluated_rows(monkeypatch):
    dev, fold, _ = dev_of(synth_frame(1500))
    _Recorder.calls = []
    monkeypatch.setattr(Candidate, "build", lambda self, seed=42: _Recorder())
    rec = oof_recommendations(dev, fold, Candidate("t_learner", {}))
    assert len(rec) == len(dev) and (rec == 1).all()
    assert len(_Recorder.calls) == 5
    evaluated = [idx for _, idx in _Recorder.calls]
    assert set().union(*evaluated) == set(dev.index)  # every row scored exactly once
    assert sum(len(i) for i in evaluated) == len(dev)
    for train, val in _Recorder.calls:
        assert not train & val


def test_nuisance_is_cross_fitted_with_provenance_and_deterministic():
    dev, fold, _ = dev_of(synth_frame(2000))
    a, b = oof_nuisance(dev, fold), oof_nuisance(dev, fold)
    assert isinstance(a, OutcomeModelPredictions)
    np.testing.assert_array_equal(a.mu, b.mu)
    np.testing.assert_array_equal(a.group, fold)
    for k, train in a.trained_on.items():
        assert not set(train) & set(np.flatnonzero(fold == k))
        assert len(train) == (fold != k).sum()
    assert ((a.mu > 0) & (a.mu < 1)).all()


# ---------- selection rule on hand-built scores ----------


def _psi(means, n=4000, seed=0, noise=0.4):
    """Per-customer scores with given means and shared noise so paired SEs are small."""
    rng = np.random.default_rng(seed)
    shared = rng.normal(0, noise, n)
    return {k: m + shared + rng.normal(0, 0.01, n) for k, m in means.items()}


def _family_best(names):
    return {
        f: Candidate(f, {"C": 1.0})
        if f == "logistic"
        else Candidate(f, {"num_leaves": 4, "min_child_samples": 300})
        for f in names
    }


def test_baseline_wins_when_model_is_within_one_se():
    fb = _family_best(["t_learner"])
    psi = _psi(
        {
            "no_email": 0.10,
            "mens_for_all": 0.18,
            "womens_for_all": 0.15,
            "random": 0.14,
            fb["t_learner"].name: 0.1802,
        }
    )
    out = select_final(fb, psi)
    assert out["chosen"] == "mens_for_all"


def test_personalized_model_wins_when_clearly_better_and_simpler_model_within_one_se():
    fb = _family_best(["logistic", "t_learner"])
    base = {"no_email": 0.10, "mens_for_all": 0.18, "womens_for_all": 0.15, "random": 0.14}
    # logistic is clearly (many SEs) behind the best model: the best model is chosen
    far = select_final(fb, _psi({**base, fb["logistic"].name: 0.215, fb["t_learner"].name: 0.22}))
    assert far["best_by_value"] == "t_learner" and far["chosen"] == "t_learner"
    # logistic is within one paired SE of the best model (gap 0.0001 vs SE ~ 0.00016): simpler wins
    psi = _psi({**base, fb["logistic"].name: 0.22, fb["t_learner"].name: 0.22})
    wiggle = 0.01 * (-1.0) ** np.arange(len(psi["no_email"]))
    psi[fb["logistic"].name] = psi[fb["t_learner"].name] + wiggle - 0.0001
    near = select_final(fb, psi)
    assert near["best_by_value"] == "t_learner" and near["chosen"] == "logistic"


def test_model_below_best_baseline_is_ineligible():
    fb = _family_best(["s_learner", "dr_learner"])
    psi = _psi(
        {
            "no_email": 0.10,
            "mens_for_all": 0.18,
            "womens_for_all": 0.15,
            "random": 0.14,
            fb["s_learner"].name: 0.179,
            fb["dr_learner"].name: 0.25,
        }
    )
    out = select_final(fb, psi)
    assert out["ineligible_models"] == ["s_learner"]
    assert out["chosen"] == "dr_learner"


def test_fixed_baselines_can_win_and_tie_goes_to_simplest():
    fb = _family_best(["t_learner"])
    same = 0.18
    psi = _psi(
        {
            "no_email": 0.10,
            "mens_for_all": same,
            "womens_for_all": 0.15,
            "random": 0.14,
            fb["t_learner"].name: same,
        }
    )
    assert select_final(fb, psi)["chosen"] == "mens_for_all"
    assert (
        one_se_choice(["a", "b"], {"a": np.zeros(10) + np.arange(10), "b": np.arange(10.0)}, "b")
        == "a"
    )


def test_choose_in_family_prefers_simplest_within_one_se():
    cands = [
        Candidate("t_learner", {"num_leaves": 16, "min_child_samples": 100}),
        Candidate("t_learner", {"num_leaves": 4, "min_child_samples": 300}),
        Candidate("t_learner", {"num_leaves": 8, "min_child_samples": 100}),
    ]
    psi = _psi({cands[0].name: 0.20, cands[1].name: 0.1999, cands[2].name: 0.19})
    assert choose_in_family(cands, psi) == cands[1]
    psi2 = _psi({cands[0].name: 0.20, cands[1].name: 0.19, cands[2].name: 0.1995})
    assert choose_in_family(cands, psi2) == cands[0]  # simpler ones are > 1 SE worse


def test_selection_is_deterministic():
    fb = _family_best(["logistic", "dr_learner"])
    means = {
        "no_email": 0.10,
        "mens_for_all": 0.18,
        "womens_for_all": 0.15,
        "random": 0.14,
        fb["logistic"].name: 0.19,
        fb["dr_learner"].name: 0.185,
    }
    assert select_final(fb, _psi(means)) == select_final(fb, _psi(means))


# ---------- end-to-end on synthetic dev data ----------


def test_end_to_end_baseline_comparisons_and_determinism(small_grid):
    df = synth_frame(6000, seed=1, mode="hetero")
    dev, fold, _ = dev_of(df)
    r1 = run_model_selection(dev, fold, n_boot=60, verbose=False)
    r2 = run_model_selection(dev, fold, n_boot=60, verbose=False)
    pd.testing.assert_frame_equal(r1["cv_results"], r2["cv_results"])
    pd.testing.assert_frame_equal(r1["summary"], r2["summary"])
    assert r1["decision"] == r2["decision"]

    cv = r1["cv_results"].set_index("policy")
    arm = np.array([ARMS.index(s) for s in dev["segment"]])
    y = dev["visit"].to_numpy()
    for a, name in enumerate(["no_email", "mens_for_all", "womens_for_all"]):
        assert cv.loc[name, "ipw_value"] == pytest.approx(((arm == a) * y).mean() * 3)
    assert cv.loc["random", "ipw_value"] == pytest.approx(
        np.mean([cv.loc[b, "ipw_value"] for b in BASELINES[:3]])
    )
    s = r1["summary"].set_index("policy")
    for f in ("logistic", "t_learner"):
        for b in BASELINES:
            assert s.loc[f, f"diff_vs_{b}"] == pytest.approx(
                s.loc[f, "aipw_value"] - s.loc[b, "aipw_value"]
            )
            assert (
                s.loc[f, f"diff_vs_{b}_ci_low"]
                <= s.loc[f, f"diff_vs_{b}"]
                <= s.loc[f, f"diff_vs_{b}_ci_high"]
            )
    assert cv[[f"fold_{k}_aipw" for k in range(5)]].notna().all().all()


def test_heterogeneous_effects_select_a_personalized_policy(small_grid):
    dev, fold, _ = dev_of(synth_frame(9000, seed=2, mode="hetero"))
    r = run_model_selection(dev, fold, n_boot=50, verbose=False)
    chosen = r["decision"]["chosen"]
    assert chosen not in BASELINES
    s = r["summary"].set_index("policy")
    assert s.loc[chosen, "diff_vs_mens_for_all_ci_low"] > 0


def test_homogeneous_effects_keep_the_fixed_baseline(small_grid):
    dev, fold, _ = dev_of(synth_frame(9000, seed=3, mode="homog"))
    r = run_model_selection(dev, fold, n_boot=50, verbose=False)
    assert r["decision"]["chosen"] == "mens_for_all"


# ---------- test-set isolation ----------


def test_load_dev_data_drops_test_rows():
    df = synth_frame(3000)
    split = make_split(df)
    dev, fold = load_dev_data(df, split)
    assert len(dev) == len(split.dev_idx) == len(fold)
    assert set(np.unique(fold)) == {0, 1, 2, 3, 4}


def test_results_do_not_depend_on_test_set_outcomes(small_grid):
    df = synth_frame(5000, seed=4)
    split = make_split(df)
    poisoned = df.copy()
    t = split.test_idx
    poisoned.loc[t, "visit"] = 1 - poisoned.loc[t, "visit"]
    poisoned.loc[t, "segment"] = np.random.default_rng(0).permutation(poisoned.loc[t, "segment"])
    a, fa = load_dev_data(df, split)
    b, fb = load_dev_data(poisoned, split)
    pd.testing.assert_frame_equal(a, b)
    ra = run_model_selection(a, fa, n_boot=30, verbose=False)
    rb = run_model_selection(b, fb, n_boot=30, verbose=False)
    pd.testing.assert_frame_equal(ra["cv_results"], rb["cv_results"])
    assert ra["decision"] == rb["decision"]


def test_selection_modules_never_reference_test_rows():
    for name in ("select.py", "select_report.py"):
        src = (Path(sel.__file__).parent / name).read_text()
        assert "test_idx" not in src


# ---------- frozen config and artifacts ----------


def test_frozen_config_and_artifacts(small_grid, tmp_path):
    df = synth_frame(5000, seed=5)
    dev, fold, split = dev_of(df)
    r = run_model_selection(dev, fold, n_boot=40, verbose=False)
    cfg = frozen_config(r, split, 42, 40, "abc123")
    assert cfg["seed"] == 42 and cfg["data_sha256"] == "abc123"
    assert cfg["split"]["fingerprint"] == split.fingerprint()
    assert cfg["split"]["n_dev"] == len(dev) and cfg["test_set_accessed"] is False
    assert cfg["chosen_policy"]["kind"] in ("baseline", "model")
    assert cfg["evaluation"]["propensity"] == pytest.approx(1 / 3)
    assert cfg["evaluation"]["bootstrap"]["n_boot"] == 40
    json.dumps(cfg, default=float)
    write_artifacts(r, cfg, tmp_path)
    for f in (
        "cv_results.csv",
        "selection_summary.csv",
        "frozen_config.json",
        "report.md",
        "value_comparison.png",
    ):
        assert (tmp_path / f).exists()
    assert json.loads((tmp_path / "frozen_config.json").read_text())["split"]["n_folds"] == 5
    assert "test set was not accessed" in (tmp_path / "report.md").read_text()


def test_get_split_saves_reloads_and_detects_mismatch(tmp_path):
    df = synth_frame(2000)
    path = tmp_path / "split.json"
    s1 = get_split(df, path)
    assert path.exists()
    assert get_split(df, path).fingerprint() == s1.fingerprint()
    with pytest.raises(RuntimeError):
        get_split(synth_frame(2000, seed=9).assign(visit=lambda d: 1 - d["visit"]), path)


def test_feature_whitelist_unchanged():
    assert len(FEATURES) == 7
