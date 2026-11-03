import dataclasses
import json
import shutil

import numpy as np
import pandas as pd
import pytest

import upliftiq.evaluation.final_evaluation as fe
from upliftiq.data.registry import HILLSTROM
from upliftiq.evaluation.final_evaluation import (
    FROZEN_PATH,
    FrozenConfigError,
    SpendNuisance,
    _git_ok,
    evaluate_on_holdout,
    holdout_nuisance,
    load_frozen,
    policy_value_tables,
    recommend,
    rehearse,
    run_final_evaluation,
    strongest_alternative_baseline,
    verify_frozen,
)
from upliftiq.evaluation.policy import OutcomeModelPredictions
from upliftiq.modeling.features import ARMS, encode_arm
from upliftiq.modeling.select import sha256_file
from upliftiq.modeling.split import make_split


def synth(n=6000, seed=0, mode="homog", start=0):
    """homog: Men's +0.09, Women's +0.04 on a 0.10 base. hetero: Men's helps mens==1 (+0.18),
    Women's helps womens==1 (+0.18)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "recency": rng.integers(1, 13, n),
            "history_segment": "x",
            "history": rng.exponential(200, n) + 30,
            "mens": rng.integers(0, 2, n),
            "womens": rng.integers(0, 2, n),
            "zip_code": rng.choice(["Urban", "Surburban", "Rural"], n),
            "newbie": rng.integers(0, 2, n),
            "channel": rng.choice(["Web", "Phone", "Multichannel"], n),
        }
    )
    arm = rng.integers(0, 3, n)
    if mode == "homog":
        tau = np.column_stack([np.zeros(n), np.full(n, 0.09), np.full(n, 0.04)])
    elif mode == "null":
        tau = np.zeros((n, 3))
    else:
        tau = np.column_stack([np.zeros(n), 0.18 * df["mens"], 0.18 * df["womens"]])
    visit = (rng.random(n) < (0.10 + tau)[np.arange(n), arm]).astype(int)
    conv = visit * (rng.random(n) < 0.2).astype(int)
    df["segment"] = np.array(ARMS)[arm]
    df["visit"], df["conversion"] = visit, conv
    df["spend"] = conv * rng.exponential(80, n)
    df.index = start + np.arange(n)  # distinct row ids for separate train / holdout frames
    return df


BASE_CFG = {"chosen_policy": {"kind": "baseline", "name": "mens_for_all"}, "seed": 42}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Synthetic dataset + a matching frozen config, with the registry pointed at it."""
    df = synth(5000, seed=1)
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)
    sha = sha256_file(data_path)
    monkeypatch.setattr(fe, "HILLSTROM", dataclasses.replace(HILLSTROM, sha256=sha, n_rows=len(df)))
    split = make_split(df)
    cfg = {
        **BASE_CFG,
        "split": {
            "seed": 42,
            "n_folds": 5,
            "fingerprint": split.fingerprint(),
            "n_dev": len(split.dev_idx),
        },
        "data_sha256": sha,
        "test_set_accessed": False,
    }
    cfg_path = tmp_path / "frozen.json"
    cfg_path.write_text(json.dumps(cfg))
    cv = pd.DataFrame(
        {
            "policy": ["no_email", "mens_for_all", "womens_for_all", "random"],
            "family": "baseline",
            "aipw_value": [0.10, 0.19, 0.15, 0.1467],
        }
    )
    cv_path = tmp_path / "cv.csv"
    cv.to_csv(cv_path, index=False)
    return dict(
        df=df, data_path=data_path, cfg=cfg, cfg_path=cfg_path, cv_path=cv_path, split=split
    )


# ---------- frozen-config enforcement ----------


def test_verify_frozen_accepts_matching_setup(world):
    split = verify_frozen(world["cfg"], world["data_path"], world["df"])
    assert split.fingerprint() == world["cfg"]["split"]["fingerprint"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.update(data_sha256="0" * 64),
        lambda c: c["split"].update(fingerprint="deadbeef"),
        lambda c: c["split"].update(n_dev=1),
        lambda c: c["split"].update(seed=7),
        lambda c: c.update(test_set_accessed=True),
        lambda c: c.pop("seed"),
        lambda c: c.pop("chosen_policy"),
        lambda c: c.update(chosen_policy={"kind": "baseline", "name": "kids_for_all"}),
        lambda c: c.update(chosen_policy={"kind": "magic"}),
    ],
    ids=[
        "sha",
        "fingerprint",
        "n_dev",
        "split_seed",
        "accessed",
        "no_seed",
        "no_policy",
        "bad_baseline",
        "bad_kind",
    ],
)
def test_verify_frozen_rejects_tampered_config(world, mutate):
    cfg = json.loads(json.dumps(world["cfg"]))
    mutate(cfg)
    with pytest.raises(FrozenConfigError):
        verify_frozen(cfg, world["data_path"], world["df"])


def test_verify_frozen_rejects_modified_data_and_wrong_row_count(world, tmp_path):
    other = tmp_path / "other.csv"
    other.write_text(world["data_path"].read_text() + "\n")
    with pytest.raises(FrozenConfigError):
        verify_frozen(world["cfg"], other, world["df"])
    with pytest.raises(FrozenConfigError, match="rows"):
        verify_frozen(world["cfg"], world["data_path"], world["df"].iloc[:-1])


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_real_frozen_config_is_committed_and_unmodified(tmp_path):
    assert _git_ok(FROZEN_PATH)
    stray = tmp_path / "x.json"
    stray.write_text("{}")
    assert not _git_ok(stray)


def test_real_frozen_config_content_matches_stage4_decision():
    cfg, sha = load_frozen()
    assert cfg["chosen_policy"] == {"kind": "baseline", "name": "mens_for_all"}
    assert cfg["test_set_accessed"] is False and len(sha) == 64
    assert cfg["data_sha256"] == HILLSTROM.sha256
    assert strongest_alternative_baseline(cfg) == "womens_for_all"


def test_primary_comparator_excludes_the_winner(world):
    assert strongest_alternative_baseline(world["cfg"], world["cv_path"]) == "womens_for_all"
    other = {"chosen_policy": {"kind": "baseline", "name": "womens_for_all"}}
    assert strongest_alternative_baseline(other, world["cv_path"]) == "mens_for_all"


# ---------- estimation correctness against known truth ----------


def test_policy_values_and_paired_comparisons_cover_the_truth():
    df = synth(30000, seed=11, start=10**6)
    arm = encode_arm(df["segment"])
    y = df["visit"].to_numpy(float)
    rec = np.ones(len(df), dtype=int)
    nuis = holdout_nuisance(synth(20000, seed=12), df, "visit", 42)
    vals, comps = policy_value_tables(
        y, arm, rec, nuis, "mens_for_all", "visit", 300, 42, "womens_for_all"
    )
    truth = {
        "frozen_policy": 0.19,
        "no_email": 0.10,
        "mens_for_all": 0.19,
        "womens_for_all": 0.14,
        "random": (0.10 + 0.19 + 0.14) / 3,
    }
    for _, r in vals.iterrows():
        # one fixed sample: within 4 SE (CI coverage rates are validated by Stage 2 simulations)
        assert abs(r["value"] - truth[r["policy"]]) < 4 * r["se"], (r["estimator"], r["policy"])
        assert r["ci_low"] < r["value"] < r["ci_high"]
    # IPW of a fixed arm = 3 x mean of (assigned to that arm) * y
    ipw = vals[vals["estimator"] == "IPW"].set_index("policy")
    assert ipw.loc["mens_for_all", "value"] == pytest.approx(3 * ((arm == 1) * y).mean())
    assert ipw.loc["random", "value"] == pytest.approx(
        y[:, None].mean() * 0 + np.mean([3 * ((arm == a) * y).mean() for a in range(3)])
    )
    # the winner is not compared with itself
    assert set(comps["baseline"]) == {"no_email", "womens_for_all", "random"}
    true_diff = {"no_email": 0.09, "womens_for_all": 0.05, "random": 0.19 - 0.14333}
    for _, r in comps.iterrows():
        assert abs(r["diff"] - true_diff[r["baseline"]]) < 4 * r["paired_se"]
        assert r["ci_low"] < r["diff"] < r["ci_high"]
        assert r["p_value_holm"] >= r["p_value"] and r["p_value"] < 1e-3
        assert r["primary"] == (r["baseline"] == "womens_for_all")
    assert (comps["ci_high"] > comps["ci_low"]).all()


def test_no_effect_data_gives_no_significant_differences():
    df = synth(20000, seed=13, mode="null", start=10**6)
    arm = encode_arm(df["segment"])
    nuis = holdout_nuisance(synth(10000, seed=14, mode="null"), df, "visit", 42)
    _, comps = policy_value_tables(
        df["visit"].to_numpy(float),
        arm,
        np.ones(len(df), dtype=int),
        nuis,
        "mens_for_all",
        "visit",
        200,
        42,
    )
    assert (comps["p_value_holm"] > 0.05).all()
    assert ((comps["ci_low"] < 0) & (comps["ci_high"] > 0)).all()


def test_spend_keeps_zeros_and_aipw_agrees_with_ipw():
    train, df = synth(20000, seed=15), synth(20000, seed=16, start=10**6)
    arm = encode_arm(df["segment"])
    nuis = holdout_nuisance(train, df, "spend", 42)
    assert (nuis.mu >= 0).all() and nuis.mu.shape == (len(df), 3)
    vals, _ = policy_value_tables(
        df["spend"].to_numpy(float),
        arm,
        np.ones(len(df), dtype=int),
        nuis,
        "mens_for_all",
        "spend",
        200,
        42,
    )
    m = vals[vals["policy"] == "mens_for_all"].set_index("estimator")["value"]
    assert (df["spend"] == 0).mean() > 0.9  # zero-heavy, and all rows are scored
    assert m["AIPW"] == pytest.approx(m["IPW"], abs=0.5)
    assert vals["se"].gt(0).all()


# ---------- data isolation ----------


@pytest.mark.parametrize("outcome", ["visit", "conversion", "spend"])
def test_nuisance_models_ignore_holdout_outcomes_and_use_only_train(outcome):
    train, holdout = synth(4000, seed=21), synth(1500, seed=22, start=10**6)
    a = holdout_nuisance(train, holdout, outcome, 42)
    poisoned = holdout.copy()
    poisoned[["visit", "conversion", "spend"]] = (
        poisoned[["visit", "conversion", "spend"]].iloc[::-1].to_numpy()
    )
    poisoned["segment"] = np.random.default_rng(0).permutation(poisoned["segment"])
    b = holdout_nuisance(train, poisoned, outcome, 42)
    np.testing.assert_array_equal(a.mu, b.mu)
    assert isinstance(a, OutcomeModelPredictions)
    assert set(a.trained_on[0]) == set(train.index)
    assert not set(a.trained_on[0]) & set(a.row_ids)
    changed = train.copy()
    changed[outcome] = changed[outcome].iloc[::-1].to_numpy()
    assert not np.array_equal(holdout_nuisance(changed, holdout, outcome, 42).mu, a.mu)


def test_spend_nuisance_fit_uses_only_given_rows():
    train = synth(3000, seed=23)
    arm = encode_arm(train["segment"])
    m = SpendNuisance(42).fit(train, arm, train["spend"].to_numpy(float))
    assert list(m.pre_.feature_names_in_) == fe.select_features(train).columns.tolist()


def test_recommend_baselines_and_dev_fitted_model():
    train = synth(5000, seed=31, mode="hetero")
    holdout = synth(2000, seed=32, mode="hetero", start=10**6)
    assert (recommend(BASE_CFG, train, holdout) == 1).all()
    rnd = recommend(
        {"chosen_policy": {"kind": "baseline", "name": "random"}, "seed": 42}, train, holdout
    )
    assert rnd.shape == (len(holdout), 3)
    model_cfg = {
        "chosen_policy": {"kind": "model", "family": "logistic", "params": {"C": 1.0}},
        "seed": 42,
    }
    rec = recommend(model_cfg, train, holdout)
    poisoned = holdout.assign(visit=1 - holdout["visit"], segment="No E-Mail")
    np.testing.assert_array_equal(rec, recommend(model_cfg, train, poisoned))
    assert len(set(rec)) > 1  # personalizes on the planted heterogeneity


def test_model_policy_end_to_end_beats_no_email_on_heterogeneous_data():
    train = synth(12000, seed=33, mode="hetero")
    holdout = synth(8000, seed=34, mode="hetero", start=10**6)
    cfg = {
        "chosen_policy": {"kind": "model", "family": "logistic", "params": {"C": 1.0}},
        "seed": 42,
    }
    res = evaluate_on_holdout(train, holdout, cfg, "womens_for_all", 100, 42)
    c = res["comparisons"]
    vis = c[(c["outcome"] == "visit") & (c["estimator"] == "AIPW")].set_index("baseline")
    assert set(vis.index) == {"no_email", "mens_for_all", "womens_for_all", "random"}
    assert (vis["diff"] > 0).all()
    assert sum(res["recommendation_shares"].values()) == pytest.approx(1.0)


# ---------- reproducibility, one-shot guard, artifacts ----------


def test_evaluation_is_reproducible():
    train, holdout = synth(3000, seed=41), synth(1500, seed=42, start=10**6)
    a = evaluate_on_holdout(train, holdout, BASE_CFG, "womens_for_all", 60, 42)
    b = evaluate_on_holdout(train, holdout, BASE_CFG, "womens_for_all", 60, 42)
    for key in ("policy_values", "comparisons", "observed_arms", "observed_effects"):
        pd.testing.assert_frame_equal(a[key], b[key])


def test_one_shot_run_writes_artifacts_and_refuses_a_second_run(world, tmp_path):
    out = tmp_path / "final"
    kwargs = dict(
        config_path=world["cfg_path"], cv_path=world["cv_path"], n_boot=50, check_git=False
    )
    res = run_final_evaluation(world["df"], world["data_path"], out, **kwargs)
    for f in (
        "report.md",
        "policy_values.csv",
        "comparisons.csv",
        "observed_arms.csv",
        "observed_effects.csv",
        "evaluation_record.json",
        "figures/primary_visit_policy_values.png",
        "figures/secondary_policy_values.png",
        "figures/observed_outcomes.png",
    ):
        assert (out / f).exists(), f
    rec = json.loads((out / "evaluation_record.json").read_text())
    split = world["split"]
    assert rec["n_test"] == len(split.test_idx) == res["n_holdout"]
    assert rec["n_dev"] == len(split.dev_idx) == res["n_train"]
    assert rec["split_fingerprint"] == split.fingerprint()
    assert rec["primary_comparator"] == "womens_for_all"
    assert rec["nuisance_models_fitted_on"] == "development rows only"
    report = (out / "report.md").read_text()
    assert "exploratory" in report.lower() and "primary" in report.lower()
    before = (out / "report.md").read_text()
    with pytest.raises(FrozenConfigError, match="already been evaluated"):
        run_final_evaluation(world["df"], world["data_path"], out, **kwargs)
    assert (out / "report.md").read_text() == before


def test_run_refuses_when_config_does_not_match_data(world, tmp_path):
    bad = json.loads(world["cfg_path"].read_text())
    bad["data_sha256"] = "1" * 64
    world["cfg_path"].write_text(json.dumps(bad))
    with pytest.raises(FrozenConfigError):
        run_final_evaluation(
            world["df"],
            world["data_path"],
            tmp_path / "o",
            config_path=world["cfg_path"],
            cv_path=world["cv_path"],
            n_boot=20,
            check_git=False,
        )
    assert not (tmp_path / "o" / "evaluation_record.json").exists()


def test_rehearsal_uses_development_rows_only(world, tmp_path):
    df, split = world["df"], world["split"]
    poisoned = df.copy()
    t = split.test_idx
    # (visit/segment are excluded: they define the stratified split itself)
    poisoned.loc[t, "conversion"] = 1 - poisoned.loc[t, "conversion"]
    poisoned.loc[t, "spend"] = 1e6
    poisoned.loc[t, "recency"] = 99
    a = rehearse(df, world["cfg"], world["cv_path"], tmp_path / "a", n_boot=40)
    b = rehearse(poisoned, world["cfg"], world["cv_path"], tmp_path / "b", n_boot=40)
    for key in ("policy_values", "comparisons", "observed_arms"):
        pd.testing.assert_frame_equal(a[key], b[key])
    assert a["n_train"] + a["n_holdout"] == len(split.dev_idx)
