import numpy as np
import pandas as pd
import pytest

from upliftiq.evaluation.policy import OutcomeModelPredictions, evaluate_policy
from upliftiq.modeling.features import FEATURES
from upliftiq.modeling.learners import MODELS, make_model
from upliftiq.modeling.split import make_split

ALL = list(MODELS)
FAST = {"s_learner": {}, "t_learner": {}, "dr_learner": {}, "logistic": {}}


def synth(n=12000, seed=0, effects=True):
    """Synthetic experiment. Men's e-mail helps customers with mens==1 (+0.15), Women's e-mail
    helps womens==1 (+0.15); everyone else is unaffected. Baseline depends on history."""
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
    base = 0.10 + 0.05 * (df["history"].to_numpy() > 200)
    tau = np.column_stack([np.zeros(n), 0.15 * effects * df["mens"], 0.15 * effects * df["womens"]])
    p = base[:, None] + tau  # true P(Y=1 | x, arm)
    y = (rng.random(n) < p[np.arange(n), arm]).astype(float)
    return df, arm, y, tau, p


def fitted(name, df, arm, y, seed=42, **kw):
    return make_model(name, seed=seed, **kw).fit(df, arm, y)


def true_value(policy, p):
    return float(p[np.arange(len(p)), policy].mean())


@pytest.fixture(scope="module")
def data():
    return synth()


@pytest.fixture(scope="module")
def models(data):
    df, arm, y, *_ = data
    return {name: fitted(name, df, arm, y) for name in ALL}


# ---------- interface: shapes, ranges, indexing ----------


@pytest.mark.parametrize("name", ALL)
def test_prediction_shapes_and_ranges(name, data, models):
    df = data[0]
    m = models[name]
    uplift, rec = m.predict_uplift(df), m.recommend(df)
    assert uplift.shape == (len(df), 3)
    assert (uplift[:, 0] == 0).all()
    assert np.isfinite(uplift).all()
    assert rec.shape == (len(df),) and set(np.unique(rec)) <= {0, 1, 2}
    assert np.issubdtype(rec.dtype, np.integer)
    if name != "dr_learner":
        mu = m.predict_outcomes(df)
        assert mu.shape == (len(df), 3) and ((mu > 0) & (mu < 1)).all()
        np.testing.assert_allclose(uplift, mu - mu[:, [0]])


def test_dr_learner_has_no_outcome_levels(models, data):
    with pytest.raises(NotImplementedError):
        models["dr_learner"].predict_outcomes(data[0])


@pytest.mark.parametrize("name", ALL)
def test_recommend_is_argmax_of_uplift_with_ties_to_control(name, data, models):
    df = data[0].iloc[:500]
    m = models[name]
    up = m.predict_uplift(df)
    rec = m.recommend(df)
    assert (rec == up.argmax(axis=1)).all()
    assert (up[np.arange(len(rec)), rec] >= 0).all()  # never recommends a negative-uplift arm


@pytest.mark.parametrize("name", ALL)
def test_treatment_indexing_columns_track_the_right_arm(name, data, models):
    df, _, _, tau, _ = data
    up = models[name].predict_uplift(df)
    mens, womens = df["mens"].to_numpy() == 1, df["womens"].to_numpy() == 1
    # Men's column is high exactly where the Men's effect lives, Women's column likewise
    assert up[mens, 1].mean() - up[~mens, 1].mean() > 0.07
    assert up[womens, 2].mean() - up[~womens, 2].mean() > 0.07


@pytest.mark.parametrize("name", ALL)
def test_swapping_arm_labels_swaps_prediction_columns(name, data, models):
    df, arm, y, *_ = data
    swapped = np.where(arm == 1, 2, np.where(arm == 2, 1, arm))
    m2 = fitted(name, df, swapped, y)
    a, b = models[name].predict_uplift(df), m2.predict_uplift(df)
    assert np.corrcoef(a[:, 1], b[:, 2])[0, 1] > 0.9
    assert np.corrcoef(a[:, 2], b[:, 1])[0, 1] > 0.9


# ---------- model behavior on synthetic data ----------


@pytest.mark.parametrize("name", ALL)
def test_models_recover_heterogeneous_effects_and_policy_value(name, data, models):
    df, _, _, tau, p = data
    m = models[name]
    up = m.predict_uplift(df)
    for k in (1, 2):
        assert np.corrcoef(up[:, k], tau[:, k])[0, 1] > 0.8
        assert up[:, k].mean() == pytest.approx(tau[:, k].mean(), abs=0.03)
    learned = true_value(m.recommend(df), p)
    oracle = true_value(tau.argmax(axis=1), p)
    assert learned > true_value(np.zeros(len(df), dtype=int), p) + 0.05
    assert learned > true_value(np.ones(len(df), dtype=int), p)
    assert learned > oracle - 0.02


@pytest.mark.parametrize("name", ALL)
def test_aa_data_gives_near_zero_uplift(name):
    df, arm, y, tau, p = synth(effects=False, seed=5)
    m = fitted(name, df, arm, y)
    up = m.predict_uplift(df)
    assert np.abs(up).max() < 0.15
    # policy value of whatever it recommends equals no-email value (no true effects)
    assert abs(true_value(m.recommend(df), p) - true_value(np.zeros(len(df), dtype=int), p)) < 1e-9


def test_dr_learner_pseudo_outcomes_are_unbiased_for_true_effect(data, models):
    _, _, _, tau, _ = data
    pseudo = models["dr_learner"].pseudo_outcomes_
    assert pseudo.shape == (len(tau), 2)
    se = pseudo.std(axis=0) / np.sqrt(len(pseudo))
    assert abs(pseudo[:, 0].mean() - tau[:, 1].mean()) < 4 * se[0]
    assert abs(pseudo[:, 1].mean() - tau[:, 2].mean()) < 4 * se[1]


def test_dr_nuisance_is_cross_fitted(models, data):
    n = len(data[0])
    nuis = models["dr_learner"].oof_nuisance_
    assert isinstance(nuis, OutcomeModelPredictions)
    assert set(np.unique(nuis.group)) == {0, 1}
    for g, train_rows in nuis.trained_on.items():
        predicted = nuis.row_ids[nuis.group == g]
        assert np.intersect1d(predicted, train_rows).size == 0
        assert len(predicted) + len(train_rows) == n


# ---------- leakage prevention ----------


@pytest.mark.parametrize("name", ALL)
def test_outcomes_and_treatment_columns_never_enter_the_model(name, data, models):
    df, arm, y, *_ = data
    leaky = df.assign(
        visit=y,
        conversion=y,
        spend=y * 100,
        segment=np.array(["No E-Mail", "Mens E-Mail", "Womens E-Mail"])[arm],
        history_segment="x",
    )
    m = fitted(name, leaky, arm, y)
    assert list(m.preprocessor_.feature_names_in_) == FEATURES
    np.testing.assert_allclose(m.predict_uplift(df), models[name].predict_uplift(df))
    np.testing.assert_allclose(m.predict_uplift(leaky), models[name].predict_uplift(df))


@pytest.mark.parametrize("name", ALL)
def test_missing_feature_raises(name, data, models):
    with pytest.raises(KeyError):
        models[name].predict_uplift(data[0].drop(columns=["channel"]))


# ---------- reproducibility ----------


@pytest.mark.parametrize("name", ALL)
def test_same_seed_gives_identical_predictions(name, data, models):
    df, arm, y, *_ = data
    again = fitted(name, df, arm, y, seed=42)
    np.testing.assert_array_equal(again.predict_uplift(df), models[name].predict_uplift(df))


@pytest.mark.parametrize("name", ["s_learner", "t_learner", "dr_learner"])
def test_different_seed_changes_lightgbm_models(name, data, models):
    df, arm, y, *_ = data
    other = fitted(name, df, arm, y, seed=7)
    assert not np.array_equal(other.predict_uplift(df), models[name].predict_uplift(df))


# ---------- validation ----------


def test_fit_validation_and_unfitted_model():
    df, arm, y, *_ = synth(600)
    with pytest.raises(RuntimeError):
        make_model("logistic").predict_uplift(df)
    m = make_model("logistic")
    with pytest.raises(ValueError):
        m.fit(df, arm, y + 2)  # non-binary
    with pytest.raises(ValueError):
        m.fit(df, arm + 1, y)  # arm code 3
    with pytest.raises(ValueError):
        m.fit(df, np.where(arm == 2, 1, arm), y)  # arm 2 missing
    with pytest.raises(ValueError):
        m.fit(df.iloc[:10], arm, y)  # length mismatch
    with pytest.raises(ValueError):
        make_model("xgboost")


def test_single_class_arm_falls_back_to_constant_probability():
    df, arm, y, *_ = synth(900, seed=3)
    y = np.where(arm == 2, 0.0, y)  # arm 2 never visits
    m = fitted("t_learner", df, arm, y)
    assert (m.predict_outcomes(df)[:, 2] == 0).all()


def test_early_stopping_can_be_disabled():
    df, arm, y, *_ = synth(2000, seed=4)
    m = fitted("t_learner", df, arm, y, early_stopping_rounds=None, params={"n_estimators": 20})
    assert all(mod.booster_.num_trees() == 20 for mod in m.models_)


# ---------- compatibility with the Stage 2 policy evaluator and Stage 1 folds ----------


@pytest.mark.parametrize("name", ALL)
def test_recommendations_feed_policy_evaluator_on_a_cv_fold(name):
    df, arm, y, tau, p = synth(8000, seed=9)
    full = df.assign(segment=np.array(["No E-Mail", "Mens E-Mail", "Womens E-Mail"])[arm], visit=y)
    split = make_split(full)
    train_idx, val_idx = split.fold_indices(0)
    m = fitted(name, df.iloc[train_idx], arm[train_idx], y[train_idx])
    rec = m.recommend(df.iloc[val_idx])
    table = evaluate_policy(y[val_idx], arm[val_idx], rec, n_boot=100)
    assert table["policy"].iloc[0] == "candidate"
    assert table.loc[0, "value"] > table.set_index("policy").loc["no_email", "value"]
    # outcome-model learners also supply out-of-sample nuisance predictions for AIPW
    if name != "dr_learner":
        nuis = OutcomeModelPredictions.from_holdout(
            m.predict_outcomes(df.iloc[val_idx]), val_idx, train_idx
        )
        aipw = evaluate_policy(y[val_idx], arm[val_idx], rec, nuis, n_boot=100)
        assert (aipw["estimator"] == "AIPW").all()
