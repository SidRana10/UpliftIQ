"""Criteo binary-treatment learners on synthetic 85/15 experiments with known effects."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from upliftiq.data.criteo import FEATURES
from upliftiq.evaluation import binary_policy as bp
from upliftiq.evaluation.policy import OutcomeModelPredictions
from upliftiq.modeling import criteo_learners as cl

ALL = list(cl.MODELS)
FAST = {"n_estimators": 60}  # keeps tests quick; defaults are exercised in the probe/smoke run


def experiment(n, seed, aa=False, p_treat=0.85):
    """Visit-like outcome: p0 in [0.22, 0.38]; true tau = 0.18 * tanh(1.5 x0) (0 under A/A)."""
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, 12)).astype(np.float32)
    p0 = 0.3 + 0.08 * np.tanh(X[:, 1])
    tau = np.zeros(n) if aa else 0.18 * np.tanh(1.5 * X[:, 0])
    t = (r.random(n) < p_treat).astype(np.int8)
    y = (r.random(n) < p0 + t * tau).astype(float)
    return X, t, y, tau


def build(name, **kw):
    params = {} if name == "logistic" else {"params": FAST}
    return cl.make_model(name, n_jobs=2, **params, **kw)


@pytest.fixture(scope="module")
def data():
    return experiment(40_000, 1), experiment(20_000, 2)


@pytest.fixture(scope="module")
def fitted(data):
    (X, t, y, _), _ = data
    return {name: build(name).fit(X, t, y) for name in ALL}


@pytest.mark.parametrize("name", ALL)
def test_shapes_direction_and_recovery_of_planted_heterogeneity(name, data, fitted):
    _, (Xv, _, _, tau) = data
    m = fitted[name]
    u = m.predict_uplift(Xv)
    assert u.shape == (len(Xv),)
    assert np.corrcoef(u, tau)[0, 1] > 0.8
    assert u[Xv[:, 0] > 0.5].mean() > 0.05 > -0.05 > u[Xv[:, 0] < -0.5].mean()  # direction
    rec = m.recommend(Xv)
    assert rec.dtype == np.int8 and np.array_equal(rec, (u > 0).astype(np.int8))
    assert rec[Xv[:, 0] > 0.5].mean() > 0.9 and rec[Xv[:, 0] < -0.5].mean() < 0.1
    top = m.recommend(Xv, budget=0.2)
    assert top.sum() == int(np.floor(0.2 * len(Xv))) and tau[top == 1].mean() > tau.mean()
    if name != "dr_learner":
        mu = m.predict_outcomes(Xv)
        assert mu.shape == (len(Xv), 2) and ((mu >= 0) & (mu <= 1)).all()
        np.testing.assert_allclose(mu[:, 1] - mu[:, 0], u, atol=1e-12)
    else:
        with pytest.raises(NotImplementedError):
            m.predict_outcomes(Xv)


@pytest.mark.parametrize("name", ALL)
def test_aa_experiment_gives_no_apparent_uplift(name):
    X, t, y, _ = experiment(40_000, 3, aa=True)
    Xv, tv, yv, _ = experiment(60_000, 4, aa=True)
    m = build(name).fit(X, t, y)
    u = m.predict_uplift(Xv)
    assert np.abs(u).mean() < 0.04
    res = bp.evaluate_targeting(yv, tv, {name: u}, n_boot=100)
    assert ((res.qini.ci_low <= 0) & (res.qini.ci_high >= 0)).all()
    pol = res.policies.query("policy == 'budget'")
    assert ((pol.diff_vs_random_ci_low <= 0) & (pol.diff_vs_random_ci_high >= 0)).all()


@pytest.mark.parametrize("name", ["logistic", "s_learner", "t_learner", "dr_learner"])
def test_treatment_indexing_flipping_labels_flips_the_effect(name, data):
    (X, t, y, _), (Xv, _, _, tau) = data
    kw = {"p_treat": 0.15} if name == "dr_learner" else {}
    flipped = build(name, **kw).fit(X, 1 - t, y)
    assert np.corrcoef(flipped.predict_uplift(Xv), tau)[0, 1] < -0.8


def test_reproducible_and_seed_sensitive(data):
    (X, t, y, _), (Xv, *_) = data
    for name in ALL:
        a, b = build(name).fit(X, t, y), build(name).fit(X, t, y)
        np.testing.assert_array_equal(a.predict_uplift(Xv), b.predict_uplift(Xv))
    for name in ("s_learner", "t_learner", "dr_learner"):  # subsampling and splits use the seed
        c = build(name, seed=7).fit(X, t, y)
        assert not np.array_equal(c.predict_uplift(Xv), build(name).fit(X, t, y).predict_uplift(Xv))


@pytest.mark.parametrize("name", ALL)
def test_only_f0_f11_are_used_extra_columns_are_ignored(name, data):
    (X, t, y, _), (Xv, *_) = data
    n = len(X)
    rng = np.random.default_rng(0)

    def frame(Z, extras):
        df = pd.DataFrame(Z, columns=FEATURES)
        for k, v in extras.items():
            df[k] = v
        return df

    clean = frame(X, {})
    poisoned = frame(
        X,
        {
            "treatment": rng.integers(0, 2, n),
            "exposure": rng.integers(0, 2, n),
            "visit": y,
            "conversion": rng.integers(0, 2, n),
        },
    )
    a = build(name).fit(clean, t, y)
    b = build(name).fit(poisoned, t, y)
    vp = frame(Xv, {"treatment": 1, "exposure": 1, "visit": 1, "conversion": 1})
    np.testing.assert_array_equal(a.predict_uplift(frame(Xv, {})), b.predict_uplift(vp))
    np.testing.assert_array_equal(a.predict_uplift(frame(Xv, {})), a.predict_uplift(Xv))
    with pytest.raises(ValueError):
        a.predict_uplift(Xv[:, :11])
    with pytest.raises(KeyError):
        a.predict_uplift(clean.drop(columns=["f3"]))


def test_logistic_scaling_comes_from_training_rows_only(data):
    (X, t, y, _), (Xv, *_) = data
    m = build("logistic").fit(X, t, y)
    before = m.scaler_.to_dict()
    m.predict_uplift(Xv * 1000)  # wildly different prediction rows
    assert m.scaler_.to_dict() == before
    fit_stats = cl.fit_scaler(X, np.arange(len(X))).to_dict()
    assert before == fit_stats


def test_dr_cross_fitting_never_predicts_a_row_with_a_model_trained_on_it(fitted):
    m = fitted["dr_learner"]
    nuis = m.oof_nuisance_
    n = len(m.pseudo_outcomes_)
    assert isinstance(nuis, OutcomeModelPredictions) and nuis.n_arms == 2
    assert set(np.unique(nuis.group)) == set(range(m.n_cross_fit))
    for g, fit_rows in nuis.trained_on.items():
        held = nuis.row_ids[nuis.group == g]
        assert not set(fit_rows) & set(held) and len(fit_rows) + len(held) == n
    # the pseudo-outcome has mean near the true average effect (unbiased score)
    assert m.pseudo_outcomes_.mean() == pytest.approx(0.0, abs=0.03)
    # the evaluator's provenance guard rejects an in-sample variant of the same predictions
    with pytest.raises(ValueError, match="out of sample"):
        OutcomeModelPredictions.from_holdout(nuis.mu, np.arange(n), np.arange(100), n_arms=2)


def test_models_plug_into_the_c2_evaluator_with_out_of_sample_nuisance(data, fitted):
    (X, t, y, _), (Xv, tv, yv, tau) = data
    n_tr, n_v = len(X), len(Xv)
    ref = fitted["t_learner"]  # outcome model trained on the fit rows, applied to disjoint rows
    nuis = OutcomeModelPredictions.from_holdout(
        ref.predict_outcomes(Xv), np.arange(n_tr, n_tr + n_v), np.arange(n_tr), n_arms=2
    )
    scores = {name: m.predict_uplift(Xv) for name, m in fitted.items()}
    res = bp.evaluate_targeting(yv, tv, scores, nuis, n_boot=60, row_ids=nuis.row_ids)
    q = res.qini.query("estimator == 'AIPW'").set_index("model")
    assert (q.ci_low > 0).all()  # every informative model beats the random line
    pol = res.policies.query(
        "estimator == 'AIPW' and policy == 'budget' and budget_requested == 0.3"
    )
    assert (pol.diff_vs_random_ci_low > 0).all()
    # a recommend(budget=...) policy is exactly the evaluator's budget policy
    rec = fitted["s_learner"].recommend(Xv, budget=0.3, row_ids=nuis.row_ids)
    direct = bp.budget_policy(scores["s_learner"], 0.3, nuis.row_ids)
    assert np.array_equal(rec, direct)
    tau_hat = bp.incremental_scores(yv, tv, nuis)
    row = pol.set_index("model").loc["s_learner"]
    assert row.value == pytest.approx(bp.policy_value(tau_hat, rec), abs=1e-12)


def test_input_validation(data):
    (X, t, y, _), _ = data
    with pytest.raises(RuntimeError):
        build("t_learner").predict_uplift(X)
    for bad_t in (t + 1, t.astype(float), np.zeros_like(t), np.ones_like(t)):
        with pytest.raises(ValueError):
            build("t_learner").fit(X, bad_t, y)
    with pytest.raises(ValueError):
        build("t_learner").fit(X, t, y * 2)
    with pytest.raises(ValueError):
        build("t_learner").fit(X, t[:-1], y)
    with pytest.raises(ValueError):
        build("t_learner").fit(np.where(np.isnan(X), 0, X)[:, :5], t, y)
    with pytest.raises(ValueError):
        cl.make_model("tarnet")
