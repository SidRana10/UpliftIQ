import numpy as np
import pandas as pd
import pytest

from upliftiq.evaluation.policy import (
    OutcomeModelPredictions,
    arm_scores,
    baseline_policies,
    bootstrap_means,
    estimate_policy_value,
    evaluate_policy,
    policy_scores,
)
from upliftiq.modeling.split import make_split

# ---------- synthetic experiment with known potential-outcome probabilities ----------

GRID = (np.arange(200_000) + 0.5) / 200_000


def probs(x, effects=True):
    """True P(Y=1 | x, arm) for arms 0, 1, 2. Arm 1 helps high x, arm 2 helps low x."""
    base = 0.10 + 0.05 * x
    if not effects:
        return np.column_stack([base, base, base])
    return np.column_stack([base, base + 0.12 * x, base + 0.06 * (1 - x)])


def candidate(x):
    return np.where(x > 0.5, 1, 2)


def true_value(policy_fn, effects=True):
    p = probs(GRID, effects)
    return float(p[np.arange(len(GRID)), policy_fn(GRID)].mean())


def true_baselines(effects=True):
    p = probs(GRID, effects).mean(axis=0)
    return {"no_email": p[0], "mens_for_all": p[1], "womens_for_all": p[2], "random": p.mean()}


def simulate(n, rng, effects=True, arm_probs=(1 / 3, 1 / 3, 1 / 3)):
    x = rng.random(n)
    arm = rng.choice(3, size=n, p=arm_probs)
    y = (rng.random(n) < probs(x, effects)[np.arange(n), arm]).astype(float)
    return x, arm, y


def oracle_nuisance(x, effects=True):
    n = len(x)
    return OutcomeModelPredictions.from_holdout(
        probs(x, effects), np.arange(n), np.array([], dtype=int)
    )


# ---------- estimator correctness and indexing ----------


def test_ipw_and_aipw_scores_match_hand_computation():
    y = np.array([1.0, 0.0, 1.0])
    arm = np.array([0, 1, 2])
    ipw = arm_scores(y, arm)
    np.testing.assert_allclose(ipw, [[3, 0, 0], [0, 0, 0], [0, 0, 3]])
    mu = np.array([[0.5, 0.2, 0.1]] * 3)
    nuis = OutcomeModelPredictions.from_holdout(mu, [0, 1, 2], [10, 11])
    aipw = arm_scores(y, arm, nuis)
    # row 0, arm 0 assigned: mu + 3*(y-mu) = 0.5 + 3*0.5; other arms: mu only
    np.testing.assert_allclose(aipw[0], [0.5 + 3 * 0.5, 0.2, 0.1])
    np.testing.assert_allclose(aipw[1], [0.5, 0.2 + 3 * (0 - 0.2), 0.1])


def test_policy_scores_use_the_right_arm_column():
    y = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    arm = np.array([2, 2, 2, 0, 1, 1])  # only Womens (code 2) converts
    s = arm_scores(y, arm)
    assert policy_scores(s, np.full(6, 2)).mean() == pytest.approx(1.5)
    assert policy_scores(s, np.full(6, 1)).mean() == 0
    assert policy_scores(s, np.zeros(6, dtype=int)).mean() == 0
    mixed = np.array([2, 0, 1, 2, 0, 1])
    np.testing.assert_allclose(policy_scores(s, mixed), s[np.arange(6), mixed])


def test_stochastic_policy_matches_mixture_and_random_baseline():
    rng = np.random.default_rng(0)
    x, arm, y = simulate(500, rng)
    s = arm_scores(y, arm)
    base = baseline_policies(len(y))
    np.testing.assert_allclose(policy_scores(s, base["random"]), s.mean(axis=1))
    onehot = np.eye(3)[candidate(x)]
    np.testing.assert_allclose(policy_scores(s, onehot), policy_scores(s, candidate(x)))


def test_ipw_arm_values_equal_observed_arm_rates_for_exact_balance():
    arm = np.repeat([0, 1, 2], 100)
    y = np.concatenate(
        [np.ones(10), np.zeros(90), np.ones(30), np.zeros(70), np.ones(20), np.zeros(80)]
    )
    est = [estimate_policy_value(y, arm, np.full(300, a)).value for a in range(3)]
    np.testing.assert_allclose(est, [0.10, 0.30, 0.20])


def test_non_uniform_propensity_vector_is_handled():
    rng = np.random.default_rng(1)
    vals = []
    for _ in range(200):
        x, arm, y = simulate(3000, rng, arm_probs=(0.5, 0.25, 0.25))
        vals.append(
            estimate_policy_value(
                y, arm, np.ones(3000, dtype=int), propensity=[0.5, 0.25, 0.25]
            ).value
        )
    truth = true_baselines()["mens_for_all"]
    assert np.mean(vals) == pytest.approx(truth, abs=4 * np.std(vals) / np.sqrt(200))
    # using the wrong (uniform) propensity is biased
    x, arm, y = simulate(40000, rng, arm_probs=(0.5, 0.25, 0.25))
    wrong = estimate_policy_value(y, arm, np.ones(40000, dtype=int), propensity=1 / 3).value
    assert abs(wrong - truth) > 0.02


def test_input_validation():
    y, arm = np.array([1.0, 0.0]), np.array([0, 1])
    s = arm_scores(y, arm)
    for bad_arm in (np.array([0, 3]), np.array([0, -1]), np.array([0.0, 1.0])):
        with pytest.raises(ValueError):
            arm_scores(y, bad_arm)
    with pytest.raises(ValueError):
        arm_scores(np.array([np.nan, 0.0]), arm)
    with pytest.raises(ValueError):
        arm_scores(y, arm[:1])
    with pytest.raises(ValueError):
        arm_scores(np.array([]), np.array([], dtype=int))
    for bad_p in (0, -0.1, 1.5):
        with pytest.raises(ValueError):
            arm_scores(y, arm, propensity=bad_p)
    with pytest.raises(ValueError):
        policy_scores(s, np.array([0, 3]))
    with pytest.raises(ValueError):
        policy_scores(s, np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        policy_scores(s, np.array([[0.5, 0.5, 0.5], [1, 0, 0]]))
    with pytest.raises(ValueError):
        policy_scores(s, np.array([0, 1, 2]))


def test_single_observation_has_nan_se():
    est = estimate_policy_value([1.0], [0], [0])
    assert est.n == 1 and np.isnan(est.se)


# ---------- out-of-sample safeguard for AIPW nuisance predictions ----------


def test_in_sample_nuisance_predictions_are_rejected():
    mu = np.full((4, 3), 0.1)
    with pytest.raises(ValueError, match="out of sample"):
        OutcomeModelPredictions.from_holdout(mu, [0, 1, 2, 3], [3, 4, 5])
    with pytest.raises(ValueError, match="no training rows"):
        OutcomeModelPredictions(mu, np.arange(4), np.array([0, 0, 1, 1]), {0: np.array([9])})


def test_raw_arrays_and_misaligned_predictions_rejected():
    y, arm = np.array([1.0, 0.0]), np.array([0, 1])
    with pytest.raises(TypeError):
        arm_scores(y, arm, np.full((2, 3), 0.1))
    nuis = OutcomeModelPredictions.from_holdout(np.full((3, 3), 0.1), [0, 1, 2], [9])
    with pytest.raises(ValueError):
        arm_scores(y, arm, nuis)
    with pytest.raises(ValueError):
        OutcomeModelPredictions.from_holdout(np.full((2, 2), 0.1), [0, 1], [9])
    with pytest.raises(ValueError):
        OutcomeModelPredictions.from_holdout(np.full((2, 3), np.nan), [0, 1], [9])


def test_cross_fitted_predictions_from_stage1_folds_are_accepted():
    n = 1000
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "segment": rng.choice(["No E-Mail", "Mens E-Mail", "Womens E-Mail"], n),
            "visit": rng.integers(0, 2, n),
        }
    )
    split = make_split(df)
    dev = split.dev_idx
    mu = np.full((len(dev), 3), 0.15)
    trained_on = {k: split.fold_indices(k)[0] for k in range(split.n_folds)}
    ok = OutcomeModelPredictions(mu, dev, split.assignment[dev], trained_on)
    assert len(ok.mu) == len(dev)
    # same predictions, but pretend every fold's model was trained on all of dev: in-sample
    leaky = {k: dev for k in range(split.n_folds)}
    with pytest.raises(ValueError, match="out of sample"):
        OutcomeModelPredictions(mu, dev, split.assignment[dev], leaky)


# ---------- evaluation table and bootstrap ----------


def test_evaluate_policy_table_structure_and_reproducibility():
    rng = np.random.default_rng(2)
    x, arm, y = simulate(3000, rng)
    a = evaluate_policy(y, arm, candidate(x), n_boot=200, seed=5)
    b = evaluate_policy(y, arm, candidate(x), n_boot=200, seed=5)
    c = evaluate_policy(y, arm, candidate(x), n_boot=200, seed=6)
    pd.testing.assert_frame_equal(a, b)
    assert not a["ci_low"].equals(c["ci_low"])
    assert a["policy"].tolist() == [
        "candidate",
        "no_email",
        "mens_for_all",
        "womens_for_all",
        "random",
    ]
    assert (a["estimator"] == "IPW").all()
    assert ((a["ci_low"] <= a["value"]) & (a["value"] <= a["ci_high"])).all()
    rnd = a.set_index("policy")
    arm_vals = [estimate_policy_value(y, arm, np.full(len(y), k)).value for k in range(3)]
    assert rnd.loc["random", "value"] == pytest.approx(np.mean(arm_vals))
    assert a.loc[0, "diff"] != a.loc[0, "diff"]  # NaN on candidate row


def test_paired_bootstrap_identical_policy_has_zero_width_difference_ci():
    rng = np.random.default_rng(3)
    x, arm, y = simulate(1000, rng)
    t = evaluate_policy(y, arm, np.ones(1000, dtype=int), n_boot=100).set_index("policy")
    row = t.loc["mens_for_all"]
    assert row["diff"] == 0 and row["diff_ci_low"] == 0 and row["diff_ci_high"] == 0


def test_bootstrap_means_shape_seed_and_validation():
    psi = np.random.default_rng(0).random((50, 3))
    d1, d2 = bootstrap_means(psi, 120, seed=1), bootstrap_means(psi, 120, seed=1)
    assert d1.shape == (120, 3)
    np.testing.assert_array_equal(d1, d2)
    assert abs(d1.mean() - psi.mean()) < 0.02
    with pytest.raises(ValueError):
        bootstrap_means(psi, 0)


def test_aipw_table_uses_nuisance_and_labels_estimator():
    rng = np.random.default_rng(4)
    x, arm, y = simulate(2000, rng)
    t = evaluate_policy(y, arm, candidate(x), oracle_nuisance(x), n_boot=100)
    assert (t["estimator"] == "AIPW").all()


# ---------- validation against synthetic truth ----------


def _run_sims(n_sims, n, effects, seed, use_aipw, mu_kind="oracle"):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_sims):
        x, arm, y = simulate(n, rng, effects)
        nuis = None
        if use_aipw:
            if mu_kind == "oracle":
                nuis = oracle_nuisance(x, effects)
            else:  # badly wrong but out-of-sample outcome model
                nuis = OutcomeModelPredictions.from_holdout(
                    np.column_stack([np.full(n, 0.5), np.full(n, 0.05), np.full(n, 0.4)]),
                    np.arange(n),
                    np.array([], dtype=int),
                )
        est = estimate_policy_value(y, arm, candidate(x), nuis)
        out.append((est.value, est.se))
    return np.array(out)


@pytest.mark.parametrize(
    "use_aipw,mu_kind",
    [(False, None), (True, "oracle"), (True, "wrong")],
    ids=["ipw", "aipw", "aipw_wrong_mu"],
)
def test_estimators_unbiased_and_ci_coverage_with_known_effects(use_aipw, mu_kind):
    truth = true_value(candidate)
    sims = _run_sims(400, 3000, True, 10, use_aipw, mu_kind)
    vals, ses = sims[:, 0], sims[:, 1]
    assert vals.mean() == pytest.approx(truth, abs=4 * vals.std() / np.sqrt(len(vals)))
    covered = np.abs(vals - truth) <= 1.96 * ses
    assert 0.92 <= covered.mean() <= 0.99


def test_aipw_with_good_outcome_model_has_lower_variance_than_ipw():
    ipw = _run_sims(300, 3000, True, 11, False)[:, 0]
    aipw = _run_sims(300, 3000, True, 11, True, "oracle")[:, 0]
    # Bernoulli noise dominates at ~15% rates, so the oracle gain is modest but must be positive
    assert aipw.std() < 0.97 * ipw.std()


@pytest.mark.parametrize("effects", [True, False], ids=["effects", "A/A"])
@pytest.mark.parametrize("use_aipw", [False, True], ids=["ipw", "aipw"])
def test_bootstrap_difference_ci_coverage_and_AA_null(effects, use_aipw):
    rng = np.random.default_rng(20)
    truth_cand = true_value(candidate, effects)
    truth_base = true_baselines(effects)
    n_sims, n = 120, 2000
    covered, diffs = [], []
    for _ in range(n_sims):
        x, arm, y = simulate(n, rng, effects)
        nuis = oracle_nuisance(x, effects) if use_aipw else None
        t = evaluate_policy(y, arm, candidate(x), nuis, n_boot=300, seed=int(rng.integers(1 << 30)))
        t = t.set_index("policy").loc[["no_email", "mens_for_all", "womens_for_all", "random"]]
        true_diff = np.array([truth_cand - truth_base[p] for p in t.index])
        covered.append(
            ((t["diff_ci_low"] <= true_diff) & (true_diff <= t["diff_ci_high"])).to_numpy()
        )
        diffs.append(t["diff"].to_numpy())
    covered, diffs = np.array(covered), np.array(diffs)
    assert covered.mean() >= 0.90
    if not effects:  # A/A: no true effect, estimates centre on zero and CIs rarely exclude it
        assert np.abs(diffs.mean(axis=0)).max() < 4 * diffs.std(axis=0).max() / np.sqrt(n_sims)
        assert (1 - covered).mean() <= 0.10


def test_clearly_better_policy_is_detected_with_known_effects():
    rng = np.random.default_rng(30)
    x, arm, y = simulate(30000, rng)
    t = evaluate_policy(y, arm, candidate(x), n_boot=300).set_index("policy")
    assert t.loc["no_email", "diff_ci_low"] > 0
    assert t.loc["random", "diff_ci_low"] > 0
    truth = true_value(candidate)
    assert t.loc["candidate", "ci_low"] <= truth <= t.loc["candidate", "ci_high"]
