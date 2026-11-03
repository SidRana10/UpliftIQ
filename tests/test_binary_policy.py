"""Binary-treatment policy evaluation on synthetic 85/15 experiments with known effects."""

from __future__ import annotations

import inspect
import tracemalloc

import numpy as np
import pytest

from upliftiq.evaluation import binary_policy as bp
from upliftiq.evaluation.policy import OutcomeModelPredictions, arm_scores


def experiment(n, seed=0, aa=False, p_treat=0.85):
    """Visit-like outcome: p0 in [0.10, 0.20], true tau = 0.03 + 0.06 tanh(x0) (0 under A/A)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    p0 = 0.15 + 0.05 * np.tanh(x[:, 1])
    tau = np.zeros(n) if aa else 0.03 + 0.06 * np.tanh(x[:, 0])
    t = (rng.random(n) < p_treat).astype(np.int8)
    y = (rng.random(n) < p0 + t * tau).astype(np.float64)
    return {"x": x, "p0": p0, "tau": tau, "t": t, "y": y}


def true_mu(d, noise=0.0, seed=1):
    mu = np.column_stack([d["p0"], d["p0"] + d["tau"]])
    return mu + np.random.default_rng(seed).normal(0, noise, mu.shape)


def holdout(mu, n):
    """Nuisance from a model 'trained' on rows disjoint from the n scored rows."""
    return OutcomeModelPredictions.from_holdout(mu, np.arange(n), np.arange(n, n + 10), n_arms=2)


def truth_gain(tau_true, score, budget, ids=None):
    n = len(tau_true)
    top = bp.rank_order(score, ids)[: bp.n_treated_for_budget(budget, n)]
    return tau_true[top].sum() / n


# ---------- estimators ----------


def test_ate_and_budget_gain_are_unbiased_and_covered(  # noqa: PLR0915
):
    d = experiment(300_000, seed=3)
    n = len(d["y"])
    res = bp.evaluate_targeting(
        d["y"], d["t"], {"oracle": d["tau"]}, holdout(true_mu(d, 0.02), n), n_boot=100
    )
    for est in ("AIPW", "IPW"):
        pol = res.policies[(res.policies.estimator == est) & (res.policies.policy == "budget")]
        row = pol[pol.budget_requested == 0.2].iloc[0]
        truth = truth_gain(d["tau"], d["tau"], 0.2)
        assert abs(row.value - truth) < 4 * row.se
        assert row.ci_low <= truth <= row.ci_high
        ref = res.references
        ate = ref[(ref.estimator == est) & (ref.policy == "treat_all")].iloc[0]
        assert abs(ate.value - d["tau"].mean()) < 4 * 0.0006 + 0.002


def test_interval_coverage_across_repeated_experiments():
    hits_ate, hits_gain, trials = 0, 0, 40
    for r in range(trials):
        d = experiment(20_000, seed=100 + r)
        res = bp.evaluate_targeting(d["y"], d["t"], {"oracle": d["tau"]}, n_boot=60, seed=r)
        pol = res.policies[res.policies.budget_requested == 0.3].iloc[0]
        hits_gain += pol.ci_low <= truth_gain(d["tau"], d["tau"], 0.3) <= pol.ci_high
        ate = res.references.query("policy == 'treat_all'").iloc[0]
        hits_ate += ate.ci_low <= d["tau"].mean() <= ate.ci_high
    assert hits_gain / trials >= 0.85 and hits_ate / trials >= 0.85


def test_aipw_with_good_outcome_model_has_smaller_standard_error_than_ipw():
    d = experiment(100_000, seed=5)
    n = len(d["y"])
    res = bp.evaluate_targeting(
        d["y"], d["t"], {"oracle": d["tau"]}, holdout(true_mu(d, 0.01), n), n_boot=20
    )
    pol = res.policies.query("policy == 'budget' and budget_requested == 0.5")
    se = pol.set_index("estimator").se
    assert se["AIPW"] < 0.97 * se["IPW"]


def test_aa_experiment_has_no_apparent_uplift():
    d = experiment(200_000, seed=7, aa=True)
    n = len(d["y"])
    rng = np.random.default_rng(8)
    scores = {"noise": rng.normal(size=n), "feature": d["x"][:, 0]}  # uninformative scores
    res = bp.evaluate_targeting(d["y"], d["t"], scores, holdout(true_mu(d, 0.01), n), n_boot=100)
    assert ((res.qini.ci_low <= 0) & (res.qini.ci_high >= 0)).all()
    pol = res.policies.query("policy == 'budget'")
    assert ((pol.ci_low <= 0) & (pol.ci_high >= 0)).all()
    assert ((pol.diff_vs_random_ci_low <= 0) & (pol.diff_vs_random_ci_high >= 0)).all()


def test_informative_scores_beat_random_and_uninformative_scores_do_not():
    d = experiment(300_000, seed=9)
    n = len(d["y"])
    scores = {"oracle": d["tau"], "noise": np.random.default_rng(1).normal(size=n)}
    res = bp.evaluate_targeting(d["y"], d["t"], scores, n_boot=100)
    q = res.qini.set_index("model")
    assert q.loc["oracle", "ci_low"] > 0
    assert q.loc["noise", "ci_low"] < 0 < q.loc["noise", "ci_high"]
    # paired: the noise ranking minus the oracle (reference) excludes zero
    pair = res.pairwise.query("metric == 'qini'").iloc[0]
    assert pair.ci_high < 0 and pair["diff"] < 0


def test_qini_matches_the_analytic_area_for_the_oracle_ranking():
    d = experiment(400_000, seed=11)
    res = bp.evaluate_targeting(d["y"], d["t"], {"oracle": d["tau"]}, n_boot=10)
    n = len(d["tau"])
    g = np.cumsum(np.sort(d["tau"])[::-1]) / n
    grid = [int(np.floor(i / 100 * n + 1e-9)) for i in range(1, 101)]
    f = np.r_[0, np.array(grid) / n]
    gg = np.r_[0, g[np.array(grid) - 1]]
    truth = np.sum(0.5 * (gg[1:] + gg[:-1]) * np.diff(f)) - 0.5 * g[-1]
    est = res.qini.set_index("estimator").loc["IPW"]
    assert est.qini == pytest.approx(truth, abs=0.0025)
    # the curve ends at the average treatment effect (85/15 allocation does not distort it)
    last = res.curves.query("estimator == 'IPW'").iloc[-1]
    assert last.budget == 1.0 and last.gain == pytest.approx(d["tau"].mean(), abs=0.004)


# ---------- policies and budgets ----------


def test_budget_policy_treats_exactly_the_requested_top_share():
    rng = np.random.default_rng(0)
    s = rng.normal(size=1001)
    for b in (0.0, 0.1, 0.333, 0.5, 1.0):
        pol = bp.budget_policy(s, b)
        k = int(np.floor(b * 1001 + 1e-9))
        assert pol.sum() == k and pol.dtype == np.int8
        if 0 < k < 1001:
            assert s[pol == 1].min() >= s[pol == 0].max()
    with pytest.raises(ValueError):
        bp.budget_policy(s, 1.2)
    with pytest.raises(ValueError):
        bp.budget_policy(np.r_[s[:5], np.nan], 0.5)


def test_tie_breaking_is_deterministic_and_independent_of_row_order():
    n = 1000
    s = np.zeros(n)
    ids = np.arange(n)
    a = bp.budget_policy(s, 0.2, ids)
    assert a.sum() == 200 and np.array_equal(a, bp.budget_policy(s, 0.2, ids))
    assert not a[:200].all()  # not "first rows by position"
    perm = np.random.default_rng(1).permutation(n)
    b = bp.budget_policy(s[perm], 0.2, ids[perm])
    assert set(ids[perm][b == 1]) == set(ids[a == 1])  # same customers whatever the row order
    assert set(bp.budget_policy(s, 0.2, ids, seed=7).nonzero()[0]) != set(a.nonzero()[0])


def test_policy_table_matches_direct_computation_and_labels_feasibility():
    d = experiment(50_000, seed=13)
    n = len(d["y"])
    nuis = holdout(true_mu(d, 0.02), n)
    score = d["x"][:, 0] + np.random.default_rng(2).normal(0, 0.5, n)
    res = bp.evaluate_targeting(d["y"], d["t"], {"m": score}, nuis, n_boot=5)
    tau = bp.incremental_scores(d["y"], d["t"], nuis)
    pol = res.policies.query("estimator == 'AIPW'").set_index(["policy", "budget_requested"])
    for b in bp.DEFAULT_BUDGETS:
        direct = bp.policy_value(tau, bp.budget_policy(score, b, nuis.row_ids))
        assert pol.loc[("budget", b), "value"] == pytest.approx(direct, abs=1e-12)
        k = pol.loc[("budget", b), "n_treated"]
        assert k == int(np.floor(b * n + 1e-9))
        rnd = res.references.query(f"estimator == 'AIPW' and policy == 'random@{b:.0%}'").iloc[0]
        assert rnd.value == pytest.approx(k / n * tau.mean(), abs=1e-12)
        assert "feasible" in rnd.feasibility and "unconstrained" not in rnd.feasibility
    thr = res.policies.query("estimator == 'AIPW' and policy == 'uplift_gt_0'").iloc[0]
    assert thr.n_treated == (score > 0).sum()
    assert thr.value == pytest.approx(bp.policy_value(tau, bp.threshold_policy(score)), abs=1e-12)
    refs = res.references.query("estimator == 'AIPW'").set_index("policy")
    assert refs.loc["treat_nobody", "value"] == 0.0
    assert "unconstrained" in refs.loc["treat_all", "feasibility"]
    assert refs.loc["treat_all", "value"] == pytest.approx(tau.mean(), abs=1e-12)
    # budget-100% gain equals treat-all, and gains are consistent with the cumulative curve
    curve = res.curves.query("estimator == 'AIPW'")
    assert curve.iloc[-1].gain == pytest.approx(tau.mean(), abs=1e-12)


def test_treat_nobody_value_is_zero_and_zero_budget_policy_is_nobody():
    assert bp.budget_policy(np.arange(10.0), 0.0).sum() == 0
    assert bp.policy_value(np.ones(10), np.zeros(10, dtype=int)) == 0.0


# ---------- propensity, indexing, input checks ----------


def test_wrong_propensity_biases_ipw_and_is_flagged():
    d = experiment(200_000, seed=15)
    n = len(d["y"])
    truth = d["tau"].mean()
    right = bp.incremental_scores(d["y"], d["t"], None, 0.85).mean()
    assert right == pytest.approx(truth, abs=0.004)
    wrong = bp.incremental_scores(d["y"], d["t"], None, 0.5).mean()
    assert abs(wrong - truth) > 0.02
    # with a correct outcome model, AIPW stays near the truth even under a wrong propensity
    aipw_wrong = bp.incremental_scores(d["y"], d["t"], holdout(true_mu(d), n), 0.5).mean()
    assert aipw_wrong == pytest.approx(truth, abs=0.004)
    with pytest.warns(UserWarning, match="treated share"):
        bp.evaluate_targeting(d["y"], d["t"], {"s": d["tau"]}, p_treat=0.5, n_boot=2)
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            bp.incremental_scores(d["y"], d["t"], None, bad)


def test_treatment_indexing_swapping_labels_negates_the_effect():
    d = experiment(50_000, seed=17)
    n = len(d["y"])
    mu = true_mu(d, 0.02)
    tau = bp.incremental_scores(d["y"], d["t"], holdout(mu, n), 0.85)
    swapped = bp.incremental_scores(d["y"], 1 - d["t"], holdout(mu[:, ::-1], n), 0.15)
    np.testing.assert_allclose(swapped, -tau, atol=1e-12)
    assert bp.incremental_scores(d["y"], d["t"]).mean() > 0  # treated visit more: tau > 0
    for bad in (d["t"] + 1, d["t"].astype(float), np.where(d["t"] == 1, 2, 0)):
        with pytest.raises(ValueError):
            bp.incremental_scores(d["y"], bad)
    ok = bp.incremental_scores(d["y"], d["t"].astype(np.uint8))  # Criteo stores uint8
    assert np.array_equal(ok, bp.incremental_scores(d["y"], d["t"]))


def test_input_validation():
    y, t = np.array([0.0, 1.0, 1.0]), np.array([0, 1, 1])
    with pytest.raises(ValueError):
        bp.incremental_scores(y[:2], t)
    with pytest.raises(ValueError):
        bp.incremental_scores(np.array([0.0, np.nan, 1.0]), t)
    with pytest.raises(ValueError):
        bp.evaluate_targeting(y, t, {})
    with pytest.raises(ValueError):
        bp.evaluate_targeting(y, t, {"s": np.zeros(2)})


# ---------- provenance and held-out safety ----------


def test_nuisance_must_be_out_of_sample_and_two_armed():
    d = experiment(2_000, seed=19)
    n = len(d["y"])
    mu = true_mu(d)
    with pytest.raises(ValueError, match="out of sample"):
        OutcomeModelPredictions.from_holdout(mu, np.arange(n), np.arange(100), n_arms=2)
    with pytest.raises(ValueError, match="shape"):
        OutcomeModelPredictions.from_holdout(mu, np.arange(n), np.arange(n, n + 5))  # 3-arm default
    three_arm = OutcomeModelPredictions.from_holdout(np.zeros((n, 3)), np.arange(n), [n + 1])
    with pytest.raises(ValueError, match="2-arm"):
        bp.incremental_scores(d["y"], d["t"], three_arm)
    with pytest.raises(TypeError):
        bp.incremental_scores(d["y"], d["t"], mu)
    # the three-arm Hillstrom evaluator rejects two-arm nuisance predictions
    with pytest.raises(ValueError, match="arms"):
        arm_scores(d["y"], d["t"], holdout(mu, n))
    with pytest.raises(ValueError, match="match the nuisance"):
        bp.evaluate_targeting(
            d["y"], d["t"], {"s": d["tau"]}, holdout(mu, n), row_ids=np.arange(n)[::-1], n_boot=2
        )


@pytest.mark.filterwarnings("ignore:observed treated share")
def test_protected_test_rows_are_refused_and_module_loads_no_data():
    d = experiment(2_000, seed=21)
    n = len(d["y"])
    ids = np.arange(n)
    test_ids = np.arange(n - 50, n + 500)
    with pytest.raises(ValueError, match="protected"):
        bp.evaluate_targeting(
            d["y"], d["t"], {"s": d["tau"]}, row_ids=ids, protected_ids=test_ids, n_boot=2
        )
    ok = bp.evaluate_targeting(
        d["y"][:1000],
        d["t"][:1000],
        {"s": d["tau"][:1000]},
        row_ids=ids[:1000],
        protected_ids=test_ids,
        n_boot=2,
    )
    assert ok.meta["n"] == 1000
    src = inspect.getsource(bp)
    for forbidden in ("criteo_prepare", "load_arrays", "np.load", "memmap", "test_idx", "open("):
        assert forbidden not in src


# ---------- bootstrap ----------


def test_results_are_reproducible_and_seed_sensitive():
    d = experiment(20_000, seed=23)
    kw = dict(y=d["y"], t=d["t"], scores={"a": d["tau"], "b": d["x"][:, 0]}, n_boot=30)
    r1, r2 = bp.evaluate_targeting(**kw), bp.evaluate_targeting(**kw)
    for name in ("policies", "references", "curves", "qini", "pairwise"):
        assert getattr(r1, name).equals(getattr(r2, name))
    r3 = bp.evaluate_targeting(**kw, seed=99)
    assert not r1.qini.ci_low.equals(r3.qini.ci_low)


def test_bootstrap_weights_are_shared_so_identical_policies_have_zero_difference():
    d = experiment(20_000, seed=25)
    res = bp.evaluate_targeting(
        d["y"], d["t"], {"a": d["x"][:, 0], "b": d["x"][:, 0].copy()}, n_boot=40
    )
    assert (res.pairwise["diff"] == 0).all()
    assert (res.pairwise.ci_low == 0).all() and (res.pairwise.ci_high == 0).all()


def test_paired_difference_interval_is_narrower_than_independent_intervals_imply():
    d = experiment(60_000, seed=27)
    n = len(d["y"])
    s_a = d["tau"]
    s_b = d["tau"] + np.random.default_rng(3).normal(0, 0.02, n)
    res = bp.evaluate_targeting(d["y"], d["t"], {"a": s_a, "b": s_b}, n_boot=100)
    p = res.pairwise.query("metric == 'value@30%'").iloc[0]
    pol = res.policies.query("policy == 'budget' and budget_requested == 0.3").set_index("model")
    indep = np.hypot(pol.loc["a", "se"], pol.loc["b", "se"]) * 1.96 * 2
    assert (p.ci_high - p.ci_low) < 0.7 * indep


def test_bootstrap_interval_shrinks_like_root_n_and_tracks_the_analytic_se():
    widths, ses = [], []
    for n in (20_000, 80_000):
        d = experiment(n, seed=29)
        res = bp.evaluate_targeting(d["y"], d["t"], {"s": d["tau"]}, n_boot=200)
        row = res.policies.query(
            "estimator == 'IPW' and policy == 'budget' and budget_requested == 0.5"
        ).iloc[0]
        widths.append(row.ci_high - row.ci_low)
        ses.append((row.se, (row.ci_high - row.ci_low) / (2 * 1.96)))
    assert 1.5 < widths[0] / widths[1] < 2.6
    for se, boot_se in ses:
        assert 0.8 < boot_se / se < 1.25


def test_memory_does_not_grow_with_bootstrap_replicates():
    d = experiment(300_000, seed=31)

    def peak(n_boot):
        tracemalloc.start()
        bp.evaluate_targeting(d["y"], d["t"], {"s": d["tau"]}, n_boot=n_boot)
        _, p = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return p

    p_small, p_big = peak(4), peak(40)
    n = len(d["y"])
    assert p_big < 1.15 * p_small
    assert p_big < 120 * n  # bytes per row: far below an n x n_boot matrix (40 x 8 = 320 B/row)
