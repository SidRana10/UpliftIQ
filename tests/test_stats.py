import numpy as np
import pytest
from scipy import stats

from upliftiq.evaluation.stats import (
    holm_adjust,
    mean_effect,
    standardized_mean_difference,
    two_proportion_effect,
    wilson_interval,
)


def test_wilson_matches_known_value():
    lo, hi = wilson_interval(10, 100)
    assert lo == pytest.approx(0.0552, abs=1e-3)
    assert hi == pytest.approx(0.1744, abs=1e-3)


def test_wilson_edges_stay_in_unit_interval():
    assert wilson_interval(0, 50)[0] == 0.0
    assert wilson_interval(50, 50)[1] == 1.0
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
    with pytest.raises(ValueError):
        wilson_interval(5, 4)


def test_two_proportion_matches_chi_square():
    r = two_proportion_effect(180, 1000, 120, 1000)
    chi2_p = stats.chi2_contingency([[180, 820], [120, 880]], correction=False)[1]
    assert r.abs_effect == pytest.approx(0.06)
    assert r.rel_effect == pytest.approx(0.5)
    assert r.p_value == pytest.approx(chi2_p, rel=1e-6)
    assert r.abs_ci[0] < 0.06 < r.abs_ci[1]
    assert r.rel_ci[0] < 0.5 < r.rel_ci[1]


def test_two_proportion_no_difference():
    r = two_proportion_effect(50, 1000, 50, 1000)
    assert r.abs_effect == 0
    assert r.p_value == pytest.approx(1.0)
    assert r.abs_ci[0] < 0 < r.abs_ci[1]


def test_two_proportion_zero_control_events_gives_nan_relative():
    r = two_proportion_effect(5, 100, 0, 100)
    assert np.isnan(r.rel_effect)
    assert np.isfinite(r.abs_ci[0]) and np.isfinite(r.abs_ci[1])


def test_two_proportion_all_zero_is_p_one():
    assert two_proportion_effect(0, 100, 0, 100).p_value == 1.0


def test_mean_effect_detects_real_difference_and_is_reproducible():
    rng = np.random.default_rng(0)
    control = np.where(rng.random(2000) < 0.02, rng.exponential(100, 2000), 0.0)
    treated = np.where(rng.random(2000) < 0.06, rng.exponential(100, 2000), 0.0)
    a = mean_effect(treated, control, n_boot=500, n_perm=500, seed=1)
    b = mean_effect(treated, control, n_boot=500, n_perm=500, seed=1)
    assert a == b
    assert a.abs_effect == pytest.approx(treated.mean() - control.mean())
    assert a.abs_ci[0] < a.abs_effect < a.abs_ci[1]
    assert a.p_value < 0.01


def test_mean_effect_null_has_large_p_and_ci_covers_zero():
    rng = np.random.default_rng(3)
    x = np.where(rng.random(1500) < 0.05, rng.exponential(50, 1500), 0.0)
    y = np.where(rng.random(1500) < 0.05, rng.exponential(50, 1500), 0.0)
    r = mean_effect(x, y, n_boot=500, n_perm=500, seed=2)
    assert r.p_value > 0.05
    assert r.abs_ci[0] < 0 < r.abs_ci[1]


def test_mean_effect_all_zero_control_and_empty_input():
    r = mean_effect(np.array([0.0, 10.0, 0.0]), np.zeros(3), n_boot=50, n_perm=50)
    assert np.isnan(r.rel_effect)
    with pytest.raises(ValueError):
        mean_effect(np.array([]), np.array([1.0]))


def test_holm_adjust():
    adj = holm_adjust([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06])
    np.testing.assert_allclose(holm_adjust([0.5]), [0.5])
    assert holm_adjust([0.4, 0.6]).max() <= 1.0
    assert (holm_adjust([0.01, 0.2, 0.9]) >= np.array([0.01, 0.2, 0.9])).all()


def test_smd():
    a = np.array([1.0, 2.0, 3.0])
    assert standardized_mean_difference(a, a) == 0
    assert standardized_mean_difference(a + 1, a) == pytest.approx(1.0)
    assert standardized_mean_difference(np.ones(3), np.ones(3)) == 0.0
    assert standardized_mean_difference(np.ones(3), np.zeros(3)) == float("inf")
