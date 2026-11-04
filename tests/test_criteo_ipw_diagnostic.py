"""IPW vs AIPW diagnostic helpers on synthetic data with and without arm imbalance."""

from __future__ import annotations

import numpy as np
import pytest

from upliftiq.evaluation import criteo_ipw_diagnostic as dx


def world(n=200_000, seed=0, imbalance=0.0):
    """Skewed baseline (like Criteo visits); effect rises with it; optional arm imbalance."""
    r = np.random.default_rng(seed)
    z = r.normal(size=n)
    hi = 1 / (1 + np.exp(-3 * (z - 1.3)))  # smooth: a minority with a high baseline
    p0 = 0.01 + 0.30 * hi
    tau = 0.005 + 0.09 * hi
    e = np.clip(0.85 + imbalance * hi, 0.01, 0.99)
    t = (r.random(n) < e).astype(np.int8)
    y = (r.random(n) < p0 + t * tau).astype(float)
    mu0, mu1 = p0, p0 + tau  # a perfect, independent outcome model
    return y, t, mu0, mu1, tau.mean()


def test_independent_formulas_match_library_and_decomposition_is_exact():
    y, t, mu0, mu1, _ = world(seed=1, imbalance=0.05)
    res = dx.run_diagnostic(y, t, mu0, mu1, n_boot=200, n_perm=500)
    chk = res["implementation_check"]
    assert chk["ipw_max_abs_difference"] < 1e-12 and chk["aipw_max_abs_difference"] < 1e-12
    assert abs(chk["decomposition_residual"]) < 1e-12
    assert abs(chk["gcomp_plus_augmentations_residual"]) < 1e-12
    y2, t2, m0, m1, _ = world(seed=5)  # randomized: weights sum to ~1, so Hajek ~ Horvitz-Thompson
    clean = dx.independent_estimates(y2, t2, m0, m1)
    assert clean["ipw_hajek"] == pytest.approx(clean["ipw"], abs=2e-3)
    assert clean["weight_sum_treated"] == pytest.approx(1.0, abs=0.01)


def test_randomized_data_shows_no_gap():
    y, t, mu0, mu1, ate = world(seed=2)
    res = dx.run_diagnostic(y, t, mu0, mu1, n_boot=200, n_perm=500)
    est, perm = res["estimates"], res["permutation_test"]
    assert abs(perm["z"]) < 3.5 and perm["p_two_sided"] > 0.001
    iv = {r["quantity"]: r for r in res["paired_intervals"]}
    d = iv["ipw_minus_aipw"]
    assert d["ci_low"] < 0 < d["ci_high"]
    assert est["aipw"] == pytest.approx(ate, abs=0.006) and est["ipw"] == pytest.approx(
        ate, abs=0.008
    )


def test_imbalanced_arms_are_flagged_and_aipw_and_poststratification_stay_near_truth():
    y, t, mu0, mu1, ate = world(
        seed=3, imbalance=0.08
    )  # high-baseline customers treated more often
    res = dx.run_diagnostic(y, t, mu0, mu1, n_boot=200, n_perm=500)
    est, perm = res["estimates"], res["permutation_test"]
    assert perm["z"] > 6 and perm["p_two_sided"] < 0.01
    assert est["ipw"] - ate > 0.004  # unadjusted contrast is inflated
    assert est["aipw"] == pytest.approx(ate, abs=0.006)
    assert res["poststratified_on_predicted_baseline"]["estimate"] == pytest.approx(ate, abs=0.01)
    iv = {r["quantity"]: r for r in res["paired_intervals"]}
    assert iv["ipw_minus_aipw"]["ci_low"] > 0
    # the control-arm term (lower baseline among controls) carries the gap
    assert est["term_control_arm"] < 0 and est["ipw_minus_aipw"] > 0
    shares = [s["treated_share"] for s in res["strata"]]
    assert shares[-1] > shares[0]


def test_report_states_no_bug_and_names_the_mechanism():
    y, t, mu0, mu1, _ = world(n=60_000, seed=4, imbalance=0.08)
    res = dx.run_diagnostic(y, t, mu0, mu1, n_boot=200, n_perm=500)
    text = dx.build_report(res, {"n": len(y), "e": 0.85, "nuisance_n_train": 1})
    assert "No implementation error found" in text and "predicted baseline" in text
    assert "propensity-free" in text
