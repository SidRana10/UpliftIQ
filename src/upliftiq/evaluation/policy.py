"""Off-policy evaluation for the randomized three-arm experiment (IPW and AIPW).

Arms are integer codes (0 = No E-Mail, 1 = Mens, 2 = Womens), see ``modeling.features.ARMS``.
Because assignment was randomized with known probabilities (1/3 each), the value of any policy
``pi`` can be estimated without a propensity model. Per-customer scores for "give customer i arm a":

    IPW:   psi_i(a) = 1{A_i = a} / e_a * Y_i
    AIPW:  psi_i(a) = mu_i(a) + 1{A_i = a} / e_a * (Y_i - mu_i(a))

A policy's score is ``psi_i(pi_i)`` (or a probability-weighted mix for stochastic policies) and its
value is the mean score. IPW is unbiased by randomization alone. AIPW stays unbiased for any ``mu``
that is not fitted on the row being scored; ``OutcomeModelPredictions`` enforces that provenance.

This module works on arrays only. It never loads the dataset, so it cannot touch the test set by
itself; callers decide which rows (dev out-of-fold rows, or test once at Stage 5) to pass in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from upliftiq.modeling.features import PROPENSITY

N_ARMS = 3
BASELINE_NAMES = ["no_email", "mens_for_all", "womens_for_all", "random"]


@dataclass(frozen=True)
class OutcomeModelPredictions:
    """Outcome-model predictions ``mu[i, a] = E[Y | X_i, arm=a]`` with out-of-sample provenance.

    ``group[i]`` identifies the fitted model that produced row i's prediction, and
    ``trained_on[g]`` lists the row ids that model was trained on. Construction fails if any row
    was predicted by a model trained on that same row (in-sample prediction).
    Use ``from_holdout`` for one model, or pass several groups for cross-fitting.
    """

    mu: np.ndarray
    row_ids: np.ndarray
    group: np.ndarray
    trained_on: dict[int, np.ndarray]

    def __post_init__(self) -> None:
        mu = np.asarray(self.mu, dtype=float)
        n = len(mu)
        if mu.ndim != 2 or mu.shape[1] != N_ARMS:
            raise ValueError(f"mu must have shape (n, {N_ARMS})")
        if not np.isfinite(mu).all():
            raise ValueError("mu contains non-finite values")
        if len(self.row_ids) != n or len(self.group) != n:
            raise ValueError("mu, row_ids and group must have the same length")
        for g in np.unique(self.group):
            if g not in self.trained_on:
                raise ValueError(f"no training rows given for model group {g}")
            if np.intersect1d(self.row_ids[self.group == g], self.trained_on[g]).size:
                raise ValueError(
                    f"model group {g} was trained on rows it predicts: AIPW outcome "
                    "predictions must be out of sample"
                )

    @classmethod
    def from_holdout(cls, mu, row_ids, train_ids) -> OutcomeModelPredictions:
        """One model trained on ``train_ids`` and applied to the (disjoint) ``row_ids``."""
        row_ids = np.asarray(row_ids)
        return cls(
            np.asarray(mu, dtype=float),
            row_ids,
            np.zeros(len(row_ids), dtype=int),
            {0: np.asarray(train_ids)},
        )


def _propensity_vector(propensity) -> np.ndarray:
    e = np.broadcast_to(np.asarray(propensity, dtype=float), (N_ARMS,)).copy()
    if not ((e > 0) & (e <= 1)).all():
        raise ValueError("propensities must be in (0, 1]")
    return e


def _check_data(y, arm) -> tuple[np.ndarray, np.ndarray]:
    y, arm = np.asarray(y, dtype=float), np.asarray(arm)
    if y.ndim != 1 or arm.shape != y.shape:
        raise ValueError("y and arm must be 1-D arrays of the same length")
    if len(y) == 0:
        raise ValueError("no observations")
    if not np.isfinite(y).all():
        raise ValueError("y contains non-finite values")
    if not np.issubdtype(arm.dtype, np.integer) or arm.min() < 0 or arm.max() >= N_ARMS:
        raise ValueError(f"arm must be integer codes in 0..{N_ARMS - 1}")
    return y, arm


def arm_scores(
    y, arm, nuisance: OutcomeModelPredictions | None = None, propensity=PROPENSITY
) -> np.ndarray:
    """(n, 3) per-customer scores ``psi_i(a)``: AIPW if ``nuisance`` is given, else IPW."""
    y, arm = _check_data(y, arm)
    e = _propensity_vector(propensity)
    assigned = arm[:, None] == np.arange(N_ARMS)[None, :]
    if nuisance is None:
        return assigned / e * y[:, None]
    if not isinstance(nuisance, OutcomeModelPredictions):
        raise TypeError("nuisance must be OutcomeModelPredictions (with out-of-sample provenance)")
    if len(nuisance.mu) != len(y):
        raise ValueError("nuisance predictions must align with y and arm")
    mu = nuisance.mu
    return mu + assigned / e * (y[:, None] - mu)


def policy_scores(scores: np.ndarray, policy) -> np.ndarray:
    """Per-customer scores of a policy given the (n, 3) ``arm_scores`` matrix.

    ``policy`` is either an integer array of recommended arms, shape (n,), or a probability
    matrix, shape (n, 3), for stochastic policies (rows sum to 1).
    """
    policy = np.asarray(policy)
    n = len(scores)
    if policy.ndim == 1:
        if policy.shape != (n,) or not np.issubdtype(policy.dtype, np.integer):
            raise ValueError("a deterministic policy must be an integer array of length n")
        if policy.min() < 0 or policy.max() >= N_ARMS:
            raise ValueError(f"policy arm codes must be in 0..{N_ARMS - 1}")
        return scores[np.arange(n), policy]
    if policy.shape != (n, N_ARMS):
        raise ValueError(f"a stochastic policy must have shape (n, {N_ARMS})")
    if (policy < 0).any() or not np.allclose(policy.sum(axis=1), 1.0):
        raise ValueError("policy probabilities must be non-negative and sum to 1 per row")
    return (scores * policy).sum(axis=1)


def baseline_policies(n: int) -> dict[str, np.ndarray]:
    """The four reference policies. ``random`` is uniform over arms (exact expectation)."""
    return {
        "no_email": np.zeros(n, dtype=int),
        "mens_for_all": np.ones(n, dtype=int),
        "womens_for_all": np.full(n, 2, dtype=int),
        "random": np.full((n, N_ARMS), 1 / N_ARMS),
    }


@dataclass(frozen=True)
class ValueEstimate:
    value: float
    se: float
    n: int


def estimate_policy_value(
    y, arm, policy, nuisance: OutcomeModelPredictions | None = None, propensity=PROPENSITY
) -> ValueEstimate:
    """Point estimate and analytic standard error of one policy's value."""
    psi = policy_scores(arm_scores(y, arm, nuisance, propensity), policy)
    se = float(psi.std(ddof=1) / np.sqrt(len(psi))) if len(psi) > 1 else float("nan")
    return ValueEstimate(float(psi.mean()), se, len(psi))


def bootstrap_means(
    psi: np.ndarray, n_boot: int = 2000, seed: int = 42, chunk: int = 50
) -> np.ndarray:
    """Customer-level bootstrap of column means of ``psi`` (n, k) -> (n_boot, k).

    All columns share the same resampled customers, so differences between columns are paired.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    psi = np.asarray(psi, dtype=float)
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, psi.shape[1]))
    for start in range(0, n_boot, chunk):
        k = min(chunk, n_boot - start)
        idx = rng.integers(0, len(psi), size=(k, len(psi)))
        out[start : start + k] = psi[idx].mean(axis=1)
    return out


def evaluate_policy(
    y,
    arm,
    policy,
    nuisance: OutcomeModelPredictions | None = None,
    propensity=PROPENSITY,
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Candidate policy vs the four baselines, with paired bootstrap 95% CIs.

    One row per policy (``candidate`` first). ``value``/``se`` are the estimator and its analytic
    standard error; ``ci_*`` is the percentile bootstrap interval; ``diff`` is
    candidate minus that policy, with its paired bootstrap interval (NaN on the candidate row).
    """
    scores = arm_scores(y, arm, nuisance, propensity)
    policies = {"candidate": policy, **baseline_policies(len(scores))}
    psi = np.column_stack([policy_scores(scores, p) for p in policies.values()])
    draws = bootstrap_means(psi, n_boot, seed)
    n = len(psi)
    se = psi.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(psi.shape[1], np.nan)
    diff = psi[:, [0]].mean(axis=0) - psi.mean(axis=0)
    diff_draws = draws[:, [0]] - draws

    rows = []
    for j, name in enumerate(policies):
        is_cand = j == 0
        rows.append(
            {
                "policy": name,
                "estimator": "IPW" if nuisance is None else "AIPW",
                "value": psi[:, j].mean(),
                "se": se[j],
                "ci_low": np.percentile(draws[:, j], 2.5),
                "ci_high": np.percentile(draws[:, j], 97.5),
                "diff": np.nan if is_cand else diff[j],
                "diff_ci_low": np.nan if is_cand else np.percentile(diff_draws[:, j], 2.5),
                "diff_ci_high": np.nan if is_cand else np.percentile(diff_draws[:, j], 97.5),
            }
        )
    return pd.DataFrame(rows)
