"""Small, dependency-light statistical helpers for randomized two-arm comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

Z95 = float(stats.norm.ppf(0.975))


@dataclass(frozen=True)
class EffectResult:
    """Treatment-vs-control effect on one outcome."""

    control_estimate: float
    treated_estimate: float
    abs_effect: float
    abs_ci: tuple[float, float]
    rel_effect: float  # treated / control - 1 (nan if control estimate is 0)
    rel_ci: tuple[float, float]
    p_value: float
    method: str


def wilson_interval(x: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= x <= n:
        raise ValueError("x must be between 0 and n")
    p = x / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    lower = 0.0 if x == 0 else max(0.0, centre - half)
    upper = 1.0 if x == n else min(1.0, centre + half)
    return float(lower), float(upper)


def two_proportion_effect(x1: int, n1: int, x0: int, n0: int) -> EffectResult:
    """Treated (x1/n1) vs control (x0/n0) for a binary outcome.

    - Absolute difference: Newcombe hybrid-score CI (good for rare outcomes).
    - P-value: pooled two-sided z-test (equivalent to the chi-square test without correction).
    - Relative lift (p1/p0 - 1): log risk-ratio (Katz) CI.
    """
    l1, u1 = wilson_interval(x1, n1)
    l0, u0 = wilson_interval(x0, n0)
    p1, p0 = x1 / n1, x0 / n0
    diff = p1 - p0
    abs_ci = (
        float(diff - np.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2)),
        float(diff + np.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)),
    )

    pooled = (x1 + x0) / (n1 + n0)
    se0 = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n0))
    p_value = 1.0 if se0 == 0 else float(2 * stats.norm.sf(abs(diff) / se0))

    if x0 == 0 or x1 == 0:
        rel, rel_ci = (float("nan"), (float("nan"), float("nan")))
    else:
        log_rr = np.log(p1 / p0)
        se = np.sqrt(1 / x1 - 1 / n1 + 1 / x0 - 1 / n0)
        rel = float(np.exp(log_rr) - 1)
        rel_ci = (float(np.exp(log_rr - Z95 * se) - 1), float(np.exp(log_rr + Z95 * se) - 1))
    return EffectResult(p0, p1, float(diff), abs_ci, rel, rel_ci, p_value, "z-test / Newcombe")


def mean_effect(
    treated: np.ndarray,
    control: np.ndarray,
    n_boot: int = 5000,
    n_perm: int = 20000,
    seed: int = 42,
) -> EffectResult:
    """Treated vs control mean for a skewed, zero-heavy outcome (e.g. spend).

    No normality assumption is relied on:
    - CIs (absolute and relative): percentile bootstrap, resampling each arm independently.
    - P-value: two-sided permutation test on the difference in means (Welch's t-test
      p-value is a sanity check done by the caller if desired).
    """
    treated = np.asarray(treated, dtype=float)
    control = np.asarray(control, dtype=float)
    if treated.size == 0 or control.size == 0:
        raise ValueError("both arms need at least one observation")
    rng = np.random.default_rng(seed)
    m1, m0 = treated.mean(), control.mean()
    diff = m1 - m0

    boot1 = _boot_means(treated, n_boot, rng)
    boot0 = _boot_means(control, n_boot, rng)
    d = boot1 - boot0
    abs_ci = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
    if m0 == 0:
        rel, rel_ci = float("nan"), (float("nan"), float("nan"))
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = boot1 / boot0 - 1
        r = r[np.isfinite(r)]
        rel = float(m1 / m0 - 1)
        rel_ci = (float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)))

    pooled = np.concatenate([treated, control])
    n1 = treated.size
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        if abs(perm[:n1].mean() - perm[n1:].mean()) >= abs(diff) - 1e-12:
            count += 1
    p_value = (count + 1) / (n_perm + 1)
    return EffectResult(
        float(m0),
        float(m1),
        float(diff),
        abs_ci,
        rel,
        rel_ci,
        p_value,
        "bootstrap CI / permutation test",
    )


def _boot_means(x: np.ndarray, n_boot: int, rng: np.random.Generator, chunk: int = 250):
    out = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        k = min(chunk, n_boot - start)
        out[start : start + k] = x[rng.integers(0, x.size, size=(k, x.size))].mean(axis=1)
    return out


def holm_adjust(p_values: list[float] | np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (controls family-wise error rate)."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    """(mean(a) - mean(b)) / pooled SD. Returns 0 if both arms are constant and equal."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if sd == 0:
        return 0.0 if a.mean() == b.mean() else float("inf")
    return float((a.mean() - b.mean()) / sd)
