"""Off-policy evaluation for a binary treatment (Criteo: 0 = control, 1 = treatment).

Same estimators as ``evaluation.policy`` with two arms and known propensities (treatment 0.85,
control 0.15 by the publisher's global subsampling; an assumption, see ``reports/criteo``):

    IPW:   tau_i = Y_i * (T_i / e - (1 - T_i) / (1 - e))
    AIPW:  tau_i = mu1_i - mu0_i + T_i / e * (Y_i - mu1_i) - (1 - T_i) / (1 - e) * (Y_i - mu0_i)

``tau_i`` is a per-customer unbiased score of the incremental effect, so the incremental value of
treating a set S relative to treating nobody is ``mean(1{i in S} * tau_i)``. Because the scores
are propensity-weighted, the cumulative-gain (Qini/AUUC-style) curve below is in units of
expected outcome gain per evaluated customer and is not distorted by the 85/15 allocation.

Policies rank customers by a predicted-uplift score and treat the top share (a contact budget).
Budget-constrained policies and random targeting at the same budget are feasible; "treat
everybody" is an unconstrained reference for any budget below 100%. Budgets are contact shares,
not monetary costs.

Uncertainty is a Poisson bootstrap: each replicate draws Poisson(1) weights per customer from
``(seed, replicate)`` only, so every policy and model evaluated with the same seed and n shares
identical weights (paired), and nothing of size n_rows x n_boot is ever allocated.

This module works on arrays only and never loads data. ``protected_ids`` lets a caller assert that
no held-out (test) row ids are being scored.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from upliftiq.evaluation.policy import OutcomeModelPredictions

P_TREAT = 0.85
SEED = 42
N_GRID = 100
DEFAULT_BUDGETS = (0.10, 0.20, 0.30, 0.50)
PROPENSITY_TOLERANCE = 0.01  # warn if the observed treated share differs from p_treat by more


def _check_inputs(y, t, p_treat) -> tuple[np.ndarray, np.ndarray]:
    y, t = np.asarray(y), np.asarray(t)
    if y.ndim != 1 or t.shape != y.shape:
        raise ValueError("y and t must be 1-D arrays of the same length")
    if len(y) == 0:
        raise ValueError("no observations")
    y = y.astype(np.float64)
    if not np.isfinite(y).all():
        raise ValueError("y contains non-finite values")
    if not np.issubdtype(t.dtype, np.integer) and t.dtype != bool:
        raise ValueError("t must hold integer codes 0 (control) and 1 (treatment)")
    t = t.astype(np.int8)
    if t.min() < 0 or t.max() > 1:
        raise ValueError("t must be 0 or 1")
    if not 0 < p_treat < 1:
        raise ValueError("p_treat must be in (0, 1)")
    return y, t


def incremental_scores(
    y, t, nuisance: OutcomeModelPredictions | None = None, p_treat: float = P_TREAT
) -> np.ndarray:
    """Per-customer score ``psi_i(1) - psi_i(0)``: AIPW if ``nuisance`` is given, else IPW."""
    y, t = _check_inputs(y, t, p_treat)
    treated, control = t == 1, t == 0
    if nuisance is None:
        return y * (treated / p_treat - control / (1 - p_treat))
    if not isinstance(nuisance, OutcomeModelPredictions):
        raise TypeError("nuisance must be OutcomeModelPredictions (with out-of-sample provenance)")
    if nuisance.n_arms != 2:
        raise ValueError(
            "binary evaluation needs 2-arm nuisance predictions (columns: control, treated)"
        )
    if len(nuisance.mu) != len(y):
        raise ValueError("nuisance predictions must align with y and t")
    mu0, mu1 = nuisance.mu[:, 0], nuisance.mu[:, 1]
    return (mu1 - mu0) + treated / p_treat * (y - mu1) - control / (1 - p_treat) * (y - mu0)


def _tie_priority(row_ids: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic pseudo-random tie-break key from the row id (splitmix64 finalizer).

    Ties are broken by this key, not by row position, because the Criteo file is in row order
    (single-treatment runs); position-based tie-breaking would favour early rows.
    """
    x = np.asarray(row_ids).astype(np.uint64) + np.uint64((seed * 0x9E3779B97F4A7C15) % 2**64)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return x


def rank_order(score, row_ids=None, seed: int = SEED) -> np.ndarray:
    """Row positions from highest to lowest predicted uplift, ties broken deterministically."""
    score = np.asarray(score, dtype=np.float64)
    if score.ndim != 1:
        raise ValueError("score must be 1-D")
    if not np.isfinite(score).all():
        raise ValueError("score contains non-finite values")
    ids = np.arange(len(score)) if row_ids is None else np.asarray(row_ids)
    if ids.shape != score.shape:
        raise ValueError("row_ids must align with score")
    return np.lexsort((_tie_priority(ids, seed), -score))


def n_treated_for_budget(budget: float, n: int) -> int:
    """Customers treated at a contact budget: floor(budget * n), with float-noise protection."""
    if not 0 <= budget <= 1:
        raise ValueError("budget must be in [0, 1]")
    return int(np.floor(budget * n + 1e-9))


def budget_policy(score, budget: float, row_ids=None, seed: int = SEED) -> np.ndarray:
    """0/1 treatment recommendation: treat the top ``budget`` share by predicted uplift."""
    order = rank_order(score, row_ids, seed)
    treat = np.zeros(len(order), dtype=np.int8)
    treat[order[: n_treated_for_budget(budget, len(order))]] = 1
    return treat


def threshold_policy(score, threshold: float = 0.0) -> np.ndarray:
    """0/1 recommendation: treat when predicted uplift exceeds ``threshold``."""
    score = np.asarray(score, dtype=np.float64)
    if not np.isfinite(score).all():
        raise ValueError("score contains non-finite values")
    return (score > threshold).astype(np.int8)


def policy_value(tau, treat) -> float:
    """Incremental value of a 0/1 policy relative to treating nobody (mean of ``treat * tau``)."""
    tau, treat = np.asarray(tau, dtype=np.float64), np.asarray(treat)
    if tau.shape != treat.shape:
        raise ValueError("tau and treat must align")
    if not np.isin(treat, (0, 1)).all():
        raise ValueError("treat must be 0 or 1")
    return float((tau * treat).mean())


@dataclass
class _Model:
    name: str
    edges: np.ndarray  # sorted unique counts of treated customers k (0..n)
    row_bin: np.ndarray  # per row: number of edges <= its rank; bin j holds ranks in [e_{j-1}, e_j)
    k_threshold: int  # customers with score > 0 (the "uplift > 0" policy)


def _prepare(name, score, ids, seed, extra_ks, n) -> _Model:
    order = rank_order(score, ids, seed)
    ks = {int(np.floor(g / N_GRID * n + 1e-9)) for g in range(1, N_GRID + 1)} | set(extra_ks)
    k_thr = int((np.asarray(score) > 0).sum())
    edges = np.array(sorted(ks | {k_thr}), dtype=np.int64)
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    return _Model(name, edges, np.searchsorted(edges, rank, side="right").astype(np.int32), k_thr)


def _vector(cum: np.ndarray, w_total: float, edges: np.ndarray, n: int, grid_pos) -> np.ndarray:
    """Statistics of one (estimator, model) in one replicate, concatenated.

    [gain at each edge | gain minus random line | gain minus treat-all | Qini coefficient]
    """
    gain = cum / w_total
    ate = gain[-1]  # last edge is k = n (treat everybody)
    f = edges / n
    g, fg = gain[grid_pos], f[grid_pos]
    area = float(np.sum(0.5 * (g[1:] + g[:-1]) * np.diff(fg)) + 0.5 * g[0] * fg[0])
    return np.concatenate([gain, gain - f * ate, gain - ate, [area - 0.5 * ate]])


def _pct(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return lo, hi


@dataclass
class TargetingResult:
    policies: pd.DataFrame  # candidate policies (budget and uplift>0) with paired CIs
    references: pd.DataFrame  # treat nobody / treat everybody / random at each budget
    curves: pd.DataFrame  # cumulative gain and uplift-per-treated on a 1% grid
    qini: pd.DataFrame  # Qini coefficient per model and estimator
    pairwise: pd.DataFrame  # each model minus the first model, paired
    meta: dict


def evaluate_targeting(
    y,
    t,
    scores: dict[str, np.ndarray],
    nuisance: OutcomeModelPredictions | None = None,
    p_treat: float = P_TREAT,
    budgets=DEFAULT_BUDGETS,
    n_boot: int = 200,
    seed: int = SEED,
    row_ids=None,
    protected_ids=None,
) -> TargetingResult:
    """Evaluate uplift-score rankings as contact-budget policies.

    ``scores`` maps model name to predicted uplift (higher = treat first). The estimator is AIPW
    when ``nuisance`` is given (IPW is then reported alongside as a cross-check), otherwise IPW.
    """
    if not scores:
        raise ValueError("scores is empty")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    for b in budgets:
        n_treated_for_budget(b, 1)
    n = len(np.asarray(y))
    ids = nuisance.row_ids if (row_ids is None and nuisance is not None) else row_ids
    if ids is not None:
        ids = np.asarray(ids)
        if ids.shape != (n,):
            raise ValueError("row_ids must align with y")
        if nuisance is not None and not np.array_equal(ids, nuisance.row_ids):
            raise ValueError("row_ids must match the nuisance predictions' row ids")
        if protected_ids is not None and np.intersect1d(ids, protected_ids).size:
            raise ValueError("row_ids include protected (held-out) rows; refusing to score them")
    y, t = _check_inputs(y, t, p_treat)
    for name, s in scores.items():
        if np.asarray(s).shape != (n,):
            raise ValueError(f"score '{name}' must align with y")

    share = float(t.mean())
    if abs(share - p_treat) > PROPENSITY_TOLERANCE:
        warnings.warn(
            f"observed treated share {share:.4f} differs from p_treat={p_treat}; "
            "the propensity assumption may be wrong",
            stacklevel=2,
        )

    taus = {"IPW": incremental_scores(y, t, None, p_treat)}
    if nuisance is not None:
        taus = {"AIPW": incremental_scores(y, t, nuisance, p_treat), **taus}
    budget_ks = {b: n_treated_for_budget(b, n) for b in budgets}
    models = [_prepare(m, s, ids, seed, budget_ks.values(), n) for m, s in scores.items()]
    layout = [(est, m) for est in taus for m in models]
    grid_k = [int(np.floor(g / N_GRID * n + 1e-9)) for g in range(1, N_GRID + 1)]
    grid_pos = {id(m): np.searchsorted(m.edges, grid_k) for m in models}

    def replicate(w: np.ndarray) -> np.ndarray:
        w_total = float(w.sum())
        parts = []
        for tau in taus.values():
            wt = w * tau
            for m in models:
                cum = np.cumsum(np.bincount(m.row_bin, weights=wt, minlength=len(m.edges)))
                parts.append(_vector(cum, w_total, m.edges, n, grid_pos[id(m)]))
        return np.concatenate(parts)

    point = replicate(np.ones(n))
    draws = np.empty((n_boot, len(point)))
    for b in range(n_boot):
        draws[b] = replicate(np.random.default_rng([seed, b]).poisson(1.0, n).astype(np.float64))
    lo, hi = _pct(draws)

    sizes = [3 * len(m.edges) + 1 for _est, m in layout]
    starts = np.concatenate([[0], np.cumsum(sizes)])
    pol_rows, ref_rows, curve_rows, qini_rows, pair_rows = [], [], [], [], []
    base = {}
    for (est, m), s0 in zip(layout, starts[:-1], strict=True):
        e = len(m.edges)
        blocks = {
            key: slice(s0 + j * e, s0 + (j + 1) * e)
            for j, key in enumerate(("gain", "rand", "all"))
        }
        qi = s0 + 3 * e
        tau = taus[est]
        z1 = np.bincount(m.row_bin, weights=tau, minlength=e)
        z2 = np.bincount(m.row_bin, weights=tau**2, minlength=e)
        cum1, cum2 = np.cumsum(z1), np.cumsum(z2)
        se = (
            np.sqrt(np.maximum(cum2 / n - (cum1 / n) ** 2, 0) / (n - 1)) if n > 1 else np.nan * cum1
        )
        ate = point[blocks["gain"]][-1]
        kinds = [("budget", b, k) for b, k in budget_ks.items()] + [
            ("uplift_gt_0", None, m.k_threshold)
        ]
        for kind, b, k in kinds:
            j = int(np.searchsorted(m.edges, k))
            row = {
                "model": m.name,
                "estimator": est,
                "policy": kind,
                "budget_requested": b,
                "budget": k / n,
                "n_treated": k,
                "value": point[blocks["gain"]][j],
                "se": float(se[j]),
                "ci_low": lo[blocks["gain"]][j],
                "ci_high": hi[blocks["gain"]][j],
                "random_value": k / n * ate,
                "diff_vs_random": point[blocks["rand"]][j],
                "diff_vs_random_ci_low": lo[blocks["rand"]][j],
                "diff_vs_random_ci_high": hi[blocks["rand"]][j],
                "treat_all_value": ate,
                "diff_vs_treat_all": point[blocks["all"]][j],
                "diff_vs_treat_all_ci_low": lo[blocks["all"]][j],
                "diff_vs_treat_all_ci_high": hi[blocks["all"]][j],
                "feasibility": "feasible (budget-constrained)",
            }
            pol_rows.append(row)
        for gi, k in enumerate(grid_k):
            j = grid_pos[id(m)][gi]
            f = k / n
            curve_rows.append(
                {
                    "model": m.name,
                    "estimator": est,
                    "budget": f,
                    "gain": point[blocks["gain"]][j],
                    "gain_ci_low": lo[blocks["gain"]][j],
                    "gain_ci_high": hi[blocks["gain"]][j],
                    "random_line": f * ate,
                    "uplift_per_treated": point[blocks["gain"]][j] / f if f else np.nan,
                }
            )
        qini_rows.append(
            {
                "model": m.name,
                "estimator": est,
                "qini": point[qi],
                "ci_low": lo[qi],
                "ci_high": hi[qi],
                "ate_treat_all": ate,
                "note": f"{N_GRID}-point trapezoid, area between gain curve and random line",
            }
        )
        base.setdefault(est, (m, s0, blocks, qi))
        ref_m, ref_s0, ref_blocks, ref_qi = base[est]
        if m is not ref_m:
            for b, k in budget_ks.items():
                j, jr = int(np.searchsorted(m.edges, k)), int(np.searchsorted(ref_m.edges, k))
                d = draws[:, blocks["gain"]][:, j] - draws[:, ref_blocks["gain"]][:, jr]
                pair_rows.append(
                    {
                        "model": m.name,
                        "reference_model": ref_m.name,
                        "estimator": est,
                        "metric": f"value@{b:.0%}",
                        "diff": point[blocks["gain"]][j] - point[ref_blocks["gain"]][jr],
                        "ci_low": np.percentile(d, 2.5),
                        "ci_high": np.percentile(d, 97.5),
                    }
                )
            d = draws[:, qi] - draws[:, ref_qi]
            pair_rows.append(
                {
                    "model": m.name,
                    "reference_model": ref_m.name,
                    "estimator": est,
                    "metric": "qini",
                    "diff": point[qi] - point[ref_qi],
                    "ci_low": np.percentile(d, 2.5),
                    "ci_high": np.percentile(d, 97.5),
                }
            )

    # references are model-independent: take them from the first model's blocks
    for est in taus:
        m, s0, blocks, _ = base[est]
        gain = blocks["gain"]
        ate_idx = len(m.edges) - 1
        ref_rows.append(
            {
                "policy": "treat_nobody",
                "estimator": est,
                "budget": 0.0,
                "value": 0.0,
                "ci_low": 0.0,
                "ci_high": 0.0,
                "feasibility": "reference (budget 0)",
            }
        )
        ref_rows.append(
            {
                "policy": "treat_all",
                "estimator": est,
                "budget": 1.0,
                "value": point[gain][ate_idx],
                "ci_low": lo[gain][ate_idx],
                "ci_high": hi[gain][ate_idx],
                "feasibility": "unconstrained reference (feasible only at budget 100%)",
            }
        )
        draws_ate = draws[:, gain][:, ate_idx]
        for b, k in budget_ks.items():
            f = k / n
            ref_rows.append(
                {
                    "policy": f"random@{b:.0%}",
                    "estimator": est,
                    "budget": f,
                    "value": f * point[gain][ate_idx],
                    "ci_low": f * np.percentile(draws_ate, 2.5),
                    "ci_high": f * np.percentile(draws_ate, 97.5),
                    "feasibility": "feasible (budget-constrained, expectation over random draws)",
                }
            )
    meta = {
        "n": n,
        "n_boot": n_boot,
        "seed": seed,
        "p_treat": p_treat,
        "observed_treated_share": share,
        "estimators": list(taus),
        "bootstrap": "Poisson(1) customer weights from (seed, replicate); paired across policies",
        "tie_break": "seeded hash of row id (deterministic, independent of row order)",
    }
    return TargetingResult(
        pd.DataFrame(pol_rows),
        pd.DataFrame(ref_rows),
        pd.DataFrame(curve_rows),
        pd.DataFrame(qini_rows),
        pd.DataFrame(pair_rows),
        meta,
    )
