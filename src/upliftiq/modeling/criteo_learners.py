"""Uplift learners for the binary-treatment Criteo experiment (0 = control, 1 = treatment).

Same interface as ``modeling.learners`` (Hillstrom, untouched), specialised to two arms:

    model = make_model("t_learner", seed=42).fit(X_train, treatment_train, visit_train)
    model.predict_uplift(X)   # (n,) predicted effect of treatment vs control, P(Y|x,1) - P(Y|x,0)
    model.recommend(X)        # (n,) 0/1: treat when predicted uplift > 0
    model.recommend(X, budget=0.2, row_ids=ids)   # treat the top 20% (C2 budget policy)
    model.predict_outcomes(X) # (n, 2): P(Y=1 | x, t) for t = 0, 1 -- outcome-model learners only

``X`` is a float array of shape (n, 12) holding f0-f11, or a DataFrame from which only f0-f11 are
read (treatment, exposure and outcome columns are never features; the treatment enters only as the
indicator the learner architecture requires). Each model fits its own preprocessing on the rows it
is given (train-only, like the C0 scaler), early-stops on a split of those same rows, and uses the
publisher-supported propensity (treatment 0.85) where the DR learner needs one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

from upliftiq.data.criteo import FEATURES, select_features
from upliftiq.data.criteo_prepare import FeatureScaler, fit_scaler
from upliftiq.evaluation.binary_policy import P_TREAT, budget_policy, incremental_scores
from upliftiq.evaluation.policy import OutcomeModelPredictions

SEED = 42
N_FEATURES = len(FEATURES)

# Features are anonymized dense floats: no categorical handling or encoding is needed.
LGBM_DEFAULTS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 500,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
}
# The DR pseudo-outcome is extremely noisy (weights up to 1/0.15), so it is regularized harder.
DR_REGRESSOR_DEFAULTS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 2000,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
}


class _ConstantProb:
    """Fallback when a training subset contains a single outcome class."""

    def __init__(self, p: float):
        self.p = p

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)])


def _lgbm_kwargs(defaults: dict, overrides: dict | None, seed: int, n_jobs: int) -> dict:
    return {
        **defaults,
        **(overrides or {}),
        "random_state": seed,
        "deterministic": True,
        "force_row_wise": True,
        "n_jobs": n_jobs,
        "verbose": -1,
    }


def _fit_lgbm(model, X, y, es_rounds, val_fraction, seed, strata=None):
    """Fit a LightGBM estimator, early-stopping on a held-out part of the *training* rows."""
    if es_rounds is None:
        return model.fit(X, y)
    tr, va = train_test_split(
        np.arange(len(y)), test_size=val_fraction, random_state=seed, stratify=strata
    )
    return model.fit(
        X[tr],
        y[tr],
        eval_X=X[va],
        eval_y=y[va],
        callbacks=[early_stopping(es_rounds, verbose=False)],
    )


def _fit_classifier(X, y, params, es_rounds, val_fraction, seed):
    if len(np.unique(y)) < 2:
        return _ConstantProb(float(y.mean()))
    return _fit_lgbm(LGBMClassifier(**params), X, y, es_rounds, val_fraction, seed, strata=y)


def _proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def _features(X) -> np.ndarray:
    """f0-f11 as a float32 array; extra DataFrame columns are ignored, never used."""
    if isinstance(X, pd.DataFrame):
        X = select_features(X).to_numpy(dtype=np.float32)
    else:
        X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != N_FEATURES:
        raise ValueError(f"X must have {N_FEATURES} feature columns (f0-f11)")
    if not np.isfinite(X).all():
        raise ValueError("X contains non-finite values")
    return X


class UpliftModel:
    """Base class: input validation and the shared prediction API."""

    name = "base"

    def __init__(self, seed: int = SEED, params: dict | None = None, n_jobs: int = 4):
        self.seed = seed
        self.params = params
        self.n_jobs = n_jobs
        self.is_fitted_ = False

    def fit(self, X, treatment, y) -> UpliftModel:
        X = _features(X)
        t, y = np.asarray(treatment), np.asarray(y, dtype=float)
        if len(X) != len(t) or len(X) != len(y):
            raise ValueError("X, treatment and y must have the same length")
        if (
            not (np.issubdtype(t.dtype, np.integer) or t.dtype == bool)
            or not np.isin(t, (0, 1)).all()
        ):
            raise ValueError("treatment must be integer codes 0 (control) and 1 (treatment)")
        t = t.astype(np.int8)
        if t.min() == t.max():
            raise ValueError("both treatment arms must appear in the training data")
        if not np.isin(y, [0.0, 1.0]).all():
            raise ValueError("y must be binary (0/1)")
        self._fit(X, t, y)
        self.is_fitted_ = True
        return self

    def _check(self, X) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("model is not fitted")
        return _features(X)

    def predict_outcomes(self, X) -> np.ndarray:
        raise NotImplementedError(f"{self.name} does not model outcome levels")

    def predict_uplift(self, X) -> np.ndarray:
        mu = self.predict_outcomes(X)
        return mu[:, 1] - mu[:, 0]

    def recommend(self, X, budget: float | None = None, row_ids=None) -> np.ndarray:
        """0/1 treatment: uplift > 0, or the top ``budget`` share (deterministic ties)."""
        uplift = self.predict_uplift(X)
        if budget is None:
            return (uplift > 0).astype(np.int8)
        return budget_policy(uplift, budget, row_ids, self.seed)


class LogisticUplift(UpliftModel):
    """L2 logistic regression on ``[z, t, t * z]`` with train-only clipped/standardized features."""

    name = "logistic"

    def __init__(self, seed=SEED, params=None, n_jobs=4, C: float = 0.1):
        super().__init__(seed, params, n_jobs)
        self.C = C

    @staticmethod
    def _design(Z: np.ndarray, t: np.ndarray) -> np.ndarray:
        tc = t[:, None].astype(np.float32)
        return np.hstack([Z, tc, tc * Z])

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return self.scaler_.transform(X, np.arange(len(X)))

    def _fit(self, X, t, y):
        self.scaler_: FeatureScaler = fit_scaler(X, np.arange(len(X)))
        self.model_ = LogisticRegression(C=self.C, max_iter=2000, random_state=self.seed)
        self.model_.fit(self._design(self._scale(X), t), y)

    def predict_outcomes(self, X):
        Z = self._scale(self._check(X))
        return np.column_stack(
            [self.model_.predict_proba(self._design(Z, np.full(len(Z), a)))[:, 1] for a in (0, 1)]
        )


class _TreeModel(UpliftModel):
    def __init__(
        self,
        seed=SEED,
        params=None,
        n_jobs=4,
        early_stopping_rounds: int | None = 30,
        val_fraction: float = 0.15,
    ):
        super().__init__(seed, params, n_jobs)
        self.early_stopping_rounds = early_stopping_rounds
        self.val_fraction = val_fraction

    def _classifier(self, X, y, params=None):
        kw = _lgbm_kwargs(
            LGBM_DEFAULTS, params if params is not None else self.params, self.seed, self.n_jobs
        )
        return _fit_classifier(
            X, y.astype(int), kw, self.early_stopping_rounds, self.val_fraction, self.seed
        )


class SLearner(_TreeModel):
    """One LightGBM classifier on ``[x, t]``; effect = difference of its two predictions."""

    name = "s_learner"

    def _fit(self, X, t, y):
        self.model_ = self._classifier(np.column_stack([X, t]), y)

    def predict_outcomes(self, X):
        X = self._check(X)
        return np.column_stack(
            [_proba(self.model_, np.column_stack([X, np.full(len(X), a)])) for a in (0, 1)]
        )


def _fit_arm_models(model: _TreeModel, X, t, y, params=None) -> list:
    return [model._classifier(X[t == a], y[t == a], params) for a in (0, 1)]


class TLearner(_TreeModel):
    """Separate LightGBM classifiers for control and treated customers."""

    name = "t_learner"

    def _fit(self, X, t, y):
        self.models_ = _fit_arm_models(self, X, t, y)

    def predict_outcomes(self, X):
        X = self._check(X)
        return np.column_stack([_proba(m, X) for m in self.models_])


class DRLearner(_TreeModel):
    """Doubly robust learner: cross-fitted outcome models, known propensity, effect regression.

    1. Training rows are split into ``n_cross_fit`` folds; per-arm LightGBM outcome models fitted on
       the other folds predict ``mu0, mu1`` for the held-out fold, so no row's pseudo-outcome comes
       from a model that saw that row (checked by ``OutcomeModelPredictions``).
    2. Pseudo-outcome ``Gamma_i = mu1 - mu0 + T/e (Y - mu1) - (1 - T)/(1 - e) (Y - mu0)`` with
       ``e = 0.85`` (the C2 AIPW score).
    3. A strongly regularized LightGBM regressor of ``Gamma`` on f0-f11 estimates the effect.
    """

    name = "dr_learner"

    def __init__(
        self,
        seed=SEED,
        params=None,
        n_jobs=4,
        early_stopping_rounds: int | None = 30,
        val_fraction: float = 0.15,
        nuisance_params: dict | None = None,
        n_cross_fit: int = 2,
        p_treat: float = P_TREAT,
    ):
        super().__init__(seed, params, n_jobs, early_stopping_rounds, val_fraction)
        self.nuisance_params = nuisance_params
        self.n_cross_fit = n_cross_fit
        self.p_treat = p_treat

    def _fit(self, X, t, y):
        n = len(y)
        mu = np.empty((n, 2))
        group = np.empty(n, dtype=int)
        trained_on = {}
        skf = StratifiedKFold(self.n_cross_fit, shuffle=True, random_state=self.seed)
        for f, (fit_rows, held) in enumerate(skf.split(X, t * 2 + y.astype(int))):
            models = _fit_arm_models(
                self, X[fit_rows], t[fit_rows], y[fit_rows], self.nuisance_params or {}
            )
            mu[held] = np.column_stack([_proba(m, X[held]) for m in models])
            group[held] = f
            trained_on[f] = fit_rows
        self.oof_nuisance_ = OutcomeModelPredictions(mu, np.arange(n), group, trained_on, n_arms=2)
        self.pseudo_outcomes_ = incremental_scores(y, t, self.oof_nuisance_, self.p_treat)
        kw = _lgbm_kwargs(DR_REGRESSOR_DEFAULTS, self.params, self.seed, self.n_jobs)
        self.effect_model_ = _fit_lgbm(
            LGBMRegressor(**kw),
            X,
            self.pseudo_outcomes_,
            self.early_stopping_rounds,
            self.val_fraction,
            self.seed,
            strata=t,
        )

    def predict_uplift(self, X):
        return self.effect_model_.predict(self._check(X))


MODELS = {
    "logistic": LogisticUplift,
    "s_learner": SLearner,
    "t_learner": TLearner,
    "dr_learner": DRLearner,
}


def make_model(name: str, seed: int = SEED, **kwargs) -> UpliftModel:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](seed=seed, **kwargs)
