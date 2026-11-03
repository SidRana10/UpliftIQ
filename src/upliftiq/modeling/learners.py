"""Uplift learners for the three-arm experiment (binary outcome, e.g. visit).

Common interface (all models):

    model = make_model("t_learner", seed=42).fit(X_train, arm_train, y_train)
    model.predict_uplift(X)   # (n, 3): effect vs No E-Mail per arm; column 0 is exactly 0
    model.recommend(X)        # (n,) arm codes (argmax of predict_uplift; ties -> No E-Mail),
                              # directly usable as a policy in ``evaluation.policy``
    model.predict_outcomes(X) # (n, 3): P(Y=1 | x, arm) -- outcome-model learners only

``X`` is a DataFrame; only the seven approved pre-treatment features are ever read (extra columns
such as outcomes or the treatment are ignored), and each model fits its own preprocessor on the
training rows it is given, so fitting inside a CV fold is leakage-safe. ``arm`` uses the integer
codes of ``modeling.features.ARMS``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

from upliftiq.evaluation.policy import OutcomeModelPredictions, arm_scores
from upliftiq.modeling.features import (
    PROPENSITY,
    make_preprocessor,
    select_features,
    tree_categorical_indices,
)

N_ARMS = 3
SEED = 42

LGBM_CLASSIFIER_DEFAULTS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 8,
    "min_child_samples": 100,
}
# The DR regression target (a pseudo-outcome) is very noisy, so it is regularized harder.
DR_REGRESSOR_DEFAULTS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 4,
    "min_child_samples": 300,
}


class _ConstantProb:
    """Fallback when a training subset contains a single outcome class."""

    def __init__(self, p: float):
        self.p = p

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)])


def _lgbm_kwargs(defaults: dict, overrides: dict | None, seed: int) -> dict:
    return {
        **defaults,
        **(overrides or {}),
        "random_state": seed,
        "deterministic": True,
        "force_row_wise": True,
        "n_jobs": 1,
        "verbose": -1,
    }


def _fit_lgbm(model, X, y, cat_idx, es_rounds, val_fraction, seed, strata=None):
    """Fit a LightGBM estimator, early-stopping on a held-out part of the *training* rows."""
    if es_rounds is None:
        return model.fit(X, y, categorical_feature=cat_idx)
    tr, va = train_test_split(
        np.arange(len(y)), test_size=val_fraction, random_state=seed, stratify=strata
    )
    return model.fit(
        X[tr],
        y[tr],
        eval_X=X[va],
        eval_y=y[va],
        categorical_feature=cat_idx,
        callbacks=[early_stopping(es_rounds, verbose=False)],
    )


def _fit_classifier(X, y, params, cat_idx, es_rounds, val_fraction, seed):
    if len(np.unique(y)) < 2:
        return _ConstantProb(float(y.mean()))
    return _fit_lgbm(
        LGBMClassifier(**params), X, y, cat_idx, es_rounds, val_fraction, seed, strata=y
    )


def _proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


class UpliftModel:
    """Base class: input validation, feature selection and the shared prediction API."""

    name = "base"

    def __init__(self, seed: int = SEED, params: dict | None = None):
        self.seed = seed
        self.params = params
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame, arm, y) -> UpliftModel:
        arm, y = np.asarray(arm), np.asarray(y, dtype=float)
        if len(X) != len(arm) or len(X) != len(y):
            raise ValueError("X, arm and y must have the same length")
        if not np.issubdtype(arm.dtype, np.integer) or arm.min() < 0 or arm.max() >= N_ARMS:
            raise ValueError(f"arm must be integer codes in 0..{N_ARMS - 1}")
        if set(np.unique(arm)) != set(range(N_ARMS)):
            raise ValueError("every treatment arm must appear in the training data")
        if not np.isin(y, [0.0, 1.0]).all():
            raise ValueError("y must be binary (0/1)")
        self._fit(select_features(X), arm, y)
        self.is_fitted_ = True
        return self

    def _check(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("model is not fitted")
        return select_features(X)

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        mu = self.predict_outcomes(X)
        return mu - mu[:, [0]]

    def predict_outcomes(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError(f"{self.name} does not model outcome levels")

    def recommend(self, X: pd.DataFrame) -> np.ndarray:
        """Arm with the largest predicted uplift; No E-Mail unless an e-mail has uplift > 0."""
        return np.argmax(self.predict_uplift(X), axis=1).astype(int)


class LogisticUplift(UpliftModel):
    """L2 logistic regression on ``[z, arm dummies, arm dummies x z]`` (control = reference)."""

    name = "logistic"

    def __init__(self, seed: int = SEED, params: dict | None = None, C: float = 1.0):
        super().__init__(seed, params)
        self.C = C

    def _design(self, Z: np.ndarray, arm: np.ndarray) -> np.ndarray:
        dummies = np.column_stack([arm == 1, arm == 2]).astype(float)
        inter = [dummies[:, [j]] * Z for j in range(2)]
        return np.hstack([Z, dummies, *inter])

    def _fit(self, X, arm, y):
        self.preprocessor_ = make_preprocessor("linear")
        Z = self.preprocessor_.fit_transform(X)
        self.model_ = LogisticRegression(C=self.C, max_iter=2000, random_state=self.seed)
        self.model_.fit(self._design(Z, arm), y)

    def predict_outcomes(self, X):
        X = self._check(X)
        Z = self.preprocessor_.transform(X)
        cols = [
            self.model_.predict_proba(self._design(Z, np.full(len(Z), a)))[:, 1]
            for a in range(N_ARMS)
        ]
        return np.column_stack(cols)


class SLearner(UpliftModel):
    """One LightGBM classifier with the arm as a categorical feature."""

    name = "s_learner"

    def __init__(
        self, seed=SEED, params=None, early_stopping_rounds: int | None = 30, val_fraction=0.15
    ):
        super().__init__(seed, params)
        self.early_stopping_rounds = early_stopping_rounds
        self.val_fraction = val_fraction

    def _with_arm(self, T: np.ndarray, arm: np.ndarray) -> np.ndarray:
        return np.column_stack([T, arm])

    def _fit(self, X, arm, y):
        self.preprocessor_ = make_preprocessor("tree")
        T = self.preprocessor_.fit_transform(X)
        self.cat_idx_ = [*tree_categorical_indices(), T.shape[1]]  # last column = arm
        params = _lgbm_kwargs(LGBM_CLASSIFIER_DEFAULTS, self.params, self.seed)
        self.model_ = _fit_classifier(
            self._with_arm(T, arm),
            y.astype(int),
            params,
            self.cat_idx_,
            self.early_stopping_rounds,
            self.val_fraction,
            self.seed,
        )

    def predict_outcomes(self, X):
        X = self._check(X)
        T = self.preprocessor_.transform(X)
        cols = [_proba(self.model_, self._with_arm(T, np.full(len(T), a))) for a in range(N_ARMS)]
        return np.column_stack(cols)


class TLearner(UpliftModel):
    """Three LightGBM classifiers, one per arm, each trained only on that arm's customers."""

    name = "t_learner"

    def __init__(
        self, seed=SEED, params=None, early_stopping_rounds: int | None = 30, val_fraction=0.15
    ):
        super().__init__(seed, params)
        self.early_stopping_rounds = early_stopping_rounds
        self.val_fraction = val_fraction

    def _fit(self, X, arm, y):
        self.preprocessor_ = make_preprocessor("tree")
        T = self.preprocessor_.fit_transform(X)
        self.models_ = _fit_arm_models(T, arm, y, self.params, self)

    def predict_outcomes(self, X):
        X = self._check(X)
        T = self.preprocessor_.transform(X)
        return np.column_stack([_proba(m, T) for m in self.models_])


def _fit_arm_models(T, arm, y, params, cfg) -> list:
    kw = _lgbm_kwargs(LGBM_CLASSIFIER_DEFAULTS, params, cfg.seed)
    return [
        _fit_classifier(
            T[arm == a],
            y[arm == a].astype(int),
            kw,
            tree_categorical_indices(),
            cfg.early_stopping_rounds,
            cfg.val_fraction,
            cfg.seed,
        )
        for a in range(N_ARMS)
    ]


class DRLearner(UpliftModel):
    """Doubly robust learner with known propensities and cross-fitted nuisance models.

    1. Split the training rows into ``n_cross_fit`` folds. Per-arm LightGBM outcome models are
       fitted on all other folds and predict ``mu_hat`` for the held-out fold, so no row's
       pseudo-outcome uses a model trained on that row.
    2. Pseudo-outcome for arm k: ``Gamma_k = psi_k - psi_0`` where ``psi_a`` is the AIPW score of
       ``evaluation.policy.arm_scores`` (known propensity 1/3).
    3. A strongly regularized LightGBM regressor of ``Gamma_k`` on the features gives
       ``tau_k(x)``. ``tau_0 = 0``.
    """

    name = "dr_learner"

    def __init__(
        self,
        seed=SEED,
        params=None,
        nuisance_params=None,
        n_cross_fit: int = 2,
        early_stopping_rounds: int | None = 30,
        val_fraction=0.15,
    ):
        super().__init__(seed, params)
        self.nuisance_params = nuisance_params
        self.n_cross_fit = n_cross_fit
        self.early_stopping_rounds = early_stopping_rounds
        self.val_fraction = val_fraction

    def _fit(self, X, arm, y):
        self.preprocessor_ = make_preprocessor("tree")
        T = self.preprocessor_.fit_transform(X)
        n = len(y)
        cat_idx = tree_categorical_indices()

        # --- cross-fitted nuisance predictions (each row predicted by models not trained on it)
        mu = np.empty((n, N_ARMS))
        group = np.empty(n, dtype=int)
        trained_on = {}
        strata = arm * 2 + y.astype(int)
        skf = StratifiedKFold(self.n_cross_fit, shuffle=True, random_state=self.seed)
        for f, (fit_rows, held) in enumerate(skf.split(T, strata)):
            models = _fit_arm_models(
                T[fit_rows], arm[fit_rows], y[fit_rows], self.nuisance_params, self
            )
            mu[held] = np.column_stack([_proba(m, T[held]) for m in models])
            group[held] = f
            trained_on[f] = fit_rows
        self.oof_nuisance_ = OutcomeModelPredictions(mu, np.arange(n), group, trained_on)

        # --- doubly robust pseudo-outcomes with the known propensity
        psi = arm_scores(y, arm, self.oof_nuisance_, PROPENSITY)
        self.pseudo_outcomes_ = psi[:, 1:] - psi[:, [0]]  # (n, 2): arms 1 and 2 vs control

        # --- effect regressors
        kw = _lgbm_kwargs(DR_REGRESSOR_DEFAULTS, self.params, self.seed)
        self.effect_models_ = [
            _fit_lgbm(
                LGBMRegressor(**kw),
                T,
                self.pseudo_outcomes_[:, j],
                cat_idx,
                self.early_stopping_rounds,
                self.val_fraction,
                self.seed,
                strata=arm,
            )
            for j in range(2)
        ]

    def predict_uplift(self, X):
        X = self._check(X)
        T = self.preprocessor_.transform(X)
        return np.column_stack([np.zeros(len(T)), *(m.predict(T) for m in self.effect_models_)])


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
