"""TARNet on synthetic 85/15 binary-treatment experiments with known effects."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from upliftiq.data.criteo import FEATURES
from upliftiq.evaluation import binary_policy as bp
from upliftiq.evaluation.policy import OutcomeModelPredictions
from upliftiq.modeling import criteo_tarnet as tn

FAST = {"max_epochs": 12, "patience": 3, "device": "cpu"}


def experiment(n, seed, aa=False):
    """p0 in [0.22, 0.38]; true tau = 0.18 * tanh(1.5 x0) (0 under A/A); 85% treated."""
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, 12)).astype(np.float32)
    p0 = 0.3 + 0.08 * np.tanh(X[:, 1])
    tau = np.zeros(n) if aa else 0.18 * np.tanh(1.5 * X[:, 0])
    t = (r.random(n) < 0.85).astype(np.int8)
    y = (r.random(n) < p0 + t * tau).astype(float)
    return X, t, y, tau


@pytest.fixture(scope="module")
def data():
    return experiment(40_000, 1), experiment(20_000, 2)


@pytest.fixture(scope="module")
def model(data):
    (X, t, y, _), _ = data
    return tn.TARNet(**FAST).fit(X, t, y)


def test_shapes_probabilities_direction_and_recovery(model, data):
    _, (Xv, _, _, tau) = data
    mu = model.predict_outcomes(Xv)
    assert mu.shape == (len(Xv), 2) and ((mu > 0) & (mu < 1)).all()
    u = model.predict_uplift(Xv)
    assert u.shape == (len(Xv),)
    np.testing.assert_allclose(u, mu[:, 1] - mu[:, 0], atol=1e-12)
    assert np.corrcoef(u, tau)[0, 1] > 0.8
    assert u[Xv[:, 0] > 0.5].mean() > 0.05 > -0.05 > u[Xv[:, 0] < -0.5].mean()
    rec = model.recommend(Xv)
    assert np.array_equal(rec, (u > 0).astype(np.int8))
    top = model.recommend(Xv, budget=0.2)
    assert top.sum() == int(np.floor(0.2 * len(Xv))) and tau[top == 1].mean() > tau.mean()


def test_loss_trains_only_the_assigned_head():
    logits = torch.zeros(4, 2, requires_grad=True)
    t = torch.tensor([1.0, 1.0, 0.0, 1.0])
    y = torch.tensor([1.0, 0.0, 1.0, 1.0])
    tn.factual_loss(logits, t, y).backward()
    g = logits.grad
    assert g[2, 0] != 0 and g[2, 1] == 0  # control row -> control head only
    assert (g[[0, 1, 3], 0] == 0).all() and (g[[0, 1, 3], 1] != 0).all()
    # equals the plain BCE of the picked logits
    picked = torch.tensor([0.3, -1.0, 0.5, 2.0])
    lg = torch.stack([picked * 0 + picked, picked], dim=1)
    ref = torch.nn.functional.binary_cross_entropy_with_logits(picked, y)
    assert tn.factual_loss(lg, torch.zeros(4), y) == pytest.approx(float(ref))


def test_flipping_treatment_labels_flips_the_effect(data):
    (X, t, y, _), (Xv, _, _, tau) = data
    m = tn.TARNet(**FAST).fit(X, 1 - t, y)
    assert np.corrcoef(m.predict_uplift(Xv), tau)[0, 1] < -0.8


def test_aa_experiment_gives_no_apparent_uplift():
    X, t, y, _ = experiment(40_000, 3, aa=True)
    Xv, tv, yv, _ = experiment(60_000, 4, aa=True)
    u = tn.TARNet(**FAST).fit(X, t, y).predict_uplift(Xv)
    assert np.abs(u).mean() < 0.04
    res = bp.evaluate_targeting(yv, tv, {"tarnet": u}, n_boot=100)
    assert ((res.qini.ci_low <= 0) & (res.qini.ci_high >= 0)).all()


def test_cpu_fits_are_reproducible_and_seed_sensitive(data):
    (X, t, y, _), (Xv, *_) = data
    kw = {**FAST, "max_epochs": 4}
    a = tn.TARNet(**kw).fit(X[:10_000], t[:10_000], y[:10_000])
    b = tn.TARNet(**kw).fit(X[:10_000], t[:10_000], y[:10_000])
    np.testing.assert_array_equal(a.predict_uplift(Xv), b.predict_uplift(Xv))
    c = tn.TARNet(**kw, seed=7).fit(X[:10_000], t[:10_000], y[:10_000])
    assert not np.array_equal(a.predict_uplift(Xv), c.predict_uplift(Xv))


def test_training_is_tracked_and_early_stopping_restores_the_best_epoch(model, data):
    h = pd.DataFrame(model.history_)
    assert {"epoch", "train_loss", "val_loss", "seconds"} <= set(h.columns)
    assert h.train_loss.iloc[-1] < h.train_loss.iloc[0]
    assert model.best_epoch_ == int(h.loc[h.val_loss.idxmin(), "epoch"])
    assert len(h) <= FAST["max_epochs"]
    model.predict_uplift(data[1][0])
    s = model.training_summary()
    assert s["device"] == "cpu" and s["fit_seconds"] > 0
    assert s["last_predict_ms_per_1k_rows"] > 0


def test_only_f0_f11_are_used_and_scaling_is_train_only(data):
    (X, t, y, _), (Xv, *_) = data
    kw = {**FAST, "max_epochs": 3}
    clean = pd.DataFrame(X[:8000], columns=FEATURES)
    poisoned = clean.assign(
        treatment=np.random.default_rng(0).integers(0, 2, 8000),
        exposure=1,
        visit=y[:8000],
        conversion=0,
    )
    a = tn.TARNet(**kw).fit(clean, t[:8000], y[:8000])
    b = tn.TARNet(**kw).fit(poisoned, t[:8000], y[:8000])
    np.testing.assert_array_equal(a.predict_uplift(Xv[:500]), b.predict_uplift(Xv[:500]))
    before = a.scaler_.to_dict()
    a.predict_uplift(Xv[:500] * 1000)
    assert a.scaler_.to_dict() == before
    with pytest.raises(ValueError):
        a.predict_uplift(Xv[:, :11])


def test_plugs_into_the_c2_evaluator_with_out_of_sample_nuisance(model, data):
    (X, *_), (Xv, tv, yv, _) = data
    n_tr = len(X)
    nuis = OutcomeModelPredictions.from_holdout(
        model.predict_outcomes(Xv), np.arange(n_tr, n_tr + len(Xv)), np.arange(n_tr), n_arms=2
    )
    u = model.predict_uplift(Xv)
    res = bp.evaluate_targeting(yv, tv, {"tarnet": u}, nuis, n_boot=60, row_ids=nuis.row_ids)
    assert (res.qini.query("estimator == 'AIPW'").ci_low > 0).all()
    rec = model.recommend(Xv, budget=0.3, row_ids=nuis.row_ids)
    assert np.array_equal(rec, bp.budget_policy(u, 0.3, nuis.row_ids))


def test_device_selection_and_cpu_fallback(monkeypatch):
    assert tn.resolve_device("cpu").type == "cpu"
    assert tn.resolve_device("auto").type in ("cpu", "mps")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert tn.resolve_device("auto").type == "cpu"
    with pytest.raises(RuntimeError):
        tn.resolve_device("mps")
    with pytest.raises(ValueError):
        tn.resolve_device("cuda")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_training_works_and_matches_cpu_quality(data):
    (X, t, y, _), (Xv, _, _, tau) = data
    m = tn.TARNet(max_epochs=8, device="mps").fit(X, t, y)
    assert m.training_summary()["device"] == "mps"
    assert np.corrcoef(m.predict_uplift(Xv), tau)[0, 1] > 0.8


def test_input_validation(data):
    (X, t, y, _), _ = data
    with pytest.raises(RuntimeError):
        tn.TARNet(**FAST).predict_uplift(X)
    for bad in (t + 1, np.zeros_like(t)):
        with pytest.raises(ValueError):
            tn.TARNet(**FAST).fit(X, bad, y)
