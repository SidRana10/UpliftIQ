"""C7 pipeline on a small synthetic prepared dataset: guards, rehearsal, one-shot evaluation."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil

import numpy as np
import pytest

from test_criteo_benchmark import N_EXCL, N_NUIS, N_VAL, make_dataset
from upliftiq.data.criteo_prepare import load_arrays, load_split
from upliftiq.evaluation import binary_policy as bp
from upliftiq.evaluation import criteo_benchmark as cb
from upliftiq.evaluation import criteo_heldout as ch
from upliftiq.evaluation.policy import OutcomeModelPredictions

SEED = 42


def make_frozen(data):
    """A frozen config computed independently of the C7 workers (in-process fit and scoring)."""
    split = load_split(data)
    X, labels = load_arrays(data)
    t = np.asarray(labels["treatment"])
    rows = split.train_idx
    model = cb.build_model("s_learner", SEED, {})
    model.fit(np.asarray(X[rows], np.float32), t[rows], np.asarray(labels["visit"][rows]))
    val = cb.validation_subset(split, t, N_VAL, SEED)
    uplift = np.asarray(model.predict_uplift(np.asarray(X[val], np.float32)), dtype=np.float32)
    path = data / "independent_uplift.npy"
    np.save(path, uplift)
    cb.ensure_nuisance(data, split, N_VAL, SEED, False, N_NUIS, N_EXCL)  # validated nuisance
    nu_path = data / "benchmark" / f"nuisance_n{N_NUIS}_val{N_VAL}.npy"
    nrows = cb.nuisance_train_rows(split, t, N_NUIS, N_EXCL, SEED)
    nuisance = OutcomeModelPredictions.from_holdout(
        np.load(nu_path).astype(np.float64), val, nrows, n_arms=2
    )
    value = (
        bp.evaluate_targeting(
            np.asarray(labels["visit"][val], dtype=float), t[val], {"s_learner": uplift},
            nuisance, budgets=cb.BUDGETS, n_boot=40, seed=SEED, row_ids=val,
        ).policies.query("estimator == 'AIPW' and policy == 'budget' and budget_requested == 0.2")
        .iloc[0].value
    )  # fmt: skip
    return {
        "test_set_accessed": False,
        "selected": {
            "name": "s_learner",
            "family": "s_learner",
            "constructor_kwargs": {},
            "hyperparameters": json.loads(
                json.dumps(cb.effective_hyperparameters(model), default=float)
            ),
            "seed": SEED,
            "training_rows": len(split.train_idx),
            "validation_uplift_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        "selection": {"candidates": {"s_learner": {"value": float(value)}}},
        "dataset": {"source_sha256": "synthetic", "rows": len(split.assignment)},
        "split": {"fingerprint": split.fingerprint(), "seed": SEED},
        "evaluation": {
            "budgets": list(cb.BUDGETS),
            "propensity": bp.P_TREAT,
            "n_boot": 40,
            "validation_subset": {"rows": N_VAL, "seed": SEED},
            "nuisance": {
                "train_rows": N_NUIS,
                "train_rows_fingerprint": hashlib.sha256(nrows.tobytes()).hexdigest()[:16],
                "disjoint_from_train_subsets_up_to": N_EXCL,
            },
        },
    }


def write_cfg(path, cfg):
    path.write_text(json.dumps(cfg, indent=2))
    return path


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("c7")
    data = make_dataset(root / "a")
    cfg = make_frozen(data)
    frozen = write_cfg(root / "frozen_config.json", cfg)
    shutil.copytree(data, root / "b")  # untouched twin: only its test labels will be altered
    return {"root": root, "a": data, "b": root / "b", "cfg": cfg, "frozen": frozen}


@pytest.fixture(scope="module")
def rehearsed(world):
    out = world["root"] / "out_a"
    record = ch.run_rehearsal(world["a"], world["frozen"], out, check_git=False)
    assert not list(ch.work_dir(world["a"]).glob("test_*"))  # no test artifact exists yet
    return {**world, "out": out, "record": record}


@pytest.fixture(scope="module")
def finalized(rehearsed):
    real, seen = ch._spawn, []

    def spy(args):
        if "test" in args:
            seen.append(json.loads((rehearsed["out"] / ch.RECORD_NAME).read_text())["status"])
        real(args)

    mp = pytest.MonkeyPatch()
    mp.setattr(ch, "_spawn", spy)
    try:
        ch.run_final(
            rehearsed["a"], rehearsed["frozen"], rehearsed["out"], confirm=True, check_git=False
        )
    finally:
        mp.undo()
    return {**rehearsed, "scoring_record_states": seen}


def test_rehearsal_passes_without_touching_test_rows_and_is_labelled_dev_only(rehearsed):
    rec = rehearsed["record"]
    assert rec["status"] == "passed" and rec["test_set_accessed"] is False
    assert rec["checks"]["refit validation-score SHA-256 equals frozen hash"] is True
    text = (rehearsed["out"] / "rehearsal" / "final_report.md").read_text()
    assert "NOT test results" in text and "validation split" in text
    assert not (rehearsed["out"] / ch.RECORD_NAME).exists()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["dataset"].update(source_sha256="deadbeef"),
        lambda c: c["split"].update(fingerprint="0" * 64),
        lambda c: c["selected"]["hyperparameters"]["lightgbm"].update(n_estimators=401),
        lambda c: c["evaluation"].update(budgets=[0.1, 0.2, 0.5]),
        lambda c: c["evaluation"].update(propensity=0.8),
        lambda c: c["selected"].update(training_rows=1),
        lambda c: c.update(test_set_accessed=True),
        lambda c: c["selected"].update(name="not_a_model"),
        lambda c: c["selected"].update(name="tarnet_cpu", family="tarnet"),
    ],
)
def test_any_frozen_config_mismatch_is_refused_before_data_access(world, mutate):
    cfg = copy.deepcopy(world["cfg"])
    mutate(cfg)
    with pytest.raises(ch.FrozenConfigError):
        ch.verify_frozen(cfg, world["a"])


def test_valid_config_verifies_and_tampered_hash_aborts_reproducibility(rehearsed):
    split = ch.verify_frozen(rehearsed["cfg"], rehearsed["a"])
    assert split.fingerprint() == rehearsed["cfg"]["split"]["fingerprint"]
    sha = hashlib.sha256(rehearsed["frozen"].read_bytes()).hexdigest()
    ch.check_reproducibility(rehearsed["a"], rehearsed["cfg"], sha)
    bad = copy.deepcopy(rehearsed["cfg"])
    bad["selected"]["validation_uplift_sha256"] = "0" * 64
    with pytest.raises(ch.ReproducibilityError, match="frozen hash"):
        ch.check_reproducibility(rehearsed["a"], bad, sha)


def test_final_refuses_without_confirmation_or_rehearsal_and_writes_nothing(world, tmp_path):
    with pytest.raises(ch.FrozenConfigError, match="confirmation"):
        ch.run_final(world["a"], world["frozen"], tmp_path, confirm=False, check_git=False)
    with pytest.raises(ch.FrozenConfigError, match="rehearsal"):
        ch.run_final(world["b"], world["frozen"], tmp_path, confirm=True, check_git=False)
    assert not (tmp_path / ch.RECORD_NAME).exists()
    assert not list(ch.work_dir(world["b"]).glob("test_*"))


def test_test_scoring_worker_needs_the_one_shot_record(rehearsed):
    args = type("A", (), {"data_dir": rehearsed["a"], "holdout": "test", "intent": None})()
    with pytest.raises(ch.FrozenConfigError, match="one-shot record"):
        ch.run_score_worker(args)


def test_final_is_one_shot_records_before_scoring_and_scores_only_the_frozen_model(finalized):
    assert finalized["scoring_record_states"] == ["started"]  # record exists before test scoring
    rec = json.loads((finalized["out"] / ch.RECORD_NAME).read_text())
    assert rec["status"] == "completed" and rec["models_evaluated_on_test"] == ["s_learner"]
    assert rec["test_rows"] == len(load_split(finalized["a"]).test_idx)
    before = (finalized["out"] / ch.RECORD_NAME).read_bytes()
    with pytest.raises(ch.FrozenConfigError, match="must not be re-run"):
        ch.run_final(
            finalized["a"], finalized["frozen"], finalized["out"], confirm=True, check_git=False
        )
    assert (finalized["out"] / ch.RECORD_NAME).read_bytes() == before


def test_final_outputs_report_limits_references_and_exploratory_conversion(finalized):
    out = finalized["out"]
    flat = " ".join((out / "final_report.md").read_text().split())
    for phrase in (
        "evaluated **once**",
        "conditional estimates for the released Criteo sample",
        "IPW minus AIPW",
        "unconstrained reference",
        "Exploratory: conversion uplift",
        "AIPW is primary",
    ):
        assert phrase in flat
    res = json.loads((out / "final_results.json").read_text())
    assert res["kind"] == "final" and res["model"] == "s_learner"
    value = {r["budget"]: r for r in res["tables"]["value"]}
    assert set(value) == {"10%", "20%", "30%"} and value["20%"]["primary"] == "yes"
    types = {r["policy"]: r["type"] for r in res["tables"]["references"]}
    assert types["treat_all"] == "unconstrained reference"
    assert types["treat_nobody"] == "zero baseline" and types["random@20%"] == "budget-feasible"
    assert res["exploratory"]["note"].startswith("exploratory only")
    for f in ("heldout_gain_curve.png", "heldout_value_vs_random.png", "heldout_value.csv"):
        assert (out / f).exists()


def test_test_labels_never_influence_the_fit_or_the_scores(finalized):
    b, root = finalized["b"], finalized["root"]
    split = load_split(b)
    for name in ("visit", "conversion"):
        y = np.load(b / f"{name}.npy")
        y[split.test_idx] = 1 - y[split.test_idx]
        np.save(b / f"{name}.npy", y)
    out_b = root / "out_b"
    ch.run_rehearsal(b, finalized["frozen"], out_b, check_git=False)
    ch.run_final(b, finalized["frozen"], out_b, confirm=True, check_git=False)
    wa, wb = ch.work_dir(finalized["a"]), ch.work_dir(b)
    for f in ("test_visit_uplift.npy", "test_conversion_uplift.npy", "test_visit_nuisance.npy"):
        assert (wa / f).read_bytes() == (wb / f).read_bytes()
    ra = json.loads((finalized["out"] / "final_results.json").read_text())
    rb = json.loads((out_b / "final_results.json").read_text())
    assert ra["policies"] != rb["policies"]  # outcomes do enter the *evaluation*, as they must
