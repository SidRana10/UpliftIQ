"""C6 final: selection rule, comparison with the earlier rung, and the frozen configuration."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from test_criteo_benchmark import N_EXCL, N_NUIS, N_VAL, make_dataset
from upliftiq.data.criteo_prepare import load_split
from upliftiq.evaluation import criteo_benchmark as cb
from upliftiq.evaluation import criteo_final as cf

ORDER = ("logistic", "s_learner", "tarnet_cpu")


def fake_bundle(values, beats, diffs, qini=None):
    """Minimal scored bundle: value@20% per model, vs-random lower bounds, paired differences."""
    pol = pd.DataFrame(
        [
            {
                "model": n,
                "estimator": "AIPW",
                "policy": "budget",
                "budget_requested": 0.2,
                "value": values[n],
                "ci_low": values[n] - 0.1,
                "ci_high": values[n] + 0.1,
                "diff_vs_random": 0.3,
                "diff_vs_random_ci_low": 0.1 if beats[n] else -0.1,
                "diff_vs_random_ci_high": 0.5,
            }
            for n in ORDER
        ]
    )
    qini = qini or {n: values[n] for n in ORDER}
    q = pd.DataFrame([{"model": n, "estimator": "AIPW", "qini": qini[n]} for n in ORDER])
    pw = pd.DataFrame(
        [
            {
                "model": m,
                "estimator": "AIPW",
                "metric": "value@20%",
                "diff": d,
                "ci_low": d - 1.96 * se,
                "ci_high": d + 1.96 * se,
            }
            for (m, ref), (d, se) in diffs.items()
            for _ in [None]
            if ref == "logistic"
        ]
    )
    pw2 = pd.DataFrame(
        [
            {
                "model": m,
                "estimator": "AIPW",
                "metric": "value@20%",
                "diff": d,
                "ci_low": d - 1.96 * se,
                "ci_high": d + 1.96 * se,
            }
            for (m, ref), (d, se) in diffs.items()
            if ref == "tarnet_cpu"
        ]
    )
    res = lambda p: SimpleNamespace(policies=pol, qini=q, pairwise=p)  # noqa: E731
    return {"results": {"logistic": res(pw), "tarnet_cpu": res(pw2)}}


DIFFS = {
    ("s_learner", "logistic"): (0.10, 0.04),
    ("tarnet_cpu", "logistic"): (0.05, 0.04),
    ("logistic", "tarnet_cpu"): (-0.05, 0.04),
    ("s_learner", "tarnet_cpu"): (0.05, 0.04),
}
VALUES = {"logistic": 0.60, "s_learner": 0.70, "tarnet_cpu": 0.65}


def test_paired_difference_is_signed_and_symmetric():
    b = fake_bundle(VALUES, dict.fromkeys(ORDER, True), DIFFS)
    ab = cf.paired_difference(b, "s_learner", "logistic", "value@20%")
    ba = cf.paired_difference(b, "logistic", "s_learner", "value@20%")
    assert ab["diff"] == pytest.approx(0.10) and ab["se"] == pytest.approx(0.04, rel=1e-3)
    assert ba["diff"] == pytest.approx(-0.10) and ba["ci_low"] == pytest.approx(-ab["ci_high"])
    with pytest.raises(KeyError):
        cf.paired_difference(b, "s_learner", "s_learner", "value@20%")


def test_clearly_better_complex_model_wins_and_simpler_within_one_se_wins_ties():
    b = fake_bundle(VALUES, dict.fromkeys(ORDER, True), DIFFS)
    sel = cf.select(b, ORDER)
    # S beats logistic by 0.10 > one SE (0.04): the simpler model is rejected and S is chosen
    assert sel["best_by_value"] == "s_learner" and sel["chosen"] == "s_learner"
    assert [s["within_one_se"] for s in sel["steps"]] == [False]
    close = dict(DIFFS) | {("s_learner", "logistic"): (0.03, 0.04)}
    sel = cf.select(
        fake_bundle({**VALUES, "logistic": 0.67}, dict.fromkeys(ORDER, True), close), ORDER
    )
    assert sel["chosen"] == "logistic"  # within one paired SE of the best -> simplest wins
    assert sel["qini_supports_choice"] is False


def test_model_that_does_not_beat_random_is_ineligible_even_if_best():
    beats = {"logistic": True, "s_learner": False, "tarnet_cpu": True}
    sel = cf.select(fake_bundle(VALUES, beats, DIFFS), ORDER)
    assert "s_learner" not in sel["eligible"] and sel["best_by_value"] == "tarnet_cpu"
    none = cf.select(fake_bundle(VALUES, dict.fromkeys(ORDER, False), DIFFS), ORDER)
    assert none["chosen"] is None and "random" in none["reason"]


# ------------------------------------------------------------ end to end on a small dataset

N_BASE = 8_000
SPECS = {
    "logistic": ("logistic", {}),
    "s_learner": ("s_learner", {"params": {"n_estimators": 30}, "n_jobs": 2}),
    "tarnet_cpu": ("tarnet", {"device": "cpu", "max_epochs": 3}),
}


@pytest.fixture(scope="module")
def final(tmp_path_factory):
    root = tmp_path_factory.mktemp("c6final")
    data = make_dataset(root / "data")
    cb.run_ladder(
        [N_BASE], list(ORDER), data, root / "base", n_val=N_VAL, n_boot=40,
        nuisance_rows=N_NUIS, nuisance_exclude=N_EXCL, specs=SPECS,
    )  # fmt: skip
    report = cf.run(
        data, root / "full", ORDER, n_val=N_VAL, n_boot=40, base_size=N_BASE,
        nuisance_rows=N_NUIS, nuisance_exclude=N_EXCL, specs=SPECS,
    )  # fmt: skip
    return {"data": data, "out": report.parent, "root": root}


def test_full_run_trains_on_entire_train_split_and_freezes_before_test(final):
    split = load_split(final["data"])
    frozen = json.loads((final["out"] / "frozen_config.json").read_text())
    sel = frozen["selected"]
    assert sel["training_rows"] == len(split.train_idx)
    assert frozen["test_set_accessed"] is False
    assert frozen["split"]["fingerprint"] == split.fingerprint()
    assert frozen["dataset"]["source_sha256"] == "synthetic"
    assert frozen["dataset"]["rows"] == len(split.assignment)
    assert sel["name"] in ORDER and sel["seed"] == 42 and sel["hyperparameters"]
    assert frozen["evaluation"]["validation_subset"]["rows"] == N_VAL
    assert frozen["evaluation"]["budgets"] == list(cb.BUDGETS)
    run_dir = cb.rung_dir(final["data"], len(split.train_idx), N_VAL)
    digest = hashlib.sha256((run_dir / f"{sel['name']}_uplift.npy").read_bytes()).hexdigest()
    assert sel["validation_uplift_sha256"] == digest


def test_report_states_limits_preserves_ipw_aipw_and_compares_with_earlier_rung(final):
    flat = " ".join((final["out"] / "final_report.md").read_text().split())
    assert "test set was not accessed" in flat and "not population-level" in flat
    assert "IPW minus AIPW" in flat and "Selected finalist" in flat
    assert "Change from the previous 2M benchmark" in flat and "change (pp, paired)" in flat
    res = json.loads((final["out"] / "final_results.json").read_text())
    assert res["status"].startswith("validation-only") and res["n_fit"] > N_BASE
    assert {c["model"] for c in res["changes_vs_base"]} == set(ORDER)
    for c in res["changes_vs_base"]:
        assert c["change"][1] <= c["change"][0] <= c["change"][2]
    for m in res["manifests"].values():
        assert m["hyperparameters"] and m["omp_num_threads"] == "1"


def test_missing_earlier_rung_is_an_error_not_a_silent_retrain(final):
    with pytest.raises(FileNotFoundError, match="missing"):
        cf.run(
            final["data"], final["root"] / "x", ORDER, n_val=N_VAL, base_size=7_777,
            specs=SPECS, nuisance_rows=N_NUIS, nuisance_exclude=N_EXCL,
        )  # fmt: skip


def test_rows_of_the_test_set_never_change_the_final_selection(tmp_path):
    a = make_dataset(tmp_path / "a")
    import shutil

    shutil.copytree(a, tmp_path / "b")
    b = tmp_path / "b"
    test_idx = load_split(b).test_idx
    X, y = np.load(b / "X.npy"), np.load(b / "visit.npy")
    X[test_idx] += 1000.0
    y[test_idx] = 1 - y[test_idx]
    np.save(b / "X.npy", X)
    np.save(b / "visit.npy", y)
    one = {"logistic": SPECS["logistic"]}
    outs = []
    for d, o in ((a, "ra"), (b, "rb")):
        cb.run_ladder([N_BASE], ["logistic"], d, tmp_path / (o + "0"), n_val=N_VAL, n_boot=40,
                      nuisance_rows=N_NUIS, nuisance_exclude=N_EXCL, specs=one)  # fmt: skip
        r = cf.run(d, tmp_path / o, ["logistic"], n_val=N_VAL, n_boot=40, base_size=N_BASE,
                   nuisance_rows=N_NUIS, nuisance_exclude=N_EXCL, specs=one)  # fmt: skip
        outs.append(json.loads((r.parent / "final_results.json").read_text()))
    assert outs[0]["policies"] == outs[1]["policies"]
    assert outs[0]["changes_vs_base"] == outs[1]["changes_vs_base"]
