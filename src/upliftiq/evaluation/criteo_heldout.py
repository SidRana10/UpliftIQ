"""C7: one-shot held-out evaluation of the frozen Criteo model.

Flow (``docs/criteo_modeling_plan.md`` section 6, frozen config from C6):

    verify    check the frozen config, dataset checksum, split fingerprint, hyperparameters and
              evaluation protocol. No test row is read.
    rehearse  development-only dry run of the *same* pipeline with the validation split standing in
              for the test set: refit the frozen model, prove reproducibility against the stored
              validation-score hash, and re-derive the frozen validation number.
    final     only after a passing rehearsal and with ``--confirm-one-shot``: write the one-shot
              record, score the test features, then evaluate once. Never repeatable.

Heavy work runs in isolated worker processes (``OMP_NUM_THREADS=1`` set before Python starts):
``fit`` trains the frozen S-learner and the AIPW nuisance on training rows only; ``score`` predicts a
holdout's *features* and never opens its labels. The driver process touches test labels only inside
``run_final``, after the one-shot record exists. Only the frozen model is evaluated: no losing
finalist is ever scored on the test set, and test outcomes cannot change the model or policy.

AIPW nuisance (visit and, exploratory, conversion): the same LightGBM T-learner configuration and the
same 500K training rows as in validation, fitted on training data only (no validation or test rows).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from upliftiq.data.criteo_prepare import (  # noqa: E402
    PROCESSED_DIR,
    REPORT_DIR,
    CriteoSplit,
    load_arrays,
    load_split,
    sha256_file,
)
from upliftiq.data.registry import CRITEO, PROJECT_ROOT  # noqa: E402
from upliftiq.evaluation import binary_policy as bp  # noqa: E402
from upliftiq.evaluation import criteo_benchmark as cb  # noqa: E402
from upliftiq.evaluation.policy import OutcomeModelPredictions  # noqa: E402

FROZEN_PATH = REPORT_DIR / "benchmark_full" / "frozen_config.json"
FINAL_DIR = REPORT_DIR / "final"
RAW_PATH = PROJECT_ROOT / "data" / "raw" / CRITEO.filename
RECORD_NAME = "evaluation_record.json"
REHEARSAL_NAME = "rehearsal.json"
TARGETS = ("visit", "conversion")
VALUE_TOL = 1e-9

LIMITATIONS = """\
**Read first: identification and scope limits.**
- Values are **conditional estimates for the released Criteo sample**, not population-level
  incrementality, advertising ROI, or guaranteed causal gains. Criteo's outcome-dependent privacy
  subsampling limits causal interpretation.
- The constant 0.85 treatment propensity is publisher-supported but cannot be verified from the
  file. IPW (which uses it directly) and AIPW disagree on this file (documented in
  `reports/criteo/benchmark/ipw_aipw_diagnostic.md`): IPW is likely biased upward, so **AIPW is
  primary** and IPW is shown only as a cross-check. AIPW is not guaranteed unbiased if the
  propensity assumption and the outcome nuisance are both wrong.
- The model was selected on validation data."""

FINAL_NOTE = (
    "- The held-out test set was used once, for this report only; nothing was changed afterwards."
)


class FrozenConfigError(RuntimeError):
    """The frozen configuration, data, split or protocol does not match what was frozen."""


class ReproducibilityError(RuntimeError):
    """A refit did not reproduce the stored validation predictions."""


# ---------------------------------------------------------------- helpers


def _json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float) + "\n")


def _bytes_sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _git_ok(path: Path) -> bool:
    """True if ``path`` is tracked by git and unchanged since HEAD."""
    import subprocess

    def run(*a):
        return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True)

    try:
        return (
            run("ls-files", "--error-unmatch", str(path)).returncode == 0
            and run("diff", "--quiet", "HEAD", "--", str(path)).returncode == 0
        )
    except OSError:
        return False


def work_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "final"


def load_frozen(path: Path = FROZEN_PATH) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------- verification (no test access)


def verify_frozen(
    cfg: dict,
    data_dir: Path,
    config_path: Path | None = None,
    raw_path: Path | None = None,
) -> CriteoSplit:
    """Check the frozen config against the data, split, model settings and protocol constants.

    Reads no label or feature row of the held-out set. ``config_path`` (if given) must be tracked
    and unmodified in git; ``raw_path`` (if given) is re-hashed against the frozen checksum.
    """
    for key in ("selected", "selection", "dataset", "split", "evaluation", "test_set_accessed"):
        if key not in cfg:
            raise FrozenConfigError(f"frozen config is missing '{key}'")
    if cfg["test_set_accessed"]:
        raise FrozenConfigError("frozen config says the test set was already accessed")
    sel, ev = cfg["selected"], cfg["evaluation"]
    name = sel["name"]
    if name not in cb.MODEL_SPECS or sel["family"] != cb.MODEL_SPECS[name][0]:
        raise FrozenConfigError(f"unknown frozen model {name!r}")
    if sel["family"] == "tarnet":
        raise FrozenConfigError("C7 supports non-neural frozen models only (no torch in this path)")
    if sel["constructor_kwargs"] != cb.MODEL_SPECS[name][1]:
        raise FrozenConfigError("frozen constructor settings differ from the benchmark spec")

    fresh = cb.effective_hyperparameters(
        cb.build_model(sel["family"], sel["seed"], sel["constructor_kwargs"])
    )
    if json.loads(json.dumps(fresh, default=float)) != sel["hyperparameters"]:
        raise FrozenConfigError(
            "model hyperparameters in code differ from the frozen configuration"
        )
    if tuple(ev["budgets"]) != tuple(cb.BUDGETS) or ev["propensity"] != bp.P_TREAT:
        raise FrozenConfigError("evaluation budgets or propensity differ from the frozen protocol")

    manifest = json.loads((Path(data_dir) / "manifest.json").read_text())
    if manifest.get("source_sha256") != cfg["dataset"]["source_sha256"]:
        raise FrozenConfigError("dataset checksum does not match the frozen configuration")
    if raw_path is not None and sha256_file(Path(raw_path)) != cfg["dataset"]["source_sha256"]:
        raise FrozenConfigError("raw dataset file does not match the frozen checksum")
    split = load_split(Path(data_dir))
    if split.fingerprint() != cfg["split"]["fingerprint"]:
        raise FrozenConfigError("split fingerprint does not match the frozen configuration")
    if len(split.assignment) != cfg["dataset"]["rows"]:
        raise FrozenConfigError("dataset row count does not match the frozen configuration")
    if len(split.train_idx) != sel["training_rows"]:
        raise FrozenConfigError("training split size does not match the frozen configuration")
    if config_path is not None and not _git_ok(Path(config_path)):
        raise FrozenConfigError("frozen_config.json is not committed or was modified since")
    return split


# ---------------------------------------------------------------- worker processes


def _fit_key(cfg_sha: str, target: str) -> str:
    return cb.config_key({"frozen": cfg_sha, "target": target})


def run_fit_worker(args) -> None:
    """Fit the frozen model and its AIPW nuisance on *training rows only*, then self-check."""
    from upliftiq.modeling.criteo_learners import make_model

    data_dir, wd = Path(args.data_dir), work_dir(args.data_dir)
    cfg, cfg_sha = load_frozen(Path(args.frozen))
    sel, ev = cfg["selected"], cfg["evaluation"]
    split = load_split(data_dir)
    _, labels = load_arrays(data_dir)
    treatment = np.asarray(labels["treatment"])
    train_rows = split.train_idx
    assert len(train_rows) == sel["training_rows"]
    Xt, tt, yt = cb._arrays(data_dir, train_rows)
    yt = np.asarray(labels[args.target][train_rows])

    t0 = time.perf_counter()
    model = cb.build_model(sel["family"], sel["seed"], sel["constructor_kwargs"])
    model.fit(Xt, tt, yt)
    fit_seconds = time.perf_counter() - t0
    del Xt, tt, yt

    nu = ev["nuisance"]
    nrows = cb.nuisance_train_rows(
        split, treatment, nu["train_rows"], nu["disjoint_from_train_subsets_up_to"], sel["seed"]
    )
    fingerprint = hashlib.sha256(nrows.tobytes()).hexdigest()[:16]
    if fingerprint != nu["train_rows_fingerprint"]:
        raise FrozenConfigError("nuisance training rows differ from the frozen configuration")
    assert np.isin(nrows, split.train_idx).all()  # training rows only: never validation or test
    Xn, tn, _ = cb._arrays(data_dir, nrows)
    yn = np.asarray(labels[args.target][nrows])
    nuisance = make_model("t_learner", seed=sel["seed"]).fit(Xn, tn, yn)

    wd.mkdir(parents=True, exist_ok=True)
    with (wd / f"{args.target}.pkl").open("wb") as f:
        pickle.dump({"model": model, "nuisance": nuisance}, f)

    val_rows = cb.validation_subset(split, treatment, ev["validation_subset"]["rows"], sel["seed"])
    assert np.isin(val_rows, split.val_idx).all()
    Xv, _, _ = cb._arrays(data_dir, val_rows)
    uplift = np.asarray(model.predict_uplift(Xv), dtype=np.float32)
    nuis = nuisance.predict_outcomes(Xv).astype(np.float32)
    np.save(wd / f"{args.target}_val_uplift.npy", uplift)
    np.save(wd / f"{args.target}_val_nuisance.npy", nuis)
    _json(
        wd / f"{args.target}_fit.json",
        {
            "key": _fit_key(cfg_sha, args.target),
            "target": args.target,
            "frozen_config_sha256": cfg_sha,
            "fit_seconds": fit_seconds,
            "n_train": len(train_rows),
            "n_nuisance": len(nrows),
            "nuisance_rows_fingerprint": fingerprint,
            "validation_uplift_sha256": sha256_file(wd / f"{args.target}_val_uplift.npy"),
            "validation_nuisance_sha256": sha256_file(wd / f"{args.target}_val_nuisance.npy"),
            "libraries": cb.library_versions(),
            "code": cb.code_state(),
        },
    )


def run_score_worker(args) -> None:
    """Predict a holdout's uplift and nuisance from its features only; labels are never opened."""
    data_dir, wd = Path(args.data_dir), work_dir(args.data_dir)
    if args.holdout == "test":
        intent = Path(args.intent) if args.intent else None
        state = json.loads(intent.read_text()).get("status") if intent and intent.exists() else None
        if state != "started":
            raise FrozenConfigError("test scoring requires the one-shot record written first")
    split = load_split(data_dir)
    rows = split.test_idx if args.holdout == "test" else split.val_idx
    X, _ = load_arrays(data_dir)  # label arrays are memory-mapped but never indexed
    Xh = np.asarray(X[rows], dtype=np.float32)
    for target in TARGETS:
        with (wd / f"{target}.pkl").open("rb") as f:
            fitted = pickle.load(f)
        np.save(
            wd / f"{args.holdout}_{target}_uplift.npy",
            np.asarray(fitted["model"].predict_uplift(Xh), dtype=np.float32),
        )
        np.save(
            wd / f"{args.holdout}_{target}_nuisance.npy",
            fitted["nuisance"].predict_outcomes(Xh).astype(np.float32),
        )


def _spawn(args: list[str]) -> None:
    """Run a worker in a fresh interpreter with OMP_NUM_THREADS=1 set before Python starts."""
    import os
    import subprocess
    import sys

    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    src = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "upliftiq.evaluation.criteo_heldout", *args]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"worker failed ({' '.join(args[:3])}):\n{result.stderr[-2000:]}")


def _module_args(sub: str, **kw) -> list[str]:
    out = [sub]
    for k, v in kw.items():
        out += [f"--{k.replace('_', '-')}", str(v)]
    return out


# ---------------------------------------------------------------- reproducibility


def ensure_fitted(data_dir: Path, frozen_path: Path, cfg_sha: str, force: bool = False) -> None:
    wd = work_dir(data_dir)
    for target in TARGETS:
        meta = wd / f"{target}_fit.json"
        fresh = (
            meta.exists()
            and json.loads(meta.read_text()).get("key") == _fit_key(cfg_sha, target)
            and (wd / f"{target}.pkl").exists()
        )
        if force or not fresh:
            _spawn(_module_args("fit", data_dir=data_dir, frozen=frozen_path, target=target))


def check_reproducibility(data_dir: Path, cfg: dict, cfg_sha: str) -> dict:
    """The refit must reproduce the stored validation predictions bit for bit (SHA-256)."""
    wd = work_dir(data_dir)
    visit = json.loads((wd / "visit_fit.json").read_text())
    if visit["key"] != _fit_key(cfg_sha, "visit"):
        raise ReproducibilityError("fit artifacts belong to a different frozen configuration")
    want = cfg["selected"]["validation_uplift_sha256"]
    if visit["validation_uplift_sha256"] != want:
        raise ReproducibilityError(
            "refit validation predictions do not match the frozen hash; refusing to continue"
        )
    nu = cfg["evaluation"]["nuisance"]
    cached = (
        Path(data_dir)
        / "benchmark"
        / (f"nuisance_n{nu['train_rows']}_val{cfg['evaluation']['validation_subset']['rows']}.npy")
    )
    nuisance_ok = visit["validation_nuisance_sha256"] == sha256_file(cached)
    if not nuisance_ok:
        raise ReproducibilityError("refit AIPW nuisance does not match the validated nuisance")
    return {
        "validation_uplift_sha256": visit["validation_uplift_sha256"],
        "matches_frozen_hash": True,
        "validation_nuisance_matches_cached": True,
        "fit_seconds": {
            t: json.loads((wd / f"{t}_fit.json").read_text())["fit_seconds"] for t in TARGETS
        },
    }


# ---------------------------------------------------------------- evaluation core


def _nuisance(arr: np.ndarray, rows: np.ndarray, nuisance_train_rows: np.ndarray):
    return OutcomeModelPredictions.from_holdout(
        arr.astype(np.float64), rows, nuisance_train_rows, n_arms=2
    )


def evaluate_holdout(
    rows, treatment, visit, conversion, scores, nuisance_arrays, nuisance_rows, cfg, protected_ids
) -> dict:
    """Score the frozen visit ranking (primary) and the exploratory conversion ranking on a holdout.

    The same function runs in the rehearsal (validation rows) and the final evaluation (test rows).
    """
    ev = cfg["evaluation"]
    common = {
        "budgets": tuple(ev["budgets"]),
        "n_boot": ev["n_boot"],
        "seed": cfg["selected"]["seed"],
        "row_ids": rows,
        "protected_ids": protected_ids,
        "p_treat": ev["propensity"],
    }
    name = cfg["selected"]["name"]
    nu_v = _nuisance(nuisance_arrays["visit"], rows, nuisance_rows)
    primary = bp.evaluate_targeting(visit, treatment, {name: scores["visit"]}, nu_v, **common)
    nu_c = _nuisance(nuisance_arrays["conversion"], rows, nuisance_rows)
    exploratory = bp.evaluate_targeting(
        conversion,
        treatment,
        {"visit_policy": scores["visit"], "conversion_model": scores["conversion"]},
        nu_c,
        **common,
    )
    return {
        "n": len(rows),
        "name": name,
        "primary": primary,
        "exploratory": exploratory,
        "assumption_check": cb.assumption_check(visit, treatment, nu_v),
        "assumption_check_conversion": cb.assumption_check(conversion, treatment, nu_c),
    }


def _pol(res, model, estimator, b):
    p = res.policies
    return p[
        (p.model == model)
        & (p.estimator == estimator)
        & (p.policy == "budget")
        & (p.budget_requested == b)
    ].iloc[0]


def build_tables(result: dict, cfg: dict) -> dict[str, pd.DataFrame]:
    name, res, ex = result["name"], result["primary"], result["exploratory"]
    budgets = cfg["evaluation"]["budgets"]
    value, ref_rows = [], []
    for b in budgets:
        a, i = _pol(res, name, "AIPW", b), _pol(res, name, "IPW", b)
        value.append(
            {
                "budget": f"{b:.0%}",
                "customers treated (share)": f"{a.n_treated / result['n']:.1%}",
                "AIPW value (pp)": cb._ci(a.value, a.ci_low, a.ci_high),
                "random at same budget (pp)": f"{a.random_value * 100:.3f}",
                "value minus random (pp)": cb._ci(
                    a.diff_vs_random, a.diff_vs_random_ci_low, a.diff_vs_random_ci_high
                ),
                "excludes 0": bool(a.diff_vs_random_ci_low > 0 or a.diff_vs_random_ci_high < 0),
                "IPW value (pp, cross-check)": f"{i.value * 100:.3f}",
                "primary": "yes" if abs(b - 0.2) < 1e-9 else "",
            }
        )
    r = res.references
    r = r[(r.estimator == "AIPW")]
    for q in r.itertuples():
        if q.policy == "treat_nobody":
            kind = "zero baseline"
        elif q.feasibility.startswith("feasible ("):
            kind = "budget-feasible"
        else:
            kind = "unconstrained reference"
        ref_rows.append(
            {
                "policy": q.policy,
                "budget": f"{q.budget:.0%}",
                "AIPW value (pp)": cb._ci(q.value, q.ci_low, q.ci_high),
                "type": kind,
            }
        )
    qa = res.qini[(res.qini.estimator == "AIPW")].iloc[0]
    qi = res.qini[(res.qini.estimator == "IPW")].iloc[0]
    qini = pd.DataFrame(
        [
            {
                "estimator": "AIPW (primary)",
                "Qini (pp)": cb._ci(qa.qini, qa.ci_low, qa.ci_high),
                "Qini / treat-all effect": round(qa.qini / qa.ate_treat_all, 3),
            },
            {
                "estimator": "IPW (cross-check)",
                "Qini (pp)": f"{qi.qini * 100:.3f}",
                "Qini / treat-all effect": round(qi.qini / qi.ate_treat_all, 3),
            },
        ]
    )
    conv = []
    for b in budgets:
        v = _pol(ex, "visit_policy", "AIPW", b)
        c = _pol(ex, "conversion_model", "AIPW", b)
        conv.append(
            {
                "budget": f"{b:.0%}",
                "frozen visit policy: conversion value (pp)": cb._ci(
                    v.value, v.ci_low, v.ci_high, digits=4
                ),
                "conversion-model ranking (pp)": cb._ci(c.value, c.ci_low, c.ci_high, digits=4),
                "random (pp)": f"{v.random_value * 100:.4f}",
            }
        )
    ta = ex.references[
        (ex.references.estimator == "AIPW") & (ex.references.policy == "treat_all")
    ].iloc[0]
    conv.append(
        {
            "budget": "100% (treat all, unconstrained)",
            "frozen visit policy: conversion value (pp)": cb._ci(
                ta.value, ta.ci_low, ta.ci_high, digits=4
            ),
            "conversion-model ranking (pp)": "",
            "random (pp)": "",
        }
    )
    return {
        "value": pd.DataFrame(value),
        "references": pd.DataFrame(ref_rows),
        "qini": qini,
        "conversion": pd.DataFrame(conv),
    }


def _figures(result: dict, cfg: dict, tables: dict, out_dir: Path) -> None:
    name, res = result["name"], result["primary"]
    c = res.curves[res.curves.model == name]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    a, i = c[c.estimator == "AIPW"], c[c.estimator == "IPW"]
    ax.plot(a.budget * 100, a.gain * 100, lw=1.6, label="frozen S-learner, AIPW (primary)")
    ax.fill_between(a.budget * 100, a.gain_ci_low * 100, a.gain_ci_high * 100, alpha=0.2)
    ax.plot(i.budget * 100, i.gain * 100, "--", lw=1, label="same ranking, IPW (cross-check)")
    ax.plot(a.budget * 100, a.random_line * 100, "k:", lw=1, label="random targeting (AIPW)")
    for b in cfg["evaluation"]["budgets"]:
        ax.axvline(b * 100, color="grey", lw=0.5)
    ax.set_xlabel("contact budget (% of customers treated)")
    ax.set_ylabel("incremental visit rate vs treat nobody (pp)")
    ax.set_title("Held-out cumulative policy value (released-sample estimate)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "heldout_gain_curve.png", dpi=130)
    plt.close(fig)

    budgets = cfg["evaluation"]["budgets"]
    pts = [_pol(res, name, "AIPW", b) for b in budgets]
    x = np.arange(len(budgets))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.18, [p.value * 100 for p in pts], 0.36, label="frozen S-learner",
           yerr=[[(p.value - p.ci_low) * 100 for p in pts], [(p.ci_high - p.value) * 100 for p in pts]], capsize=3)  # fmt: skip
    ax.bar(
        x + 0.18, [p.random_value * 100 for p in pts], 0.36, label="random targeting", color="grey"
    )
    ax.set_xticks(x, [f"{b:.0%}" for b in budgets])
    ax.set_xlabel("contact budget")
    ax.set_ylabel("AIPW incremental visit rate (pp)")
    ax.set_title("Held-out policy value vs random (95% intervals)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "heldout_value_vs_random.png", dpi=130)
    plt.close(fig)


def build_report(result: dict, cfg: dict, tables: dict, kind: str, extra: dict) -> str:
    md = lambda df: df.to_markdown(index=False)  # noqa: E731
    ac, acc = result["assumption_check"], result["assumption_check_conversion"]
    z = ac["predicted_baseline_gap"] / ac["predicted_baseline_gap_se"]
    res = result["primary"]
    ate = res.references[
        (res.references.estimator == "AIPW") & (res.references.policy == "treat_all")
    ].iloc[0]
    sel = cfg["selected"]
    if kind == "final":
        banner = (
            "# Criteo final held-out evaluation (one-shot)\n\n"
            f"The frozen `{sel['name']}` was evaluated **once** on the untouched held-out test set "
            f"({result['n']:,} customers). No other model was scored on it, nothing was retuned, and "
            "these results were not used to change the model, hyperparameters, selection rule or policy."
        )
    else:
        banner = (
            "# Criteo C7 development-only rehearsal (NOT test results)\n\n"
            f"This dry run uses the **validation split** ({result['n']:,} customers) as a stand-in "
            "holdout to exercise the exact final pipeline. The held-out test set was not accessed. "
            "Numbers below are validation numbers and carry no weight as final results."
        )
    checks = "\n".join(f"- {k}: {v}" for k, v in extra.get("checks", {}).items())
    return f"""{banner}

{LIMITATIONS}\n{FINAL_NOTE if kind == "final" else ""}

Units: percentage points (pp) of visit rate per evaluated customer, relative to treating nobody.
Budgets are shares of customers contacted, not monetary cost.

## Setup and verification
- Frozen model: `{sel["name"]}` (LightGBM, effective hyperparameters in `frozen_config.json`), seed {sel["seed"]}, trained on the full training split ({sel["training_rows"]:,} rows).
- Dataset sha256 `{cfg["dataset"]["source_sha256"][:16]}…`; split fingerprint `{cfg["split"]["fingerprint"][:16]}…`.
- AIPW nuisance: LightGBM T-learner, same configuration as validation, fitted on {cfg["evaluation"]["nuisance"]["train_rows"]:,} **training rows only** (fingerprint `{cfg["evaluation"]["nuisance"]["train_rows_fingerprint"]}`); no validation or test rows. The plan's wording (train+validation nuisance for test) was replaced by this training-only rule at the owner's direction.
- Bootstrap: {cfg["evaluation"]["n_boot"]} Poisson replicates, shared weights (paired); seed {sel["seed"]}.
{checks}

## Primary result: AIPW policy value by contact budget (20% is primary)
`value minus random` is the paired difference from random targeting at the same budget.

{md(tables["value"])}

## Reference policies
Budget-feasible policies respect the contact budget; treat-all is an unconstrained reference
(feasible only at a 100% budget) and treat-nobody is the zero baseline.

{md(tables["references"])}

Treat-all effect (AIPW): {ate.value * 100:.3f} pp [{ate.ci_low * 100:.3f}, {ate.ci_high * 100:.3f}].

## Qini / AUUC
Area between the cumulative-gain curve and the random line (100-point trapezoid).

{md(tables["qini"])}

## IPW vs AIPW (known limitation, preserved)
| quantity | value |
|:--|--:|
| treated share | {ac["treated_share"]:.4f} |
| treat-all effect, IPW (treated minus control mean), pp | {ac["ate_ipw"] * 100:.3f} |
| treat-all effect, AIPW, pp | {ac["ate_aipw"] * 100:.3f} |
| IPW minus AIPW, pp (SE) | {ac["ate_ipw_minus_aipw"] * 100:.3f} ({ac["ate_ipw_minus_aipw_se"] * 100:.3f}) |
| predicted baseline, treated vs control, pp | {ac["predicted_baseline_treated"] * 100:.3f} vs {ac["predicted_baseline_control"] * 100:.3f} (z = {z:.1f}) |

The IPW column in the table above is higher than AIPW at every budget for the same reason as on
validation: the arms differ in predicted baseline, which the constant-propensity estimator ignores.

## Exploratory: conversion uplift (not used for any decision)
Conversion is very rare, so intervals are wide and the conversion nuisance (trained on the same
{cfg["evaluation"]["nuisance"]["train_rows"]:,} rows) is weak. Rows show the **frozen visit policy**
(unchanged) scored on conversion, and, for description only, a conversion-trained S-learner of the
same family and fixed hyperparameters. Neither affects the selected model or targeting policy.
Conversion IPW vs AIPW treat-all: {acc["ate_ipw"] * 100:.4f} vs {acc["ate_aipw"] * 100:.4f} pp.

{md(tables["conversion"])}

Figures: `heldout_gain_curve.png`, `heldout_value_vs_random.png`.

## Caveats
- One seed, one split, one fixed configuration; intervals reflect scoring noise on this sample, not refit variability.
- Released-sample estimates only: no claim about advertising ROI, cost, or any customer population outside this file.
"""


def write_artifacts(result: dict, cfg: dict, out_dir: Path, kind: str, extra: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = build_tables(result, cfg)
    _figures(result, cfg, tables, out_dir)
    for k, df in tables.items():
        df.to_csv(out_dir / f"heldout_{k}.csv", index=False)
    (out_dir / "final_report.md").write_text(build_report(result, cfg, tables, kind, extra))
    payload = {
        "kind": kind,
        "n": result["n"],
        "model": result["name"],
        "tables": {k: v.to_dict("records") for k, v in tables.items()},
        "policies": result["primary"].policies.to_dict("records"),
        "references": result["primary"].references.to_dict("records"),
        "qini": result["primary"].qini.to_dict("records"),
        "exploratory": {
            "policies": result["exploratory"].policies.to_dict("records"),
            "references": result["exploratory"].references.to_dict("records"),
            "qini": result["exploratory"].qini.to_dict("records"),
            "note": "exploratory only; never used to select a model or policy",
        },
        "assumption_check": result["assumption_check"],
        "assumption_check_conversion": result["assumption_check_conversion"],
        "meta": result["primary"].meta,
        **extra,
    }
    _json(out_dir / "final_results.json", payload)
    return payload


# ---------------------------------------------------------------- drivers


def _load_scores(wd: Path, holdout: str):
    s = {t: np.load(wd / f"{holdout}_{t}_uplift.npy") for t in TARGETS}
    n = {t: np.load(wd / f"{holdout}_{t}_nuisance.npy") for t in TARGETS}
    return s, n


def _nuisance_rows(split, treatment, cfg):
    nu = cfg["evaluation"]["nuisance"]
    return cb.nuisance_train_rows(
        split,
        treatment,
        nu["train_rows"],
        nu["disjoint_from_train_subsets_up_to"],
        cfg["selected"]["seed"],
    )


def run_rehearsal(
    data_dir=PROCESSED_DIR,
    frozen_path=FROZEN_PATH,
    out_dir=None,
    check_git=True,
    raw_path=None,
    force=False,
) -> dict:
    """Development-only dry run on the validation split; test rows and labels are never opened."""
    data_dir, out_dir = Path(data_dir), Path(out_dir or FINAL_DIR)
    cfg, sha = load_frozen(frozen_path)
    split = verify_frozen(cfg, data_dir, frozen_path if check_git else None, raw_path)
    ensure_fitted(data_dir, Path(frozen_path), sha, force)
    repro = check_reproducibility(data_dir, cfg, sha)
    wd = work_dir(data_dir)
    _spawn(_module_args("score", data_dir=data_dir, holdout="validation"))
    _, labels = load_arrays(data_dir)
    rows = split.val_idx
    treatment = np.asarray(labels["treatment"][rows])
    visit = np.asarray(labels["visit"][rows], dtype=np.float64)
    conv = np.asarray(labels["conversion"][rows], dtype=np.float64)
    scores, nuis = _load_scores(wd, "validation")
    nrows = _nuisance_rows(split, np.asarray(labels["treatment"]), cfg)
    result = evaluate_holdout(
        rows, treatment, visit, conv, scores, nuis, nrows, cfg, split.test_idx
    )

    # the frozen 500K validation number must be re-derived by this very pipeline
    sel = cfg["selected"]
    sub = cb.validation_subset(
        split,
        np.asarray(labels["treatment"]),
        cfg["evaluation"]["validation_subset"]["rows"],
        sel["seed"],
    )
    pos = np.searchsorted(rows, sub)
    assert np.array_equal(rows[pos], sub)
    sub_res = evaluate_holdout(
        sub, treatment[pos], visit[pos], conv[pos],
        {k: v[pos] for k, v in scores.items()}, {k: v[pos] for k, v in nuis.items()},
        nrows, cfg, split.test_idx,
    )  # fmt: skip
    got = float(_pol(sub_res["primary"], sel["name"], "AIPW", 0.2).value)
    want = cfg["selection"]["candidates"][sel["name"]]["value"]
    if abs(got - want) > VALUE_TOL:
        raise ReproducibilityError(
            f"pipeline did not reproduce the frozen value@20%: {got} vs {want}"
        )

    checks = {
        "frozen config, dataset checksum, split fingerprint, hyperparameters, protocol": "verified",
        "refit validation-score SHA-256 equals frozen hash": repro["matches_frozen_hash"],
        "refit AIPW nuisance equals the validated nuisance (SHA-256)": repro[
            "validation_nuisance_matches_cached"
        ],
        "pipeline reproduces frozen validation AIPW value@20% (500K subset)": f"{got * 100:.6f} pp (frozen {want * 100:.6f} pp)",
        "test rows opened": "no (workers scored the validation split only)",
    }
    extra = {"checks": checks, "reproducibility": repro, "frozen_config_sha256": sha}
    write_artifacts(result, cfg, out_dir / "rehearsal", "rehearsal", extra)
    record = {
        "status": "passed",
        "frozen_config_sha256": sha,
        "checks": checks,
        "rehearsal_holdout": "validation split (development only)",
        "test_set_accessed": False,
    }
    _json(out_dir / REHEARSAL_NAME, record)
    return record


def run_final(
    data_dir=PROCESSED_DIR,
    frozen_path=FROZEN_PATH,
    out_dir=None,
    confirm=False,
    check_git=True,
    raw_path=None,
) -> dict:
    """The one-shot evaluation on the held-out test set. Refuses to run twice."""
    data_dir, out_dir = Path(data_dir), Path(out_dir or FINAL_DIR)
    if not confirm:
        raise FrozenConfigError("the one-shot test evaluation needs explicit confirmation")
    cfg, sha = load_frozen(frozen_path)
    record_path = out_dir / RECORD_NAME
    if record_path.exists():
        raise FrozenConfigError(
            "a one-shot evaluation record already exists; the held-out evaluation must not be re-run"
        )
    split = verify_frozen(cfg, data_dir, frozen_path if check_git else None, raw_path)
    reh = out_dir / REHEARSAL_NAME
    if not reh.exists() or json.loads(reh.read_text()).get("frozen_config_sha256") != sha:
        raise FrozenConfigError(
            "a passing development rehearsal for this frozen config is required first"
        )
    if json.loads(reh.read_text())["status"] != "passed":
        raise FrozenConfigError("the development rehearsal did not pass")
    repro = check_reproducibility(data_dir, cfg, sha)  # fit artifacts re-verified, no test access
    wd = work_dir(data_dir)

    started = datetime.now(UTC).isoformat(timespec="seconds")
    head = cb.code_state()
    record = {
        "status": "started",
        "frozen_config_sha256": sha,
        "selected_model": cfg["selected"]["name"],
        "started_at": started,
        "code_commit": head["commit"],
        "code_dirty": head["dirty"],
        "one_shot": True,
    }
    _json(record_path, record)  # written BEFORE any test row is read
    try:
        _spawn(_module_args("score", data_dir=data_dir, holdout="test", intent=record_path))
        _, labels = load_arrays(data_dir)  # test labels are first indexed here
        rows = split.test_idx
        t_all = np.asarray(labels["treatment"])
        treatment = t_all[rows]
        visit = np.asarray(labels["visit"][rows], dtype=np.float64)
        conv = np.asarray(labels["conversion"][rows], dtype=np.float64)
        scores, nuis = _load_scores(wd, "test")
        nrows = _nuisance_rows(split, t_all, cfg)
        result = evaluate_holdout(rows, treatment, visit, conv, scores, nuis, nrows, cfg, None)
        extra = {
            "checks": {
                "frozen config, dataset checksum, split fingerprint, hyperparameters, protocol": "verified before test access",
                "refit validation-score SHA-256 equals frozen hash": repro["matches_frozen_hash"],
                "development rehearsal": "passed before this evaluation",
                "models scored on the test set": cfg["selected"]["name"] + " only",
            },
            "frozen_config_sha256": sha,
        }
        payload = write_artifacts(result, cfg, out_dir, "final", extra)
        record.update(
            {
                "status": "completed",
                "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "test_rows": len(rows),
                "test_ids_sha256": _bytes_sha(rows),
                "test_uplift_sha256": sha256_file(wd / "test_visit_uplift.npy"),
                "dataset_sha256": cfg["dataset"]["source_sha256"],
                "split_fingerprint": split.fingerprint(),
                "primary_metric": "AIPW policy value at 20% contact budget",
                "n_boot": cfg["evaluation"]["n_boot"],
                "seed": cfg["selected"]["seed"],
                "nuisance_fitted_on": "training rows only",
                "models_evaluated_on_test": [cfg["selected"]["name"]],
                "conversion": "exploratory only; no effect on model or policy",
                "libraries": cb.library_versions(),
                "results_sha256": hashlib.sha256(
                    (out_dir / "final_results.json").read_bytes()
                ).hexdigest(),
                "headline_value_at_20pct_pp": float(
                    _pol(result["primary"], result["name"], "AIPW", 0.2).value * 100
                ),
            }
        )
        _json(record_path, record)
        return payload
    except BaseException as e:
        record.update({"status": "failed", "error": repr(e)})
        _json(record_path, record)
        raise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Criteo C7 held-out evaluation.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("verify", "rehearse", "final"):
        p = sub.add_parser(cmd)
        p.add_argument("--data-dir", default=str(PROCESSED_DIR))
        p.add_argument("--frozen", default=str(FROZEN_PATH))
        p.add_argument("--out-dir", default=str(FINAL_DIR))
        if cmd == "final":
            p.add_argument("--confirm-one-shot", action="store_true")
    f = sub.add_parser("fit")
    s = sub.add_parser("score")
    for p in (f, s):
        p.add_argument("--data-dir", required=True)
    f.add_argument("--frozen", required=True)
    f.add_argument("--target", required=True, choices=TARGETS)
    s.add_argument("--holdout", required=True, choices=("validation", "test"))
    s.add_argument("--intent", default=None)
    a = ap.parse_args(argv)
    raw = RAW_PATH if RAW_PATH.exists() else None
    if a.cmd == "fit":
        run_fit_worker(a)
    elif a.cmd == "score":
        run_score_worker(a)
    elif a.cmd == "verify":
        cfg, sha = load_frozen(Path(a.frozen))
        verify_frozen(cfg, Path(a.data_dir), Path(a.frozen), raw)
        print("frozen configuration verified:", sha)
    elif a.cmd == "rehearse":
        print(json.dumps(run_rehearsal(a.data_dir, a.frozen, a.out_dir, raw_path=raw), indent=2))
    else:
        run_final(a.data_dir, a.frozen, a.out_dir, a.confirm_one_shot, raw_path=raw)
        print("final evaluation written to", a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
