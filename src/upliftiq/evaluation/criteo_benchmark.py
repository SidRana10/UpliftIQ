"""Criteo C5: preliminary validation benchmark of the five uplift models.

Pipeline (one rung = one training-subset size):

1. A *nuisance* process trains one outcome model (LightGBM T-learner) on train rows that are
   disjoint from every candidate's training subset, and predicts mu0/mu1 for the validation subset.
   All models are scored with this same out-of-sample AIPW nuisance (cached, built once).
2. One isolated *worker* process per model (and TARNet device) loads the prepared arrays, fits on
   the identical nested, treatment-stratified training subset, predicts the fixed validation
   subset, and writes uplift scores plus a manifest (config, fingerprints, library versions,
   timings, latency percentiles, peak memory). Workers start with ``OMP_NUM_THREADS=1`` set before
   Python launches, which avoids the macOS OpenMP clash between PyTorch and LightGBM.
3. The driver (numpy only: it never imports LightGBM or PyTorch) evaluates all scores jointly with
   the C2 evaluator (shared Poisson bootstrap weights, so model differences are paired) and writes
   tables, figures and a report.

Only train and validation rows are touched; the test rows are passed to the evaluator as protected
ids. Results are preliminary validation benchmarks, not held-out results, and no model is selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from upliftiq.data.criteo_prepare import (  # noqa: E402
    PROCESSED_DIR,
    REPORT_DIR,
    SEED,
    CriteoSplit,
    load_arrays,
    load_split,
    peak_rss_gb,
    train_subset,
)
from upliftiq.data.registry import PROJECT_ROOT  # noqa: E402
from upliftiq.evaluation import binary_policy as bp  # noqa: E402
from upliftiq.evaluation.policy import OutcomeModelPredictions  # noqa: E402

BUDGETS = (0.10, 0.20, 0.30)
LADDER = (100_000, 500_000)
VAL_ROWS = 500_000
NUISANCE_ROWS = 500_000
NUISANCE_EXCLUDE = 2_000_000  # nuisance rows avoid the largest planned candidate subset (S2)
LATENCY_PLAN = ((1, 200), (1_000, 30), (100_000, 3))  # (batch size, repetitions)
N_BOOT = 200

# name -> (family, constructor kwargs); everything else uses the C3/C4 defaults (no tuning).
MODEL_SPECS: dict[str, tuple[str, dict]] = {
    "logistic": ("logistic", {}),
    "s_learner": ("s_learner", {}),
    "t_learner": ("t_learner", {}),
    "dr_learner": ("dr_learner", {}),
    "tarnet_cpu": ("tarnet", {"device": "cpu"}),
    "tarnet_mps": ("tarnet", {"device": "mps"}),
}
DEFAULT_MODELS = list(MODEL_SPECS)


# ---------------------------------------------------------------- shared helpers


def stratified_subset(idx: np.ndarray, treatment: np.ndarray, n: int, seed: int = SEED):
    """Treatment-stratified random subset of ``idx`` (sorted); all of ``idx`` if n is large."""
    if n >= len(idx):
        return np.sort(idx)
    rng = np.random.default_rng(seed)
    chosen = []
    for t in (0, 1):
        rows = idx[np.asarray(treatment)[idx] == t]
        chosen.append(rng.permutation(rows)[: round(n * len(rows) / len(idx))])
    return np.sort(np.concatenate(chosen))


def validation_subset(split: CriteoSplit, treatment, n: int, seed: int = SEED) -> np.ndarray:
    return stratified_subset(split.val_idx, treatment, n, seed)


def nuisance_train_rows(split: CriteoSplit, treatment, n: int, exclude_n: int, seed: int = SEED):
    """Train rows disjoint from the candidates' (nested) training subsets, for the nuisance model."""
    excluded = train_subset(split, treatment, exclude_n, seed)
    pool = np.setdiff1d(split.train_idx, excluded)
    return stratified_subset(pool, treatment, n, seed + 1)


def latency_stats(predict, X: np.ndarray, plan=LATENCY_PLAN) -> list[dict]:
    """p50/p95 latency (ms) and throughput (rows/s) per batch size; ``predict`` maps rows -> scores."""
    out = []
    for batch, reps in plan:
        if batch * 1 > len(X):
            continue
        predict(X[:batch])  # warm-up
        times = []
        for r in range(reps):
            start = (r * batch) % max(len(X) - batch, 1)
            rows = X[start : start + batch]
            t0 = time.perf_counter()
            predict(rows)
            times.append(time.perf_counter() - t0)
        times = np.asarray(times)
        out.append(
            {
                "batch": batch,
                "reps": reps,
                "p50_ms": float(np.percentile(times, 50) * 1e3),
                "p95_ms": float(np.percentile(times, 95) * 1e3),
                "rows_per_s": float(batch / np.median(times)),
            }
        )
    return out


def current_rss_gb() -> float:
    """Resident memory of this process right now (ru_maxrss only gives the high-water mark)."""
    r = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True)
    return int(r.stdout.strip() or 0) / 1024**2


class RssSampler:
    """Polls the process's current RSS in a thread to get a peak for one phase (e.g. fit)."""

    def __init__(self, interval: float = 0.05):
        self.interval, self.peak_gb = interval, 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak_gb = max(self.peak_gb, current_rss_gb())
            self._stop.wait(self.interval)

    def __enter__(self):
        self.peak_gb = current_rss_gb()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak_gb = max(self.peak_gb, current_rss_gb())


def config_key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[
        :16
    ]


def code_state() -> dict:
    def git(*args):
        r = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def library_versions() -> dict:
    import lightgbm
    import sklearn

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "platform": platform.platform(),
    }
    if "torch" in sys.modules:
        versions["torch"] = sys.modules["torch"].__version__
    return versions


def data_provenance(data_dir: Path, split: CriteoSplit) -> dict:
    manifest = data_dir / "manifest.json"
    sha = json.loads(manifest.read_text()).get("source_sha256") if manifest.exists() else None
    return {"split_fingerprint": split.fingerprint(), "split_seed": split.seed, "data_sha256": sha}


def rung_dir(data_dir: Path, size: int, n_val: int) -> Path:
    return data_dir / "benchmark" / f"fit{size}_val{n_val}"


def _json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float) + "\n")


# ---------------------------------------------------------------- worker processes


def _arrays(data_dir: Path, rows: np.ndarray):
    X, labels = load_arrays(data_dir)
    return (
        np.asarray(X[rows], dtype=np.float32),
        np.asarray(labels["treatment"][rows]),
        np.asarray(labels["visit"][rows]),
    )


def build_model(family: str, seed: int, kwargs: dict):
    if family == "tarnet":
        from upliftiq.modeling.criteo_tarnet import TARNet

        return TARNet(seed=seed, **kwargs)
    from upliftiq.modeling.criteo_learners import make_model

    return make_model(family, seed=seed, **kwargs)


def run_model_worker(args) -> None:
    data_dir = Path(args.data_dir)
    split, spec = load_split(data_dir), json.loads(args.spec)
    _, labels = load_arrays(data_dir)
    treatment = np.asarray(labels["treatment"])
    fit_rows = train_subset(split, treatment, args.size, args.seed)
    val_rows = validation_subset(split, treatment, args.n_val, args.seed)
    assert np.isin(fit_rows, split.train_idx).all() and np.isin(val_rows, split.val_idx).all()
    Xf, tf, yf = _arrays(data_dir, fit_rows)
    Xv, _, _ = _arrays(data_dir, val_rows)
    del labels
    rss_loaded = current_rss_gb()

    model = build_model(spec["family"], args.seed, spec["kwargs"])
    t0 = time.perf_counter()
    with RssSampler() as sampler:
        model.fit(Xf, tf, yf)
    fit_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    uplift = np.asarray(model.predict_uplift(Xv), dtype=np.float32)
    predict_seconds = time.perf_counter() - t0
    latency = latency_stats(model.predict_uplift, Xv)

    out = Path(args.out_dir)
    np.save(out / f"{args.name}_uplift.npy", uplift)
    manifest = {
        "model": args.name,
        "family": spec["family"],
        "config_key": args.config_key,
        "kwargs": spec["kwargs"],
        "seed": args.seed,
        "n_fit": len(fit_rows),
        "n_val": len(val_rows),
        "provenance": data_provenance(data_dir, split),
        "fit_seconds": fit_seconds,
        "predict_val_seconds": predict_seconds,
        "latency": latency,
        "rss_gb_before_fit": rss_loaded,
        "rss_gb_peak_during_fit": sampler.peak_gb,
        "peak_rss_gb": peak_rss_gb(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "libraries": library_versions(),
        "code": code_state(),
        "uplift_mean": float(uplift.mean()),
        "uplift_std": float(uplift.std()),
    }
    if hasattr(model, "training_summary"):
        manifest["training"] = model.training_summary()
        manifest["training"]["history"] = model.history_
    _json(out / f"{args.name}.json", manifest)


def run_nuisance_worker(args) -> None:
    from upliftiq.modeling.criteo_learners import make_model

    data_dir = Path(args.data_dir)
    split = load_split(data_dir)
    _, labels = load_arrays(data_dir)
    treatment = np.asarray(labels["treatment"])
    train_rows = nuisance_train_rows(split, treatment, args.size, args.exclude, args.seed)
    val_rows = validation_subset(split, treatment, args.n_val, args.seed)
    Xn, tn, yn = _arrays(data_dir, train_rows)
    Xv, _, _ = _arrays(data_dir, val_rows)
    t0 = time.perf_counter()
    model = make_model("t_learner", seed=args.seed).fit(Xn, tn, yn)
    seconds = time.perf_counter() - t0
    out = Path(args.out_path)
    np.save(out.with_suffix(".npy"), model.predict_outcomes(Xv).astype(np.float32))
    _json(
        out.with_suffix(".json"),
        {
            "config_key": args.config_key,
            "model": "LightGBM T-learner (defaults)",
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "fit_seconds": seconds,
            "train_rows_fingerprint": hashlib.sha256(train_rows.tobytes()).hexdigest()[:16],
            "disjoint_from_candidate_subsets_up_to": args.exclude,
            "provenance": data_provenance(data_dir, split),
            "libraries": library_versions(),
            "code": code_state(),
        },
    )


# ---------------------------------------------------------------- driver


def _spawn(module_args: list[str]) -> None:
    """Run a worker in a fresh interpreter with OMP_NUM_THREADS=1 set before Python starts."""
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    src = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "upliftiq.evaluation.criteo_benchmark", *module_args]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"worker failed ({' '.join(module_args[:4])}):\n{result.stderr[-2000:]}")


def _fresh(path: Path, key: str) -> bool:
    return path.exists() and json.loads(path.read_text()).get("config_key") == key


def ensure_nuisance(
    data_dir, split, n_val, seed, force=False, size=NUISANCE_ROWS, exclude=NUISANCE_EXCLUDE
) -> Path:
    base = Path(data_dir) / "benchmark" / f"nuisance_n{size}_val{n_val}"
    base.parent.mkdir(parents=True, exist_ok=True)
    key = config_key(
        {"nuisance": size, "n_val": n_val, "seed": seed, **data_provenance(data_dir, split)}
    )
    if force or not (_fresh(base.with_suffix(".json"), key) and base.with_suffix(".npy").exists()):
        _spawn(
            [
                "nuisance",
                "--data-dir",
                str(data_dir),
                "--size",
                str(size),
                "--n-val",
                str(n_val),
                "--seed",
                str(seed),
                "--exclude",
                str(exclude),
                "--out-path",
                str(base),
                "--config-key",
                key,
            ]
        )
    return base


def ensure_model(name, spec, size, data_dir, split, n_val, seed, force=False) -> Path:
    out = rung_dir(Path(data_dir), size, n_val)
    out.mkdir(parents=True, exist_ok=True)
    key = config_key(
        {
            "name": name,
            **spec,
            "size": size,
            "n_val": n_val,
            "seed": seed,
            **data_provenance(data_dir, split),
        }
    )
    if force or not (_fresh(out / f"{name}.json", key) and (out / f"{name}_uplift.npy").exists()):
        _spawn(
            [
                "model",
                "--data-dir",
                str(data_dir),
                "--name",
                name,
                "--spec",
                json.dumps(spec),
                "--size",
                str(size),
                "--n-val",
                str(n_val),
                "--seed",
                str(seed),
                "--out-dir",
                str(out),
                "--config-key",
                key,
            ]
        )
    return out


def evaluate_rung(data_dir, split, size, n_val, names, nuisance_base, seed, n_boot) -> dict:
    """Score every model's validation uplift with the shared nuisance (numpy only)."""
    data_dir = Path(data_dir)
    out = rung_dir(data_dir, size, n_val)
    _, labels = load_arrays(data_dir)
    treatment = np.asarray(labels["treatment"])
    val_rows = validation_subset(split, treatment, n_val, seed)
    y = np.asarray(labels["visit"][val_rows], dtype=np.float64)
    t = treatment[val_rows]
    nuis_meta = json.loads(nuisance_base.with_suffix(".json").read_text())
    nuis_rows = nuisance_train_rows(
        split,
        treatment,
        nuis_meta["n_train"],
        nuis_meta["disjoint_from_candidate_subsets_up_to"],
        seed,
    )
    nuisance = OutcomeModelPredictions.from_holdout(
        np.load(nuisance_base.with_suffix(".npy")).astype(np.float64), val_rows, nuis_rows, n_arms=2
    )
    scores = {n: np.load(out / f"{n}_uplift.npy") for n in names}
    protected = split.test_idx
    results = {}
    refs = [names[0], *(["tarnet_cpu"] if "tarnet_cpu" in names[1:] else [])]
    for ref in refs:
        order = [ref, *[n for n in names if n != ref]]
        results[ref] = bp.evaluate_targeting(
            y,
            t,
            {n: scores[n] for n in order},
            nuisance,
            budgets=BUDGETS,
            n_boot=n_boot,
            seed=seed,
            row_ids=val_rows,
            protected_ids=protected,
        )
    manifests = {n: json.loads((out / f"{n}.json").read_text()) for n in names}
    check = assumption_check(y, t, nuisance)
    return {
        "results": results,
        "manifests": manifests,
        "nuisance": nuis_meta,
        "n": len(y),
        "assumption_check": check,
    }


def assumption_check(y, t, nuisance: OutcomeModelPredictions) -> dict:
    """IPW vs AIPW agreement and arm balance in predicted baseline (no propensity model fitted).

    With a correct constant propensity, IPW and AIPW estimate the same effect. A gap, together with
    different predicted baseline visit rates in the two arms, points to covariate imbalance that the
    constant-propensity IPW estimator ignores and the outcome-model adjustment in AIPW partly removes.
    """
    ipw, aipw = bp.incremental_scores(y, t), bp.incremental_scores(y, t, nuisance)
    mu0, treated = nuisance.mu[:, 0], t == 1
    gap = float(mu0[treated].mean() - mu0[~treated].mean())
    se = float(np.sqrt(mu0[treated].var() / treated.sum() + mu0[~treated].var() / (~treated).sum()))
    d = ipw - aipw
    return {
        "treated_share": float(t.mean()),
        "ate_ipw": float(ipw.mean()),
        "ate_aipw": float(aipw.mean()),
        "ate_ipw_minus_aipw": float(d.mean()),
        "ate_ipw_minus_aipw_se": float(d.std(ddof=1) / np.sqrt(len(d))),
        "nuisance_ate": float((nuisance.mu[:, 1] - mu0).mean()),
        "predicted_baseline_treated": float(mu0[treated].mean()),
        "predicted_baseline_control": float(mu0[~treated].mean()),
        "predicted_baseline_gap": gap,
        "predicted_baseline_gap_se": se,
    }


# ---------------------------------------------------------------- report


def _ci(v, lo, hi, scale=100.0, digits=3) -> str:
    return f"{v * scale:.{digits}f} [{lo * scale:.{digits}f}, {hi * scale:.{digits}f}]"


def build_tables(bundle: dict) -> dict[str, pd.DataFrame]:
    first = next(iter(bundle["results"].values()))
    pol, ref = first.policies, first.references
    names = list(bundle["manifests"])
    rows = []
    for n in names:
        row = {"model": n}
        for b in BUDGETS:
            p = pol[
                (pol.model == n)
                & (pol.estimator == "AIPW")
                & (pol.policy == "budget")
                & (pol.budget_requested == b)
            ].iloc[0]
            ip = pol[
                (pol.model == n)
                & (pol.estimator == "IPW")
                & (pol.policy == "budget")
                & (pol.budget_requested == b)
            ].iloc[0]
            row[f"AIPW value@{b:.0%} (pp)"] = _ci(p.value, p.ci_low, p.ci_high)
            row[f"vs random@{b:.0%} (pp)"] = _ci(
                p.diff_vs_random, p.diff_vs_random_ci_low, p.diff_vs_random_ci_high
            )
            row[f"IPW value@{b:.0%} (pp)"] = f"{ip.value * 100:.3f}"
        rows.append(row)
    value = pd.DataFrame(rows)

    refs = ref[ref.estimator == "AIPW"][
        ["policy", "budget", "value", "ci_low", "ci_high", "feasibility"]
    ]
    refs = refs.assign(
        value=lambda d: [
            _ci(v, lo, hi) for v, lo, hi in zip(d.value, d.ci_low, d.ci_high, strict=True)
        ]
    )
    refs = refs[["policy", "budget", "value", "feasibility"]].rename(
        columns={"value": "AIPW value (pp)"}
    )

    q = first.qini
    qini = []
    for n in names:
        a = q[(q.model == n) & (q.estimator == "AIPW")].iloc[0]
        i = q[(q.model == n) & (q.estimator == "IPW")].iloc[0]
        qini.append(
            {
                "model": n,
                "Qini AIPW (pp)": _ci(a.qini, a.ci_low, a.ci_high),
                "Qini IPW (pp)": f"{i.qini * 100:.3f}",
                "Qini / treat-all effect": a.qini / a.ate_treat_all,
            }
        )
    qini = pd.DataFrame(qini)

    pair_rows = []
    for ref_name, res in bundle["results"].items():
        if res.pairwise.empty:
            continue
        pw = res.pairwise[res.pairwise.estimator == "AIPW"]
        for _, r in pw.iterrows():
            pair_rows.append(
                {
                    "reference": ref_name,
                    "model": r.model,
                    "metric": r.metric,
                    "difference (pp)": _ci(r["diff"], r.ci_low, r.ci_high),
                    "excludes 0": bool(r.ci_low > 0 or r.ci_high < 0),
                }
            )
    pairs = pd.DataFrame(
        pair_rows, columns=["reference", "model", "metric", "difference (pp)", "excludes 0"]
    )

    eng = []
    for n, m in bundle["manifests"].items():
        lat = {d["batch"]: d for d in m["latency"]}
        eng.append(
            {
                "model": n,
                "n_fit": m["n_fit"],
                "train_s": m["fit_seconds"],
                "predict 500k rows (s)": m["predict_val_seconds"],
                "rows/s @100k": f"{lat[100_000]['rows_per_s']:,.0f}" if 100_000 in lat else "",
                "p50/p95 ms @1": f"{lat[1]['p50_ms']:.2f} / {lat[1]['p95_ms']:.2f}"
                if 1 in lat
                else "",
                "p50/p95 ms @1k": f"{lat[1000]['p50_ms']:.1f} / {lat[1000]['p95_ms']:.1f}"
                if 1000 in lat
                else "",
                "p50/p95 ms @100k": f"{lat[100000]['p50_ms']:.0f} / {lat[100000]['p95_ms']:.0f}"
                if 100000 in lat
                else "",
                "RSS before fit (GB)": m["rss_gb_before_fit"],
                "RSS peak in fit (GB)": m["rss_gb_peak_during_fit"],
                "process peak RSS (GB)": m["peak_rss_gb"],
                "epochs": m.get("training", {}).get("epochs_run", ""),
            }
        )
    engineering = pd.DataFrame(eng)

    cost = []
    for n, m in bundle["manifests"].items():
        a = q[(q.model == n) & (q.estimator == "AIPW")].iloc[0]
        cost.append(
            {
                "model": n,
                "train_s": m["fit_seconds"],
                "Qini AIPW (pp)": a.qini * 100,
                "Qini CI low": a.ci_low * 100,
                "Qini CI high": a.ci_high * 100,
            }
        )
    return {
        "value": value,
        "references": refs,
        "qini": qini,
        "pairwise": pairs,
        "engineering": engineering,
        "cost_quality": pd.DataFrame(cost),
    }


def _figures(bundle: dict, tables: dict, path_prefix: Path) -> None:
    first = next(iter(bundle["results"].values()))
    curves = first.curves[first.curves.estimator == "AIPW"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for n, c in curves.groupby("model"):
        ax.plot(c.budget * 100, c.gain * 100, label=n, lw=1.4)
    any_c = curves[curves.model == curves.model.iloc[0]]
    ax.plot(any_c.budget * 100, any_c.random_line * 100, "k--", lw=1, label="random targeting")
    ax.set_xlabel("contact budget (% of customers treated)")
    ax.set_ylabel("AIPW incremental visit rate vs treat nobody (pp)")
    ax.set_title("Cumulative policy value (validation subset, preliminary)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_prefix.with_name(path_prefix.name + "_gain_curves.png"), dpi=130)
    plt.close(fig)

    cq = tables["cost_quality"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(
        cq.train_s,
        cq["Qini AIPW (pp)"],
        yerr=[cq["Qini AIPW (pp)"] - cq["Qini CI low"], cq["Qini CI high"] - cq["Qini AIPW (pp)"]],
        fmt="o",
        capsize=3,
    )
    for _, r in cq.iterrows():
        ax.annotate(
            r.model,
            (r.train_s, r["Qini AIPW (pp)"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xscale("log")
    ax.set_xlabel("training time (s, log scale)")
    ax.set_ylabel("Qini coefficient, AIPW (pp)")
    ax.set_title("Training cost vs uplift quality (preliminary)")
    fig.tight_layout()
    fig.savefig(path_prefix.with_name(path_prefix.name + "_cost_quality.png"), dpi=130)
    plt.close(fig)


def build_report(bundle: dict, tables: dict, size: int, prefix: str) -> str:
    m0 = next(iter(bundle["manifests"].values()))
    prov = m0["provenance"]
    libs = {}
    for m in bundle["manifests"].values():
        libs.update(m["libraries"])
    md = lambda df, **kw: df.to_markdown(index=False, **kw)  # noqa: E731
    first = next(iter(bundle["results"].values()))
    meta = first.meta
    ac = bundle["assumption_check"]
    z = ac["predicted_baseline_gap"] / ac["predicted_baseline_gap_se"]
    if abs(ac["ate_ipw_minus_aipw"]) > 2 * ac["ate_ipw_minus_aipw_se"] or abs(z) > 3:
        ac_text = (
            "**Flag:** IPW and AIPW disagree by more than twice their paired standard error, or the arms "
            f"differ in predicted baseline (z = {z:.1f}). This is consistent with covariate imbalance between "
            "treated and control in the released file, i.e. the constant 0.85 propensity does not fully hold "
            "even though the C1 balance thresholds passed. AIPW (primary) adjusts through the outcome model "
            "and stays consistent when that model is adequate; IPW values above are shown only as a check "
            "and are likely biased upward. All models are scored with the same estimator, so model "
            "comparisons are less affected, but absolute gains should be read with this caveat."
        )
    else:
        ac_text = "IPW and AIPW agree within noise and the arms are balanced in predicted baseline."
    pair_view = tables["pairwise"]
    key_pairs = pair_view[pair_view.metric.isin(["qini", "value@20%"])]
    return f"""# Criteo C5 preliminary validation benchmark: {size:,} training rows

**Preliminary.** These are validation-subset benchmarks on training subsets of {size:,} rows. They are
not final held-out results: the Criteo test set was not touched, and **no model has been selected or
frozen**. Hyperparameters are the fixed C3/C4 defaults (no tuning). Differences between models are
reported with paired intervals and should be read as effect sizes, not as a ranking decision.

Units: gains are **percentage points of visit rate per evaluated customer relative to treating
nobody**, estimated with AIPW (IPW as a check). Budgets are contact shares, not monetary cost.
They describe this released sample, not original-population incrementality or advertising ROI:
the constant 0.85 propensity is publisher-supported but cannot be verified from the file, and
Criteo's outcome-dependent privacy subsampling limits causal interpretation.

## Setup
- Training rows: {size:,} (nested, treatment-stratified subset of the train split; identical for every model).
- Validation subset: {bundle["n"]:,} rows (treatment-stratified, fixed). Bootstrap: {meta["n_boot"]} Poisson replicates, seed {meta["seed"]}, shared weights (paired).
- AIPW nuisance: one LightGBM T-learner trained on {bundle["nuisance"]["n_train"]:,} train rows disjoint from every candidate subset up to {bundle["nuisance"]["disjoint_from_candidate_subsets_up_to"]:,} rows, predicting the validation subset (out of sample, same for all models).
- Split fingerprint `{prov["split_fingerprint"][:16]}…`, data sha256 `{(prov["data_sha256"] or "n/a")[:16]}…`, seed {m0["seed"]}.
- Code commit `{(m0["code"]["commit"] or "n/a")[:10]}` (dirty: {m0["code"]["dirty"]}); python {libs["python"]}, lightgbm {libs["lightgbm"]}, torch {libs.get("torch", "n/a")}, scikit-learn {libs["scikit-learn"]}; `OMP_NUM_THREADS={m0["omp_num_threads"]}`; each model ran in its own process.

## Policy value at matched contact budgets (AIPW)
Value = expected visit-rate gain vs treating nobody (pp) if the top share by predicted uplift is treated.
"vs random" is the paired difference from random targeting at the same budget.

{md(tables["value"])}

References (feasible = respects the budget; treat-all is an unconstrained reference below a 100% budget):

{md(tables["references"])}

## Qini (AUUC-style) ranking metric
Area between the cumulative-gain curve and the random line; curve in `{prefix}_gain_curves.png`.

{md(tables["qini"].round({"Qini / treat-all effect": 3}))}

## Paired differences between models (AIPW, selected metrics)
Reference rows come from two joint bootstrap runs with shared weights; full table in the JSON.

{md(key_pairs)}

## Engineering: training cost, latency, memory
Latency measured on the validation rows with `predict_uplift` (includes preprocessing), CPU inference
for tree/linear models; TARNet on its own device. Peak RSS is the process high-water mark, including
data loading (the memory-mapped feature file is paged in while gathering rows). "RSS peak in fit"
is sampled every 50 ms during `fit`; MPS unified memory is not fully visible in RSS. TARNet
epochs: 30 is the epoch cap (early stopping did not trigger).

{md(tables["engineering"].round({"train_s": 1, "predict 500k rows (s)": 2, "RSS before fit (GB)": 2, "RSS peak in fit (GB)": 2, "process peak RSS (GB)": 2}))}

## Training cost vs quality
`{prefix}_cost_quality.png`.

{md(tables["cost_quality"].round(4))}

## Assumption check: IPW vs AIPW
No propensity model was fitted; this only compares the two estimators and arm balance.

| quantity | value |
|:--|--:|
| treated share (validation subset) | {ac["treated_share"]:.4f} |
| treat-all effect, IPW (= treated minus control mean), pp | {ac["ate_ipw"] * 100:.3f} |
| treat-all effect, AIPW, pp | {ac["ate_aipw"] * 100:.3f} |
| IPW minus AIPW, pp (SE) | {ac["ate_ipw_minus_aipw"] * 100:.3f} ({ac["ate_ipw_minus_aipw_se"] * 100:.3f}) |
| predicted baseline visit rate, treated vs control, pp | {ac["predicted_baseline_treated"] * 100:.3f} vs {ac["predicted_baseline_control"] * 100:.3f} |
| difference (SE), pp | {ac["predicted_baseline_gap"] * 100:.3f} ({ac["predicted_baseline_gap_se"] * 100:.3f}) |

{ac_text}

## Caveats
- A single preliminary rung and a single seed; model orderings may change with more training data, tuning, or other seeds.
- TARNet appears on CPU and MPS: CPU is bit-reproducible, MPS is not guaranteed to be. Selection (later) will use one device.
- The shared nuisance is itself a LightGBM T-learner; AIPW remains unbiased for any out-of-sample nuisance, and IPW is shown as a check.
"""


def write_outputs(bundle: dict, size: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"fit{size}"
    tables = build_tables(bundle)
    _figures(bundle, tables, out_dir / prefix)
    (out_dir / f"{prefix}_report.md").write_text(build_report(bundle, tables, size, prefix))
    first = next(iter(bundle["results"].values()))
    summary = {
        "status": "preliminary validation benchmark; test set untouched; no model selected",
        "n_fit": size,
        "n_val": bundle["n"],
        "assumption_check": bundle["assumption_check"],
        "meta": first.meta,
        "nuisance": bundle["nuisance"],
        "tables": {k: v.to_dict("records") for k, v in tables.items()},
        "manifests": {
            n: {k: v for k, v in m.items() if k != "training"}
            | (
                {"training": {k: v for k, v in m["training"].items() if k != "history"}}
                if "training" in m
                else {}
            )
            for n, m in bundle["manifests"].items()
        },
        "policies": {k: r.policies.to_dict("records") for k, r in bundle["results"].items()},
        "qini": {k: r.qini.to_dict("records") for k, r in bundle["results"].items()},
        "pairwise": {k: r.pairwise.to_dict("records") for k, r in bundle["results"].items()},
    }
    _json(out_dir / f"{prefix}_results.json", summary)
    return out_dir / f"{prefix}_report.md"


def run_ladder(
    sizes,
    names,
    data_dir=PROCESSED_DIR,
    out_dir=None,
    n_val=VAL_ROWS,
    seed=SEED,
    n_boot=N_BOOT,
    force=False,
    nuisance_rows=NUISANCE_ROWS,
    nuisance_exclude=NUISANCE_EXCLUDE,
    specs=None,
) -> list[Path]:
    specs = specs or MODEL_SPECS
    data_dir = Path(data_dir)
    out_dir = Path(out_dir) if out_dir else REPORT_DIR / "benchmark"
    split = load_split(data_dir)
    nuisance = ensure_nuisance(data_dir, split, n_val, seed, force, nuisance_rows, nuisance_exclude)
    reports = []
    for size in sizes:
        for name in names:
            ensure_model(
                name,
                {"family": specs[name][0], "kwargs": specs[name][1]},
                size,
                data_dir,
                split,
                n_val,
                seed,
                force,
            )
        bundle = evaluate_rung(data_dir, split, size, n_val, list(names), nuisance, seed, n_boot)
        reports.append(write_outputs(bundle, size, out_dir))
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Criteo C5 benchmark (driver and workers).")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser(
        "run", help="drive the ladder: train in isolated workers, evaluate, report"
    )
    run.add_argument("--sizes", type=int, nargs="+", default=[LADDER[0]])
    run.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=DEFAULT_MODELS)
    run.add_argument("--data-dir", default=str(PROCESSED_DIR))
    run.add_argument("--out-dir", default=None)
    run.add_argument("--n-val", type=int, default=VAL_ROWS)
    run.add_argument("--seed", type=int, default=SEED)
    run.add_argument("--n-boot", type=int, default=N_BOOT)
    run.add_argument("--force", action="store_true")
    w = sub.add_parser("model")
    for a in ("--data-dir", "--name", "--spec", "--out-dir", "--config-key"):
        w.add_argument(a, required=True)
    for a in ("--size", "--n-val", "--seed"):
        w.add_argument(a, type=int, required=True)
    nu = sub.add_parser("nuisance")
    for a in ("--data-dir", "--out-path", "--config-key"):
        nu.add_argument(a, required=True)
    for a in ("--size", "--n-val", "--seed", "--exclude"):
        nu.add_argument(a, type=int, required=True)
    args = parser.parse_args(argv)
    if args.cmd == "model":
        run_model_worker(args)
    elif args.cmd == "nuisance":
        run_nuisance_worker(args)
    else:
        for path in run_ladder(
            args.sizes,
            args.models,
            args.data_dir,
            args.out_dir,
            args.n_val,
            args.seed,
            args.n_boot,
            args.force,
        ):
            print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
