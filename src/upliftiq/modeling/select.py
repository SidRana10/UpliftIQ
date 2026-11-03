"""Stage 4: cross-validated model selection on the development data only.

Implements docs/causal_modeling_plan.md sections 5-8 for the primary outcome (visit):

1. For every candidate (family x fixed hyperparameter grid) fit on 4 CV folds, recommend arms for
   the held-out fold, and stack the out-of-fold recommendations over all dev rows.
2. Score each policy with AIPW (primary) and IPW (check) using the known propensities. AIPW
   outcome predictions come from one fixed T-learner, cross-fitted on the same folds, so every
   customer's nuisance prediction comes from a model that never trained on that customer.
3. Choose hyperparameters per family, then the final policy, with the one-standard-error rule
   (simplest eligible option wins). Fixed baselines compete and may win.

Everything here takes the development rows only. ``load_dev_data`` is the single place the split
is applied, and it drops the test rows before any outcome is used.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from upliftiq.data.registry import HILLSTROM, PROJECT_ROOT, RAW_DIR
from upliftiq.data.validate import validate_hillstrom
from upliftiq.evaluation.policy import (
    PROPENSITY,
    OutcomeModelPredictions,
    arm_scores,
    baseline_policies,
    evaluate_policy,
    policy_scores,
)
from upliftiq.modeling.features import TREATMENT_COL, encode_arm
from upliftiq.modeling.learners import TLearner, make_model
from upliftiq.modeling.split import Split, load_split, make_split, save_split

SEED = 42
N_BOOT = 2000
OUT_DIR = PROJECT_ROOT / "reports" / "model_selection"
SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "hillstrom_split.json"

LOGISTIC_GRID = [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)]
LGBM_GRID = [{"num_leaves": nl, "min_child_samples": mc} for nl in (4, 8, 16) for mc in (100, 300)]
FAMILIES = ["logistic", "s_learner", "t_learner", "dr_learner"]  # simplest first
BASELINES = ["no_email", "mens_for_all", "womens_for_all", "random"]  # simplest first


@dataclass(frozen=True)
class Candidate:
    family: str
    params: dict

    @property
    def name(self) -> str:
        return self.family + ":" + ",".join(f"{k}={v}" for k, v in self.params.items())

    @property
    def complexity(self) -> tuple:
        """Within a family, smaller is simpler (fewer leaves, larger leaves, stronger shrinkage)."""
        if self.family == "logistic":
            return (self.params["C"],)
        return (self.params["num_leaves"], -self.params["min_child_samples"])

    def build(self, seed: int = SEED):
        if self.family == "logistic":
            return make_model("logistic", seed=seed, **self.params)
        return make_model(self.family, seed=seed, params=dict(self.params))


def candidate_grid() -> list[Candidate]:
    grids = {"logistic": LOGISTIC_GRID}
    return [Candidate(f, p) for f in FAMILIES for p in grids.get(f, LGBM_GRID)]


def load_dev_data(df: pd.DataFrame, split: Split) -> tuple[pd.DataFrame, np.ndarray]:
    """Development rows only, plus each row's CV fold. Test rows are dropped here."""
    dev_idx = split.dev_idx
    dev = df.iloc[dev_idx].reset_index(drop=True)
    return dev, split.assignment[dev_idx].astype(int)


def _fold_rows(fold: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    return np.flatnonzero(fold != k), np.flatnonzero(fold == k)


def oof_recommendations(dev: pd.DataFrame, fold: np.ndarray, cand: Candidate, seed=SEED):
    """Stacked out-of-fold arm recommendations for every dev row."""
    arm, y = encode_arm(dev[TREATMENT_COL]), dev["visit"].to_numpy(dtype=float)
    rec = np.empty(len(dev), dtype=int)
    for k in range(int(fold.max()) + 1):
        tr, va = _fold_rows(fold, k)
        model = cand.build(seed).fit(dev.iloc[tr], arm[tr], y[tr])
        rec[va] = model.recommend(dev.iloc[va])
    return rec


def oof_nuisance(dev: pd.DataFrame, fold: np.ndarray, seed=SEED) -> OutcomeModelPredictions:
    """Fixed default T-learner, cross-fitted on the CV folds, for AIPW outcome predictions."""
    arm, y = encode_arm(dev[TREATMENT_COL]), dev["visit"].to_numpy(dtype=float)
    n_folds = int(fold.max()) + 1
    mu = np.empty((len(dev), 3))
    trained_on = {}
    for k in range(n_folds):
        tr, va = _fold_rows(fold, k)
        model = TLearner(seed=seed).fit(dev.iloc[tr], arm[tr], y[tr])
        mu[va] = model.predict_outcomes(dev.iloc[va])
        trained_on[k] = tr
    return OutcomeModelPredictions(mu, np.arange(len(dev)), fold, trained_on)


def paired_gap(psi_best: np.ndarray, psi_other: np.ndarray) -> tuple[float, float]:
    """(mean difference best - other, paired standard error)."""
    d = psi_best - psi_other
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def one_se_choice(order: list[str], psi: dict[str, np.ndarray], best: str) -> str:
    """Simplest option (earliest in ``order``) whose value is within 1 paired SE of ``best``."""
    for name in order:
        if name == best:
            return best
        gap, se = paired_gap(psi[best], psi[name])
        if gap <= se:
            return name
    return best


def best_by_value(names: list[str], psi: dict[str, np.ndarray]) -> str:
    return max(names, key=lambda n: (psi[n].mean(), -names.index(n)))


def choose_in_family(cands: list[Candidate], psi: dict[str, np.ndarray]) -> Candidate:
    ordered = sorted(cands, key=lambda c: c.complexity)
    names = [c.name for c in ordered]
    pick = one_se_choice(names, psi, best_by_value(names, psi))
    return ordered[names.index(pick)]


def select_final(family_best: dict[str, Candidate], psi: dict[str, np.ndarray]) -> dict:
    """Final policy: baselines and family winners compete under the one-SE rule.

    A personalized model whose CV value is below the best fixed baseline is ineligible.
    """
    best_baseline = best_by_value(BASELINES, psi)
    eligible_models = [
        f
        for f in FAMILIES
        if f in family_best and psi[family_best[f].name].mean() >= psi[best_baseline].mean()
    ]
    order = BASELINES + eligible_models  # simplest first
    key = {f: family_best[f].name for f in eligible_models}
    scores = {**{b: psi[b] for b in BASELINES}, **{f: psi[key[f]] for f in eligible_models}}
    best = best_by_value(order, scores)
    chosen = one_se_choice(order, scores, best)
    return {
        "chosen": chosen,
        "best_by_value": best,
        "best_baseline": best_baseline,
        "eligible_models": eligible_models,
        "ineligible_models": [f for f in FAMILIES if f in family_best and f not in eligible_models],
    }


def run_model_selection(
    dev: pd.DataFrame, fold: np.ndarray, seed: int = SEED, n_boot: int = N_BOOT, verbose=True
) -> dict:
    """Run the full CV selection on development data and return all tables and the decision."""
    arm = encode_arm(dev[TREATMENT_COL])
    y = dev["visit"].to_numpy(dtype=float)
    nuisance = oof_nuisance(dev, fold, seed)
    aipw = arm_scores(y, arm, nuisance, PROPENSITY)
    ipw = arm_scores(y, arm, None, PROPENSITY)

    policies: dict[str, np.ndarray] = dict(baseline_policies(len(dev)))
    cands = candidate_grid()
    for i, c in enumerate(cands, 1):
        policies[c.name] = oof_recommendations(dev, fold, c, seed)
        if verbose:
            print(f"[{i}/{len(cands)}] {c.name}", flush=True)
    psi = {n: policy_scores(aipw, p) for n, p in policies.items()}
    psi_ipw = {n: policy_scores(ipw, p) for n, p in policies.items()}

    family_best = {f: choose_in_family([c for c in cands if c.family == f], psi) for f in FAMILIES}
    decision = select_final(family_best, psi)

    n = len(dev)
    rows = []
    for name in [*BASELINES, *(c.name for c in cands)]:
        cand = next((c for c in cands if c.name == name), None)
        fold_vals = [psi[name][fold == k].mean() for k in range(int(fold.max()) + 1)]
        rows.append(
            {
                "policy": name,
                "family": cand.family if cand else "baseline",
                "aipw_value": psi[name].mean(),
                "aipw_se": psi[name].std(ddof=1) / np.sqrt(n),
                "ipw_value": psi_ipw[name].mean(),
                "ipw_se": psi_ipw[name].std(ddof=1) / np.sqrt(n),
                "share_no_email": float(np.mean(policies[name] == 0)) if cand else np.nan,
                "share_mens": float(np.mean(policies[name] == 1)) if cand else np.nan,
                "share_womens": float(np.mean(policies[name] == 2)) if cand else np.nan,
                "family_best": bool(cand and family_best[cand.family].name == name),
                **{f"fold_{k}_aipw": v for k, v in enumerate(fold_vals)},
            }
        )
    cv_results = pd.DataFrame(rows)

    # Paired bootstrap CIs for the family winners against all four baselines (AIPW)
    summary = []
    for f in FAMILIES:
        tbl = evaluate_policy(
            y, arm, policies[family_best[f].name], nuisance, PROPENSITY, n_boot, seed
        ).set_index("policy")
        row = {"policy": f, "config": family_best[f].name, **_ci_cols(tbl.loc["candidate"])}
        for b in BASELINES:
            row.update(_diff_cols(tbl.loc[b], b))
        row["ipw_value"] = psi_ipw[family_best[f].name].mean()
        summary.append(row)
        base_tbl = tbl
    for b in BASELINES:
        row = {"policy": b, "config": "fixed baseline", **_ci_cols(base_tbl.loc[b])}
        row["ipw_value"] = psi_ipw[b].mean()
        summary.append(row)
    summary = pd.DataFrame(summary)

    return {
        "cv_results": cv_results,
        "summary": summary,
        "family_best": family_best,
        "decision": decision,
        "psi": psi,
        "policies": policies,
        "nuisance": nuisance,
        "n_dev": n,
    }


def _ci_cols(r) -> dict:
    return {"aipw_value": r["value"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]}


def _diff_cols(r, b: str) -> dict:
    return {
        f"diff_vs_{b}": r["diff"],
        f"diff_vs_{b}_ci_low": r["diff_ci_low"],
        f"diff_vs_{b}_ci_high": r["diff_ci_high"],
    }


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def frozen_config(result: dict, split: Split, seed: int, n_boot: int, data_sha256: str) -> dict:
    """Everything needed to reproduce the selected policy and run the one-shot test evaluation."""
    d = result["decision"]
    chosen = d["chosen"]
    if chosen in BASELINES:
        policy = {"kind": "baseline", "name": chosen}
    else:
        c = result["family_best"][chosen]
        policy = {"kind": "model", "family": c.family, "params": c.params, "config": c.name}
    return {
        "chosen_policy": policy,
        "selection": {k: v for k, v in d.items() if k != "chosen"},
        "family_best_configs": {f: c.params for f, c in result["family_best"].items()},
        "seed": seed,
        "split": {
            "fingerprint": split.fingerprint(),
            "seed": split.seed,
            "n_folds": split.n_folds,
            "n_dev": result["n_dev"],
        },
        "evaluation": {
            "primary_outcome": "visit",
            "primary_metric": "cv_aipw_policy_value",
            "check_metric": "cv_ipw_policy_value",
            "propensity": PROPENSITY,
            "nuisance_model": "TLearner(default params), cross-fitted on the CV folds",
            "bootstrap": {"n_boot": n_boot, "seed": seed, "unit": "customer", "ci": 0.95},
            "selection_rule": "one paired SE; simplest eligible; baselines may win; "
            "models below best baseline ineligible",
            "model_grids": {"logistic": LOGISTIC_GRID, "lightgbm": LGBM_GRID},
        },
        "data_sha256": data_sha256,
        "code_commit": _git("rev-parse", "HEAD"),
        "code_dirty": bool(_git("status", "--porcelain")),
        "test_set_accessed": False,
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_split(df: pd.DataFrame, path: Path = SPLIT_PATH) -> Split:
    """Load the saved split if present (verifying it matches a fresh build), else create it."""
    fresh = make_split(df)
    if path.exists():
        saved = load_split(path, n_rows=len(df))
        if saved.fingerprint() != fresh.fingerprint():
            raise RuntimeError("saved split differs from the reproducible split")
        return saved
    save_split(fresh, path)
    return fresh


def main(argv: list[str] | None = None) -> int:
    from upliftiq.modeling.select_report import write_artifacts

    parser = argparse.ArgumentParser(description="Hillstrom CV model selection (dev data only).")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args(argv)

    data_path = RAW_DIR / HILLSTROM.filename
    df = validate_hillstrom(data_path)
    split = get_split(df)
    dev, fold = load_dev_data(df, split)
    del df  # the test rows are not needed (or used) from here on
    result = run_model_selection(dev, fold, SEED, args.n_boot)
    cfg = frozen_config(result, split, SEED, args.n_boot, sha256_file(data_path))
    write_artifacts(result, cfg, args.out)
    print(f"Chosen policy: {cfg['chosen_policy']}")
    print(f"Artifacts written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
