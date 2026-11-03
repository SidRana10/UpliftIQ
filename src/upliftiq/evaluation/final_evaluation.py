"""Stage 5: one-shot evaluation of the frozen policy on the held-out test set.

Flow: verify the frozen config (config file, dataset checksum, split fingerprint) -> take the dev
and test rows -> recommend with the frozen policy -> fit any outcome (nuisance) models on **dev
only** -> score the policy and the baselines on the test rows with IPW and AIPW.

The core ``evaluate_on_holdout(train, holdout, ...)`` knows nothing about which rows are "the test
set": the real run passes (dev, test), a rehearsal passes (dev folds 1-4, dev fold 0). The test
rows are only ever used to compute features for prediction and to score outcomes; no component is
fitted on them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import stats

from upliftiq.data.registry import HILLSTROM, PROJECT_ROOT, RAW_DIR
from upliftiq.data.validate import validate_hillstrom
from upliftiq.evaluation.hillstrom_analysis import treatment_effects
from upliftiq.evaluation.policy import (
    PROPENSITY,
    OutcomeModelPredictions,
    arm_scores,
    baseline_policies,
    bootstrap_means,
    policy_scores,
)
from upliftiq.evaluation.stats import holm_adjust, wilson_interval
from upliftiq.modeling.features import (
    ARMS,
    TREATMENT_COL,
    encode_arm,
    make_preprocessor,
    select_features,
    tree_categorical_indices,
)
from upliftiq.modeling.learners import TLearner
from upliftiq.modeling.select import BASELINES, Candidate, sha256_file
from upliftiq.modeling.split import Split, make_split

FROZEN_PATH = PROJECT_ROOT / "reports" / "model_selection" / "frozen_config.json"
CV_RESULTS_PATH = PROJECT_ROOT / "reports" / "model_selection" / "cv_results.csv"
OUT_DIR = PROJECT_ROOT / "reports" / "final_evaluation"
RECORD_NAME = "evaluation_record.json"
OUTCOMES = ["visit", "conversion", "spend"]
PRIMARY_OUTCOME = "visit"
ESTIMATORS = ["IPW", "AIPW"]


class FrozenConfigError(RuntimeError):
    """The frozen configuration, data or split does not match what was frozen."""


# ---------------------------------------------------------------- verification


def load_frozen(path: Path = FROZEN_PATH) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _git_ok(path: Path) -> bool:
    """True if ``path`` is tracked by git and unchanged since HEAD."""
    try:
        run = lambda *a: subprocess.run(  # noqa: E731
            ["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True
        )
        return (
            run("ls-files", "--error-unmatch", str(path)).returncode == 0
            and run("diff", "--quiet", "HEAD", "--", str(path)).returncode == 0
        )
    except OSError:
        return False


def verify_frozen(
    cfg: dict, data_path: Path, df: pd.DataFrame, config_path: Path | None = None
) -> Split:
    """Check the frozen config against the data and split; return the verified split.

    Raises ``FrozenConfigError`` on any mismatch. ``config_path`` (if given) must also be tracked
    and unmodified in git, so the config cannot have been edited after selection.
    """
    for key in ("chosen_policy", "seed", "split", "data_sha256", "test_set_accessed"):
        if key not in cfg:
            raise FrozenConfigError(f"frozen config is missing '{key}'")
    if cfg["test_set_accessed"]:
        raise FrozenConfigError("frozen config says the test set was already accessed")
    policy = cfg["chosen_policy"]
    if policy.get("kind") == "baseline":
        if policy.get("name") not in BASELINES:
            raise FrozenConfigError(f"unknown baseline {policy.get('name')!r}")
    elif policy.get("kind") != "model":
        raise FrozenConfigError("chosen_policy.kind must be 'baseline' or 'model'")

    sha = sha256_file(data_path)
    if sha != cfg["data_sha256"] or sha != HILLSTROM.sha256:
        raise FrozenConfigError("dataset checksum does not match the frozen configuration")
    if len(df) != HILLSTROM.n_rows:
        raise FrozenConfigError(f"expected {HILLSTROM.n_rows} rows, got {len(df)}")

    split = make_split(df, seed=cfg["split"]["seed"], n_folds=cfg["split"]["n_folds"])
    if split.fingerprint() != cfg["split"]["fingerprint"]:
        raise FrozenConfigError("split fingerprint does not match the frozen configuration")
    if len(split.dev_idx) != cfg["split"]["n_dev"]:
        raise FrozenConfigError("development set size does not match the frozen configuration")
    if config_path is not None and not _git_ok(config_path):
        raise FrozenConfigError("frozen_config.json is not committed or was modified since")
    return split


def strongest_alternative_baseline(cfg: dict, cv_path: Path = CV_RESULTS_PATH) -> str:
    """Primary comparator: best development-CV fixed baseline other than the winner."""
    cv = pd.read_csv(cv_path)
    cv = cv[cv["family"] == "baseline"]
    chosen = cfg["chosen_policy"].get("name")
    cv = cv[cv["policy"] != chosen]
    return str(cv.sort_values("aipw_value", ascending=False).iloc[0]["policy"])


# ---------------------------------------------------------------- policy and nuisance


def recommend(cfg: dict, train: pd.DataFrame, holdout: pd.DataFrame) -> np.ndarray:
    """Frozen policy's arm recommendations for the holdout rows (fitted on ``train`` if a model)."""
    policy = cfg["chosen_policy"]
    if policy["kind"] == "baseline":
        return baseline_policies(len(holdout))[policy["name"]]
    cand = Candidate(policy["family"], policy["params"])
    model = cand.build(cfg["seed"]).fit(
        train, encode_arm(train[TREATMENT_COL]), train["visit"].to_numpy(dtype=float)
    )
    return model.recommend(holdout)


class SpendNuisance:
    """Per-arm LightGBM Tweedie regressors for expected spend (zeros included); fixed, untuned."""

    def __init__(self, seed: int):
        self.seed = seed

    def fit(self, X: pd.DataFrame, arm: np.ndarray, y: np.ndarray) -> SpendNuisance:
        self.pre_ = make_preprocessor("tree")
        T = self.pre_.fit_transform(select_features(X))
        self.models_ = [
            LGBMRegressor(
                objective="tweedie",
                tweedie_variance_power=1.5,
                n_estimators=150,
                learning_rate=0.03,
                num_leaves=4,
                min_child_samples=300,
                random_state=self.seed,
                deterministic=True,
                force_row_wise=True,
                n_jobs=1,
                verbose=-1,
            ).fit(T[arm == a], y[arm == a], categorical_feature=tree_categorical_indices())
            for a in range(len(ARMS))
        ]
        return self

    def predict_outcomes(self, X: pd.DataFrame) -> np.ndarray:
        T = self.pre_.transform(select_features(X))
        return np.column_stack([m.predict(T) for m in self.models_])


def holdout_nuisance(
    train: pd.DataFrame, holdout: pd.DataFrame, outcome: str, seed: int
) -> OutcomeModelPredictions:
    """Outcome-model predictions for the holdout from models fitted on ``train`` only."""
    arm = encode_arm(train[TREATMENT_COL])
    y = train[outcome].to_numpy(dtype=float)
    if outcome == "spend":
        model = SpendNuisance(seed).fit(train, arm, y)
    else:  # binary outcomes: the same default T-learner used for AIPW in model selection
        model = TLearner(seed=seed).fit(train, arm, y)
    mu = model.predict_outcomes(holdout)
    return OutcomeModelPredictions.from_holdout(
        mu, holdout.index.to_numpy(), train.index.to_numpy()
    )


# ---------------------------------------------------------------- estimation


def policy_value_tables(
    y: np.ndarray,
    arm: np.ndarray,
    rec,
    nuisance: OutcomeModelPredictions,
    chosen: str,
    outcome: str,
    n_boot: int,
    seed: int,
    primary: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """IPW and AIPW policy values and paired comparisons for one outcome.

    Returns (values, comparisons). Comparisons are frozen policy minus each baseline other than
    itself, with paired bootstrap CIs, an analytic paired z-test p-value and Holm adjustment across
    those baselines (within outcome and estimator).
    """
    names = ["frozen_policy", *BASELINES]
    policies = {"frozen_policy": rec, **baseline_policies(len(y))}
    value_rows, comp_rows = [], []
    for est in ESTIMATORS:
        scores = arm_scores(y, arm, nuisance if est == "AIPW" else None, PROPENSITY)
        psi = np.column_stack([policy_scores(scores, policies[n]) for n in names])
        draws = bootstrap_means(psi, n_boot, seed)
        n = len(y)
        for j, name in enumerate(names):
            value_rows.append(
                {
                    "outcome": outcome,
                    "estimator": est,
                    "policy": name,
                    "value": psi[:, j].mean(),
                    "se": psi[:, j].std(ddof=1) / np.sqrt(n),
                    "ci_low": np.percentile(draws[:, j], 2.5),
                    "ci_high": np.percentile(draws[:, j], 97.5),
                }
            )
        rows = []
        for j, name in enumerate(names[1:], start=1):
            if name == chosen:
                continue  # a baseline policy compared with itself
            d = psi[:, 0] - psi[:, j]
            dd = draws[:, 0] - draws[:, j]
            se = d.std(ddof=1) / np.sqrt(n)
            z = d.mean() / se if se > 0 else 0.0
            rows.append(
                {
                    "outcome": outcome,
                    "estimator": est,
                    "baseline": name,
                    "diff": d.mean(),
                    "ci_low": np.percentile(dd, 2.5),
                    "ci_high": np.percentile(dd, 97.5),
                    "paired_se": se,
                    "p_value": float(2 * stats.norm.sf(abs(z))),
                    "primary": name == primary,
                }
            )
        adj = holm_adjust([r["p_value"] for r in rows])
        for r, a in zip(rows, adj, strict=True):
            comp_rows.append({**r, "p_value_holm": a})
    return pd.DataFrame(value_rows), pd.DataFrame(comp_rows)


def observed_arm_table(holdout: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Raw randomized outcomes per arm (no model, no policy): rates and mean spend with 95% CIs."""
    rows = []
    for arm in ARMS:
        g = holdout[holdout[TREATMENT_COL] == arm]
        n = len(g)
        v_lo, v_hi = wilson_interval(int(g["visit"].sum()), n)
        c_lo, c_hi = wilson_interval(int(g["conversion"].sum()), n)
        boot = bootstrap_means(g[["spend"]].to_numpy(dtype=float), n_boot, seed)[:, 0]
        rows.append(
            {
                "arm": arm,
                "n": n,
                "visit_rate": g["visit"].mean(),
                "visit_ci_low": v_lo,
                "visit_ci_high": v_hi,
                "conversion_rate": g["conversion"].mean(),
                "conversion_ci_low": c_lo,
                "conversion_ci_high": c_hi,
                "mean_spend_per_customer": g["spend"].mean(),
                "spend_ci_low": np.percentile(boot, 2.5),
                "spend_ci_high": np.percentile(boot, 97.5),
                "share_zero_spend": float((g["spend"] == 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_on_holdout(
    train: pd.DataFrame, holdout: pd.DataFrame, cfg: dict, primary: str, n_boot: int, seed: int
) -> dict:
    """All Stage 5 tables. ``train`` is the only data any model is fitted on."""
    arm = encode_arm(holdout[TREATMENT_COL])
    rec = recommend(cfg, train, holdout)
    chosen = cfg["chosen_policy"].get("name", "model")
    values, comps = [], []
    for outcome in OUTCOMES:
        nuis = holdout_nuisance(train, holdout, outcome, seed)
        v, c = policy_value_tables(
            holdout[outcome].to_numpy(dtype=float),
            arm,
            rec,
            nuis,
            chosen,
            outcome,
            n_boot,
            seed,
            primary,
        )
        values.append(v)
        comps.append(c)
    return {
        "policy_values": pd.concat(values, ignore_index=True),
        "comparisons": pd.concat(comps, ignore_index=True),
        "observed_arms": observed_arm_table(holdout, n_boot, seed),
        "observed_effects": treatment_effects(holdout, seed=seed),
        "n_train": len(train),
        "n_holdout": len(holdout),
        "recommendation_shares": {a: float(np.mean(rec == i)) for i, a in enumerate(ARMS)}
        if np.ndim(rec) == 1
        else {a: 1 / len(ARMS) for a in ARMS},
    }


# ---------------------------------------------------------------- orchestration


def _git_head() -> tuple[str | None, bool | None]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True
            ).stdout.strip()
        )
        return head or None, dirty
    except OSError:
        return None, None


def run_final_evaluation(
    df: pd.DataFrame,
    data_path: Path,
    out_dir: Path,
    config_path: Path = FROZEN_PATH,
    cv_path: Path = CV_RESULTS_PATH,
    n_boot: int = 2000,
    check_git: bool = True,
) -> dict:
    """Verify, then evaluate the frozen policy once on the test rows and write all artifacts."""
    from upliftiq.evaluation.final_report import write_artifacts

    cfg, cfg_sha = load_frozen(config_path)
    record_path = out_dir / RECORD_NAME
    if (
        record_path.exists()
        and json.loads(record_path.read_text()).get("frozen_config_sha256") == cfg_sha
    ):
        raise FrozenConfigError(
            "this frozen configuration has already been evaluated on the test set; "
            "results are in the output directory and must not be re-run or revised"
        )
    split = verify_frozen(cfg, data_path, df, config_path if check_git else None)
    primary = strongest_alternative_baseline(cfg, cv_path)
    train, test = df.iloc[split.dev_idx], df.iloc[split.test_idx]  # original index kept
    result = evaluate_on_holdout(train, test, cfg, primary, n_boot, cfg["seed"])
    head, dirty = _git_head()
    record = {
        "frozen_config_sha256": cfg_sha,
        "chosen_policy": cfg["chosen_policy"],
        "primary_outcome": PRIMARY_OUTCOME,
        "primary_comparator": primary,
        "data_sha256": cfg["data_sha256"],
        "split_fingerprint": split.fingerprint(),
        "n_dev": len(train),
        "n_test": len(test),
        "seed": cfg["seed"],
        "n_boot": n_boot,
        "code_commit": head,
        "code_dirty": dirty,
        "nuisance_models_fitted_on": "development rows only",
    }
    write_artifacts(result, cfg, record, out_dir)
    return result


def rehearse(df: pd.DataFrame, cfg: dict, cv_path: Path, out_dir: Path, n_boot: int = 200) -> dict:
    """Dry run of the whole pipeline on development data only (fold 0 plays the test set)."""
    from upliftiq.evaluation.final_report import write_artifacts
    from upliftiq.modeling.select import load_dev_data

    split = make_split(df, seed=cfg["split"]["seed"], n_folds=cfg["split"]["n_folds"])
    dev, fold = load_dev_data(df, split)
    primary = strongest_alternative_baseline(cfg, cv_path)
    result = evaluate_on_holdout(dev[fold != 0], dev[fold == 0], cfg, primary, n_boot, cfg["seed"])
    write_artifacts(result, cfg, {"rehearsal": True, "primary_comparator": primary}, out_dir)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 5: held-out evaluation of the frozen policy."
    )
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument(
        "--rehearsal", action="store_true", help="dry run on development data only (no test rows)"
    )
    args = parser.parse_args(argv)
    data_path = RAW_DIR / HILLSTROM.filename
    df = validate_hillstrom(data_path)
    if args.rehearsal:
        cfg, _ = load_frozen()
        verify_frozen(cfg, data_path, df, FROZEN_PATH)
        rehearse(df, cfg, CV_RESULTS_PATH, args.out, min(args.n_boot, 200))
        print(f"Rehearsal (development data only) written to {args.out}")
        return 0
    run_final_evaluation(df, data_path, args.out, n_boot=args.n_boot)
    print(f"Final evaluation written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
