"""Read-only endpoints over committed reports. Hillstrom and Criteo are kept strictly apart.

Nothing is recomputed: each handler reads a report file written by the completed experiments.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from upliftiq.backend.reports import Reports, ReportUnavailable
from upliftiq.backend.schemas import Envelope, Health

router = APIRouter()

HILLSTROM_FROZEN = "reports/model_selection/frozen_config.json"
CRITEO_FROZEN = "reports/criteo/benchmark_full/frozen_config.json"

HILLSTROM_CAVEATS = [
    "Randomized 3-arm email test (Men's / Women's / no email); business outcomes in business units.",
    "Visit is the primary, pre-specified outcome; conversion and spend are exploratory (rare / 99% zeros).",
    "The frozen decision is a fixed policy (Men's E-Mail for all); no personalised model beat it.",
]
CRITEO_CAVEATS = [
    "Anonymized benchmark: features f0-f11 have no business meaning; gains are percentage points of visit rate.",
    "Conditional estimates for the released sample; not population ROI or guaranteed causal gains.",
    "The constant 0.85 propensity is publisher-supported but unverifiable; IPW and AIPW disagree on this file (AIPW primary, IPW a cross-check).",
    "Outcome-dependent privacy subsampling limits causal interpretation.",
]

HILLSTROM_FIGURES = {
    "covariate_balance": "reports/hillstrom/figures/covariate_balance.png",
    "covariate_distributions": "reports/hillstrom/figures/covariate_distributions.png",
    "outcomes_by_group": "reports/hillstrom/figures/outcomes_by_group.png",
    "spend_distribution": "reports/hillstrom/figures/spend_distribution.png",
    "treatment_effects": "reports/hillstrom/figures/treatment_effects.png",
    "observed_outcomes": "reports/final_evaluation/figures/observed_outcomes.png",
    "primary_visit_policy_values": "reports/final_evaluation/figures/primary_visit_policy_values.png",
    "secondary_policy_values": "reports/final_evaluation/figures/secondary_policy_values.png",
}
CRITEO_FIGURES = {
    "heldout_gain_curve": "reports/criteo/final/heldout_gain_curve.png",
    "heldout_value_vs_random": "reports/criteo/final/heldout_value_vs_random.png",
    "scaling_curves": "reports/criteo/benchmark_2m/scaling_curves.png",
    "treatment_balance": "reports/criteo/treatment_balance.png",
}


MODEL_FIELDS = (
    "family", "config_key", "n_fit", "fit_seconds", "predict_val_seconds", "latency",
    "rss_gb_before_fit", "rss_gb_peak_during_fit", "peak_rss_gb", "omp_num_threads",
    "hyperparameters", "training", "libraries", "code", "uplift_mean", "uplift_std",
)  # fmt: skip


def _heldout_numeric(final: dict) -> dict:
    """Unrounded policy values, intervals and Qini as stored (the display tables are strings)."""
    return {k: final[k] for k in ("policies", "references", "qini")}


def _reports(request: Request) -> Reports:
    return request.app.state.reports


def _sha(reports: Reports, relative: str) -> str | None:
    try:
        return hashlib.sha256(reports.path(relative).read_bytes()).hexdigest()
    except ReportUnavailable:
        return None


def envelope(
    reports: Reports, dataset: str, source: list[str], data: Any, frozen: str | None = None
) -> Envelope:
    caveats = {"hillstrom": HILLSTROM_CAVEATS, "criteo": CRITEO_CAVEATS}.get(dataset, [])
    return Envelope(
        dataset=dataset,
        source=source,
        provenance="historical",
        frozen_config_sha256=_sha(reports, frozen) if frozen else None,
        caveats=caveats,
        data=data,
    )


def _figure(reports: Reports, figures: dict[str, str], name: str) -> FileResponse:
    if name not in figures:
        raise HTTPException(404, f"unknown figure '{name}'; available: {sorted(figures)}")
    return FileResponse(reports.path(figures[name]), media_type="image/png")


# ------------------------------------------------------------------ platform


@router.get("/health", response_model=Health, tags=["platform"])
def health(request: Request):
    app, reports = request.app, _reports(request)
    checks = {
        "hillstrom_final_evaluation": "reports/final_evaluation/policy_values.csv",
        "hillstrom_frozen_config": HILLSTROM_FROZEN,
        "criteo_heldout": "reports/criteo/final/final_results.json",
        "criteo_frozen_config": CRITEO_FROZEN,
        "criteo_model_manifest": "reports/criteo/model_package.json",
    }
    available = {k: reports.exists(v) for k, v in checks.items()}
    model = app.state.model
    ok = model is not None and all(available.values())
    return Health(
        status="ok" if ok else "degraded",
        api_version="v1",
        criteo_model_loaded=model is not None,
        model_version=model.version if model else None,
        model_error=app.state.model_error,
        reports_available=available,
    )


@router.get("/datasets", response_model=Envelope, tags=["platform"])
def datasets(request: Request):
    r = _reports(request)
    data = [
        {
            "id": "hillstrom",
            "name": "Hillstrom MineThatData E-Mail Analytics",
            "kind": "business-facing randomized email experiment",
            "rows": 64000,
            "arms": ["No E-Mail", "Mens E-Mail", "Womens E-Mail"],
            "outcomes": {"primary": "visit", "exploratory": ["conversion", "spend"]},
            "frozen_decision": "Men's E-Mail for all",
            "heldout_customers": r.json("reports/final_evaluation/evaluation_record.json")[
                "n_test"
            ],
            "units": "proportions; spend in dollars",
        },
        {
            "id": "criteo",
            "name": "Criteo Uplift Prediction Dataset v2.1",
            "kind": "anonymized benchmark (no business interpretation, no ROI)",
            "rows": r.json("reports/criteo/dataset_summary.json")["n_rows"],
            "arms": ["control", "treatment"],
            "outcomes": {"primary": "visit", "exploratory": ["conversion"]},
            "features": [f"f{i}" for i in range(12)],
            "frozen_model": "LightGBM S-learner; primary metric AIPW value at 20% contact budget",
            "heldout_customers": r.json("reports/criteo/final/final_results.json")["n"],
            "units": "percentage points of visit rate vs treating nobody",
        },
    ]
    return envelope(r, "platform", ["reports/final_evaluation/evaluation_record.json",
                                    "reports/criteo/dataset_summary.json",
                                    "reports/criteo/final/final_results.json"], data)  # fmt: skip


# ------------------------------------------------------------------ overview


@router.get("/overview", response_model=Envelope, tags=["overview"])
def overview(request: Request):
    """Headline of each study side by side, never merged or ranked against each other."""
    r = _reports(request)
    values = r.csv("reports/final_evaluation/policy_values.csv")
    comps = r.csv("reports/final_evaluation/comparisons.csv")
    record = r.json("reports/final_evaluation/evaluation_record.json")
    frozen = next(
        v for v in values
        if v["outcome"] == "visit" and v["estimator"] == "AIPW" and v["policy"] == "frozen_policy"
    )  # fmt: skip
    primary = next(
        c for c in comps if c["outcome"] == "visit" and c["estimator"] == "AIPW" and c["primary"]
    )
    held = r.csv("reports/criteo/final/heldout_value.csv")
    refs = r.csv("reports/criteo/final/heldout_references.csv")
    final = r.json("reports/criteo/final/final_results.json")
    record_c = r.json("reports/criteo/final/evaluation_record.json")
    data = {
        "hillstrom": {
            "frozen_policy": record["chosen_policy"]["name"],
            "primary_metric": "AIPW expected visit rate (proportion)",
            "value": {k: frozen[k] for k in ("value", "ci_low", "ci_high")},
            "vs_strongest_alternative_baseline": {
                "baseline": primary["baseline"],
                **{k: primary[k] for k in ("diff", "ci_low", "ci_high", "p_value_holm")},
            },
            "heldout_customers": record["n_test"],
            "caveats": HILLSTROM_CAVEATS,
        },
        "criteo": {
            "model": final["model"],
            "primary_metric": "AIPW policy value at 20% contact budget (pp of visit rate)",
            "budgets": held,
            "treat_all_reference": next(x for x in refs if x["policy"] == "treat_all"),
            "heldout_customers": final["n"],
            "evaluation_record_status": record_c["status"],
            "code_dirty": record_c["code_dirty"],
            "caveats": CRITEO_CAVEATS,
        },
    }
    return envelope(r, "platform", ["reports/final_evaluation", "reports/criteo/final"], data)


# ------------------------------------------------------------------ hillstrom


@router.get("/hillstrom/groups", response_model=Envelope, tags=["hillstrom"])
def hillstrom_groups(request: Request):
    r = _reports(request)
    src = "reports/hillstrom/tables/group_summary.csv"
    return envelope(r, "hillstrom", [src], r.csv(src))


@router.get("/hillstrom/effects", response_model=Envelope, tags=["hillstrom"])
def hillstrom_effects(request: Request):
    """Observed randomized effects: full sample and the held-out customers."""
    r = _reports(request)
    srcs = ["reports/hillstrom/tables/effects.csv", "reports/final_evaluation/observed_effects.csv"]
    data = {"full_sample": r.csv(srcs[0]), "heldout": r.csv(srcs[1])}
    return envelope(r, "hillstrom", srcs, data, HILLSTROM_FROZEN)


@router.get("/hillstrom/balance", response_model=Envelope, tags=["hillstrom"])
def hillstrom_balance(request: Request):
    r = _reports(request)
    src = "reports/hillstrom/tables/balance.csv"
    return envelope(r, "hillstrom", [src], r.csv(src))


@router.get("/hillstrom/models", response_model=Envelope, tags=["hillstrom"])
def hillstrom_models(request: Request):
    """Model-selection summary (cross-validated AIPW value per candidate policy)."""
    r = _reports(request)
    srcs = ["reports/model_selection/selection_summary.csv", HILLSTROM_FROZEN]
    frozen = r.json(HILLSTROM_FROZEN)
    data = {"candidates": r.csv(srcs[0]), "selection": frozen["selection"]}
    return envelope(r, "hillstrom", srcs, data, HILLSTROM_FROZEN)


@router.get("/hillstrom/decision", response_model=Envelope, tags=["hillstrom"])
def hillstrom_decision(request: Request):
    """Frozen policy, its held-out policy values and paired comparisons (visit primary)."""
    r = _reports(request)
    srcs = [
        "reports/final_evaluation/policy_values.csv",
        "reports/final_evaluation/comparisons.csv",
        "reports/final_evaluation/evaluation_record.json",
    ]
    data = {
        "chosen_policy": r.json(HILLSTROM_FROZEN)["chosen_policy"],
        "policy_values": r.csv(srcs[0]),
        "comparisons": r.csv(srcs[1]),
        "evaluation_record": r.json(srcs[2]),
        "outcome_roles": {"visit": "primary", "conversion": "exploratory", "spend": "exploratory"},
    }
    return envelope(r, "hillstrom", srcs + [HILLSTROM_FROZEN], data, HILLSTROM_FROZEN)


@router.get("/hillstrom/figures/{name}", tags=["hillstrom"], response_class=FileResponse)
def hillstrom_figure(name: str, request: Request):
    return _figure(_reports(request), HILLSTROM_FIGURES, name)


# ------------------------------------------------------------------ criteo


@router.get("/criteo/diagnostics", response_model=Envelope, tags=["criteo"])
def criteo_diagnostics(request: Request):
    r = _reports(request)
    srcs = [
        "reports/criteo/dataset_summary.json",
        "reports/criteo/treatment_diagnostics.json",
        "reports/criteo/split_summary.json",
    ]
    data = {"dataset": r.json(srcs[0]), "treatment": r.json(srcs[1]), "split": r.json(srcs[2])}
    return envelope(r, "criteo", srcs, data, CRITEO_FROZEN)


@router.get("/criteo/ipw-aipw", response_model=Envelope, tags=["criteo"])
def criteo_ipw_aipw(request: Request):
    """The documented IPW vs AIPW discrepancy (shown, never hidden)."""
    r = _reports(request)
    src = "reports/criteo/benchmark/ipw_aipw_diagnostic.json"
    return envelope(r, "criteo", [src], r.json(src), CRITEO_FROZEN)


@router.get("/criteo/benchmark", response_model=Envelope, tags=["criteo"])
def criteo_benchmark(request: Request):
    """Validation comparison of the three finalists (not held-out) and the learning curve."""
    r = _reports(request)
    srcs = ["reports/criteo/benchmark_full/final_results.json",
            "reports/criteo/benchmark_2m/scaling_results.json"]  # fmt: skip
    full, scaling = r.json(srcs[0]), r.json(srcs[1])
    data = {
        "split": "validation (500K subset); the model was selected on these numbers",
        "n_train": full["n_fit"],
        "tables": {
            k: full["tables"][k] for k in ("value", "references", "qini", "pairwise", "engineering")
        },
        "selection": full["selection"],
        "change_from_2m": full["changes_vs_base"],
        "learning_curve": {
            "levels": scaling["levels"],
            "changes": scaling["changes"],
            "engineering": scaling["engineering"],
        },
        "numeric": {
            # every model is scored identically under each reference run: any one is the full set
            "policies": next(iter(full["policies"].values())),
            "qini": next(iter(full["qini"].values())),
            "pairwise": full["pairwise"],
            "models": {
                name: {k: m[k] for k in MODEL_FIELDS if k in m}
                for name, m in full["manifests"].items()
            },
        },
    }
    return envelope(r, "criteo", srcs, data, CRITEO_FROZEN)


@router.get("/criteo/heldout", response_model=Envelope, tags=["criteo"])
def criteo_heldout(request: Request):
    """One-shot held-out result of the frozen S-learner (AIPW primary, IPW cross-check)."""
    r = _reports(request)
    srcs = [
        "reports/criteo/final/final_results.json",
        "reports/criteo/final/evaluation_record.json",
    ]
    final = r.json(srcs[0])
    data = {
        "model": final["model"],
        "n": final["n"],
        "primary_metric": "AIPW policy value at 20% contact budget",
        "value": final["tables"]["value"],
        "references": final["tables"]["references"],
        "qini": final["tables"]["qini"],
        "assumption_check": final["assumption_check"],
        "exploratory_conversion": {
            "table": final["tables"]["conversion"],
            "note": "exploratory only; never used for any decision",
        },
        "evaluation_record": r.json(srcs[1]),
        "numeric": _heldout_numeric(final),
    }
    return envelope(r, "criteo", srcs, data, CRITEO_FROZEN)


@router.get("/criteo/curves", response_model=Envelope, tags=["criteo"])
def criteo_curves(request: Request):
    """Validated policy-value points at the evaluated budgets plus the stored curve figures.

    Only budgets 10%, 20% and 30% were evaluated; no values are produced for other budgets.
    """
    r = _reports(request)
    srcs = ["reports/criteo/final/final_results.json"]
    final = r.json(srcs[0])
    data = {
        "evaluated_budgets": [0.1, 0.2, 0.3],
        "points": final["tables"]["value"],
        "references": final["tables"]["references"],
        "qini": final["tables"]["qini"],
        "figures": {n: f"/api/v1/criteo/figures/{n}" for n in CRITEO_FIGURES},
        "numeric": _heldout_numeric(final),
    }
    return envelope(r, "criteo", srcs, data, CRITEO_FROZEN)


@router.get("/criteo/figures/{name}", tags=["criteo"], response_class=FileResponse)
def criteo_figure(name: str, request: Request):
    return _figure(_reports(request), CRITEO_FIGURES, name)


# ------------------------------------------------------------------ models


@router.get("/models", response_model=Envelope, tags=["models"])
def models(request: Request):
    r, app = _reports(request), request.app
    model = app.state.model
    data = [
        {
            "dataset": "hillstrom",
            "name": "mens_for_all",
            "kind": "fixed policy (no served model)",
            "version": None,
            "available": True,
        },
        {
            "dataset": "criteo",
            "name": "criteo_s_learner",
            "kind": "LightGBM S-learner",
            "version": model.version if model else None,
            "available": model is not None,
            "error": app.state.model_error,
        },
    ]
    return envelope(r, "platform", [], data)


@router.get("/models/criteo", response_model=Envelope, tags=["models"])
def model_criteo(request: Request):
    """Manifest of the packaged model: schema, fingerprints, checksums, dependencies."""
    r, app = _reports(request), request.app
    model = app.state.model
    manifest = model.manifest if model else None
    if manifest is None:
        manifest = r.json("reports/criteo/model_package.json")
    data = {
        "loaded": model is not None,
        "load_error": app.state.model_error,
        "manifest": manifest,
        "evaluation_record": r.json("reports/criteo/final/evaluation_record.json"),
    }
    return envelope(
        r,
        "criteo",
        ["reports/criteo/model_package.json", "reports/criteo/final/evaluation_record.json"],
        data,
        CRITEO_FROZEN,
    )


@router.get("/models/criteo/runs", response_model=Envelope, tags=["models"])
def model_runs(request: Request):
    """MLflow runs (local store). Historical imports and new executions are labelled apart."""
    r, tracking = _reports(request), request.app.state.settings.tracking_dir
    db = tracking / "mlflow.db"
    if not db.is_file():
        raise ReportUnavailable("mlruns/mlflow.db")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    client = mlflow.MlflowClient(f"sqlite:///{db.resolve()}")
    exp = client.get_experiment_by_name("UpliftIQ-Criteo")
    runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"]) if exp else []
    data = [
        {
            "run_id": x.info.run_id,
            "name": x.info.run_name,
            "provenance": x.data.tags.get("provenance"),
            "executed_by_this_run": x.data.tags.get("executed_by_this_run"),
            "model_version": x.data.tags.get("model_version"),
            "start_time": x.info.start_time,
            "metrics": x.data.metrics,
        }
        for x in runs
    ]
    return envelope(r, "criteo", ["mlruns/mlflow.db"], data, CRITEO_FROZEN)
