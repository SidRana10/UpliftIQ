"""Criteo inference and budget-constrained targeting with the approved S-learner."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Request

from upliftiq.backend.schemas import (
    BatchRequest,
    BatchResponse,
    CriteoRecord,
    Prediction,
    TargetItem,
    TargetRequest,
    TargetResponse,
)
from upliftiq.evaluation.binary_policy import n_treated_for_budget, rank_order
from upliftiq.modeling.criteo_package import InputValidationError

router = APIRouter(prefix="/criteo", tags=["criteo inference"])

CAVEATS = [
    "Anonymized Criteo benchmark model: features have no business meaning.",
    "Uplift is a ranking signal estimated on a released sample; absolute calibration was not validated.",
    "No financial optimization or ROI is implied; budgets are shares of supplied customers.",
]


def _model(request: Request):
    from upliftiq.backend.app import ModelUnavailable

    model = request.app.state.model
    if model is None:
        raise ModelUnavailable(
            f"approved Criteo model is not loaded ({request.app.state.model_error})"
        )
    return model


def _check_size(request: Request, n: int) -> None:
    limit = request.app.state.settings.max_batch
    if n > limit:
        raise InputValidationError(f"batch of {n} exceeds the API limit of {limit} records")


def _predict(request: Request, records: list[dict]):
    model = _model(request)
    with request.app.state.model_lock:
        return model, model.predict(records)


@router.post("/predict", response_model=Prediction, summary="Single-record uplift inference")
def predict(record: CriteoRecord, request: Request):
    model, out = _predict(request, [record.model_dump()])
    row = out.iloc[0]
    return Prediction(
        p_visit_control=row.p_visit_control,
        p_visit_treatment=row.p_visit_treatment,
        uplift=row.uplift,
        recommend_treatment=bool(row.recommend_treatment),
        model_version=model.version,
    )


@router.post("/predict/batch", response_model=BatchResponse, summary="Batch uplift inference")
def predict_batch(body: BatchRequest, request: Request):
    _check_size(request, len(body.records))
    model, out = _predict(request, [r.model_dump() for r in body.records])
    return BatchResponse(
        model_version=model.version,
        n=len(out),
        predictions=[
            Prediction(
                p_visit_control=r.p_visit_control,
                p_visit_treatment=r.p_visit_treatment,
                uplift=r.uplift,
                recommend_treatment=bool(r.recommend_treatment),
                model_version=model.version,
            )
            for r in out.itertuples()
        ],
        caveats=CAVEATS,
    )


@router.post("/target", response_model=TargetResponse, summary="Contact-budget targeting")
def target(body: TargetRequest, request: Request):
    """Treat the top ``budget`` share of the supplied customers by predicted uplift.

    Ranking is deterministic: descending uplift, ties broken by a seeded hash of the customer id
    (supplied ids, else list position), so the result does not depend on record order when ids
    are given. This is the same rule used in the validated policy evaluation.
    """
    _check_size(request, len(body.records))
    rows = [r.model_dump(exclude={"id"}) for r in body.records]
    model, out = _predict(request, rows)
    ids = (
        np.arange(len(rows))
        if body.records[0].id is None
        else np.array([r.id for r in body.records])
    )
    seed = model.manifest.get("frozen", {}).get("seed", 42)
    order = rank_order(out.uplift.to_numpy(), ids, seed)
    k = n_treated_for_budget(body.budget, len(order))
    items = [
        TargetItem(
            id=int(ids[pos]),
            rank=rank,
            selected=rank <= k,
            p_visit_control=out.p_visit_control.iloc[pos],
            p_visit_treatment=out.p_visit_treatment.iloc[pos],
            uplift=out.uplift.iloc[pos],
        )
        for rank, pos in enumerate(order, start=1)
    ]
    return TargetResponse(
        model_version=model.version,
        n=len(order),
        budget=body.budget,
        n_selected=k,
        ranking="descending predicted uplift; ties by seeded hash of customer id",
        items=items,
        caveats=CAVEATS,
    )
