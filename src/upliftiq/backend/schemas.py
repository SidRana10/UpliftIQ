"""Pydantic request/response models."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from upliftiq.modeling.criteo_package import FEATURES

# one float field per anonymized feature; unknown fields (treatment, outcomes, ...) are rejected
CriteoFeatures = create_model(
    "CriteoFeatures",
    __config__=ConfigDict(extra="forbid", json_schema_extra={"description": "Anonymized f0-f11"}),
    **{name: (float, Field(allow_inf_nan=False)) for name in FEATURES},
)


class CriteoRecord(CriteoFeatures):  # type: ignore[misc, valid-type]
    pass


class TargetRecord(CriteoFeatures):  # type: ignore[misc, valid-type]
    id: Annotated[
        int | None, Field(ge=0, description="optional customer id used for tie-breaking")
    ] = None


class Envelope(BaseModel):
    dataset: Literal["hillstrom", "criteo", "platform"]
    source: list[str]
    provenance: Literal["historical", "new_execution"]
    frozen_config_sha256: str | None = None
    caveats: list[str]
    data: Any


class Prediction(BaseModel):
    p_visit_control: float
    p_visit_treatment: float
    uplift: float
    recommend_treatment: bool
    model_version: str


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[CriteoRecord] = Field(min_length=1)


class BatchResponse(BaseModel):
    model_version: str
    n: int
    predictions: list[Prediction]
    caveats: list[str]


class TargetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[TargetRecord] = Field(min_length=1)
    budget: float = Field(gt=0, le=1, description="share of the supplied customers to contact")

    @model_validator(mode="after")
    def _ids(self):
        ids = [r.id for r in self.records]
        if any(i is not None for i in ids):
            if any(i is None for i in ids):
                raise ValueError("either every record has an id or none does")
            if len(set(ids)) != len(ids):
                raise ValueError("record ids must be unique")
        return self


class TargetItem(BaseModel):
    id: int
    rank: int = Field(description="1 = highest predicted uplift")
    selected: bool
    p_visit_control: float
    p_visit_treatment: float
    uplift: float


class TargetResponse(BaseModel):
    model_version: str
    n: int
    budget: float
    n_selected: int
    ranking: str
    items: list[TargetItem]
    caveats: list[str]


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    api_version: str
    criteo_model_loaded: bool
    model_version: str | None
    model_error: str | None
    reports_available: dict[str, bool]
