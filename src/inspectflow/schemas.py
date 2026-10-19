"""Contracts shared by the vision pipeline, the API, and the ERP inbox."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Decision(str, Enum):
    accept = "accept"
    reject = "reject"
    review = "review"


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class Detection(BaseModel):
    bbox: BoundingBox
    score: float = 1.0
    label: str = "part"


class DefectPrediction(BaseModel):
    label: Literal["ok", "scratch"]
    score: float
    bbox: BoundingBox


class InspectionResult(BaseModel):
    station_id: str
    work_order_id: str | None
    expected_work_order_id: str | None = None
    operator_id: str | None = None
    counted_quantity: int
    expected_quantity: int | None = None
    defects: list[DefectPrediction]
    decision: Decision
    latency_ms: float
    model_versions: dict[str, str]
    frame_counts: list[int] = Field(default_factory=list)


class ErpInspection(BaseModel):
    """Quality-inspection document an ERP module would receive."""

    station_id: str
    work_order_id: str | None
    decision: Decision
    counted_quantity: int
    expected_quantity: int | None
    defects: list[DefectPrediction]
    model_versions: dict[str, str]


class ChatAnswer(BaseModel):
    answer: str
    citations: list[str]


class ExperimentRecord(BaseModel):
    run_id: str
    task: str
    data_version: str
    model_name: str
    model_version: str
    params: dict[str, float | int | str]
    metrics: dict[str, float]
    notes: str = ""
