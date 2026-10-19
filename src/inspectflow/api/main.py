"""REST gateway: camera station in, ERP quality document out."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from inspectflow import __version__
from inspectflow.config import load_config
from inspectflow.cv.face import EnrollmentStore, embed_image
from inspectflow.pipeline import StationPipeline
from inspectflow.rag.assistant import answer_question, load_corpus
from inspectflow.schemas import ChatAnswer, ErpInspection

ROOT = Path.cwd()
CONFIG = load_config(ROOT / "configs" / "pipeline.yaml")
PIPELINE = StationPipeline(CONFIG, ROOT)
CORPUS = load_corpus(ROOT / CONFIG.rag.corpus_dir)
ENROLLMENTS = EnrollmentStore(ROOT / CONFIG.face.enrollment_dir)
ERP_INBOX: list[ErpInspection] = []

app = FastAPI(title="InspectFlow", version=__version__)


class ChatRequest(BaseModel):
    question: str


class VerifyResponse(BaseModel):
    operator_id: str | None
    score: float
    verified: bool


def _read_upload(payload: bytes) -> np.ndarray:
    array = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="image decode failed")
    return image


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__, "station_id": CONFIG.station_id}


@app.post("/api/v1/inspections/frame")
async def inspect_frame(
    image: UploadFile = File(...),
    expected_quantity: int | None = Form(default=None),
    expected_work_order_id: str | None = Form(default=None),
    operator_id: str | None = Form(default=None),
    operator_verified: bool | None = Form(default=None),
) -> dict:
    frame = _read_upload(await image.read())
    result = PIPELINE.inspect_frame(
        frame,
        expected_quantity=expected_quantity,
        expected_work_order_id=expected_work_order_id,
        operator_verified=operator_verified,
        operator_id=operator_id,
    )
    document = ErpInspection(
        station_id=result.station_id,
        work_order_id=result.work_order_id,
        decision=result.decision,
        counted_quantity=result.counted_quantity,
        expected_quantity=result.expected_quantity,
        defects=result.defects,
        model_versions=result.model_versions,
    )
    ERP_INBOX.append(document)
    return {"inspection": result.model_dump(mode="json"), "erp_document": document.model_dump(mode="json")}


@app.get("/api/v1/erp/inspections")
def list_erp_inspections() -> list[dict]:
    return [item.model_dump(mode="json") for item in ERP_INBOX]


@app.post("/api/v1/operators/enroll")
async def enroll_operator(operator_id: str = Form(...), image: UploadFile = File(...)) -> dict[str, str]:
    frame = _read_upload(await image.read())
    ENROLLMENTS.enroll(operator_id, embed_image(frame))
    return {"operator_id": operator_id, "stored": "embedding"}


@app.post("/api/v1/operators/verify", response_model=VerifyResponse)
async def verify_operator(image: UploadFile = File(...)) -> VerifyResponse:
    frame = _read_upload(await image.read())
    operator_id, score = ENROLLMENTS.identify(embed_image(frame), threshold=CONFIG.face.threshold)
    return VerifyResponse(operator_id=operator_id, score=score, verified=operator_id is not None)


@app.post("/api/v1/chat", response_model=ChatAnswer)
def chat(body: ChatRequest) -> ChatAnswer:
    return answer_question(body.question, CORPUS, top_k=CONFIG.rag.top_k)
