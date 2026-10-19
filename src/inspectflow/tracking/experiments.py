"""Append-only experiment and inspection logs so a run can be repeated."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from inspectflow.schemas import ExperimentRecord, InspectionResult


def dataset_version(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:12]


class ExperimentLogger:
    def __init__(self, runs_path: Path, inspections_path: Path) -> None:
        self.runs_path = runs_path
        self.inspections_path = inspections_path
        self.runs_path.parent.mkdir(parents=True, exist_ok=True)
        self.inspections_path.parent.mkdir(parents=True, exist_ok=True)

    def log_run(self, record: ExperimentRecord) -> ExperimentRecord:
        row = record.model_dump()
        row["logged_at"] = datetime.now(UTC).isoformat()
        with self.runs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        return record

    def new_run_id(self) -> str:
        return uuid4().hex[:12]

    def log_inspection(self, result: InspectionResult) -> None:
        row = result.model_dump(mode="json")
        row["logged_at"] = datetime.now(UTC).isoformat()
        with self.inspections_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
