"""One station pass: detect, classify, read the work order, decide, log."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from inspectflow.config import PipelineConfig
from inspectflow.cv.classifier import classify_detections, load_classifier
from inspectflow.cv.detection import aggregate_quantity, detect_parts
from inspectflow.cv.ocr import read_work_order
from inspectflow.decisions import decide
from inspectflow.schemas import InspectionResult
from inspectflow.tracking.experiments import ExperimentLogger


class StationPipeline:
    def __init__(self, config: PipelineConfig, root: Path | None = None) -> None:
        self.config = config
        self.root = root or Path.cwd()
        weights = self.root / config.classifier.weights
        self.classifier = load_classifier(weights)
        self.logger = ExperimentLogger(
            self.root / config.experiments.runs_path,
            self.root / config.experiments.inspections_path,
        )

    def model_versions(self) -> dict[str, str]:
        classifier_name = self.config.classifier.name if self.classifier else "edge-variance"
        return {
            "detector": f"{self.config.detector.name}-{self.config.detector.version}",
            "classifier": f"{classifier_name}-{self.config.classifier.version}",
            "ocr": f"{self.config.ocr.name}-{self.config.ocr.version}",
            "face": f"{self.config.face.name}-{self.config.face.version}",
        }

    def inspect_frame(
        self,
        image: np.ndarray,
        *,
        expected_quantity: int | None = None,
        expected_work_order_id: str | None = None,
        operator_verified: bool | None = None,
        operator_id: str | None = None,
    ) -> InspectionResult:
        started = perf_counter()
        detections = detect_parts(
            image,
            min_area=self.config.detector.min_area,
            max_area=self.config.detector.max_area,
        )
        predictions = classify_detections(
            image,
            detections,
            model=self.classifier,
            image_size=self.config.classifier.image_size,
            scratch_edge_threshold=self.config.classifier.scratch_edge_threshold,
        )
        work_order_id = read_work_order(
            image,
            font_scale=self.config.ocr.font_scale,
            thickness=self.config.ocr.thickness,
        )
        if expected_work_order_id is None or work_order_id is None:
            work_order_matches = None
        else:
            work_order_matches = work_order_id == expected_work_order_id
        defect_count = sum(1 for item in predictions if item.label != "ok")
        decision = decide(
            counted=len(detections),
            expected_quantity=expected_quantity,
            defect_count=defect_count,
            work_order_matches=work_order_matches,
            operator_verified=operator_verified,
        )
        result = InspectionResult(
            station_id=self.config.station_id,
            work_order_id=work_order_id,
            expected_work_order_id=expected_work_order_id,
            operator_id=operator_id,
            counted_quantity=len(detections),
            expected_quantity=expected_quantity,
            defects=[item for item in predictions if item.label != "ok"],
            decision=decision,
            latency_ms=(perf_counter() - started) * 1000,
            model_versions=self.model_versions(),
        )
        self.logger.log_inspection(result)
        return result

    def inspect_frames(self, frames: list[np.ndarray], *, expected_quantity: int | None = None) -> InspectionResult:
        counts: list[int] = []
        last: InspectionResult | None = None
        for frame in frames:
            last = self.inspect_frame(frame, expected_quantity=expected_quantity)
            counts.append(last.counted_quantity)
        if last is None:
            raise ValueError("video sample is empty")
        counted = aggregate_quantity(counts)
        decision = decide(
            counted=counted,
            expected_quantity=expected_quantity,
            defect_count=len(last.defects),
            work_order_matches=None,
            operator_verified=None,
        )
        return last.model_copy(
            update={
                "counted_quantity": counted,
                "decision": decision,
                "frame_counts": counts,
            }
        )
