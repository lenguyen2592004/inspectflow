from pathlib import Path

import cv2
import numpy as np

from inspectflow.config import load_config
from inspectflow.cv.detection import aggregate_quantity, detect_parts
from inspectflow.cv.face import EnrollmentStore, cosine, embed_image
from inspectflow.cv.ocr import read_work_order
from inspectflow.data.synthetic import make_frame
from inspectflow.decisions import decide
from inspectflow.metrics import box_iou, confusion_matrix, detection_scores, precision_recall_f1
from inspectflow.schemas import BoundingBox
from inspectflow.pipeline import StationPipeline
from inspectflow.schemas import Decision


ROOT = Path(__file__).resolve().parents[1]


def test_detection_iou_matching() -> None:
    left = BoundingBox(x=0, y=0, width=10, height=10)
    right = BoundingBox(x=0, y=0, width=10, height=10)
    shifted = BoundingBox(x=20, y=20, width=10, height=10)
    assert box_iou(left, right) == 1.0
    scores = detection_scores([left, shifted], [right])
    assert scores["true_positive"] == 1
    assert scores["false_positive"] == 1
    assert scores["false_negative"] == 0


def test_metrics_perfect_split() -> None:
    matrix = confusion_matrix([0, 0, 1, 1], [0, 0, 1, 1], num_classes=2)
    scores = precision_recall_f1(matrix)
    assert scores["accuracy"] == 1.0
    assert scores["f1_macro"] == 1.0


def test_decision_rejects_scratch_and_reviews_count_gap() -> None:
    assert decide(counted=3, expected_quantity=3, defect_count=1, work_order_matches=True, operator_verified=True) is Decision.reject
    assert decide(counted=2, expected_quantity=3, defect_count=0, work_order_matches=True, operator_verified=True) is Decision.review
    assert decide(counted=3, expected_quantity=3, defect_count=0, work_order_matches=True, operator_verified=True) is Decision.accept


def test_synthetic_frame_counts_and_reads_work_order() -> None:
    frame = make_frame(work_order_id="WO-1001", ok_count=2, scratch_count=1, seed=1)
    config = load_config(ROOT / "configs" / "pipeline.yaml")
    found = detect_parts(frame.image, min_area=config.detector.min_area, max_area=config.detector.max_area)
    assert len(found) == 3
    assert read_work_order(frame.image, font_scale=config.ocr.font_scale, thickness=config.ocr.thickness) == "WO-1001"


def test_station_pipeline_rejects_scratched_lot(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "pipeline.yaml")
    config.classifier.weights = str(tmp_path / "missing.pt")
    config.experiments.runs_path = str(tmp_path / "runs.jsonl")
    config.experiments.inspections_path = str(tmp_path / "inspections.jsonl")
    pipeline = StationPipeline(config, ROOT)
    frame = make_frame(ok_count=2, scratch_count=1, seed=2)
    result = pipeline.inspect_frame(frame.image, expected_quantity=3, expected_work_order_id="WO-1001")
    assert result.counted_quantity == 3
    assert result.decision is Decision.reject
    assert result.defects
    assert (tmp_path / "inspections.jsonl").exists()


def test_video_count_uses_median() -> None:
    assert aggregate_quantity([3, 3, 1, 3, 3]) == 3


def test_face_enrollment_matches_same_pattern(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    pattern = rng.integers(0, 255, size=(96, 96), dtype=np.uint8)
    image = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)
    other = cv2.cvtColor(rng.integers(0, 255, size=(96, 96), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    store = EnrollmentStore(tmp_path)
    embedding = embed_image(image)
    store.enroll("OP-01", embedding)
    matched, score = store.identify(embed_image(image), threshold=0.9)
    rejected, _ = store.identify(embed_image(other), threshold=0.99)
    assert matched == "OP-01"
    assert score > 0.9
    assert rejected is None
    assert cosine(embedding, embedding) > 0.99
