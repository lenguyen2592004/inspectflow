"""Held-out station benchmark. Writes results/benchmark.json.

Training frames use seeds 100-179. This script uses seeds 1000 and above.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inspectflow.config import load_config
from inspectflow.cv.classifier import load_classifier, predict_crop
from inspectflow.cv.detection import aggregate_quantity, detect_parts
from inspectflow.cv.face import cosine, embed_image
from inspectflow.cv.ocr import read_work_order
from inspectflow.data.synthetic import make_frame
from inspectflow.decisions import decide
from inspectflow.metrics import confusion_matrix, detection_scores, precision_recall_f1
from inspectflow.pipeline import StationPipeline
from inspectflow.schemas import Decision

WORK_ORDERS = ("WO-1001", "WO-2048", "WO-3370", "AB-9152")


def camera_noise(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    alpha = float(rng.uniform(0.85, 1.15))
    beta = float(rng.integers(-12, 13))
    noisy = np.clip(image.astype(np.float32) * alpha + beta, 0, 255)
    blurred = cv2.GaussianBlur(noisy, (3, 3), 0)
    grain = rng.normal(0, 3, blurred.shape)
    return np.clip(blurred + grain, 0, 255).astype(np.uint8)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def evaluate_station(pipeline: StationPipeline, *, frames: int, seed: int, perturb: bool) -> dict[str, float]:
    config = pipeline.config
    rng = np.random.default_rng(seed)
    det_tp = det_fp = det_fn = 0
    count_hits = 0
    ocr_hits = 0
    ocr_total = 0
    decision_hits = 0
    latencies: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []

    for index in range(frames):
        scratch = int(rng.integers(0, 2))
        ok = int(rng.integers(1, 4))
        work_order = WORK_ORDERS[int(rng.integers(0, len(WORK_ORDERS)))]
        frame = make_frame(
            work_order_id=work_order,
            ok_count=ok,
            scratch_count=scratch,
            seed=seed + index,
        )
        image = camera_noise(frame.image, rng) if perturb else frame.image
        expected_quantity = ok + scratch
        if rng.random() < 0.25 and scratch == 0:
            expected_quantity += 1
        expected_work_order = work_order
        if rng.random() < 0.2 and scratch == 0 and expected_quantity == ok + scratch:
            expected_work_order = "WO-0000"

        result = pipeline.inspect_frame(
            image,
            expected_quantity=expected_quantity,
            expected_work_order_id=expected_work_order,
        )
        latencies.append(result.latency_ms)
        counted_true = ok + scratch
        count_hits += int(result.counted_quantity == counted_true)
        boxes = detect_parts(image, min_area=config.detector.min_area, max_area=config.detector.max_area)
        scores = detection_scores(
            [item.bbox for item in boxes],
            [part.bbox for part in frame.parts],
        )
        det_tp += int(scores["true_positive"])
        det_fp += int(scores["false_positive"])
        det_fn += int(scores["false_negative"])

        ocr_total += 1
        ocr_hits += int(result.work_order_id == work_order)

        truth = decide(
            counted=counted_true,
            expected_quantity=expected_quantity,
            defect_count=scratch,
            work_order_matches=work_order == expected_work_order,
            operator_verified=None,
        )
        decision_hits += int(result.decision is truth)

        model = pipeline.classifier
        for part in frame.parts:
            box = part.bbox
            crop = image[box.y : box.y + box.height, box.x : box.x + box.width]
            label, _score = predict_crop(
                crop,
                model=model,
                image_size=config.classifier.image_size,
                scratch_edge_threshold=config.classifier.scratch_edge_threshold,
            )
            y_true.append(0 if part.label == "ok" else 1)
            y_pred.append(0 if label == "ok" else 1)

    precision = det_tp / (det_tp + det_fp) if det_tp + det_fp else 0.0
    recall = det_tp / (det_tp + det_fn) if det_tp + det_fn else 0.0
    det_f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    cls = precision_recall_f1(confusion_matrix(y_true, y_pred, num_classes=2))
    return {
        "frames": float(frames),
        "detection_precision_iou50": precision,
        "detection_recall_iou50": recall,
        "detection_f1_iou50": det_f1,
        "count_exact_match": count_hits / frames,
        "ocr_exact_match": ocr_hits / ocr_total if ocr_total else 0.0,
        "decision_accuracy": decision_hits / frames,
        "defect_f1_macro": cls["f1_macro"],
        "defect_accuracy": cls["accuracy"],
        "latency_ms_median": _median(latencies),
    }


def evaluate_video(config_min_area: int, config_max_area: int) -> dict[str, float]:
    hits = 0
    trials = 30
    for index in range(trials):
        full = make_frame(ok_count=3, scratch_count=0, seed=3000 + index)
        partial = make_frame(ok_count=1, scratch_count=0, seed=4000 + index)
        counts = []
        for frame in (full, full, partial, full, full):
            counts.append(
                len(detect_parts(frame.image, min_area=config_min_area, max_area=config_max_area))
            )
        hits += int(aggregate_quantity(counts) == 3)
    return {"video_count_exact_match": hits / trials, "video_trials": float(trials)}


def evaluate_faces(threshold: float) -> dict[str, float]:
    """1:1 verification. Impostors are noisy probes of a different identity."""
    rng = np.random.default_rng(11)
    identities = 12
    enrolled: list[np.ndarray] = []
    probes: list[np.ndarray] = []
    for identity in range(identities):
        enrolled_card = _operator_card(identity, rng)
        probe_card = _operator_card(identity, rng)
        enrolled.append(embed_image(cv2.cvtColor(enrolled_card, cv2.COLOR_GRAY2BGR)))
        probes.append(embed_image(cv2.cvtColor(probe_card, cv2.COLOR_GRAY2BGR)))
    genuine = [cosine(enrolled[index], probes[index]) for index in range(identities)]
    impostor = [
        cosine(enrolled[left], probes[right])
        for left in range(identities)
        for right in range(identities)
        if left != right
    ]
    tar = sum(score >= threshold for score in genuine) / len(genuine)
    far = sum(score >= threshold for score in impostor) / len(impostor)
    return {
        "identities": float(identities),
        "impostor_pairs": float(len(impostor)),
        "tar": tar,
        "far": far,
        "genuine_min": min(genuine),
        "impostor_max": max(impostor),
        "threshold": threshold,
    }


def _operator_card(identity: int, rng: np.random.Generator) -> np.ndarray:
    """Fixed mosaic per identity, plus sensor noise on the probe."""
    local = np.random.default_rng(identity + 50)
    mosaic = (local.integers(0, 2, size=(8, 8)) * 255).astype(np.uint8)
    image = cv2.resize(mosaic, (128, 128), interpolation=cv2.INTER_NEAREST)
    grain = rng.normal(0, 6, image.shape)
    return np.clip(image.astype(np.float32) + grain, 0, 255).astype(np.uint8)


def main() -> None:
    config = load_config(ROOT / "configs" / "pipeline.yaml")
    config.experiments.runs_path = "experiments/benchmark-runs.jsonl"
    config.experiments.inspections_path = "experiments/benchmark-inspections.jsonl"
    weights = ROOT / config.classifier.weights
    if load_classifier(weights) is None:
        raise SystemExit("train the defect CNN first: python scripts/train_defect_classifier.py")
    pipeline = StationPipeline(config, ROOT)
    clean = evaluate_station(pipeline, frames=80, seed=1000, perturb=False)
    noisy = evaluate_station(pipeline, frames=80, seed=2000, perturb=True)
    report = {
        "dataset": "synthetic line frames, held-out seeds 1000+ (train used 100-179)",
        "classifier": f"{config.classifier.name}-{config.classifier.version}",
        "detector": f"{config.detector.name}-{config.detector.version}",
        "clean": clean,
        "camera_noise": noisy,
        "video": evaluate_video(config.detector.min_area, config.detector.max_area),
        "operator_verification": evaluate_faces(config.face.threshold),
    }
    destination = ROOT / "results" / "benchmark.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
