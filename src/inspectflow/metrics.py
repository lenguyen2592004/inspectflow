"""Evaluation metrics for classification and single-class detection."""

from __future__ import annotations

import numpy as np

from inspectflow.schemas import BoundingBox


def confusion_matrix(y_true: list[int], y_pred: list[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred, strict=True):
        matrix[truth, pred] += 1
    return matrix


def precision_recall_f1(matrix: np.ndarray) -> dict[str, float]:
    """Macro precision, recall, and F1. Empty classes contribute zeros."""
    num_classes = matrix.shape[0]
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for index in range(num_classes):
        tp = float(matrix[index, index])
        predicted = float(matrix[:, index].sum())
        actual = float(matrix[index, :].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    correct = float(np.trace(matrix))
    total = float(matrix.sum())
    return {
        "accuracy": correct / total if total else 0.0,
        "precision_macro": float(np.mean(precisions)),
        "recall_macro": float(np.mean(recalls)),
        "f1_macro": float(np.mean(f1s)),
    }


def box_iou(left: BoundingBox, right: BoundingBox) -> float:
    left_x2 = left.x + left.width
    left_y2 = left.y + left.height
    right_x2 = right.x + right.width
    right_y2 = right.y + right.height
    inter_w = max(0, min(left_x2, right_x2) - max(left.x, right.x))
    inter_h = max(0, min(left_y2, right_y2) - max(left.y, right.y))
    intersection = inter_w * inter_h
    union = left.width * left.height + right.width * right.height - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def detection_scores(
    predictions: list[BoundingBox],
    truths: list[BoundingBox],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Greedy one-to-one matching. Unmatched boxes count as false positives or misses."""
    remaining = set(range(len(truths)))
    true_positive = 0
    for prediction in predictions:
        best_index = None
        best_iou = iou_threshold
        for index in remaining:
            overlap = box_iou(prediction, truths[index])
            if overlap >= best_iou:
                best_iou = overlap
                best_index = index
        if best_index is not None:
            true_positive += 1
            remaining.remove(best_index)
    false_positive = len(predictions) - true_positive
    false_negative = len(remaining)
    precision = true_positive / (true_positive + false_positive) if predictions else 0.0
    recall = true_positive / (true_positive + false_negative) if truths else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": float(true_positive),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
    }
