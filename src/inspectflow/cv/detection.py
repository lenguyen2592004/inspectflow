"""Object detection baseline: contour parts on a line camera frame.

Replace `detect_parts` with a YOLO/RT-DETR adapter that returns the same boxes
once a labeled set exists. Counting stays downstream of the boxes.
"""

from __future__ import annotations

import cv2
import numpy as np

from inspectflow.schemas import BoundingBox, Detection


def detect_parts(image: np.ndarray, *, min_area: int, max_area: int) -> list[Detection]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[Detection] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        detections.append(
            Detection(bbox=BoundingBox(x=x, y=y, width=width, height=height), score=1.0, label="part")
        )
    detections.sort(key=lambda item: (item.bbox.y, item.bbox.x))
    return detections


def aggregate_quantity(frame_counts: list[int]) -> int:
    """Stable count across sampled video frames. Median resists a partial frame."""
    if not frame_counts:
        return 0
    ordered = sorted(frame_counts)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return int(round((ordered[mid - 1] + ordered[mid]) / 2))
