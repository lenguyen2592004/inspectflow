"""Fixed-font OCR for inkjet work-order labels.

Factory printers repeat one font and size. Template matching is the baseline.
Swap in Tesseract, PaddleOCR, or a CRNN when labels vary.
"""

from __future__ import annotations

import cv2
import numpy as np

from inspectflow.data.synthetic import render_text_patch

CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"


def _templates(font_scale: float, thickness: int) -> dict[str, np.ndarray]:
    templates: dict[str, np.ndarray] = {}
    for char in CHARSET:
        patch = render_text_patch(char, font_scale=font_scale, thickness=thickness)
        ys, xs = np.where(patch > 0)
        if len(xs) == 0:
            continue
        templates[char] = patch[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    return templates


def _similarity(crop: np.ndarray, template: np.ndarray) -> float:
    resized = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
    reference = cv2.resize(template, (32, 32), interpolation=cv2.INTER_AREA)
    left = resized.astype(np.float32).ravel()
    right = reference.astype(np.float32).ravel()
    denom = float(np.linalg.norm(left) * np.linalg.norm(right) + 1e-6)
    return float(np.dot(left, right) / denom)


def read_work_order(image: np.ndarray, *, font_scale: float = 1.0, thickness: int = 2) -> str | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    banner = gray[:90, :]
    _, binary = cv2.threshold(banner, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if height < 2 or width < 2:
            continue
        candidates.append((x, y, width, height))
    letters = [box for box in candidates if box[3] >= 8]
    if not letters:
        return None
    line_y = float(np.median([box[1] + box[3] / 2 for box in letters]))
    line_h = float(np.median([box[3] for box in letters]))
    boxes = [
        box
        for box in candidates
        if box[3] >= 8 or abs((box[1] + box[3] / 2) - line_y) <= line_h * 0.6
    ]
    boxes.sort(key=lambda item: item[0])
    templates = _templates(font_scale, thickness)
    chars: list[str] = []
    for x, y, width, height in boxes:
        crop = binary[y : y + height, x : x + width]
        ranked = sorted((( _similarity(crop, patch), char) for char, patch in templates.items()), reverse=True)
        score, char = ranked[0]
        if score < 0.45:
            continue
        chars.append(char)
    text = "".join(chars)
    return text or None
