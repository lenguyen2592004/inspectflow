"""Synthetic line frames so the pipeline can be reproduced without factory data.

Swap this generator for labeled camera data using the same crop and box schema.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from inspectflow.schemas import BoundingBox

FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass(frozen=True)
class SyntheticPart:
    bbox: BoundingBox
    label: str


@dataclass(frozen=True)
class SyntheticFrame:
    image: np.ndarray
    work_order_id: str
    parts: list[SyntheticPart]


def render_text_patch(text: str, font_scale: float = 1.0, thickness: int = 2) -> np.ndarray:
    (width, height), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    patch = np.zeros((height + baseline + 8, width + 8), dtype=np.uint8)
    cv2.putText(patch, text, (4, height + 2), FONT, font_scale, 255, thickness, cv2.LINE_8)
    return patch


def make_frame(
    *,
    work_order_id: str = "WO-1001",
    ok_count: int = 2,
    scratch_count: int = 1,
    seed: int = 0,
    font_scale: float = 1.0,
    thickness: int = 2,
) -> SyntheticFrame:
    rng = np.random.default_rng(seed)
    canvas = np.full((480, 640, 3), 40, dtype=np.uint8)
    label = render_text_patch(work_order_id, font_scale=font_scale, thickness=thickness)
    canvas[12 : 12 + label.shape[0], 16 : 16 + label.shape[1], :] = label[:, :, None]

    parts: list[SyntheticPart] = []
    total = ok_count + scratch_count
    for index in range(total):
        label_name = "scratch" if index < scratch_count else "ok"
        width = int(rng.integers(90, 130))
        height = int(rng.integers(70, 110))
        x = 20 + (index % 4) * 150
        y = 120 + (index // 4) * 150
        color = (30, 90, 180) if label_name == "ok" else (40, 140, 200)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), color, thickness=-1)
        if label_name == "scratch":
            for _ in range(6):
                x1 = int(rng.integers(x + 4, x + width - 4))
                y1 = int(rng.integers(y + 4, y + height - 4))
                x2 = int(np.clip(x1 + rng.integers(-30, 30), x + 2, x + width - 2))
                y2 = int(np.clip(y1 + rng.integers(-20, 20), y + 2, y + height - 2))
                cv2.line(canvas, (x1, y1), (x2, y2), (250, 250, 250), 2, cv2.LINE_8)
        parts.append(SyntheticPart(bbox=BoundingBox(x=x, y=y, width=width, height=height), label=label_name))
    return SyntheticFrame(image=canvas, work_order_id=work_order_id, parts=parts)


def make_frame_crops(
    *,
    frames: int = 40,
    image_size: int = 64,
    seed: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Crops from the same renderer the station uses. Label 0 is ok, 1 is scratch."""
    images: list[np.ndarray] = []
    labels: list[int] = []
    for index in range(frames):
        frame = make_frame(ok_count=2, scratch_count=1, seed=seed + index)
        for part in frame.parts:
            box = part.bbox
            crop = frame.image[box.y : box.y + box.height, box.x : box.x + box.width]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
            images.append(resized)
            labels.append(0 if part.label == "ok" else 1)
    stacked = np.stack(images)[:, None, :, :]
    return stacked, np.asarray(labels, dtype=np.int64)


def make_crop_dataset(
    *,
    per_class: int = 80,
    image_size: int = 64,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return grayscale crops shaped (N, 1, S, S) and labels: 0 ok, 1 scratch."""
    rng = np.random.default_rng(seed)
    images: list[np.ndarray] = []
    labels: list[int] = []
    for class_index, name in ((0, "ok"), (1, "scratch")):
        for _ in range(per_class):
            crop = np.full((image_size, image_size), int(rng.integers(70, 150)), dtype=np.uint8)
            if name == "scratch":
                for _stroke in range(5):
                    p1 = (int(rng.integers(2, image_size - 2)), int(rng.integers(2, image_size - 2)))
                    p2 = (int(rng.integers(2, image_size - 2)), int(rng.integers(2, image_size - 2)))
                    cv2.line(crop, p1, p2, 245, 1, cv2.LINE_8)
            images.append(crop)
            labels.append(class_index)
    stacked = np.stack(images)[:, None, :, :]
    return stacked, np.asarray(labels, dtype=np.int64)
