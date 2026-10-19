"""Consented operator check at the station.

The baseline embeds a face crop with HOG and matches cosine similarity against
embeddings enrolled on site. Raw photos are not stored. Upgrade the embedder
to ArcFace or FaceNet without changing enrollment or the verify API.

Only enroll people who agreed to identification at this station.
The shipped threshold is 0.9, chosen so enrolled mosaic probes stay above it
and other identities stay below it in `scripts/evaluate.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_face(image_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = _CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        return None
    x, y, width, height = max(faces, key=lambda item: item[2] * item[3])
    return int(x), int(y), int(width), int(height)


def embed_crop(gray_face: np.ndarray) -> np.ndarray:
    resized = cv2.resize(gray_face, (64, 64), interpolation=cv2.INTER_AREA)
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    vector = hog.compute(resized).reshape(-1).astype(np.float32)
    norm = float(np.linalg.norm(vector) + 1e-6)
    return vector / norm


def embed_image(image_bgr: np.ndarray) -> np.ndarray:
    """Embed a full frame, or the whole image when it is already a face crop."""
    box = detect_face(image_bgr)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if box is None:
        return embed_crop(gray)
    x, y, width, height = box
    return embed_crop(gray[y : y + height, x : x + width])


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-6))


class EnrollmentStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def enroll(self, operator_id: str, embedding: np.ndarray) -> None:
        path = self.directory / f"{operator_id}.json"
        payload = {"operator_id": operator_id, "embedding": embedding.astype(float).tolist()}
        path.write_text(json.dumps(payload), encoding="utf-8")

    def identify(self, embedding: np.ndarray, *, threshold: float) -> tuple[str | None, float]:
        best_id: str | None = None
        best_score = -1.0
        for path in self.directory.glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            score = cosine(embedding, np.asarray(payload["embedding"], dtype=np.float32))
            if score > best_score:
                best_score = score
                best_id = payload["operator_id"]
        if best_id is None or best_score < threshold:
            return None, best_score
        return best_id, best_score
