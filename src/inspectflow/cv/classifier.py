"""Defect classifier: a small PyTorch CNN plus an OpenCV edge-density fallback."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from inspectflow.schemas import DefectPrediction

CLASS_NAMES = ("ok", "scratch")


class DefectCNN(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(inputs))


def edge_variance(crop_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def predict_crop(
    crop_bgr: np.ndarray,
    *,
    model: DefectCNN | None,
    image_size: int,
    scratch_edge_threshold: float,
    device: torch.device | None = None,
) -> tuple[str, float]:
    if model is None:
        score = edge_variance(crop_bgr)
        label = "scratch" if score >= scratch_edge_threshold else "ok"
        confidence = min(score / (scratch_edge_threshold * 2), 0.99) if label == "scratch" else 0.5
        return label, confidence

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    model.eval()
    with torch.no_grad():
        logits = model(tensor.to(device or torch.device("cpu")))
        probabilities = torch.softmax(logits, dim=1)[0]
    index = int(torch.argmax(probabilities).item())
    return CLASS_NAMES[index], float(probabilities[index].item())


def load_classifier(weights: Path, device: torch.device | None = None) -> DefectCNN | None:
    if not weights.exists():
        return None
    model = DefectCNN()
    state = torch.load(weights, map_location=device or torch.device("cpu"), weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def classify_detections(
    image: np.ndarray,
    boxes: list,
    *,
    model: DefectCNN | None,
    image_size: int,
    scratch_edge_threshold: float,
) -> list[DefectPrediction]:
    predictions: list[DefectPrediction] = []
    height, width = image.shape[:2]
    for detection in boxes:
        box = detection.bbox
        x2 = min(box.x + box.width, width)
        y2 = min(box.y + box.height, height)
        crop = image[box.y : y2, box.x : x2]
        if crop.size == 0:
            continue
        label, score = predict_crop(
            crop,
            model=model,
            image_size=image_size,
            scratch_edge_threshold=scratch_edge_threshold,
        )
        predictions.append(DefectPrediction(label=label, score=score, bbox=box))
    return predictions
