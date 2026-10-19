"""Export the trained defect CNN for an edge runtime.

    python scripts/export_onnx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inspectflow.config import load_config
from inspectflow.cv.classifier import load_classifier


def main() -> None:
    config = load_config(ROOT / "configs" / "pipeline.yaml")
    weights = ROOT / config.classifier.weights
    model = load_classifier(weights)
    if model is None:
        raise SystemExit(f"missing weights at {weights}; train first")
    size = config.classifier.image_size
    dummy = torch.zeros(1, 1, size, size)
    destination = ROOT / "artifacts" / "defect_cnn.onnx"
    model.eval()
    torch.onnx.export(model, dummy, destination, input_names=["crop"], output_names=["logits"], dynamo=False)
    print(destination)


if __name__ == "__main__":
    main()
