"""Train the defect CNN and append an experiment record.

Run from the repo root:
    python scripts/train_defect_classifier.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inspectflow.config import load_config
from inspectflow.cv.classifier import CLASS_NAMES, DefectCNN
from inspectflow.data.augment import augment_crop
from inspectflow.data.synthetic import make_frame_crops
from inspectflow.metrics import confusion_matrix, precision_recall_f1
from inspectflow.schemas import ExperimentRecord
from inspectflow.tracking.experiments import ExperimentLogger, dataset_version


def main() -> None:
    config = load_config(ROOT / "configs" / "pipeline.yaml")
    seed = 7
    torch.manual_seed(seed)
    images, labels = make_frame_crops(frames=80, image_size=config.classifier.image_size, seed=100)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    images, labels = images[order], labels[order]
    split = int(len(labels) * 0.8)
    train_x = np.stack([augment_crop(image, rng) for image in images[:split]])[:, None, :, :]
    train_y = labels[:split]
    val_x, val_y = images[split:], labels[split:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DefectCNN(num_classes=len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x).float() / 255.0, torch.from_numpy(train_y).long()),
        batch_size=32,
        shuffle=True,
    )
    epochs = 4
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(val_x).float().to(device) / 255.0)
        predicted = torch.argmax(logits, dim=1).cpu().numpy().tolist()
    matrix = confusion_matrix(val_y.tolist(), predicted, num_classes=len(CLASS_NAMES))
    metrics = precision_recall_f1(matrix)

    weights = ROOT / config.classifier.weights
    weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), weights)

    logger = ExperimentLogger(ROOT / config.experiments.runs_path, ROOT / config.experiments.inspections_path)
    record = ExperimentRecord(
        run_id=logger.new_run_id(),
        task="defect-classification",
        data_version=dataset_version(images.tobytes()[:4096] + labels.tobytes()),
        model_name=config.classifier.name,
        model_version=config.classifier.version,
        params={"epochs": epochs, "lr": 1e-3, "image_size": config.classifier.image_size, "seed": seed},
        metrics=metrics,
        notes="Synthetic line crops from make_frame. Replace with labeled station crops before quoting accuracy.",
    )
    logger.log_run(record)
    print(record.model_dump())


if __name__ == "__main__":
    main()
