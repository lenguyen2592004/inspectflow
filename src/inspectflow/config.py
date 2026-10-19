"""Load the station pipeline config."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class DetectorConfig(BaseModel):
    name: str
    version: str
    min_area: int
    max_area: int


class ClassifierConfig(BaseModel):
    name: str
    version: str
    weights: str
    image_size: int
    scratch_edge_threshold: float


class OcrConfig(BaseModel):
    name: str
    version: str
    font_scale: float
    thickness: int


class FaceConfig(BaseModel):
    name: str
    version: str
    threshold: float
    enrollment_dir: str


class VideoConfig(BaseModel):
    sample_every_n_frames: int


class ExperimentConfig(BaseModel):
    runs_path: str
    inspections_path: str


class RagConfig(BaseModel):
    corpus_dir: str
    top_k: int


class PipelineConfig(BaseModel):
    station_id: str
    detector: DetectorConfig
    classifier: ClassifierConfig
    ocr: OcrConfig
    face: FaceConfig
    video: VideoConfig
    experiments: ExperimentConfig
    rag: RagConfig


def load_config(path: Path | None = None) -> PipelineConfig:
    config_path = path or Path("configs/pipeline.yaml")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return PipelineConfig.model_validate(raw)
