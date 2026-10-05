"""Project settings, loaded from config/config.yaml and validated with pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

MODEL_NAMES = {
    "mini": "asafaya/bert-mini-arabic",
    "base": "asafaya/bert-base-arabic",
}


def _resolve_from_root(path: Path) -> Path:
    # Relative paths are relative to the project root, not the working directory.
    return path if path.is_absolute() else PROJECT_ROOT / path


class DataConfig(BaseModel):
    data_dir: Path = Path("data")
    val_size: float = Field(0.1, gt=0, lt=1)

    _resolve_data_dir = field_validator("data_dir")(_resolve_from_root)


class PreprocessingConfig(BaseModel):
    remove_emojis: bool = True


class ModelConfig(BaseModel):
    version: Literal["mini", "base"] = "mini"
    max_length: int = Field(128, gt=0, le=512)
    hidden_dim: int = Field(50, gt=0)
    dropout: float = Field(0.5, ge=0, lt=1)
    freeze_bert: bool = False

    @property
    def name(self) -> str:
        return MODEL_NAMES[self.version]


class TrainingConfig(BaseModel):
    batch_size: int = Field(16, gt=0)
    epochs: int = Field(2, gt=0)
    learning_rate: float = Field(5e-5, gt=0)
    adam_eps: float = Field(1e-8, gt=0)
    warmup_steps: int = Field(0, ge=0)
    max_grad_norm: float = Field(1.0, gt=0)
    log_every: int = Field(20, gt=0)
    device: Literal["auto", "cpu", "cuda"] = "auto"


class EvaluationConfig(BaseModel):
    threshold: float = Field(0.5, gt=0, lt=1)


class ArtifactsConfig(BaseModel):
    output_dir: Path = Path("models")

    _resolve_output_dir = field_validator("output_dir")(_resolve_from_root)


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class MLflowConfig(BaseModel):
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "arabic-sentiment-bert"
    registered_model_name: str = "arabic_sentiment_champion"


class Settings(BaseModel):
    seed: int = 2020
    data: DataConfig = DataConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    artifacts: ArtifactsConfig = ArtifactsConfig()
    logging: LoggingConfig = LoggingConfig()
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)

    @property
    def run_dir(self) -> Path:
        """Where this run's checkpoint, metrics and plots are saved."""
        return self.artifacts.output_dir / f"bert-{self.model.version}"

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> Settings:
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f) or {})
