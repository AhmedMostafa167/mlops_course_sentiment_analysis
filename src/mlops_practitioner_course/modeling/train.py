"""Training loop for BertClassifier."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import asdict, dataclass

from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from mlops_practitioner_course.config import Settings, TrainingConfig

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed every RNG the training run touches, for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available")
    return torch.device(name)


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float | None
    val_accuracy: float | None
    seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


class Trainer:
    """Fine-tunes a classifier with AdamW, a linear LR schedule and gradient clipping."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
        settings: Settings | None = None,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
        # Only optimize trainable parameters, so freeze_bert=True works as expected.
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            eps=config.adam_eps,
        )

        if settings is not None:
            self.settings = settings
        elif isinstance(config, Settings):
            self.settings = config
            self.config = config.training
        else:
            try:
                self.settings = Settings.from_yaml()
            except Exception:
                self.settings = Settings()

        self.run_id: str | None = None
        mlflow.set_tracking_uri(self.settings.mlflow.tracking_uri)
        mlflow.set_experiment(self.settings.mlflow.experiment_name)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> list[EpochResult]:
        """Train for `config.epochs` epochs, evaluating on `val_loader` after each one."""
        with mlflow.start_run() as run:
            self.run_id = run.info.run_id

            # Log hyperparameters
            mlflow.log_params({
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "epochs": self.config.epochs,
                "model_name": self.settings.model.name,
                "max_length": self.settings.model.max_length,
                "seed": self.settings.seed,
                "optimizer_type": type(self.optimizer).__name__,
            })

            total_steps = len(train_loader) * self.config.epochs
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=total_steps,
            )
            logger.info(
                "Training on %s: %d epochs x %d batches = %d steps",
                self.device, self.config.epochs, len(train_loader), total_steps,
            )

            history = []
            for epoch in range(1, self.config.epochs + 1):
                start = time.perf_counter()
                train_loss = self._train_epoch(train_loader, scheduler, epoch)
                train_acc = getattr(self, "last_train_acc", 0.0)
                if val_loader is not None:
                    val_loss, val_accuracy, val_f1 = self.evaluate_with_f1(val_loader)
                else:
                    val_loss, val_accuracy, val_f1 = None, None, None

                result = EpochResult(epoch, train_loss, val_loss, val_accuracy, time.perf_counter() - start)
                history.append(result)

                # Log training and validation metrics for this epoch
                epoch_metrics = {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                }
                if val_loss is not None:
                    epoch_metrics["val_loss"] = val_loss
                if val_accuracy is not None:
                    epoch_metrics["val_acc"] = val_accuracy
                if val_f1 is not None:
                    epoch_metrics["val_f1"] = val_f1

                mlflow.log_metrics(epoch_metrics, step=epoch)

                if val_loader is not None:
                    logger.info(
                        "Epoch %d done | train loss %.4f | val loss %.4f | val acc %.2f%% | val f1 %.4f | %.0fs",
                        epoch, train_loss, val_loss, 100 * val_accuracy, val_f1, result.seconds,
                    )
                else:
                    logger.info("Epoch %d done | train loss %.4f | %.0fs", epoch, train_loss, result.seconds)

            logger.info("Training complete")

            # Post-training:
            # 1. Log saved evaluation artifacts if present (e.g., metrics_val.json, roc_val.png)
            run_dir = getattr(self.settings, "run_dir", None)
            if run_dir:
                run_path = Path(run_dir)
                for artifact_name in [
                    "metrics_val.json",
                    "roc_val.png",
                    "metrics_test.json",
                    "roc_test.png",
                    "history.json",
                ]:
                    artifact_path = run_path / artifact_name
                    if artifact_path.exists():
                        mlflow.log_artifact(str(artifact_path))

            # 2. Register the PyTorch model via mlflow.pytorch.log_model() under the registered model name
            reg_model_name = self.settings.mlflow.registered_model_name
            model_info = mlflow.pytorch.log_model(
                pytorch_model=self.model,
                artifact_path="model",
                registered_model_name=reg_model_name,
                serialization_format=mlflow.pytorch.SERIALIZATION_FORMAT_PICKLE,
            )

            # 3. Use MlflowClient to set the alias "champion" on the latest registered model version
            client = MlflowClient()
            model_version = getattr(model_info, "registered_model_version", None)
            if model_version is None:
                versions = client.search_model_versions(f"name='{reg_model_name}'")
                if versions:
                    sorted_versions = sorted(versions, key=lambda v: int(v.version))
                    model_version = sorted_versions[-1].version

            if model_version is not None:
                client.set_registered_model_alias(
                    name=reg_model_name,
                    alias="champion",
                    version=str(model_version),
                )
                logger.info(
                    "Set alias 'champion' on model '%s' version %s",
                    reg_model_name,
                    model_version,
                )

        return history

    def _train_epoch(self, loader: DataLoader, scheduler, epoch: int) -> float:
        self.model.train()
        total_loss, window_loss, window_steps = 0.0, 0.0, 0
        total_correct, total_samples = 0, 0
        window_start = time.perf_counter()

        for step, batch in enumerate(loader, start=1):
            input_ids, attention_mask, labels = (t.to(self.device) for t in batch)

            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.loss_fn(logits, labels)
            loss.backward()
            # Clip gradients to prevent exploding gradients.
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            window_loss += loss.item()
            window_steps += 1

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            if step % self.config.log_every == 0 or step == len(loader):
                logger.info(
                    "Epoch %d | batch %d/%d | train loss %.4f | %.1fs",
                    epoch, step, len(loader), window_loss / window_steps,
                    time.perf_counter() - window_start,
                )
                window_loss, window_steps = 0.0, 0
                window_start = time.perf_counter()

        self.last_train_acc = total_correct / total_samples if total_samples > 0 else 0.0
        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        """Return (mean loss, accuracy in [0, 1]) over every sample in `loader`."""
        val_loss, val_acc, _ = self.evaluate_with_f1(loader)
        return val_loss, val_acc

    @torch.no_grad()
    def evaluate_with_f1(self, loader: DataLoader) -> tuple[float, float, float]:
        """Return (mean loss, accuracy in [0, 1], f1) over every sample in `loader`."""
        self.model.eval()
        total_loss, correct, n_samples = 0.0, 0, 0
        all_preds = []
        all_labels = []

        for batch in loader:
            input_ids, attention_mask, labels = (t.to(self.device) for t in batch)
            logits = self.model(input_ids, attention_mask)
            # Weight by batch size so a smaller final batch doesn't skew the averages.
            total_loss += self.loss_fn(logits, labels).item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            n_samples += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        val_loss = total_loss / n_samples
        val_acc = correct / n_samples
        val_f1 = float(f1_score(all_labels, all_preds, zero_division=0))
        return val_loss, val_acc, val_f1
