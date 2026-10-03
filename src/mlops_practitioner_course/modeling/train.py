"""Training loop for BertClassifier."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from mlops_practitioner_course.config import TrainingConfig

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

    def __init__(self, model: nn.Module, config: TrainingConfig, device: torch.device) -> None:
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

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> list[EpochResult]:
        """Train for `config.epochs` epochs, evaluating on `val_loader` after each one."""
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
            val_loss, val_accuracy = self.evaluate(val_loader) if val_loader is not None else (None, None)
            result = EpochResult(epoch, train_loss, val_loss, val_accuracy, time.perf_counter() - start)
            history.append(result)

            if val_loader is not None:
                logger.info(
                    "Epoch %d done | train loss %.4f | val loss %.4f | val acc %.2f%% | %.0fs",
                    epoch, train_loss, val_loss, 100 * val_accuracy, result.seconds,
                )
            else:
                logger.info("Epoch %d done | train loss %.4f | %.0fs", epoch, train_loss, result.seconds)

        logger.info("Training complete")
        return history

    def _train_epoch(self, loader: DataLoader, scheduler, epoch: int) -> float:
        self.model.train()
        total_loss, window_loss, window_steps = 0.0, 0.0, 0
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

            if step % self.config.log_every == 0 or step == len(loader):
                logger.info(
                    "Epoch %d | batch %d/%d | train loss %.4f | %.1fs",
                    epoch, step, len(loader), window_loss / window_steps,
                    time.perf_counter() - window_start,
                )
                window_loss, window_steps = 0.0, 0
                window_start = time.perf_counter()

        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        """Return (mean loss, accuracy in [0, 1]) over every sample in `loader`."""
        self.model.eval()
        total_loss, correct, n_samples = 0.0, 0, 0

        for batch in loader:
            input_ids, attention_mask, labels = (t.to(self.device) for t in batch)
            logits = self.model(input_ids, attention_mask)
            # Weight by batch size so a smaller final batch doesn't skew the averages.
            total_loss += self.loss_fn(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            n_samples += labels.size(0)

        return total_loss / n_samples, correct / n_samples
