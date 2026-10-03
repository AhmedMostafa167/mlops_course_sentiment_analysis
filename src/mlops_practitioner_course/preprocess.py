"""Tweet cleaning and BERT tokenization."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Sequence

import emoji
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from mlops_practitioner_course.config import MODEL_NAMES, Settings

logger = logging.getLogger(__name__)


class TweetCleaner:
    """Normalizes raw tweet text before tokenization. Call it like a function."""

    MENTION_RE = re.compile(r"@\w+")
    URL_RE = re.compile(r"https?://\S+")
    WHITESPACE_RE = re.compile(r"\s+")

    def __init__(self, remove_emojis: bool = True, url_token: str = "<URL>") -> None:
        self.remove_emojis = remove_emojis
        self.url_token = url_token

    def __call__(self, text: str) -> str:
        # Normalize unicode encoding
        text = unicodedata.normalize("NFC", text)
        # Remove '@name'
        text = self.MENTION_RE.sub(" ", text)
        # Replace '&amp;' with '&'
        text = text.replace("&amp;", "&")
        # Replace URLs anywhere in the text
        text = self.URL_RE.sub(self.url_token, text)
        if self.remove_emojis:
            text = emoji.replace_emoji(text, replace="")
        # Collapse whitespace last, after the removals above leave gaps
        return self.WHITESPACE_RE.sub(" ", text).strip()


class BertPreprocessor:
    """Cleans and tokenizes tweets for an Arabic BERT model."""

    def __init__(
        self,
        version: str = "mini",
        max_length: int = 128,
        batch_size: int = 16,
        seed: int = 2020,
        cleaner: TweetCleaner | None = None,
    ) -> None:
        if version not in MODEL_NAMES:
            raise ValueError(f"Unknown version {version!r}, expected one of {list(MODEL_NAMES)}")
        self.model_name = MODEL_NAMES[version]
        self.max_length = max_length
        self.batch_size = batch_size
        self.seed = seed
        self.cleaner = cleaner or TweetCleaner()
        # Load once; loading per sentence is the main cost of the notebook version.
        logger.info("Loading tokenizer %s (max_length=%d)", self.model_name, max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    @classmethod
    def from_settings(cls, settings: Settings) -> BertPreprocessor:
        return cls(
            version=settings.model.version,
            max_length=settings.model.max_length,
            batch_size=settings.training.batch_size,
            seed=settings.seed,
            cleaner=TweetCleaner(remove_emojis=settings.preprocessing.remove_emojis),
        )

    def encode(self, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (input_ids, attention_mask) tensors of shape (n_texts, max_length)."""
        encoded = self.tokenizer(
            [self.cleaner(t) for t in texts],
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"]

    def build_dataloader(
        self,
        texts: Sequence[str],
        labels: Sequence[int] | None = None,
        shuffle: bool = False,
    ) -> DataLoader:
        """Tokenize texts and wrap them in a DataLoader (shuffle=True for training).

        Batches are (input_ids, attention_mask, labels), or just the first two
        when `labels` is None (inference on unlabelled text).
        """
        input_ids, attention_mask = self.encode(texts)
        tensors = [input_ids, attention_mask]
        if labels is not None:
            tensors.append(torch.tensor(labels))
        dataset = TensorDataset(*tensors)
        truncated = int((attention_mask.sum(dim=1) == self.max_length).sum())
        logger.info(
            "Encoded %d texts, %d (%.1f%%) hit max_length=%d",
            len(texts), truncated, 100 * truncated / max(len(texts), 1), self.max_length,
        )
        # Seeded generator makes the shuffle order reproducible across runs.
        generator = torch.Generator().manual_seed(self.seed) if shuffle else None
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, generator=generator)
