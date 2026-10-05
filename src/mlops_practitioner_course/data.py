"""Loading the Arabic sentiment tweets dataset (positive / negative TSV files)."""

from __future__ import annotations

import csv
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from mlops_practitioner_course.config import PROJECT_ROOT, Settings

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

logger = logging.getLogger(__name__)

Split = Literal["train", "test"]


@dataclass(frozen=True)
class DataSplit:
    """Parallel lists of tweets and their integer labels (0 = negative, 1 = positive)."""

    texts: list[str]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.texts)

    def describe(self) -> str:
        """One-line summary: size and per-class counts/shares, for logging."""
        counts = Counter(self.labels)
        balance = ", ".join(
            f"label {label}: {n} ({n / len(self):.1%})" for label, n in sorted(counts.items())
        )
        return f"{len(self)} samples [{balance}]"


class ArabicTweetsLoader:
    """Reads the train/test TSV files and returns labelled splits."""

    FILE_TEMPLATE = "{split}_Arabic_tweets_{sentiment}_20190413.tsv"
    LABELS = {"negative": 0, "positive": 1}
    S3_BASE_URL = "https://sentiment-analysis-amr-data.s3.eu-north-1.amazonaws.com"

    def __init__(
        self,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        val_size: float = 0.1,
        seed: int = 2020,
    ) -> None:
        if not 0 < val_size < 1:
            raise ValueError(f"val_size must be in (0, 1), got {val_size}")
        self.data_dir = Path(data_dir)
        self.val_size = val_size
        self.seed = seed

    @classmethod
    def from_settings(cls, settings: Settings) -> ArabicTweetsLoader:
        return cls(
            data_dir=settings.data.data_dir,
            val_size=settings.data.val_size,
            seed=settings.seed,
        )

    def _read_file(self, split: Split, sentiment: str) -> pd.DataFrame:
        filename = self.FILE_TEMPLATE.format(split=split, sentiment=sentiment)
        local_path = self.data_dir / filename
        if local_path.exists():
            logger.info("Loading %s locally from %s", filename, local_path)
            df = pd.read_csv(
                local_path,
                sep="\t",
                header=None,
                names=["label", "text"],
                quoting=csv.QUOTE_NONE,
            )
        else:
            s3_url = f"{self.S3_BASE_URL}/{filename}"
            logger.info("Local file %s not found. Streaming in-memory from %s", local_path, s3_url)
            df = pd.read_csv(
                s3_url,
                sep="\t",
                header=None,
                names=["label", "text"],
                quoting=csv.QUOTE_NONE,
            )

        df["label"] = self.LABELS[sentiment]
        if "tweet" not in df.columns:
            df["tweet"] = df["text"]
        return df

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load raw train and test DataFrames, either from local disk or streamed from S3.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
        """
        train_df = self.load_split("train")
        test_df = self.load_split("test")
        return train_df, test_df

    def load_split(self, split: Split) -> pd.DataFrame:
        """Return all tweets of one split as a DataFrame with `tweet` and `label` columns."""
        frames = [self._read_file(split, sentiment) for sentiment in self.LABELS]
        return pd.concat(frames, ignore_index=True)

    def load_train_val(self) -> tuple[DataSplit, DataSplit]:
        df = self.load_split("train")
        X_train, X_val, y_train, y_val = train_test_split(
            df["tweet"].tolist(),
            df["label"].tolist(),
            test_size=self.val_size,
            random_state=self.seed,
            stratify=df["label"],
        )
        train, val = DataSplit(X_train, y_train), DataSplit(X_val, y_val)
        logger.info("Train split: %s", train.describe())
        logger.info("Val split:   %s", val.describe())
        return train, val

    def load_test(self) -> DataSplit:
        df = self.load_split("test")
        test = DataSplit(df["tweet"].tolist(), df["label"].tolist())
        logger.info("Test split:  %s", test.describe())
        return test
