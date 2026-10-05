import pandas as pd
import pytest

from mlops_practitioner_course.config import Settings
from mlops_practitioner_course.data import ArabicTweetsLoader


def write_split(data_dir, split, n_per_class):
    # Same layout as the real files: no header, "label<TAB>tweet".
    for sentiment in ArabicTweetsLoader.LABELS:
        rows = [f"{sentiment[:3]}\t{sentiment} tweet {i}" for i in range(n_per_class)]
        path = data_dir / ArabicTweetsLoader.FILE_TEMPLATE.format(split=split, sentiment=sentiment)
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")


@pytest.fixture
def data_dir(tmp_path):
    write_split(tmp_path, "train", n_per_class=10)
    write_split(tmp_path, "test", n_per_class=3)
    return tmp_path


def test_load_train_val_is_stratified(data_dir):
    train, val = ArabicTweetsLoader(data_dir, val_size=0.2).load_train_val()
    assert (len(train), len(val)) == (16, 4)
    assert sorted(val.labels) == [0, 0, 1, 1]
    assert "label 0: 8 (50.0%)" in train.describe()


def test_load_test_maps_file_sentiment_to_label(data_dir):
    test = ArabicTweetsLoader(data_dir).load_test()
    assert test.labels == [0, 0, 0, 1, 1, 1]
    assert test.texts[0] == "negative tweet 0"


def test_from_settings(data_dir):
    settings = Settings.model_validate({"seed": 7, "data": {"data_dir": str(data_dir), "val_size": 0.3}})
    loader = ArabicTweetsLoader.from_settings(settings)
    assert (loader.data_dir, loader.val_size, loader.seed) == (data_dir, 0.3, 7)


@pytest.mark.parametrize("val_size", [0, 1, 1.5])
def test_rejects_invalid_val_size(val_size):
    with pytest.raises(ValueError, match="val_size"):
        ArabicTweetsLoader(val_size=val_size)


def test_missing_file_falls_back_to_s3(tmp_path, monkeypatch):
    # Stub out the network read and record which sources were requested.
    sources = []

    def fake_read_csv(source, **kwargs):
        sources.append(str(source))
        return pd.DataFrame({"label": ["neg"], "text": ["remote tweet"]})

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)
    test = ArabicTweetsLoader(tmp_path).load_test()

    assert all(s.startswith(ArabicTweetsLoader.S3_BASE_URL) for s in sources)
    assert test.texts == ["remote tweet", "remote tweet"]
    assert test.labels == [0, 1]
