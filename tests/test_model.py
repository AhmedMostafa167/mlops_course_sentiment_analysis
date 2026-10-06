import pytest

from mlops_practitioner_course.config import Settings
from mlops_practitioner_course.modeling.evaluate import evaluate_predictions
from mlops_practitioner_course.preprocess import BertPreprocessor, TweetCleaner


def test_cleaner_replaces_urls_and_removes_emojis():
    cleaned = TweetCleaner()("@user مرحبا 😀 https://t.co/x")
    assert cleaned == "مرحبا <URL>"


def test_evaluate_predictions_metrics():
    report = evaluate_predictions([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])
    assert report.accuracy == 0.5
    assert report.confusion_matrix == [[1, 1], [1, 1]]


def test_settings_load_from_yaml():
    settings = Settings.from_yaml()
    assert 0 < settings.evaluation.threshold < 1


def test_preprocessor_rejects_unknown_version():
    # Raises before the tokenizer is downloaded, so no network is needed.
    with pytest.raises(ValueError, match="Unknown version"):
        BertPreprocessor(version="large")
