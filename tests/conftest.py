import numpy as np
import pytest
from fastapi.testclient import TestClient

from mlops_practitioner_course.api import app
from mlops_practitioner_course.config import Settings


class FakePredictor:
    """Stands in for the real model so API tests need no weights or download."""

    LABEL_NAMES = ("negative", "positive")
    threshold = 0.5

    def predict_proba(self, texts):
        return np.full(len(texts), 0.9)


@pytest.fixture
def sample_reviews():
    return ["المنتج رائع جدا", "سيء ولا أنصح به", "عادي"]


@pytest.fixture
def client(monkeypatch):
    # Default Settings, not config.yaml, so tests don't depend on the local config.
    monkeypatch.setattr(app.state, "settings", Settings(), raising=False)
    monkeypatch.setattr(app.state, "predictor", FakePredictor(), raising=False)
    # Not used as a context manager, so the lifespan (real model loading) never runs.
    return TestClient(app)
