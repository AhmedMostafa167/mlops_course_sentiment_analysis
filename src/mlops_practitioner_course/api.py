"""HTTP API serving the sentiment model.

    uv run uvicorn mlops_practitioner_course.api:app --reload
"""

from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from typing import Annotated

import logging
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, StringConstraints

from mlops_practitioner_course.config import Settings
from mlops_practitioner_course.modeling.checkpoint import CHECKPOINT_FILENAME
from mlops_practitioner_course.modeling.predict import OnnxSentimentPredictor

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 64

# Whitespace-only text is rejected with a 422 before it reaches the tokenizer.
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class PredictRequest(BaseModel):
    text: Text


class BatchPredictRequest(BaseModel):
    texts: list[Text] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class Prediction(BaseModel):
    text: str
    label: str
    probability: float = Field(description="P(positive)")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:  # pragma: no cover
    # Load the model once at startup, not on every request.
    settings = Settings.from_yaml()
    app.state.settings = settings
    
    model_path = settings.run_dir / CHECKPOINT_FILENAME
    logger.info("Starting up FastAPI application...")
    logger.info(f"Loading model checkpoint from {model_path} on device '{settings.training.device}'...")
    
    app.state.predictor = OnnxSentimentPredictor.from_checkpoint(
        model_path, settings.training.device
    )
    
    logger.info("Model loaded successfully. Ready to serve requests!")
    yield


app = FastAPI(title="Arabic tweet sentiment API", lifespan=lifespan)

# --- Prometheus metrics ---
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
    inprogress_labels=True,
    excluded_handlers=["/health", "/metrics"],
).instrument(app).expose(app, endpoint="/metrics")


def get_predictor(request: Request) -> OnnxSentimentPredictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predictor


def predict_texts(predictor: OnnxSentimentPredictor, texts: Sequence[str]) -> list[Prediction]:
    probs = predictor.predict_proba(texts)
    return [
        Prediction(
            text=text,
            label=predictor.LABEL_NAMES[int(prob >= predictor.threshold)],
            probability=float(prob),
        )
        for text, prob in zip(texts, probs)
    ]


@app.get("/")
def root(request: Request) -> dict[str, str | float]:  # pragma: no cover
    settings: Settings = request.app.state.settings
    return {
        "service": app.title,
        "model_version": settings.model.version,
        "threshold": get_predictor(request).threshold,
        "docs": "/docs",
    }


@app.get("/health")
def health(request: Request) -> dict[str, str | bool]:
    get_predictor(request)  # 503 until the model is loaded
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict(body: PredictRequest, request: Request) -> Prediction:
    return predict_texts(get_predictor(request), [body.text])[0]


@app.post("/predict/batch")
def predict_batch(body: BatchPredictRequest, request: Request) -> list[Prediction]:
    return predict_texts(get_predictor(request), body.texts)
