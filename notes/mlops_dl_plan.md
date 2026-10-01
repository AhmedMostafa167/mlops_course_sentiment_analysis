# MLOps DL Project: Minimal Plan (Arabic Sentiment, CPU)

**Loop for every change:** change → measure (same protocol) → record → decide.

---

## 1. Package
- **Do:** `src/` layout + `pyproject.toml`. One model class: `load()`, `predict(text)`, `predict_batch(texts)`.
- **In:** small pretrained Arabic encoder.
- **Out:** `pip install -e .` works; `predict()` returns label + confidence.
- **Tip:** `pip install torch --index-url https://download.pytorch.org/whl/cpu` (no CUDA, image is GBs smaller).

## 2. API
- **Do:** FastAPI: `POST /predict`, `POST /predict_batch`, `GET /health`. Pydantic rejects empty input (422). Load the model once at startup.
- **Out:** `curl` works on all three; bad input returns 422.

## 3. Version data + pipeline (DVC)
```bash
dvc init
dvc add data/raw/reviews.csv
dvc remote add -d storage <path-or-url>
dvc push
```
- **Do:** `dvc.yaml` stages: `prepare` → `train` → `evaluate`; hyperparameters in `params.yaml`; output `metrics.json`. Fix seeds.
- **Out:** `dvc repro` rebuilds everything; fresh clone + `dvc pull && dvc repro` gives the same metrics.

## 4. Track experiments (MLflow), one run
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```
```python
with mlflow.start_run():
    mlflow.log_params({...})                       # model, lr, batch size, epochs, seed, data hash
    mlflow.log_metric("val_loss", v, step=epoch)   # per epoch
    mlflow.log_metric("test_f1_macro", f1)         # final, held-out test set
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.pytorch.log_model(model, "model")       # log the tokenizer next to it
```
- **Rule:** same metric names in every run; the run that trains the model logs it.
- **Out:** repeat with one variable changed each time; compare runs in the UI.

## 5. Registry + auto-promote
- **Do:** register each run's model as `ArabicSentiment`, alias it `staging`. `scripts/promote.py` compares `staging` vs `production` on `test_f1_macro` (same test set); if better by a small margin, set alias `production`.
- **Out:** the API loads `models:/ArabicSentiment@production`, so promotion changes what is served with no code change.

## 6. Docker (multi-stage)
- **Do:** builder stage installs CPU torch + package into a venv; final stage copies only the venv and `models/production/` (exported by the promote script). Non-root user, `HEALTHCHECK` on `/health`, `docker-compose.yml`.
- **Out:** `docker compose up && curl localhost:8000/predict` works. **Record image size.**

## 7. Baseline load test (Locust)
- **Do:** realistic Arabic texts of varied length against `/predict`. Fix users / spawn rate / duration / CPU limit; never change them.
- **Out:** `reports/locust_fastapi.html` with p50/p95/p99, RPS, failures.

## 8. Optimize (one technique at a time)
For each: apply → check accuracy → measure size → load test → log to MLflow.
1. Distillation (smaller student).
2. ONNX export + ONNX Runtime.
3. INT8 quantization.

- **Out:** benchmark table (original / distilled / ONNX / INT8 × accuracy, size, p95). ONNX serving drops PyTorch from the image; rebuild and compare size.
- **Note:** TensorRT needs an NVIDIA GPU; use Colab/Kaggle or document ONNX CPU as the substitute.

## 9. Serve with BentoML
- **Do:** wrap the chosen model in a BentoML service with a batchable endpoint (max batch size + max wait). Save to the Bento store, build the container.
- **Out:** same API contract; `reports/locust_bento.html` under the identical protocol; compare to Step 8.

## 10. Tests
- **Do:** `pytest` for model class, API (valid, empty, oversized batch), data split, promote logic, and a quality check against `baseline.json`.
- **Out:** fast, network-free suite using a tiny fixture model.

## 11. CI/CD (GitHub Actions)
- **PR:** `ruff check` → `pytest --cov=src --cov-fail-under=75` → F1 gate (fail if below baseline) → `docker build`.
- **Merge to main:** build with the production model, push to Docker Hub, tag with commit SHA.
- **Out:** green pipeline; a deliberately bad PR fails; image appears on Docker Hub.

## 12. Batch scoring + canary
- **Do:** script scores ≥1,000 texts and writes label, confidence, text length, timestamp to `data/scoring/output/`. nginx routes 5% new / 95% old; document rollout stages and rollback rule.

## 13. Monitoring
- **Do:** compute PSI (text length, confidence) of each batch vs training data, run Evidently, save the report. Service exposes `/metrics` (latency histogram, PSI gauge) → Prometheus scrapes → Grafana panels (p95, PSI). Alert when PSI > 0.25. Put all in `docker-compose.yml`.
- **Out:** shifted data (very short texts) triggers the alert.

## 14. Deliver
- **Do:** README with 3-command setup, architecture diagram (data → train → registry → serve → monitor), benchmark table, per-phase changelog; `reports/` with all screenshots and reports. Test from a fresh clone. Write the ≥300-word peer review.
