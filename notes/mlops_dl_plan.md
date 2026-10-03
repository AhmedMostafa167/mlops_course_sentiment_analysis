# MLOps DL Project: Minimal Plan (Arabic Sentiment, CPU)

**Loop for every change:** change → measure (same protocol) → record → decide.

---

## Progress (as of 2026-10-03)

| Step | Status |
|---|---|
| 1. Package | ✅ Done (CPU torch index still to switch) |
| 2. API | ✅ Done |
| 3. DVC | 🟡 Partial: data/model tracked; no pipeline yet |
| 4. MLflow | ⬜ Not started |
| 5. Registry + promote | ⬜ Not started |
| 6. Docker | ⬜ Not started (`.Dockerfile` is empty) |
| 7. Locust baseline | ⬜ Not started |
| 8. Optimize | ⬜ Not started |
| 9. BentoML | ⬜ Not started |
| 10. Tests | ⬜ Not started (`tests/` is empty) |
| 11. CI/CD | ⬜ Not started |
| 12. Batch scoring + canary | ⬜ Not started |
| 13. Monitoring | ⬜ Not started |
| 14. Deliver | ⬜ Not started (`README.md` is empty) |

---

## 1. Package
- **Do:** `src/` layout + `pyproject.toml`. One model class: `load()`, `predict(text)`, `predict_batch(texts)`.
- **In:** small pretrained Arabic encoder.
- **Out:** `pip install -e .` works; `predict()` returns label + confidence.
- **Tip:** `pip install torch --index-url https://download.pytorch.org/whl/cpu` (no CUDA, image is GBs smaller).
- **Status: ✅ Done.**
  - Done: `src/` layout, `pyproject.toml`, CLI (`train` / `evaluate` / `predict`). `SentimentPredictor` loads from a checkpoint (`from_checkpoint`); single-text and batch prediction with label + probability are exposed through the API (`/predict`, `/predict/batch`). Encoder: `asafaya/bert-mini-arabic`. Config validated with pydantic from `config/config.yaml`.
  - Missing: torch still comes from the **cu130** index in `pyproject.toml`; switch to the CPU index before Docker.

## 2. API
- **Do:** FastAPI: `POST /predict`, `POST /predict_batch`, `GET /health`. Pydantic rejects empty input (422). Load the model once at startup.
- **Out:** `curl` works on all three; bad input returns 422.
- **Status: ✅ Done.**
  - Done: `GET /health`, `POST /predict`, `POST /predict/batch` (max 64 texts); model loaded once in the lifespan hook; empty/whitespace text → 422; 503 if the model isn't loaded.
  - Note: batch route is `/predict/batch` (plan says `/predict_batch`); response field is `probability` = P(positive).

## 3. Version data + pipeline (DVC)
```bash
dvc init
dvc add data/raw/reviews.csv
dvc remote add -d storage <path-or-url>
dvc push
```
- **Do:** `dvc.yaml` stages: `prepare` → `train` → `evaluate`; hyperparameters in `params.yaml`; output `metrics.json`. Fix seeds.
- **Out:** `dvc repro` rebuilds everything; fresh clone + `dvc pull && dvc repro` gives the same metrics.
- **Status: 🟡 Partial.**
  - Done: `dvc init`; remote `storage` configured; `data.dvc` (whole `data/` dir) and `models/bert-mini.dvc` tracked; seeds fixed (`seed: 2020`, seeded shuffle).
  - Missing: `dvc.yaml` (`prepare` → `train` → `evaluate`); `params.yaml` (or point DVC `params` at `config/config.yaml`); single `metrics.json` (training currently writes `metrics_val.json` / `metrics_test.json`); `dvc repro` reproducibility check.
  - Risk: remote is a local OneDrive path, so fresh clone / CI can't `dvc pull`. Move it to a shared remote (S3, GDrive, etc.).

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
- **Status: ⬜ Not started.** MLflow isn't a dependency yet. Current metrics use binary `f1` (need `test_f1_macro`); ROC plots are saved but no confusion-matrix image; `history.json` has per-epoch `val_loss` ready to log.

## 5. Registry + auto-promote
- **Do:** register each run's model as `ArabicSentiment`, alias it `staging`. `scripts/promote.py` compares `staging` vs `production` on `test_f1_macro` (same test set); if better by a small margin, set alias `production`.
- **Out:** the API loads `models:/ArabicSentiment@production`, so promotion changes what is served with no code change.
- **Status: ⬜ Not started.** API currently loads local `models/bert-mini/model.pt`; no `scripts/promote.py`.

## 6. Docker (multi-stage)
- **Do:** builder stage installs CPU torch + package into a venv; final stage copies only the venv and `models/production/` (exported by the promote script). Non-root user, `HEALTHCHECK` on `/health`, `docker-compose.yml`.
- **Out:** `docker compose up && curl localhost:8000/predict` works. **Record image size.**
- **Status: ⬜ Not started.** `.Dockerfile` exists but is empty (rename to `Dockerfile`); no `docker-compose.yml`.

## 7. Baseline load test (Locust)
- **Do:** realistic Arabic texts of varied length against `/predict`. Fix users / spawn rate / duration / CPU limit; never change them.
- **Out:** `reports/locust_fastapi.html` with p50/p95/p99, RPS, failures.
- **Status: ⬜ Not started.** No `reports/` folder.

## 8. Optimize (one technique at a time)
For each: apply → check accuracy → measure size → load test → log to MLflow.
1. Distillation (smaller student).
2. ONNX export + ONNX Runtime.
3. INT8 quantization.

- **Out:** benchmark table (original / distilled / ONNX / INT8 × accuracy, size, p95). ONNX serving drops PyTorch from the image; rebuild and compare size.
- **Note:** TensorRT needs an NVIDIA GPU; use Colab/Kaggle or document ONNX CPU as the substitute.
- **Status: ⬜ Not started.**

## 9. Serve with BentoML
- **Do:** wrap the chosen model in a BentoML service with a batchable endpoint (max batch size + max wait). Save to the Bento store, build the container.
- **Out:** same API contract; `reports/locust_bento.html` under the identical protocol; compare to Step 8.
- **Status: ⬜ Not started.**

## 10. Tests
- **Do:** `pytest` for model class, API (valid, empty, oversized batch), data split, promote logic, and a quality check against `baseline.json`.
- **Out:** fast, network-free suite using a tiny fixture model.
- **Status: ⬜ Not started.** `tests/` is empty; no pytest dependency, fixture model or `baseline.json`.

## 11. CI/CD (GitHub Actions)
- **PR:** `ruff check` → `pytest --cov=src --cov-fail-under=75` → F1 gate (fail if below baseline) → `docker build`.
- **Merge to main:** build with the production model, push to Docker Hub, tag with commit SHA.
- **Out:** green pipeline; a deliberately bad PR fails; image appears on Docker Hub.
- **Status: ⬜ Not started.** No `.github/workflows/`, no ruff config.

## 12. Batch scoring + canary
- **Do:** script scores ≥1,000 texts and writes label, confidence, text length, timestamp to `data/scoring/output/`. nginx routes 5% new / 95% old; document rollout stages and rollback rule.
- **Status: ⬜ Not started.**

## 13. Monitoring
- **Do:** compute PSI (text length, confidence) of each batch vs training data, run Evidently, save the report. Service exposes `/metrics` (latency histogram, PSI gauge) → Prometheus scrapes → Grafana panels (p95, PSI). Alert when PSI > 0.25. Put all in `docker-compose.yml`.
- **Out:** shifted data (very short texts) triggers the alert.
- **Status: ⬜ Not started.**

## 14. Deliver
- **Do:** README with 3-command setup, architecture diagram (data → train → registry → serve → monitor), benchmark table, per-phase changelog; `reports/` with all screenshots and reports. Test from a fresh clone. Write the ≥300-word peer review.
- **Status: ⬜ Not started.** `README.md` is empty.
