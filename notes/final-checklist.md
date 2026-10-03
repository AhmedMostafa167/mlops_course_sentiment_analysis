# Final Project Checklist

Review before closing the project. Source: findings from [dataset-analysis.ipynb](../src/dataset-analysis.ipynb).

## Data & preprocessing decisions
- [ ] **Emoji handling decided and justified.** ASTC has emojis in 79% of tweets vs 0–7% in the other datasets, and its labels were likely derived from emojis (label leakage). Remove/mask emojis in training, or show a with/without ablation.
- [ ] **Arabic text normalization applied.** Alef/yaa/taa-marbuta variants, tashkeel, tatweel, repeated letters, URLs, mentions.
- [ ] **Vocabulary gap addressed.** Vocab IoU between datasets is only 0.05–0.13. Use subword/pretrained Arabic models (AraBERT, MARBERT, CAMeLBERT) or justify a word-level approach.
- [ ] **Max sequence length chosen from text-length distributions.** Reviews are much longer than tweets.

## Data quality checks
- [ ] Duplicates checked within ASTC train/test and across all datasets (retweets, repeated texts).
- [ ] Class balance verified with `value_counts()` for every dataset. Don't rely on the README table.
- [ ] SS2030 labels confirmed as 0/1 with no NaNs.
- [ ] Train/val split moved into the modeling pipeline and stratified (`stratify=y`).

## Evaluation
- [ ] Metrics reported **per dataset** (ASTC, SS2030, 100k reviews, ArSAS), not as one blended number.
- [ ] Generalization gap explained: domain shift (tweets vs reviews), dialect (Saudi-only SS2030, mixed ArSAS, mostly MSA reviews).

## Notebook fixes
- [ ] `contains_emoji` checks each character (or uses `emoji.emoji_count`) instead of splitting on spaces, which undercounts.
- [ ] `emoji` version pinned, or `UNICODE_EMOJI` replaced with `emoji.is_emoji` (removed in `emoji>=2.0`).
- [ ] Emoji printout shows a real percentage (it currently prints a fraction labeled "%").
- [ ] Word histograms exclude stopwords or show the most informative words per class.

## Serving (FastAPI)
Source: building [api.py](../src/mlops_practitioner_course/api.py).
- [ ] **API versioning decided.** Put routes under a `/v1` prefix, or justify skipping it. It was deferred while building the first version.
- [ ] **Model info comes from the checkpoint, not `config.yaml`.** `GET /` reports `model_version` from the config file, while the threshold comes from the checkpoint, so the two can disagree after a retrain. Read both from the checkpoint's saved settings.
- [ ] **Readiness behavior decided.** If the checkpoint is missing, `lifespan` currently crashes the server at startup, so `/health` never returns 503. Either keep fail-fast (and document it) or catch the load error and serve 503s.
- [ ] **Request limits justified.** The batch cap of 64 should be checked against latency and memory on the serving machine. Add a max character length per text, since the tokenizer truncates at `max_length` (128 tokens) anyway.
- [ ] **Config path overridable.** The API always loads `config/config.yaml`. Allow an environment variable for Docker and deployment.
- [ ] **API tests added** in `tests/` with `TestClient`: each endpoint, the 422 cases (empty text, batch over the cap) and the 503 case.
- [ ] `httpx2` installed for the test client (Starlette warns that `httpx` with `TestClient` is deprecated).
- [ ] Code split once it grows: schemas to `schemas.py` beyond a few models, `api/routers/` beyond about 6–8 routes.
- [ ] Serve command (`uv run uvicorn mlops_practitioner_course.api:app`) added to the README and Dockerfile.
