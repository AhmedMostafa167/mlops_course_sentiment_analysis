# From Notebook to Production: Project Notes

These are my working notes from turning a Jupyter notebook into a real, maintainable ML project. They cover the tools, the best practices, and the mistakes I hit along the way, so you can skip them.

---

## Contents

1. [What this project is](#1-what-this-project-is)
2. [The big idea: three layers of tools](#2-the-big-idea-three-layers-of-tools)
3. [Setting up the project](#3-setting-up-the-project)
4. [Organizing the code](#4-organizing-the-code)
5. [Best practices for writing the code](#5-best-practices-for-writing-the-code)
6. [Pitfalls when moving notebook code into scripts](#6-pitfalls-when-moving-notebook-code-into-scripts)
7. [Building the API](#7-building-the-api)
8. [Pitfalls when building the API](#8-pitfalls-when-building-the-api)
9. [Environment and tooling issues](#9-environment-and-tooling-issues)
10. [Quick command reference](#10-quick-command-reference)
11. [Glossary](#11-glossary)

---

## 1. What this project is

- **Task:** sentiment analysis of Arabic tweets. Each tweet is classified as **positive** or **negative**.
- **Model:** a small pretrained Arabic BERT (`asafaya/bert-mini-arabic` from Hugging Face), fine-tuned with a small classification head on top.
- **Starting point:** a single Jupyter notebook that did everything: loading data, cleaning, training, evaluation.
- **Goal:** turn that notebook into a proper project: Python scripts, versioned data, a config file, and an HTTP API that serves predictions.

> The approach follows a course video that starts from a notebook and shows how to turn it into a better project structure.

---

## 2. The big idea: three layers of tools

The video groups the tools by **when** in the project you need them:

| Layer | When | Tools | What they do |
|---|---|---|---|
| **Before** | Setting up | `uv`, `dvc`, `git` | Manage the environment, version the data, version the code |
| **During** | Building and experimenting | `mlflow`, `fastapi` | Track experiments, serve the model as an API |
| **After v1** | Shipping | `docker` | Package everything so it runs the same anywhere |

**Main rule:** code belongs in Python script files.
- ✅ Notebooks are fine for **data exploration**.
- ❌ Data cleaning, processing pipelines and training must move into **standalone scripts**.

---

## 3. Setting up the project

### 3.1 Python environment with `uv`

`uv` is a fast tool for managing Python versions, virtual environments and dependencies.

```bash
uv init              # create the project in the current folder (no path needed)
uv add <package>     # add a dependency; the first call also creates .venv
```

- `uv add` installs straight into `.venv`. You only need `uv sync` after a fresh clone, a `git pull`, or a hand edit of `pyproject.toml`.
- **GPU vs CPU torch:** to train on a GPU, install torch from the GPU package index and register that index with uv. For the Docker image, though, the **CPU version** is much smaller (GBs smaller) and is the better choice.
- **Hugging Face:** if you download models from Hugging Face, remember to set up your authentication token.

### 3.2 Versioning data with DVC

Git isn't built for large data files. **DVC** (Data Version Control) stores the data somewhere else and keeps a small pointer file (`.dvc`) in git.

```bash
uv add dvc
dvc init
dvc add train.csv          # or a whole folder, e.g. dvc add data
```

- For practice, you can push the data to a **local folder** or a **Google Drive folder** as the remote storage.
- A processing pipeline can later be defined in DVC and rerun with `dvc repro`.
- **Trained models should be tracked with DVC too**, not git:

  ```bash
  dvc add models/bert-mini
  ```

### 3.3 Versioning code with git

Once the structure is in place, record it with git:

```bash
git init                            # skip if the folder is already a repo
git commit -m "your message"
git remote add origin <repo-link>
```

⚠️ Make sure the **data path is in `.gitignore`**, so the raw data never ends up in git.

---

## 4. Organizing the code

### 4.1 Split scripts by job

Give each job its own file: data, training, prediction and so on. A good reference layout (from the *Made With ML* course):

```
madewithml/
├── config.py      # settings
├── data.py        # loading and splitting data
├── evaluate.py    # metrics
├── models.py      # model definition
├── predict.py     # inference
├── serve.py       # API
├── train.py       # training loop
├── tune.py        # hyperparameter tuning
└── utils.py       # shared helpers
```

### 4.2 This project's layout

```
mlops-practitioner-course/
├── config/config.yaml           # all settings: paths, split, model, training
├── data/                        # tweets (tracked by DVC, ignored by git)
├── models/bert-mini/            # trained model + metrics (tracked by DVC)
├── notebooks/                   # exploration only
├── notes/                       # notes like this one
├── pyproject.toml               # dependencies + entry point
└── src/mlops_practitioner_course/
    ├── config.py                # reads and validates config.yaml
    ├── data.py                  # loads the tweet files, train/val split
    ├── preprocess.py            # text cleaning + tokenization
    ├── main.py                  # command line: train / evaluate / predict
    ├── api.py                   # FastAPI app
    └── modeling/
        ├── model.py             # BERT + classifier head
        ├── train.py             # training loop
        ├── evaluate.py          # metrics, ROC plots
        ├── predict.py           # inference
        └── checkpoint.py        # save / load the model
```

---

## 5. Best practices for writing the code

### 5.1 Design

1. **One job per class.** Loading, cleaning and tokenizing are separate classes, so each can be tested and swapped on its own.
2. **Load expensive things once.** Tokenizers and models are loaded in `__init__`, never inside a loop.
3. **No hidden globals.** Every function gets its inputs as arguments. (The old notebook function `data_to_tensor` silently depended on notebook variables.)
4. **Pass dependencies in explicitly.** Model, data, device and config are all arguments, so each part can be tested on its own.
5. **Only optimize trainable parameters.** That way `freeze_bert: true` really trains only the classifier head.

### 5.2 Configuration

6. **No magic numbers.** Seed, split size, max length and batch size are arguments, not hard-coded values.
7. **Put settings in `config/`.** Use a YAML file (or a pydantic Settings class) for values like max length, model name, batch size and seed.
8. **Validate the config.** Read the YAML through pydantic models, so a wrong value fails with a clear message.
   - **Rule of thumb:** everything about the data goes in the config: paths, split ratios, target column, and feature choices (missing-value strategy, columns to keep or drop). Model hyperparameters go there too.

### 5.3 Reliability and readability

9. **Fail early with clear errors.** A bad `version` or `val_size`, or a missing file, raises a readable message.
10. **Type hints and docstrings** on everything that's called from outside the class. Docstrings everywhere help you understand your own code when you come back to it later.
11. **Use `logging`, not `print`.** For example, log split sizes and class balance when loading data.
12. **Logs vs. output.** Progress messages go to logging; prediction results go to stdout, so other tools can read them.

### 5.4 Reproducibility and results

13. **Fixed seed plus a stratified split**, so every run gets the same data split.
14. **Write results to files** (metrics JSON, ROC curve PNG). They can be compared between runs and tracked with DVC.

### 5.5 Saving models safely

15. **Save weights plus config, never a pickled object.** Load with `torch.load(..., weights_only=True)`, which refuses to run code hidden inside the file.
16. **Train and predict with the same preprocessing.** The checkpoint stores `remove_emojis`, `max_length` and the model name, and the predictor rebuilds its preprocessor from them. That way the API can never clean text differently from training.

### 5.6 Packaging

17. **Make the project an installable package.** In `pyproject.toml`:
    - add a `[build-system]` section using `uv_build` (it finds the package from the project name),
    - add `pyyaml` as a dependency,
    - add a command entry point.

    Then `uv run mlops-practitioner-course` runs your main file from inside the package.

---

## 6. Pitfalls when moving notebook code into scripts

| Problem | What happened | Fix |
|---|---|---|
| **Hidden dependencies** | Code copied from cells relied on things defined elsewhere: missing imports (`csv`, `torch`, `np`), the wrong class (`BertModel` imported but `AutoModel` used), globals like `device`, `optimizer`, `scheduler`, `train_dataloader`. | Import and run each new module once before moving on. |
| **Library APIs change** | `from transformers import AdamW` fails in transformers 5. | Use `torch.optim.AdamW`. Check each import against the installed version. |
| **Expensive calls in loops** | The notebook reloaded the tokenizer **for every tweet**. That line also hard-coded `mini`, so `version="base"` was silently ignored. | Load the tokenizer once and tokenize the whole list in one call. |
| **Broken cleaning regexes** | The URL regex `^https?://.*` only matched a URL at the start of the text, then deleted everything after it. The `@mention` regex missed mentions at the end of a tweet. | Test regexes on a few hand-written examples. |
| **Comparing to old scores** | Changing the cleaning changes what the model sees. | Retrain, and don't compare new scores with the old notebook numbers. |
| **`max_length` confusion** | `max_length` counts **tokens**, not characters. 280 is Twitter's *character* limit; on this data no tweet came close to 128 tokens. | Log how many texts hit `max_length` and choose the value from that. |
| **Relative paths** | Paths like `"../data"` only work when run from one folder. | Build paths from the file's location (`Path(__file__).resolve().parents[2]`) or from the config. |
| **Package name** | `mlops-practitioner-course` can't be imported (hyphens aren't allowed). | Name the package folder `mlops_practitioner_course`. The project name in `pyproject.toml` can keep its hyphens. |
| **Wrong accuracy** | Averaging per-batch accuracies makes the smaller last batch count as much as a full one. | Count correct predictions over **all samples**. |
| **Plots in scripts** | `plt.show()` does nothing in a script and blocks on a server. | Save figures with `fig.savefig(...)`. |
| **Unbalanced splits** | A random split can change the class balance. | Pass `stratify=y` to the train/val split. |

---

## 7. Building the API

The API serves the trained model over HTTP using **FastAPI**.

### 7.1 Endpoints

| Method | Path | What it does |
|---|---|---|
| `GET` | `/` | Service info: name, model version, threshold, link to `/docs` |
| `GET` | `/health` | Returns `model_loaded: true`, or **503** if no model is loaded |
| `POST` | `/predict` | Predicts one text |
| `POST` | `/predict/batch` | Predicts a list of texts (up to 64) |

### 7.2 How to run it

```bash
uv add "fastapi[standard]"   # FastAPI + the uvicorn server + the fastapi CLI
uv run uvicorn mlops_practitioner_course.api:app --reload
```

- `mlops_practitioner_course.api` is the module path (underscores).
- `:app` is the name of the FastAPI variable inside the file.
- `--reload` restarts on code changes. **Development only.**
- The run command points at the **API module**, not at `main.py`.
- In production, set the **host and port with environment variables** instead of hard-coding them.

### 7.3 Best practices

1. **Install a server too.** FastAPI only defines the app; it can't serve it alone. `fastapi[standard]` brings `uvicorn`.
2. **Give the API its own module.** `api.py` sits next to `main.py` (the CLI), and both import `SentimentPredictor`. The API never imports the CLI, which would pull in training, `argparse` and plotting.
3. **Shared helpers go in shared modules.** `run_dir()` was in `main.py`, but the API needed it to find the checkpoint, so it moved to `config.py` as `Settings.run_dir`. If two entry points need the same thing, move it down a layer.
4. **Load the model once, at startup.** Use FastAPI's `lifespan` to load the checkpoint and keep the predictor on `app.state`. Loading inside a route reloads the model on every request.
5. **Always have a health check, and make it meaningful.** It should prove the **model is ready**, not just that the process is running.
6. **Don't leave `/` empty.** Show the service name, model version, threshold and a link to `/docs`. When predictions look wrong, you can see right away which model is answering.
7. **One function for single and batch prediction.** `/predict` wraps its text in a list and calls the same `predict_texts()` as `/predict/batch`, so the label and threshold logic lives in one place.
8. **Let pydantic reject bad input** before it reaches the model:
   - strip whitespace and require at least 1 character (an empty tweet gets a **422**),
   - cap the batch size (64) so one request can't run the server out of memory.
9. **Use plain `def` for inference routes, not `async def`.** Model inference blocks. FastAPI runs `def` routes in a thread pool; a blocking torch call inside `async def` freezes every other request.
10. **Start with one file.** Four endpoints read fine in a single `api.py`. Split schemas into `schemas.py`, or routes into `routers/`, only once it gets crowded.
11. **Test without starting a server.** Use FastAPI's `TestClient` inside a `with` block:

    ```python
    with TestClient(app) as client:
        client.get("/health")
    ```

    Without `with`, the `lifespan` never runs and every route finds no model.

---

## 8. Pitfalls when building the API

- **Arabic text crashes the Windows console.** Printing API responses failed with `UnicodeEncodeError: 'charmap' codec`. The API was fine; the terminal uses the cp1252 encoding. **Fix:** set `PYTHONIOENCODING=utf-8` before running scripts that print non-English text.
- **Report model info from one source.** `GET /` read `model_version` from `config.yaml` but the threshold from the checkpoint. After retraining with a different config, they can disagree. **The checkpoint is the source of truth** for what's actually being served.
- **Decide what happens when the model file is missing.** With `lifespan`, a missing checkpoint crashes the server at startup, so the 503 in `/health` can never be reached. Both options are valid (fail fast, or start and report "not ready"), but choose one on purpose.
- **Type hints change between versions.** `-> AsyncIterator[None]` on an `@asynccontextmanager` function is flagged as deprecated. The current form is `-> AsyncGenerator[None]`.
- **Read test warnings.** `TestClient` warns that using it with `httpx` is deprecated and asks for `httpx2`. Don't scroll past warnings.
- **Line endings on Windows.** Editing a file with a script rewrote it with CRLF line endings. Check `git diff --stat` after scripted edits: if a one-line change shows up as a whole-file change, the line endings flipped.

---

## 9. Environment and tooling issues

### Windows + Jupyter: "Access is denied" during `uv sync`
A running notebook kernel keeps torch's DLL files locked, so `uv sync` fails and can leave torch half-installed. **Closing the notebook tab isn't enough.**
1. In VS Code, run **"Jupyter: Shut Down All Kernels"** (or end the `ipykernel_launcher` process).
2. Then run:
   ```bash
   uv sync --reinstall-package torch
   ```

### DVC and `.gitignore`
Don't add the `models` folder to the root `.gitignore`. Otherwise `dvc add models/bert-mini` fails with *"bad DVC file name ... is git-ignored"*. `dvc add` writes its own `models/.gitignore`, which hides the model but keeps the small `.dvc` pointer file visible to git.

### Git habits
- **Moving notebooks:** use `git mv` so git records a rename. If the file isn't tracked yet, `git add` it first, then `git mv`.
- **Branches:** you can create one at any time with `git switch -c <name>`. Uncommitted changes come along and belong to whichever branch you commit them on.
- **Commit after each working step.** Large uncommitted refactors are easy to lose.

---

## 10. Quick command reference

```bash
# Environment
uv init
uv add <package>
uv sync                                   # after clone / pull / editing pyproject.toml
uv sync --reinstall-package torch         # fix a half-installed torch

# Data and model versioning
dvc init
dvc add data
dvc add models/bert-mini
dvc push / dvc pull
dvc repro                                 # rerun the pipeline

# Code versioning
git init
git commit -m "message"
git remote add origin <repo-link>
git switch -c <branch-name>
git mv <old-path> <new-path>

# Run the project
uv run mlops-practitioner-course train
uv run mlops-practitioner-course evaluate
uv run mlops-practitioner-course predict "تغريدة"
uv run uvicorn mlops_practitioner_course.api:app --reload
```

---

## 11. Glossary

| Term | Meaning |
|---|---|
| **uv** | Fast Python package and environment manager. |
| **DVC** | Data Version Control: versions large files (data, models) outside git, keeping small pointer files in git. |
| **MLflow** | Tracks experiments: parameters, metrics and models for each training run. |
| **FastAPI** | Python framework for building HTTP APIs. |
| **uvicorn** | The server that actually runs a FastAPI app. |
| **Docker** | Packages the app and its environment into an image that runs the same everywhere. |
| **Checkpoint** | A saved file with the model's weights (and here, its settings). |
| **Tokenizer** | Splits text into tokens (word pieces) the model understands. |
| **`max_length`** | Maximum number of tokens per text; longer texts are cut. |
| **Stratified split** | A train/val split that keeps the same class balance in both parts. |
| **pydantic** | Library that validates data (config values, API requests). |
| **lifespan** | FastAPI hook that runs code once at startup and shutdown. |
| **422 / 503** | HTTP codes: 422 = invalid input, 503 = service not ready. |
