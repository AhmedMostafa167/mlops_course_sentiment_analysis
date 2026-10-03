* Idea: the video starts from a notebook and how it should be translated to a better project structure.
* Idea: the video splits the tools or thinking style into three layers: before[dvc, uv, git], during[mlflow, fastapi], and after version 1[docker]. 

* The code need to be in python script files. Data Exploration can still be kept in notebooks but data cleaning pipelines, and processing must be moved into standalone scripts.

* Using the UUV tool to initiate the coding environment:
    * `uv init` with no path for venv creation
    * `uv add package` for dependencies and starting the `.venv` creation process

* Commit the data to the dvc repo: 
    * `uv add dvc`
    * `dvc init`
    * `dvc add train.csv`
    * For trial and simplification purposes data can be pushed to local only or remote GDrive folder.
    * The processing pipeline can be added with the `dvc repro` 

* Now that the structure is fully prepared it's time to record it with `git`.
    * Using the commands `git init` # can be skipped now
    * and `git commit -m ""`
    * then `git remote add origin repo:link`
    * It's important to make sure the data path/ file is added to the `.gitignore` file

* The coding itself and how to structure a DL python scripts, then structure a FastAPI end point
    * A good idea to split the scripts as work loads i.e. train, data, predict, etc.  
    ```
        madewithml/
            ├── config.py
            ├── data.py
            ├── evaluate.py
            ├── models.py
            ├── predict.py
            ├── serve.py
            ├── train.py
            ├── tune.py
            └── utils.py
    ```
    * For using GPU make sure to install the torch-gpu index and add to UV. But to minimize the docker image it's best to use the CPU version.
    * If you plan to import models from hugging face remember to add the authentication token.  
* OOP best practices and code writing:
    Best practices behind these changes
    1. Single responsibility: loading, cleaning and tokenizing are separate classes, so each one can be tested and swapped on its own.
    1. Load expensive things once: tokenizers and models go in __init__, never inside a loop.
    1. No hidden globals: every function gets its inputs as arguments. The old data_to_tensor depended on notebook variables.
    1. Avoid magic numbers: seed, split size, max length and batch size are now constructor arguments.
    1. Fail early with clear errors: bad version or val_size values, or a missing file, raise a readable message.
    1. Type hints and docstrings on everything you call from outside the class.
    1. Reproducibility: a fixed seed plus a stratified split.
    1. Use logging instead of print, to log split sizes and class balance when loading.
    1. Put settings in config/: a YAML file or a pydantic Settings class for MAX_LEN, the model name, batch size and seed. 
    1. Add a [build-system] section using uv_build, which finds the package from the project name, and pyyaml as a dependency and a command entry point, so `uv run mlops-practitioner-course` runs your main file inside the project package.
    1. Save weights plus config, never a pickled object. Loading uses torch.load(weights_only=True), which refuses to run code hidden in a file.
    1. Train and predict with the same preprocessing. The checkpoint stores remove_emojis, max_length and the model name, and the predictor rebuilds its preprocessor from those. That way the API can't clean text differently from training.
    1. Pass dependencies in explicitly. Model, data, device and config are all arguments, so each part can be tested on its own.
    1. Only optimize trainable parameters, so freeze_bert: true actually trains just the head.
    1. Write results as files (metrics JSON, ROC PNG). They can be compared between runs and tracked with DVC.
    1. Logs vs. output: progress goes to logging, while predict results go to stdout so other tools can read them.
    1. Adding the training and path configuration to a .yaml file then write a config validation script with pydantic models to read these values. Rule of Thumb: Put everything related to your data in the config (paths, split ratios, target column). This includes your feature selection (missing value strategy, which columns to include or drop, etc.) and then also adds the model hyperparameters.  

* Internal models saved should be added to dvc: `dvc add models/bert-mini` 
* FastAPI run: within the code write the uvcorn run command, add the port number and the host link as env variables in a production environment. 

* Pitfalls found while moving the notebook into scripts:
    * Code copied from notebook cells depends on things defined elsewhere in the notebook: missing imports (`csv`, `torch`, `np`), the wrong class name (`BertModel` imported but `AutoModel` used), and globals like `device`, `optimizer`, `scheduler` or `train_dataloader`. Import and run each new module once before moving on.
    * Library APIs change between versions. `from transformers import AdamW` fails in transformers 5, so use `torch.optim.AdamW`. Read each import back against the installed version.
    * Look for expensive calls inside loops. The notebook reloaded the tokenizer for every tweet, and that line also hard-coded `mini`, so `version="base"` was silently ignored. Load the tokenizer once and tokenize the whole list in one call.
    * Test the text-cleaning regexes on a few hand-written examples. The URL regex `^https?://.*` only matched a URL at the start of the text and deleted everything after it, and the `@mention` regex missed a mention at the end of a tweet.
    * Once the cleaning changes, the model sees different input. Retrain, and don't compare the new scores with the old notebook numbers.
    * `max_length` counts tokens, not characters. 280 is Twitter's character limit; measured on this data, no tweet came close to 128 tokens. Log how many texts reach `max_length` and pick the value from that.
    * Paths like `"../data"` only work from one folder. Build them from the file's location (`Path(__file__).resolve().parents[2]`) or from the config.
    * The package folder must be a valid Python name: `mlops_practitioner_course`, not `mlops-practitioner-course`. Hyphens can't be imported. The project name in `pyproject.toml` can keep its hyphens.
    * Count accuracy over all samples, not as an average of per-batch accuracies. Otherwise the smaller last batch counts as much as a full one.
    * `plt.show()` does nothing in a script and blocks on a server. Save figures to files with `fig.savefig(...)`.
    * Train/val split: pass `stratify=y` so both splits keep the same class balance.

* Environment and tooling issues:
    * Windows + Jupyter: a notebook kernel keeps torch's DLLs locked, so `uv sync` fails with "Access is denied" and can leave torch half-installed. Closing the notebook tab is not enough. Run "Jupyter: Shut Down All Kernels" in VS Code (or end the `ipykernel_launcher` process), then run `uv sync --reinstall-package torch`.
    * DVC and `.gitignore`: don't add the models folder to the root `.gitignore`. Otherwise `dvc add models/bert-mini` fails with "bad DVC file name ... is git-ignored". `dvc add` writes its own `models/.gitignore` that hides the model while keeping the small `.dvc` pointer file visible to git.
    * Moving notebooks: use `git mv` so git records a rename. An untracked file needs `git add` first, then `git mv`.
    * You can create a branch at any time with `git switch -c <name>`. Uncommitted changes come along, and they belong to whichever branch you commit them on.
    * Commit after each working step. Large uncommitted refactors are easy to lose.
    * Add docstrings every where so that the next time you open it you can easily understand it.

* API struuctuuring:
    * Always add a health check
    * remember to implement the default path don't leave it empty
    * the app start should go to the scripts related to the api files not always the main file
    * to implement one function with the same logic but can accept a single input or a series
    
* Building the API (best practices):
    * Install a server along with FastAPI. FastAPI only defines the app, it can't serve it on its own. `uv add "fastapi[standard]"` brings `uvicorn` and the `fastapi` CLI. `uv add` already installs into `.venv`, so `uv sync` is only needed after a clone, a pull or a hand edit of `pyproject.toml`.
    * Give the API its own module next to the CLI, not inside it. Here `api.py` sits beside `main.py`, and both import `SentimentPredictor`. The API should never import the CLI file, because that pulls in training, `argparse` and plotting.
    * Shared helpers belong in shared modules. `run_dir()` lived in `main.py`, and the API needed it to find the checkpoint, so it moved to `config.py` as `Settings.run_dir`. If two entry points need the same thing, move it down a layer.
    * Load the model once, at startup. Use FastAPI's `lifespan` to load the checkpoint and keep the predictor on `app.state`. Loading inside a route reloads the model on every request.
    * A health check should prove the model is ready, not just that the process is running. `/health` returns `model_loaded: true`, or a 503 when there's no predictor.
    * Use the default path `/` for something useful: service name, model version, threshold and a link to `/docs`. When predictions look wrong, you can see right away which model is answering.
    * Single and batch prediction share one function. `/predict` wraps its text in a list and calls the same `predict_texts()` as `/predict/batch`, so there's only one place where the label and threshold logic lives.
    * Let pydantic reject bad input before it reaches the model: strip whitespace and require at least 1 character (an empty tweet gets a 422), and cap the batch size (64 here) so one request can't run the server out of memory.
    * Model inference blocks, so write those routes with plain `def`, not `async def`. FastAPI runs `def` routes in a thread pool. A blocking torch call inside `async def` freezes every other request.
    * Start with everything in one file. Four endpoints read fine in a single `api.py`. Split the schemas into `schemas.py`, or the routes into `routers/`, only once the file gets crowded.
    * Run command: `uv run uvicorn mlops_practitioner_course.api:app --reload`. The module path uses underscores, `:app` is the variable name inside the file, and `--reload` is for development only.
    * Test the API without starting a server, using `TestClient`, inside a `with TestClient(app) as client:` block. Without `with`, the `lifespan` never runs and every route finds no model.

* Errors and pitfalls while building the API:
    * Windows console: printing the Arabic API responses crashed with `UnicodeEncodeError: 'charmap' codec`. The API was fine; the terminal uses cp1252. Set `PYTHONIOENCODING=utf-8` before running test scripts that print non-English text.
    * Report model info from a single source. `GET /` takes `model_version` from `config.yaml`, but the threshold from the checkpoint. After retraining with a different config, the two can disagree. The checkpoint is the source of truth for what's actually being served.
    * Decide what happens when the model file is missing. With `lifespan`, a missing checkpoint crashes the server at startup, so the 503 in `/health` can never be reached. Both options are valid (fail fast, or start and report "not ready"), but pick one on purpose.
    * Type hints change between versions too. The editor flagged `-> AsyncIterator[None]` on an `@asynccontextmanager` function as deprecated. The current form is `-> AsyncGenerator[None]`.
    * `TestClient` warns that using it with `httpx` is deprecated, and asks for `httpx2` instead. Read warnings during tests instead of scrolling past them.
    * Windows line endings: editing a file with a script rewrote it with CRLF endings. Check `git diff --stat` after scripted edits. A one-line change that shows up as a whole-file change means the line endings flipped.
