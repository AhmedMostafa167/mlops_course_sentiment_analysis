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

* FastAPI run: within the code write the uvcorn run command, add the port number and the host link as env variables in a production environment. 