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

* Now that the structure is fully prepared it's time to record it with `git`.
    * Using the commands `git init` # can be skipped now
    * and `git commit -m ""`
    * then `git remote add origin repo:link`
    * It's important to make sure the data path/ file is added to the `.gitignore` file

* The coding itself and how to structure a DL python scripts, then structure a FastAPI end point
    * FastAPI run: within the code write the uvcorn run command, add the port number and the host link as env variables in a production environment. 