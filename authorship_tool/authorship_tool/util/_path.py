import os
from pathlib import Path


class PathUtil:
    PROJECT_ROOT_PATH: Path | None = None

    @classmethod
    def initialize_project_root_path(cls):
        file_dir: Path = Path(os.path.dirname("__file__")).resolve()

        if file_dir.joinpath("pyproject.toml").exists():
            PathUtil.PROJECT_ROOT_PATH = file_dir
            return

        for directory in file_dir.parents:
            if directory.joinpath("pyproject.toml").exists():
                PathUtil.PROJECT_ROOT_PATH = directory
                return

        print(
            "Path: $PROJECT_ROOT_PATH could not be found. Skip and continue processing."
        )


PathUtil.initialize_project_root_path()
