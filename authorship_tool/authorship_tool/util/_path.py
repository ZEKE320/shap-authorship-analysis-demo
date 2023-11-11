import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class PathUtil:
    PROJECT_ROOT_PATH: Path
    LGBM_MODEL_PATH: Path
    DATASET_PATH: Path
    SHAP_FIGURE_PATH: Path

    @classmethod
    def initialize_all_path(cls) -> None:
        cls.initialize_project_root_path()
        cls.initialize_lgbm_model_path()
        cls.initialize_dataset_path()
        cls.initialize_shap_figure_path()

    @classmethod
    def initialize_project_root_path(cls) -> None:
        file_dir: Path = Path(os.path.dirname("__file__")).resolve()

        for directory in [*file_dir.parents, file_dir]:
            if directory.joinpath("pyproject.toml").exists():
                PathUtil.PROJECT_ROOT_PATH = directory
                return

        raise ValueError("File: 'pyproject.toml' could not be found.")

    @classmethod
    def initialize_lgbm_model_path(cls) -> None:
        PathUtil.LGBM_MODEL_PATH = cls.__initialize_file_path("path_lgbm_model")

    @classmethod
    def initialize_dataset_path(cls) -> None:
        PathUtil.DATASET_PATH = cls.__initialize_file_path("path_dataset")

    @classmethod
    def initialize_shap_figure_path(cls) -> None:
        PathUtil.SHAP_FIGURE_PATH = cls.__initialize_file_path("path_shap_figure")

    @classmethod
    def __initialize_file_path(cls, env_name) -> Path:
        if cls.PROJECT_ROOT_PATH is None:
            raise ValueError("Path: `PROJECT_ROOT_PATH` is not initialized.")

        if not (rel_path := os.getenv(env_name)):
            raise ValueError(f"Env: `{env_name}` could not be found.")

        if not (abs_path := cls.PROJECT_ROOT_PATH.joinpath(rel_path)):
            raise FileNotFoundError(f"File: `{abs_path}` could not be found.")

        return abs_path


PathUtil.initialize_all_path()
