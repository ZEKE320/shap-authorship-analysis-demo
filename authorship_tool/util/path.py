import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class PathUtil:
    PROJECT_ROOT: Path
    LGBM_MODEL_DIR: Path
    DATASET_DIR: Path
    SHAP_FIGURE_DIR: Path
    PAST_PARTICIPLE_ADJECTIVE_DATASET: Path

    @classmethod
    def initialize_all_paths(cls) -> None:
        cls.PROJECT_ROOT = cls.__initialize_project_root()
        cls.LGBM_MODEL_DIR = cls.__initialize_path("path_dump_lgbm_model_dir")
        cls.DATASET_DIR = cls.__initialize_path("path_dump_dataset_dir")
        cls.SHAP_FIGURE_DIR = cls.__initialize_path("path_dump_shap_figure_dir")
        cls.PAST_PARTICIPLE_ADJECTIVE_DATASET = cls.__initialize_path(
            "path_adjective_past_participle_dataset"
        )

    @classmethod
    def __initialize_project_root(cls) -> Path:
        file_dir: Path = Path(os.path.dirname("__file__")).resolve()

        for directory in [*file_dir.parents, file_dir]:
            if directory.joinpath("pyproject.toml").exists():
                print(f"Project root: {directory}")
                return directory

        raise ValueError("File: 'pyproject.toml' could not be found.")

    @classmethod
    def __initialize_path(cls, env_key: str) -> Path:
        if cls.PROJECT_ROOT is None:
            raise ValueError("Path: `PROJECT_ROOT_PATH` is not initialized.")

        if not (rel_path := os.getenv(env_key)):
            raise ValueError(f"Env: `{env_key}` could not be found.")

        if not (abs_path := cls.PROJECT_ROOT.joinpath(Path(rel_path))):
            raise FileNotFoundError(f"File: `{abs_path}` could not be found.")

        print(f"Path: {env_key} = {abs_path}")
        return abs_path


PathUtil.initialize_all_paths()
