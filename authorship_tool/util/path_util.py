"""パスユーティリティモジュール (Path utility module)"""
import os
from pathlib import Path

from authorship_tool.config import PATHS


class PathUtil:
    """パスユーティリティクラス (Path utility class)"""

    PROJECT_ROOT: Path
    TEXT_DATA_DIR: Path
    DATASET_DIR: Path
    PAST_PARTICIPLE_ADJECTIVE_DATASET: Path
    LGBM_MODEL_DIR: Path
    SHAP_FIGURE_DIR: Path

    @classmethod
    def initialize_all_paths(cls) -> None:
        """
        環境変数からパスを初期化する
        Initialize all paths from environment variables
        """

        cls.PROJECT_ROOT = cls.__initialize_project_root()
        cls.TEXT_DATA_DIR = cls.__initialize_path("path_text_data_dir")
        cls.DATASET_DIR = cls.__initialize_path("path_dump_dataset_dir")
        cls.PAST_PARTICIPLE_ADJECTIVE_DATASET = cls.__initialize_path(
            "path_adjective_past_participle_dataset"
        )
        cls.LGBM_MODEL_DIR = cls.__initialize_path("path_dump_lgbm_model_dir")
        cls.SHAP_FIGURE_DIR = cls.__initialize_path("path_dump_shap_figure_dir")

    @staticmethod
    def __initialize_project_root() -> Path:
        """
        プロジェクトのルートディレクトリを初期化する
        Initialize the project root directory

        Raises:
            ValueError: pyproject.tomlが見つからない場合 (If pyproject.toml is not found)

        Returns:
            Path: プロジェクトのルートディレクトリ (The project root directory)
        """

        file_dir: Path = Path(os.path.dirname("__file__")).resolve()

        for directory in [*file_dir.parents, file_dir]:
            if directory.joinpath("pyproject.toml").exists():
                print(f"Project root: {directory}")
                return directory

        raise ValueError("File: 'pyproject.toml' could not be found.")

    @classmethod
    def __initialize_path(cls, env_key: str) -> Path:
        """
        環境変数からパスを初期化する
        Initialize the path from the environment variable

        Args:
            env_key (str): 環境変数のキー (Environment variable key)

        Raises:
            ValueError: PROJECT_ROOT_PATHが初期化されていない場合 (If PROJECT_ROOT_PATH is not initialized)
            ValueError: 環境変数が見つからない場合 (If the environment variable is not found)
            FileNotFoundError: ファイルが見つからない場合 (If the file is not found)

        Returns:
            Path: パス (Path)
        """

        if cls.PROJECT_ROOT is None:
            raise ValueError("Path: `PROJECT_ROOT_PATH` is not initialized.")

        if not (rel_path := PATHS[env_key]):
            raise ValueError(f"Env: `{env_key}` could not be found.")

        if not (abs_path := cls.PROJECT_ROOT.joinpath(Path(rel_path))):
            raise FileNotFoundError(f"File: `{abs_path}` could not be found.")

        print(f"Path: {env_key} = {abs_path}")
        return abs_path


PathUtil.initialize_all_paths()
