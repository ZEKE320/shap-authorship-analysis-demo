"""パスユーティリティモジュール (Path utility module)"""
import os
from pathlib import Path
from typing import Final


class PathUtil:
    @staticmethod
    def init_project_root() -> Path:
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
    def init_path(cls, rel_path_str: str) -> Path:
        """
        相対パスから絶対パスを初期化する
        Initialize the absolute path from the relative path

        Args:
            rel_path (str): 相対パス (Relative path)

        Raises:
            ValueError: PROJECT_ROOT_PATHが初期化されていない場合 (If PROJECT_ROOT_PATH is not initialized)
            FileNotFoundError: ファイルが見つからない場合 (If the file is not found)

        Returns:
            Path: パス (Path)
        """

        if cls.PROJECT_ROOT_PATH is None:
            raise ValueError("Path: `PROJECT_ROOT_PATH` is not initialized.")

        if not (abs_path := cls.PROJECT_ROOT_PATH.joinpath(Path(rel_path_str))):
            raise FileNotFoundError(f"File: `{abs_path}` could not be found.")

        print(f"Path: {rel_path_str} = {abs_path}")
        return abs_path

    PROJECT_ROOT_PATH: Final[Path] = init_project_root()


PATHS: Final[dict[str, Path]] = {
    "text_data_dir": PathUtil.init_path(
        "dump/text_data",
    ),
    "dataset_dir": PathUtil.init_path(
        "dump/dataset",
    ),
    "past_participle_jj_dataset": PathUtil.init_path(
        "data/john_blake_2023/wordLists/adjectivesPastParticiple",
    ),
    "lgbm_model_dir": PathUtil.init_path(
        "dump/lgbm/model",
    ),
    "shap_figure_dir": PathUtil.init_path(
        "dump/shap/figure",
    ),
}
