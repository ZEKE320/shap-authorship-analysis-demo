"""パスユーティリティモジュール (Path utility module)"""
import abc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final


class PathUtil:
    """
    パスユーティリティ
    Path utility
    """

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


@dataclass(frozen=True, init=False)
class DatasetPaths:
    """
    データセットパス
    Dataset paths
    """

    text_data_dir: Path = PathUtil.init_path(
        "dump/text_data",
    )
    past_participle_jj_dataset: Path = PathUtil.init_path(
        "data/john_blake_2023/wordLists/adjectivesPastParticiple",
    )
    vijini_dataset: Path = PathUtil.init_path(
        "data/liyanage_vijini_2022",
    )
    uoa_thesis_dataset: Path = PathUtil.init_path(
        "data/uoa-thesis-2014-2017",
    )

    def __new__(cls) -> None:
        cls.text_data_dir.mkdir(parents=True, exist_ok=True)
        cls.past_participle_jj_dataset.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, init=False)
class CommonOutputPaths:
    """
    共通出力パス
    Common output paths
    """

    processed_text_dir: Path = PathUtil.init_path(
        "dump/processed_text",
    )
    dataset_dump_dir: Path = PathUtil.init_path(
        "dump/dataset",
    )
    lgbm_model_dir: Path = PathUtil.init_path(
        "dump/lgbm/model",
    )
    shap_figure_dir: Path = PathUtil.init_path(
        "dump/shap/figure",
    )


@dataclass(frozen=True, init=False)
class BasePaths(metaclass=abc.ABCMeta):
    """
    ベースパス
    Base paths
    """

    basename: str
    processed_text_dir: Path
    dataset_dump_dir: Path
    lgbm_model_dir: Path
    shap_figure_dir: Path

    @classmethod
    def init_paths(cls) -> None:
        cls.processed_text_dir = CommonOutputPaths.processed_text_dir.joinpath(
            cls.basename
        )
        cls.dataset_dump_dir = CommonOutputPaths.dataset_dump_dir.joinpath(cls.basename)
        cls.lgbm_model_dir = CommonOutputPaths.lgbm_model_dir.joinpath(cls.basename)
        cls.shap_figure_dir = CommonOutputPaths.shap_figure_dir.joinpath(cls.basename)

        cls.processed_text_dir.mkdir(parents=True, exist_ok=True)
        cls.dataset_dump_dir.mkdir(parents=True, exist_ok=True)
        cls.lgbm_model_dir.mkdir(parents=True, exist_ok=True)
        cls.shap_figure_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, init=False)
class VijiniDatasetPaths(BasePaths):
    """
    Vijini氏データセット関連パス
    Vijini dataset paths
    """

    basename: str = "liyanage_vijini_2022"


VijiniDatasetPaths.init_paths()


@dataclass(frozen=True, init=False)
class UoaThesisDatasetPaths(BasePaths):
    """
    UoA論文データセット関連パス
    UoA thesis dataset paths
    """

    basename: str = "uoa_thesis_2014_2017"


UoaThesisDatasetPaths.init_paths()


@dataclass(frozen=True, init=False)
class InauguralPaths(BasePaths):
    """
    就任演説関連パス
    Inaugural paths
    """

    basename: str = "inaugural"


InauguralPaths.init_paths()


@dataclass(frozen=True, init=False)
class InauguralLoocvPaths(BasePaths):
    """
    就任演説関連パス (LOOCV)
    Inaugural paths (LOOCV)
    """

    basename: str = "inaugural_loocv"


InauguralLoocvPaths.init_paths()


@dataclass(frozen=True, init=False)
class GutenbergPaths(BasePaths):
    """
    Gutenberg関連パス
    Gutenberg paths
    """

    basename: str = "gutenberg"


GutenbergPaths.init_paths()
