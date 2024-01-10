"""LGBMに利用するモデルデータ定義モジュール"""
import pickle
from pathlib import Path
from typing import Final, TypedDict

import numpy as np
import pandas as pd
from attr import dataclass
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from pandas import DataFrame
from shap import Explainer

from authorship_tool.util.path_util import PathUtil


@dataclass(frozen=True)
class LGBMSource(TypedDict):
    """LGBMのモデル作成用データクラス"""

    feature_data_frame: DataFrame
    nd_category: NDArray[np.bool_]


@dataclass(frozen=True)
class SplittedDataset(TypedDict):
    train_data: DataFrame
    test_data: DataFrame
    train_ans: NDArray
    test_ans: NDArray


@dataclass(frozen=True)
class Prediction(TypedDict):
    pred_prob: NDArray
    pred_ans: NDArray


@dataclass(frozen=True)
class Score(TypedDict):
    f1_score: np.float64 | NDArray[np.float64]
    accuracy_score: np.float64
    auc_roc_score: np.float64 | None


@dataclass(frozen=True)
class ShapData(TypedDict):
    explainer: Explainer
    shap_positive_vals: NDArray
    shap_positive_expected_val: np.float64


@dataclass(frozen=True)
class LGBMResult(TypedDict):
    """LGBMのモデル作成結果データクラス"""

    model: LGBMClassifier
    splitted_dataset: SplittedDataset
    prediction: Prediction
    shap_data: ShapData
    score: Score | None


@dataclass(frozen=True)
class LGBMCvResult(TypedDict):
    models: list[LGBMClassifier]
    splitted_datasets: list[SplittedDataset]
    predictions: list[Prediction]
    shap_data_list: list[ShapData]


@dataclass(frozen=True)
class CvViewData(TypedDict):
    test_ans: DataFrame
    pred_ans: DataFrame
    pred_prob: DataFrame
    shap_positive_vals: DataFrame
    shap_expected_val: np.float64


def dump(result: LGBMResult, title: str | None = None) -> None:
    """
    作成したモデル、データを保存する

    Args:
        title (Optional[str]): フォルダ名に使用するタイトル
    """
    if title is None:
        title = "_output_"

    LGBM_MODEL_DIR: Final[Path] = PathUtil.LGBM_MODEL_DIR.joinpath(title)
    DATASET_DIR: Final[Path] = PathUtil.DATASET_DIR.joinpath(title)

    LGBM_MODEL_DIR.mkdir(exist_ok=True)
    LGBM_MODEL_DIR.mkdir(exist_ok=True)

    with open(LGBM_MODEL_DIR.joinpath("lgbm_model.pkl"), "wb") as f:
        pickle.dump(result["model"], f)

    result["splitted_dataset"]["train_data"].to_csv(
        DATASET_DIR.joinpath("train_data.csv"), index=False
    )
    result["splitted_dataset"]["test_data"].to_csv(
        DATASET_DIR.joinpath("test_data.csv"), index=False
    )
    DataFrame(result["splitted_dataset"]["train_ans"]).to_csv(
        DATASET_DIR.joinpath("train_ans.csv"),
        index=False,
        header=False,
    )
    DataFrame(result["splitted_dataset"]["test_ans"]).to_csv(
        DATASET_DIR.joinpath("test_ans.csv"),
        index=False,
        header=False,
    )

    DataFrame(result["prediction"]["pred_prob"]).to_csv(
        DATASET_DIR.joinpath("ans_pred_prob.csv"),
        index=False,
        header=False,
    )
    DataFrame(result["prediction"]["pred_ans"]).to_csv(
        DATASET_DIR.joinpath("ans_pred.csv"),
        index=False,
        header=False,
    )

    DataFrame(result["shap_data"]["shap_positive_vals"]).to_csv(
        DATASET_DIR.joinpath("test_shap_val.csv"),
        index=False,
        header=False,
    )


def pred_crosstab(result: LGBMResult) -> DataFrame:
    """
    予測結果のクロス集計を行う

    Returns:
        DataFrame: クロス集計結果
    """
    return pd.crosstab(
        result["splitted_dataset"]["test_ans"],
        result["prediction"]["pred_ans"],
        rownames=["actual"],
        colnames=["predicted"],
    )
