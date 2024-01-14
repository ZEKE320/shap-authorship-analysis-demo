"""LGBMに利用するモデルデータ定義モジュール"""
import pickle
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from attr import dataclass
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from pandas import DataFrame
from shap import Explainer

from authorship_tool.util.path_util import PathUtil


@dataclass(frozen=True)
class LGBMSource:
    """LGBMのモデル作成用データクラス"""

    feature_data_frame: DataFrame
    nd_category: NDArray[np.bool_]


@dataclass(frozen=True)
class SplittedDataset:
    """学習データとテストデータ用データクラス"""

    train_data: DataFrame
    test_data: DataFrame
    train_ans: NDArray
    test_ans: NDArray


@dataclass(frozen=True)
class Prediction:
    """予測結果データクラス"""

    pred_prob: NDArray
    pred_ans: NDArray


@dataclass(frozen=True)
class Score:
    """評価スコアデータクラス"""

    f1_score: np.float64
    accuracy_score: np.float64
    auc_roc_score: np.float64


@dataclass(frozen=True)
class ShapData:
    """Shapデータクラス"""

    explainer: Explainer
    shap_vals: NDArray[np.float64]
    shap_expected_val: np.float64


@dataclass(frozen=True)
class TrainingResult:
    """LGBMのモデル学習結果データクラス"""

    model: LGBMClassifier
    splitted_dataset: SplittedDataset
    prediction: Prediction
    shap_data: ShapData
    score: Score | None


@dataclass(frozen=True)
class CrossValidationResult:
    """LGBMのクロスバリデーション結果データクラス"""

    models: list[LGBMClassifier]
    splitted_datasets: list[SplittedDataset]
    predictions: list[Prediction]
    shap_data_list: list[ShapData]


@dataclass(frozen=True)
class CrossValidationView:
    """クロスバリデーション結果の可視化用データクラス"""

    test_data: pd.DataFrame
    test_ans: NDArray[np.bool_]
    pred_ans: NDArray[np.bool_]
    pred_prob: NDArray[np.float64]
    shap_positive_vals: NDArray[np.float64]
    shap_expected_val: NDArray[np.float64]


def dump(result: TrainingResult, title: str | None = None) -> None:
    """
    作成したモデル、データを保存する

    Args:
        title (Optional[str]): フォルダ名に使用するタイトル
    """
    if title is None:
        title = "_output_"

    lgbm_model_dir: Final[Path] = PathUtil.LGBM_MODEL_DIR.joinpath(title)
    dataset_dir: Final[Path] = PathUtil.DATASET_DIR.joinpath(title)

    lgbm_model_dir.mkdir(exist_ok=True)
    lgbm_model_dir.mkdir(exist_ok=True)

    with open(lgbm_model_dir.joinpath("lgbm_model.pkl"), "wb") as f:
        pickle.dump(result.model, f)

    result.splitted_dataset.train_data.to_csv(
        dataset_dir.joinpath("train_data.csv"), index=False
    )
    result.splitted_dataset.test_data.to_csv(
        dataset_dir.joinpath("test_data.csv"), index=False
    )
    DataFrame(result.splitted_dataset.train_ans).to_csv(
        dataset_dir.joinpath("train_ans.csv"),
        index=False,
        header=False,
    )
    DataFrame(result.splitted_dataset.test_ans).to_csv(
        dataset_dir.joinpath("test_ans.csv"),
        index=False,
        header=False,
    )

    DataFrame(result.prediction.pred_prob).to_csv(
        dataset_dir.joinpath("ans_pred_prob.csv"),
        index=False,
        header=False,
    )
    DataFrame(result.prediction.pred_ans).to_csv(
        dataset_dir.joinpath("ans_pred.csv"),
        index=False,
        header=False,
    )

    DataFrame(result.shap_data.shap_vals).to_csv(
        dataset_dir.joinpath("test_shap_val.csv"),
        index=False,
        header=False,
    )


def pred_crosstab_normal(result: TrainingResult) -> DataFrame:
    """
    予測結果のクロス集計を行う

    Returns:
        DataFrame: クロス集計結果
    """
    return pd.crosstab(
        result.splitted_dataset.test_ans,
        result.prediction.pred_ans,
        rownames=["actual"],
        colnames=["predicted"],
    )


def pred_crosstab_loocv(cv_result: CrossValidationView) -> DataFrame:
    """
    予測結果のクロス集計を行う

    Returns:
        DataFrame: クロス集計結果
    """
    return pd.crosstab(
        cv_result.test_ans,
        cv_result.pred_ans,
        rownames=["actual"],
        colnames=["predicted"],
    )
