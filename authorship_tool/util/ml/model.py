"""LGBMに利用するモデルデータ定義モジュール"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from pandas import DataFrame

from authorship_tool.util.path_util import BasePaths


@dataclass(frozen=True)
class LGBMSource:
    """LGBMのモデル作成用データクラス"""

    feature_data_frame: DataFrame
    nd_category: npt.NDArray[np.bool_]


@dataclass(frozen=True)
class SplittedDataset:
    """学習データとテストデータ用データクラス"""

    train_data: DataFrame
    test_data: DataFrame
    train_ans: npt.NDArray[np.bool_]
    test_ans: npt.NDArray[np.bool_]


@dataclass(frozen=True)
class Prediction:
    """予測結果データクラス"""

    pred_prob: npt.NDArray
    pred_ans: npt.NDArray


@dataclass(frozen=True)
class Score:
    """評価スコアデータクラス"""

    f1_score: float
    accuracy_score: float
    auc_roc_score: float


@dataclass(frozen=True)
class ShapData:
    """Shapデータクラス"""

    explainer: shap.Explainer
    explanation: shap.Explanation
    base_values: np.float64
    shap_values: npt.NDArray[np.float64]
    data: DataFrame


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

    models: tuple[LGBMClassifier, ...]
    splitted_datasets: tuple[SplittedDataset, ...]
    predictions: tuple[Prediction, ...]
    shap_data_tuple: tuple[ShapData, ...]
    scores: tuple[Score, ...]


@dataclass(frozen=True)
class CvGlobalExplanationData:
    """クロスバリデーション結果の可視化用データクラス"""

    test_data: pd.DataFrame
    test_ans: npt.NDArray[np.bool_]
    pred_ans: npt.NDArray[np.bool_]
    pred_prob: npt.NDArray[np.float64]
    shap_vals: npt.NDArray[np.float64]


def dump(result: TrainingResult, path: type[BasePaths]) -> None:
    """
    作成したモデル、データを保存する

    Args:
        result (TrainingResult): 学習結果
        path (BasePaths): パス
    """

    lgbm_model_dir: Final[Path] = path.lgbm_model_dir
    dataset_dir: Final[Path] = path.dataset_output_dir

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

    DataFrame(result.shap_data.shap_values).to_csv(
        dataset_dir.joinpath("test_shap_val.csv"),
        index=False,
        header=False,
    )


def pred_crosstab(
    test_ans: npt.NDArray[np.bool_], pred_ans: npt.NDArray[np.bool_]
) -> DataFrame:
    """
    予測結果のクロス集計を行う

    Returns:
        DataFrame: クロス集計結果
    """
    return pd.crosstab(
        test_ans,
        pred_ans,
        rownames=["actual"],
        colnames=["predicted"],
    )
