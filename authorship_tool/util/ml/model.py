"""LGBMに利用するモデルデータ定義モジュール"""
import pickle
from pathlib import Path
from typing import Final, Optional

import pandas as pd
from attr import dataclass
from lightgbm import LGBMClassifier
from numpy import ndarray
from pandas import DataFrame
from shap import Explainer

from authorship_tool.util.path_util import PathUtil


@dataclass(frozen=True)
class LGBMSource:
    """LGBMのモデル作成用データクラス"""

    feature_data_frame: DataFrame
    nd_category: ndarray


@dataclass(frozen=True)
class SplittedDataset:
    train_data: DataFrame
    test_data: DataFrame
    train_ans: ndarray
    test_ans: ndarray


@dataclass(frozen=True)
class Prediction:
    pred_prob: ndarray
    pred_ans: ndarray


@dataclass(frozen=True)
class Score:
    f1_score: float
    accuracy_score: float
    auc_roc_score: Optional[float] = None


@dataclass(frozen=True)
class ShapData:
    explainer: Explainer
    shap_positive_vals: ndarray
    shap_positive_expected_val: float


@dataclass(frozen=True)
class LGBMResult:
    """LGBMのモデル作成結果データクラス"""

    model: LGBMClassifier
    splitted_dataset: SplittedDataset
    prediction: Prediction
    shap_data: ShapData
    score: Optional[Score] = None

    def dump(self, title: Optional[str] = None) -> None:
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
            pickle.dump(self.model, f)

        self.splitted_dataset.train_data.to_csv(
            DATASET_DIR.joinpath("train_data.csv"), index=False
        )
        self.splitted_dataset.test_data.to_csv(
            DATASET_DIR.joinpath("test_data.csv"), index=False
        )
        DataFrame(self.splitted_dataset.train_ans).to_csv(
            DATASET_DIR.joinpath("train_ans.csv"),
            index=False,
            header=False,
        )
        DataFrame(self.splitted_dataset.test_ans).to_csv(
            DATASET_DIR.joinpath("test_ans.csv"),
            index=False,
            header=False,
        )

        DataFrame(self.prediction.pred_prob).to_csv(
            DATASET_DIR.joinpath("ans_pred_prob.csv"),
            index=False,
            header=False,
        )
        DataFrame(self.prediction.pred_ans).to_csv(
            DATASET_DIR.joinpath("ans_pred.csv"),
            index=False,
            header=False,
        )

        DataFrame(self.shap_data.shap_positive_vals).to_csv(
            DATASET_DIR.joinpath("test_shap_val.csv"),
            index=False,
            header=False,
        )

    def pred_crosstab(self) -> DataFrame:
        """
        予測結果のクロス集計を行う

        Returns:
            DataFrame: クロス集計結果
        """
        return pd.crosstab(
            self.splitted_dataset.test_ans,
            self.prediction.pred_ans,
            rownames=["actual"],
            colnames=["predicted"],
        )


@dataclass(frozen=True)
class LGBMCvResult:
    models: list[LGBMClassifier]
    splitted_datasets: list[SplittedDataset]
    predictions: list[Prediction]
    shap_data_list: list[ShapData]
