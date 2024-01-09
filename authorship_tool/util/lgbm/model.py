"""LGBMに利用するモデルデータ定義モジュール"""
import pickle
from os import makedirs
from typing import Optional

import pandas as pd
from attr import dataclass
from lightgbm import LGBMClassifier
from numpy import ndarray
from pandas import DataFrame

from authorship_tool.util.path_util import PathUtil


@dataclass(frozen=True)
class LGBMSourceModel:
    """LGBMのモデル作成用データクラス"""

    df: DataFrame
    nd_correctness: ndarray


@dataclass(frozen=True)
class LGBMResultModel:
    """LGBMのモデル作成結果データクラス"""

    model: LGBMClassifier
    train_data: DataFrame
    test_data: DataFrame
    train_ans: ndarray
    test_ans: ndarray
    ans_pred_prob: ndarray
    ans_pred: ndarray
    auc_roc_score: float

    def dump(self, title: Optional[str] = None) -> None:
        """
        作成したモデル、データを保存する

        Args:
            title (Optional[str]): フォルダ名に使用するタイトル
        """
        if title is None:
            title = "_output_"

        makedirs(PathUtil.LGBM_MODEL_DIR.joinpath(title), exist_ok=True)
        makedirs(PathUtil.DATASET_DIR.joinpath(title), exist_ok=True)

        with open(PathUtil.LGBM_MODEL_DIR.joinpath(title, "lgbm_model.pkl"), "wb") as f:
            pickle.dump(self.model, f)

        self.train_data.to_csv(
            PathUtil.DATASET_DIR.joinpath(title, "train_data.csv"), index=False
        )
        self.test_data.to_csv(
            PathUtil.DATASET_DIR.joinpath(title, "test_data.csv"), index=False
        )
        DataFrame(self.train_ans).to_csv(
            PathUtil.DATASET_DIR.joinpath(title, "train_ans.csv"),
            index=False,
            header=False,
        )
        DataFrame(self.test_ans).to_csv(
            PathUtil.DATASET_DIR.joinpath(title, "test_ans.csv"),
            index=False,
            header=False,
        )
        DataFrame(self.ans_pred_prob).to_csv(
            PathUtil.DATASET_DIR.joinpath(title, "ans_pred_prob.csv"),
            index=False,
            header=False,
        )
        DataFrame(self.ans_pred).to_csv(
            PathUtil.DATASET_DIR.joinpath(title, "ans_pred.csv"),
            index=False,
            header=False,
        )

    def pred_crosstab(self) -> DataFrame:
        """
        予測結果のクロス集計を行う

        Returns:
            DataFrame: クロス集計結果
        """
        return pd.crosstab(self.test_ans, self.ans_pred)
