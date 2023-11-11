import pickle
from os import makedirs
from typing import Final

import pandas as pd
from lightgbm import LGBMClassifier
from numpy import ndarray
from pandas import DataFrame

from authorship_tool.util import PathUtil


class LGBMSourceModel:
    def __init__(
        self, desired_score: float, df: DataFrame, nd_correctness: ndarray
    ) -> None:
        self.__DESIRED_SCORE: Final[float] = desired_score
        self.__DF: Final[DataFrame] = df
        self.__ND_CORRECTNESS: Final[ndarray] = nd_correctness

    @property
    def desired_score(self) -> float:
        return self.__DESIRED_SCORE

    @property
    def df(self) -> DataFrame:
        return self.__DF

    @property
    def nd_correctness(self) -> ndarray:
        return self.__ND_CORRECTNESS


class LGBMResultModel:
    def __init__(
        self,
        model: LGBMClassifier,
        train_data: DataFrame,
        test_data: DataFrame,
        train_ans: ndarray,
        test_ans: ndarray,
        ans_pred_prob: ndarray,
        ans_pred: ndarray,
        auc_roc_score: float,
    ) -> None:
        self.__MODEL: Final[LGBMClassifier] = model
        self.__TRAIN_DATA: Final[DataFrame] = train_data
        self.__TEST_DATA: Final[DataFrame] = test_data
        self.__TRAIN_ANS: Final[ndarray] = train_ans
        self.__TEST_ANS: Final[ndarray] = test_ans
        self.__ANS_PRED_PROB: Final[ndarray] = ans_pred_prob
        self.__ANS_PRED: Final[ndarray] = ans_pred
        self.__AUC_ROC_SCORE: Final[float] = auc_roc_score

    @property
    def model(self) -> LGBMClassifier:
        return self.__MODEL

    @property
    def test_data(self) -> DataFrame:
        return self.__TEST_DATA

    @property
    def auc_roc_score(self) -> float:
        return self.__AUC_ROC_SCORE

    def dump(self) -> None:
        makedirs(PathUtil.LGBM_MODEL_PATH, exist_ok=True)
        makedirs(PathUtil.DATASET_PATH, exist_ok=True)

        with open(PathUtil.LGBM_MODEL_PATH.joinpath("lgbm_model.pkl"), "wb") as f:
            pickle.dump(self.__MODEL, f)

        self.__TRAIN_DATA.to_csv(
            PathUtil.DATASET_PATH.joinpath("train_data.csv"), index=False
        )
        self.__TEST_DATA.to_csv(
            PathUtil.DATASET_PATH.joinpath("test_data.csv"), index=False
        )
        DataFrame(self.__TRAIN_ANS).to_csv(
            PathUtil.DATASET_PATH.joinpath("train_ans.csv"), index=False, header=False
        )
        DataFrame(self.__TEST_ANS).to_csv(
            PathUtil.DATASET_PATH.joinpath("test_ans.csv"), index=False, header=False
        )
        DataFrame(self.__ANS_PRED_PROB).to_csv(
            PathUtil.DATASET_PATH.joinpath("ans_pred_prob.csv"),
            index=False,
            header=False,
        )
        DataFrame(self.__ANS_PRED).to_csv(
            PathUtil.DATASET_PATH.joinpath("ans_pred.csv"), index=False, header=False
        )

    def pred_crosstab(self) -> DataFrame:
        return pd.crosstab(self.__TEST_ANS, self.__ANS_PRED)
