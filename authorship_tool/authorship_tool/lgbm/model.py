import os
import pickle
from os import makedirs, path
from typing import Final

import pandas as pd
from dotenv import load_dotenv
from lightgbm import LGBMClassifier
from numpy import ndarray
from pandas import DataFrame

load_dotenv()
MODEL_PATH: Final[str] = path.join(
    path.dirname(path.abspath(".env")),
    os.getenv("path_lgbm_model") or "",
)
DATASET_PATH: Final[str] = path.join(
    path.dirname(path.abspath(".env")),
    os.getenv("path_dataset") or "",
)


class LGBMSourceModel:
    __DESIRED_SCORE: Final[float]
    __DF: Final[DataFrame]
    __ND_CORRECTNESS: Final[ndarray]

    def __init__(
        self, desired_score: float, df: DataFrame, nd_correctness: ndarray
    ) -> None:
        self.__DESIRED_SCORE = desired_score
        self.__DF = df
        self.__ND_CORRECTNESS = nd_correctness

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
    __MODEL: Final[LGBMClassifier]
    __TRAIN_DATA: Final[DataFrame]
    __TEST_DATA: Final[DataFrame]
    __TRAIN_ANS: Final[ndarray]
    __TEST_ANS: Final[ndarray]
    __ANS_PRED_PROB: Final[ndarray]
    __ANS_PRED: Final[ndarray]
    __AUC_ROC_SCORE: Final[float]

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
        self.__MODEL = model
        self.__TRAIN_DATA = train_data
        self.__TEST_DATA = test_data
        self.__TRAIN_ANS = train_ans
        self.__TEST_ANS = test_ans
        self.__ANS_PRED_PROB = ans_pred_prob
        self.__ANS_PRED = ans_pred
        self.__AUC_ROC_SCORE = auc_roc_score

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
        makedirs(MODEL_PATH, exist_ok=True)
        makedirs(DATASET_PATH, exist_ok=True)

        with open(path.join(MODEL_PATH, "lgbm_model.pkl"), "wb") as f:
            pickle.dump(self.__MODEL, f)

        self.__TRAIN_DATA.to_csv(path.join(DATASET_PATH, "train_data.csv"), index=False)
        self.__TEST_DATA.to_csv(path.join(DATASET_PATH, "test_data.csv"), index=False)
        DataFrame(self.__TRAIN_ANS).to_csv(
            path.join(DATASET_PATH, "train_ans.csv"), index=False, header=False
        )
        DataFrame(self.__TEST_ANS).to_csv(
            path.join(DATASET_PATH, "test_ans.csv"), index=False, header=False
        )
        DataFrame(self.__ANS_PRED_PROB).to_csv(
            path.join(DATASET_PATH, "and_pred_prob.csv"), index=False, header=False
        )
        DataFrame(self.__ANS_PRED).to_csv(
            path.join(DATASET_PATH, "ans_pred.csv"), index=False, header=False
        )

    def pred_crosstab(self) -> DataFrame:
        return pd.crosstab(self.__TEST_ANS, self.__ANS_PRED)
