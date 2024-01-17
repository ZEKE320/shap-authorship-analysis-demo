"""
分類器トレーナーモジュール
Classifier trainer module
"""


from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from authorship_tool.util.ml.analyzer import create_shap_data
from authorship_tool.util.ml.model import (
    LGBMSource,
    Prediction,
    Score,
    ShapData,
    SplittedDataset,
    TrainingResult,
)

SCORE_CALC_DEFAULT: Final[bool] = False


def train(
    splitted_dataset: SplittedDataset, use_score_calc: bool = SCORE_CALC_DEFAULT
) -> TrainingResult:
    """
    LightGBMを使ってモデルをトレーニングします。
    Train the model using LightGBM.

    Args:
        splitted_dataset (SplittedDataset): トレーニングデータとテストデータ
        use_score_calc (bool, optional): スコア計算の有無. Defaults to False.

    Returns:
        TrainingResult: トレーニング結果
    """

    model = LGBMClassifier()
    model.fit(X=splitted_dataset.train_data.values, y=splitted_dataset.train_ans)

    prediction: Prediction = predict(model, splitted_dataset.test_data)
    shap_data: ShapData = create_shap_data(model, splitted_dataset.test_data)

    score: Score | None = calc_score(
        prediction, splitted_dataset.test_ans, use_score_calc
    )

    return TrainingResult(
        model,
        splitted_dataset,
        prediction,
        shap_data,
        score,
    )


def predict(model: LGBMClassifier, test_data: pd.DataFrame) -> Prediction:
    """
    モデルによる予測を行います。
    Perform prediction by the model.

    Args:
        model (LGBMClassifier): モデル
        test_data (pd.DataFrame): テストデータ

    Returns:
        Prediction: 予測結果
    """

    return Prediction(
        pred_prob=np.array(model.predict_proba(X=test_data), dtype=np.float64)[:, 1],
        pred_ans=np.array(model.predict(X=test_data), dtype=bool),
    )


def train_once(training_source: LGBMSource) -> TrainingResult:
    """LightGBMを使って、著者推定モデルを学習します。

    Args:
        training_source (LGBMSource): LGBMのモデル作成用ソースデータ (LGBM model creation source data)

    Returns:
        TrainingResult: トレーニング結果 (Training result)
    """

    train_data, test_data, train_ans, test_ans = train_test_split(
        training_source.feature_data_frame,
        training_source.nd_category,
        shuffle=True,
        random_state=0,
        test_size=0.20,
    )

    dataset = SplittedDataset(
        train_data,
        test_data,
        train_ans,
        test_ans,
    )

    return train(dataset, use_score_calc=True)


def calc_score(
    prediction: Prediction, test_ans: npt.NDArray, use_score: bool
) -> Score | None:
    """
    スコアを計算します。

    Args:
        prediction (Prediction): 予測結果
        test_ans (NDArray): 正解ラベル
        use_score (bool): スコア計算の有無

    Returns:
        Score | None: スコアデータ
    """

    if len(test_ans) <= 1 or not use_score:
        return None

    f1_result = f1_score(y_true=test_ans, y_pred=prediction.pred_ans)
    accuracy_result = accuracy_score(y_true=test_ans, y_pred=prediction.pred_ans)
    auc_roc_result = roc_auc_score(y_true=test_ans, y_score=prediction.pred_prob)

    return Score(
        f1_result,
        accuracy_result,
        auc_roc_result,
    )
