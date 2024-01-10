"""lightgbmトレーナー"""

from typing import Final, Optional

import shap
from lightgbm import LGBMClassifier
from numpy import ndarray
from pandas import DataFrame
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, train_test_split

from authorship_tool.util.ml.model import (
    LGBMCvResult,
    LGBMResult,
    LGBMSource,
    Prediction,
    Score,
    ShapData,
    SplittedDataset,
)

SHAP_VALUE_POSITIVE_IDX: Final[int] = 1


def train(dataset: SplittedDataset, use_score: bool = False) -> LGBMResult:
    model = LGBMClassifier()
    model.fit(X=dataset.train_data.values, y=dataset.train_ans)

    prediction: Prediction = predict(model, dataset.test_data)
    shap_result: ShapData = create_shap_data(model, dataset.test_data)

    score: Score | None = calc_score(prediction, dataset.test_ans, use_score)

    return LGBMResult(
        model,
        dataset,
        prediction,
        shap_result,
        score if use_score else None,
    )


def predict(model: LGBMClassifier, test_data: DataFrame) -> Prediction:
    ans_pred_prob = model.predict_proba(X=test_data)[:, 1]
    ans_pred = model.predict(X=test_data)

    return Prediction(ans_pred_prob, ans_pred)


def calc_score(
    prediction: Prediction, test_ans: ndarray, use_score: bool
) -> Optional[Score]:
    if len(test_ans) <= 1 or not use_score:
        return None

    f1_result = f1_score(y_true=test_ans, y_pred=prediction.ans_pred)
    accuracy_result = accuracy_score(y_true=test_ans, y_pred=prediction.ans_pred)
    auc_roc_result = roc_auc_score(y_true=test_ans, y_score=prediction.ans_pred_prob)

    return Score(f1_result, accuracy_result, auc_roc_result)


def create_shap_data(model: LGBMClassifier, test_data: DataFrame) -> ShapData:
    tree_explainer = shap.TreeExplainer(model)
    test_shap_val = tree_explainer.shap_values(test_data)[SHAP_VALUE_POSITIVE_IDX]
    test_shap_expected_val = list(tree_explainer.expected_value)[
        SHAP_VALUE_POSITIVE_IDX
    ]

    return ShapData(tree_explainer, test_shap_val, test_shap_expected_val)


def train_once(training_source: LGBMSource) -> LGBMResult:
    """LightGBMを使って、著者推定モデルを学習します。

    Args:
        df (pd.DataFrame): 特徴量のデータフレーム
        nd_correctness (np.ndarray): 正解ラベルの配列

    Returns:
        tuple:
        学習済みモデル、学習用データ、テスト用データ、学習用正解ラベル、テスト用正解ラベル、
        テスト用予測確率、テスト用予測ラベル、ROC AUCスコア
    """

    dataset = SplittedDataset(
        *train_test_split(
            training_source.feature_data_frame, training_source.nd_category
        )
    )

    return train(dataset, use_score=True)


def train_by_index(
    source: LGBMSource,
    train_indices: ndarray,
    test_index: ndarray,
    use_score: bool = False,
) -> LGBMResult:
    X_train, X_test = (
        source.feature_data_frame.iloc[train_indices],
        source.feature_data_frame.iloc[test_index],
    )
    y_train, y_test = (
        source.nd_category[train_indices],
        source.nd_category[test_index],
    )
    splitted_dataset = SplittedDataset(X_train, X_test, y_train, y_test)

    return train(splitted_dataset, use_score)


def loocv_train(source: LGBMSource) -> LGBMCvResult:
    loo = LeaveOneOut()

    results: list[LGBMResult] = [
        train_by_index(source, train_indices, test_index, use_score=False)
        for train_indices, test_index in loo.split(source.feature_data_frame)
    ]

    # cv_result: LGBMCvResult = LGBMCvResult(
    #     [result.model for result in results],
    #     [result.splitted_dataset for result in results],
    #     [result.prediction for result in results],
    #     [result.shap_data for result in results],
    #     None,
    # )

    # return cv_result
    return results
