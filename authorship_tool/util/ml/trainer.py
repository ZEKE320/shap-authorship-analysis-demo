"""lightgbmトレーナー"""

from typing import Final
from numpy import ndarray
from pandas import DataFrame
import shap
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, train_test_split

from authorship_tool.util.ml.model import (
    LGBMResult,
    LGBMSource,
    SplittedDataset,
    Prediction,
    Score,
    ShapResult,
)

SHAP_VALUE_POSITIVE_IDX: Final[int] = 1


def train(dataset: SplittedDataset) -> LGBMResult:
    model = LGBMClassifier()
    model.fit(X=dataset.train_data.values, y=dataset.train_ans)

    prediction = predict(model, dataset.test_data)
    score = calc_score(prediction, dataset.test_ans)
    shap_result = create_shap_data(model, dataset.test_data)

    return LGBMResult(
        model,
        dataset,
        prediction,
        score,
        shap_result,
    )


def predict(model: LGBMClassifier, test_data: DataFrame) -> Prediction:
    ans_pred_prob = model.predict_proba(X=test_data)[:, 1]
    ans_pred = model.predict(X=test_data)

    return Prediction(ans_pred_prob, ans_pred)


def calc_score(prediction: Prediction, test_ans: ndarray) -> Score:
    auc_roc_result = roc_auc_score(y_true=test_ans, y_score=prediction.ans_pred_prob)
    f1_result = f1_score(y_true=test_ans, y_pred=prediction.ans_pred)
    accuracy_result = accuracy_score(y_true=test_ans, y_pred=prediction.ans_pred)

    return Score(auc_roc_result, f1_result, accuracy_result)


def create_shap_data(model: LGBMClassifier, test_data: DataFrame) -> ShapResult:
    tree_explainer = shap.TreeExplainer(model)
    test_shap_val = tree_explainer.shap_values(test_data)[SHAP_VALUE_POSITIVE_IDX]
    test_shap_expected_val = list(tree_explainer.expected_value)[
        SHAP_VALUE_POSITIVE_IDX
    ]

    return ShapResult(tree_explainer, test_shap_val, test_shap_expected_val)


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
            training_source.feature_data_frame, training_source.nd_correctness
        )
    )

    return train(dataset)


def train_by_idx(source: LGBMSource, train_index, test_index) -> LGBMResult:
    X_train, X_test = (
        source.feature_data_frame.iloc[train_index],
        source.feature_data_frame.iloc[test_index],
    )
    y_train, y_test = (
        source.nd_correctness[train_index],
        source.nd_correctness[test_index],
    )
    splitted_dataset = SplittedDataset(X_train, X_test, y_train, y_test)

    return train(splitted_dataset)


def loocv_train(source: LGBMSource) -> list[LGBMResult]:
    loo = LeaveOneOut()

    results: list[LGBMResult] = [
        train_by_idx(source, train_index, test_index)
        for train_index, test_index in loo.split(source.feature_data_frame)
    ]

    return results
