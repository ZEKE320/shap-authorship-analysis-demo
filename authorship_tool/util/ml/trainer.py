"""lightgbmトレーナー"""

from typing import Final

import numpy as np
import shap
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
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
    model.fit(X=dataset["train_data"].values, y=dataset["train_ans"])

    prediction: Prediction = predict(model, dataset["test_data"])
    shap_result: ShapData = create_shap_data(model, dataset["test_data"])

    score: Score | None = calc_score(prediction, dataset["test_ans"], use_score)

    return {
        "model": model,
        "splitted_dataset": dataset,
        "prediction": prediction,
        "shap_data": shap_result,
        "score": score if use_score else None,
    }


def predict(model: LGBMClassifier, test_data: DataFrame) -> Prediction:
    pred_prob: NDArray[np.float64] = np.array(
        model.predict_proba(X=test_data), dtype=np.float64
    )[:, 1]
    pred_ans: NDArray[np.bool_] = np.array(model.predict(X=test_data), dtype=np.bool_)

    return {"pred_prob": pred_prob, "pred_ans": pred_ans}


def calc_score(
    prediction: Prediction, test_ans: NDArray, use_score: bool
) -> Score | None:
    if len(test_ans) <= 1 or not use_score:
        return None

    f1_result = f1_score(y_true=test_ans, y_pred=prediction["pred_ans"])
    accuracy_result = accuracy_score(y_true=test_ans, y_pred=prediction["pred_ans"])
    auc_roc_result = roc_auc_score(y_true=test_ans, y_score=prediction["pred_prob"])

    return {
        "f1_score": f1_result,
        "accuracy_score": accuracy_result,
        "auc_roc_score": auc_roc_result,
    }


def create_shap_data(model: LGBMClassifier, test_data: DataFrame) -> ShapData:
    tree_explainer = shap.TreeExplainer(model)
    test_shap_val = tree_explainer.shap_values(test_data)[SHAP_VALUE_POSITIVE_IDX]
    test_shap_expected_val = np.array(tree_explainer.expected_value)[
        SHAP_VALUE_POSITIVE_IDX
    ]

    return {
        "explainer": tree_explainer,
        "shap_positive_vals": test_shap_val,
        "shap_positive_expected_val": test_shap_expected_val,
    }


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

    train_data, test_data, train_ans, test_ans = train_test_split(
        training_source["feature_data_frame"], training_source["nd_category"]
    )

    dataset: SplittedDataset = {
        "train_data": train_data,
        "test_data": test_data,
        "train_ans": train_ans,
        "test_ans": test_ans,
    }

    return train(dataset, use_score=True)


def train_by_index(
    source: LGBMSource,
    train_indices: NDArray,
    test_index: NDArray,
    use_score: bool = False,
) -> LGBMResult:
    X_train, X_test = (
        source["feature_data_frame"].iloc[train_indices],
        source["feature_data_frame"].iloc[test_index],
    )
    y_train, y_test = (
        source["nd_category"][train_indices],
        source["nd_category"][test_index],
    )
    splitted_dataset: SplittedDataset = {
        "train_data": X_train,
        "test_data": X_test,
        "train_ans": y_train,
        "test_ans": y_test,
    }

    return train(splitted_dataset, use_score)


def train_loocv(source: LGBMSource) -> LGBMCvResult:
    loo = LeaveOneOut()

    results: list[LGBMResult] = [
        train_by_index(source, train_indices, test_index, use_score=False)
        for train_indices, test_index in loo.split(source["feature_data_frame"])
    ]

    cv_result: LGBMCvResult = convert_results_to_cv_result(results)

    return cv_result


def convert_results_to_cv_result(results: list[LGBMResult]) -> LGBMCvResult:
    models, datasets, predictions, shap_data_list = map(
        list,
        zip(
            *(
                (
                    result["model"],
                    result["splitted_dataset"],
                    result["prediction"],
                    result["shap_data"],
                )
                for result in results
            )
        ),
    )

    return {
        "models": models,
        "splitted_datasets": datasets,
        "predictions": predictions,
        "shap_data_list": shap_data_list,
    }
