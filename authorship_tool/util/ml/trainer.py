"""lightgbmトレーナー"""

from dataclasses import astuple
from typing import Final

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, train_test_split

from authorship_tool.util.ml.model import (
    CrossValidationResult,
    CrossValidationView,
    LGBMSource,
    Prediction,
    Score,
    ShapData,
    SplittedDataset,
    TrainingResult,
)

SHAP_VALUE_POSITIVE_IDX: Final[int] = 1


def train(
    splitted_dataset: SplittedDataset, use_score_calc: bool = False
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
        pred_ans=np.array(model.predict(X=test_data), dtype=np.bool_),
    )


def create_shap_data(model: LGBMClassifier, test_data: pd.DataFrame) -> ShapData:
    """
    SHAPデータを作成します。

    Args:
        model (LGBMClassifier): モデル
        test_data (pd.DataFrame): テストデータ

    Returns:
        ShapData: SHAPデータ
    """

    tree_explainer = shap.TreeExplainer(model)
    shap_vals = tree_explainer.shap_values(test_data)[SHAP_VALUE_POSITIVE_IDX]
    shap_expected_val = np.array(tree_explainer.expected_value)[SHAP_VALUE_POSITIVE_IDX]

    return ShapData(
        explainer=tree_explainer,
        shap_vals=shap_vals,
        shap_expected_val=shap_expected_val,
    )


def train_once(training_source: LGBMSource) -> TrainingResult:
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
        training_source.feature_data_frame, training_source.nd_category
    )

    dataset = SplittedDataset(
        train_data,
        test_data,
        train_ans,
        test_ans,
    )

    return train(dataset, use_score_calc=True)


def train_by_index(
    source: LGBMSource,
    train_indices: NDArray,
    test_index: NDArray,
    use_score_calc: bool = False,
) -> TrainingResult:
    """
    指定したインデックスのデータを用いて学習を行います。

    Args:
        source (LGBMSource): LGBMのモデル作成用ソースデータ
        train_indices (NDArray): トレーニングデータのインデックス一覧
        test_index (NDArray): テストデータのインデックス一覧
        use_score_calc (bool, optional): スコア計算の有無. Defaults to False.

    Returns:
        LGBMResult: _description_
    """
    splitted_dataset = SplittedDataset(
        train_data=source.feature_data_frame.iloc[train_indices],
        test_data=source.feature_data_frame.iloc[test_index],
        train_ans=source.nd_category[train_indices],
        test_ans=source.nd_category[test_index],
    )

    return train(splitted_dataset, use_score_calc)


def train_loocv(source: LGBMSource) -> CrossValidationView:
    """
    LOOCVで学習を行います。

    Args:
        source (LGBMSource): LGBMのモデル作成用ソースデータ

    Returns:
        CvViewData: Cvの結果を表示するためのデータ
    """
    loo = LeaveOneOut()

    results: list[TrainingResult] = [
        train_by_index(source, train_indices, test_index, use_score_calc=False)
        for train_indices, test_index in loo.split(source.feature_data_frame)
    ]

    cv_result: CrossValidationResult = convert_results_to_cv_result(results)
    cv_view_data: CrossValidationView = convert_cv_result_for_view(cv_result)

    return cv_view_data


def convert_results_to_cv_result(
    results: list[TrainingResult],
) -> CrossValidationResult:
    """
    LGBMResultのリストからLGBMCvResultを生成します。

    Args:
        results (list[LGBMResult]): LGBMResultのリスト

    Returns:
        LGBMCvResult: LGBMCvResultインスタンス
    """

    models_zip, splitted_datasets_zip, predictions_zip, shap_data_zip = zip(
        *map(astuple, results)
    )

    return CrossValidationResult(
        list(models_zip),
        list(splitted_datasets_zip),
        list(predictions_zip),
        list(shap_data_zip),
    )


def convert_cv_result_for_view(cv_result: CrossValidationResult) -> CrossValidationView:
    """
    LGBMCvResultをCvViewDataに変換します。

    Args:
        cv_result (LGBMCvResult): LGBMCvResultインスタンス

    Returns:
        CvViewData: Cvの結果を表示するためのデータ
    """

    (_, splitted_datasets_zip, predictions_zip, shap_data_collection_zip) = astuple(
        cv_result
    )

    (_, test_data_zip, _, test_ans_zip) = zip(
        *map(astuple, splitted_datasets_zip),
    )
    (pred_prob_zip, pred_ans_zip) = zip(
        *map(astuple, predictions_zip),
    )
    (_, shap_vals_zip, shap_expected_val_zip) = zip(
        *map(astuple, shap_data_collection_zip),
    )

    test_data: pd.DataFrame = pd.concat(test_data_zip)
    test_ans: NDArray[np.bool_] = np.concatenate(test_ans_zip)

    pred_ans: NDArray[np.bool_] = np.concatenate(pred_ans_zip)
    pred_prob: NDArray[np.float64] = np.concatenate(pred_prob_zip)
    shap_vals: NDArray[np.float64] = np.concatenate(shap_vals_zip)
    shap_expected_val: NDArray[np.float64] = np.concatenate(
        np.full(
            shape=shap_vals_zip,
            fill_value=shap_expected_val_zip,
        )
    )

    return CrossValidationView(
        test_data,
        test_ans,
        pred_ans,
        pred_prob,
        shap_vals,
        shap_expected_val,
    )


def calc_score(
    prediction: Prediction, test_ans: NDArray, use_score: bool
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
        f1_result,  # type: ignore
        accuracy_result,  # type: ignore
        auc_roc_result,  # type: ignore
    )


def calculate_score_by_loocv(cv_view_data: CrossValidationView) -> Score:
    """
    LOOCVの結果からスコアを計算します。

    Args:
        cv_view_data (CvViewData): Cvの結果を表示するためのデータ

    Returns:
        Score: スコアデータ
    """

    f1_result = f1_score(y_true=cv_view_data.test_ans, y_pred=cv_view_data.pred_ans)
    accuracy_result = accuracy_score(
        y_true=cv_view_data.test_ans, y_pred=cv_view_data.pred_ans
    )
    auc_roc_result = roc_auc_score(
        y_true=cv_view_data.test_ans, y_score=cv_view_data.pred_prob
    )

    return Score(
        f1_result,  # type: ignore
        accuracy_result,  # type: ignore
        auc_roc_result,  # type: ignore
    )
