"""
分類器 交差検証モジュール
Classifier cross validation module
"""

from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, LeaveOneOut

from authorship_tool.util.ml.model import (
    CrossValidationResult,
    CvGlobalExplanationData,
    LGBMSource,
    Score,
    SplittedDataset,
    TrainingResult,
)
from authorship_tool.util.ml.trainer import train

SCORE_CALC_DEFAULT: Final[bool] = False


def train_loocv(source: LGBMSource) -> list[TrainingResult]:
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

    return results


def train_kfold(source: LGBMSource, k: int) -> list[TrainingResult]:
    """
    KFoldで学習を行います。

    Args:
        source (LGBMSource): LGBMのモデル作成用ソースデータ
        k (int): KFoldの分割数

    Returns:
        CvViewData: Cvの結果を表示するためのデータ
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=0)

    results: list[TrainingResult] = [
        train_by_index(source, train_indices, test_index, use_score_calc=False)
        for train_indices, test_index in kf.split(source.feature_data_frame)
    ]

    return results


def train_by_index(
    source: LGBMSource,
    train_indices: npt.NDArray,
    test_index: npt.NDArray,
    use_score_calc: bool = SCORE_CALC_DEFAULT,
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

    models_zip, splitted_datasets_zip, predictions_zip, shap_data_zip, scores = zip(
        *[
            (
                result.model,
                result.splitted_dataset,
                result.prediction,
                result.shap_data,
                result.score,
            )
            for result in results
        ],
        strict=True,
    )

    return CrossValidationResult(
        tuple(models_zip),
        tuple(splitted_datasets_zip),
        tuple(predictions_zip),
        tuple(shap_data_zip),
        tuple(scores),
    )


def convert_cv_result_to_global_exp_data(
    cv_result: CrossValidationResult,
) -> CvGlobalExplanationData:
    """
    LGBMCvResultをCvViewDataに変換します。

    Args:
        cv_result (LGBMCvResult): LGBMCvResultインスタンス

    Returns:
        CvViewData: Cvの結果を表示するためのデータ
    """

    (test_data_zip, test_ans_zip) = zip(
        *[
            (splitted_dataset.test_data, splitted_dataset.test_ans)
            for splitted_dataset in cv_result.splitted_datasets
        ],
        strict=True,
    )
    (pred_prob_zip, pred_ans_zip) = zip(
        *[
            (prediction.pred_prob, prediction.pred_ans)
            for prediction in cv_result.predictions
        ],
        strict=True,
    )
    shap_vals_list: list[npt.NDArray[np.float64]] = [
        shap_data.shap_values for shap_data in cv_result.shap_data_tuple
    ]
    explanation_list = [
        shap_data.explanation for shap_data in cv_result.shap_data_tuple
    ]
    explanation = shap.Explanation(
        values=np.vstack([e.values for e in explanation_list]),
        base_values=np.hstack([e.base_values for e in explanation_list]),
        data=np.vstack([e.data for e in explanation_list]),
        feature_names=explanation_list[0].feature_names,
    )

    test_data: pd.DataFrame = pd.concat(test_data_zip)
    test_ans: npt.NDArray[np.bool_] = np.concatenate(test_ans_zip)

    pred_ans: npt.NDArray[np.bool_] = np.concatenate(pred_ans_zip)
    pred_prob: npt.NDArray[np.float64] = np.concatenate(pred_prob_zip)

    shap_vals: npt.NDArray[np.float64] = np.concatenate(shap_vals_list)

    return CvGlobalExplanationData(
        test_data,
        test_ans,
        pred_ans,
        pred_prob,
        shap_vals,
        explanation
    )


def calc_score_for_cv(cv_view_data: CvGlobalExplanationData) -> Score:
    """
    LOOCVの結果からスコアを計算します。

    Args:
        cv_view_data (CvViewData): Cvの結果を表示するためのデータ

    Returns:
        Score: スコアデータ
    """

    f1_result: float = f1_score(
        y_true=cv_view_data.test_ans, y_pred=cv_view_data.pred_ans
    )
    accuracy_result: float = accuracy_score(
        y_true=cv_view_data.test_ans, y_pred=cv_view_data.pred_ans
    )
    auc_roc_result: float = roc_auc_score(
        y_true=cv_view_data.test_ans, y_score=cv_view_data.pred_prob
    )

    return Score(
        f1_result,
        accuracy_result,
        auc_roc_result,
    )
