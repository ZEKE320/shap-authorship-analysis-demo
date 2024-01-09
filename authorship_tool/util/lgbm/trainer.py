"""lightgbmトレーナー"""

import sklearn.metrics
import sklearn.model_selection
from lightgbm import LGBMClassifier

from authorship_tool.util.lgbm.model import LGBMResultModel, LGBMSourceModel


# def learn_until_succeed(training_source: LGBMSourceModel) -> LGBMResultModel:
#     """指定したROC AUCスコアを超えるまで、著者推定モデルを学習します。

#     Args:
#         df (pd.DataFrame): _description_
#         nd_correctness (np.ndarray): _description_
#         desired_score (float): _description_

#     Returns:
#         tuple:
#         学習済みモデル、学習用データ、テスト用データ、学習用正解ラベル、テスト用正解ラベル、
#         テスト用予測確率、テスト用予測ラベル、ROC AUCスコア
#     """

#     while True:
#         training_result: LGBMResultModel = learn(training_source)
#         if training_result.auc_roc_score >= training_source.desired_score:
#             break

#     return training_result


def learn(training_dto: LGBMSourceModel) -> LGBMResultModel:
    """LightGBMを使って、著者推定モデルを学習します。

    Args:
        df (pd.DataFrame): 特徴量のデータフレーム
        nd_correctness (np.ndarray): 正解ラベルの配列

    Returns:
        tuple:
        学習済みモデル、学習用データ、テスト用データ、学習用正解ラベル、テスト用正解ラベル、
        テスト用予測確率、テスト用予測ラベル、ROC AUCスコア
    """

    (
        train_data,
        test_data,
        train_ans,
        test_ans,
    ) = sklearn.model_selection.train_test_split(
        training_dto.df, training_dto.nd_correctness
    )

    model = LGBMClassifier()
    model.fit(X=train_data.values, y=train_ans)

    ans_pred_prob = model.predict_proba(X=test_data)[:, 1]
    ans_pred = model.predict(X=test_data)

    auc_roc_score = sklearn.metrics.roc_auc_score(
        y_true=test_ans, y_score=ans_pred_prob
    )

    return LGBMResultModel(
        model,
        train_data,
        test_data,
        train_ans,
        test_ans,
        ans_pred_prob,
        ans_pred,
        auc_roc_score,
    )


def loocv_learn():
    pass
