"""
分類機 分析モジュール
Classifier analyzer module
"""

from typing import Final

import shap
from lightgbm import LGBMClassifier
from pandas import DataFrame

from authorship_tool.util.ml.model import ShapData

SHAP_VALUE_POSITIVE_IDX: Final[int] = 1


def create_shap_data(model: LGBMClassifier, test_data: DataFrame) -> ShapData:
    """
    SHAPデータを作成します。

    Args:
        model (LGBMClassifier): モデル
        test_data (pd.DataFrame): テストデータ

    Returns:
        ShapData: SHAPデータ
    """

    tree_explainer = shap.TreeExplainer(model)

    return ShapData(
        explainer=tree_explainer,
        explanation=tree_explainer(test_data),
        shap_values=tree_explainer.shap_values(test_data),
        base_values=tree_explainer.expected_value,
        data=test_data,
    )
