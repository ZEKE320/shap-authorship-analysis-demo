"""
分類機 分析モジュール
Classifier analyzer module
"""

from typing import Final

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier

from authorship_tool.util.ml.model import ShapData

SHAP_VALUE_POSITIVE_IDX: Final[int] = 1


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
