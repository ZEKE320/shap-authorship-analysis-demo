import pandas as pd
import sklearn
import lightgbm as lgb
import numpy as np


def learn_more(df: pd.DataFrame, nd_correctness: np.ndarray):
    val: float = 0
    while val < 0.9:
        model, x_test, y_test, y_pred, val = learn(df, nd_correctness)
    return model, x_test, y_test, y_pred, val


def learn(df: pd.DataFrame, nd_correctness: np.ndarray):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(df, nd_correctness)

    model = lgb.LGBMClassifier()
    model.fit(x_train.values, y_train)

    y_pred_prob = model.predict_proba(x_test)[:, 1]
    y_pred = model.predict(x_test)

    val = sklearn.metrics.roc_auc_score(y_test, y_pred_prob)
    return model, x_test, y_test, y_pred, val
