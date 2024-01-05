"""テーブルを表示するためのモジュール"""
import pandas
from tabulate import tabulate


def display(data_frame: pandas.DataFrame):
    """テーブルを表示する

    Args:
        data_frame (pandas.DataFrame): 表示したいデータフレーム
    """
    print(
        tabulate(
            data_frame.values.tolist(),
            headers=data_frame.columns.tolist(),
            tablefmt="psql",
        )
    )
