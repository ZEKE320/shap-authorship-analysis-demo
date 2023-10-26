import pandas as pd
from tabulate import tabulate


def display(df: pd.DataFrame):
    print(tabulate(df, headers="keys", tablefmt="psql"))
