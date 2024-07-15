import re

import numpy as np  # linear algebra
import polars as pl  # data processing, CSV file I/O (e.g. pl.read_csv)
from dateutil import parser
from IPython.display import display

from authorship_tool.util.path_util import DatasetPaths


def generate_datasets():
    # %%
    datasetPaths = DatasetPaths()

    # chunk = pd.read_csv(datasetPaths.enron_dataset, chunksize=5000)
    # data = next(chunk)
    data = pl.read_csv(datasetPaths.enron_dataset).with_row_index(name="index")

    # %%
    print(data.get_column("message")[2])

    # %%
    display(data.head())

    # %%
    x = len(data)
    headers = ["From: ", "Subject: "]

    print("---\nRemoving emails without headers...\n---")
    for i, v in enumerate(headers):
        print(f"---\nChecking `{v}`...\n---")
        data = standard_format(data, data.get_column("message"), v)
        print("Done✅")
    data = data.drop("index").with_row_index(name="index")
    print(
        "Got rid of {} useless emails! That's {}% of the total number of messages in this dataset.".format(
            x - len(data),
            np.round(((x - len(data)) / x) * 100, decimals=2),
        )
    )

    # %%
    print("---\nExtracting text from message...\n---")
    texts = pl.Series("text", get_text(data.get_column("message"), 15))
    print("Done✅")

    # %%
    print("---\nExtracting date from message...\n---")
    date_rows = pl.Series("date", get_row(data.get_column("message"), 1))
    print("Done✅")

    # %%
    print("---\nExtracting senders from message...\n---")
    sender_rows = pl.Series("sender", get_row(data.get_column("message"), 2))
    print("Done✅")

    # %%
    print("---\nExtracting subject from message...\n---")
    subject_rows = pl.Series("subject", get_row(data.get_column("message"), 4))
    print("Done✅")

    # %%
    print("---\nConverting date to datetime object... (This takes a while)\n---")

    dates_str = date_rows.str.strip_prefix("Date: ")
    datetime_objects = [parser.parse(date) for date in dates_str]
    dates = pl.Series("date", datetime_objects)

    print("Done✅")

    # %%
    print("---\nStripping headers from text...\n---")

    subjects = subject_rows.str.strip_prefix("Subject: ")
    senders = sender_rows.str.strip_prefix("From: ")
    print("Done✅")

    # %%
    print("---\nFinalizing dataset...\n---")

    data_cleaned = data.with_columns([dates, senders, subjects, texts]).drop(
        ["file", "message"]
    )
    print("Done✅")

    # %%
    display(data_cleaned.head())

    # %%
    data_cleaned.write_csv(datasetPaths.enron_dataset_cleaned)


# %%
def get_text(series: pl.Series, row_num_slicer: int) -> list[str]:
    result = []
    for row, message in enumerate(series):
        if row % 1000 == 0:
            print(f"{row / len(series) * 100:.1f}%...")
        message_words = message.split("\n")
        del message_words[:row_num_slicer]
        result.append("\n".join(message_words))
    return result


def get_row(series: pl.Series, row_num: int) -> list[str]:
    result = []
    for row, message in enumerate(series):
        if row % 1000 == 0:
            print(f"{row / len(series) * 100:.1f}%...")
        message_words = message.split("\n")
        message_words = message_words[row_num]
        result.append(message_words)
    return result


def get_address(
    series: pl.Series,
) -> tuple[list[str]]:
    address = re.compile("[\w\.-]+@[\w\.-]+\.\w+")
    result1 = []
    for i in range(len(series)):
        if i % 1000 == 0:
            print(f"{i / len(series) * 100:.1f}%...")
        for message in series:
            correspondents = re.findall(address, message)
            result1.append(correspondents[0])
    return result1


def standard_format(
    df: pl.DataFrame,
    series: pl.Series,
    string: str,
) -> pl.DataFrame:
    rows = []
    for row, message in enumerate(series):
        if row % 1000 == 0:
            print(f"{row / len(series) * 100:.1f}%...")
        message_words = message.split("\n")[:20]
        if not any(line.startswith(string) for line in message_words):
            rows.append(row)

    df = df.filter(~pl.col("index").is_in(rows))
    return df


if __name__ == "__main__":
    generate_datasets()
