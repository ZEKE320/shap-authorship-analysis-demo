"""
シリアライザーモジュール
Serializer module
"""

from pathlib import Path

import pandas as pd

from authorship_tool.types_ import AuthorColl4dStr
from authorship_tool.util.table_util import display


def author_collection_to_csv(
    author_collection: AuthorColl4dStr, author_name: str, file_path: Path
) -> None:
    """
    1人の著者の文書集合をcsvファイルにシリアライズする。
    Serializes an AuthorCollection4dStr to a csv file.

    Args:
        author_collection (AuthorCollection4dStr): 1人の著者の文書集合 (Author's collection of documents)
    """

    fixed_cols: list[str] = [
        "Author name",
        "Document index",
        "Paragraph index",
        "Sentence index",
        "Sentence length",
    ]

    rows: list[list[int | str]] = [
        [author_name, doc_idx, para_idx, sent_idx, len(sent)] + sent
        for doc_idx, doc in enumerate(author_collection)
        for para_idx, para in enumerate(doc)
        for sent_idx, sent in enumerate(para)
    ]

    max_sent_size: int = max(len(row) - len(fixed_cols) for row in rows)

    cols: list[str] = fixed_cols + [f"Word {i+1}" for i in range(max_sent_size)]
    df = pd.DataFrame(rows, columns=cols)

    display(df.head())

    df.to_csv(file_path.joinpath(f"{author_name}.csv"), index=False)
