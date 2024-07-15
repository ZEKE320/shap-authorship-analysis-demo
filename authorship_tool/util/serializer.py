"""
シリアライザーモジュール
Serializer module
"""

from pathlib import Path

import pandas as pd
from IPython.display import display

from authorship_tool.types_ import Doc3dStr, Title


def author_collection_to_csv(
    doc_by_title: dict[Title, Doc3dStr], author_name: str, file_path: Path
) -> None:
    """
    1人の著者の文書集合をcsvファイルにシリアライズする。
    Serializes an AuthorCollection4dStr to a csv file.

    Args:
        author_collection (AuthorCollection4dStr): 1人の著者の文書集合 (Author's collection of documents)
    """

    fixed_cols: list[str] = [
        "Author name",
        "Document name",
        "Paragraph number",
        "Sentence number",
    ]

    rows: list[list[int | str | object]] = [
        [author_name, title, para_idx + 1, sent_idx + 1] + sent
        for title, doc in doc_by_title.items()
        for para_idx, para in enumerate(doc)
        for sent_idx, sent in enumerate(para)
    ]

    max_sent_size: int = max(len(row) - len(fixed_cols) for row in rows)

    cols: list[str] = fixed_cols + [f"Word {i+1}" for i in range(max_sent_size)]
    df = pd.DataFrame(rows, columns=cols)

    display(df.head())

    df.to_csv(file_path.joinpath(f"{author_name}.csv"), index=False)
