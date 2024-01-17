"""
テキスト読み込みモジュール
Text reader module
"""

from pathlib import Path

from authorship_tool.types_ import Document, Paragraph


def read_document(file_path: Path) -> list[Paragraph]:
    """
    ファイルから文書を読み込む
    Read a document from a file

    Args:
        file_path (Path): ファイルパス (File path)

    Raises:
        FileNotFoundError: ファイルが見つからない場合 (If the file is not found)

    Returns:
        str: 段落 (Document)
    """

    if not file_path.exists():
        raise FileNotFoundError(f"File: `{file_path}` could not be found.")

    with open(file_path, mode="r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def read_multi_documents(
    file_paths: list[Path], base_path: Path | None = None
) -> list[Document]:
    """
    同一のフォルダにある複数のファイルから文書を読み込む
    Read documents from multiple files in the same folder

    Args:
        base_path (Path): フォルダパス (Folder path)
        file_paths (list[Path]): ファイルパスのリスト (List of file paths)

    Raises:
        FileNotFoundError: ファイルが見つからない場合 (If the file is not found)

    Returns:
        list[Document]: 文書のリスト (List of documents)
    """

    if base_path is None:
        base_path = Path()

    return [read_document(base_path.joinpath(path)) for path in file_paths]
