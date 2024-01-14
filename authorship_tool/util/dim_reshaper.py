"""配列の次元を変換するモジュール"""

from authorship_tool.types_ import Para2dStr, Sent1dStr


def reduce_dim(two_dim_str: list | set | tuple, /) -> list:
    """段落のリストを文章のリストに変換する"""
    return [single_str for one_dim_str in two_dim_str for single_str in one_dim_str]


def one_dim_to_str(sent: Sent1dStr, /) -> str:
    """文章のリストを文字列に変換する"""
    return " ".join(sent)


def two_dim_to_str(para: Para2dStr, /) -> str:
    """段落のリストを文字列に変換する"""
    return one_dim_to_str(reduce_dim(para))
