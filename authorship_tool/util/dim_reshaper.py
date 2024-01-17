"""配列の次元を変換するモジュール"""

from authorship_tool.types_ import Para2dStr, Sent1dStr


def reduce_dim(coll: list | set | tuple, times: int = 1, /) -> list:
    """段落のリストを文章のリストに変換する"""

    for _ in range(times):
        coll = [str_1d for str_2d in coll for str_1d in str_2d]

    return list(coll)


def one_dim_to_str(sent: Sent1dStr, /) -> str:
    """文章のリストを文字列に変換する"""
    return " ".join(sent)


def two_dim_to_str(para: Para2dStr, /) -> str:
    """段落のリストを文字列に変換する"""
    return one_dim_to_str(reduce_dim(para))
