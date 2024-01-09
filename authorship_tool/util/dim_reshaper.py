"""配列の次元を変換するモジュール"""


from authorship_tool.types import TwoDimStr, OneDimStr


def reduce_dim(para: TwoDimStr, /) -> OneDimStr:
    """段落のリストを文章のリストに変換する"""
    return [word for sent in para for word in sent]


def one_dim_to_str(sent: OneDimStr, /) -> str:
    """文章のリストを文字列に変換する"""
    return " ".join(sent)


def two_dim_to_str(para: TwoDimStr, /) -> str:
    """段落のリストを文字列に変換する"""
    return one_dim_to_str(reduce_dim(para))
