"""配列の次元を変換するモジュール"""


from authorship_tool.types import Para2dStr, Sent1dStr


def para_to_1d(para: Para2dStr, /) -> Sent1dStr:
    """段落のリストを文章のリストに変換する"""
    return [word for sent in para for word in sent]


def sent_to_str(sent: Sent1dStr, /) -> str:
    """文章のリストを文字列に変換する"""
    return " ".join(sent)


def para_to_str(para: Para2dStr, /) -> str:
    """段落のリストを文字列に変換する"""
    return sent_to_str(para_to_1d(para))
