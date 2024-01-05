"""配列の次元を変換するモジュール"""


def para2sent(para: list[list[str]], /) -> list[str]:
    """段落のリストを文章のリストに変換する"""
    return [word for sent in para for word in sent]


def sent2str(sent: list[str], /) -> str:
    """文章のリストを文字列に変換する"""
    return " ".join(sent)


def para2str(para: list[list[str]], /) -> str:
    """段落のリストを文字列に変換する"""
    return sent2str(para2sent(para))
