"""配列の次元を変換するモジュール"""


class ArrayDimensionReshaper:
    """配列の次元を変換するクラス"""

    @classmethod
    def para2sent(cls, para: list[list[str]], /) -> list[str]:
        """段落のリストを文章のリストに変換する"""
        return [word for sent in para for word in sent]

    @classmethod
    def sent2str(cls, sent: list[str], /) -> str:
        """文章のリストを文字列に変換する"""
        return " ".join(sent)

    @classmethod
    def para2str(cls, para: list[list[str]], /) -> str:
        """段落のリストを文字列に変換する"""
        return cls.sent2str(cls.para2sent(para))
