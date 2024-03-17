"""authorship_tool.util.tokenizerのテストモジュール"""
from authorship_tool.util import tokenizer

TEXT = "This is a pen, Mr. Smith. This pen is so expensive, don't you think? Oh, no! He stole my pen...!"


def test_tokenize_para():
    """段落をトークン化するテスト"""

    assert tokenizer.tokenize_para(TEXT) == [
        ["This", "is", "a", "pen", ",", "Mr.", "Smith", "."],
        ["This", "pen", "is", "so", "expensive", ",", "do", "n't", "you", "think", "?"],
        ["Oh", ",", "no", "!"],
        ["He", "stole", "my", "pen", "...", "!"],
    ]
