"""authorship_tool.util.tokenizerのテストモジュール"""
from authorship_tool.util import tokenizer

text = "This is a pen, Mr. Smith. This pen is so expensive, don't you think? Oh, no! He stole my pen...!"


def test_tokenizeToPara():
    assert tokenizer.tokenize_to_para(text) == [
        ["This", "is", "a", "pen", ",", "Mr.", "Smith", "."],
        ["This", "pen", "is", "so", "expensive", ",", "do", "n't", "you", "think", "?"],
        ["Oh", ",", "no", "!"],
        ["He", "stole", "my", "pen", "...", "!"],
    ]
