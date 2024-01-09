"""トークナイザーモジュール"""
import nltk
from authorship_tool.types import TwoDimStr


def tokenize_to_para(paragraph_text: str) -> TwoDimStr:
    """
    文章を単語のリストのリストに分割する

    Args:
        text (str): テキスト

    Returns:
        Para2dStr: トークン化された段落
    """
    return [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(paragraph_text)]
