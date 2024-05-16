"""トークナイザーモジュール"""

import nltk

from authorship_tool.types_ import Para2dStr, Paragraph


def tokenize_para(paragraph_text: Paragraph) -> Para2dStr:
    """
    文章を単語のリストのリストに分割する

    Args:
        text (str): テキスト

    Returns:
        Para2dStr: トークン化された段落
    """
    return [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(paragraph_text)]
