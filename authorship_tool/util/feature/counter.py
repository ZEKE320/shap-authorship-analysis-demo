"""文章の特徴量を計算するモジュール"""

import re

import nltk

from authorship_tool.types import Sent


def sentence_length(sent: Sent) -> int:
    """文章中に出現する単語数を計算する"""
    return len(sent)


def count_each_tokens(sent: Sent) -> int:
    """文章中に出現する単語の種類数を計算する"""
    return len(set(sent))


def count_selected_character(sent: Sent, character: str) -> int:
    """文章内で出現する指定した文字の合計を計算する"""
    return sent.count(character)


def count_non_alphabetic_characters(sent: Sent) -> int:
    """文章内で出現する記号の合計を計算する"""
    pattern = r"[^a-zA-Z\s]"
    matches: list[str] = re.findall(pattern=pattern, string=" ".join(sent))
    return len(matches)


def count_numeric_characters(sent: Sent) -> int:
    """文章内で出現する数字の合計を計算する"""
    pattern = r"[\d]"
    matches: list[str] = re.findall(pattern=pattern, string=" ".join(sent))
    return len(matches)


def count_uncommon_words(sent: Sent) -> int:
    """ストップワードではない単語の数を計算する"""
    stop_words = set(nltk.corpus.stopwords.words("english"))
    return len([word for word in sent if word not in stop_words])
