import re

import nltk
from nltk.corpus import stopwords


def sentence_length(sentence: list[str]) -> int:
    """文章中に出現する単語数を計算する"""
    return len(sentence)


def count_word_types(sentence: list[str]) -> int:
    """文章中に出現する単語の種類数を計算する"""
    return len(set(sentence))


def count_character(sentence: list[str], character: str) -> int:
    """文章内で出現する指定した文字の合計を計算する"""
    return sentence.count(character)


def count_comma(sentence: list[str]) -> int:
    """文章内で出現するカンマの合計を計算する"""
    return count_character(sentence, ",")


def count_period(sentence: list[str]) -> int:
    """文章内で出現するピリオドの合計を計算する"""
    return count_character(sentence, ".")


def count_attention_mark(sentence: list[str]) -> int:
    """文章内で出現する感嘆符の合計を計算する"""
    return count_character(sentence, "!")


def count_question_mark(sentence: list[str]) -> int:
    """文章内で出現する疑問符の合計を計算する"""
    return count_character(sentence, "?")


def count_double_quotation(sentence: list[str]) -> int:
    """文章内で出現する二重引用符の合計を計算する"""
    return count_character(sentence, '"')


def count_single_quotation(sentence: list[str]) -> int:
    """文章内で出現する一重引用符の合計を計算する"""
    return count_character(sentence, "'")


def count_semicolon(sentence: list[str]) -> int:
    """文章内で出現するセミコロンの合計を計算する"""
    return count_character(sentence, ";")


def count_colon(sentence: list[str]) -> int:
    """文章内で出現するコロンの合計を計算する"""
    return count_character(sentence, ":")


def count_non_alphabetic_characters(sentence: list[str]) -> int:
    """文章内で出現する記号の合計を計算する"""
    pattern = r"[^a-zA-Z\s]"
    matches = re.findall(pattern, sentence)
    return len(matches)


def count_uncommon_words(sentence: list[str]) -> int:
    """ストップワードではない単語の数を計算する"""
    stop_words = set(stopwords.words("english"))
    return len([word for word in sentence if word not in stop_words])
