import nltk
from nltk import FreqDist
from authorship_tool.util import feature_counter as fct


def word_variation(sentence: list[str]) -> float:
    """文章中の単語の豊富さを計算する: (単語の種類の数)/(文章中の単語数)"""
    return fct.count_word_types(sentence) / fct.sentence_length(sentence)


def average_word_length(sentence: list[str]) -> float:
    """文章中の単語の平均文字数を計算する"""
    return sum([len(word) for word in sentence]) / fct.sentence_length(sentence)


def comma_frequency(sentence: list[str]) -> float:
    """文章内で出現するカンマの割合を計算する"""
    return fct.count_comma(sentence) / fct.sentence_length(sentence)


def period_frequency(sentence: list[str]) -> float:
    """文章内で出現するピリオドの割合を計算する"""
    return fct.count_period(sentence) / fct.sentence_length(sentence)


def attention_mark_frequency(sentence: list[str]) -> float:
    """文章内で出現する感嘆符の割合を計算する"""
    return fct.count_attention_mark(sentence) / fct.sentence_length(sentence)


def question_mark_frequency(sentence: list[str]) -> float:
    """文章内で出現する疑問符の割合を計算する"""
    return fct.count_question_mark(sentence) / fct.sentence_length(sentence)


def double_quotation_frequency(sentence: list[str]) -> float:
    """文章内で出現する二重引用符の割合を計算する"""
    return fct.count_double_quotation(sentence) / fct.sentence_length(sentence)


def single_quotation_frequency(sentence: list[str]) -> float:
    """文章内で出現する一重引用符の割合を計算する"""
    return fct.count_single_quotation(sentence) / fct.sentence_length(sentence)


def semicolon_frequency(sentence: list[str]) -> float:
    """文章内で出現するセミコロンの割合を計算する"""
    return fct.count_semicolon(sentence) / fct.sentence_length(sentence)


def colon_frequency(sentence: list[str]) -> float:
    """文章内で出現するコロンの割合を計算する"""
    return fct.count_colon(sentence) / fct.sentence_length(sentence)


def non_alphabetic_characters_frequency(sentence: list[str]) -> float:
    """文章内で出現する記号の割合を計算する"""
    return fct.count_non_alphabetic_characters(sentence) / fct.sentence_length(sentence)


def uncommon_word_frequency(sentence: list[str]) -> float:
    """ストップワードではない単語の割合を計算する"""
    return fct.count_uncommon_words(sentence) / fct.sentence_length(sentence)


def all_pos_frequency(sentence: list[str]) -> dict[str, int]:
    """文章中の全ての品詞の割合を計算する"""
    pos_list = nltk.pos_tag(sentence)
    fd = FreqDist(pos_list)

    total_tags = fd.N()
    return {tag[1]: count / total_tags for tag, count in fd.items()}
