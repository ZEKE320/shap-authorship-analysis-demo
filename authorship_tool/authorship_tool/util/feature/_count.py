"""文章の特徴量を計算するモジュール"""

import re

import nltk

from authorship_tool.type_alias import Sent


class FeatureCounter:
    """文章の特徴量を計算するクラス"""

    @classmethod
    def sentence_length(cls, words: list[str]) -> int:
        """文章中に出現する単語数を計算する"""
        return len(words)

    @classmethod
    def count_word_types(cls, words: list[str]) -> int:
        """文章中に出現する単語の種類数を計算する"""
        return len(set(words))

    @classmethod
    def count_character(cls, words: list[str], character: str) -> int:
        """文章内で出現する指定した文字の合計を計算する"""
        return words.count(character)

    @classmethod
    def count_comma(cls, words: list[str]) -> int:
        """文章内で出現するカンマの合計を計算する"""
        return cls.count_character(words, ",")

    @classmethod
    def count_period(cls, words: list[str]) -> int:
        """文章内で出現するピリオドの合計を計算する"""
        return cls.count_character(words, ".")

    @classmethod
    def count_attention_mark(cls, words: list[str]) -> int:
        """文章内で出現する感嘆符の合計を計算する"""
        return cls.count_character(words, "!")

    @classmethod
    def count_question_mark(cls, words: list[str]) -> int:
        """文章内で出現する疑問符の合計を計算する"""
        return cls.count_character(words, "?")

    @classmethod
    def count_double_quotation(cls, words: list[str]) -> int:
        """文章内で出現する二重引用符の合計を計算する"""
        return cls.count_character(words, '"')

    @classmethod
    def count_single_quotation(cls, words: list[str]) -> int:
        """文章内で出現する一重引用符の合計を計算する"""
        return cls.count_character(words, "'")

    @classmethod
    def count_semicolon(cls, words: list[str]) -> int:
        """文章内で出現するセミコロンの合計を計算する"""
        return cls.count_character(words, ";")

    @classmethod
    def count_colon(cls, words: list[str]) -> int:
        """文章内で出現するコロンの合計を計算する"""
        return cls.count_character(words, ":")

    @classmethod
    def count_non_alphabetic_characters(cls, words: list[str]) -> int:
        """文章内で出現する記号の合計を計算する"""
        pattern = r"[^a-zA-Z\s]"
        matches = re.findall(pattern, " ".join(words))
        return len(matches)

    @classmethod
    def count_uncommon_words(cls, words: list[str]) -> int:
        """ストップワードではない単語の数を計算する"""
        stop_words = set(nltk.corpus.stopwords.words("english"))
        return len([word for word in words if word not in stop_words])
