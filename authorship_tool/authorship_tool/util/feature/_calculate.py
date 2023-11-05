"""特徴計算モジュール"""

import nltk

from authorship_tool.util import FeatureCounter, PosFeature


class FeatureCalculator:
    """特徴量を計算するクラス"""

    @classmethod
    def word_variation(cls, words: list[str]) -> float:
        """文章中の単語の豊富さを計算する: (単語の種類の数)/(文章中の単語数)"""
        return FeatureCounter.count_word_types(words) / FeatureCounter.sentence_length(
            words
        )

    @classmethod
    def average_word_length(cls, words: list[str]) -> float:
        """文章中の単語の平均文字数を計算する"""
        return sum([len(word) for word in words]) / FeatureCounter.sentence_length(
            words
        )

    @classmethod
    def comma_frequency(cls, words: list[str]) -> float:
        """文章内で出現するカンマの割合を計算する"""
        return FeatureCounter.count_comma(words) / FeatureCounter.sentence_length(words)

    @classmethod
    def period_frequency(cls, words: list[str]) -> float:
        """文章内で出現するピリオドの割合を計算する"""
        return FeatureCounter.count_period(words) / FeatureCounter.sentence_length(
            words
        )

    @classmethod
    def attention_mark_frequency(cls, words: list[str]) -> float:
        """文章内で出現する感嘆符の割合を計算する"""
        return FeatureCounter.count_attention_mark(
            words
        ) / FeatureCounter.sentence_length(words)

    @classmethod
    def question_mark_frequency(cls, words: list[str]) -> float:
        """文章内で出現する疑問符の割合を計算する"""
        return FeatureCounter.count_question_mark(
            words
        ) / FeatureCounter.sentence_length(words)

    @classmethod
    def double_quotation_frequency(cls, words: list[str]) -> float:
        """文章内で出現する二重引用符の割合を計算する"""
        return FeatureCounter.count_double_quotation(
            words
        ) / FeatureCounter.sentence_length(words)

    @classmethod
    def single_quotation_frequency(cls, words: list[str]) -> float:
        """文章内で出現する一重引用符の割合を計算する"""
        return FeatureCounter.count_single_quotation(
            words
        ) / FeatureCounter.sentence_length(words)

    @classmethod
    def semicolon_frequency(cls, words: list[str]) -> float:
        """文章内で出現するセミコロンの割合を計算する"""
        return FeatureCounter.count_semicolon(words) / FeatureCounter.sentence_length(
            words
        )

    @classmethod
    def colon_frequency(cls, words: list[str]) -> float:
        """文章内で出現するコロンの割合を計算する"""
        return FeatureCounter.count_colon(words) / FeatureCounter.sentence_length(words)

    @classmethod
    def non_alphabetic_characters_frequency(cls, words: list[str]) -> float:
        """文章内で出現する記号の割合を計算する"""
        return FeatureCounter.count_non_alphabetic_characters(
            words
        ) / FeatureCounter.sentence_length(words)

    @classmethod
    def uncommon_word_frequency(cls, words: list[str]) -> float:
        """ストップワードではない単語の割合を計算する"""
        return FeatureCounter.count_uncommon_words(
            words
        ) / FeatureCounter.sentence_length(words)

    @classmethod
    def all_pos_frequency(cls, words: list[str]) -> dict[str, float]:
        """文章中の全ての品詞の割合を計算する"""
        detailed_pos_list = PosFeature(words).subcategory().words_and_pos
        freq_dist = nltk.FreqDist(detailed_pos_list)

        total_tags = freq_dist.N()
        return {tag[1]: count / total_tags for tag, count in freq_dist.items()}
