"""文章の特徴量を計算するモジュール"""

import re

import nltk

from authorship_tool.type_alias import Sent


class FeatureCounter:
    """文章の特徴量を計算するクラス"""

    @classmethod
    def sentence_length(cls, sent: Sent) -> int:
        """文章中に出現する単語数を計算する"""
        return len(sent)

    @classmethod
    def count_each_tokens(cls, sent: Sent) -> int:
        """文章中に出現する単語の種類数を計算する"""
        return len(set(sent))

    @classmethod
    def count_selected_character(cls, sent: Sent, character: str) -> int:
        """文章内で出現する指定した文字の合計を計算する"""
        return sent.count(character)

    @classmethod
    def count_non_alphabetic_characters(cls, sent: Sent) -> int:
        """文章内で出現する記号の合計を計算する"""
        pattern = r"[^a-zA-Z\s]"
        matches = re.findall(pattern, " ".join(words))
        return len(matches)

    @classmethod
    def count_uncommon_words(cls, sent: Sent) -> int:
        """ストップワードではない単語の数を計算する"""
        stop_words = set(nltk.corpus.stopwords.words("english"))
        return len([word for word in sent if word not in stop_words])
