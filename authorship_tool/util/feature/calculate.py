"""特徴計算モジュール"""

import nltk
from rich.console import Console

from authorship_tool.util import FeatureCounter, PosFeature
from authorship_tool.type_alias import Sent, Tag, TaggedToken

console = Console(highlight=False)


class FeatureCalculator:
    """特徴量を計算するクラス"""

    @classmethod
    def word_variation(cls, sent: Sent) -> float:
        """文章中の単語の豊富さを計算する: (単語の種類の数)/(文章中の単語数)"""
        return FeatureCounter.count_each_tokens(sent) / FeatureCounter.sentence_length(
            sent
        )

    @classmethod
    def average_word_length(cls, sent: Sent) -> float:
        """文章中の単語の平均文字列長を計算する"""
        return sum(len(word) for word in sent) / FeatureCounter.sentence_length(sent)

    @classmethod
    def non_alphabetic_characters_frequency(cls, sent: Sent) -> float:
        """文章内で出現する記号の割合を計算する"""
        return FeatureCounter.count_non_alphabetic_characters(
            sent
        ) / FeatureCounter.sentence_length(sent)

    @classmethod
    def uncommon_word_frequency(cls, sent: Sent) -> float:
        """特徴的な単語の割合を計算する"""
        return FeatureCounter.count_uncommon_words(
            sent
        ) / FeatureCounter.sentence_length(sent)

    @classmethod
    def all_pos_frequency(cls, sent: Sent) -> dict[Tag, float]:
        """文章中の各品詞の割合を計算する"""
        pos_feature: PosFeature = PosFeature(sent).tag_subcategories()
        tagged_tokens: list[TaggedToken] = pos_feature.tagged_tokens

        # TODO 過去分詞形容詞を確認する為なので、後々削除する
        if "JJ_pp" in set(pos for (_, pos) in tagged_tokens):
            console.print(f"{pos_feature}\n")

        freq_dist = nltk.FreqDist(tagged_tokens)

        total_tags: int = freq_dist.N()
        return {tag[1]: count / total_tags for (tag, count) in freq_dist.items()}
