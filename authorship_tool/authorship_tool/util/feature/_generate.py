"""特徴量のデータセットを生成するモジュール"""
from typing import Optional

from authorship_tool.util import FeatureCalculator, FeatureCounter, TypeGuardUtil
from authorship_tool.type_alias import Para, Sent, Tag


class FeatureDatasetGenerator:
    """特徴量のデータセットを生成するクラス"""

    def __init__(self, tags: Optional[list[str]] = None) -> None:
        self.columns: list[str] = [
            "word variation",
            "uncommon word frequency",
            "sentence length",
            "average word length",
        ]
        if tags is not None:
            self.columns.extend(tags)

    def generate(
        self, words: list[str], tags: list[str], correctness: bool
    ) -> tuple[list[float], bool]:
        """文章のリストから特徴量のリストを生成する"""
        freq_by_pos: dict[str, float] = FeatureCalculator.all_pos_frequency(words)

        return (
            [
                FeatureCalculator.word_variation(words),
                FeatureCalculator.uncommon_word_frequency(words),
                FeatureCounter.sentence_length(words),
                FeatureCalculator.average_word_length(words),
            ]
            + [freq_by_pos.get(tag, 0) for tag in tags],
            correctness,
        )

    def reshape_and_generate(
        self,
        words_collection: list[list[str]],
        tags: list[str],
        correctness: bool,
    ) -> tuple[list[float], bool]:
        """シェイプを落としてから特徴量のリストを生成する"""
        words: list[str] = [word for words in words_collection for word in words]
        return self.generate(words, tags, correctness)
