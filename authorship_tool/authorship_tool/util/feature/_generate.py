"""特徴量のデータセットを生成するモジュール"""
from typing import Optional

from authorship_tool.util import FeatureCalculator, FeatureCounter, TypeGuardUtil
from authorship_tool.type_alias import Para, Sent, Tag


class FeatureDatasetGenerator:
    """特徴量のデータセットを生成するクラス"""

    def __init__(self, tags: Optional[set[Tag]] = None) -> None:
        if tags and not TypeGuardUtil.is_tag_set(tags):
            raise ValueError("tags must be a set of str")

        col: list[str] = [
            "word variation",
            "uncommon word frequency",
            "sentence length",
            "average word length",
        ]

        if tags:
            col.extend(sorted(tags))

        self.__columns: tuple[str, ...] = tuple(col)

    @property
    def columns(self) -> tuple[str, ...]:
        """特徴量の列名"""
        return self.__columns

    def generate_from_sentence(
        self, sent: Sent, tags: set[Tag], correctness: bool
    ) -> tuple[tuple[float, ...], bool]:
        """文章のリストから特徴量のリストを生成する"""
        freq_by_pos: dict[str, float] = FeatureCalculator.all_pos_frequency(sent)

        return (
            tuple(
                [
                    FeatureCalculator.word_variation(sent),
                    FeatureCalculator.uncommon_word_frequency(sent),
                    FeatureCounter.sentence_length(sent),
                    FeatureCalculator.average_word_length(sent),
                ]
                + [freq_by_pos.get(tag, 0.0) for tag in tags]
            ),
            correctness,
        )

    def generate_from_paragraph(
        self,
        para: Para,
        tags: set[Tag],
        correctness: bool,
    ) -> tuple[tuple[float, ...], bool]:
        """シェイプを落としてから特徴量のリストを生成する"""
        sent: Sent = [word for sent in para for word in sent]
        return self.generate_from_sentence(sent, tags, correctness)
