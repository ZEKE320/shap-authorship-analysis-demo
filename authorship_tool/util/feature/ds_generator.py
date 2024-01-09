"""
特徴量データセット生成モジュール
Feature dataset generator module
"""
from typing import Callable, Final, Optional

from authorship_tool.types import Para2dStr, Sent1dStr, Tag
from authorship_tool.util import dim_reshaper, type_guard
from authorship_tool.util.feature.calculator import (
    ParagraphCalculator,
    SentenceCalculator,
    UnivKansasFeatures,
)


class SentenceFeatureDatasetGenerator:
    """文の特徴量のデータセットを生成するクラス"""

    __COLS_AND_FUNC: Final[dict[str, Callable[[Sent1dStr], float | int]]] = {
        "word variation": SentenceCalculator.word_variation,
        "uncommon word frequency": SentenceCalculator.uncommon_word_frequency,
        "sentence length": SentenceCalculator.sentence_length,
        "average word length": SentenceCalculator.average_word_length,
    }

    def __init__(self, tags: Optional[list[Tag]] = None) -> None:
        if tags and not type_guard.is_tag_list(tags):
            raise ValueError("tags must be a list of str")

        col: list[str] = list(SentenceFeatureDatasetGenerator.__COLS_AND_FUNC.keys())

        if tags:
            col.extend(tags)

        # クラスのフィールドを定義
        self.__columns: Final[tuple[str, ...]] = tuple(col)
        self.__tags: Final[list[Tag]] = tags if tags else []

    @property
    def columns(self) -> tuple[str, ...]:
        """特徴量の列名"""
        return self.__columns

    def generate_from_sentence(
        self, sent: Sent1dStr, correctness: bool
    ) -> tuple[tuple[float, ...], bool]:
        """文字列のリストから特徴量のリストを生成する"""
        freq_by_pos: dict[str, float] = SentenceCalculator.pos_frequencies(sent)

        return (
            tuple(
                [func(sent) for func in self.__COLS_AND_FUNC.values()]
                + [freq_by_pos.get(tag, 0.0) for tag in self.__tags]
            ),
            correctness,
        )

    def generate_from_paragraph(
        self,
        para: Para2dStr,
        correctness: bool,
    ) -> tuple[tuple[float, ...], bool]:
        """文字列のリストのリストから特徴量のリストを生成する"""
        sent: Sent1dStr = dim_reshaper.para_to_1d(para)
        return self.generate_from_sentence(sent, correctness)


