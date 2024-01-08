"""特徴量のデータセットを生成するモジュール"""
from typing import Callable, Final, Optional

from authorship_tool.types import Para2dStr, Sent1dStr, Tag
from authorship_tool.util import type_guard
from authorship_tool.util.feature import calculator as f_calculator
from authorship_tool.util.feature import counter as f_counter


class FeatureDatasetGenerator:
    """特徴量のデータセットを生成するクラス"""

    def __init__(self, tags: Optional[list[Tag]] = None) -> None:
        if tags and not type_guard.is_tag_list(tags):
            raise ValueError("tags must be a list of str")

        cols_and_func: dict[str, Callable[[Sent1dStr], float]] = {
            "word variation": f_calculator.word_variation,
            "uncommon word frequency": f_calculator.uncommon_word_frequency,
            "sentence length": f_counter.sentence_length,
            "average word length": f_calculator.average_word_length,
        }

        col: list[str] = list(cols_and_func.keys())

        if tags:
            col.extend(tags)

        # クラスのフィールドを定義
        self.__cols_and_func: Final[
            dict[str, Callable[[Sent1dStr], float]]
        ] = cols_and_func
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
        freq_by_pos: dict[str, float] = f_calculator.all_pos_frequency(sent)

        return (
            tuple(
                [func(sent) for func in self.__cols_and_func.values()]
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
        sent: Sent1dStr = [word for sent in para for word in sent]
        return self.generate_from_sentence(sent, correctness)
