"""特徴量のデータセットを生成するモジュール"""
from typing import Callable, Final, Optional

from authorship_tool.util import FeatureCalculator, FeatureCounter, TypeGuardUtil
from authorship_tool.type_alias import Para, Sent, Tag


class FeatureDatasetGenerator:
    """特徴量のデータセットを生成するクラス"""

    def __init__(self, tags: Optional[tuple[Tag, ...]] = None) -> None:
        if tags and not TypeGuardUtil.is_tag_tuple(tags):
            raise ValueError("tags must be a tuple of str")

        cols_and_func: dict[str, Callable[[Sent], float]] = {
            "word variation": FeatureCalculator.word_variation,
            "uncommon word frequency": FeatureCalculator.uncommon_word_frequency,
            "sentence length": FeatureCounter.sentence_length,
            "average word length": FeatureCalculator.average_word_length,
        }

        col: list[str] = list(cols_and_func.keys())

        if tags:
            col.extend(tags)

        # クラスのフィールドを定義
        self.__cols_and_func: Final[dict[str, Callable[[Sent], float]]] = cols_and_func
        self.__columns: Final[tuple[str, ...]] = tuple(col)
        self.__tags: Final[tuple[Tag, ...]] = tags if tags else tuple()

    @property
    def columns(self) -> tuple[str, ...]:
        """特徴量の列名"""
        return self.__columns

    def generate_from_sentence(
        self, sent: Sent, correctness: bool
    ) -> tuple[tuple[float, ...], bool]:
        """文章のリストから特徴量のリストを生成する"""
        freq_by_pos: dict[str, float] = FeatureCalculator.all_pos_frequency(sent)

        return (
            tuple(
                [func(sent) for func in self.__cols_and_func.values()]
                + [freq_by_pos.get(tag, 0.0) for tag in self.__tags]
            ),
            correctness,
        )

    def generate_from_paragraph(
        self,
        para: Para,
        correctness: bool,
    ) -> tuple[tuple[float, ...], bool]:
        """シェイプを落としてから特徴量のリストを生成する"""
        sent: Sent = [word for sent in para for word in sent]
        return self.generate_from_sentence(sent, correctness)
