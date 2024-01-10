"""
特徴量データセット生成モジュール
Feature dataset generator module
"""
from typing import Callable, Final

import numpy as np
from numpy.typing import NDArray

from authorship_tool.types import Para2dStr, Sent1dStr, Tag
from authorship_tool.util import dim_reshaper, type_guard
from authorship_tool.util.feature.calculator import (
    ParagraphCalculator,
    SentenceCalculator,
    UnivKansasFeatures,
)


class SentenceFeatureDatasetGenerator:
    """文の特徴量のデータセットを生成するクラス"""

    __COLS_AND_FUNC: Final[dict[str, Callable[[Sent1dStr], np.float64 | int]]] = {
        "word variation": SentenceCalculator.word_variation,
        "uncommon word frequency": SentenceCalculator.uncommon_word_frequency,
        "sentence length": SentenceCalculator.sentence_length,
        "average word length": SentenceCalculator.average_token_length,
    }

    def __init__(self, tags: tuple[Tag, ...] | None = None) -> None:
        if tags and not type_guard.is_tag_tuple(tags):
            raise ValueError("tags must be a list of str")

        col: list[str] = list(SentenceFeatureDatasetGenerator.__COLS_AND_FUNC.keys())

        if tags:
            col.extend(tags)

        # クラスのフィールドを定義
        self.__columns: Final[tuple[str, ...]] = tuple(col)
        self.__tags: Final[tuple[Tag, ...]] = tags if tags else ()

    @property
    def columns(self) -> tuple[str, ...]:
        """特徴量の列名"""
        return self.__columns

    def generate_from_sentence(self, sent: Sent1dStr, category: bool) -> NDArray:
        """文字列のリストから特徴量のリストを生成する"""

        if not type_guard.is_sent(sent):
            raise ValueError("sent must be list[str]")

        freq_by_pos: dict[str, np.float64] = SentenceCalculator.pos_frequencies(sent)

        return np.hstack(
            (
                np.array(
                    [func(sent) for func in self.__COLS_AND_FUNC.values()],
                    dtype=np.float64,
                ),
                np.array(
                    [freq_by_pos.get(tag, 0.0) for tag in self.__tags], dtype=np.float64
                ),
                np.array([category], dtype=bool),
            )
        )

    def generate_from_paragraph(
        self,
        para: Para2dStr,
        correctness: bool,
    ) -> NDArray:
        """文字列のリストのリストから特徴量のリストを生成する"""

        sent: Sent1dStr = dim_reshaper.reduce_dim(para)
        return self.generate_from_sentence(sent, correctness)


class ParagraphFeatureDatasetGenerator:
    """
    段落特徴量データセット生成クラス
    Paragraph feature dataset generator class
    """

    __COLS_AND_FUNC: Final[dict[str, Callable[[Para2dStr], np.float64 | int]]] = {
        "v1 sentences per paragraph": UnivKansasFeatures.v1_sentences_per_paragraph,
        "v2 words per paragraph": UnivKansasFeatures.v2_words_per_paragraph,
        "v3 close parenthesis present": UnivKansasFeatures.v3_close_parenthesis_present,
        "v4 dash present": UnivKansasFeatures.v4_dash_present,
        "v5 semi-colon or colon present": UnivKansasFeatures.v5_semi_colon_or_colon_present,
        "v6 question mark present": UnivKansasFeatures.v6_question_mark_present,
        "v7 apostrophe present": UnivKansasFeatures.v7_apostrophe_present,
        "v8 standard deviation of sentence length": UnivKansasFeatures.v8_standard_deviation_of_sentence_length,
        "v9 length difference for consecutive sentences": UnivKansasFeatures.v9_length_difference_for_consecutive_sentences,
        "v10 sentence with < 11 words": UnivKansasFeatures.v10_sentence_with_lt_11_words,
        "v11 sentence with > 34 words": UnivKansasFeatures.v11_sentence_with_gt_34_words,
        "v12 contains although": UnivKansasFeatures.v12_contains_although,
        "v13 contains however": UnivKansasFeatures.v13_contains_however,
        "v14 contains but": UnivKansasFeatures.v14_contains_but,
        "v15 contains because": UnivKansasFeatures.v15_contains_because,
        "v16 contains this": UnivKansasFeatures.v16_contains_this,
        "v17 contains others or researchers": UnivKansasFeatures.v17_contains_others_or_researchers,
        "v18 contains numbers": UnivKansasFeatures.v18_contains_numbers,
        "v19 contains 2 times more capitals than period": UnivKansasFeatures.v19_contains_2_times_more_capitals_than_period,
        "v20 contains et": UnivKansasFeatures.v20_contains_et,
        "word variation": ParagraphCalculator.word_variation,
        "average token length": ParagraphCalculator.average_token_length,
        "non alphabetic characters frequency": ParagraphCalculator.non_alphabetic_characters_frequency,
        "uncommon word frequency": ParagraphCalculator.uncommon_word_frequency,
    }

    def __init__(self, tags: tuple[Tag, ...] | None = None) -> None:
        if tags and not type_guard.is_tag_tuple(tags):
            raise ValueError("tags must be a list of str")

        col: list[str] = list(ParagraphFeatureDatasetGenerator.__COLS_AND_FUNC.keys())
        if tags:
            col.extend(tags)

        # クラスのフィールドを定義
        self.__columns: Final[tuple[str, ...]] = tuple(col)
        self.__tags: Final[tuple[Tag, ...]] = tags if tags else ()

    @property
    def columns(self) -> tuple[str, ...]:
        """特徴量の列名"""
        return self.__columns

    def generate_from_paragraph(
        self,
        para: Para2dStr,
        category: bool,
    ) -> NDArray:
        """文字列のリストのリストから特徴量のリストを生成する"""

        if not type_guard.is_para(para):
            raise ValueError("para must be list[list[str]]")

        freq_by_pos: dict[str, np.float64] = ParagraphCalculator.pos_frequencies(para)

        return np.hstack(
            (
                np.array(
                    [func(para) for func in self.__COLS_AND_FUNC.values()],
                    dtype=np.float64,
                ),
                np.array(
                    [freq_by_pos.get(tag, 0.0) for tag in self.__tags],
                    dtype=np.float64,
                ),
                np.array([category], dtype=bool),
            )
        )
