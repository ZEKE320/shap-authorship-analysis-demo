"""
POSタグモジュール
POS tags module
"""

from pathlib import Path
from typing import Final

import nltk

from authorship_tool.types_ import Sent1dStr, Tag, TaggedToken, TokenStr
from authorship_tool.util import type_guard
from authorship_tool.util.path_util import DatasetPaths


class PosFeature:
    """POSタグと追加の特徴タグを管理するクラス"""

    __PAST_PARTICIPLE_ADJECTIVE_DATASET: set[TokenStr] = set()
    __POS_SUBCATEGORIES: Final[set[Tag]] = set(["JJ_pp"])

    def __init__(self, words: list) -> None:
        tagged_tokens: list[TaggedToken] = []
        """単語とPOSタグのタプルのリスト"""

        if type_guard.is_sent(words):
            tagged_tokens = nltk.pos_tag(words)

        elif type_guard.is_para(words):
            tagged_tokens = [
                word_and_pos
                for words_and_pos in nltk.pos_tag_sents(words)
                for word_and_pos in words_and_pos
            ]

        elif type_guard.are_paras(words):
            sents: list[Sent1dStr] = [sent for para in words for sent in para]

            tagged_tokens = [
                word_and_pos
                for words_and_pos in nltk.pos_tag_sents(sents)
                for word_and_pos in words_and_pos
            ]

        elif type_guard.are_tagged_tokens(words):
            tagged_tokens = words.copy()

        if len(tagged_tokens) == 0 or not type_guard.are_tagged_tokens(tagged_tokens):
            raise TypeError("src type is not supported.")

        self.__tagged_tokens: Final[list[TaggedToken]] = tagged_tokens

    def __str__(self) -> str:
        return " ".join([tagged_token[0] for tagged_token in self.__tagged_tokens])

    @property
    def tagged_tokens(self) -> list[TaggedToken]:
        """単語とPOSタグのタプルのリスト

        Returns:
            list[tuple[TokenStr, Tag]]: 単語とPOSタグのタプルのリスト
        """
        return self.__tagged_tokens

    @property
    def all_pos(self) -> list[Tag]:
        """POSタグのリストを返す

        Returns:
            list[str]: POSタグのリスト
        """
        return sorted({pos for (_, pos) in self.__tagged_tokens})

    def tag_subcategories(self) -> "PosFeature":
        """サブカテゴリを追加する

        Returns:
            PosFeature: PosFeatureインスタンス
        """
        return self.__tag_jj_subcategories()

    def __tag_jj_subcategories(self) -> "PosFeature":
        """形容詞のサブカテゴリを追加する

        Returns:
            PosFeature: PosFeatureインスタンス
        """
        if "JJ" in self.all_pos:
            return self.__tag_jj_past_participle()

        return self

    def __tag_jj_past_participle(self) -> "PosFeature":
        """過去分詞形の形容詞を追加する

        Returns:
            PosFeature: PosFeatureインスタンス
        """
        return PosFeature(
            [
                (word, "JJ_pp")
                if word.strip().lower()
                in PosFeature.__PAST_PARTICIPLE_ADJECTIVE_DATASET
                and pos == "JJ"
                else (word, pos)
                for (word, pos) in self.__tagged_tokens
            ]
        )

    @classmethod
    def initialize_dataset_past_participle_adjective(cls) -> None:
        """過去分詞形の形容詞のデータセットのパスを指定する"""
        adjectives_past_participle_path: Final[
            Path | None
        ] = DatasetPaths.past_participle_jj_dataset

        if (
            adjectives_past_participle_path is None
            or not adjectives_past_participle_path.exists()
        ):
            print(
                f"File: '{adjectives_past_participle_path}' could not be found. Skip and continue processing."
            )
            return

        with open(adjectives_past_participle_path, "r", encoding="utf-8") as f:
            adjectives: set[str] = set(f.read().splitlines())
            cls.__PAST_PARTICIPLE_ADJECTIVE_DATASET = set(
                adj.strip().lower() for adj in adjectives
            )


PosFeature.initialize_dataset_past_participle_adjective()
