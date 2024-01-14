"""
POSタグモジュール
POS tags module
"""

from pathlib import Path
from typing import Final

import nltk

from authorship_tool.types_ import Sent1dStr, Tag, TaggedToken, TokenStr
from authorship_tool.util import type_guard
from authorship_tool.util.path_util import PATHS


class PosFeature:
    """POSタグと追加の特徴タグを管理するクラス"""

    __PAST_PARTICIPLE_ADJECTIVE_DATASET: set[TokenStr] = set()
    __POS_SUBCATEGORIES: Final[set[Tag]] = set(["JJ_pp"])

    def __init__(self, word_list: list) -> None:
        tagged_tokens: list[TaggedToken] = []
        """単語とPOSタグのタプルのリスト"""

        if type_guard.is_sent(word_list):
            tagged_tokens = nltk.pos_tag(word_list)

        elif type_guard.is_para(word_list):
            tagged_tokens = [
                word_and_pos
                for words_and_pos in nltk.pos_tag_sents(word_list)
                for word_and_pos in words_and_pos
            ]

        elif type_guard.are_paras(word_list):
            sents: list[Sent1dStr] = [sent for para in word_list for sent in para]

            tagged_tokens = [
                word_and_pos
                for words_and_pos in nltk.pos_tag_sents(sents)
                for word_and_pos in words_and_pos
            ]

        elif type_guard.are_tagged_tokens(word_list):
            tagged_tokens = word_list.copy()

        if len(tagged_tokens) == 0 or not type_guard.are_tagged_tokens(tagged_tokens):
            raise TypeError("src type is not supported.")

        self.__tagged_tokens: Final[list[TaggedToken]] = tagged_tokens

    def __str__(self) -> str:
        for idx, (word, pos) in enumerate(self.__tagged_tokens):
            if pos in PosFeature.__POS_SUBCATEGORIES:
                self.__tagged_tokens[idx] = (
                    f"[bold #749BC2 reverse]{word}[/bold #749BC2 reverse]",
                    pos,
                )

        return " ".join([tagged_token[0] for tagged_token in self.__tagged_tokens])

    @property
    def tagged_tokens(self) -> list[TaggedToken]:
        """単語とPOSタグのタプルのリスト

        Returns:
            list[tuple[str, str]]: 単語とPOSタグのタプルのリスト
        """
        return self.__tagged_tokens

    @property
    def all_pos(self) -> tuple[Tag, ...]:
        """POSタグのタプルを返す

        Returns:
            tuple[str]: POSタグのタプル
        """
        return tuple(sorted({pos for (_, pos) in self.__tagged_tokens}))

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
        for word, pos in self.__tagged_tokens:
            if pos == "JJ" and word in PosFeature.__PAST_PARTICIPLE_ADJECTIVE_DATASET:
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

        return self

    @classmethod
    def initialize_dataset_past_participle_adjective(cls) -> None:
        """過去分詞形の形容詞のデータセットのパスを指定する"""
        adjectives_past_participle_path: Final[Path | None] = PATHS[
            "past_participle_jj_dataset"
        ]

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
