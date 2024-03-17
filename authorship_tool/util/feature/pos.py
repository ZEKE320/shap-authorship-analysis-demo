"""
POSタグモジュール
POS tags module
"""

from enum import Enum, auto
from typing import Final

import nltk
import pandas as pd

from authorship_tool.types_ import Sent1dStr, Tag, TaggedToken, TokenStr
from authorship_tool.util import type_guard
from authorship_tool.util.path_util import DatasetPaths


class ExtrapositionAdjectiveState(Enum):
    """外置形容詞の状態を表す列挙型"""

    INITIAL = auto()
    FOUND_IT = auto()
    FOUND_VERB = auto()
    FOUND_ADJ = auto()
    FOUND_THAT_TO = auto()


class PosFeature:
    """POSタグと追加の特徴タグを管理するクラス"""

    __PAST_PARTICIPLE_ADJECTIVE_DATASET: set[TokenStr] = set()
    __PRESENT_PARTICIPLE_ADJECTIVE_DATASET: set[TokenStr] = set()
    __LIMIT_ADJECTIVE_DATASET: set[TokenStr] = set()
    __EXTRAPOSITION_ADJECTIVE_DATASET: set[TokenStr] = set()

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
            return (
                self.tag_jj_past_participle()
                .tag_jj_present_participle()
                .tag_jj_limit()
                .tag_jj_extraposition()
            )

        return self

    def tag_jj_past_participle(self) -> "PosFeature":
        """過去分詞形の形容詞を追加する

        Returns:
            PosFeature: PosFeatureインスタンス
        """
        return PosFeature(
            [
                (
                    (word, "JJ_pp")
                    if word.strip().lower()
                    in PosFeature.__PAST_PARTICIPLE_ADJECTIVE_DATASET
                    and pos == "JJ"
                    else (word, pos)
                )
                for (word, pos) in self.__tagged_tokens
            ]
        )

    def tag_jj_present_participle(self) -> "PosFeature":
        """
        現在分詞形の形容詞を追加する

        Returns:
            PosFeature: PosFeatureインスタンス
        """
        return PosFeature(
            [
                (
                    (word, "JJ_presp")
                    if word.strip().lower()
                    in PosFeature.__PRESENT_PARTICIPLE_ADJECTIVE_DATASET
                    and pos == "JJ"
                    else (word, pos)
                )
                for (word, pos) in self.__tagged_tokens
            ]
        )

    def tag_jj_limit(self) -> "PosFeature":
        """
        限定形容詞をタグ付けする

        Returns:
            PosFeature: PosFeatureインスタンス
        """
        return PosFeature(
            [
                (
                    (word, "JJ_lim")
                    if word.strip().lower() in PosFeature.__LIMIT_ADJECTIVE_DATASET
                    and pos == "JJ"
                    else (word, pos)
                )
                for (word, pos) in self.__tagged_tokens
            ]
        )

    def tag_jj_extraposition(self) -> "PosFeature":
        """
        外置形容詞をタグ付けする


        Returns:
            PosFeature: PosFeatureインスタンス
        """

        if all(
            word in PosFeature.__EXTRAPOSITION_ADJECTIVE_DATASET
            for word, _ in self.__tagged_tokens
        ):
            return self

        State = ExtrapositionAdjectiveState
        current = State.INITIAL
        jj_idx: int | None = None
        jj_token: TokenStr | None = None
        tagged_tokens: list[TaggedToken] = []

        for i, (token, tag) in enumerate(self.__tagged_tokens):

            if current == State.INITIAL and token.lower() == "it":
                current = State.FOUND_IT

            elif current == State.FOUND_IT and tag.startswith("V"):
                current = State.FOUND_VERB

            elif (
                current == State.FOUND_VERB
                and tag == "JJ"
                and token.lower() in self.__EXTRAPOSITION_ADJECTIVE_DATASET
            ):

                current = State.FOUND_ADJ
                jj_idx = i
                jj_token = token

            elif current == State.FOUND_ADJ and token.lower() in ["that", "to"]:
                if isinstance(jj_idx, int) and jj_token is not None:
                    jj_idx = int(jj_idx)  # Convert jj_idx to an integer
                    tagged_tokens[jj_idx] = (jj_token, "JJ_exp")
                current = State.INITIAL
                jj_idx = None
                jj_token = None

            elif tag in ["."]:
                current = State.INITIAL
                jj_idx = None
                jj_token = None

            tagged_tokens.append((token, tag))

        return PosFeature(tagged_tokens)

    @classmethod
    def initialize_additional_pos_dataset(cls) -> None:
        """
        追加のPOSタグデータセットを初期化する
        Initialize additional POS tag dataset
        """

        with open(DatasetPaths.past_participle_jj_dataset, "r", encoding="utf-8") as f:
            adjectives: set[str] = set(f.read().splitlines())
            cls.__PAST_PARTICIPLE_ADJECTIVE_DATASET = set(
                adj.strip().lower() for adj in adjectives
            )

        with open(
            DatasetPaths.present_participle_jj_dataset, "r", encoding="utf-8"
        ) as f:
            adjectives: set[str] = set(f.read().splitlines())
            cls.__PRESENT_PARTICIPLE_ADJECTIVE_DATASET = set(
                adj.strip().lower() for adj in adjectives
            )

        # 限定形容詞のデータセットを読み込む
        limit_adjectives_df = pd.read_csv(DatasetPaths.limit_jj_dataset)
        limit_adjectives = set(
            limit_adjectives_df["Non-gradable (limit)"].str.strip().str.lower()
        )
        cls.__LIMIT_ADJECTIVE_DATASET = limit_adjectives

        # 外置形容詞のデータセットを読み込む
        extraposition_adjectives_df = pd.read_csv(DatasetPaths.extraposition_jj_dataset)
        extraposition_adjectives = set(
            extraposition_adjectives_df["Extraposition Adjectives"]
            .str.strip()
            .str.lower()
        )
        cls.__EXTRAPOSITION_ADJECTIVE_DATASET = extraposition_adjectives


PosFeature.initialize_additional_pos_dataset()
