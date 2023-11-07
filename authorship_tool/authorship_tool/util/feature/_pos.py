import os
from pathlib import Path
from typing import Final

import nltk
from dotenv import load_dotenv

from authorship_tool.util import TypeGuardUtil, PathUtil

load_dotenv()


class PosFeature:
    """POSタグと追加の特徴タグを管理するクラス"""

    __PAST_PARTICIPLE_ADJECTIVE_DATASET: list[str] = []
    __POS_SUBCATEGORIES: list[str] = ["JJ_pp"]

    def __init__(self, word_list: list) -> None:
        self.__words_and_pos: list[tuple[str, str]]
        """単語とPOSタグのタプルのリスト"""

        if TypeGuardUtil.are_tokens(word_list):
            self.__words_and_pos = nltk.pos_tag(word_list)
            if self.__words_and_pos and TypeGuardUtil.are_tagged_tokens(
                self.__words_and_pos
            ):
                return

        elif TypeGuardUtil.are_sents(word_list):
            self.__words_and_pos = [
                word_and_pos
                for words_and_pos in nltk.pos_tag_sents(word_list)
                if TypeGuardUtil.are_tagged_tokens(words_and_pos)
                for word_and_pos in words_and_pos
            ]

            if self.__words_and_pos and TypeGuardUtil.are_tagged_tokens(
                self.__words_and_pos
            ):
                return

        elif TypeGuardUtil.are_paras(word_list):
            sents: list[list[str]] = [sent for para in word_list for sent in para]

            self.__words_and_pos = [
                word_and_pos
                for words_and_pos in nltk.pos_tag_sents(sents)
                if TypeGuardUtil.are_tagged_tokens(words_and_pos)
                for word_and_pos in words_and_pos
            ]

            if self.__words_and_pos and TypeGuardUtil.are_tagged_tokens(
                self.__words_and_pos
            ):
                return

        elif TypeGuardUtil.are_tagged_tokens(word_list):
            self.__words_and_pos = word_list.copy()
            return

        raise TypeError("src type is not supported.")

    def __str__(self) -> str:
        colored_text: str = ""
        for idx, (word, pos) in enumerate(self.__words_and_pos):
            if pos in self.__POS_SUBCATEGORIES:
                colored_text += f"[bold violet]{word}[/bold violet]"
            else:
                colored_text += f"{word}"

            if idx + 1 < len(self.__words_and_pos) and self.__words_and_pos[idx + 1][
                1
            ] not in [
                ".",
                ",",
                ":",
                ";",
                "!",
                "?",
                ")",
            ]:
                colored_text += " "

        return colored_text

    @property
    def words_and_pos(self) -> list[tuple[str, str]]:
        """単語とPOSタグのタプルのリスト

        Returns:
            list[tuple[str, str]]: 単語とPOSタグのタプルのリスト
        """
        return self.__words_and_pos

    @property
    def pos_set(self) -> set[str]:
        """POSタグの集合を返す

        Returns:
            set[str]: POSタグの集合
        """
        return set(pos for (_, pos) in self.__words_and_pos)

    def classify_subcategories(self) -> "PosFeature":
        """サブカテゴリを追加する

        Returns:
            PosFeature: PosFeatureインスタンス
        """
        return self.__classify_jj_subcategories()

    def __classify_jj_subcategories(self) -> "PosFeature":
        """形容詞のサブカテゴリを追加する

        Returns:
            PosFeature: PosFeatureインスタンス
        """
        if "JJ" in self.pos_set:
            return self.__classify_jj_past_participle()

        return self

    def __classify_jj_past_participle(self) -> "PosFeature":
        """過去分詞形の形容詞を追加する

        Returns:
            PosFeature: PosFeatureインスタンス
        """
        for word, pos in self.__words_and_pos:
            if pos == "JJ" and word in self.__PAST_PARTICIPLE_ADJECTIVE_DATASET:
                return PosFeature(
                    [
                        (word, "JJ_pp")
                        if word in self.__PAST_PARTICIPLE_ADJECTIVE_DATASET
                        and pos == "JJ"
                        else (word, pos)
                        for (word, pos) in self.__words_and_pos
                    ]
                )

        return self

    @classmethod
    def initialize_past_participle_adjective_dataset(cls) -> None:
        if (project_root_path := PathUtil.PROJECT_ROOT_PATH) is None:
            print(
                "Path: $PROJECT_ROOT_PATH could not be found. Skip and continue processing."
            )
            return

        dataset_path: str | None = os.getenv("path_adjective_past_participle_dataset")
        adjectives_past_participle_path: Final[Path | None] = (
            project_root_path.joinpath(dataset_path)
            if project_root_path.exists() and dataset_path
            else None
        )

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
            PosFeature.__PAST_PARTICIPLE_ADJECTIVE_DATASET = sorted(
                {adj.strip() for adj in adjectives}
            )


PosFeature.initialize_past_participle_adjective_dataset()
