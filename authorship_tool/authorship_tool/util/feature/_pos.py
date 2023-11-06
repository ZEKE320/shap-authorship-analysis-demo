import os
from os import path
from typing import Final

import nltk
from dotenv import load_dotenv

from authorship_tool.util import TypeGuardUtil

load_dotenv()
ADJECTIVES_PAST_PARTICIPLE_PATH = ""
if dataset_path := os.getenv("path_dataset_adjective_past_participle"):
    ADJECTIVES_PAST_PARTICIPLE_PATH = path.join(
        path.dirname(path.abspath(".env")), dataset_path
    )


try:
    with open(ADJECTIVES_PAST_PARTICIPLE_PATH, "r", encoding="utf-8") as f:
        PAST_PARTICIPLE_ADJECTIVES: list[str] | None = f.read().splitlines()
except FileNotFoundError:
    print(
        f"Path: '{ADJECTIVES_PAST_PARTICIPLE_PATH}' could not be found. Skip and continue processing."
    )


class PosFeature:
    """POSタグと追加の特徴タグを管理するクラス"""

    __words_and_pos: list[tuple[str, str]]
    """単語とPOSタグのタプルのリスト"""

    def __init__(self, word_list: list[tuple[str, str]] | list[str]) -> None:
        if TypeGuardUtil.is_str_list(word_list):
            self.__words_and_pos = nltk.pos_tag(word_list)
        elif TypeGuardUtil.is_pos_list(word_list):
            self.__words_and_pos = word_list
        else:
            raise TypeError("src must be list[tuple[str, str]] or list[str]")

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
        return {pos for (_, pos) in self.__words_and_pos}

    def add_subcategory(self) -> "PosFeature":
        """サブカテゴリを追加する

        Returns:
            PosFeature: PosFeatureインスタンス
        """
        return self.add_jj_subcategory()

    def add_jj_subcategory(self) -> "PosFeature":
        """形容詞のサブカテゴリを追加する

        Returns:
            PosFeature: PosFeatureインスタンス
        """
        if "JJ" not in {pos for (_, pos) in self.__words_and_pos}:
            return self

        return self.add_jj_past_participle()

    def add_jj_past_participle(self) -> "PosFeature":
        """過去分詞形の形容詞を追加する

        Returns:
            PosFeature: PosFeatureインスタンス
        """
        return PosFeature(
            [
                (word, "JJ_pp")
                if word in self.__PAST_PARTICIPLE_ADJECTIVE_DATASET and pos == "JJ"
                else (word, pos)
                for (word, pos) in self.__words_and_pos
            ]
        )

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
