"""タイプガード用のユーティリティモジュール"""
from typing import TypeGuard

from authorship_tool.type_alias import Para, Sent, Tag, Token, TaggedToken


class TypeGuardUtil:
    """タイプガード用のユーティリティクラス"""

    @classmethod
    def is_sent(cls, values: list, /) -> TypeGuard[Sent]:
        """strのリストであることを確認する

        Args:
            values (list): リスト

        Returns:
            TypeGuard[list[str]]: タイプガードされたリスト
        """
        return (
            bool(values)
            and isinstance(values, list)
            and all(isinstance(string, str) for string in values)
        )

    @classmethod
    def is_para(cls, values: list, /) -> TypeGuard[Para]:
        """strのリストのリストであることを確認する

        Args:
            values (list): リスト

        Returns:
            TypeGuard[list[list[str]]]: タイプガードされたリスト
        """
        return (
            bool(values)
            and isinstance(values, list)
            and all(cls.is_sent(cent) for cent in values)
        )

    @classmethod
    def are_paras(cls, values: list, /) -> TypeGuard[list[Para]]:
        """strのリストのリストのリストであることを確認する

        Args:
            values (list): リスト

        Returns:
            TypeGuard[list[list[list[str]]]]: タイプガードされたリスト
        """
        return (
            bool(values)
            and isinstance(values, list)
            and all(cls.is_para(para) for para in values)
        )

    @classmethod
    def is_tagged_token(cls, values: tuple, /) -> TypeGuard[TaggedToken]:
        """posのタプルであることを確認する

        Args:
            values (tuple): タプル

        Returns:
            TypeGuard[tuple[str, str]]: タイプガードされたタプル
        """
        return (
            bool(values)
            and isinstance(values, tuple)
            and len(values) == 2
            and isinstance(values[0], Token)
            and isinstance(values[1], Tag)
        )

    @classmethod
    def are_tagged_tokens(cls, values: list, /) -> TypeGuard[list[TaggedToken]]:
        """posのリストであることを確認する

        Args:
            values (list): リスト

        Returns:
            TypeGuard[list[tuple[str, str]]]: タイプガードされたリスト
        """
        return (
            bool(values)
            and isinstance(values, list)
            and all(cls.is_tagged_token(tpl) for tpl in values)
        )

    @classmethod
    def are_tagged_sents(cls, values: list, /) -> TypeGuard[list[list[TaggedToken]]]:
        """posのリストのリストであることを確認する

        Args:
            values (list): リスト

        Returns:
            TypeGuard[list[list[tuple[str, str]]]]: タイプガードされたリスト
        """
        return (
            bool(values)
            and isinstance(values, list)
            and all(cls.are_tagged_tokens(pos_list) for pos_list in values)
        )

    @classmethod
    def is_tag_list(cls, values: list, /) -> TypeGuard[list[Tag]]:
        """posのリストであることを確認する

        Args:
            values (list): リスト

        Returns:
            TypeGuard[list[str]]: タイプガードされたリスト
        """
        return (
            bool(values)
            and isinstance(values, list)
            and all(isinstance(s, Tag) for s in values)
        )

    @classmethod
    def is_tag_set(cls, values: set, /) -> TypeGuard[set[Tag]]:
        """posのセットであることを確認する

        Args:
            values (set): セット

        Returns:
            TypeGuard[set[str]]: タイプガードされたセット
        """
        return (
            bool(values)
            and isinstance(values, set)
            and all(isinstance(s, Tag) for s in values)
        )

    @classmethod
    def is_tag_tuple(cls, values: tuple, /) -> TypeGuard[tuple[Tag, ...]]:
        """posのタプルであることを確認する

        Args:
            values (tuple): タプル

        Returns:
            TypeGuard[tuple[str, ...]]: タイプガードされたタプル
        """
        return (
            bool(values)
            and isinstance(values, tuple)
            and all(isinstance(s, Tag) for s in values)
        )
