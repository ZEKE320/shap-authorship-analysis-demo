"""タイプガード用のユーティリティモジュール"""
from typing import TypeGuard

from authorship_tool.types import TwoDimStr, OneDimStr, Tag, TaggedToken, TokenStr


def is_sent(values: list, /) -> TypeGuard[OneDimStr]:
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


def is_para(values: list, /) -> TypeGuard[TwoDimStr]:
    """strのリストのリストであることを確認する

    Args:
        values (list): リスト

    Returns:
        TypeGuard[list[list[str]]]: タイプガードされたリスト
    """
    return (
        bool(values)
        and isinstance(values, list)
        and all(is_sent(cent) for cent in values)
    )


def are_paras(values: list, /) -> TypeGuard[list[TwoDimStr]]:
    """strのリストのリストのリストであることを確認する

    Args:
        values (list): リスト

    Returns:
        TypeGuard[list[list[list[str]]]]: タイプガードされたリスト
    """
    return (
        bool(values)
        and isinstance(values, list)
        and all(is_para(para) for para in values)
    )


def is_tagged_token(values: tuple, /) -> TypeGuard[TaggedToken]:
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
        and isinstance(values[0], TokenStr)
        and isinstance(values[1], Tag)
    )


def are_tagged_tokens(values: list, /) -> TypeGuard[list[TaggedToken]]:
    """posのリストであることを確認する

    Args:
        values (list): リスト

    Returns:
        TypeGuard[list[tuple[str, str]]]: タイプガードされたリスト
    """
    return (
        bool(values)
        and isinstance(values, list)
        and all(is_tagged_token(tpl) for tpl in values)
    )


def are_tagged_sents(values: list, /) -> TypeGuard[list[list[TaggedToken]]]:
    """posのリストのリストであることを確認する

    Args:
        values (list): リスト

    Returns:
        TypeGuard[list[list[tuple[str, str]]]]: タイプガードされたリスト
    """
    return (
        bool(values)
        and isinstance(values, list)
        and all(are_tagged_tokens(pos_list) for pos_list in values)
    )


def is_tag_list(values: list, /) -> TypeGuard[list[Tag]]:
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


def is_tag_set(values: set, /) -> TypeGuard[set[Tag]]:
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


def is_tag_tuple(values: tuple, /) -> TypeGuard[tuple[Tag, ...]]:
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
