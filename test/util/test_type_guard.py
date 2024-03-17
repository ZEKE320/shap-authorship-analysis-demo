"""タイプガード用のユーティリティモジュールテスト"""
import pytest

from authorship_tool.util import type_guard


def test_is_sent() -> None:
    """strのリストであることを確認できること"""
    assert type_guard.is_sent([""])
    assert type_guard.is_sent(["a", "b"])
    assert not type_guard.is_sent([])
    assert not type_guard.is_sent(["a", 1])
    assert not type_guard.is_sent([1, 2])
    assert not type_guard.is_sent("a")
    assert not type_guard.is_sent(1)


def test_is_pos_list() -> None:
    """posのリストであることを確認できること"""
    assert type_guard.are_tagged_tokens([("", "")])
    assert type_guard.are_tagged_tokens([("a", "b"), ("c", "d")])
    assert not type_guard.are_tagged_tokens([])
    assert not type_guard.are_tagged_tokens([("a", 1)])
    assert not type_guard.are_tagged_tokens([("a", "b", "c")])
    assert not type_guard.are_tagged_tokens("a")
    assert not type_guard.are_tagged_tokens(1)

    pytest.main(["-v"])
