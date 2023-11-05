"""タイプガード用のユーティリティモジュールテスト"""
import nltk
import pytest

from authorship_tool.util import TypeGuardUtil


def test_is_str_list() -> None:
    """strのリストであることを確認できること"""
    assert TypeGuardUtil.is_str_list([])
    assert TypeGuardUtil.is_str_list([""])
    assert TypeGuardUtil.is_str_list(["a", "b"])
    assert not TypeGuardUtil.is_str_list(["a", 1])
    assert not TypeGuardUtil.is_str_list([1, 2])
    assert not TypeGuardUtil.is_str_list("a")  # type: ignore
    assert not TypeGuardUtil.is_str_list(1)  # type: ignore


def test_is_pos_list() -> None:
    """posのリストであることを確認できること"""
    assert TypeGuardUtil.is_pos_list([])
    assert TypeGuardUtil.is_pos_list([("", "")])
    assert TypeGuardUtil.is_pos_list([("a", "b"), ("c", "d")])
    assert not TypeGuardUtil.is_pos_list([("a", 1)])
    assert not TypeGuardUtil.is_pos_list([("a", "b", "c")])
    assert not TypeGuardUtil.is_pos_list("a")  # type: ignore
    assert not TypeGuardUtil.is_pos_list(1)  # type: ignore

    assert TypeGuardUtil.is_pos_list(nltk.pos_tag(["She", "looks", "busy", "."]))


if __name__ == "__main__":
    pytest.main(["-v"])
