"""タイプガード用のユーティリティモジュール"""
from typing import TypeGuard


class TypeGuardUtil:
    """タイプガード用のユーティリティクラス"""

    @classmethod
    def is_sentence(cls, values: list, /) -> TypeGuard[list[str]]:
        """strのリストであることを確認する

        Args:
            values (list): リスト

        Returns:
            TypeGuard[list[str]]: タイプガードされたリスト
        """
        return isinstance(values, list) and all(
            isinstance(string, str) for string in values
        )

    @classmethod
    def is_paragraph(cls, values: list, /) -> TypeGuard[list[list[str]]]:
        """strのリストのリストであることを確認する

        Args:
            values (list): リスト

        Returns:
            TypeGuard[list[list[str]]]: タイプガードされたリスト
        """
        return isinstance(values, list) and all(
            cls.is_sentence(cent) for cent in values
        )

    @classmethod
    def is_pos_list(cls, values: list, /) -> TypeGuard[list[tuple[str, str]]]:
        """posのリストであることを確認する

        Args:
            values (list): リスト

        Returns:
            TypeGuard[list[tuple[str, str]]]: タイプガードされたリスト
        """
        return isinstance(values, list) and all(
            isinstance(tpl, tuple)
            and all(isinstance(s, str) for s in tpl)
            and len(tpl) == 2
            for tpl in values
        )
