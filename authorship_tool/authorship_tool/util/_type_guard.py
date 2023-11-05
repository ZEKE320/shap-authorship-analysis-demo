"""タイプガード用のユーティリティモジュール"""
from typing import TypeGuard


class TypeGuardUtil:
    """タイプガード用のユーティリティクラス"""

    @classmethod
    def is_str_list(cls, values: list, /) -> TypeGuard[list[str]]:
        """strのリストであることを確認する

        Args:
            values (list): リスト

        Returns:
            TypeGuard[list[str]]: タイプガードされたリスト
        """
        return isinstance(values, list) and all(isinstance(s, str) for s in values)

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
