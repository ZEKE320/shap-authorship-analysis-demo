"""プロジェクト全般の型エイリアス"""
from typing import TypeAlias

# 文章関連の型エイリアス

Tag: TypeAlias = str
"""タグ"""
TokenStr: TypeAlias = str
"""単語"""
Sent1dStr: TypeAlias = list[TokenStr]
"""文"""
Para2dStr: TypeAlias = list[Sent1dStr]
"""段落"""
TaggedTokens: TypeAlias = tuple[TokenStr, Tag]
"""タグ付けされた単語"""
