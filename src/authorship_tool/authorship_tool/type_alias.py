"""プロジェクト全般の型エイリアス"""
from typing import TypeAlias

# 文章関連の型エイリアス

Tag: TypeAlias = str
"""タグ"""
Token: TypeAlias = str
"""単語"""
Sent: TypeAlias = list[Token]
"""文"""
Para: TypeAlias = list[Sent]
"""段落"""
Book: TypeAlias = list[Sent]
"""本"""
TaggedToken: TypeAlias = tuple[Token, Tag]
"""タグ付けされた単語"""
