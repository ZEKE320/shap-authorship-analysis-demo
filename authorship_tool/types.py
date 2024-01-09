"""プロジェクト全般の型エイリアス"""
from typing import TypeAlias

# 文章関連の型エイリアス

Tag: TypeAlias = str
"""タグ"""
TokenStr: TypeAlias = str
"""単語"""
OneDimStr: TypeAlias = list[TokenStr]
"""文"""
TwoDimStr: TypeAlias = list[OneDimStr]
"""段落"""
TaggedToken: TypeAlias = tuple[TokenStr, Tag]
"""タグ付けされた単語"""
