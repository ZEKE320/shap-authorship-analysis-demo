"""プロジェクト全般の型エイリアス"""
from typing import TypeAlias

# 文章関連 型エイリアス

Tag: TypeAlias = str
"""タグ"""

TokenStr: TypeAlias = str
"""単語"""

Sent1dStr: TypeAlias = list[TokenStr]
"""文"""

Para2dStr: TypeAlias = list[Sent1dStr]
"""段落"""

TaggedToken: TypeAlias = tuple[TokenStr, Tag]
"""タグ付けされた単語"""

# 特徴量関連 型エイリアス

FeatureLabel: TypeAlias = str
"""特徴ラベル"""
