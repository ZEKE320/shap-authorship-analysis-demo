"""プロジェクト全般の型エイリアス"""
from typing import TypeAlias

# 文章関連

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

# 特徴量関連

FeatureLabel: TypeAlias = str
"""特徴ラベル"""

Char: TypeAlias = str
"""文字"""

# 環境変数

EnvKey: TypeAlias = str
"""環境変数のキー"""

PathStr: TypeAlias = str
"""パス文字列"""
