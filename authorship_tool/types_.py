"""プロジェクト全般の型エイリアス"""
from typing import TypeAlias

# 文章関連

Tag: TypeAlias = str
"""タグ (Tag)"""

TokenStr: TypeAlias = str
"""単語 (Token)"""

Sent1dStr: TypeAlias = list[TokenStr]
"""文 (Sentence)"""

Para2dStr: TypeAlias = list[Sent1dStr]
"""段落 (Paragraph)"""

Docs3dStr: TypeAlias = list[Para2dStr]
"""文書 (Document)"""

AuthorCollection4dStr: TypeAlias = list[Docs3dStr]
"""1人の著者に関連する文書の集合 (Author's document collection)"""

TaggedToken: TypeAlias = tuple[TokenStr, Tag]
"""タグ付けされた単語 (Tagged tokens)"""

# 特徴量関連

FeatureLabel: TypeAlias = str
"""特徴ラベル (Feature label)"""

Char: TypeAlias = str
"""文字 (Character)"""

# 環境変数

EnvKey: TypeAlias = str
"""環境変数のキー (Environment variable key)"""

PathStr: TypeAlias = str
"""パス文字列 (Path string)"""
