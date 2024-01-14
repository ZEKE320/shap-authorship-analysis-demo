"""
正規表現モジュール
Regex module
"""

import re

NUMERIC_VALUE_PATTERN: re.Pattern[str] = re.compile(
    r"[+-]?(?:\d+\.?\d*|\.\d+)(?:(?:[eE][+-]?\d+)|(?:\*10\^[+-]?\d+))"
)
"""
数値を表す文字列の正規表現パターン
Regular expression pattern for strings representing numbers
e.g.) 1.23, 1.23e+4, 1.23*10^4, -1.23e+4
"""
