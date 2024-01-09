"""
正規表現モジュール
Regex module
"""

import re

NUMERIC_VALUE_PATTERN: re.Pattern[str] = re.compile(
    r"[+-]?(?:\d+\.?\d*|\.\d+)(?:(?:[eE][+-]?\d+)|(?:\*10\^[+-]?\d+))"
)
