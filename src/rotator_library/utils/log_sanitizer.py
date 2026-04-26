import re
import json
from typing import Any

# Regex patterns for various API key formats
# 1. OpenAI: sk-..., followed by alphanumeric/dashes
# 2. Anthropic: sk-ant-..., followed by alphanumeric/dashes
# 3. Google: AIza...
# 4. Generic Bearer tokens and generic long strings
_API_KEY_PATTERN = re.compile(
    r'(?:sk-(?:ant-)?[a-zA-Z0-9_-]{20,}|AIza[0-9a-zA-Z-_]{35,}|Bearer\s+[a-zA-Z0-9\._-]{20,}|(?:key|api|token)-[a-zA-Z0-9_-]{20,})',
    re.IGNORECASE
)

def _sanitize_value(value: Any) -> Any:
    """Recursively sanitize dictionary/list/string values."""
    if isinstance(value, dict):
        return {k: _sanitize_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_sanitize_value(i) for i in value]
    elif isinstance(value, str):
        return _API_KEY_PATTERN.sub('[REDACTED]', value)
    return value

def sanitize_for_log(data: Any) -> Any:
    """Recursively mask API keys in dicts, lists, or strings for logging."""
    return _sanitize_value(data)
