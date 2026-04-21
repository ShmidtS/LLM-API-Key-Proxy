# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""High-performance JSON serialization utilities using orjson.

orjson is significantly faster than the standard json module for
serialization and deserialization of large payloads in hot paths
(API request/response transformation, streaming, message cloning).
"""

import json
import orjson
import re
from typing import Any, Dict, Optional

JSONDecodeError = getattr(orjson, "JSONDecodeError", json.JSONDecodeError)

STREAM_DONE = object()
"""Sentinel marker indicating stream completion in internal pipeline."""

# Pre-compiled pattern to extract the first JSON object from a string
_JSON_OBJECT_RE = re.compile(r"(\{.*\})", re.DOTALL)


def extract_json_object(text: str) -> Optional[str]:
    """Extract the first JSON object from arbitrary text.

    Returns the matched JSON string (stripped) if found, otherwise None.
    Uses a pre-compiled regex for performance.
    """
    if not text:
        return None
    match = _JSON_OBJECT_RE.search(text)
    if match:
        return match.group(1)
    return None

def json_dumps(obj: Any) -> bytes:
    """Fast JSON serialization using orjson. Returns UTF-8 encoded bytes."""
    return orjson.dumps(obj)


def json_dumps_str(obj: Any) -> str:
    """Fast JSON serialization returning a UTF-8 string."""
    return orjson.dumps(obj).decode("utf-8")


def json_loads(s: str) -> Any:
    """Fast JSON deserialization using orjson."""
    return orjson.loads(s)


def sse_data_event(data: Any) -> bytes:
    """Format a dict as an SSE data event bytes.

    Produces ``b"data: {...}\\n\\n"`` suitable for yielding in SSE streaming responses.
    """
    return b"data: " + orjson.dumps(data) + b"\n\n"


def json_deep_copy(obj: Any) -> Any:
    """Deep copy for JSON-serializable data via orjson round-trip.

    2-3x faster than copy.deepcopy for dicts/lists of JSON primitives.
    Only safe for JSON-serializable objects (no tuples, sets, bytes, custom types).
    Tuples become lists; non-JSON types raise orjson.JSONEncodeError.
    """
    return orjson.loads(orjson.dumps(obj))


def extract_reasoning(data: Dict[str, Any]) -> Optional[str]:
    """Extract reasoning or reasoning_content from an OpenAI-compatible response dict.

    Checks top-level ``reasoning``, then ``choices[0].message.reasoning``,
    then ``choices[0].message.reasoning_content``.
    """
    if not isinstance(data, dict):
        return None
    if "reasoning" in data:
        return data["reasoning"]
    if "choices" in data and data["choices"]:
        message = data["choices"][0].get("message", {})
        if "reasoning" in message:
            return message["reasoning"]
        if "reasoning_content" in message:
            return message["reasoning_content"]
    return None
