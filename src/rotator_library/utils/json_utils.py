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

# Pre-compiled pattern to extract the first JSON object from a string.
# Greedy — used as a fast path only; falls back to brace-counting parser on failure.
_JSON_OBJECT_RE = re.compile(r"(\{.*\})", re.DOTALL)


def _extract_balanced_json(text: str) -> Optional[str]:
    """Extract the first balanced JSON object by counting brace depth.

    Scans character-by-character, skipping braces inside string literals
    (handles escaped quotes within strings). Returns the substring from the
    first ``{`` to the matching ``}`` where depth returns to 0, or None if
    no balanced object is found.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1  # skip escaped character
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        i += 1
    return None


def extract_json_object(text: str) -> Optional[str]:
    """Extract the first JSON object from arbitrary text.

    Returns the matched JSON string (stripped) if found, otherwise None.
    Tries the pre-compiled regex first (fast path), then falls back to a
    brace-counting parser for cases the regex cannot handle correctly.
    """
    if not text:
        return None
    match = _JSON_OBJECT_RE.search(text)
    if match:
        candidate = match.group(1)
        try:
            orjson.loads(candidate)
            return candidate
        except (JSONDecodeError, ValueError):
            pass  # fall through to balanced parser
    return _extract_balanced_json(text)

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

    Intentional: callers are in logging/audit paths, not per-chunk streaming.
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
