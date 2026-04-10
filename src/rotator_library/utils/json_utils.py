# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""High-performance JSON serialization utilities using orjson.

orjson is significantly faster than the standard json module for
serialization and deserialization of large payloads in hot paths
(API request/response transformation, streaming, message cloning).
"""

import orjson
from typing import Any

STREAM_DONE = object()
"""Sentinel marker indicating stream completion in internal pipeline."""

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
