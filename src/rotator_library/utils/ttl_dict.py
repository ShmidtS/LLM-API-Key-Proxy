# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# src/rotator_library/utils/ttl_dict.py
"""
Thread-safe dictionary with per-entry TTL and optional max-size eviction.

Replaces unbounded Dict[str, Any] caches with a bounded, auto-expiring
alternative that prevents memory leaks from long-running processes.

Usage:
    cache = TTLDict(maxsize=500, default_ttl=3600)
    cache["key"] = value         # stores with default TTL
    cache.set("key", value, ttl=120)  # stores with custom TTL
    cache.get("key")             # returns None if expired
    "key" in cache               # False if expired
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Optional


class TTLDict:
    """
    OrderedDict with per-entry TTL and max-size LRU eviction.

    - O(1) get/set/delete via OrderedDict
    - Automatic expiry on access (lazy eviction)
    - Max-size LRU eviction on set
    - Thread-safe with optional locking
    - Supports ``in``, ``get``, ``pop``, ``keys``, ``values``, ``items``

    Args:
        maxsize: Maximum entries before LRU eviction (0 = unlimited)
        default_ttl: Default time-to-live in seconds for entries
    """

    __slots__ = ("_data", "_maxsize", "_default_ttl", "_lock")

    def __init__(self, maxsize: int = 0, default_ttl: float = 3600.0):
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._maxsize = maxsize
        self._default_ttl = default_ttl
        self._lock = threading.Lock()

    def _is_alive(self, entry: tuple[float, Any]) -> bool:
        return time.time() <= entry[0]

    def _evict_expired_key(self, key: str) -> None:
        """Remove key if expired. Call while holding lock."""
        entry = self._data.get(key)
        if entry is not None and not self._is_alive(entry):
            del self._data[key]

    def _enforce_maxsize(self) -> None:
        """LRU-evict oldest entries until under maxsize. Call while holding lock."""
        if self._maxsize > 0:
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    # ---- Core dict interface ----

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value, ttl=self._default_ttl)

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Store value with given TTL (uses default_ttl if None)."""
        deadline = time.time() + (ttl if ttl is not None else self._default_ttl)
        with self._lock:
            self._data[key] = (deadline, value)
            self._data.move_to_end(key)
            self._enforce_maxsize()

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            self._evict_expired_key(key)
            entry = self._data[key]  # raises KeyError if missing/expired
            if not self._is_alive(entry):
                del self._data[key]
                raise KeyError(key)
            self._data.move_to_end(key)
            return entry[1]

    def __contains__(self, key: str) -> bool:
        with self._lock:
            self._evict_expired_key(key)
            return key in self._data and self._is_alive(self._data[key])

    def __delitem__(self, key: str) -> None:
        with self._lock:
            del self._data[key]

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __bool__(self) -> bool:
        return len(self) > 0

    def get(self, key: str, default: Any = None) -> Any:
        """Get value if not expired, else return default."""
        with self._lock:
            self._evict_expired_key(key)
            entry = self._data.get(key)
            if entry is not None and self._is_alive(entry):
                self._data.move_to_end(key)
                return entry[1]
            return default

    def pop(self, key: str, *default: Any) -> Any:
        """Remove and return value, or default if missing/expired."""
        with self._lock:
            self._evict_expired_key(key)
            if key in self._data:
                _, value = self._data.pop(key)
                return value
            if default:
                return default[0]
            raise KeyError(key)

    def keys(self) -> list[str]:
        with self._lock:
            return [k for k, v in self._data.items() if self._is_alive(v)]

    def values(self) -> list[Any]:
        with self._lock:
            return [v[1] for v in self._data.values() if self._is_alive(v)]

    def items(self) -> list[tuple[str, Any]]:
        with self._lock:
            return [(k, v[1]) for k, v in self._data.items() if self._is_alive(v)]

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def cleanup(self) -> int:
        """Remove all expired entries. Returns count removed."""
        with self._lock:
            expired = [k for k, v in self._data.items() if not self._is_alive(v)]
            for k in expired:
                del self._data[k]
            return len(expired)
