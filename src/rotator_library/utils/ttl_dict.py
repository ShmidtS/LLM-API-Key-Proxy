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

import asyncio
import threading
import time
from collections import OrderedDict
from typing import Any, Optional


class TTLDict:
    """
    OrderedDict with per-entry TTL and max-size LRU eviction.

    Dual-lock design for safe use from both sync and async contexts:

    - **Sync callers** use ``set``, ``get``, ``pop``, ``keys``, etc.
      These acquire ``threading.Lock`` and never block the event loop
      when called from non-async code.
    - **Async callers** use ``aset``, ``aget``, ``apop``, ``akeys``, etc.
      These acquire ``asyncio.Lock`` and yield to the event loop
      while waiting, preventing event-loop stalls.

    The two lock domains are independent: sync methods do NOT try to
    detect or acquire the async lock, and vice versa.  Callers must
    pick the correct domain for their context.  Mixing sync and async
    access to the same instance from concurrent threads/tasks is safe
    for individual operations but does NOT guarantee cross-domain
    atomicity (e.g. an async cleanup and a sync set may interleave).

    - O(1) get/set/delete via OrderedDict
    - Automatic expiry on access (lazy eviction)
    - Max-size LRU eviction on set
    - Thread-safe (sync) and async-safe (async) locking
    - Supports ``in``, ``get``, ``pop``, ``keys``, ``values``, ``items``
      and their async counterparts with ``a`` prefix

    Args:
        maxsize: Maximum entries before LRU eviction (0 = unlimited)
        default_ttl: Default time-to-live in seconds for entries
    """

    __slots__ = ("_data", "_maxsize", "_default_ttl", "_lock", "_async_lock")

    def __init__(self, maxsize: int = 0, default_ttl: float = 3600.0):
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._maxsize = maxsize
        self._default_ttl = default_ttl
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()

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

    # ---- Async counterparts (use asyncio.Lock) ----

    async def aset(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Async store value with given TTL (uses default_ttl if None)."""
        deadline = time.time() + (ttl if ttl is not None else self._default_ttl)
        async with self._async_lock:
            self._data[key] = (deadline, value)
            self._data.move_to_end(key)
            self._enforce_maxsize()

    async def aget(self, key: str, default: Any = None) -> Any:
        """Async get value if not expired, else return default."""
        async with self._async_lock:
            self._evict_expired_key(key)
            entry = self._data.get(key)
            if entry is not None and self._is_alive(entry):
                self._data.move_to_end(key)
                return entry[1]
            return default

    async def acontains(self, key: str) -> bool:
        """Async check if key exists and is not expired."""
        async with self._async_lock:
            self._evict_expired_key(key)
            return key in self._data and self._is_alive(self._data[key])

    async def apop(self, key: str, *default: Any) -> Any:
        """Async remove and return value, or default if missing/expired."""
        async with self._async_lock:
            self._evict_expired_key(key)
            if key in self._data:
                _, value = self._data.pop(key)
                return value
            if default:
                return default[0]
            raise KeyError(key)

    async def akeys(self) -> list[str]:
        """Async list of non-expired keys."""
        async with self._async_lock:
            return [k for k, v in self._data.items() if self._is_alive(v)]

    async def avalues(self) -> list[Any]:
        """Async list of non-expired values."""
        async with self._async_lock:
            return [v[1] for v in self._data.values() if self._is_alive(v)]

    async def aitems(self) -> list[tuple[str, Any]]:
        """Async list of non-expired (key, value) pairs."""
        async with self._async_lock:
            return [(k, v[1]) for k, v in self._data.items() if self._is_alive(v)]

    async def aclear(self) -> None:
        """Async remove all entries."""
        async with self._async_lock:
            self._data.clear()

    async def acleanup(self) -> int:
        """Async remove all expired entries. Returns count removed."""
        async with self._async_lock:
            expired = [k for k, v in self._data.items() if not self._is_alive(v)]
            for k in expired:
                del self._data[k]
            return len(expired)
