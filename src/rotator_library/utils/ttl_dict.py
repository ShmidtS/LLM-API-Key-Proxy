# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# src/rotator_library/utils/ttl_dict.py
# pylint: disable=R0903

import asyncio
import threading
import time
from collections import OrderedDict
from typing import Any, Optional


class TTLDict:
    __slots__ = (
        '_data',
        '_maxsize',
        '_default_ttl',
        '_lock',
        '_async_lock',
        '_cleanup_task',
        '_cleanup_interval',
    )

    def __init__(self, maxsize: int = 0, default_ttl: float = 3600.0, cleanup_interval: float = 60.0):
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._maxsize = maxsize
        self._default_ttl = default_ttl
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task[None]] = None
        self._cleanup_interval = cleanup_interval

    def _is_alive(self, entry: tuple[float, Any]) -> bool:
        return time.time() <= entry[0]

    def _evict_expired_key(self, key: str) -> None:
        entry = self._data.get(key)
        if entry is not None and not self._is_alive(entry):
            del self._data[key]

    def _enforce_maxsize(self) -> None:
        if self._maxsize > 0:
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value, ttl=self._default_ttl)

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        deadline = time.time() + (ttl if ttl is not None else self._default_ttl)
        with self._lock:
            self._data[key] = (deadline, value)
            self._data.move_to_end(key)
            self._enforce_maxsize()

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            self._evict_expired_key(key)
            entry = self._data[key]
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
        with self._lock:
            self._evict_expired_key(key)
            entry = self._data.get(key)
            if entry is not None and self._is_alive(entry):
                self._data.move_to_end(key)
                return entry[1]
            return default

    def pop(self, key: str, *default: Any) -> Any:
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
        with self._lock:
            expired = [k for k, v in self._data.items() if not self._is_alive(v)]
            for k in expired:
                del self._data[k]
            return len(expired)

    async def aset(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        self._start_cleanup_task()
        deadline = time.time() + (ttl if ttl is not None else self._default_ttl)
        async with self._async_lock:
            self._data[key] = (deadline, value)
            self._data.move_to_end(key)
            self._enforce_maxsize()

    async def aget(self, key: str, default: Any = None) -> Any:
        self._start_cleanup_task()
        async with self._async_lock:
            self._evict_expired_key(key)
            entry = self._data.get(key)
            if entry is not None and self._is_alive(entry):
                self._data.move_to_end(key)
                return entry[1]
            return default

    async def acontains(self, key: str) -> bool:
        self._start_cleanup_task()
        async with self._async_lock:
            self._evict_expired_key(key)
            return key in self._data and self._is_alive(self._data[key])

    async def apop(self, key: str, *default: Any) -> Any:
        self._start_cleanup_task()
        async with self._async_lock:
            self._evict_expired_key(key)
            if key in self._data:
                _, value = self._data.pop(key)
                return value
            if default:
                return default[0]
            raise KeyError(key)

    async def akeys(self) -> list[str]:
        async with self._async_lock:
            return [k for k, v in self._data.items() if self._is_alive(v)]

    async def avalues(self) -> list[Any]:
        async with self._async_lock:
            return [v[1] for v in self._data.values() if self._is_alive(v)]

    async def aitems(self) -> list[tuple[str, Any]]:
        async with self._async_lock:
            return [(k, v[1]) for k, v in self._data.items() if self._is_alive(v)]

    async def aclear(self) -> None:
        async with self._async_lock:
            self._data.clear()

    async def acleanup(self) -> int:
        async with self._async_lock:
            expired = [k for k, v in self._data.items() if not self._is_alive(v)]
            for k in expired:
                del self._data[k]
            return len(expired)

    def _start_cleanup_task(self) -> None:
        if self._cleanup_task is not None and not self._cleanup_task.done():
            return
        try:
            asyncio.get_running_loop()
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        except RuntimeError:
            pass

    async def _cleanup_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self.acleanup()
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    def close(self) -> None:
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            self._cleanup_task = None