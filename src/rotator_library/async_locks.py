# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# src/rotator_library/async_locks.py
"""
Optimized async locking primitives for high-throughput scenarios.

Provides alternatives to asyncio.Lock with better performance
for specific access patterns:
- ReadWriteLock: Multiple readers, single writer
"""

import asyncio
from contextlib import asynccontextmanager


class ReadWriteLock:
    """
    A read-write lock that allows multiple concurrent readers
    but exclusive access for writers.

    This is more efficient than a standard Lock when:
    - Read operations are frequent
    - Read operations are fast
    - Write operations are infrequent

    Usage:
    lock = ReadWriteLock()

    # Read lock (multiple can hold simultaneously)
    async with lock.read():
        data = shared_resource.read()

    # Write lock (exclusive)
    async with lock.write():
        shared_resource.write(new_value)
    """

    def __init__(self):
        self._readers = 0
        self._writer_waiting = 0
        self._writer_active = False
        self._condition = asyncio.Condition()

    async def acquire_read(self) -> None:
        """Acquire read lock."""
        async with self._condition:
            while self._writer_active or self._writer_waiting > 0:
                await self._condition.wait()
            self._readers += 1

    async def release_read(self) -> None:
        """Release read lock."""
        async with self._condition:
            self._readers -= 1
            if self._readers == 0:
                self._condition.notify_all()

    async def acquire_write(self) -> None:
        """Acquire write lock."""
        async with self._condition:
            self._writer_waiting += 1
            try:
                while self._readers > 0 or self._writer_active:
                    await self._condition.wait()
                self._writer_active = True
            finally:
                self._writer_waiting -= 1

    async def release_write(self) -> None:
        """Release write lock."""
        async with self._condition:
            self._writer_active = False
            self._condition.notify_all()

    @asynccontextmanager
    async def read(self):
        """Context manager for read lock."""
        await self.acquire_read()
        try:
            yield
        finally:
            await self.release_read()

    @asynccontextmanager
    async def write(self):
        """Context manager for write lock."""
        await self.acquire_write()
        try:
            yield
        finally:
            await self.release_write()
