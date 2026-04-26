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
import logging
from contextlib import asynccontextmanager
from typing import Optional

logger = logging.getLogger(__name__)

ASYNC_LOCK_TIMEOUT = 30.0

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

    _MAX_READER_BATCH = 8

    def __init__(self):
        self._readers = 0
        self._writer_waiting = 0
        self._writer_active = False
        self._reader_epoch = 0
        self._condition = asyncio.Condition()

    @property
    def locked(self):
        return self._writer_active or self._readers > 0

    async def acquire_read(self, timeout: Optional[float] = None) -> None:
        """Acquire read lock with optional timeout."""
        try:
            await asyncio.wait_for(self._acquire_read_logic(), timeout=timeout or ASYNC_LOCK_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("Read lock acquisition timed out")
            raise

    async def _acquire_read_logic(self) -> None:
        async with self._condition:
            while self._writer_active or (self._writer_waiting > 0 and self._reader_epoch >= self._MAX_READER_BATCH):
                await self._condition.wait()
            self._readers += 1
            if self._writer_waiting > 0:
                self._reader_epoch += 1

    async def release_read(self) -> None:
        """Release read lock."""
        async with self._condition:
            self._readers -= 1
            if self._readers == 0 and self._writer_waiting > 0:
                self._condition.notify_all()

    async def acquire_write(self, timeout: Optional[float] = None) -> None:
        """Acquire write lock with optional timeout."""
        try:
            await asyncio.wait_for(self._acquire_write_logic(), timeout=timeout or ASYNC_LOCK_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("Write lock acquisition timed out")
            raise

    async def _acquire_write_logic(self) -> None:
        async with self._condition:
            self._writer_waiting += 1
            try:
                while self._readers > 0 or self._writer_active:
                    await self._condition.wait()
                self._writer_active = True
                self._reader_epoch = 0
            finally:
                self._writer_waiting -= 1

    async def release_write(self) -> None:
        """Release write lock."""
        async with self._condition:
            self._writer_active = False
            self._condition.notify_all()

    @asynccontextmanager
    async def read(self, timeout: Optional[float] = None):
        """Context manager for read lock."""
        await self.acquire_read(timeout=timeout)
        try:
            yield
        finally:
            await self.release_read()

    @asynccontextmanager
    async def write(self, timeout: Optional[float] = None):
        """Context manager for write lock."""
        await self.acquire_write(timeout=timeout)
        try:
            yield
        finally:
            await self.release_write()
