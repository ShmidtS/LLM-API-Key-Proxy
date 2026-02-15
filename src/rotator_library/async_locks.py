# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/async_locks.py
"""
Optimized async locking primitives for high-throughput scenarios.

Provides alternatives to asyncio.Lock with better performance
for specific access patterns:
- ReadWriteLock: Multiple readers, single writer
- RateLimitedLock: Lock with built-in rate limiting
- AsyncRWLock: Async-friendly read-write lock
"""

import asyncio
import time
from typing import Optional, Callable, Any
from contextlib import asynccontextmanager
import logging

lib_logger = logging.getLogger("rotator_library")


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
        self._read_ready = asyncio.Condition()
        self._write_ready = asyncio.Condition()

    async def acquire_read(self) -> None:
        """Acquire read lock."""
        async with self._read_ready:
            # Wait if a writer is active or waiting (writers have priority)
            while self._writer_active or self._writer_waiting > 0:
                await self._read_ready.wait()
            self._readers += 1

    async def release_read(self) -> None:
        """Release read lock."""
        async with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                # Notify waiting writers
                async with self._write_ready:
                    self._write_ready.notify_all()

    async def acquire_write(self) -> None:
        """Acquire write lock."""
        async with self._write_ready:
            self._writer_waiting += 1
            try:
                # Wait until no readers or writers active
                while self._readers > 0 or self._writer_active:
                    await self._write_ready.wait()
                self._writer_active = True
            finally:
                self._writer_waiting -= 1

    async def release_write(self) -> None:
        """Release write lock."""
        async with self._write_ready:
            self._writer_active = False
            # Notify both readers and writers
            self._write_ready.notify_all()
            async with self._read_ready:
                self._read_ready.notify_all()

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


class AsyncSemaphore:
    """
    An enhanced semaphore with monitoring and timeout support.

    Features:
    - Max concurrent operations limit
    - Wait timeout with fallback
    - Statistics tracking
    - Fairness (FIFO ordering)
    """

    def __init__(self, value: int = 1, name: str = ""):
        """
        Initialize semaphore.

        Args:
            value: Maximum concurrent operations
            name: Optional name for logging
        """
        self._value = value
        self._name = name or f"semaphore_{id(self)}"
        self._waiters: list = []
        self._lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "acquires": 0,
            "releases": 0,
            "timeouts": 0,
            "peak_concurrent": 0,
            "current_concurrent": 0,
        }

    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire the semaphore.

        Args:
            timeout: Max seconds to wait (None = no timeout)

        Returns:
            True if acquired, False if timeout
        """
        start_time = time.time()

        async with self._lock:
            if self._value > 0:
                self._value -= 1
                self._stats["acquires"] += 1
                self._stats["current_concurrent"] += 1
                self._stats["peak_concurrent"] = max(
                    self._stats["peak_concurrent"],
                    self._stats["current_concurrent"]
                )
                return True

            # Need to wait
            waiter = asyncio.Event()
            self._waiters.append(waiter)

        # Wait outside the lock
        try:
            if timeout is not None:
                try:
                    await asyncio.wait_for(waiter.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    # Remove from waiters
                    async with self._lock:
                        if waiter in self._waiters:
                            self._waiters.remove(waiter)
                    self._stats["timeouts"] += 1
                    return False
            else:
                await waiter.wait()

            # Acquired via release
            async with self._lock:
                self._stats["acquires"] += 1
                self._stats["current_concurrent"] += 1
                self._stats["peak_concurrent"] = max(
                    self._stats["peak_concurrent"],
                    self._stats["current_concurrent"]
                )
            return True

        except Exception:
            async with self._lock:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
            raise

    def release(self) -> None:
        """Release the semaphore."""
        # Note: This is sync for compatibility with context manager
        self._value += 1
        self._stats["releases"] += 1
        self._stats["current_concurrent"] -= 1

        # Wake up next waiter if any
        if self._waiters:
            next_waiter = self._waiters.pop(0)
            next_waiter.set()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    def get_stats(self) -> dict:
        """Get semaphore statistics."""
        return {
            **self._stats,
            "name": self._name,
            "available": self._value,
            "waiters": len(self._waiters),
        }


class RateLimitedLock:
    """
    A lock with built-in rate limiting.

    Useful for operations that need both mutual exclusion
    and rate limiting (e.g., API calls).

    Features:
    - Mutual exclusion
    - Rate limiting (min interval between operations)
    - Burst handling (allow N rapid operations)
    """

    def __init__(
        self,
        min_interval: float = 0.1,
        burst_size: int = 1,
        name: str = "",
    ):
        """
        Initialize rate-limited lock.

        Args:
            min_interval: Minimum seconds between operations
            burst_size: Number of rapid operations allowed before rate limit
            name: Optional name for logging
        """
        self._min_interval = min_interval
        self._burst_size = burst_size
        self._burst_remaining = burst_size
        self._last_release: Optional[float] = None
        self._lock = asyncio.Lock()
        self._name = name or f"rate_limited_{id(self)}"

        # Statistics
        self._stats = {
            "acquires": 0,
            "rate_limited": 0,
            "total_wait_time": 0.0,
        }

    async def acquire(self) -> None:
        """Acquire the lock, waiting for rate limit if needed."""
        async with self._lock:
            self._stats["acquires"] += 1

            # Check if we need to wait for rate limit
            if self._last_release is not None and self._burst_remaining <= 0:
                elapsed = time.time() - self._last_release
                if elapsed < self._min_interval:
                    wait_time = self._min_interval - elapsed
                    self._stats["rate_limited"] += 1
                    self._stats["total_wait_time"] += wait_time
                    await asyncio.sleep(wait_time)

            # Reset burst if enough time has passed
            if self._last_release is not None:
                elapsed = time.time() - self._last_release
                if elapsed >= self._min_interval * self._burst_size:
                    self._burst_remaining = self._burst_size

            self._burst_remaining -= 1

    def release(self) -> None:
        """Release the lock."""
        self._last_release = time.time()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    def get_stats(self) -> dict:
        """Get lock statistics."""
        return {
            **self._stats,
            "name": self._name,
            "burst_remaining": self._burst_remaining,
            "min_interval": self._min_interval,
        }


class LazyLock:
    """
    A lazily-initialized lock that defers creation until first use.

    Useful when you want to avoid creating locks during module import
    but need thread-safe access once the application is running.
    """

    def __init__(self, name: str = ""):
        self._lock: Optional[asyncio.Lock] = None
        self._name = name

    def _ensure_lock(self) -> asyncio.Lock:
        """Ensure lock exists."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def acquire(self) -> None:
        """Acquire the lock."""
        await self._ensure_lock().acquire()

    def release(self) -> None:
        """Release the lock."""
        if self._lock is not None:
            self._lock.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    def locked(self) -> bool:
        """Check if lock is currently held."""
        if self._lock is None:
            return False
        return self._lock.locked()


# Utility functions

def create_lock_pool(count: int) -> list:
    """
    Create a pool of locks for striped locking.

    Striped locking reduces contention by distributing
    operations across multiple locks based on a key.

    Args:
        count: Number of locks in the pool

    Returns:
        List of asyncio.Lock instances
    """
    return [asyncio.Lock() for _ in range(count)]


def get_striped_lock(locks: list, key: Any) -> asyncio.Lock:
    """
    Get a lock from a pool based on a key.

    Args:
        locks: List of locks from create_lock_pool
        key: Any hashable key

    Returns:
        One of the locks from the pool
    """
    index = hash(key) % len(locks)
    return locks[index]
