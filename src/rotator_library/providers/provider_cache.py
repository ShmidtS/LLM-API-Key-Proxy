# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# src/rotator_library/providers/provider_cache.py
"""
Shared cache utility for providers.

A modular, async-capable cache system supporting:
- Dual-TTL: short-lived memory cache, longer-lived disk persistence
- Background persistence with batched writes
- Automatic cleanup of expired entries
- Generic key-value storage for any provider-specific needs

Usage examples:
- Gemini 3: thoughtSignatures (tool_call_id → encrypted signature)
- Claude: Thinking content (composite_key → thinking text + signature)
- General: Any transient data that benefits from persistence across requests
"""

import asyncio
from orjson import JSONDecodeError
import logging
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

from ..async_locks import ReadWriteLock
from ..config import env_bool as _env_bool, env_int as _env_int
from ..utils.json_utils import json_loads
from ..utils.resilient_io import safe_write_json

lib_logger = logging.getLogger("rotator_library")


def _read_file_sync(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# =============================================================================
# PROVIDER CACHE CLASS
# =============================================================================


class ProviderCache:
    """
    Server-side cache for provider conversation state preservation.

    A generic, modular cache supporting any key-value data that providers need
    to persist across requests. Features:

    - Dual-TTL system: entries live in memory for memory_ttl, but persist on
      disk for the longer disk_ttl. Memory cleanup does NOT affect disk entries.
    - Merge-on-save: disk writes merge current memory with existing disk entries,
      preserving disk-only entries until they exceed disk_ttl
    - Async disk persistence with batched writes
    - Background cleanup task for memory-expired entries (disk untouched)
    - Statistics tracking (hits, misses, writes, disk preservation)

    Args:
        cache_file: Path to disk cache file
        memory_ttl_seconds: In-memory entry lifetime (default: 1 hour)
        disk_ttl_seconds: Disk entry lifetime (default: 48 hours)
        enable_disk: Whether to enable disk persistence (default: from env or True)
        write_interval: Seconds between background disk writes (default: 60)
        cleanup_interval: Seconds between expired entry cleanup (default: 30 min)
        env_prefix: Environment variable prefix for configuration overrides

    Environment Variables (with default prefix "PROVIDER_CACHE"):
        {PREFIX}_ENABLE: Enable/disable disk persistence
        {PREFIX}_WRITE_INTERVAL: Background write interval in seconds
        {PREFIX}_CLEANUP_INTERVAL: Cleanup interval in seconds
    """

    def __init__(
        self,
        cache_file: Path,
        memory_ttl_seconds: int = 3600,
        disk_ttl_seconds: int = 172800,  # 48 hours
        enable_disk: Optional[bool] = None,
        write_interval: Optional[int] = None,
        cleanup_interval: Optional[int] = None,
        env_prefix: str = "PROVIDER_CACHE",
        max_entries: int = 10000,
    ):
        # In-memory cache (OrderedDict for O(1) LRU eviction): {cache_key: {"value", "timestamp", "accessed"}}
        self._cache: OrderedDict[str, Dict[str, float | str]] = OrderedDict()
        self._memory_ttl = memory_ttl_seconds
        self._disk_ttl = disk_ttl_seconds
        self._rw_lock = ReadWriteLock()   # Read-write lock for cache access (read-heavy optimization)
        self._disk_lock = asyncio.Lock()
        self._max_entries = max_entries
        self._evicted_count = 0

        # Disk persistence configuration
        self._cache_file = cache_file
        self._enable_disk = (
            enable_disk
            if enable_disk is not None
            else _env_bool(f"{env_prefix}_ENABLE", True)
        )
        self._dirty = False
        self._dirty_generation = 0
        self._write_interval = write_interval or _env_int(
            f"{env_prefix}_WRITE_INTERVAL", 60
        )
        self._cleanup_interval = cleanup_interval or _env_int(
            f"{env_prefix}_CLEANUP_INTERVAL", 1800
        )

        # Background tasks
        self._writer_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._init_task: Optional[asyncio.Task] = None
        self._pending_tasks: set = set()
        self._running = False

        # Statistics
        self._stats = {
            "memory_hits": 0,
            "disk_hits": 0,
            "misses": 0,
            "writes": 0,
            "disk_errors": 0,
        }

        # Track disk health for monitoring
        self._disk_available = True

        # In-memory index of disk entries for O(1) lookup (invalidated on write)
        self._DISK_ENTRY_CACHE_MAXSIZE: int = _env_int(f"{env_prefix}_DISK_ENTRY_CACHE_MAXSIZE", 4096)
        self._disk_entry_cache: Dict[str, Dict] = {}
        self._disk_cache_valid = False

        # Metadata about this cache instance
        self._cache_name = cache_file.stem if cache_file else "unnamed"

        self._initialized = not self._enable_disk  # Memory-only caches are ready immediately

        if self._enable_disk:
            lib_logger.debug(
                f"ProviderCache[{self._cache_name}]: Disk enabled "
                f"(memory_ttl={memory_ttl_seconds}s, disk_ttl={disk_ttl_seconds}s)"
            )
            self._init_task = asyncio.create_task(self._async_init())
        else:
            lib_logger.debug(f"ProviderCache[{self._cache_name}]: Memory-only mode")

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    async def _async_init(self) -> None:
        """Async initialization: load from disk and start background tasks."""
        try:
            await self._load_from_disk()
            await self._start_background_tasks()
            self._initialized = True
        except Exception as e:
            lib_logger.error(
                f"ProviderCache[{self._cache_name}] async init failed: {e}"
            )
            self._initialized = True  # Allow operation even if init failed

    async def _load_from_disk(self) -> None:
        """Load cache from disk file with TTL validation."""
        if not self._enable_disk or not self._cache_file.exists():
            return

        try:
            async with self._disk_lock:
                raw = await asyncio.to_thread(_read_file_sync, self._cache_file)
                data = json_loads(raw)

                if data.get("version") != "1.0":
                    lib_logger.warning(
                        f"ProviderCache[{self._cache_name}]: Version mismatch, starting fresh"
                    )
                    return

                now = time.time()
                entries = data.get("entries", {})
                loaded = expired = 0

                for cache_key, entry in entries.items():
                    age = now - entry.get("timestamp", 0)
                    if age <= self._disk_ttl:
                        value = entry.get(
                            "value", entry.get("signature", "")
                        )  # Support both formats
                        if value:
                            async with self._rw_lock.write():
                                self._cache[cache_key] = {
                                    "value": value,
                                    "timestamp": entry["timestamp"],
                                    "accessed": now,
                                }
                                self._cache.move_to_end(cache_key)
                            loaded += 1
                    else:
                        expired += 1

                lib_logger.debug(
                    f"ProviderCache[{self._cache_name}]: Loaded {loaded} entries ({expired} expired)"
                )
        except JSONDecodeError as e:
            lib_logger.warning(
                f"ProviderCache[{self._cache_name}]: File corrupted: {e}"
            )
        except Exception as e:
            lib_logger.error(f"ProviderCache[{self._cache_name}]: Load failed: {e}")

    # =========================================================================
    # DISK PERSISTENCE
    # =========================================================================

    async def _save_to_disk(self) -> bool:
        """Persist cache to disk using atomic write with health tracking.

        Implements dual-TTL preservation: merges current memory state with
        existing disk entries that haven't exceeded disk_ttl. This ensures
        entries persist on disk for the full disk_ttl even after they expire
        from memory (which uses the shorter memory_ttl).

        Returns:
            True if write succeeded, False otherwise.
        """
        if not self._enable_disk:
            return True  # Not an error if disk is disabled

        async with self._disk_lock:
            now = time.time()

            # Step 1: Use in-memory disk cache if valid, otherwise read from disk
            existing_entries: Dict[str, Dict[str, Any]] = {}
            if self._disk_cache_valid:
                existing_entries = self._disk_entry_cache
            elif self._cache_file.exists():
                try:
                    raw = await asyncio.to_thread(_read_file_sync, self._cache_file)
                    data = json_loads(raw)
                    existing_entries = data.get("entries", {})
                except (JSONDecodeError, IOError, OSError):
                    pass  # Start fresh if corrupted or unreadable

            # Step 2: Filter existing disk entries by disk_ttl (not memory_ttl)
            # This preserves entries that expired from memory but are still valid on disk
            valid_disk_entries = {
                k: v
                for k, v in existing_entries.items()
                if now - v.get("timestamp", 0) <= self._disk_ttl
            }

            # Step 3: Merge - memory entries take precedence (fresher timestamps)
            merged_entries = valid_disk_entries.copy()
            async with self._rw_lock.read():
                cache_snapshot = list(self._cache.items())
            for key, entry in cache_snapshot:
                merged_entries[key] = {"value": entry["value"], "timestamp": entry["timestamp"]}

            # Count entries that were preserved from disk (not in memory)
            memory_keys = {k for k, _ in cache_snapshot}
            preserved_from_disk = len(
                [k for k in valid_disk_entries if k not in memory_keys]
            )

            # Step 4: Build and save merged cache data
            cache_data = {
                "version": "1.0",
                "memory_ttl_seconds": self._memory_ttl,
                "disk_ttl_seconds": self._disk_ttl,
                "entries": merged_entries,
                "statistics": {
                    "total_entries": len(merged_entries),
                    "memory_entries": len(cache_snapshot),
                    "disk_preserved": preserved_from_disk,
                    "last_write": now,
                    **self._stats,
                },
            }

            if safe_write_json(
                self._cache_file, cache_data, lib_logger, secure_permissions=True
            ):
                self._stats["writes"] += 1
                self._disk_available = True
                # Cache the merged entries as the in-memory disk snapshot
                # so next save merges from memory instead of re-reading the file
                self._disk_entry_cache = merged_entries
                self._trim_disk_entry_cache()
                self._disk_cache_valid = True
                # Log merge info only when we preserved disk-only entries (infrequent)
                if preserved_from_disk > 0:
                    lib_logger.debug(
                        f"ProviderCache[{self._cache_name}]: Saved {len(merged_entries)} entries "
                        f"(memory={len(cache_snapshot)}, preserved_from_disk={preserved_from_disk})"
                    )
                return True
            else:
                self._stats["disk_errors"] += 1
                self._disk_available = False
                return False

    # =========================================================================
    # BACKGROUND TASKS
    # =========================================================================

    async def _start_background_tasks(self) -> None:
        """Start background writer and cleanup tasks."""
        if not self._enable_disk or self._running:
            return

        self._running = True
        self._writer_task = asyncio.create_task(self._writer_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        lib_logger.debug(f"ProviderCache[{self._cache_name}]: Started background tasks")

    async def _writer_loop(self) -> None:
        """Background task: periodically flush dirty cache to disk."""
        try:
            while self._running:
                await asyncio.sleep(self._write_interval)
                async with self._rw_lock.read():
                    if not self._dirty:
                        continue
                    dirty_generation = self._dirty_generation
                try:
                    success = await self._save_to_disk()
                    async with self._rw_lock.write():
                        if success and self._dirty_generation == dirty_generation:
                            self._dirty = False
                        elif not success:
                            self._dirty = True
                except Exception as e:
                    async with self._rw_lock.write():
                        self._dirty = True
                    lib_logger.error(
                        f"ProviderCache[{self._cache_name}]: Writer error: {e}"
                    )
        except asyncio.CancelledError:
            lib_logger.debug(f"ProviderCache[{self._cache_name}]: Writer loop cancelled")
            raise

    async def _cleanup_loop(self) -> None:
        """Background task: periodically clean up expired entries."""
        try:
            while self._running:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired()
        except asyncio.CancelledError:
            lib_logger.debug(f"ProviderCache[{self._cache_name}]: Cleanup loop cancelled")
            raise

    async def _cleanup_expired(self) -> None:
        """Remove expired entries from memory cache.

        Only cleans memory - disk entries are preserved and cleaned during
        _save_to_disk() based on their own disk_ttl.
        """
        async with self._rw_lock.write():
            now = time.time()
            while len(self._cache) >= self._max_entries:
                # popitem(last=False) removes the least-recently-used entry
                self._cache.popitem(last=False)
                self._evicted_count += 1

            expired = [
                k
                for k, entry in self._cache.items()
                if now - entry["timestamp"] > self._memory_ttl
            ]
            for k in expired:
                del self._cache[k]
            # Don't set dirty flag: memory cleanup shouldn't trigger disk write
            # Disk entries are cleaned separately in _save_to_disk() by disk_ttl
            if expired:
                lib_logger.debug(
                    f"ProviderCache[{self._cache_name}]: Cleaned {len(expired)} expired entries from memory"
                )

    # =========================================================================
    # CORE OPERATIONS
    # =========================================================================

    def _schedule_task(self, coro) -> None:
        """Create a tracked background task that auto-removes when done."""
        task = asyncio.create_task(coro)
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    def store(self, key: str, value: str) -> None:
        """
        Store a value synchronously (schedules async storage).

        Args:
            key: Cache key
            value: Value to store (typically JSON-serialized data)
        """
        try:
            self._schedule_task(self._async_store(key, value))
        except RuntimeError:
            lib_logger.warning(
                f"ProviderCache[{self._cache_name}]: store() called outside event loop"
            )

    async def _async_store(self, key: str, value: str) -> None:
        """Async implementation of store."""
        now = time.time()
        async with self._rw_lock.write():
            self._cache[key] = {"value": value, "timestamp": now, "accessed": now}
            self._cache.move_to_end(key)
            self._dirty = True
            self._dirty_generation += 1

    async def store_async(self, key: str, value: str) -> None:
        """
        Store a value asynchronously (awaitable).

        Use this when you need to ensure the value is stored before continuing.
        """
        await self._async_store(key, value)

    async def _touch_key(self, key: str) -> None:
        """Move key to end of LRU (called from sync retrieve via create_task)."""
        async with self._rw_lock.write():
            if key in self._cache:
                self._cache.move_to_end(key)

    async def _remove_expired_key(self, key: str) -> None:
        """Remove an expired key from memory cache (called from sync retrieve via create_task)."""
        async with self._rw_lock.write():
            self._cache.pop(key, None)

    def retrieve(self, key: str) -> Optional[str]:
        """
        Retrieve a value by key (synchronous, with optional async disk fallback).

        Args:
            key: Cache key

        Returns:
            Cached value if found and not expired, None otherwise
        """
        if not self._initialized:
            lib_logger.warning(
                f"ProviderCache[{self._cache_name}]: retrieve('{key}') called "
                "before async init completed — disk data not yet loaded"
            )
        # Read-only access: dict.__getitem__ and time.time() are atomic in CPython.
        # Mutation (move_to_end, del) requires async lock — defer to cleanup.
        entry = self._cache.get(key)
        if entry is not None:
            if time.time() - entry["timestamp"] <= self._memory_ttl:
                self._stats["memory_hits"] += 1
                # Schedule LRU re-order and expiry cleanup asynchronously
                try:
                    self._schedule_task(self._touch_key(key))
                except RuntimeError:
                    lib_logger.warning(
                        f"ProviderCache[{self._cache_name}]: retrieve() called outside event loop; touch skipped"
                    )
                return entry["value"]
            else:
                # Entry expired — schedule removal via async path (no race)
                try:
                    self._schedule_task(self._remove_expired_key(key))
                except RuntimeError:
                    lib_logger.warning(
                        f"ProviderCache[{self._cache_name}]: retrieve() called outside event loop; expired key removal skipped"
                    )

        self._stats["misses"] += 1
        return None

    async def retrieve_async(self, key: str) -> Optional[str]:
        """
        Retrieve a value asynchronously (checks disk if not in memory).

        Use this when you can await and need guaranteed disk fallback.
        """
        # Check memory first under read-lock
        async with self._rw_lock.read():
            if key in self._cache:
                entry = self._cache[key]
                if time.time() - entry["timestamp"] <= self._memory_ttl:
                    self._stats["memory_hits"] += 1
                    return entry["value"]
                # Entry expired — need write-lock to remove; fall through below
            else:
                # Key not in cache at all — skip write-lock, go to disk
                pass

        # If we reach here, key was in cache but expired — remove under write-lock
        async with self._rw_lock.write():
            # Re-check: another task may have updated or removed the key
            if key in self._cache:
                entry = self._cache[key]
                if time.time() - entry["timestamp"] <= self._memory_ttl:
                    # Another task refreshed it while we waited
                    self._stats["memory_hits"] += 1
                    entry["accessed"] = time.time()
                    return entry["value"]
                # Still expired — remove from memory only
                # Don't set dirty flag: disk copy should persist until disk_ttl
                del self._cache[key]

        # Check disk
        if self._enable_disk:
            return await self._disk_lookup(key, return_value=True)

        self._stats["misses"] += 1
        return None

    def _trim_disk_entry_cache(self) -> None:
        """Evict oldest entries from _disk_entry_cache if it exceeds max size."""
        if len(self._disk_entry_cache) > self._DISK_ENTRY_CACHE_MAXSIZE:
            # Remove oldest entries by timestamp (ascending)
            sorted_keys = sorted(
                self._disk_entry_cache,
                key=lambda k: self._disk_entry_cache[k].get("timestamp", 0),
            )
            excess = len(self._disk_entry_cache) - self._DISK_ENTRY_CACHE_MAXSIZE
            for k in sorted_keys[:excess]:
                del self._disk_entry_cache[k]

    async def _load_disk_index(self) -> Dict[str, Dict]:
        """Load and cache all disk entries. Returns dict of key -> entry.

        Reads the file once, parses all entries, and caches them in memory
        for O(1) lookups until invalidated by a write.
        """
        if self._disk_cache_valid and self._disk_entry_cache:
            return self._disk_entry_cache

        try:
            async with self._disk_lock:
                if not self._cache_file.exists():
                    self._disk_entry_cache = {}
                    self._disk_cache_valid = True
                    return self._disk_entry_cache

                raw = await asyncio.to_thread(_read_file_sync, self._cache_file)
                data = json_loads(raw)

                entries = data.get("entries", {})
                # Filter by disk_ttl at load time to keep index lean
                now = time.time()
                self._disk_entry_cache = {
                    k: v for k, v in entries.items()
                    if now - v.get("timestamp", 0) <= self._disk_ttl
                }
                self._trim_disk_entry_cache()
                self._disk_cache_valid = True
        except (JSONDecodeError, IOError, OSError):
            self._disk_entry_cache = {}
            self._disk_cache_valid = True
        except Exception as e:
            lib_logger.debug(
                f"ProviderCache[{self._cache_name}]: Disk index load failed: {e}"
            )
            self._disk_entry_cache = {}
            self._disk_cache_valid = True
        return self._disk_entry_cache

    async def _disk_lookup(self, key: str, return_value: bool = False) -> Optional[str]:
        """Look up a key in the disk index and load into memory if found.

        Args:
            key: Cache key to look up.
            return_value: If True, return the value (for retrieve_async).
                If False, just load into memory (for background fallback).

        Returns:
            The cached value if return_value=True and found, None otherwise.
        """
        try:
            entries = await self._load_disk_index()
            entry = entries.get(key)
            if entry is None:
                if return_value:
                    self._stats["misses"] += 1
                return None

            ts = entry.get("timestamp", 0)
            value = entry.get("value", entry.get("signature", ""))
            if not value:
                if return_value:
                    self._stats["misses"] += 1
                return None

            async with self._rw_lock.write():
                now = time.time()
                self._cache[key] = {"value": value, "timestamp": ts, "accessed": now}
                self._cache.move_to_end(key)
            self._stats["disk_hits"] += 1
            if not return_value:
                lib_logger.debug(
                    f"ProviderCache[{self._cache_name}]: Loaded {key} from disk"
                )
            return value if return_value else None
        except Exception as e:
            lib_logger.debug(
                f"ProviderCache[{self._cache_name}]: Disk lookup failed: {e}"
            )
            if return_value:
                self._stats["misses"] += 1
            return None

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def contains(self, key: str) -> bool:
        """Check if key exists in memory cache (without updating stats)."""
        entry = self._cache.get(key)
        if entry is not None:
            return time.time() - entry["timestamp"] <= self._memory_ttl
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics including disk health."""
        memory_entries = len(self._cache)
        return {
            **self._stats,
            "memory_entries": memory_entries,
            "dirty": self._dirty,
            "disk_enabled": self._enable_disk,
            "disk_available": self._disk_available,
        }

    async def clear(self) -> None:
        """Clear all cached data."""
        async with self._rw_lock.write():
            self._cache.clear()
            self._dirty = True
            self._dirty_generation += 1
        if self._enable_disk:
                await self._save_to_disk()

    async def shutdown(self) -> None:
        """Graceful shutdown: flush pending writes and stop background tasks."""
        lib_logger.info(f"ProviderCache[{self._cache_name}]: Shutting down...")
        self._running = False

        # Cancel init task if still running
        if self._init_task and not self._init_task.done():
            self._init_task.cancel()
            try:
                await self._init_task
            except asyncio.CancelledError:
                raise
            except Exception as e:
                lib_logger.debug(f"ProviderCache[{self._cache_name}]: Init task error during shutdown: {e}")

        # Cancel background tasks
        for task in (self._writer_task, self._cleanup_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    lib_logger.debug(f"ProviderCache[{self._cache_name}]: Shutdown cancelled")
                except Exception as e:
                    lib_logger.debug(f"ProviderCache[{self._cache_name}]: Shutdown task error: {e}")

        # Wait for pending tasks to complete
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
            self._pending_tasks.clear()

        # Final save
        if self._dirty and self._enable_disk:
            await self._save_to_disk()

        lib_logger.info(
            f"ProviderCache[{self._cache_name}]: Shutdown complete "
            f"(stats: mem_hits={self._stats['memory_hits']}, "
            f"disk_hits={self._stats['disk_hits']}, misses={self._stats['misses']})"
        )
