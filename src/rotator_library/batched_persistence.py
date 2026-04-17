# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# src/rotator_library/batched_persistence.py
"""
Batched disk persistence for high-throughput state updates.

Provides background batched writes to avoid blocking request paths
with synchronous disk I/O. Similar to ProviderCache pattern but
generalized for any state data.
"""

import asyncio
import json
import logging
from .utils.json_utils import json_loads
import orjson
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass

import aiofiles

from .config import env_bool as _env_bool, env_float as _env_float
from .utils.resilient_io import safe_write_json

lib_logger = logging.getLogger("rotator_library")
_PENDING_STATE_EMPTY = object()


@dataclass
class PersistenceConfig:
    """Configuration for batched persistence."""
    write_interval: float = 5.0  # Seconds between writes
    max_dirty_age: float = 30.0  # Max age before forced write
    enable_disk: bool = True
    env_prefix: str = "BATCHED_PERSISTENCE"

class BatchedPersistence:
    """
    Manages batched disk writes for state data.

    Instead of writing to disk on every state change, this class:
    - Keeps state in memory
    - Marks state as "dirty" on changes
    - Writes to disk periodically in background
    - Ensures final write on shutdown

    This dramatically reduces disk I/O in high-throughput scenarios
    while ensuring data durability.

    Usage:
        persistence = BatchedPersistence(
            file_path=Path("data/state.json"),
            serializer=lambda data: orjson.dumps(data, option=orjson.OPT_INDENT_2).decode(),
        )
        await persistence.start()

        # Update state (fast, in-memory)
        persistence.update({"key": "value"})

        # On shutdown
        await persistence.stop()
    """

    def __init__(
        self,
        file_path: Path,
        serializer: Optional[Callable[[Any], str]] = None,
        config: Optional[PersistenceConfig] = None,
    ):
        """
        Initialize batched persistence.

        Args:
            file_path: Path to the state file
            serializer: Function to serialize state to string (default: JSON indent=2)
            config: Persistence configuration
        """
        self._file_path = file_path
        self._serializer = serializer or (lambda d: orjson.dumps(d, option=orjson.OPT_INDENT_2).decode())
        self._config = config or PersistenceConfig()

        # Override from environment
        env_prefix = self._config.env_prefix
        self._config.write_interval = _env_float(
            f"{env_prefix}_WRITE_INTERVAL", self._config.write_interval
        )
        self._config.max_dirty_age = _env_float(
            f"{env_prefix}_MAX_DIRTY_AGE", self._config.max_dirty_age
        )
        self._config.enable_disk = _env_bool(
            f"{env_prefix}_ENABLE", self._config.enable_disk
        )

        # State
        self._state: Any = None
        self._dirty = False
        self._last_write: Optional[float] = None
        self._last_change: Optional[float] = None
        self._lock = asyncio.Lock()

        # Background task
        self._writer_task: Optional[asyncio.Task] = None
        self._pending_update_task: Optional[asyncio.Task] = None
        self._pending_state: Any = _PENDING_STATE_EMPTY
        self._running = False

        # Statistics
        self._stats = {
            "updates": 0,
            "writes": 0,
            "write_errors": 0,
            "bytes_written": 0,
        }

    async def start(self) -> None:
        """Start the background writer task."""
        if not self._config.enable_disk or self._running:
            return

        # Load existing state from disk
        await self._load_from_disk()

        # Start background writer
        self._running = True
        self._writer_task = asyncio.create_task(self._writer_loop())

        lib_logger.debug(
            f"BatchedPersistence started for {self._file_path.name} "
            f"(interval={self._config.write_interval}s)"
        )

    async def _load_from_disk(self) -> None:
        """Load state from disk file if it exists."""
        if not self._file_path.exists():
            return

        try:
            async with aiofiles.open(self._file_path, "r", encoding="utf-8") as f:
               content = await f.read()
               self._state = json_loads(content)
            lib_logger.debug(f"Loaded state from {self._file_path.name}")
        except (json.JSONDecodeError, IOError, OSError) as e:
            lib_logger.warning(f"Failed to load state from {self._file_path.name}: {e}")

    async def _writer_loop(self) -> None:
        """Background task: periodically write dirty state to disk."""
        try:
            while self._running:
                await asyncio.sleep(self._config.write_interval)

                # Check if we need to write
                if not self._dirty:
                    continue

                # Check if enough time has passed or max age exceeded
                if self._last_change is not None:
                    age = time.monotonic() - self._last_change
                    if age >= self._config.write_interval or age >= self._config.max_dirty_age:
                        await self._write_to_disk()
        except asyncio.CancelledError:
            lib_logger.debug("Writer loop cancelled")

    async def _write_to_disk(self) -> bool:
        """Write current state to disk (non-blocking)."""
        async with self._lock:
            if not self._config.enable_disk or self._state is None:
                return True
            try:
                # Ensure directory exists
                self._file_path.parent.mkdir(parents=True, exist_ok=True)

                # Serialize and write off the event loop to avoid blocking
                data = self._state if isinstance(self._state, dict) else {"data": self._state}
                write_started_at = self._last_change
                success = await asyncio.to_thread(
                    safe_write_json,
                    self._file_path,
                    data,
                    lib_logger,
                    True,
                )

                if success:
                    if self._last_change == write_started_at:
                        self._dirty = False
                    self._last_write = time.monotonic()
                    self._stats["writes"] += 1
                    try:
                        self._stats["bytes_written"] += self._file_path.stat().st_size
                    except OSError:
                        lib_logger.debug("Could not stat file for bytes_written tracking")
                    return True
                else:
                    self._stats["write_errors"] += 1
                    return False

            except Exception as e:
                lib_logger.error(f"Failed to write state to {self._file_path.name}: {e}")
                self._stats["write_errors"] += 1
                return False

    def _apply_update(self, state: Any) -> None:
        """Internal mutation logic for state updates."""
        self._state = state
        self._dirty = True
        self._last_change = time.monotonic()
        self._stats["updates"] += 1

    def update(self, state: Any) -> None:
        """
        Update state (in-memory, marks dirty).

        When an event loop is running, schedules at most one update task
        and coalesces bursty updates into the latest pending state.
        When no loop is running, mutates directly (safe — no concurrent writer).

        Args:
            state: New state value
        """
        try:
            loop = asyncio.get_running_loop()
            self._pending_state = state
            if self._pending_update_task is None or self._pending_update_task.done():
                self._pending_update_task = loop.create_task(self._flush_pending_update())
        except RuntimeError:
            self._apply_update(state)

    async def update_async(self, state: Any) -> None:
        """
        Update state asynchronously (thread-safe).

        Args:
            state: New state value
        """
        async with self._lock:
            self._apply_update(state)

    async def _flush_pending_update(self) -> None:
        while True:
            async with self._lock:
                state = self._pending_state
                self._pending_state = _PENDING_STATE_EMPTY
                if state is not _PENDING_STATE_EMPTY:
                    self._apply_update(state)

            if self._pending_state is _PENDING_STATE_EMPTY:
                return

    def get_state(self) -> Any:
        """Get current state (from memory)."""
        return self._state

    async def force_write(self) -> bool:
        """
        Force immediate write to disk.

        Returns:
            True if write succeeded
        """
        if self._pending_update_task:
            await self._pending_update_task
        return await self._write_to_disk()

    async def stop(self) -> None:
        """Stop background writer and force final write."""
        self._running = False

        if self._writer_task:
            self._writer_task.cancel()
            try:
                await self._writer_task
            except asyncio.CancelledError:
                lib_logger.debug("Writer task cancelled during stop")

        if self._pending_update_task:
            await self._pending_update_task

        # Final write
        if self._dirty and self._state is not None:
            await self._write_to_disk()

        lib_logger.info(
            f"BatchedPersistence stopped for {self._file_path.name} "
            f"(writes={self._stats['writes']}, errors={self._stats['write_errors']})"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get persistence statistics."""
        return {
            **self._stats,
            "dirty": self._dirty,
            "last_write": self._last_write,
            "last_change": self._last_change,
            "file_path": str(self._file_path),
            "running": self._running,
        }

    @property
    def is_dirty(self) -> bool:
        """Check if there are pending writes."""
        return self._dirty


class UsagePersistenceManager:
    """
    Specialized batched persistence for usage data.

    Manages the key_usage.json file with optimized batching
    for high-frequency usage updates.
    """

    def __init__(self, file_path: Path):
        """Initialize usage persistence manager."""
        self._persistence = BatchedPersistence(
            file_path=file_path,
            config=PersistenceConfig(
                write_interval=5.0,  # Write every 5 seconds max
                max_dirty_age=15.0,  # Force write after 15 seconds
                env_prefix="USAGE_PERSISTENCE",
            )
        )
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize and start background writer."""
        if self._initialized:
            return

        await self._persistence.start()
        self._initialized = True

    def update_usage(self, usage_data: Dict[str, Any]) -> None:
        """
        Update usage data (fast, in-memory).

        Args:
            usage_data: Usage data dictionary
        """
        self._persistence.update(usage_data)

    async def force_save(self) -> bool:
        """Force immediate save to disk."""
        return await self._persistence.force_write()

    def get_usage(self) -> Optional[Dict[str, Any]]:
        """Get current usage data."""
        return self._persistence.get_state()

    async def shutdown(self) -> None:
        """Shutdown and save final state."""
        await self._persistence.stop()

    def get_stats(self) -> Dict[str, Any]:
        """Get persistence statistics."""
        return self._persistence.get_stats()
