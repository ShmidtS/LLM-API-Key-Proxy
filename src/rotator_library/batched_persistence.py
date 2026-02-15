# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

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
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass, field

from .utils.resilient_io import safe_write_json

lib_logger = logging.getLogger("rotator_library")


@dataclass
class PersistenceConfig:
    """Configuration for batched persistence."""
    write_interval: float = 5.0  # Seconds between writes
    max_dirty_age: float = 30.0  # Max age before forced write
    enable_disk: bool = True
    env_prefix: str = "BATCHED_PERSISTENCE"


def _env_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    """Get boolean from environment variable."""
    return os.getenv(key, str(default).lower()).lower() in ("true", "1", "yes")


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
            serializer=lambda data: json.dumps(data, indent=2),
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
        self._serializer = serializer or (lambda d: json.dumps(d, indent=2))
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
            with open(self._file_path, "r", encoding="utf-8") as f:
                content = f.read()
            self._state = json.loads(content)
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
                    age = time.time() - self._last_change
                    if age >= self._config.write_interval or age >= self._config.max_dirty_age:
                        await self._write_to_disk()
        except asyncio.CancelledError:
            pass

    async def _write_to_disk(self) -> bool:
        """Write current state to disk."""
        if not self._config.enable_disk or self._state is None:
            return True

        async with self._lock:
            try:
                # Ensure directory exists
                self._file_path.parent.mkdir(parents=True, exist_ok=True)

                # Serialize and write
                content = self._serializer(self._state)

                success = safe_write_json(
                    self._file_path,
                    self._state if isinstance(self._state, dict) else {"data": self._state},
                    lib_logger,
                    atomic=True,
                    indent=2,
                )

                if success:
                    self._dirty = False
                    self._last_write = time.time()
                    self._stats["writes"] += 1
                    self._stats["bytes_written"] += len(content)
                    return True
                else:
                    self._stats["write_errors"] += 1
                    return False

            except Exception as e:
                lib_logger.error(f"Failed to write state to {self._file_path.name}: {e}")
                self._stats["write_errors"] += 1
                return False

    def update(self, state: Any) -> None:
        """
        Update state (in-memory, marks dirty).

        This is a fast in-memory operation. Disk write happens
        in background.

        Args:
            state: New state value
        """
        self._state = state
        self._dirty = True
        self._last_change = time.time()
        self._stats["updates"] += 1

    async def update_async(self, state: Any) -> None:
        """
        Update state asynchronously (thread-safe).

        Args:
            state: New state value
        """
        async with self._lock:
            self.update(state)

    def get_state(self) -> Any:
        """Get current state (from memory)."""
        return self._state

    async def force_write(self) -> bool:
        """
        Force immediate write to disk.

        Returns:
            True if write succeeded
        """
        return await self._write_to_disk()

    async def stop(self) -> None:
        """Stop background writer and force final write."""
        self._running = False

        if self._writer_task:
            self._writer_task.cancel()
            try:
                await self._writer_task
            except asyncio.CancelledError:
                pass

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
