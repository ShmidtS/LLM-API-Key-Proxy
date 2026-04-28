# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

from ._constants import lib_logger
from typing import Dict, Any
from ..utils.json_utils import json_loads
from ..batched_persistence import UsagePersistenceManager
import aiofiles
import copy
import json
import os


class UsageManagerPersistenceMixin:
    async def _lazy_init(self):
        """Initializes the usage data by loading it from the file asynchronously."""
        async with self._init_lock:
            if not self._initialized.is_set():
                await self._load_usage()
                await self._reset_daily_stats_if_needed()

                # Initialize batch persistence if enabled
                if self._use_batch_persistence:
                    from pathlib import Path

                    self._batch_persistence = UsagePersistenceManager(
                        Path(self.file_path)
                    )
                    await self._batch_persistence.initialize()
                    lib_logger.info("Batch persistence enabled for usage data")

                self._initialized.set()

    async def _load_usage(self):
        """Loads usage data from the JSON file asynchronously with resilience."""
        async with self._data_lock.write():
            if not os.path.exists(self.file_path):
                self._usage_data = {}
                return

            try:
                async with aiofiles.open(self.file_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                    self._usage_data = json_loads(content) if content.strip() else {}
            except FileNotFoundError:
                # File deleted between exists check and open
                self._usage_data = {}
            except json.JSONDecodeError as e:
                lib_logger.warning(
                    f"Corrupted usage file {self.file_path}: {e}. Starting fresh."
                )
                self._usage_data = {}
            except (OSError, PermissionError, IOError) as e:
                lib_logger.warning(
                    f"Cannot read usage file {self.file_path}: {e}. Using empty state."
                )
                self._usage_data = {}

            # Restore fair cycle state from persisted data
            fair_cycle_data = self._usage_data.get("__fair_cycle__", {})
            if fair_cycle_data:
                self._deserialize_cycle_state(fair_cycle_data)

    async def _save_usage(self):
        """Saves the current usage data using the resilient state writer or batch persistence."""
        if self._usage_data is None:
            return

        async with self._data_lock.write():
            # Add human-readable timestamp fields before saving
            self._add_readable_timestamps(self._usage_data)

            # Persist fair cycle state (separate from credential data)
            if self._cycle_exhausted:
                self._usage_data["__fair_cycle__"] = self._serialize_cycle_state()
            elif "__fair_cycle__" in self._usage_data:
                # Clean up empty cycle data
                del self._usage_data["__fair_cycle__"]

            snapshot = copy.deepcopy(self._usage_data)

        # Use batch persistence if enabled (high-throughput mode)
        if self._use_batch_persistence and self._batch_persistence:
            self._batch_persistence.update_usage(snapshot)
        else:
            # Hand off to resilient writer - handles retries and disk failures
            await self._state_writer.write(snapshot)

    async def _get_usage_data_snapshot(self) -> Dict[str, Any]:
        """
        Get a shallow copy of the current usage data.

        Returns:
            Copy of usage data dict (safe for reading without lock)
        """
        await self._lazy_init()
        async with self._data_lock.read():
            return dict(self._usage_data) if self._usage_data else {}

    async def reload_from_disk(self) -> None:
        """
        Force reload usage data from disk.

        Useful when another process may have updated the file.
        """
        async with self._init_lock:
            self._initialized.clear()
            await self._load_usage()
            await self._reset_daily_stats_if_needed()
            self._initialized.set()

