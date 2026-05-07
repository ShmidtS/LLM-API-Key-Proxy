# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

"""Manages pending changes to .env"""

import logging
import os
from typing import Dict, Optional, List

from dotenv import load_dotenv
from dotenv import set_key, unset_key

from rotator_library.utils.paths import get_data_file

from proxy_app.settings_dialogs._common import _NOT_FOUND

logger = logging.getLogger(__name__)


class AdvancedSettings:
    """Manages pending changes to .env"""

    def __init__(self):
        self.env_file = get_data_file(".env")
        self.pending_changes = {}  # key -> value (None means delete)
        self.load_current_settings()

    def load_current_settings(self):
        """Load current .env values into env vars"""
        load_dotenv(self.env_file, override=True)

    def set(self, key: str, value: str):
        """Stage a change"""
        self.pending_changes[key] = value

    def remove(self, key: str):
        """Stage a removal"""
        self.pending_changes[key] = None

    def save(self):
        """Write pending changes to .env"""
        for key, value in self.pending_changes.items():
            if value is None:
                # Remove key
                unset_key(str(self.env_file), key)
            else:
                # Set key
                set_key(str(self.env_file), key, value)

        self.pending_changes.clear()
        self.load_current_settings()

    def discard(self):
        """Discard pending changes"""
        self.pending_changes.clear()

    def has_pending(self) -> bool:
        """Check if there are pending changes"""
        return bool(self.pending_changes)

    def get_pending_value(self, key: str):
        """Get pending value for a key. Returns sentinel _NOT_FOUND if no pending change."""
        return self.pending_changes.get(key, _NOT_FOUND)

    def get_original_value(self, key: str) -> Optional[str]:
        """Get the current .env value (before pending changes)"""
        return os.getenv(key)

    def get_change_type(self, key: str) -> Optional[str]:
        """Returns 'add', 'edit', 'remove', or None if no pending change"""
        if key not in self.pending_changes:
            return None
        if self.pending_changes[key] is None:
            return "remove"
        elif os.getenv(key) is not None:
            return "edit"
        else:
            return "add"

    def get_pending_keys_by_pattern(
        self, prefix: str = "", suffix: str = ""
    ) -> List[str]:
        """Get all pending change keys that match prefix and/or suffix"""
        return [
            k
            for k in self.pending_changes.keys()
            if k.startswith(prefix) and k.endswith(suffix)
        ]

    def get_changes_summary(self) -> Dict[str, List[tuple]]:
        """Get categorized summary of all pending changes.
        Returns dict with 'add', 'edit', 'remove' keys,
        each containing list of (key, old_val, new_val) tuples.
        """
        summary: Dict[str, List[tuple]] = {"add": [], "edit": [], "remove": []}
        for key, new_val in self.pending_changes.items():
            old_val = os.getenv(key)
            change_type = self.get_change_type(key)
            if change_type:
                summary[change_type].append((key, old_val, new_val))
        # Sort each list alphabetically by key
        for change_type in summary:
            summary[change_type].sort(key=lambda x: x[0])
        return summary

    def get_pending_counts(self) -> Dict[str, int]:
        """Get counts of pending changes by type"""
        adds = len(
            [
                k
                for k, v in self.pending_changes.items()
                if v is not None and os.getenv(k) is None
            ]
        )
        edits = len(
            [
                k
                for k, v in self.pending_changes.items()
                if v is not None and os.getenv(k) is not None
            ]
        )
        removes = len([k for k, v in self.pending_changes.items() if v is None])
        return {"add": adds, "edit": edits, "remove": removes}
