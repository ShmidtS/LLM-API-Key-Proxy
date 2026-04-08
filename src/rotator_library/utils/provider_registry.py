# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/utils/provider_registry.py
"""
Shared provider instance registry (singleton).

Eliminates the duplicate ``_provider_instances`` dicts maintained
independently by ``client.py`` and ``usage_manager.py``.  Both
modules now delegate to this single registry so that a provider
is instantiated exactly once and shared across the process.

Thread-safe via double-checked locking with ``threading.Lock``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

lib_logger = logging.getLogger("rotator_library")


class ProviderRegistry:
    """
    Global registry of instantiated provider objects.

    Providers are lazily created on first access and then cached.
    The registry is process-wide (singleton) so that ``client.py``
    and ``usage_manager.py`` share the same instances.

    Usage::

        from .utils.provider_registry import get_provider_registry

        registry = get_provider_registry()
        instance = registry.get_or_create("antigravity", plugin_class)
    """

    def __init__(self) -> None:
        self._instances: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def get_or_create(
        self,
        provider_name: str,
        plugin_entry: Any = None,
    ) -> Optional[Any]:
        """
        Get a cached provider instance or create one from *plugin_entry*.

        Args:
            provider_name: Provider identifier (e.g. ``"antigravity"``).
            plugin_entry: A class to instantiate, an already-created
                          instance to store, or ``None`` to just look up.

        Returns:
            The provider instance, or ``None`` if not cached and
            *plugin_entry* is ``None``.
        """
        # Fast path — already cached (no lock needed for read under GIL)
        if provider_name in self._instances:
            return self._instances[provider_name]

        if plugin_entry is None:
            return None

        # Slow path — create under lock
        with self._lock:
            # Double-check after acquiring lock
            if provider_name in self._instances:
                return self._instances[provider_name]

            if isinstance(plugin_entry, type):
                instance = plugin_entry()
            else:
                instance = plugin_entry

            self._instances[provider_name] = instance
            return instance

    def get(self, provider_name: str) -> Optional[Any]:
        """Return cached instance or ``None`` without creating."""
        return self._instances.get(provider_name)

    def register(self, provider_name: str, instance: Any) -> None:
        """
        Explicitly register an already-created instance.

        Thread-safe; overwrites any existing entry for *provider_name*.
        """
        with self._lock:
            self._instances[provider_name] = instance

    def __contains__(self, provider_name: str) -> bool:
        return provider_name in self._instances

    def contains(self, provider_name: str) -> bool:
        """Check whether a provider is already registered."""
        return provider_name in self._instances

    def clear(self) -> None:
        """Remove all cached provider instances."""
        with self._lock:
            self._instances.clear()

    def keys(self) -> list:
        """Return a snapshot of registered provider names."""
        return list(self._instances.keys())

    def items(self) -> list:
        """Return a snapshot of ``(name, instance)`` pairs."""
        return list(self._instances.items())


# ---------------------------------------------------------------------------
# Singleton accessor (thread-safe, double-checked locking)
# ---------------------------------------------------------------------------

_REGISTRY_INSTANCE: Optional[ProviderRegistry] = None
_REGISTRY_LOCK = threading.Lock()


def get_provider_registry() -> ProviderRegistry:
    """Return the global ``ProviderRegistry`` singleton (thread-safe)."""
    global _REGISTRY_INSTANCE
    if _REGISTRY_INSTANCE is None:
        with _REGISTRY_LOCK:
            if _REGISTRY_INSTANCE is None:
                _REGISTRY_INSTANCE = ProviderRegistry()
    return _REGISTRY_INSTANCE
