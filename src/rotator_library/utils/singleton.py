# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# src/rotator_library/utils/singleton.py
"""
Unified singleton metaclass for thread-safe, reset-aware singletons.

Replaces the three inconsistent patterns previously used across the codebase:
- ``__new__`` override (ModelDefinitions, ReauthCoordinator)
- Module-level global + factory function (HttpClientPool, CredentialWeightCache)
- Class-level ``_instance`` + ``get_instance()`` classmethod (BufferedWriteRegistry)

All singletons now share a single pattern with consistent thread-safety
and ``reset()`` support for config reload scenarios.
"""

import threading
from typing import Any, Dict, Optional, Type, TypeVar

T = TypeVar("T", bound=object)


class SingletonMeta(type):
    """
    Thread-safe singleton metaclass with reset support.

    Usage::

        class MyClass(metaclass=SingletonMeta):
            def __init__(self, value: str = "default"):
                self.value = value

        # First call creates the instance
        obj = MyClass()
        # Subsequent calls return the same instance
        obj2 = MyClass()
        assert obj is obj2

        # Reset for config reload
        MyClass.reset()

    Features:
    - Thread-safe via ``threading.Lock``
    - ``__init__`` only runs once (on first instantiation)
    - ``reset()`` classmethod clears the singleton for re-creation
    - ``_instance`` class attribute for introspection
    - ``is_initialized()`` classmethod to check if singleton exists
    """

    _instances: Dict[Any, Any] = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        # Fast path: instance already exists
        if cls in cls._instances:
            return cls._instances[cls]

        # Slow path: acquire lock and double-check
        with cls._lock:
            if cls in cls._instances:
                return cls._instances[cls]

            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
            return instance

    def reset(cls) -> None:
        """
        Reset the singleton instance.

        The next call to ``cls()`` will create a fresh instance.
        Thread-safe: acquires the class-level lock.
        """
        with cls._lock:
            cls._instances.pop(cls, None)

    def is_initialized(cls) -> bool:
        """Check if the singleton instance has been created."""
        return cls in cls._instances

    def get_instance(cls) -> Optional[Any]:
        """Get the current instance without creating one. Returns None if not initialized."""
        return cls._instances.get(cls)
