# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .client import RotatingClient
from .utils.json_utils import STREAM_DONE

# For type checkers (Pylint, mypy), import PROVIDER_PLUGINS statically
# At runtime, it's lazy-loaded via __getattr__
if TYPE_CHECKING:
    from .providers import PROVIDER_PLUGINS

__all__ = [
    "RotatingClient",
    "STREAM_DONE",
    "PROVIDER_PLUGINS",
]


_LAZY_IMPORTS = {
    "PROVIDER_PLUGINS": (".providers", "PROVIDER_PLUGINS"),
}


def __getattr__(name: str) -> Any:
    """Lazy-load heavy modules to speed up initial import."""
    entry = _LAZY_IMPORTS.get(name)
    if entry is not None:
        import importlib

        module_path, attr_name = entry
        module = importlib.import_module(module_path, __name__)
        value = module if attr_name is None else getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
