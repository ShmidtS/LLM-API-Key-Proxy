# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

from pathlib import Path

_src_pkg = Path(__file__).resolve().parents[1] / "src" / "rotator_library"
if _src_pkg.is_dir():
    __path__.append(str(_src_pkg))

__all__ = ["RotatingClient", "STREAM_DONE", "PROVIDER_PLUGINS"]


_LAZY_IMPORTS = {
    "RotatingClient": (".client", "RotatingClient"),
    "STREAM_DONE": (".utils.json_utils", "STREAM_DONE"),
    "PROVIDER_PLUGINS": (".providers", "PROVIDER_PLUGINS"),
}


def __getattr__(name: str):
    entry = _LAZY_IMPORTS.get(name)
    if entry is not None:
        import importlib

        module_path, attr_name = entry
        module = importlib.import_module(module_path, __name__)
        value = module if attr_name is None else getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
