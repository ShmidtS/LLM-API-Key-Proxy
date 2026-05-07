# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# src/rotator_library/credential_io.py

import orjson
from pathlib import Path

from .utils.json_utils import json_loads
from .utils.paths import get_oauth_dir, get_data_file


def _get_oauth_base_dir() -> Path:
    """Get the OAuth base directory (lazy, respects EXE vs script mode)."""
    oauth_dir = get_oauth_dir()
    oauth_dir.mkdir(parents=True, exist_ok=True)
    return oauth_dir


def _get_env_file() -> Path:
    """Get the .env file path (lazy, respects EXE vs script mode)."""
    return get_data_file(".env")


def _read_text_file(path: Path | str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_text_file(path: Path | str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _read_json_file(path: Path | str) -> dict:
    return json_loads(_read_text_file(path))


def _write_json_file(path: Path | str, data: dict) -> None:
    _write_text_file(path, orjson.dumps(data, option=orjson.OPT_INDENT_2).decode("utf-8"))
