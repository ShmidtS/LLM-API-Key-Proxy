# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""Environment variable parsing helpers."""

import os


def env_int(key: str, default: int) -> int:
    """Get an integer from an environment variable, falling back to default."""
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default
