# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

from pathlib import Path

_src_pkg = Path(__file__).resolve().parents[1] / "src" / "proxy_app"
if _src_pkg.is_dir():
    __path__.append(str(_src_pkg))
