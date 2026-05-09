# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import time
from pathlib import Path
from typing import Any, Dict, Optional


class UsageManagerStatisticsMixin:
    from ._statistics_formatters import get_stats_for_endpoint

