# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

"""Shared imports and constants for settings_dialogs package."""

import json
import orjson
import logging
import os
from typing import Dict, Any, Optional, List

from rich.console import Console
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.panel import Panel
from dotenv import set_key, unset_key

from rotator_library.utils.paths import get_data_file
from rotator_library.utils.terminal_utils import clear_screen

logger = logging.getLogger(__name__)
console = Console()

# Sentinel value for distinguishing "no pending change" from "pending change to None"
_NOT_FOUND = object()

from proxy_app._provider_settings import PROVIDER_SETTINGS_MAP  # noqa: F401
