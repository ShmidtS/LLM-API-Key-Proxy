# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

"""Lifespan management for the FastAPI application.

Extracted from main.py to reduce file size and improve maintainability.
"""

import asyncio
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import orjson

from fastapi import FastAPI
from rotator_library import PROVIDER_PLUGINS, RotatingClient
from rotator_library.client._models import _MODEL_FETCH_BG_TIMEOUT
from rotator_library.credential_manager import CredentialManager
from rotator_library.dns_fix import close_doh_client, close_dns_executor
from rotator_library.model_info_service import init_model_info_service
from proxy_app.batch_manager import EmbeddingBatcher
from proxy_app.config import (
    DEFAULT_GLOBAL_TIMEOUT,
    MAX_GLOBAL_TIMEOUT,
    MIN_GLOBAL_TIMEOUT,
)

logger = logging.getLogger(__name__)



from ._lifecycle_impl import LifespanConfig, create_lifespan, suppress_connection_reset
