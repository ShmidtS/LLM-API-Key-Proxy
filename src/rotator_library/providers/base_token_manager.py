"""
Base Token Manager - Common state infrastructure for token refresh queue management.

Shared by GoogleOAuthBase, QwenAuthBase, IFlowAuthBase.
Contains only the __init__ state initialization that is identical across all.
"""

import asyncio
import threading
from typing import Any, Dict, Optional

from ..utils.ttl_dict import TTLDict


class BaseTokenManager:
    """Common state and infrastructure for token refresh queue management.

    Shared by GoogleOAuthBase, QwenAuthBase, IFlowAuthBase.
    Contains only the __init__ state initialization that is identical across all.
    """

    def __init__(self):
        # Cache and lock management
        self._credentials_cache: TTLDict = TTLDict(maxsize=200, default_ttl=3600.0)
        self._refresh_locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock = threading.Lock()  # Thread-safe, non-async for dict access

        # Refresh failure tracking
        self._refresh_failures: Dict[str, int] = {}
        self._next_refresh_after: Dict[str, float] = {}

        # Queue infrastructure
        self._refresh_queue: asyncio.Queue = asyncio.Queue()
        self._queue_processor_task: Optional[asyncio.Task] = None
        self._reauth_queue: asyncio.Queue = asyncio.Queue()
        self._reauth_processor_task: Optional[asyncio.Task] = None

        # Queue deduplication and unavailability tracking
        self._queued_credentials: set = set()
        self._unavailable_credentials: Dict[str, float] = {}
        self._unavailable_ttl_seconds: int = 360
        self._queue_tracking_lock = asyncio.Lock()
        self._queue_retry_count: Dict[str, int] = {}

        # Timing constants
        self._refresh_timeout_seconds: int = 15
        self._refresh_interval_seconds: int = 30
        self._refresh_max_retries: int = 3
        self._reauth_timeout_seconds: int = 300
