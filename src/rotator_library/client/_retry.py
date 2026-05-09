# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""Compatibility imports for retry mixins."""

from __future__ import annotations

from ..error_handler import (
    classify_error,
    get_retry_backoff,
    should_retry_same_key,
    should_rotate_on_error,
    validate_response_quality,
)
from ..failure_logger import log_failure
from .retry.context_builder import RetryContextBuilderMixin
from .retry.mixins import RetryMixin
from .retry.non_streaming import NonStreamingRetryMixin
from .retry.streaming import StreamingRetryMixin

__all__ = [
    "NonStreamingRetryMixin",
    "RetryContextBuilderMixin",
    "RetryMixin",
    "StreamingRetryMixin",
    "classify_error",
    "log_failure",
    "should_retry_same_key",
    "should_rotate_on_error",
]
