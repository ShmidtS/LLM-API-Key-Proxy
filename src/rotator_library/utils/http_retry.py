# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""HTTP retry utilities with exponential backoff."""

import asyncio
import random
from typing import Optional


async def exponential_backoff_with_jitter(
    attempt: int,
    base: float = 2.0,
    max_wait: float = 60.0,
    jitter: float = 0.1,
    retry_after: Optional[float] = None,
) -> None:
    """
    Exponential backoff with optional jitter and retry-after.

    Args:
        attempt: Current attempt number (0-indexed)
        base: Base for exponential calculation
        max_wait: Maximum wait time in seconds
        jitter: Jitter factor (0.0-1.0)
        retry_after: Optional override wait time from server

    Example:
        for attempt in range(max_retries):
            try:
                return await make_request()
            except RetryableError:
                await exponential_backoff_with_jitter(attempt)
    """
    if retry_after is not None:
        wait_time = retry_after
    else:
        wait_time = min(base**attempt, max_wait)

    if jitter > 0:
        jitter_amount = random.uniform(0, wait_time * jitter)
        wait_time += jitter_amount

    await asyncio.sleep(wait_time)
