# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""Streaming retry execution."""

from __future__ import annotations

import asyncio
import codecs
import logging
import re
import time
from typing import Any, AsyncGenerator, Callable, Optional

import httpx
import litellm  # type: ignore[import-untyped]
from litellm.exceptions import APIConnectionError, APIError as LiteLLMAPIError, BadRequestError, InternalServerError, InvalidRequestError, RateLimitError, ServiceUnavailableError  # type: ignore[import-untyped]

from ..retry_base import HalfOpenSlot
from ...error_handler import (
    classify_error,
    get_retry_backoff,
    should_retry_same_key,
    should_rotate_on_error,
    validate_response_quality,
)
from ...error_types import (
    GarbageResponseError,
    NoAvailableKeysError,
    mask_credential,
)
from ...failure_logger import log_failure
from ...utils.chunk_aggregator import ChunkAggregator
from ...utils.json_utils import STREAM_DONE, json_loads, JSONDecodeError

lib_logger = logging.getLogger("rotator_library")


class StreamingRetryMixin:
    """Streaming retry execution."""

    def _parse_streaming_error_payload(self, original_exc):
        """Parse error JSON from streaming exception string representation.

        Returns (error_payload: dict, cleaned_str: str|None).
        """
        error_payload = {}
        cleaned_str = None
        try:
            json_str_match = re.search(
                r"(\{.*\})", str(original_exc), re.DOTALL
            )
            if json_str_match:
                cleaned_str = codecs.decode(
                    json_str_match.group(1), "unicode_escape"
                )
                error_payload = json_loads(cleaned_str)
        except (JSONDecodeError, TypeError):
            error_payload = {}
        return error_payload, cleaned_str

    def _extract_quota_info(self, error_details):
        """Extract quota value and ID from streaming error details.

        Returns (quota_value: str, quota_id: str).
        """
        quota_value = "N/A"
        quota_id = "N/A"
        if "details" in error_details and isinstance(
            error_details.get("details"), list
        ):
            for detail in error_details["details"]:
                if isinstance(detail.get("violations"), list):
                    for violation in detail["violations"]:
                        if "quotaValue" in violation:
                            quota_value = violation["quotaValue"]
                        if "quotaId" in violation:
                            quota_id = violation["quotaId"]
                        if quota_value != "N/A" and quota_id != "N/A":
                            break
        return quota_value, quota_id

    from ._streaming_orchestrator import _streaming_acompletion_with_retry

    async def _rate_limited_streaming(self, request, pre_request_callback=None, **kwargs):
        """Backpressure-gated entry point for streaming API calls.

        The semaphore is acquired per-attempt inside
        _streaming_acompletion_with_retry, not around the entire retry
        cycle, so backoff sleeps don't hold a semaphore slot.
        """
        async for chunk in self._streaming_acompletion_with_retry(
            request, pre_request_callback=pre_request_callback, **kwargs
        ):
            yield chunk

    async def _forced_streaming_acompletion(
        self,
        request: Optional[Any] = None,
        pre_request_callback: Optional[Callable] = None,
        **kwargs,
    ) -> Any:
        """
        Execute a streaming request internally but return a non-streaming ModelResponse.

        Used when a provider requires stream=true (e.g., Fireworks with max_tokens > 4096)
        but the client requested a non-streaming response. This method streams from the
        provider, collects all chunks, and assembles them into a single ModelResponse
        identical to what litellm.acompletion(stream=False) would return.
        """
        model = kwargs.get("model", "")
        provider = model.split("/")[0] if "/" in model else ""
        # Get the streaming generator from the normal streaming path
        stream_generator = self._streaming_acompletion_with_retry(
            request=request,
            pre_request_callback=pre_request_callback,
            **kwargs,
        )

        # Collect dict chunks and assemble into a non-streaming response
        aggregator = ChunkAggregator(model=model, llm_provider=provider)

        async for chunk in stream_generator:
            # STREAM_DONE sentinel: stream is complete
            if chunk is STREAM_DONE:
                break

            if not isinstance(chunk, dict):
                continue

            aggregator.check_error_payload(chunk)
            aggregator.add_chunk(chunk)

        model_response = aggregator.build_model_response(model=model)

        lib_logger.debug(
            "Forced streaming completion assembled: model=%s, "
            "finish_reason=%s, usage=%s",
            aggregator.first_chunk_meta.get("model") if aggregator.first_chunk_meta else None,
            aggregator.finish_reason,
            aggregator.usage_data,
        )

        try:
            validate_response_quality(model_response, provider=provider, model=model)
        except GarbageResponseError as exc:
            lib_logger.warning(
                "Garbage response detected for %s/%s, rotating to next credential: %s",
                provider, model, exc.message if hasattr(exc, 'message') else exc,
            )
            raise

        return model_response
