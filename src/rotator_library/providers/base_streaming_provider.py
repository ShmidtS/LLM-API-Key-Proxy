# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Base mixin classes for shared streaming and quota-refresh logic
across multiple API providers.
"""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

import orjson
from rotator_library.utils.json_utils import json_deep_copy
import litellm

lib_logger = logging.getLogger("rotator_library")


async def parse_sse_stream(
    response: Any,
    provider_name: str = "",
    on_line: Optional[Callable[[str], None]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Async generator that parses SSE (Server-Sent Events) lines from an
    httpx streaming response and yields decoded JSON chunks.

    Handles both ``data: ...`` (with space) and ``data:...`` (without space)
    prefixes.  Terminates on ``[DONE]`` sentinel.

    Args:
        response: An httpx streaming response (already inside ``async with``).
        provider_name: Name of the calling provider for log messages.
        on_line: Optional callback invoked for every raw line (useful for
                 transaction logging, e.g. ``file_logger.log_response_chunk``).

    Yields:
        Parsed JSON dicts from each SSE ``data`` line.
    """
    async for line in response.aiter_lines():
        if on_line is not None:
            on_line(line)

        # Skip non-data lines (event:, comments, empty lines)
        if not line.startswith("data:"):
            continue

        # Extract data after "data:" prefix, handling both "data: " and "data:"
        if line.startswith("data: "):
            data_str = line[6:]
        else:
            data_str = line[5:]

        # Check for [DONE] sentinel (with optional surrounding whitespace)
        if data_str.strip() == "[DONE]":
            break

        try:
            chunk = orjson.loads(data_str)
            yield chunk
        except (orjson.JSONDecodeError, json.JSONDecodeError):
            name = provider_name or "unknown"
            lib_logger.warning(
                f"Could not decode JSON from {name}: {line}"
            )


class StreamingResponseMixin:
    """
    Mixin providing shared streaming response utilities.

    Used by providers that implement custom streaming (Qwen, iFlow, Gemini CLI):
      - _stream_to_completion_response()
      - _clean_tool_schemas()
      - _clean_schema_properties()
    """

    # -------------------------------------------------------------------------
    # Tool schema cleaning
    # -------------------------------------------------------------------------

    def _clean_tool_schemas(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Removes unsupported properties from tool schemas to prevent API errors.
        """
        cleaned_tools = []

        for tool in tools:
            # Shallow copy top-level dict, then deep-copy only the "function" value
            # if it needs mutation. Avoids full orjson round-trip serialization.
            cleaned_tool = dict(tool)

            if "function" in cleaned_tool:
                func = json_deep_copy(cleaned_tool["function"])
                cleaned_tool["function"] = func

                # Remove strict mode (not supported by most providers)
                func.pop("strict", None)

                # Clean parameter schema if present
                if "parameters" in func and isinstance(func["parameters"], dict):
                    params = func["parameters"]

                    # Remove additionalProperties if present
                    params.pop("additionalProperties", None)

                    # Recursively clean nested properties
                    if "properties" in params:
                        self._clean_schema_properties(params["properties"])

            cleaned_tools.append(cleaned_tool)

        return cleaned_tools

    def _clean_schema_properties(self, properties: Dict[str, Any]) -> None:
        """Recursively cleans schema properties."""
        for prop_name, prop_schema in properties.items():
            if isinstance(prop_schema, dict):
                # Remove unsupported fields
                prop_schema.pop("strict", None)
                prop_schema.pop("additionalProperties", None)

                # Recurse into nested properties
                if "properties" in prop_schema:
                    self._clean_schema_properties(prop_schema["properties"])

                # Recurse into array items
                if "items" in prop_schema and isinstance(prop_schema["items"], dict):
                    self._clean_schema_properties({"item": prop_schema["items"]})

    # -------------------------------------------------------------------------
    # Stream-to-completion reassembly
    # -------------------------------------------------------------------------

    def _stream_to_completion_response(
        self, chunks: List[litellm.ModelResponse]
    ) -> litellm.ModelResponse:
        """
        Manually reassembles streaming chunks into a complete response.

        Key behavior:
        - Determines finish_reason based on accumulated state (tool_calls vs stop)
        - Properly initializes tool_calls with type field
        - Handles usage data extraction from chunks
        """
        if not chunks:
            raise ValueError("No chunks provided for reassembly")

        # Initialize the final response structure
        final_message: Dict[str, Any] = {"role": "assistant"}
        aggregated_tool_calls: Dict[int, Dict[str, Any]] = {}
        usage_data = None
        chunk_finish_reason = None

        # Get the first chunk for basic response metadata
        first_chunk = chunks[0]

        # Process each chunk to aggregate content
        for chunk in chunks:
            if not hasattr(chunk, "choices") or not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.get("delta", {})

            # Aggregate content
            if "content" in delta and delta["content"] is not None:
                if "content" not in final_message:
                    final_message["content"] = ""
                final_message["content"] += delta["content"]

            # Aggregate reasoning content
            if "reasoning_content" in delta and delta["reasoning_content"] is not None:
                if "reasoning_content" not in final_message:
                    final_message["reasoning_content"] = ""
                final_message["reasoning_content"] += delta["reasoning_content"]

            # Aggregate tool calls with proper initialization
            if "tool_calls" in delta and delta["tool_calls"]:
                for tc_chunk in delta["tool_calls"]:
                    index = tc_chunk.get("index", 0)
                    if index not in aggregated_tool_calls:
                        # Initialize with type field for OpenAI compatibility
                        aggregated_tool_calls[index] = {
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if "id" in tc_chunk:
                        aggregated_tool_calls[index]["id"] = tc_chunk["id"]
                    if "type" in tc_chunk:
                        aggregated_tool_calls[index]["type"] = tc_chunk["type"]
                    if "function" in tc_chunk:
                        if (
                            "name" in tc_chunk["function"]
                            and tc_chunk["function"]["name"] is not None
                        ):
                            tc_fn = aggregated_tool_calls[index]["function"]
                            if "_name_chunks" not in tc_fn:
                                tc_fn["_name_chunks"] = []
                            tc_fn["_name_chunks"].append(tc_chunk["function"]["name"])
                        if (
                            "arguments" in tc_chunk["function"]
                            and tc_chunk["function"]["arguments"] is not None
                        ):
                            tc_fn = aggregated_tool_calls[index]["function"]
                            if "_args_chunks" not in tc_fn:
                                tc_fn["_args_chunks"] = []
                            tc_fn["_args_chunks"].append(
                                tc_chunk["function"]["arguments"]
                            )

            # Aggregate function calls (legacy format)
            if "function_call" in delta and delta["function_call"] is not None:
                if "function_call" not in final_message:
                    final_message["function_call"] = {"_name_chunks": [], "_args_chunks": []}
                if (
                    "name" in delta["function_call"]
                    and delta["function_call"]["name"] is not None
                ):
                    final_message["function_call"]["_name_chunks"].append(
                        delta["function_call"]["name"]
                    )
                if (
                    "arguments" in delta["function_call"]
                    and delta["function_call"]["arguments"] is not None
                ):
                    final_message["function_call"]["_args_chunks"].append(
                        delta["function_call"]["arguments"]
                    )

            # Track finish_reason from chunks
            if choice.get("finish_reason"):
                chunk_finish_reason = choice["finish_reason"]

        # Handle usage data from the last chunk that has it
        for chunk in reversed(chunks):
            if hasattr(chunk, "usage") and chunk.usage:
                usage_data = chunk.usage
                break

        # Add tool calls to final message if any
        if aggregated_tool_calls:
            for tc_index, tc in aggregated_tool_calls.items():
                fn = tc["function"]
                if "_name_chunks" in fn:
                    fn["name"] = "".join(fn.pop("_name_chunks"))
                if "_args_chunks" in fn:
                    fn["arguments"] = "".join(fn.pop("_args_chunks"))
            final_message["tool_calls"] = list(aggregated_tool_calls.values())

        # Resolve function_call chunks
        if "function_call" in final_message:
            fc = final_message["function_call"]
            if "_name_chunks" in fc:
                fc["name"] = "".join(fc.pop("_name_chunks"))
            if "_args_chunks" in fc:
                fc["arguments"] = "".join(fc.pop("_args_chunks"))

        # Ensure standard fields are present for consistent logging
        for field in ["content", "tool_calls", "function_call"]:
            if field not in final_message:
                final_message[field] = None

        # Determine finish_reason based on accumulated state
        # Priority: tool_calls wins if present, then chunk's finish_reason, then default to "stop"
        if aggregated_tool_calls:
            finish_reason = "tool_calls"
        elif chunk_finish_reason:
            finish_reason = chunk_finish_reason
        else:
            finish_reason = "stop"

        # Construct the final response
        final_choice = {
            "index": 0,
            "message": final_message,
            "finish_reason": finish_reason,
        }

        # Create the final ModelResponse
        final_response_data = {
            "id": first_chunk.id,
            "object": "chat.completion",
            "created": first_chunk.created,
            "model": first_chunk.model,
            "choices": [final_choice],
            "usage": usage_data,
        }

        return litellm.ModelResponse(**final_response_data)


class QuotaRefreshMixin:
    """
    Mixin providing shared background job boilerplate for quota refresh.

    Subclasses must define:
      - _virtual_model_name  (str): e.g. "chutes/_quota"
      - _quota_cache         (dict): credential -> usage_data cache
      - fetch_quota_usage(api_key, client) -> dict: provider-specific API call
      - provider_name         (str): for logging

    Optional:
      - _include_max_requests: whether to pass max_requests (default True)
    """

    _virtual_model_name: str = ""
    _quota_cache: Dict[str, Dict[str, Any]] = {}
    provider_name: str = ""
    _include_max_requests: bool = True

    async def run_background_job(
        self,
        usage_manager: Any,
        credentials: List[str],
        quota_fetch_concurrency: int = 5,
        get_http_pool_fn=None,
    ) -> None:
        """
        Refresh quota usage for all credentials in parallel.

        Args:
            usage_manager: UsageManager instance
            credentials: List of API keys / credential paths
            quota_fetch_concurrency: Max concurrent quota fetch tasks
            get_http_pool_fn: Callable to get HTTP pool (injected for flexibility)
        """

        async def refresh_single_credential(
            api_key: str, client: Any
        ) -> None:
            async with semaphore:
                try:
                    usage_data = await self.fetch_quota_usage(api_key, client)

                    if usage_data.get("status") == "success":
                        # Update quota cache
                        self._quota_cache[api_key] = usage_data

                        # Calculate values for usage manager
                        remaining_fraction = usage_data.get("remaining_fraction", 0.0)
                        reset_ts = usage_data.get("reset_at")

                        baseline_kwargs: Dict[str, Any] = {
                            "remaining_fraction": remaining_fraction,
                            "reset_timestamp": reset_ts,
                        }
                        if self._include_max_requests:
                            quota = usage_data.get("quota", 0)
                            baseline_kwargs["max_requests"] = quota

                        # Store baseline in usage manager
                        await usage_manager.update_quota_baseline(
                            api_key,
                            self._virtual_model_name,
                            **baseline_kwargs,
                        )

                        lib_logger.debug(
                            f"Updated {self.provider_name} quota baseline for credential: "
                            f"{usage_data.get('remaining', 0):.0f}/{usage_data.get('quota', 0)} remaining "
                            f"({remaining_fraction * 100:.0f}%)"
                        )
                    elif usage_data.get("status") == "transient_error" or usage_data.get("remaining_fraction") is None:
                        lib_logger.warning(
                            f"Transient error refreshing {self.provider_name} quota for credential "
                            f"(error: {usage_data.get('error')}), preserving previous baseline"
                        )
                    else:
                        self._quota_cache[api_key] = usage_data
                        await usage_manager.update_quota_baseline(
                            api_key,
                            self._virtual_model_name,
                            remaining_fraction=0.0,
                            reset_timestamp=usage_data.get("reset_at"),
                        )
                        lib_logger.warning(
                            f"Failed to refresh {self.provider_name} quota for credential "
                            f"(error: {usage_data.get('error')}), marking as exhausted"
                        )

                except Exception as e:
                    lib_logger.warning(
                        f"Failed to refresh {self.provider_name} quota usage: {e}"
                    )

        # Fetch all credentials in parallel with shared HTTP client
        if get_http_pool_fn is None:
            from ..http_client_pool import get_http_pool
            get_http_pool_fn = get_http_pool

        semaphore = asyncio.Semaphore(quota_fetch_concurrency)
        pool = await get_http_pool_fn()
        client = await pool.get_client_async()
        tasks = [
            refresh_single_credential(api_key, client) for api_key in credentials
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
