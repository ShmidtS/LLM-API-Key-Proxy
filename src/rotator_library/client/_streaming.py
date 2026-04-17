# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""Streaming mixin for RotatingClient — safe streaming wrapper and
transaction logging stream wrapper."""

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, Optional

import httpx

from ..error_types import mask_credential, NoAvailableKeysError
from ..utils.json_utils import STREAM_DONE

lib_logger = logging.getLogger("rotator_library")

_MAX_LOGGED_CHUNKS = 10000
_MAX_LOGGED_BYTES = 1_048_576  # 1 MB


class _StreamedException(Exception):
    """Internal exception to break out of the streaming while-loop on error."""
    pass


class StreamingMixin:
    """Mixin with streaming wrapper methods for RotatingClient."""

    async def _safe_streaming_wrapper(
        self,
        stream: Any,
        key: str,
        model: str,
        request: Optional[Any] = None,
        provider_plugin: Optional[Any] = None,
    ) -> AsyncGenerator[Any, None]:
        """
        A hybrid wrapper for streaming that buffers fragmented JSON, handles client disconnections gracefully,
        and distinguishes between content and streamed errors.

        FINISH_REASON HANDLING:
        Providers just translate chunks - this wrapper handles ALL finish_reason logic:
        1. Strip finish_reason from intermediate chunks (litellm defaults to "stop")
        2. Track accumulated_finish_reason with priority: tool_calls > length/content_filter > stop
        3. Only emit finish_reason on final chunk (detected by usage.completion_tokens > 0)
        """
        stream_completed = False
        stream_iterator = stream.__aiter__()
        json_buffer_parts: list[str] = []  # O(n) accumulation via list
        accumulated_finish_reason = None  # Track strongest finish_reason across chunks
        has_tool_calls = False  # Track if ANY tool calls were seen in stream
        chunk_index = 0  # Track chunk count for better error logging
        has_usage_data = False  # Track if we ever saw usage data

        try:
            while True:
                # Check disconnection every 200 chunks to avoid per-chunk syscall overhead
                if chunk_index % 200 == 0 and request and await request.is_disconnected():
                    lib_logger.info(
                        "Client disconnected. Aborting stream for credential %s.",
                        mask_credential(key),
                    )
                    break

                try:
                    chunk = await stream_iterator.__anext__()
                    chunk_index += 1
                    if json_buffer_parts:
                        lib_logger.warning(
                            "Discarding incomplete JSON buffer from previous chunk: %s",
                            ''.join(json_buffer_parts),
                        )
                        json_buffer_parts.clear()

                    # Convert chunk to dict, handling both litellm.ModelResponse and raw dicts
                    # Per-chunk error isolation: malformed chunks are skipped, not fatal
                    try:
                        if hasattr(chunk, "model_dump"):
                            chunk_dict = chunk.model_dump()
                        else:
                            chunk_dict = chunk
                    except (KeyError, TypeError) as e:
                        lib_logger.warning(
                            "Skipping malformed chunk at index %s for model %s: %s",
                            chunk_index, model, e,
                        )
                        continue

                    # === FINISH_REASON LOGIC ===
                    # Providers send raw chunks without finish_reason logic.
                    # This wrapper determines finish_reason based on accumulated state.
                    if "choices" in chunk_dict and chunk_dict["choices"]:
                        choice = chunk_dict["choices"][0]
                        delta = choice.get("delta", {})
                        usage = chunk_dict.get("usage", {})

                        # Check if we have usage data
                        if (
                            usage
                            and isinstance(usage, dict)
                            and usage.get("completion_tokens")
                        ):
                            has_usage_data = True

                        # Track tool_calls across ALL chunks - if we ever see one, finish_reason must be tool_calls
                        if delta.get("tool_calls"):
                            has_tool_calls = True
                            accumulated_finish_reason = "tool_calls"

                        # === STREAM ABORT DETECTION ===
                        # Check for provider abort (finish_reason='error' or native_finish_reason='abort')
                        raw_finish_reason = choice.get("finish_reason")
                        native_finish_reason = chunk_dict.get("native_finish_reason")
                        if (
                            raw_finish_reason == "error"
                            or native_finish_reason == "abort"
                        ):
                            lib_logger.warning(
                                "Stream abort detected for model %s at chunk %s. "
                                "finish_reason=%s, native_finish_reason=%s",
                                model, chunk_index, raw_finish_reason, native_finish_reason,
                            )
                            raise _StreamedException(
                                "Provider aborted stream mid-generation"
                            )

                        # Strip litellm's default finish_reason from intermediate chunks
                        # (litellm sets finish_reason="stop" on ALL chunks, not just the last)
                        if raw_finish_reason and raw_finish_reason != "error":
                            # Update accumulated reason with priority: tool_calls > length/content_filter > stop
                            if raw_finish_reason in ("tool_calls", "function_call"):
                                accumulated_finish_reason = raw_finish_reason
                                has_tool_calls = True
                            elif raw_finish_reason in ("length", "content_filter"):
                                # length/content_filter override stop, but not tool_calls
                                if not has_tool_calls:
                                    accumulated_finish_reason = raw_finish_reason
                            elif raw_finish_reason == "stop":
                                # stop is lowest priority - only set if nothing else
                                if accumulated_finish_reason is None:
                                    accumulated_finish_reason = "stop"

                            # Strip from intermediate chunks
                            choice["finish_reason"] = None

                        # Emit finish_reason ONLY on the final chunk (has usage data)
                        if has_usage_data and accumulated_finish_reason:
                            choice["finish_reason"] = accumulated_finish_reason

                        # Handle tool_calls in delta for proper finish_reason
                        if delta.get("tool_calls"):
                            has_tool_calls = True
                            accumulated_finish_reason = "tool_calls"

                    yield chunk_dict

                except StopAsyncIteration:
                    # Stream ended normally
                    stream_completed = True
                    break

                except _StreamedException:
                    # Provider aborted - re-raise to retry handler
                    stream_completed = False
                    raise

                except (httpx.ReadTimeout, httpx.PoolTimeout) as e:
                    # Timeout errors are retriable — re-raise so upstream retry logic can act
                    lib_logger.warning(
                        "Timeout during streaming for model %s, "
                        "credential %s, chunk %s: %s",
                        model, mask_credential(key), chunk_index, e,
                    )
                    stream_completed = False
                    raise

                except (httpx.RemoteProtocolError, httpx.ConnectError) as e:
                    # Protocol/connection errors are retriable — re-raise
                    lib_logger.warning(
                        "Protocol/connection error during streaming for model %s, "
                        "credential %s, chunk %s: %s",
                        model, mask_credential(key), chunk_index, e,
                    )
                    stream_completed = False
                    raise

                except (asyncio.CancelledError, MemoryError, SystemExit):
                    stream_completed = False
                    raise

                except Exception as e:
                    # Check if this is a JSON decode error from fragmented chunks
                    error_str = str(e)
                    if "Expecting value" in error_str or "Unterminated string" in error_str:
                        # Buffer the raw chunk data for re-assembly, not the error string
                        raw = chunk if isinstance(chunk, (str, bytes)) else str(chunk)
                        json_buffer_parts.append(raw)
                        continue

                    # For other errors during iteration, log and break
                    lib_logger.error(
                        "Error during streaming for model %s, "
                        "credential %s, chunk %s: %s",
                        model, mask_credential(key), chunk_index, e,
                    )
                    stream_completed = False
                    break

        except _StreamedException:
            # Re-raise provider aborts so retry logic can handle them
            raise

        except NoAvailableKeysError:
            # Re-raise key exhaustion so retry logic can try next key
            raise

        except Exception as e:
            # Catch any unexpected errors in the wrapper itself
            lib_logger.error(
                "Unexpected error in streaming wrapper for model %s: %s",
                model, e,
            )
            # Try to emit an error chunk to the client
            error_data = {
                "error": {
                    "message": f"Stream interrupted: {str(e)}",
                    "type": "proxy_stream_error",
                }
            }
            yield error_data

        finally:
            try:
                # Always emit STREAM_DONE if stream completed and client is still connected
                # This prevents sending [DONE] to a disconnected client or after an error.
                if stream_completed and (
                    not request or not await request.is_disconnected()
                ):
                    yield STREAM_DONE
            except Exception as e:
                lib_logger.exception("Error during stream cleanup: %s", e)

    async def _transaction_logging_stream_wrapper(
        self,
        stream: Any,
        transaction_logger: Optional[Any],
        request_data: Dict[str, Any],
    ) -> Any:
        """
        Wrap a stream to log chunks and final response to TransactionLogger.

        This wrapper:
        1. Yields chunks unchanged (passthrough) - dicts flow through without re-parse
        2. Logs dict chunks via transaction_logger.log_stream_chunk()
        3. Collects chunks for final response assembly
        4. After stream ends, assembles and logs final response

        Args:
            stream: The streaming generator (yields dicts or STREAM_DONE sentinel)
            transaction_logger: Optional TransactionLogger instance
            request_data: Original request data for context
        """
        from ..transaction_logger import TransactionLogger

        MAX_LOGGED_CHUNKS = _MAX_LOGGED_CHUNKS
        MAX_LOGGED_BYTES = _MAX_LOGGED_BYTES
        chunks = []
        total_bytes = 0
        try:
            async for chunk in stream:
                yield chunk

                # Log chunk if logging enabled (chunk is now a dict, no re-parse needed)
                if (
                    transaction_logger
                    and isinstance(chunk, dict)
                ):
                    if len(chunks) < MAX_LOGGED_CHUNKS and total_bytes < MAX_LOGGED_BYTES:
                        chunks.append(chunk)
                        total_bytes += len(str(chunk))
                    await transaction_logger.log_stream_chunk(chunk)
        finally:
            # Assemble and log final response after stream ends
            if transaction_logger and chunks:
                try:
                    final_response = TransactionLogger.assemble_streaming_response(
                        chunks, request_data
                    )
                    await transaction_logger.log_response(final_response)
                except Exception as e:
                    lib_logger.warning(
                        "TransactionLogger: Failed to assemble/log final response: %s",
                        e,
                    )
