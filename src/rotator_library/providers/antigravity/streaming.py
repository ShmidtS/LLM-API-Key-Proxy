# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""Streaming mixin for AntigravityProvider."""

import asyncio
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
import litellm
import json
from ...utils.json_utils import json_loads
from ...error_handler import EmptyResponseError, TransientQuotaError
from ...timeout_config import TimeoutConfig
from ...transaction_logger import AntigravityProviderLogger
from .constants import (
    EMPTY_RESPONSE_MAX_ATTEMPTS,
    EMPTY_RESPONSE_RETRY_DELAY,
    MALFORMED_CALL_MAX_RETRIES,
    MALFORMED_CALL_RETRY_DELAY,
    _MalformedFunctionCallDetected,
    _generate_request_id,
    lib_logger,
)
from ..provider_interface import build_bearer_headers


class AntigravityStreamingMixin:
    """Mixin providing streaming response handling methods."""

    async def _collect_streaming_chunks(
        self,
        streaming_generator: AsyncGenerator[litellm.ModelResponse, None],
        model: str,
        file_logger: Optional["AntigravityProviderLogger"] = None,
    ) -> litellm.ModelResponse:
        """
        Collect all chunks from a streaming generator into a single non-streaming
        ModelResponse. Used when client requests stream=False.
        """
        content_chunks: List[str] = []
        reasoning_chunks: List[str] = []
        collected_tool_calls: List[Dict[str, Any]] = []
        last_chunk = None
        usage_info = None

        async for chunk in streaming_generator:
            last_chunk = chunk
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                # delta can be a dict or a Delta object depending on litellm version
                if isinstance(delta, dict):
                    # Handle as dict
                    if delta.get("content"):
                        content_chunks.append(delta["content"])
                    if delta.get("reasoning_content"):
                        reasoning_chunks.append(delta["reasoning_content"])
                    if delta.get("tool_calls"):
                        for tc in delta["tool_calls"]:
                            self._accumulate_tool_call(tc, collected_tool_calls)
                else:
                    # Handle as object with attributes
                    if hasattr(delta, "content") and delta.content:
                        content_chunks.append(delta.content)
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        reasoning_chunks.append(delta.reasoning_content)
                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        for tc in delta.tool_calls:
                            self._accumulate_tool_call(tc, collected_tool_calls)
            if hasattr(chunk, "usage") and chunk.usage:
                usage_info = chunk.usage

        # Build final non-streaming response
        finish_reason = "stop"
        if last_chunk and hasattr(last_chunk, "choices") and last_chunk.choices:
            finish_reason = last_chunk.choices[0].finish_reason or "stop"

        # Resolve chunk-accumulated args
        for tc in collected_tool_calls:
            fn = tc["function"]
            if "_args_chunks" in fn:
                fn["arguments"] = "".join(fn.pop("_args_chunks"))

        collected_content = "".join(content_chunks)
        collected_reasoning = "".join(reasoning_chunks)
        message_dict: Dict[str, Any] = {"role": "assistant"}
        if collected_content:
            message_dict["content"] = collected_content
        if collected_reasoning:
            message_dict["reasoning_content"] = collected_reasoning
        if collected_tool_calls:
            # Convert to proper format
            message_dict["tool_calls"] = [
                {
                    "id": tc["id"] or f"call_{i}",
                    "type": "function",
                    "function": tc["function"],
                }
                for i, tc in enumerate(collected_tool_calls)
                if tc["function"]["name"]  # Only include if we have a name
            ]
            if message_dict["tool_calls"]:
                finish_reason = "tool_calls"

        # Warn if no chunks were received (edge case for debugging)
        if last_chunk is None:
            lib_logger.warning(
                f"[Antigravity] Streaming received zero chunks for {model}"
            )

        response_dict = {
            "id": last_chunk.id if last_chunk else f"chatcmpl-{model}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message_dict,
                    "finish_reason": finish_reason,
                }
            ],
        }

        if usage_info:
            response_dict["usage"] = (
                usage_info.model_dump()
                if hasattr(usage_info, "model_dump")
                else dict(usage_info)
            )

        # Log the final accumulated response
        if file_logger:
            await file_logger.log_final_response(response_dict)

        return litellm.ModelResponse(**response_dict)

    def _accumulate_tool_call(
        self, tc: Any, collected_tool_calls: List[Dict[str, Any]]
    ) -> None:
        """Accumulate a tool call from a streaming chunk into the collected list."""
        # Handle both dict and object access patterns
        if isinstance(tc, dict):
            tc_index = tc.get("index")
            tc_id = tc.get("id")
            tc_function = tc.get("function", {})
            tc_func_name = (
                tc_function.get("name") if isinstance(tc_function, dict) else None
            )
            tc_func_args = (
                tc_function.get("arguments", "")
                if isinstance(tc_function, dict)
                else ""
            )
        else:
            tc_index = getattr(tc, "index", None)
            tc_id = getattr(tc, "id", None)
            tc_function = getattr(tc, "function", None)
            tc_func_name = getattr(tc_function, "name", None) if tc_function else None
            tc_func_args = getattr(tc_function, "arguments", "") if tc_function else ""

        if tc_index is None:
            # Handle edge case where provider omits index
            lib_logger.warning(
                f"[Antigravity] Tool call received without index field, "
                f"appending sequentially: {tc}"
            )
            tc_index = len(collected_tool_calls)

        # Ensure list is long enough
        while len(collected_tool_calls) <= tc_index:
            collected_tool_calls.append(
                {
                    "id": None,
                    "type": "function",
                    "function": {"name": None, "arguments": ""},
                }
            )

        if tc_id:
            collected_tool_calls[tc_index]["id"] = tc_id
        if tc_func_name:
            collected_tool_calls[tc_index]["function"]["name"] = tc_func_name
        if tc_func_args:
            if "_args_chunks" not in collected_tool_calls[tc_index]["function"]:
                collected_tool_calls[tc_index]["function"]["_args_chunks"] = []
            collected_tool_calls[tc_index]["function"]["_args_chunks"].append(tc_func_args)

    def _inject_tool_hardening_instruction(
        self, payload: Dict[str, Any], instruction_text: str
    ) -> None:
        """Inject tool usage hardening system instruction for Gemini 3 & Claude."""
        if not instruction_text:
            return

        instruction_part = {"text": instruction_text}

        if "system_instruction" in payload:
            existing = payload["system_instruction"]
            if isinstance(existing, dict) and "parts" in existing:
                existing["parts"].insert(0, instruction_part)
            else:
                payload["system_instruction"] = {
                    "role": "user",
                    "parts": [instruction_part, {"text": str(existing)}],
                }
        else:
            payload["system_instruction"] = {
                "role": "user",
                "parts": [instruction_part],
            }

    async def _handle_streaming(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
        file_logger: Optional[AntigravityProviderLogger] = None,
        malformed_retry_num: Optional[int] = None,
    ) -> AsyncGenerator[litellm.ModelResponse, None]:
        """Handle streaming completion.

        Args:
            malformed_retry_num: If set, log response chunks to malformed_retry_N_response.log
                                 instead of the main response_stream.log
        """
        # Build tool schema map for schema-aware JSON parsing
        # NOTE: After _transform_to_antigravity_format, tools are at payload["request"]["tools"]
        tools_for_schema = payload.get("request", {}).get("tools")
        tool_schemas = self._build_tool_schema_map(tools_for_schema, model)

        # Accumulator tracks state across chunks for caching and tool indexing
        accumulator = {
            "reasoning_content": "",
            "thought_signature": "",
            "text_content": "",
            "tool_calls": [],
            "tool_idx": 0,  # Track tool call index across chunks
            "is_complete": False,  # Track if we received usageMetadata
            "last_usage": None,  # Track last received usage for final chunk
            "yielded_any": False,  # Track if we yielded any real chunks
            "tool_schemas": tool_schemas,  # For schema-aware JSON string parsing
            "malformed_call": None,  # Track MALFORMED_FUNCTION_CALL if detected
            "response_id": None,  # Track original response ID for synthetic chunks
        }

        async with client.stream(
            "POST",
            url,
            headers=headers,
            json=payload,
            timeout=TimeoutConfig.streaming(),
        ) as response:
            if response.status_code >= 400:
                # Read error body so it's available in response.text for logging
                # The actual logging happens in failure_logger via _extract_response_body
                try:
                    await response.aread()
                    # lib_logger.error(
                    #     f"API error {response.status_code}: {error_body.decode()}"
                    # )
                except (httpx.HTTPError, ValueError, Exception) as e:
                    lib_logger.debug(f"Failed to read error body for logging: {e}")

            response.raise_for_status()

            async for line in response.aiter_lines():
                if file_logger:
                    if malformed_retry_num is not None:
                        await file_logger.log_malformed_retry_response(
                            malformed_retry_num, line
                        )
                    else:
                        await file_logger.log_response_chunk(line)

                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json_loads(data_str)
                        gemini_chunk = self._unwrap_response(chunk)

                        # Capture response ID from first chunk for synthetic responses
                        if not accumulator.get("response_id"):
                            accumulator["response_id"] = gemini_chunk.get("responseId")

                        # Check for MALFORMED_FUNCTION_CALL
                        malformed_msg = self._check_for_malformed_call(gemini_chunk)
                        if malformed_msg:
                            # Store for retry handler, don't yield anything more
                            accumulator["malformed_call"] = malformed_msg
                            break

                        openai_chunk = self._gemini_to_openai_chunk(
                            gemini_chunk, model, accumulator
                        )

                        yield litellm.ModelResponse(**openai_chunk)
                        accumulator["yielded_any"] = True
                    except json.JSONDecodeError:
                        if file_logger:
                            await file_logger.log_error(f"Parse error: {data_str[:100]}")
                        continue

        # Check if we detected a malformed call - raise exception for retry handler
        if accumulator.get("malformed_call"):
            raise _MalformedFunctionCallDetected(
                accumulator["malformed_call"],
                {"accumulator": accumulator},
            )

        # Only emit synthetic final chunk if we actually received real data
        # If no data was received, the caller will detect zero chunks and retry
        if accumulator.get("yielded_any"):
            # If stream ended without usageMetadata chunk, emit a final chunk
            if not accumulator.get("is_complete"):
                final_chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                }
                # Only include usage if we received real data during streaming
                if accumulator.get("last_usage"):
                    final_chunk["usage"] = accumulator["last_usage"]
                yield litellm.ModelResponse(**final_chunk)

            # Log final assembled response for provider logging
            if file_logger:
                # Build final response from accumulated data
                final_message = {"role": "assistant"}
                if accumulator.get("text_content"):
                    final_message["content"] = accumulator["text_content"]
                if accumulator.get("reasoning_content"):
                    final_message["reasoning_content"] = accumulator[
                        "reasoning_content"
                    ]
                if accumulator.get("tool_calls"):
                    final_message["tool_calls"] = accumulator["tool_calls"]

                final_response = {
                    "id": accumulator.get("response_id")
                    or f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": final_message,
                            "finish_reason": "tool_calls"
                            if accumulator.get("tool_calls")
                            else "stop",
                        }
                    ],
                    "usage": accumulator.get("last_usage"),
                }
                await file_logger.log_final_response(final_response)

            # Cache Claude thinking after stream completes
            if (
                self._is_claude(model)
                and self._enable_signature_cache
                and accumulator.get("reasoning_content")
            ):
                self._cache_thinking(
                    accumulator["reasoning_content"],
                    accumulator["thought_signature"],
                    accumulator["text_content"],
                    accumulator["tool_calls"],
                )

    async def _streaming_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
        file_logger: Optional[AntigravityProviderLogger] = None,
        tool_schemas: Optional[Dict[str, Dict[str, Any]]] = None,
        gemini_contents: Optional[List[Dict[str, Any]]] = None,
        gemini_payload: Optional[Dict[str, Any]] = None,
        project_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> AsyncGenerator[litellm.ModelResponse, None]:
        """
        Wrapper around _handle_streaming that retries on empty responses, bare 429s,
        and MALFORMED_FUNCTION_CALL errors.

        If the stream yields zero chunks (Antigravity returned nothing) or encounters
        a bare 429 (no retry info), retry up to EMPTY_RESPONSE_MAX_ATTEMPTS times
        before giving up.

        If MALFORMED_FUNCTION_CALL is detected, inject corrective messages and retry
        up to MALFORMED_CALL_MAX_RETRIES times.
        """
        empty_error_msg = (
            "The model returned an empty response after multiple attempts. "
            "This may indicate a temporary service issue. Please try again."
        )
        transient_429_msg = (
            "The model returned transient 429 errors after multiple attempts. "
            "This may indicate a temporary service issue. Please try again."
        )

        # Track malformed call retries (separate from empty response retries)
        malformed_retry_count = 0
        current_gemini_contents = gemini_contents
        current_payload = payload

        for attempt in range(EMPTY_RESPONSE_MAX_ATTEMPTS):
            chunk_count = 0

            try:
                # Pass malformed_retry_count to log response to separate file
                retry_num = malformed_retry_count if malformed_retry_count > 0 else None
                async for chunk in self._handle_streaming(
                    client,
                    url,
                    headers,
                    current_payload,
                    model,
                    file_logger,
                    malformed_retry_num=retry_num,
                ):
                    chunk_count += 1
                    yield chunk  # Stream immediately - true streaming preserved

                if chunk_count > 0:
                    return  # Success - we got data

                # Zero chunks - empty response
                if attempt < EMPTY_RESPONSE_MAX_ATTEMPTS - 1:
                    lib_logger.warning(
                        f"[Antigravity] Empty stream from {model}, "
                        f"attempt {attempt + 1}/{EMPTY_RESPONSE_MAX_ATTEMPTS}. Retrying..."
                    )
                    await asyncio.sleep(EMPTY_RESPONSE_RETRY_DELAY)
                    continue
                else:
                    # Last attempt failed - raise without extra logging
                    # (caller will log the error)
                    raise EmptyResponseError(
                        provider="antigravity",
                        model=model,
                        message=empty_error_msg,
                    )

            except _MalformedFunctionCallDetected as e:
                # Handle MALFORMED_FUNCTION_CALL - try auto-fix first
                parsed = self._parse_malformed_call_message(e.finish_message, model)

                # Extract response_id and last_usage from accumulator for all paths
                response_id = None
                last_usage = None
                if e.raw_response and isinstance(e.raw_response, dict):
                    acc = e.raw_response.get("accumulator", {})
                    response_id = acc.get("response_id")
                    last_usage = acc.get("last_usage")

                if parsed:
                    # Try to auto-fix the malformed JSON
                    error_info = self._analyze_json_error(parsed["raw_args"])

                    if error_info.get("fixed_json"):
                        # Auto-fix successful - build synthetic response
                        lib_logger.info(
                            f"[Antigravity] Auto-fixed malformed function call for "
                            f"'{parsed['tool_name']}' from {model} (streaming)"
                        )

                        # Log the auto-fix details
                        if file_logger:
                            file_logger.log_malformed_autofix(
                                parsed["tool_name"],
                                parsed["raw_args"],
                                error_info["fixed_json"],
                            )

                        # Use chunk format for streaming with original response ID and usage
                        fixed_chunk = self._build_fixed_tool_call_chunk(
                            model,
                            parsed,
                            error_info,
                            response_id=response_id,
                            usage=last_usage,
                        )
                        if fixed_chunk:
                            yield fixed_chunk
                            return

                # Auto-fix failed - retry by asking model to fix its JSON
                # Each retry response will also attempt auto-fix first
                if malformed_retry_count < MALFORMED_CALL_MAX_RETRIES:
                    malformed_retry_count += 1
                    lib_logger.warning(
                        f"[Antigravity] MALFORMED_FUNCTION_CALL from {model} (streaming), "
                        f"retry {malformed_retry_count}/{MALFORMED_CALL_MAX_RETRIES}: "
                        f"{e.finish_message[:100]}..."
                    )

                    if parsed and gemini_payload is not None:
                        # Get schema for the failed tool
                        tool_schema = (
                            tool_schemas.get(parsed["tool_name"])
                            if tool_schemas
                            else None
                        )

                        # Build corrective messages
                        assistant_msg, user_msg = (
                            self._build_malformed_call_retry_messages(
                                parsed, tool_schema
                            )
                        )

                        # Inject into conversation
                        current_gemini_contents = list(current_gemini_contents or [])
                        current_gemini_contents.append(assistant_msg)
                        current_gemini_contents.append(user_msg)

                        # Only "contents" differs from the original payload;
                        # _transform_to_antigravity_format deep-copies internally
                        # (line 3401: orjson round-trip), so a shallow dict copy
                        # with replaced contents is sufficient here
                        gemini_payload_copy = {**gemini_payload, "contents": current_gemini_contents}
                        current_payload = self._transform_to_antigravity_format(
                            gemini_payload_copy,
                            model,
                            project_id or "",
                            max_tokens,
                            reasoning_effort,
                            tool_choice,
                        )

                        # Log the retry request in the same folder
                        if file_logger:
                            file_logger.log_malformed_retry_request(
                                malformed_retry_count, current_payload
                            )

                    await asyncio.sleep(MALFORMED_CALL_RETRY_DELAY)
                    continue  # Retry with modified payload
                else:
                    # Auto-fix failed and retries disabled/exceeded - yield fallback response
                    lib_logger.warning(
                        f"[Antigravity] MALFORMED_FUNCTION_CALL could not be auto-fixed "
                        f"for {model} (streaming): {e.finish_message[:100]}..."
                    )
                    fallback = self._build_malformed_fallback_chunk(
                        model,
                        e.finish_message,
                        response_id=response_id,
                        usage=last_usage,
                    )
                    yield fallback
                    return

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    # Check if this is a bare 429 (no retry info) vs real quota exhaustion
                    quota_info = self.parse_quota_error(e)
                    if quota_info is None:
                        # Bare 429 - retry like empty response
                        if attempt < EMPTY_RESPONSE_MAX_ATTEMPTS - 1:
                            lib_logger.warning(
                                f"[Antigravity] Bare 429 from {model}, "
                                f"attempt {attempt + 1}/{EMPTY_RESPONSE_MAX_ATTEMPTS}. Retrying..."
                            )
                            await asyncio.sleep(EMPTY_RESPONSE_RETRY_DELAY)
                            continue
                        else:
                            # Last attempt failed - raise TransientQuotaError to rotate
                            raise TransientQuotaError(
                                provider="antigravity",
                                model=model,
                                message=transient_429_msg,
                            )
                    # Has retry info - real quota exhaustion, propagate for cooldown
                    lib_logger.debug(
                        f"429 with retry info - propagating for cooldown: {e}"
                    )
                    raise
                # Other HTTP errors - raise immediately (let caller handle)
                raise

            except (httpx.HTTPError, Exception) as e:
                lib_logger.debug("Non-HTTP streaming error: %s", e)
                # Non-HTTP errors - raise immediately
                raise

        # Should not reach here, but just in case
        lib_logger.error(
            f"[Antigravity] Unexpected exit from streaming retry loop for {model}"
        )
        raise EmptyResponseError(
            provider="antigravity",
            model=model,
            message=empty_error_msg,
        )

    async def count_tokens(
        self,
        client: httpx.AsyncClient,
        credential_path: str,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        litellm_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """Count tokens for the given prompt using Antigravity :countTokens endpoint."""
        try:
            token = await self.get_valid_token(credential_path)
            internal_model = self._alias_to_internal(model)

            # Discover project ID
            project_id = await self._discover_project_id(
                credential_path, token, litellm_params or {}
            )

            system_instruction, contents = await self._transform_messages(
                messages, internal_model
            )
            contents = self._fix_tool_response_grouping(contents)

            gemini_payload = {"contents": contents}
            if system_instruction:
                gemini_payload["systemInstruction"] = system_instruction

            gemini_tools = self._build_tools_payload(tools, model)
            if gemini_tools:
                gemini_payload["tools"] = gemini_tools

            antigravity_payload = {
                "project": project_id,
                "userAgent": "antigravity",
                "requestType": "agent",  # Required per CLIProxyAPI commit 67985d8
                "requestId": _generate_request_id(),
                "model": internal_model,
                "request": gemini_payload,
            }

            url = f"{self._get_base_url()}:countTokens"
            headers = build_bearer_headers(token)

            response = await client.post(
                url, headers=headers, json=antigravity_payload, timeout=30
            )
            try:
                response.raise_for_status()
                data = response.json()
            except (httpx.HTTPStatusError, json.JSONDecodeError, ValueError) as e:
                body_preview = response.text[:200] if response.text else "<empty>"
                lib_logger.warning("OAuth/HTTP error in token count: %s — body: %s", e, body_preview)
                raise
            unwrapped = self._unwrap_response(data)
            total = unwrapped.get("totalTokens", 0)

            return {"prompt_tokens": total, "total_tokens": total}
        except (httpx.HTTPError, ValueError, json.JSONDecodeError, Exception) as e:
            lib_logger.error(f"Token counting failed: {e}")
            return {"prompt_tokens": 0, "total_tokens": 0}
