# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

"""Shared chunk aggregator for streaming response assembly.

Used by both the proxy streaming wrapper and the forced-streaming
completion path in the retry layer to avoid ~140 lines of duplicated
aggregation logic.
"""

from __future__ import annotations

import time
from typing import Any

import litellm


class ChunkAggregator:
    """Accumulates streaming chunks into a single response.

    Call ``add_chunk(chunk)`` for every dict chunk in the stream, then
    read ``.result_dict`` or ``.result_model_response(model=...)`` to
    get the assembled output.
    """

    _MAX_CONTENT_PARTS = 50000
    _MAX_GENERIC_PARTS = 5000

    def __init__(self) -> None:
        self._content_parts: list[str] = []
        self._flushed_content: str = ""
        self._generic_str_parts: dict[str, list[str]] = {}
        self._aggregated_tool_calls: dict[int, dict] = {}
        self._final_message: dict[str, Any] = {"role": "assistant"}
        self._usage_data: dict | None = None
        self._finish_reason: str | None = None
        self._first_chunk_meta: dict[str, Any] | None = None

    # -- public API ----------------------------------------------------------

    def add_chunk(self, chunk: dict) -> None:
        """Process one streaming chunk dict."""
        if not isinstance(chunk, dict):
            return

        if self._first_chunk_meta is None:
            self._first_chunk_meta = {
                "id": chunk.get("id"),
                "created": chunk.get("created"),
                "model": chunk.get("model"),
            }

        choices = chunk.get("choices")
        if choices:
            choice = choices[0]
            delta = choice.get("delta", {})

            for key, value in delta.items():
                if value is None:
                    continue
                if key == "content":
                    if value:
                        self._content_parts.append(value)
                        if len(self._content_parts) >= self._MAX_CONTENT_PARTS:
                            self._flushed_content += "".join(self._content_parts)
                            self._content_parts.clear()
                elif key == "tool_calls":
                    self._accumulate_tool_calls(value)
                elif key == "function_call":
                    self._accumulate_function_call(value)
                else:
                    if key == "role":
                        self._final_message[key] = value
                    elif isinstance(value, str):
                        parts = self._generic_str_parts.setdefault(key, [])
                        parts.append(value)
                        if len(parts) >= self._MAX_GENERIC_PARTS:
                            self._final_message.setdefault(key, "".join(parts))
                            self._generic_str_parts.pop(key, None)
                    else:
                        self._final_message[key] = value

            fr = choice.get("finish_reason")
            if fr:
                self._finish_reason = fr

        usage = chunk.get("usage")
        if usage and isinstance(usage, dict):
            self._usage_data = usage

    def check_error_payload(self, chunk: dict) -> None:
        """Raise if *chunk* is an error payload (no choices, has error key).

        Only used by the forced-streaming path in _retry.py.
        """
        if "error" in chunk and "choices" not in chunk:
            error_info = chunk["error"]
            msg = (
                error_info.get("message", str(error_info))
                if isinstance(error_info, dict)
                else str(error_info)
            )
            raise litellm.InternalServerError(msg)

    @property
    def first_chunk_meta(self) -> dict[str, Any] | None:
        return self._first_chunk_meta

    @property
    def usage_data(self) -> dict | None:
        return self._usage_data

    @property
    def finish_reason(self) -> str | None:
        return self._finish_reason

    # -- dict output (proxy streaming wrapper) -------------------------------

    def build_final_message(self) -> dict[str, Any]:
        """Return the assembled ``final_message`` dict for logging."""
        msg = dict(self._final_message)

        if self._flushed_content or self._content_parts:
            msg["content"] = self._flushed_content + "".join(self._content_parts)

        for key, parts in self._generic_str_parts.items():
            msg[key] = "".join(parts)

        if self._aggregated_tool_calls:
            tool_calls_list = []
            for tc in self._aggregated_tool_calls.values():
                fn = tc["function"]
                tool_calls_list.append(
                    {
                        "id": tc.get("id"),
                        "type": tc["type"],
                        "function": {
                            "name": "".join(fn["name_parts"]),
                            "arguments": "".join(fn["args_parts"]),
                        },
                    }
                )
            msg["tool_calls"] = tool_calls_list

        if "function_call" in msg:
            fc = msg["function_call"]
            msg["function_call"] = {
                "name": "".join(fc.get("_name_parts", [])),
                "arguments": "".join(fc.get("_args_parts", [])),
            }

        for field in ("content", "tool_calls", "function_call"):
            if field not in msg:
                msg[field] = None

        fr = self._finish_reason
        if msg.get("tool_calls") and fr != "tool_calls":
            fr = "tool_calls"
        msg.setdefault("finish_reason", fr)

        return msg

    def build_response_dict(self) -> dict[str, Any]:
        """Return a full response dict (id, object, created, model, choices, usage)."""
        if self._first_chunk_meta is None:
            return {}

        msg = self.build_final_message()
        fr = msg.pop("finish_reason", self._finish_reason)

        return {
            "id": self._first_chunk_meta.get("id"),
            "object": "chat.completion",
            "created": self._first_chunk_meta.get("created"),
            "model": self._first_chunk_meta.get("model"),
            "choices": [
                {
                    "index": 0,
                    "message": msg,
                    "finish_reason": fr,
                }
            ],
            "usage": self._usage_data,
        }

    # -- ModelResponse output (forced streaming path) ------------------------

    def build_model_response(
        self,
        model: str,
        *,
        stream_id: Any = None,
    ) -> litellm.ModelResponse:
        """Return a ``litellm.ModelResponse`` object.

        *model* is the requested model name (fallback when metadata is
        absent).  *stream_id* is used as part of the response id fallback.
        """
        meta = self._first_chunk_meta or {}
        content = (self._flushed_content + "".join(self._content_parts)) if (self._flushed_content or self._content_parts) else None

        tool_calls_list = None
        if self._aggregated_tool_calls:
            tool_calls_list = []
            for index in sorted(self._aggregated_tool_calls):
                tc = self._aggregated_tool_calls[index]
                fn = tc["function"]
                tool_calls_list.append(
                    {
                        "index": index,
                        "type": tc.get("type", "function"),
                        "id": tc.get("id", f"call_{index}"),
                        "function": {
                            "name": "".join(fn["name_parts"]),
                            "arguments": "".join(fn["args_parts"]),
                        },
                    }
                )

        final_message: dict[str, Any] = {"role": "assistant"}
        if content is not None:
            final_message["content"] = content
        if tool_calls_list:
            final_message["tool_calls"] = tool_calls_list

        fr = self._finish_reason
        if fr is None:
            if tool_calls_list:
                fr = "tool_calls"
            elif content:
                fr = "stop"
        final_message["finish_reason"] = fr or "stop"

        if "function_call" in self._final_message:
            fc = self._final_message["function_call"]
            final_message["function_call"] = {
                "name": "".join(fc.get("_name_parts", [])),
                "arguments": "".join(fc.get("_args_parts", [])),
            }

        return litellm.ModelResponse(
            id=meta.get("id") or f"chatcmpl-{stream_id or id(self)}",
            created=meta.get("created") or int(time.time()),
            model=meta.get("model") or model,
            choices=[
                {
                    "index": 0,
                    "message": final_message,
                    "finish_reason": final_message.get("finish_reason", "stop"),
                }
            ],
            usage=self._usage_data
            or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    # -- internal helpers ----------------------------------------------------

    def _accumulate_tool_calls(self, value: list[dict]) -> None:
        for tc_chunk in value:
            index = tc_chunk.get("index")
            if index is None:
                continue
            if index not in self._aggregated_tool_calls:
                self._aggregated_tool_calls[index] = {
                    "type": "function",
                    "function": {
                        "name_parts": [],
                        "args_parts": [],
                    },
                }
            tc = self._aggregated_tool_calls[index]
            if tc_chunk.get("id"):
                tc["id"] = tc_chunk["id"]
            if "function" in tc_chunk:
                fn = tc_chunk["function"]
                if fn.get("name") is not None:
                    tc["function"]["name_parts"].append(fn["name"])
                if fn.get("arguments") is not None:
                    tc["function"]["args_parts"].append(fn["arguments"])

    def _accumulate_function_call(self, value: dict) -> None:
        if "function_call" not in self._final_message:
            self._final_message["function_call"] = {
                "_name_parts": [],
                "_args_parts": [],
            }
        if value.get("name") is not None:
            self._final_message["function_call"]["_name_parts"].append(value["name"])
        if value.get("arguments") is not None:
            self._final_message["function_call"]["_args_parts"].append(
                value["arguments"]
            )
