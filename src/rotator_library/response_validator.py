# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""Lightweight response schema validation.

Pure dict checks only — no jsonschema, no pydantic — to stay under 2ms per call.
Used for both non-streaming response validation and per-chunk streaming validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


_VALID_FINISH_REASONS = frozenset({
    None, "stop", "length", "tool_calls", "content_filter", "function_call",
})


@dataclass(slots=True)
class ValidationResult:
    """Outcome of a single validation check."""

    valid: bool
    error: str = ""
    field_path: str = ""


class ResponseValidator:
    """Stateless, fast response validator using pure isinstance checks."""

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    @staticmethod
    def validate_non_streaming(
        response: object,
        level: str = "standard",
    ) -> ValidationResult:
        """Validate a complete non-streaming response dict.

        Args:
            response: The response object returned by the provider/litellm.
            level: One of ``"strict"``, ``"standard"``, ``"lenient"``, ``"off"``.

        Returns:
            :class:`ValidationResult` indicating pass/fail and the first
            violated field path.
        """
        if level == "off":
            return ValidationResult(valid=True)

        # --- Level: lenient (dict + choices is list) ---
        if not isinstance(response, dict):
            return ValidationResult(
                valid=False,
                error="Response is not a dict",
                field_path="(root)",
            )

        choices = response.get("choices")
        if choices is not None and not isinstance(choices, list):
            return ValidationResult(
                valid=False,
                error="choices is not a list",
                field_path="choices",
            )

        if level == "lenient":
            return ValidationResult(valid=True)

        # --- Level: standard (structure checks) ---
        if choices is not None and len(choices) > 0:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if message is not None:
                    if not isinstance(message, dict):
                        return ValidationResult(
                            valid=False,
                            error="choices[0].message is not a dict",
                            field_path="choices[0].message",
                        )
                    # message must contain at least one payload key
                    if not (
                        "content" in message
                        or "tool_calls" in message
                        or "function_call" in message
                    ):
                        return ValidationResult(
                            valid=False,
                            error=(
                                "choices[0].message missing content, "
                                "tool_calls, and function_call"
                            ),
                            field_path="choices[0].message",
                        )

        model = response.get("model")
        if model is not None and not isinstance(model, str):
            return ValidationResult(
                valid=False,
                error="model is not a string",
                field_path="model",
            )

        usage = response.get("usage")
        if usage is not None and not isinstance(usage, dict):
            return ValidationResult(
                valid=False,
                error="usage is not a dict",
                field_path="usage",
            )

        return ValidationResult(valid=True)

    # ------------------------------------------------------------------
    # Streaming (per-chunk)
    # ------------------------------------------------------------------

    @staticmethod
    def validate_streaming_chunk(
        chunk: object,
        level: str = "standard",
    ) -> ValidationResult:
        """Validate a single streaming chunk dict.

        Non-dict chunks pass through unchanged (existing behaviour).

        Args:
            chunk: The chunk dict yielded from the streaming pipeline.
            level: Validation level (see :meth:`validate_non_streaming`).

        Returns:
            :class:`ValidationResult`.
        """
        if level == "off":
            return ValidationResult(valid=True)

        if not isinstance(chunk, dict):
            # Non-dict chunks pass through — not a validation concern.
            return ValidationResult(valid=True)

        choices = chunk.get("choices")
        if choices is not None and not isinstance(choices, list):
            return ValidationResult(
                valid=False,
                error="choices is not a list",
                field_path="choices",
            )

        if level == "lenient":
            return ValidationResult(valid=True)

        # --- standard ---
        if choices is not None and len(choices) > 0:
            choice = choices[0]
            if isinstance(choice, dict):
                delta = choice.get("delta")
                if delta is not None and not isinstance(delta, dict):
                    return ValidationResult(
                        valid=False,
                        error="choices[0].delta is not a dict",
                        field_path="choices[0].delta",
                    )

                finish_reason = choice.get("finish_reason")
                if finish_reason not in _VALID_FINISH_REASONS:
                    return ValidationResult(
                        valid=False,
                        error=f"choices[0].finish_reason has invalid value: {finish_reason!r}",
                        field_path="choices[0].finish_reason",
                    )

        return ValidationResult(valid=True)
