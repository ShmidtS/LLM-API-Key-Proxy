# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

from typing import Dict, Any, Set, Optional, Tuple

from .token_calculator import adjust_max_tokens_in_payload


# Kilocode/OpenRouter free models often have limited parameter support
# These parameters are commonly unsupported and cause 400 errors
KILOCODE_UNSUPPORTED_PARAMS: Set[str] = {
    "stream_options",  # Not supported by many free models
    "frequency_penalty",  # Often unsupported
    "presence_penalty",  # Often unsupported
    "top_p",  # Sometimes unsupported
    "top_k",  # Sometimes unsupported
    "stop",  # Can cause issues with some models
    "n",  # Number of completions - often unsupported
    "logprobs",  # Often unsupported
    "top_logprobs",  # Often unsupported
    "user",  # User identifier - often ignored but can cause issues
    "seed",  # Not supported by all models
    "response_format",  # Only supported by some models
    "reasoning_effort",  # OpenAI-specific, not supported by Kilocode/Novita free models
}

# Fireworks AI rejects extra fields that other providers accept
FIREWORKS_UNSUPPORTED_PARAMS: Set[str] = {
    "chat_template_kwargs",
    "reasoning_budget",
}

_GEMINI_THINKING_MODELS = frozenset({"gemini/gemini-2.5-pro", "gemini/gemini-2.5-flash"})

# Models that don't support the temperature parameter at all.
# OpenAI returns 400 "Only the default (1) value is supported" for these.
_NO_TEMPERATURE_MODEL_PREFIXES = (
    "openai/gpt-5",
    "openai/o1-",
    "openai/o3-",
    "openai/o4-",
    "gpt-5",
    "o1-",
    "o3-",
    "o4-",
)


def sanitize_request_payload(
    payload: Dict[str, Any],
    model: str,
    registry: Optional[Any] = None,
    auto_adjust_max_tokens: bool = True,
) -> Tuple[Dict[str, Any], bool]:
    """
    Sanitizes and adjusts the request payload based on the model.

    This function:
    1. Removes unsupported parameters for specific providers
    2. Automatically adjusts max_tokens to prevent context window overflow

    Args:
        payload: The request payload dictionary
        model: The model identifier (e.g., "openai/gpt-4o")
        registry: Optional ModelRegistry instance for context window lookup
        auto_adjust_max_tokens: Whether to auto-adjust max_tokens (default: True)

    Returns:
        Tuple of (sanitized payload dictionary, should_reject flag).
        should_reject is True if the request should be rejected (input exceeds context window).
    """
    if isinstance(model, str):
        normalized_model = model
        if model != model.strip().lower():
            normalized_model = model.strip().lower()
    else:
        normalized_model = ""

    if "dimensions" in payload and not normalized_model.startswith(
        "openai/text-embedding-3"
    ):
        del payload["dimensions"]

    if payload.get("thinking") == {"type": "enabled", "budget_tokens": -1}:
        if normalized_model not in _GEMINI_THINKING_MODELS:
            del payload["thinking"]

    # Kilocode provider - remove unsupported parameters for free models
    # Free models through Kilocode/OpenRouter often reject extra parameters
    if normalized_model.startswith("kilocode/"):
        model_without_prefix = normalized_model.split("/", 1)[1] if "/" in normalized_model else normalized_model

        # Free models have stricter parameter requirements
        if ":free" in model_without_prefix or model_without_prefix.startswith("z-ai/"):
            for param in KILOCODE_UNSUPPORTED_PARAMS:
                if param in payload:
                    del payload[param]

    # Fireworks AI rejects extra fields with "Extra inputs are not permitted"
    if normalized_model.startswith("fireworks/"):
        for param in FIREWORKS_UNSUPPORTED_PARAMS:
            if param in payload:
                del payload[param]

    # Strip temperature for models that don't support it at all (400 error)
    if "temperature" in payload:
        for prefix in _NO_TEMPERATURE_MODEL_PREFIXES:
            if normalized_model.startswith(prefix):
                del payload["temperature"]
                break

    # Auto-adjust max_tokens to prevent context window overflow
    should_reject = False
    if auto_adjust_max_tokens and (
        "max_tokens" in payload
        or "max_completion_tokens" in payload
        or "messages" in payload
    ):
        payload, should_reject = adjust_max_tokens_in_payload(payload, model, registry)

    return payload, should_reject
