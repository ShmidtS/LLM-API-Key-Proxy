# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Automatic max_tokens calculation to prevent context window overflow errors.

This module calculates balanced max_tokens values based on:
1. Model's context window limit (from ModelRegistry)
2. Current input token count (messages + tools)
3. Safety buffer to avoid edge cases
"""

import logging
import re
from typing import Dict, Any, Optional, Tuple

from litellm.litellm_core_utils.token_counter import token_counter

logger = logging.getLogger("rotator_library")

# Default context window sizes for common models (fallback when registry unavailable)
DEFAULT_CONTEXT_WINDOWS: Dict[str, int] = {
    # OpenAI
    "gpt-4": 8192,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-3.5-turbo": 16385,
    # Anthropic
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    "claude-3.5-sonnet": 200000,
    "claude-3.5-haiku": 200000,
    "claude-sonnet-4": 200000,
    "claude-opus-4": 200000,
    # Google
    "gemini-1.5-pro": 1048576,
    "gemini-1.5-flash": 1048576,
    "gemini-2.0-flash": 1048576,
    "gemini-2.5-pro": 1048576,
    "gemini-2.5-flash": 1048576,
    # DeepSeek
    "deepseek-chat": 64000,
    "deepseek-coder": 64000,
    "deepseek-reasoner": 64000,
    # Mistral
    "mistral-large": 128000,
    "mistral-medium": 32000,
    "mistral-small": 32000,
    # Other common
    "llama-3.1-405b": 131072,
    "llama-3.1-70b": 131072,
    "llama-3.1-8b": 131072,
    # ZhipuAI / GLM models (via Kilocode, Z-AI, etc.)
    "glm-4": 128000,
    "glm-4-plus": 128000,
    "glm-4-air": 128000,
    "glm-4-flash": 128000,
    "glm-5": 202800,
    "z-ai/glm-5": 202800,
}

# Safety buffer (tokens reserved for system overhead, response formatting, etc.)
# Increased from 100 to 1000 to account for:
# - Token counting estimation errors (~5-10%)
# - Provider-specific tokenization differences
# - System message and metadata overhead
# - Tool definitions and function call overhead
DEFAULT_SAFETY_BUFFER = 1000

# Minimum max_tokens to request (avoid degenerate cases)
MIN_MAX_TOKENS = 256

# Maximum percentage of context window to use for output (prevent edge cases)
MAX_OUTPUT_RATIO = 0.75

# Maximum percentage of context window for input (leave room for output)
# If input exceeds this, messages should be trimmed or request rejected
MAX_INPUT_RATIO = 1.0

# Extra buffer for providers with known tokenization differences
PROVIDER_SAFETY_BUFFERS = {
    "kilocode": 2000,  # Kilocode has additional overhead
    "openrouter": 1500,
    "gemini": 1000,  # Gemini tokenization can vary
    "anthropic": 500,
}


def extract_model_name(model: str) -> str:
    """
    Extract the base model name from a provider-prefixed model string.

    Examples:
        "openai/gpt-4o" -> "gpt-4o"
        "anthropic/claude-3-opus" -> "claude-3-opus"
        "kilocode/z-ai/glm-5:free" -> "z-ai/glm-5:free"
    """
    if "/" in model:
        parts = model.split("/", 1)
        return parts[1] if len(parts) > 1 else model
    return model


def normalize_model_name(model: str) -> str:
    """
    Normalize model name for lookup.

    Handles common variations like:
        "gpt-4-0125-preview" -> "gpt-4-turbo"
        "claude-3-opus-20240229" -> "claude-3-opus"
    """
    model = model.lower().strip()

    # Remove version/date suffixes
    model = re.sub(r"-[0-9]{4,}$", "", model)  # Remove date like -20240229
    model = re.sub(r"-preview$", "", model)
    model = re.sub(r"-latest$", "", model)

    return model


def get_context_window(model: str, registry=None) -> Optional[int]:
    """
    Get the context window size for a model.

    Args:
        model: Full model identifier (e.g., "openai/gpt-4o")
        registry: Optional ModelRegistry instance for lookups

    Returns:
        Context window size in tokens, or None if unknown
    """
    # Try registry first if available
    if registry is not None:
        try:
            metadata = registry.lookup(model)
            if metadata and metadata.limits.context_window:
                return metadata.limits.context_window
        except Exception as e:
            logger.debug(f"Registry lookup failed for {model}: {e}")

    # Extract base model name
    base_model = extract_model_name(model)
    normalized = normalize_model_name(base_model)

    # Try direct match
    if base_model in DEFAULT_CONTEXT_WINDOWS:
        return DEFAULT_CONTEXT_WINDOWS[base_model]

    if normalized in DEFAULT_CONTEXT_WINDOWS:
        return DEFAULT_CONTEXT_WINDOWS[normalized]

    # Try partial matches
    for pattern, window in DEFAULT_CONTEXT_WINDOWS.items():
        if pattern in normalized or normalized in pattern:
            return window

    # Special handling for common prefixes
    for prefix in ["gpt-4", "gpt-3.5", "claude-3", "gemini-", "deepseek", "mistral"]:
        if normalized.startswith(prefix):
            for pattern, window in DEFAULT_CONTEXT_WINDOWS.items():
                if pattern.startswith(prefix):
                    return window

    return None


def count_input_tokens(
    messages: list,
    model: str,
    tools: Optional[list] = None,
    tool_choice: Optional[Any] = None,
) -> int:
    """
    Count total input tokens including messages and tools.

    Args:
        messages: List of message dictionaries
        model: Model identifier for token counting
        tools: Optional list of tool definitions
        tool_choice: Optional tool choice parameter

    Returns:
        Total input token count
    """
    total = 0

    # Count message tokens
    if messages:
        try:
            total += token_counter(model=model, messages=messages)
        except Exception as e:
            logger.warning(f"Failed to count message tokens: {e}")
            # Fallback: rough estimate
            total += sum(len(str(m).split()) * 4 // 3 for m in messages)

    # Count tool definition tokens
    if tools:
        try:
            import json

            tools_json = json.dumps(tools)
            total += token_counter(model=model, text=tools_json)
        except Exception as e:
            logger.debug(f"Failed to count tool tokens: {e}")
            # Fallback: rough estimate
            total += len(str(tools)) // 4

    return total


def get_provider_safety_buffer(model: str) -> int:
    """
    Get provider-specific safety buffer based on model prefix.

    Args:
        model: Full model identifier (e.g., "kilocode/z-ai/glm-5:free")

    Returns:
        Safety buffer for this provider
    """
    # Extract provider from model
    if "/" in model:
        provider = model.split("/")[0].lower()
        if provider in PROVIDER_SAFETY_BUFFERS:
            return PROVIDER_SAFETY_BUFFERS[provider]
    return DEFAULT_SAFETY_BUFFER


def calculate_max_tokens(
    model: str,
    messages: Optional[list] = None,
    tools: Optional[list] = None,
    tool_choice: Optional[Any] = None,
    requested_max_tokens: Optional[int] = None,
    registry=None,
    safety_buffer: Optional[int] = None,
) -> Tuple[Optional[int], str]:
    """
    Calculate a safe max_tokens value based on context window and input.

    Args:
        model: Full model identifier
        messages: List of message dictionaries
        tools: Optional list of tool definitions
        tool_choice: Optional tool choice parameter
        requested_max_tokens: User-requested max_tokens (if any)
        registry: Optional ModelRegistry for context window lookup
        safety_buffer: Extra buffer for safety (default: auto-detect from provider)

    Returns:
        Tuple of (calculated_max_tokens, reason) where reason explains the calculation
        Returns (None, "input_exceeds_context") if input is too large and cannot be processed
    """
    # Get context window
    context_window = get_context_window(model, registry)

    if context_window is None:
        if requested_max_tokens is not None:
            return requested_max_tokens, "unknown_context_window_using_requested"
        return None, "unknown_context_window_no_request"

    # Use provider-specific buffer if not specified
    if safety_buffer is None:
        safety_buffer = get_provider_safety_buffer(model)

    # Count input tokens
    input_tokens = 0
    if messages:
        input_tokens = count_input_tokens(messages, model, tools, tool_choice)

    # CRITICAL CHECK: Input must not exceed max allowed ratio
    max_input_allowed = int(context_window * MAX_INPUT_RATIO)
    if input_tokens > max_input_allowed:
        logger.error(
            f"Input tokens ({input_tokens}) exceed maximum allowed ({max_input_allowed}) "
            f"for context window ({context_window}). Model: {model}. "
            f"Request will fail - consider reducing conversation history."
        )
        # Return None to signal the request should be rejected
        return (
            None,
            f"input_exceeds_context_by_{input_tokens - max_input_allowed}_tokens",
        )

    # Calculate available space for output
    available_for_output = context_window - input_tokens - safety_buffer

    if available_for_output < MIN_MAX_TOKENS:
        # Input is too large - log warning but allow minimal response
        logger.warning(
            f"Input tokens ({input_tokens}) leave insufficient space for output "
            f"(available: {available_for_output}, min: {MIN_MAX_TOKENS}). Model: {model}"
        )
        return MIN_MAX_TOKENS, "input_exceeds_context_minimal_output"

    # Apply maximum output ratio
    max_allowed_by_ratio = int(context_window * MAX_OUTPUT_RATIO)
    capped_available = min(available_for_output, max_allowed_by_ratio)

    # If user requested a specific value, honor it if valid
    if requested_max_tokens is not None:
        if requested_max_tokens <= capped_available:
            return requested_max_tokens, "using_requested_within_limit"
        else:
            # User requested too much, cap it
            return (
                capped_available,
                f"capped_from_{requested_max_tokens}_to_{capped_available}",
            )

    # No specific request - use calculated value
    return (
        capped_available,
        f"calculated_from_context_{context_window}_input_{input_tokens}",
    )


def adjust_max_tokens_in_payload(
    payload: Dict[str, Any],
    model: str,
    registry=None,
) -> Tuple[Dict[str, Any], bool]:
    """
    Adjust max_tokens in a request payload to prevent context overflow.

    This function:
    1. Calculates input token count from messages + tools
    2. Gets context window for the model
    3. Sets max_tokens to a safe value if not already set or if too large
    4. Returns flag indicating if request should be rejected

    Args:
        payload: Request payload dictionary
        model: Model identifier
        registry: Optional ModelRegistry instance

    Returns:
        Tuple of (modified payload, should_reject flag)
        If should_reject is True, the input exceeds context window and request will fail
    """
    # Check if max_tokens adjustment is needed
    # Look for both max_tokens (OpenAI) and max_completion_tokens (newer OpenAI)
    requested_max = payload.get("max_tokens") or payload.get("max_completion_tokens")

    messages = payload.get("messages", [])
    tools = payload.get("tools")
    tool_choice = payload.get("tool_choice")

    # Calculate safe max_tokens
    calculated_max, reason = calculate_max_tokens(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        requested_max_tokens=requested_max,
        registry=registry,
    )

    # Check if request should be rejected due to input exceeding context
    if calculated_max is None and "input_exceeds_context" in reason:
        logger.error(
            f"Rejecting request for {model}: {reason}. "
            f"Input tokens exceed context window capacity."
        )
        return payload, True  # Signal to reject

    if calculated_max is not None:
        # Log the adjustment
        if requested_max is None:
            logger.info(
                f"Auto-setting max_tokens={calculated_max} for model {model} "
                f"(reason: {reason})"
            )
        elif calculated_max != requested_max:
            logger.info(
                f"Adjusting max_tokens from {requested_max} to {calculated_max} "
                f"for model {model} (reason: {reason})"
            )

        # Set both max_tokens and max_completion_tokens for compatibility
        # Some providers use max_tokens, others use max_completion_tokens
        payload["max_tokens"] = calculated_max

        # Only set max_completion_tokens if it was originally present or for OpenAI models
        if "max_completion_tokens" in payload or model.startswith(("openai/", "gpt")):
            payload["max_completion_tokens"] = calculated_max

    return payload, False

    return payload
