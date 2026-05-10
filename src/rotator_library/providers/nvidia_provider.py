# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import httpx
import logging
import time
import asyncio
from dataclasses import dataclass, field
from collections import deque
from typing import List, Dict, Any
from .provider_interface import ProviderInterface, build_bearer_headers  # strip_provider_prefix intentionally not used — nvidia preserves org prefix
from .utilities import fetch_provider_models
from .utilities.base_quota_tracker import BaseQuotaTracker

lib_logger = logging.getLogger('rotator_library')


@dataclass
class RateLimitWindow:
    """Sliding window for rate limit tracking."""
    requests: deque = field(default_factory=lambda: deque(maxlen=1000))
    window_seconds: int = 60  # 1 minute window

    def add_request(self, timestamp: float):
        """Add request to window."""
        self.requests.append(timestamp)

    def get_request_count(self, since: float) -> int:
        """Get request count since timestamp."""
        return sum(1 for ts in self.requests if ts >= since)

    def is_within_limit(self, limit: int) -> bool:
        """Check if within rate limit."""
        now = time.time()
        window_start = now - self.window_seconds
        return self.get_request_count(window_start) < limit

# NVIDIA NIM models that support native thinking via chat_template_kwargs
_NVIDIA_DEEPSEEK_THINKING_MODELS = frozenset([
    "deepseek-ai/deepseek-v3.1",
    "deepseek-ai/deepseek-v3.1-terminus",
    "deepseek-ai/deepseek-v3.2",
])

# Qwen models that support thinking via enable_thinking parameter
_QWEN_THINKING_MODELS = frozenset([
    "qwen/qwen3.5-397b-a17b",
    "qwen/qwen3-coder-480b-a35b-instruct",
])

# Moonshot AI models that support thinking via chat_template_kwargs.thinking
_MOONSHOT_THINKING_MODELS = frozenset([
    "moonshotai/kimi-k2.6",
])

# Top-level payload fields that are Anthropic-specific and not supported by NVIDIA NIM
_ANTHROPIC_UNSUPPORTED_FIELDS = frozenset([
    "thinking",          # Anthropic thinking control {type: "adaptive"} / {type: "enabled", budget_tokens: N}
    "betas",             # Anthropic beta features list
])


class NvidiaProvider(ProviderInterface, BaseQuotaTracker):
    skip_cost_calculation = True
    """
    Provider implementation for the NVIDIA NIM API.

    Handles translation of Anthropic-format requests (as sent by Claude Code)
    into NVIDIA-compatible OpenAI-format requests.
    """

    # Provider name for env var lookups
    provider_env_name = "NVIDIA_NIM"

    # BaseQuotaTracker configuration
    _use_integer_max_requests = True
    provider_env_prefix = ""
    cache_subdir = "nvidia"
    default_max_requests = {
        "free": {"_default": 40},
        "paid": {"_default": 500},
    }
    default_max_requests_unknown = 40
    user_to_api_model_map: Dict[str, str] = {}
    api_to_user_model_map: Dict[str, str] = {}

    def __init__(self):
        super().__init__()
        # Inline NVIDIA quota tracker state
        self._windows: Dict[str, Dict[str, RateLimitWindow]] = {}
        self._lock = None
        self.project_tier_cache: Dict[str, str] = {}
        self.project_id_cache: Dict[str, str] = {}
        self._model_quota_groups: Dict[str, List[str]] = {
            "deepseek": [
                "deepseek-ai/deepseek-v3.1",
                "deepseek-ai/deepseek-v3.1-terminus",
                "deepseek-ai/deepseek-v3.2",
                "deepseek-ai/deepseek-r1",
            ],
            "qwen": [
                "qwen/qwen3.5-397b-a17b",
                "qwen/qwen3-coder-480b-a35b-instruct",
            ],
        }
        self._quota_refresh_interval = 300
        self._learned_costs: Dict[str, Dict[str, float]] = {}
        self._learned_costs_loaded = True
        self._learned_costs_lock = None

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches the list of available models from the NVIDIA NIM API.
        """
        return await fetch_provider_models(
            client,
            "https://integrate.api.nvidia.com/v1/models",
            build_bearer_headers(api_key, content_type=None),
            "NVIDIA",
            lambda data: [
                f"nvidia/{model['id']}" for model in data.get("data", [])
                if isinstance(model, dict) and "id" in model
            ],
        )

    def _strip_cache_control_from_content(self, content: Any) -> Any:
        """
        Recursively strips 'cache_control' fields from message content blocks.

        Claude Code sends cache_control: {type: "ephemeral"} on system message blocks
        and user message content blocks. NVIDIA NIM does not support this field and
        may reject requests containing it.

        Args:
            content: The content field of a message — either a string (returned as-is)
                     or a list of content blocks (each block is cleaned in place).

        Returns:
            Cleaned content with all cache_control fields removed.
        """
        if isinstance(content, list):
            cleaned = []
            for block in content:
                if isinstance(block, dict):
                    block = {k: v for k, v in block.items() if k != "cache_control"}
                cleaned.append(block)
            return cleaned
        return content

    def _sanitize_messages(self, payload: Dict[str, Any]) -> None:
        """
        Strips Anthropic-specific fields from all messages in the payload.

        - Removes 'cache_control' from every content block in every message.
        - Removes 'thinking_signature' (Anthropic-specific) while preserving
          'reasoning_content' which NVIDIA NIM documents as a valid string field.
        - Converts messages with list content to plain strings if all blocks are
          text-only (NVIDIA NIM expects string content for many models).
        """
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return

        for msg in messages:
            if not isinstance(msg, dict):
                continue

            # Strip Anthropic-only fields that NVIDIA NIM does not document.
            # reasoning_content is intentionally preserved: NVIDIA NIM documents it
            # as a valid message field (string) for preserved reasoning content.
            msg.pop("thinking_signature", None)
            msg.pop("cache_control", None)

            content = msg.get("content")
            if content is None:
                continue

            # Strip cache_control from content blocks
            cleaned = self._strip_cache_control_from_content(content)

            # For ALL messages with list content, collapse to a single string
            # if every block is text-only.  NVIDIA NIM (OpenAI-compatible) accepts
            # string content far more reliably than array content, and some server
            # internals hash messages for deduplication / caching, which breaks
            # when a message dict contains unhashable nested dicts/lists.
            if isinstance(cleaned, list):
                text_parts = []
                all_text = True
                for block in cleaned:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    else:
                        all_text = False
                        break
                if all_text and text_parts:
                    cleaned = "\n\n".join(text_parts)

            msg["content"] = cleaned

    def _strip_anthropic_fields(self, payload: Dict[str, Any]) -> None:
        """
        Removes top-level Anthropic-specific payload fields that NVIDIA NIM does not support.

        Fields removed:
            - 'thinking': Anthropic adaptive/enabled thinking parameter
            - 'betas': Anthropic beta features list
        """
        for field in _ANTHROPIC_UNSUPPORTED_FIELDS:
            if field in payload:
                lib_logger.debug(
                    f"[NvidiaProvider] Stripping unsupported Anthropic field '{field}' from payload."
                )
                del payload[field]

    def handle_thinking_parameter(self, payload: Dict[str, Any], model: str) -> None:
        """
        Handles reasoning/thinking parameters for NVIDIA NIM models.

        Behavior per model family:

        DeepSeek models (deepseek-v3.1, deepseek-v3.2, etc.):
            - When 'reasoning_effort' is set (low/medium/high), enables native thinking
              via 'extra_body.chat_template_kwargs.thinking = True'.
            - Strips the Anthropic 'thinking' field regardless.

        Qwen models (qwen3.5, qwen3-coder, etc.):
            - When 'reasoning_effort' is set, enables native thinking
              via 'extra_body.chat_template_kwargs.enable_thinking = True'.
            - Strips the Anthropic 'thinking' field regardless.

        All other models (Llama, Mistral, etc.):
            - Strips 'thinking' field entirely (not supported by NVIDIA NIM for these models).
            - Strips 'cache_control' from all content blocks (Anthropic prompt caching).
            - Collapses list-format system messages into plain strings.

        This method is called from client.py for the 'nvidia' provider
        before the litellm call is made.

        Args:
            payload: The litellm_kwargs dict (modified in-place).
            model: Full model string, e.g. 'nvidia/qwen/qwen3.5-397b-a17b'.
        """
        # Extract the bare model name without provider prefix
        # model can be 'nvidia/qwen/qwen3.5-397b-a17b'
        # Strip the first component (provider prefix) to get the actual model ID
        model_name = model.split('/', 1)[1] if '/' in model else model

        lib_logger.info(
            f"[NvidiaProvider] handle_thinking_parameter called for model '{model_name}'. "
            f"Initial payload keys: {list(payload.keys())}"
        )

        # --- Step 1: Strip all Anthropic-specific top-level fields ---
        self._strip_anthropic_fields(payload)

        # --- Step 2: Strip cache_control and normalize messages ---
        self._sanitize_messages(payload)

        # --- Step 2.1: Strip stream_options ---
        # NVIDIA NIM OpenAI-compatible endpoint crashes with 'unhashable type: dict'
        # when stream_options (a dict) is present in the request payload.
        if payload.pop("stream_options", None) is not None:
            lib_logger.debug(
                "[NvidiaProvider] Stripped stream_options from payload for NVIDIA NIM."
            )

        # --- Step 2.5: Sanitize tool_choice ---
        # NVIDIA NIM's OpenAI-compatible endpoint does not accept a dict-form
        # tool_choice (e.g. {"type": "function", "function": {"name": ...}})
        # for several models.  Attempting to send one causes an internal server
        # error: "unhashable type: 'dict'".  Convert to "required" which forces
        # the model to use a tool without specifying the exact one.
        tool_choice = payload.get("tool_choice")
        if isinstance(tool_choice, dict):
            payload["tool_choice"] = "required"
            lib_logger.debug(
                f"[NvidiaProvider] Converted dict tool_choice to 'required' "
                f"for model '{model_name}'."
            )

        # --- Step 2.6: Strip extra_body.chat_template_kwargs ---
        # extra_body is unpacked by LiteLLM/OpenAI SDK into top-level fields.
        # chat_template_kwargs (a dict) causes NVIDIA NIM to crash with
        # "unhashable type: 'dict'" because the server attempts to hash it.
        extra_body = payload.get("extra_body")
        if isinstance(extra_body, dict) and "chat_template_kwargs" in extra_body:
            del extra_body["chat_template_kwargs"]
            lib_logger.debug(
                f"[NvidiaProvider] Stripped chat_template_kwargs from extra_body "
                f"for model '{model_name}'."
            )
            if not extra_body:
                payload.pop("extra_body", None)

        # --- Step 2.7: Simplify tools for all NVIDIA NIM models ---
        # vLLM-backed NVIDIA NIM endpoints crash with "unhashable type: 'dict'"
        # on deeply nested dicts inside tools[].function.parameters (properties,
        # required, etc.) because the server attempts to hash request parameters
        # for deduplication / caching.  Strip nested schemas while keeping the
        # top-level tools list so tool calling still works.
        if "tools" in payload:
            tools = payload.get("tools")
            if isinstance(tools, list) and tools:
                simplified = []
                for tool in tools:
                    if not isinstance(tool, dict) or tool.get("type") != "function":
                        simplified.append(tool)
                        continue
                    func = tool.get("function", {})
                    if not isinstance(func, dict):
                        simplified.append(tool)
                        continue
                    # Keep only name, description, and a minimal parameters stub.
                    # Remove deeply nested properties / required that may trigger
                    # the vLLM hashing bug.
                    clean_func = {
                        "name": func.get("name"),
                        "description": func.get("description", ""),
                    }
                    # Preserve parameters only as an empty object schema.
                    # If the model needs arguments it will still generate JSON,
                    # but we drop the property-level schemas that crash the server.
                    clean_func["parameters"] = {"type": "object"}
                    simplified.append({"type": "function", "function": clean_func})
                payload["tools"] = simplified
                lib_logger.info(
                    f"[NvidiaProvider] Simplified 'tools' ({len(tools)} items) "
                    f"for model '{model_name}' — stripped nested parameter schemas."
                )

        # --- Step 3: Handle reasoning/thinking for supported models ---
        reasoning_effort = payload.get("reasoning_effort")

        # DeepSeek models
        if model_name in _NVIDIA_DEEPSEEK_THINKING_MODELS:
            if reasoning_effort in ("low", "medium", "high"):
                payload.setdefault("extra_body", {})
                payload["extra_body"].setdefault("chat_template_kwargs", {})
                payload["extra_body"]["chat_template_kwargs"]["thinking"] = True
                lib_logger.info(
                    f"[NvidiaProvider] Enabled native thinking for DeepSeek model "
                    f"'{model_name}' (reasoning_effort='{reasoning_effort}')."
                )
            else:
                lib_logger.debug(
                    f"[NvidiaProvider] DeepSeek model '{model_name}' without reasoning_effort; "
                    f"thinking not enabled."
                )
        
        # Qwen models
        elif model_name in _QWEN_THINKING_MODELS:
            if reasoning_effort:
                payload.setdefault("extra_body", {})
                payload["extra_body"].setdefault("chat_template_kwargs", {})
                payload["extra_body"]["chat_template_kwargs"]["enable_thinking"] = True
                lib_logger.info(
                    f"[NvidiaProvider] Enabled native thinking for Qwen model "
                    f"'{model_name}' (reasoning_effort='{reasoning_effort}')."
                )
            else:
                lib_logger.debug(
                    f"[NvidiaProvider] Qwen model '{model_name}' without reasoning_effort; "
                    f"thinking not enabled."
                )

        # Moonshot AI models (kimi-k2.6, etc.)
        # NOTE: NVIDIA NIM endpoint for Moonshot crashes with
        # "unhashable type: 'dict'" when extra_body.chat_template_kwargs is present.
        # Native thinking for kimi-k2.6 on NVIDIA NIM is not enabled via extra_body.
        elif model_name in _MOONSHOT_THINKING_MODELS:
            if reasoning_effort:
                lib_logger.info(
                    f"[NvidiaProvider] Moonshot model '{model_name}' with "
                    f"reasoning_effort='{reasoning_effort}' — thinking control is not "
                    f"supported on NVIDIA NIM for this model family."
                )
            else:
                lib_logger.debug(
                    f"[NvidiaProvider] Moonshot model '{model_name}' without reasoning_effort."
                )

        # Other models
        else:
            # Non-DeepSeek/Qwen models: log if reasoning_effort was present but not actionable
            if reasoning_effort:
                lib_logger.debug(
                    f"[NvidiaProvider] Model '{model_name}' does not support native thinking; "
                    f"reasoning_effort='{reasoning_effort}' will be left for litellm to handle or ignore."
                )

        # Strip reasoning_effort after processing — it is Anthropic/OpenAI-specific
        # and NVIDIA NIM does not document this parameter.
        payload.pop("reasoning_effort", None)

        # --- Final diagnostic log ---
        # Log the remaining payload keys and the type of message content so we
        # can verify in production that sanitization actually ran.
        remaining_keys = list(payload.keys())
        msg_content_type = None
        messages = payload.get("messages")
        if isinstance(messages, list) and messages:
            first_content = messages[0].get("content") if isinstance(messages[0], dict) else None
            msg_content_type = type(first_content).__name__ if first_content is not None else "missing"
        lib_logger.info(
            f"[NvidiaProvider] Sanitization complete for '{model_name}'. "
            f"Remaining keys: {remaining_keys}, first_msg_content_type={msg_content_type}"
        )

    # =========================================================================
    # QUOTA TRACKING METHODS
    # =========================================================================

    async def track_request(self, credential: str, model: str) -> None:
        """
        Track a request for rate limit enforcement.

        Args:
            credential: Credential identifier
            model: Model name (e.g., "nvidia/deepseek-ai/deepseek-v3.1")
        """
        async with self._ensure_lock():
            if "/" in model:
                model = model.split("/", 1)[1]
            model_group = self._get_model_group(model)
            for grouped_model in self._model_quota_groups.get(model_group, [model]):
                window = self._get_window(credential, grouped_model)
                window.add_request(time.time())

    def can_make_request(self, credential: str, model: str) -> bool:
        """
        Check if request is within rate limits.

        Args:
            credential: Credential identifier
            model: Model name

        Returns:
            True if request is allowed, False otherwise
        """
        if "/" in model:
            model = model.split("/", 1)[1]
        model_group = self._get_model_group(model)
        for grouped_model in self._model_quota_groups.get(model_group, [model]):
            window = self._get_window(credential, grouped_model)
            limit = self._get_rate_limit(credential, grouped_model)
            if not window.is_within_limit(limit):
                return False
        return True

    def get_wait_time(self, credential: str, model: str) -> float:
        """
        Get seconds to wait before next request.

        Args:
            credential: Credential identifier
            model: Model name

        Returns:
            Seconds to wait (0.0 if can proceed immediately)
        """
        if "/" in model:
            model = model.split("/", 1)[1]
        model_group = self._get_model_group(model)
        max_wait = 0.0
        for grouped_model in self._model_quota_groups.get(model_group, [model]):
            window = self._get_window(credential, grouped_model)
            limit = self._get_rate_limit(credential, grouped_model)
            if not window.is_within_limit(limit):
                if window.requests:
                    oldest = min(window.requests)
                    wait_time = (oldest + window.window_seconds) - time.time()
                    max_wait = max(max_wait, wait_time)
        return max(0.0, max_wait)

    def get_quota_info(self, credential: str, model: str) -> Dict:
        """
        Get quota information for credential and model.

        Args:
            credential: Credential identifier
            model: Model name

        Returns:
            Dictionary with quota information
        """
        if "/" in model:
            model = model.split("/", 1)[1]
        model_group = self._get_model_group(model)
        stats = {
            "credential": credential,
            "model": model,
            "model_group": model_group,
            "tier": self.project_tier_cache.get(credential, "free"),
            "models": {}
        }
        for grouped_model in self._model_quota_groups.get(model_group, [model]):
            window = self._get_window(credential, grouped_model)
            limit = self._get_rate_limit(credential, grouped_model)
            now = time.time()
            window_start = now - window.window_seconds
            request_count = window.get_request_count(window_start)
            stats["models"][grouped_model] = {
                "request_count": request_count,
                "rate_limit": limit,
                "remaining": max(0, limit - request_count),
                "window_seconds": window.window_seconds,
                "is_within_limit": window.is_within_limit(limit),
            }
        return stats

    # =========================================================================
    # BACKGROUND JOB FOR QUOTA SYNC
    # =========================================================================

    def get_background_job_config(self) -> Dict[str, Any]:
        """
        Get background job configuration for NVIDIA quota sync.

        Returns:
            Dictionary with job configuration or None if no job needed
        """
        return {
            "name": "nvidia_quota_sync",
            "interval": 600,  # Run every 10 minutes
            "run_on_start": False,
        }

    async def run_background_job(
        self, usage_manager: Any, credentials: List[str]
    ) -> None:
        """
        Background job to sync NVIDIA quota windows.

        Cleans up old rate limit windows to prevent memory leaks.

        Args:
            usage_manager: UsageManager instance (not used for NVIDIA)
            credentials: List of credential paths (not used for NVIDIA)
        """
        try:
            cleaned = self.cleanup_old_windows()
            if cleaned > 0:
                lib_logger.debug(
                    f"NVIDIA quota sync: cleaned {cleaned} old windows"
                )
        except httpx.HTTPError as e:
            lib_logger.error(f"Error in NVIDIA quota sync job: {e}", exc_info=True)

    # =====================================================================
    # NVIDIA QUOTA TRACKER INTERNAL METHODS
    # =====================================================================

    def _ensure_lock(self):
        """Lazily create asyncio.Lock to avoid RuntimeError before event loop starts."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _get_window(self, credential: str, model: str) -> RateLimitWindow:
        """Get or create rate limit window for credential+model."""
        if credential not in self._windows:
            self._windows[credential] = {}
        if model not in self._windows[credential]:
            self._windows[credential][model] = RateLimitWindow()
        return self._windows[credential][model]

    def _get_model_group(self, model: str) -> str:
        """Get quota group for model."""
        for group, models in self._model_quota_groups.items():
            if model in models:
                return group
        return model  # Use model name as group if not in any group

    def _get_rate_limit(self, credential: str, model: str) -> int:
        """Get rate limit for credential and model via BaseQuotaTracker."""
        tier = self.project_tier_cache.get(credential, "free")
        return self.get_max_requests_for_model("_default", tier)

    def set_credential_tier(self, credential: str, tier: str) -> None:
        """Set tier for credential."""
        self.project_tier_cache[credential] = tier
        lib_logger.info(f"Set NVIDIA credential {credential} tier to {tier}")

    def reset_window(self, credential: str, model: str) -> None:
        """Reset rate limit window for credential and model."""
        if "/" in model:
            model = model.split("/", 1)[1]
        model_group = self._get_model_group(model)
        for grouped_model in self._model_quota_groups.get(model_group, [model]):
            if credential in self._windows and grouped_model in self._windows[credential]:
                self._windows[credential][grouped_model] = RateLimitWindow()

    def cleanup_old_windows(self, max_age_seconds: int = 3600) -> int:
        """Clean up old rate limit windows."""
        now = time.time()
        cleaned = 0
        for credential in list(self._windows.keys()):
            for model in list(self._windows[credential].keys()):
                window = self._windows[credential][model]
                if window.requests:
                    newest = max(window.requests)
                    if now - newest > max_age_seconds:
                        del self._windows[credential][model]
                        cleaned += 1
                else:
                    del self._windows[credential][model]
                    cleaned += 1
            if not self._windows[credential]:
                del self._windows[credential]
        if cleaned > 0:
            lib_logger.debug(f"Cleaned up {cleaned} old rate limit windows")
        return cleaned

    async def _fetch_quota_for_credential(self, credential_path: str) -> dict:
        return {"status": "error", "error": "NVIDIA NIM has no public quota API", "identifier": credential_path, "tier": None, "fetched_at": time.time()}

    def _extract_model_quota_from_response(self, quota_data: dict, tier: str) -> list:
        return []
