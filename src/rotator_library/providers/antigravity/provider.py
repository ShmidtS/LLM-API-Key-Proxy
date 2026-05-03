# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""AntigravityProvider - core class for Google Antigravity API."""

import os
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, cast

import httpx
import litellm  # type: ignore[import-untyped]
import json
from ...utils.json_utils import json_loads

from ...config import env_bool, env_int
from ...error_handler import EmptyResponseError, TransientQuotaError
from ...model_definitions import ModelDefinitions
from ...timeout_config import TimeoutConfig
from ...transaction_logger import AntigravityProviderLogger
from ...utils.duration import parse_duration as _parse_duration_shared
from ..provider_interface import ProviderInterface, UsageResetConfigDef, QuotaGroupMap, build_bearer_headers
from ..antigravity_auth_base import AntigravityAuthBase
from ..provider_cache import ProviderCache
from ..utilities.antigravity_quota_tracker import AntigravityQuotaTracker
from ..utilities.gemini_shared_utils import (
    GEMINI_TIER_PRIORITIES,
    GEMINI_DEFAULT_TIER_PRIORITY,
    GEMINI_DEFAULT_PRIORITY_MULTIPLIERS,
    GEMINI_DEFAULT_SEQUENTIAL_FALLBACK_MULTIPLIER,
    alias_to_internal_model,
    internal_to_alias_model,
)
from ..utilities.gemini_tool_handler import GeminiToolHandler
from ..utilities.gemini_credential_manager import GeminiCredentialManager
from .constants import (
    BASE_URLS,
    ANTIGRAVITY_HEADERS,
    AVAILABLE_MODELS,
    MODEL_ALIAS_MAP,
    MODEL_ALIAS_REVERSE,
    EXCLUDED_MODELS,
    _generate_request_id,
    _generate_project_id,
    DEFAULT_GEMINI3_SYSTEM_INSTRUCTION,
    DEFAULT_CLAUDE_SYSTEM_INSTRUCTION,
    DEFAULT_PARALLEL_TOOL_INSTRUCTION,
    ENABLE_INTERLEAVED_THINKING,
    CLAUDE_INTERLEAVED_THINKING_HINT,
    _get_gemini3_signature_cache_file,
    _get_claude_thinking_cache_file,
    lib_logger,
)
from .thinking_cache import ThinkingCacheMixin
from .message_transform import MessageTransformMixin
from .tool_recovery import ToolRecoveryMixin
from .streaming import AntigravityStreamingMixin


class AntigravityProvider(
    ThinkingCacheMixin,
    MessageTransformMixin,
    ToolRecoveryMixin,
    AntigravityStreamingMixin,
    AntigravityAuthBase,
    AntigravityQuotaTracker,
    GeminiToolHandler,
    GeminiCredentialManager,
    ProviderInterface,
):
    """
    Antigravity provider for Gemini and Claude models via Google's internal API.

    Supports:
    - Gemini 2.5 (Pro/Flash) with thinkingBudget
    - Gemini 3 (Pro/Flash/Image) with thinkingLevel
    - Claude Sonnet 4.5 via Antigravity proxy
    - Claude Opus 4.5 via Antigravity proxy

    Features:
    - Unified streaming/non-streaming handling
    - ThoughtSignature caching for multi-turn conversations
    - Automatic base URL fallback
    - Gemini 3 tool hallucination prevention
    """

    skip_cost_calculation = True

    # Sequential mode by default - preserves thinking signature caches between requests
    default_rotation_mode: str = "sequential"

    # =========================================================================
    # TIER & USAGE CONFIGURATION
    # =========================================================================

    # Provider name for env var lookups (QUOTA_GROUPS_ANTIGRAVITY_*)
    provider_env_name: str = "antigravity"

    tier_priorities = GEMINI_TIER_PRIORITIES

    default_tier_priority: int = GEMINI_DEFAULT_TIER_PRIORITY

    # Usage reset configs keyed by priority sets
    # Priorities 1-2 (paid tiers) get 5h window, others get 7d window
    usage_reset_configs = {
        frozenset({1, 2}): UsageResetConfigDef(
            window_seconds=5 * 60 * 60,  # 5 hours
            mode="per_model",
            description="5-hour per-model window (paid tier)",
            field_name="models",
        ),
        "default": UsageResetConfigDef(
            window_seconds=7 * 24 * 60 * 60,  # 7 days
            mode="per_model",
            description="7-day per-model window (free/unknown tier)",
            field_name="models",
        ),
    }

    # Model quota groups (can be overridden via QUOTA_GROUPS_ANTIGRAVITY_CLAUDE)
    # Models in the same group share quota - when one is exhausted, all are
    # Based on empirical testing - see tests/quota_verification/QUOTA_TESTING_GUIDE.md
    # Note: -thinking variants are included since they share the same quota pool
    # (users call non-thinking names, proxy maps to -thinking internally)
    # Group names are kept short for compact TUI display
    model_quota_groups: QuotaGroupMap = {
        # Claude and GPT-OSS share the same quota pool
        "claude": [
            "claude-sonnet-4-5",
            "claude-sonnet-4-5-thinking",
            "claude-opus-4-5",
            "claude-opus-4-5-thinking",
            "claude-sonnet-4.5",
            "claude-opus-4.5",
            "gpt-oss-120b-medium",
        ],
        # Gemini 3 Pro variants share quota
        "g3-pro": [
            "gemini-3-pro-high",
            "gemini-3-pro-low",
            "gemini-3-pro-preview",
        ],
        # Gemini 3 Flash (standalone)
        "g3-flash": [
            "gemini-3-flash",
        ],
        # Gemini 2.5 Flash variants share quota (verified 2026-01-07: NOT including Lite)
        "g25-flash": [
            "gemini-2.5-flash",
            "gemini-2.5-flash-thinking",
        ],
        # Gemini 2.5 Flash Lite - SEPARATE quota pool (verified 2026-01-07)
        "g25-lite": [
            "gemini-2.5-flash-lite",
        ],
    }

    # Model usage weights for grouped usage calculation
    # Opus consumes more quota per request, so its usage counts 2x when
    # comparing credentials for selection
    model_usage_weights = {}

    default_priority_multipliers = GEMINI_DEFAULT_PRIORITY_MULTIPLIERS

    default_sequential_fallback_multiplier = GEMINI_DEFAULT_SEQUENTIAL_FALLBACK_MULTIPLIER

    @classmethod
    def parse_quota_error(
        cls, error: Exception, error_body: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Parse Antigravity/Google RPC quota errors.

        Overrides base class because _quota_error_patterns cannot express:
        - Navigation of error.details[] array with @type-based dispatch
        - _parse_duration_shared for Google protobuf duration strings (retryDelay, quotaResetDelay)
        - quotaResetTimeStamp extraction and datetime-to-epoch conversion
        """
        import re as regex_module

        body = cls._extract_error_body(error, error_body)
        if not body:
            body = str(error)

        # Try to find JSON in the body
        try:
            json_match = regex_module.search(r"\{[\s\S]*\}", body)
            if not json_match:
                return None

            data = json_loads(json_match.group(0))
        except (json.JSONDecodeError, AttributeError, TypeError):
            return None

        error_obj = data.get("error", data)
        details = error_obj.get("details", [])

        result: Dict[str, Any] = {
            "retry_after": None,
            "reason": None,
            "reset_timestamp": None,
            "quota_reset_timestamp": None,
        }

        for detail in details:
            detail_type = detail.get("@type", "")

            if "RetryInfo" in detail_type:
                retry_delay = detail.get("retryDelay")
                if retry_delay:
                    parsed = _parse_duration_shared(retry_delay)
                    if parsed is not None:
                        result["retry_after"] = parsed

            elif "ErrorInfo" in detail_type:
                result["reason"] = detail.get("reason")
                metadata = detail.get("metadata", {})

                if result["retry_after"] is None:
                    quota_delay = metadata.get("quotaResetDelay")
                    if quota_delay:
                        parsed = _parse_duration_shared(quota_delay)
                        if parsed is not None:
                            result["retry_after"] = parsed

                reset_ts_str = metadata.get("quotaResetTimeStamp")
                result["reset_timestamp"] = reset_ts_str

                if reset_ts_str:
                    try:
                        reset_dt = datetime.fromisoformat(
                            reset_ts_str.replace("Z", "+00:00")
                        )
                        result["quota_reset_timestamp"] = reset_dt.timestamp()
                    except (ValueError, AttributeError) as e:
                        lib_logger.warning(
                            f"Failed to parse quota reset timestamp '{reset_ts_str}': {e}"
                        )

        if result["retry_after"] is None:
            return None

        return result

    def __init__(self):
        super().__init__()
        self.model_definitions = ModelDefinitions()

        # Base URL management
        self._base_url_index = 0
        self._current_base_url = BASE_URLS[0]

        # Configuration from environment
        memory_ttl = env_int("ANTIGRAVITY_SIGNATURE_CACHE_TTL", 3600)
        disk_ttl = env_int("ANTIGRAVITY_SIGNATURE_DISK_TTL", 86400)

        # Initialize caches using shared ProviderCache
        self._signature_cache = ProviderCache(
            _get_gemini3_signature_cache_file(),
            memory_ttl,
            disk_ttl,
            env_prefix="ANTIGRAVITY_SIGNATURE",
        )
        self._thinking_cache = ProviderCache(
            _get_claude_thinking_cache_file(),
            memory_ttl,
            disk_ttl,
            env_prefix="ANTIGRAVITY_THINKING",
        )

        # Quota tracking state
        self._learned_costs: Dict[
            str, Dict[str, int]
        ] = {}  # tier -> model -> max_requests
        self._learned_costs_loaded: bool = False
        self._quota_refresh_interval = env_int(
            "ANTIGRAVITY_QUOTA_REFRESH_INTERVAL", 300
        )  # 5 min
        self._initial_quota_fetch_done: bool = (
            False  # Track if initial full fetch completed
        )

        # Feature flags
        self._preserve_signatures_in_client = env_bool(
            "ANTIGRAVITY_PRESERVE_THOUGHT_SIGNATURES", True
        )
        self._enable_signature_cache = env_bool(
            "ANTIGRAVITY_ENABLE_SIGNATURE_CACHE", True
        )
        self._enable_dynamic_models = env_bool(
            "ANTIGRAVITY_ENABLE_DYNAMIC_MODELS", False
        )
        self._enable_gemini3_tool_fix = env_bool("ANTIGRAVITY_GEMINI3_TOOL_FIX", True)
        self._enable_claude_tool_fix = env_bool("ANTIGRAVITY_CLAUDE_TOOL_FIX", False)
        self._enable_thinking_sanitization = env_bool(
            "ANTIGRAVITY_CLAUDE_THINKING_SANITIZATION", True
        )

        # Gemini 3 tool fix configuration
        self._gemini3_tool_prefix = os.getenv(
            "ANTIGRAVITY_GEMINI3_TOOL_PREFIX", "gemini3_"
        )
        self._gemini3_description_prompt = os.getenv(
            "ANTIGRAVITY_GEMINI3_DESCRIPTION_PROMPT",
            "\n\n⚠️ STRICT PARAMETERS (use EXACTLY as shown): {params}. Do NOT use parameters from your training data - use ONLY these parameter names.",
        )
        self._gemini3_enforce_strict_schema = env_bool(
            "ANTIGRAVITY_GEMINI3_STRICT_SCHEMA", True
        )
        # Toggle for JSON string parsing in tool call arguments
        # NOTE: This is possibly redundant - modern Gemini models may not need this fix.
        # Disabled by default. Enable if you see JSON-stringified values in tool args.
        self._enable_json_string_parsing = env_bool(
            "ANTIGRAVITY_ENABLE_JSON_STRING_PARSING", True
        )
        self._gemini3_system_instruction = os.getenv(
            "ANTIGRAVITY_GEMINI3_SYSTEM_INSTRUCTION", DEFAULT_GEMINI3_SYSTEM_INSTRUCTION
        )

        # Claude tool fix configuration (separate from Gemini 3)
        self._claude_description_prompt = os.getenv(
            "ANTIGRAVITY_CLAUDE_DESCRIPTION_PROMPT", "\n\nSTRICT PARAMETERS: {params}."
        )
        self._claude_system_instruction = os.getenv(
            "ANTIGRAVITY_CLAUDE_SYSTEM_INSTRUCTION", DEFAULT_CLAUDE_SYSTEM_INSTRUCTION
        )

        # Parallel tool usage instruction configuration
        self._enable_parallel_tool_instruction_claude = env_bool(
            "ANTIGRAVITY_PARALLEL_TOOL_INSTRUCTION_CLAUDE",
            True,  # ON for Claude
        )
        self._enable_parallel_tool_instruction_gemini3 = env_bool(
            "ANTIGRAVITY_PARALLEL_TOOL_INSTRUCTION_GEMINI3",
            True,  # ON for Gemini 3
        )
        self._parallel_tool_instruction = os.getenv(
            "ANTIGRAVITY_PARALLEL_TOOL_INSTRUCTION", DEFAULT_PARALLEL_TOOL_INSTRUCTION
        )

        # Tool name sanitization: sanitized_name → original_name
        # Used to fix invalid tool names (e.g., containing '/') and restore them in responses
        self._tool_name_mapping: Dict[str, str] = {}

        # Log configuration
        self._log_config()

    def _log_config(self) -> None:
        """Log provider configuration."""
        lib_logger.debug(
            f"Antigravity config: signatures_in_client={self._preserve_signatures_in_client}, "
            f"cache={self._enable_signature_cache}, dynamic_models={self._enable_dynamic_models}, "
            f"gemini3_fix={self._enable_gemini3_tool_fix}, gemini3_strict_schema={self._gemini3_enforce_strict_schema}, "
            f"claude_fix={self._enable_claude_tool_fix}, thinking_sanitization={self._enable_thinking_sanitization}, "
            f"parallel_tool_claude={self._enable_parallel_tool_instruction_claude}, "
            f"parallel_tool_gemini3={self._enable_parallel_tool_instruction_gemini3}"
        )

    def _sanitize_tool_name(self, name: str) -> str:
        """
        Sanitize tool name to comply with Antigravity API rules.

        Rules (from ANTIGRAVITY_API_SPEC.md):
        - First char must be letter (a-z, A-Z) or underscore (_)
        - Allowed chars: a-zA-Z0-9_.:-
        - Max length: 64 characters
        - Slashes (/) not allowed

        Handles collisions by appending numeric suffix (_2, _3, etc.)

        Returns sanitized name and stores mapping for later restoration.
        """
        if not name:
            return name

        original = name
        sanitized = name

        # Replace / with _ (most common issue)
        sanitized = sanitized.replace("/", "_")

        # If starts with digit, prepend underscore
        if sanitized and sanitized[0].isdigit():
            sanitized = f"_{sanitized}"

        # Truncate to 60 chars (leave room for potential suffix)
        if len(sanitized) > 60:
            sanitized = sanitized[:60]

        # Handle collisions - check if this sanitized name already maps to a DIFFERENT original
        base_sanitized = sanitized
        suffix = 2
        existing_values = set(self._tool_name_mapping.values())
        while (
            sanitized in self._tool_name_mapping
            and self._tool_name_mapping[sanitized] != original
        ) or (sanitized in existing_values and original not in existing_values):
            # Check if sanitized name is already used for a different original
            if sanitized in self._tool_name_mapping:
                if self._tool_name_mapping[sanitized] == original:
                    break  # Same original, no collision
            sanitized = f"{base_sanitized}_{suffix}"
            suffix += 1
            if suffix > 100:  # Safety limit
                lib_logger.error(f"[Tool Name] Too many collisions for '{original}'")
                break

        # Truncate again if suffix made it too long
        if len(sanitized) > 64:
            sanitized = sanitized[:64]

        # Store mapping for restoration (only if changed)
        if sanitized != original:
            self._tool_name_mapping[sanitized] = original
            lib_logger.debug(f"[Tool Name] Sanitized: '{original}' → '{sanitized}'")

        return sanitized

    def _restore_tool_name(self, sanitized_name: str) -> str:
        """Restore original tool name from sanitized version."""
        return self._tool_name_mapping.get(sanitized_name, sanitized_name)

    def _clear_tool_name_mapping(self) -> None:
        """Clear tool name mapping at start of each request."""
        self._tool_name_mapping.clear()

    def _get_antigravity_headers(self) -> Dict[str, str]:
        """Return the Antigravity API headers. Used by quota tracker mixin."""
        return ANTIGRAVITY_HEADERS

    # =========================================================================
    # MODEL UTILITIES
    # =========================================================================

    def _alias_to_internal(self, alias: str) -> str:
        """Convert public alias to internal model name."""
        return alias_to_internal_model(alias, MODEL_ALIAS_REVERSE)

    def _internal_to_alias(self, internal: str) -> str:
        """Convert internal model name to public alias."""
        return internal_to_alias_model(internal, MODEL_ALIAS_MAP, EXCLUDED_MODELS)

    def _is_gemini_3(self, model: str) -> bool:
        """Check if model is Gemini 3 (requires special handling)."""
        internal = self._alias_to_internal(model)
        return internal.startswith("gemini-3-") or model.startswith("gemini-3-")

    def _is_claude(self, model: str) -> bool:
        """Check if model is Claude."""
        return "claude" in model.lower()

    def _strip_provider_prefix(self, model: str) -> str:
        """Strip provider prefix from model name."""
        return model.split("/")[-1] if "/" in model else model

    def normalize_model_for_tracking(self, model: str) -> str:
        """
        Normalize internal Antigravity model names to public-facing names.

        Internal variants like 'claude-sonnet-4-5-thinking' are tracked under
        their public name 'claude-sonnet-4-5'. Uses the _api_to_user_model mapping.

        Args:
            model: Model name (with or without provider prefix)

        Returns:
            Normalized public-facing model name (preserves provider prefix if present)
        """
        has_prefix = "/" in model
        if has_prefix:
            provider, clean_model = model.split("/", 1)
        else:
            clean_model = model

        normalized = self._api_to_user_model(clean_model)

        if has_prefix:
            return f"{provider}/{normalized}"
        return normalized

    # =========================================================================
    # BASE URL MANAGEMENT
    # =========================================================================

    def _get_base_url(self) -> str:
        """Get current base URL."""
        return self._current_base_url

    def _get_available_models(self) -> List[str]:
        """
        Get list of user-facing model names available via this provider.

        Used by quota tracker to filter which models to store baselines for.
        Only models in this list will have quota baselines tracked.

        Returns:
            List of user-facing model names (e.g., ["claude-sonnet-4-5", "claude-opus-4-5"])
        """
        return AVAILABLE_MODELS

    def _try_next_base_url(self) -> bool:
        """Switch to next base URL in fallback list. Returns True if successful."""
        if self._base_url_index < len(BASE_URLS) - 1:
            self._base_url_index += 1
            self._current_base_url = BASE_URLS[self._base_url_index]
            lib_logger.info(f"Switching to fallback URL: {self._current_base_url}")
            return True
        return False

    def _reset_base_url(self) -> None:
        """Reset to primary base URL."""
        self._base_url_index = 0
        self._current_base_url = BASE_URLS[0]

    # =========================================================================
    # THINKING CACHE KEY GENERATION
    # =========================================================================


    async def get_valid_token(self, credential_identifier: str) -> str:
        """Get a valid access token for the credential."""
        creds = await self._load_credentials(credential_identifier)
        if self._is_token_expired(creds):
            creds = await self._refresh_token(credential_identifier, creds)
        return creds["access_token"]

    def has_custom_logic(self) -> bool:
        """Antigravity uses custom translation logic."""
        return True

    async def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        """Get OAuth authorization header."""
        token = await self.get_valid_token(credential_identifier)
        return {"Authorization": f"Bearer {token}"}

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """Fetch available models from Antigravity."""
        if not self._enable_dynamic_models:
            lib_logger.debug("Using hardcoded model list")
            return [f"antigravity/{m}" for m in AVAILABLE_MODELS]

        try:
            token = await self.get_valid_token(api_key)
            url = f"{self._get_base_url()}/fetchAvailableModels"

            headers = {
                **build_bearer_headers(token),
                **ANTIGRAVITY_HEADERS,
            }
            payload = {
                "project": _generate_project_id(),
                "requestId": _generate_request_id(),
                "userAgent": "antigravity",
                "requestType": "agent",  # Required per CLIProxyAPI commit 67985d8
            }

            response = await client.post(
                url, json=payload, headers=headers, timeout=TimeoutConfig.provider_request()
            )
            try:
                response.raise_for_status()
                data = response.json()
            except (httpx.HTTPStatusError, json.JSONDecodeError, ValueError) as e:
                body_preview = response.text[:200] if response.text else "<empty>"
                lib_logger.warning("OAuth/HTTP error in models discovery: %s — body: %s", e, body_preview)
                raise

            models = []
            for model_info in data.get("models", []):
                internal = model_info.get("name", "").replace("models/", "")
                if internal:
                    public = self._internal_to_alias(internal)
                    if public:
                        models.append(f"antigravity/{public}")

            if models:
                lib_logger.info(f"Discovered {len(models)} models")
                return models
        except httpx.HTTPError as e:
            lib_logger.warning(f"Dynamic model discovery failed: {e}")

        return [f"antigravity/{m}" for m in AVAILABLE_MODELS]

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """
        Handle completion requests for Antigravity.

        Main entry point that:
        1. Extracts parameters and transforms messages
        2. Builds Antigravity request payload
        3. Makes API call with fallback logic
        4. Transforms response to OpenAI format
        """
        # Clear tool name mapping for fresh request
        self._clear_tool_name_mapping()

        # Extract parameters
        model = self._strip_provider_prefix(kwargs.get("model", "gemini-2.5-pro"))
        messages = kwargs.get("messages", [])
        stream = kwargs.get("stream", False)
        credential_path = kwargs.pop("credential_identifier", kwargs.get("api_key", ""))
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")
        reasoning_effort = cast(Optional[str], kwargs.get("reasoning_effort"))
        top_p = kwargs.get("top_p")
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")
        transaction_context = kwargs.pop("transaction_context", None)

        # Create provider logger from transaction context
        file_logger = AntigravityProviderLogger(transaction_context)

        # Determine if thinking is enabled for this request
        # Thinking is enabled if:
        # 1. Model is a thinking model (opus or -thinking suffix) - ALWAYS enabled, cannot be disabled
        # 2. For non-thinking models: reasoning_effort is set and not explicitly disabled
        thinking_enabled = False
        if self._is_claude(model):
            model_lower = model.lower()

            # Check if this is a thinking model by name (opus or -thinking suffix)
            is_thinking_model = "opus" in model_lower or "-thinking" in model_lower

            if is_thinking_model:
                # Thinking models ALWAYS have thinking enabled - cannot be disabled
                thinking_enabled = True
                # Note: invalid disable requests in reasoning_effort are handled later
            else:
                # Non-thinking models - reasoning_effort controls thinking
                if reasoning_effort is not None:
                    if isinstance(reasoning_effort, str):
                        effort_lower = reasoning_effort.lower().strip()
                        if effort_lower in ("disable", "none", "off", ""):
                            thinking_enabled = False
                        else:
                            thinking_enabled = True
                    elif isinstance(reasoning_effort, (int, float)):
                        # Numeric: enabled if > 0
                        thinking_enabled = float(reasoning_effort) > 0
                    else:
                        thinking_enabled = True

        # Transform messages to Gemini format FIRST
        # This restores thinking from cache if reasoning_content was stripped by client
        system_instruction, gemini_contents = await self._transform_messages(messages, model)
        gemini_contents = self._fix_tool_response_grouping(gemini_contents)

        # Sanitize thinking blocks for Claude AFTER transformation
        # Now we can see the full picture including cached thinking that was restored
        # This handles: context compression, model switching, mid-turn thinking toggle
        force_disable_thinking = False
        if self._is_claude(model) and self._enable_thinking_sanitization:
            gemini_contents, force_disable_thinking = (
                await self._sanitize_thinking_for_claude(gemini_contents, thinking_enabled)
            )

            # If we're in a mid-turn thinking toggle situation, we MUST disable thinking
            # for this request. Thinking will naturally resume on the next turn.
            if force_disable_thinking:
                thinking_enabled = False
                reasoning_effort = "disable"  # Force disable for this request

        # Clean up any empty messages left by stripping/recovery operations
        gemini_contents = self._remove_empty_messages(gemini_contents)

        # Inject interleaved thinking reminder to last real user message
        # Only if thinking is enabled and tools are present
        if (
            ENABLE_INTERLEAVED_THINKING
            and thinking_enabled
            and self._is_claude(model)
            and tools
        ):
            gemini_contents = self._inject_interleaved_thinking_reminder(
                gemini_contents
            )

        # Build payload
        gemini_payload: Dict[str, Any] = {"contents": gemini_contents}

        if system_instruction:
            gemini_payload["system_instruction"] = system_instruction

        # Inject tool usage hardening system instructions
        if tools:
            if self._is_gemini_3(model) and self._enable_gemini3_tool_fix:
                self._inject_tool_hardening_instruction(
                    gemini_payload, self._gemini3_system_instruction
                )
            elif self._is_claude(model) and self._enable_claude_tool_fix:
                self._inject_tool_hardening_instruction(
                    gemini_payload, self._claude_system_instruction
                )

            # Inject parallel tool usage encouragement (independent of tool hardening)
            if self._is_claude(model) and self._enable_parallel_tool_instruction_claude:
                self._inject_tool_hardening_instruction(
                    gemini_payload, self._parallel_tool_instruction
                )
            elif (
                self._is_gemini_3(model)
                and self._enable_parallel_tool_instruction_gemini3
            ):
                self._inject_tool_hardening_instruction(
                    gemini_payload, self._parallel_tool_instruction
                )

            # Inject interleaved thinking hint for Claude thinking models with tools
            if (
                ENABLE_INTERLEAVED_THINKING
                and self._is_claude(model)
                and thinking_enabled
            ):
                self._inject_tool_hardening_instruction(
                    gemini_payload, CLAUDE_INTERLEAVED_THINKING_HINT
                )

        # Add generation config
        gen_config = {}
        if top_p is not None:
            gen_config["topP"] = top_p

        # Handle temperature - Gemini 3 defaults to 1 if not explicitly set
        if temperature is not None:
            gen_config["temperature"] = temperature
        elif self._is_gemini_3(model):
            # Gemini 3 performs better with temperature=1 for tool use
            gen_config["temperature"] = 1.0

        thinking_config = self._get_thinking_config(cast(Optional[str], reasoning_effort), model)
        if thinking_config:
            gen_config.setdefault("thinkingConfig", {}).update(thinking_config)

        if gen_config:
            gemini_payload["generationConfig"] = gen_config

        # Add tools
        gemini_tools = self._build_tools_payload(tools, model)

        if gemini_tools:
            gemini_payload["tools"] = gemini_tools

            # Apply tool transformations
            if self._is_gemini_3(model) and self._enable_gemini3_tool_fix:
                # Gemini 3: namespace prefix + strict schema + parameter signatures
                # Tools are freshly built by _build_tools_payload — no other
                # reference exists, so in-place mutation is safe (skip deepcopy)
                gemini_payload["tools"] = self._apply_gemini3_namespace(
                    gemini_payload["tools"], copy_tools=False
                )

                if self._gemini3_enforce_strict_schema:
                    gemini_payload["tools"] = self._enforce_strict_schema_on_tools(
                        gemini_payload["tools"], copy_tools=False
                    )
                gemini_payload["tools"] = self._inject_signature_into_descriptions(
                    gemini_payload["tools"], self._gemini3_description_prompt, copy_tools=False
                )
            elif self._is_claude(model) and self._enable_claude_tool_fix:
                # Claude: parameter signatures only (no namespace prefix)
                # Freshly built tools — safe to mutate in place
                gemini_payload["tools"] = self._inject_signature_into_descriptions(
                    gemini_payload["tools"], self._claude_description_prompt, copy_tools=False
                )

        # Get access token first (needed for project discovery)
        token = await self.get_valid_token(credential_path)

        # Discover real project ID
        litellm_params = kwargs.get("litellm_params", {}) or {}
        project_id = await self._discover_project_id(
            credential_path, token, litellm_params
        )

        # Transform to Antigravity format with real project ID
        payload = self._transform_to_antigravity_format(
            gemini_payload, model, project_id, max_tokens, reasoning_effort, tool_choice
        )
        await file_logger.log_request(payload)

        # Pre-build tool schema map for malformed call handling
        # This maps original tool names (without prefix) to their schemas
        tool_schemas = self._build_tool_schema_map(gemini_payload.get("tools"), model)

        # Make API call - always use streaming endpoint internally
        # For stream=False, we collect chunks into a single response
        base_url = self._get_base_url()
        endpoint = ":streamGenerateContent"
        url = f"{base_url}{endpoint}?alt=sse"

        # These headers are REQUIRED for gemini-3-pro-high/low to work
        # Without X-Goog-Api-Client and Client-Metadata, only gemini-3-pro-preview works
        headers = {
            **build_bearer_headers(token),
            "Accept": "text/event-stream",
            **ANTIGRAVITY_HEADERS,
        }

        # Keep a mutable reference to gemini_contents for retry injection
        current_gemini_contents = gemini_contents

        # URL fallback loop - handles HTTP errors (except 429) and network errors
        # by switching to fallback URLs. Empty response retry is handled inside
        # _streaming_with_retry.
        while True:
            try:
                # Always use streaming internally - _streaming_with_retry handles
                # empty responses, bare 429s, and malformed function calls
                streaming_generator = self._streaming_with_retry(
                    client,
                    url,
                    headers,
                    payload,
                    model,
                    file_logger,
                    tool_schemas,
                    current_gemini_contents,
                    gemini_payload,
                    project_id,
                    max_tokens,
                    cast(Optional[str], reasoning_effort),
                    tool_choice,
                )

                if stream:
                    # Client requested streaming - return generator directly
                    return streaming_generator
                else:
                    # Client requested non-streaming - collect chunks into single response
                    return await self._collect_streaming_chunks(
                        streaming_generator, model, file_logger
                    )

            except httpx.HTTPStatusError as e:
                # 429 = Rate limit/quota exhausted - tied to credential, not URL
                # Do NOT retry on different URL, just raise immediately
                if e.response.status_code == 429:
                    lib_logger.debug(
                        f"429 quota error - not retrying on fallback URL: {e}"
                    )
                    raise

                # Other HTTP errors (403, 500, etc.) - try fallback URL
                if self._try_next_base_url():
                    lib_logger.warning(f"Retrying with fallback URL: {e}")
                    url = f"{self._get_base_url()}{endpoint}?alt=sse"
                    continue  # Retry with new URL
                raise  # No more fallback URLs

            except (EmptyResponseError, TransientQuotaError):
                # Already retried internally - don't catch, propagate for credential rotation
                raise

            except (httpx.HTTPError, ValueError, KeyError) as e:
                # Non-HTTP errors (network issues, timeouts, etc.) - try fallback URL
                if self._try_next_base_url():
                    lib_logger.warning(f"Retrying with fallback URL: {e}")
                    url = f"{self._get_base_url()}{endpoint}?alt=sse"
                    continue  # Retry with new URL
                raise  # No more fallback URLs
