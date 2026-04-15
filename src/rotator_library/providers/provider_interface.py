# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    List,
    Dict,
    Any,
    Optional,
    AsyncGenerator,
    Union,
    FrozenSet,
    Tuple,
)
import re
import orjson

import logging
import os
import httpx
import litellm


from ..config import (
    DEFAULT_ROTATION_MODE,
    DEFAULT_TIER_PRIORITY,
    DEFAULT_SEQUENTIAL_FALLBACK_MULTIPLIER,
    DEFAULT_FAIR_CYCLE_ENABLED,
    DEFAULT_FAIR_CYCLE_TRACKING_MODE,
    DEFAULT_FAIR_CYCLE_CROSS_TIER,
    DEFAULT_FAIR_CYCLE_DURATION,
    DEFAULT_EXHAUSTION_COOLDOWN_THRESHOLD,
)


# =============================================================================
# SHARED PROVIDER UTILITIES
# =============================================================================


def strip_provider_prefix(model: str) -> str:
    """Strip the provider prefix from a model name ("nanogpt/gpt-4o" -> "gpt-4o")."""
    return model.split("/")[-1] if "/" in model else model


def build_bearer_headers(token: str, content_type: str = "application/json") -> dict:
    """Build {"Authorization": "Bearer {token}", "Content-Type": content_type}."""
    return {"Authorization": f"Bearer {token}", "Content-Type": content_type}


# =============================================================================
# TIER & USAGE CONFIGURATION TYPES
# =============================================================================


@dataclass(frozen=True)
class UsageResetConfigDef:
    """Definition for usage reset configuration per tier type.

    Attributes:
        window_seconds: Duration of the usage tracking window in seconds.
        mode: "credential" (one window per credential) or "per_model" (separate window per model).
        description: Human-readable description for logging.
        field_name: Key in usage data JSON (e.g. "models" for per_model, "daily" for credential).
    """

    window_seconds: int
    mode: str  # "credential" or "per_model"
    description: str
    field_name: str = "daily"


# Type aliases for provider configuration
TierPriorityMap = Dict[str, int]  # tier_name -> priority
UsageConfigKey = Union[FrozenSet[int], str]  # frozenset of priorities OR "default"
UsageConfigMap = Dict[UsageConfigKey, UsageResetConfigDef]  # priority_set -> config
QuotaGroupMap = Dict[str, List[str]]  # group_name -> [models]


class ProviderInterface(ABC):
    """
    An interface for API provider-specific functionality, including model
    discovery and custom API call handling for non-standard providers.
    """

    skip_cost_calculation: bool = False

    # "balanced" = distribute load; "sequential" = exhaust one key then switch
    default_rotation_mode: str = DEFAULT_ROTATION_MODE

    # --- Tier configuration (override in subclass) ---

    # For env var lookups: QUOTA_GROUPS_{provider_env_name}_{GROUP}
    provider_env_name: str = ""
    # tier_name -> priority (1 = highest); unknowns fall back to default_tier_priority
    tier_priorities: TierPriorityMap = {}
    default_tier_priority: int = DEFAULT_TIER_PRIORITY

    # --- Usage reset configuration (override in subclass) ---

    # Keys: frozenset of priorities OR "default"
    usage_reset_configs: UsageConfigMap = {}

    # --- Model quota groups (override in subclass) ---

    # Models sharing cooldown timing. Env: QUOTA_GROUPS_{PROVIDER}_{GROUP}="m1,m2"
    model_quota_groups: QuotaGroupMap = {}
    # Per-model weight when calculating grouped usage (default 1). E.g. {"opus": 2}
    model_usage_weights: Dict[str, int] = {}

    # --- Priority concurrency multipliers (override in subclass) ---

    # priority -> multiplier. E.g. {1: 5, 2: 3}
    default_priority_multipliers: Dict[int, int] = {}
    default_sequential_fallback_multiplier: int = DEFAULT_SEQUENTIAL_FALLBACK_MULTIPLIER

    # --- Fair cycle rotation (override in subclass) ---
    # Fair cycle: each credential used once before reuse; "exhausted" = long cooldown.

    # None = derive from rotation mode. Env: FAIR_CYCLE_{PROVIDER}=true/false
    default_fair_cycle_enabled: Optional[bool] = DEFAULT_FAIR_CYCLE_ENABLED
    # "model_group" | "credential". Env: FAIR_CYCLE_TRACKING_MODE_{PROVIDER}
    default_fair_cycle_tracking_mode: str = DEFAULT_FAIR_CYCLE_TRACKING_MODE
    # False = per-tier; True = all must exhaust. Env: FAIR_CYCLE_CROSS_TIER_{PROVIDER}
    default_fair_cycle_cross_tier: bool = DEFAULT_FAIR_CYCLE_CROSS_TIER
    # Env: FAIR_CYCLE_DURATION_{PROVIDER}=<seconds>
    default_fair_cycle_duration: int = DEFAULT_FAIR_CYCLE_DURATION
    # Min cooldown to count as "exhausted". Env: EXHAUSTION_COOLDOWN_THRESHOLD_{PROVIDER}
    default_exhaustion_cooldown_threshold: int = DEFAULT_EXHAUSTION_COOLDOWN_THRESHOLD

    # --- Custom caps (override in subclass) ---
    # Keys: int|tuple[int,...]|"default". Values: {"max_requests": int|str, "cooldown_mode": str, ...}
    # Env: CUSTOM_CAP_{PROVIDER}_T{TIER}_{MODEL}, CUSTOM_CAP_COOLDOWN_...
    default_custom_caps: Dict[
        Union[int, Tuple[int, ...], str], Dict[str, Dict[str, Any]]
    ] = {}

    @abstractmethod
    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """Fetch the list of available model names from the provider's API."""
        pass

    def has_custom_logic(self) -> bool:
        """Returns True if the provider implements its own acompletion/aembedding logic."""
        return False

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """Handles the entire completion call for non-standard providers."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement custom acompletion."
        )

    async def aembedding(
        self, client: httpx.AsyncClient, **kwargs
    ) -> litellm.EmbeddingResponse:
        """Handles the entire embedding call for non-standard providers."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement custom aembedding."
        )

    async def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        """For OAuth providers, returns the Authorization header dict."""
        raise NotImplementedError("This provider does not support OAuth.")

    async def proactively_refresh(self, credential_path: str):
        """Proactively refresh a token if it's nearing expiry."""
        pass

    def _resolve_tier_priority(self, tier_name: Optional[str]) -> int:
        """Resolve priority for a tier name using provider's tier_priorities mapping."""
        if tier_name is None:
            return self.default_tier_priority
        return self.tier_priorities.get(tier_name, self.default_tier_priority)

    def get_credential_priority(self, credential: str) -> Optional[int]:
        """Resolve credential priority from tier name via tier_priorities mapping.

        Subclasses define get_credential_tier_name() for tier lookup;
        do NOT override this method.
        """
        tier_fn = getattr(self, "get_credential_tier_name", None)
        tier = tier_fn(credential) if tier_fn else None
        if tier is None:
            return None
        return self._resolve_tier_priority(tier)

    async def initialize_credentials(self, credential_paths: List[str]) -> None:
        """Called at startup to initialize provider with all available credentials."""
        pass

    # --- Sequential Rotation Support ---

    @classmethod
    def get_rotation_mode(cls, provider_name: str) -> str:
        """Get rotation mode: checks ROTATION_MODE_{PROVIDER} env, then default_rotation_mode."""
        return os.getenv(f"ROTATION_MODE_{provider_name.upper()}", cls.default_rotation_mode)

    # Providers set this to a list of pattern specs for quota error matching.
    # Each spec is one of:
    #   ("json", json_path, expected_value, default_retry_secs, reason)
    #     Dotted path navigates nested dicts; path ending "*" = substring match.
    #   ("extract", json_path, reason)
    #     Navigate to json_path, use the numeric value there as retry_after.
    #   ("body", keyword, default_retry_secs, reason)
    #     Case-insensitive keyword search in the raw body text.
    #
    # The generic parser also runs extract_retry_after_from_body() first;
    # if it finds a duration, that overrides default_retry_secs.
    _quota_error_patterns: Optional[list] = None

    @staticmethod
    def _extract_error_body(error: Exception, error_body: Optional[str] = None) -> Optional[str]:
        """Extract error body text from various exception shapes."""
        body = error_body
        if body:
            return body
        if hasattr(error, "response") and hasattr(error.response, "text"):
            try:
                return error.response.text
            except Exception:
                logging.debug("Failed to extract error response text", exc_info=True)
        if hasattr(error, "body") and error.body:
            return str(error.body)
        if hasattr(error, "message"):
            return str(error.message)
        return None

    @staticmethod
    def _navigate_json(data: Any, path: str) -> Any:
        """Navigate *data* along dotted *path*, returning the value or None."""
        node = data
        for part in path.split("."):
            if isinstance(node, dict):
                node = node.get(part)
            else:
                return None
        return node

    @classmethod
    def _match_quota_pattern(cls, data: Any, path: str, expected: Any) -> bool:
        """Navigate *data* along dotted *path* and compare to *expected*. Path ending "*" = substring match."""
        is_substring = path.endswith("*")
        node = cls._navigate_json(data, path.rstrip("*"))
        if node is None:
            return False
        if is_substring:
            return str(expected).lower() in str(node).lower()
        return node == expected

    @classmethod
    def parse_quota_error(
        cls, error: Exception, error_body: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Parse a quota/rate-limit error and extract structured information.

        When _quota_error_patterns is defined, performs generic pattern matching.
        Providers with complex logic override this method entirely.

        Returns None if not a parseable quota error, otherwise dict with:
        retry_after, reason, reset_timestamp, quota_reset_timestamp.
        """
        patterns = getattr(cls, "_quota_error_patterns", None)
        if not patterns:
            return None

        body = cls._extract_error_body(error, error_body)
        if not body:
            return None

        from ..error_handler import extract_retry_after_from_body

        retry_after = extract_retry_after_from_body(body)

        data = None
        try:
            json_match = re.search(r"\{[\s\S]*\}", body)
            if json_match:
                data = orjson.loads(json_match.group(0))
        except (orjson.JSONDecodeError, ValueError):
            pass

        for spec in patterns:
            kind = spec[0]

            if kind == "json":
                _, path, expected, default_retry, reason = spec
                if data is not None and cls._match_quota_pattern(data, path, expected):
                    final_retry = retry_after if retry_after else default_retry
                    return {"retry_after": final_retry, "reason": reason}

            elif kind == "extract":
                _, path, reason = spec
                if data is not None:
                    val = cls._navigate_json(data, path)
                    if val is not None:
                        try:
                            return {"retry_after": int(val), "reason": reason}
                        except (ValueError, TypeError):
                            pass

            elif kind == "body":
                _, keyword, default_retry, reason = spec
                if keyword.lower() in body.lower():
                    final_retry = retry_after if retry_after else default_retry
                    return {"retry_after": final_retry, "reason": reason}

        if retry_after:
            return {"retry_after": retry_after, "reason": "RATE_LIMIT_EXCEEDED"}

        return None

    # --- Usage Reset Config ---

    def get_usage_reset_config(self, credential: str) -> Optional[Dict[str, Any]]:
        """
        Get provider-specific usage tracking configuration for a credential.

        Resolves tier via get_credential_tier_name(), maps to priority via
        tier_priorities, then finds matching UsageResetConfigDef.

        Subclasses should define usage_reset_configs as a class attribute
        instead of overriding this method.

        Modes:
            - "credential": One window per credential; all models reset together.
            - "per_model": Separate window per model; models reset independently.

        Args:
            credential: The credential identifier (API key or path)

        Returns:
            None to use default daily reset, otherwise dict with:
            window_seconds, mode, priority, description, field_name.
        """
        if not self.usage_reset_configs:
            return None

        tier_fn = getattr(self, "get_credential_tier_name", None)
        tier = tier_fn(credential) if tier_fn else None
        priority = self._resolve_tier_priority(tier)

        # Check frozenset keys for explicit priority match, then "default"
        config: Optional[UsageResetConfigDef] = None
        for key, cfg in self.usage_reset_configs.items():
            if isinstance(key, frozenset) and priority in key:
                config = cfg
                break
        if config is None:
            config = self.usage_reset_configs.get("default")
        if config is None:
            return None

        return {
            "window_seconds": config.window_seconds,
            "mode": config.mode,
            "priority": priority,
            "description": config.description,
            "field_name": config.field_name,
        }

    def get_default_usage_field_name(self) -> str:
        """
        Get the default usage tracking field name for this provider.

        Providers can override this to use a custom field name for usage tracking
        when no credential-specific config is available.

        Returns:
            Field name string (default: "daily")
        """
        return "daily"

    # --- Quota Groups Logic ---

    def _get_effective_quota_groups(self) -> QuotaGroupMap:
        """Get quota groups with .env overrides applied.

        Env format: QUOTA_GROUPS_{PROVIDER}_{GROUP}="model1,model2"
        Empty string disables a default group.
        """
        if not self.provider_env_name or not self.model_quota_groups:
            return self.model_quota_groups

        result: QuotaGroupMap = {}
        for group_name, default_models in self.model_quota_groups.items():
            env_key = (
                f"QUOTA_GROUPS_{self.provider_env_name.upper()}_{group_name.upper()}"
            )
            env_value = os.getenv(env_key)
            if env_value is not None:
                if env_value.strip():
                    result[group_name] = [
                        m.strip() for m in env_value.split(",") if m.strip()
                    ]
            else:
                result[group_name] = list(default_models)
        return result

    def get_model_quota_group(self, model: str) -> Optional[str]:
        """Returns the quota group name for a model, or None if not grouped.

        Models in the same group share cooldown timing and reset together.
        Subclasses should define model_quota_groups as a class attribute
        instead of overriding this method.
        """
        clean_model = strip_provider_prefix(model)
        groups = self._get_effective_quota_groups()
        for group_name, models in groups.items():
            if clean_model in models:
                return group_name
        return None

    def get_models_in_quota_group(self, group: str) -> List[str]:
        """Returns all model names in a quota group (without provider prefix)."""
        return self._get_effective_quota_groups().get(group, [])

    def get_model_usage_weight(self, model: str) -> int:
        """Returns usage weight for a model in grouped usage (default 1)."""
        return self.model_usage_weights.get(strip_provider_prefix(model), 1)

    def normalize_model_for_tracking(self, model: str) -> str:
        """Normalize internal model names to public-facing names for usage tracking.

        Default: returns model unchanged. Override for providers with internal variants.
        """
        return model

