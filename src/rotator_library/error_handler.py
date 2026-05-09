# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import os
import logging
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from .config.defaults import COOLDOWN_RATE_LIMIT_DEFAULT
from .config.defaults import INCEPTION_BACKOFF_BASE, INCEPTION_MAX_BACKOFF
from .config.defaults import KILOCODE_BACKOFF_BASE, KILOCODE_MAX_BACKOFF
from .config.defaults import PROXY_PROVIDERS
from .error_handling.classifier import (
    classify_error,
    classify_rate_limit as _classify_rate_limit,
    classify_stream_error,
    is_provider_abort,
)
from .error_handling.quality_validator import validate_response_quality
from .error_handling.quota_parser import (
    detect_ip_throttle,
    extract_quota_details as _extract_quota_details,
    try_parse_provider_quota_error as _try_parse_provider_quota_error,
)
from .error_handling.retry_after import (
    _extract_retry_from_json_body,
    extract_retry_after_from_body,
    get_retry_after,
)
from .error_types import (
    ClassifiedError,
    CredentialNeedsReauthError,
    EmptyResponseError,
    GarbageResponseError,
    PreRequestCallbackError,
    TransientQuotaError,
    mask_credential,
)
from .ip_throttle_detector import IPThrottleDetector, ThrottleScope
from .utils.http_retry import compute_backoff_with_jitter

if TYPE_CHECKING:
    from .circuit_breaker import ProviderCircuitBreaker
    from .cooldown_manager import CooldownManager

lib_logger = logging.getLogger("rotator_library")

RATE_LIMIT_DEFAULT_COOLDOWN = COOLDOWN_RATE_LIMIT_DEFAULT
_detect_ip_throttle = detect_ip_throttle

_NON_ROTATABLE_ERRORS = frozenset(
    {
        "invalid_request",
        "context_window_exceeded",
        "pre_request_callback_error",
        "ip_rate_limit",
    }
)

_RETRYABLE_SAME_KEY_ERRORS = frozenset(
    {
        "server_error",
        "api_connection",
        "ip_rate_limit",
    }
)

# =============================================================================
# Provider-Specific Backoff Configuration
# =============================================================================
# Allows tuning retry behavior per provider for better resilience.

PROVIDER_BACKOFF_CONFIGS: Dict[str, Dict[str, float]] = {
    "kilocode": {
        "server_error_base": KILOCODE_BACKOFF_BASE,
        "connection_base": 0.5,
        "max_backoff": KILOCODE_MAX_BACKOFF,
    },
    "friendli": {  # z-ai uses Friendli backend
        "server_error_base": 1.5,
        "connection_base": 0.5,
        "max_backoff": 20.0,
    },
    "inception": {
        "server_error_base": INCEPTION_BACKOFF_BASE,
        "connection_base": 1.0,
        "max_backoff": INCEPTION_MAX_BACKOFF,
        "max_retries": 5,
    },
}

_BACKOFF_CONFIG_CACHE: dict[str, dict] = {}
_BACKOFF_ENV_CACHE: dict[str, Tuple[Optional[str], Optional[str]]] = {}


def _get_provider_backoff_config(provider: Optional[str]) -> Dict[str, float]:
    """
    Get backoff config for a provider, with env var overrides.

    Env vars:
        KILOCODE_BACKOFF_BASE - base multiplier for server errors
        KILOCODE_MAX_BACKOFF - maximum backoff in seconds

    Returns:
        Dict with server_error_base, connection_base, max_backoff
    """
    if not provider:
        return {}

    env_values: Tuple[Optional[str], Optional[str]]
    if provider == "kilocode":
        env_values = (
            os.environ.get("KILOCODE_BACKOFF_BASE"),
            os.environ.get("KILOCODE_MAX_BACKOFF"),
        )
    elif provider == 'inception':
        env_values = (
            os.environ.get('INCEPTION_BACKOFF_BASE'),
            os.environ.get('INCEPTION_MAX_BACKOFF'),
        )
    else:
        env_values = (None, None)

    cached = _BACKOFF_CONFIG_CACHE.get(provider)
    if cached is not None and _BACKOFF_ENV_CACHE.get(provider) == env_values:
        return cached

    config = PROVIDER_BACKOFF_CONFIGS.get(provider, {}).copy()

    # Env var overrides for kilocode
    if provider == "kilocode":
        backoff_base = env_values[0]
        max_backoff = env_values[1]
        if backoff_base is not None:
            try:
                config["server_error_base"] = float(backoff_base)
            except ValueError:
                lib_logger.debug("Invalid KILOCODE_BACKOFF_BASE value")
        if max_backoff is not None:
            try:
                config["max_backoff"] = float(max_backoff)
            except ValueError:
                lib_logger.debug("Invalid KILOCODE_MAX_BACKOFF value")

    # Env var overrides for inception
    if provider == 'inception':
        backoff_base = env_values[0]
        max_backoff = env_values[1]
        if backoff_base is not None:
            try:
                config['server_error_base'] = float(backoff_base)
            except ValueError:
                lib_logger.debug("Invalid INCEPTION_BACKOFF_BASE value")
        if max_backoff is not None:
            try:
                config['max_backoff'] = float(max_backoff)
            except ValueError:
                lib_logger.debug("Invalid INCEPTION_MAX_BACKOFF value")

    _BACKOFF_CONFIG_CACHE[provider] = config
    _BACKOFF_ENV_CACHE[provider] = env_values
    return config


def get_retry_backoff(
    classified_error: "ClassifiedError", attempt: int, provider: Optional[str] = None
) -> float:
    """
    Calculate retry backoff time based on error type and attempt number.

    Different strategies for different error types:
    - api_connection: More aggressive retry (network issues are transient)
    - server_error: Standard exponential backoff
    - rate_limit: Use retry_after if available, otherwise shorter default
    - ip_rate_limit: Use retry_after (from detection) or default cooldown

    Args:
        classified_error: The classified error with type and retry_after
        attempt: Current retry attempt number (0-indexed)
        provider: Optional provider name for provider-specific tuning

    Returns:
        Backoff time in seconds
    """
    # If provider specified retry_after, use it
    if classified_error.retry_after:
        return classified_error.retry_after

    error_type = classified_error.error_type

    # Provider-specific config
    config = _get_provider_backoff_config(provider)
    max_backoff = config.get("max_backoff", 60.0)

    if error_type == "api_connection":
        # More aggressive retry for network errors - they're usually transient
        base = config.get("connection_base", 0.5)
        backoff = compute_backoff_with_jitter(attempt, base=1.5, max_wait=max_backoff, jitter=0.3, min_wait=float(base))
    elif error_type == "server_error":
        # Standard exponential backoff with provider-specific base
        base = config.get("server_error_base", 2.0)
        backoff = compute_backoff_with_jitter(attempt, base=base, max_wait=max_backoff, jitter=0.3)
    elif error_type == "rate_limit":
        # Short default for transient rate limits without retry_after
        backoff = compute_backoff_with_jitter(attempt, base=2.0, max_wait=max_backoff, jitter=0.3, retry_after=5.0)
    elif error_type == "ip_rate_limit":
        # IP throttle - use default cooldown with jitter
        backoff = compute_backoff_with_jitter(attempt, base=2.0, max_wait=max_backoff, jitter=0.3, retry_after=float(RATE_LIMIT_DEFAULT_COOLDOWN))
    else:
        # Default backoff
        backoff = compute_backoff_with_jitter(attempt, base=2.0, max_wait=max_backoff, jitter=0.3)

    return backoff


# =============================================================================
# Unified 429 Error Handler
# =============================================================================

class ThrottleActionType(Enum):
    """Actions to take after processing a 429 error."""

    CREDENTIAL_COOLDOWN = "credential_cooldown"  # Single credential throttled
    PROVIDER_COOLDOWN = "provider_cooldown"  # IP-level throttle detected
    FAIL_IMMEDIATELY = "fail_immediately"  # Non-recoverable (should not happen for 429)


@dataclass
class ThrottleAction:
    """
    Result of processing a 429 error with unified handling.

    This dataclass consolidates all decisions about what to do after a 429:
    - What action to take (credential vs provider cooldown)
    - How long to wait
    - Whether to open the circuit breaker
    - Related metadata for logging/debugging
    """

    action_type: ThrottleActionType
    cooldown_seconds: int = 0
    open_circuit_breaker: bool = False
    throttle_scope: ThrottleScope = ThrottleScope.CREDENTIAL
    confidence: float = 0.0
    affected_credentials: list = dataclass_field(default_factory=list)
    reason: str = ""

    def __str__(self) -> str:
        return (
            f"ThrottleAction(action={self.action_type.value}, "
            f"cooldown={self.cooldown_seconds}s, "
            f"circuit_breaker={self.open_circuit_breaker}, "
            f"scope={self.throttle_scope.value})"
        )


async def handle_429_error(
    provider: str,
    credential: str,
    error: Exception,
    error_body: Optional[str] = None,
    retry_after: Optional[int] = None,
    ip_throttle_detector: Optional["IPThrottleDetector"] = None,
    circuit_breaker: Optional["ProviderCircuitBreaker"] = None,
    cooldown_manager: Optional["CooldownManager"] = None,
) -> ThrottleAction:
    """
    Unified handler for 429 rate limit errors.

    This function consolidates all 429 processing logic:
    1. Detects IP-level vs credential-level throttle
    2. Determines appropriate cooldown duration
    3. Decides whether to open circuit breaker
    4. Returns a ThrottleAction with all decisions

    If circuit_breaker and cooldown_manager are provided, actions are applied
    automatically. Otherwise, the caller must apply them based on the returned
    ThrottleAction.
    """
    # Get or create detector
    detector = ip_throttle_detector if ip_throttle_detector is not None else IPThrottleDetector()

    # Step 1: Check for explicit IP throttle indicators in error body
    ip_throttle_from_body = detect_ip_throttle(error_body, provider=provider)

    if ip_throttle_from_body is not None:
        # Error body explicitly indicates IP-level throttle
        cooldown = retry_after or ip_throttle_from_body
        lib_logger.warning(
            "IP-level throttle detected for provider '%s' from error body. "
            "Blocking provider for %ss.",
            provider, cooldown,
        )
        action = ThrottleAction(
            action_type=ThrottleActionType.PROVIDER_COOLDOWN,
            cooldown_seconds=cooldown,
            open_circuit_breaker=True,
            throttle_scope=ThrottleScope.IP,
            confidence=1.0,  # High confidence from explicit error body
            reason="IP-level throttle detected from error body",
        )
        # Auto-apply if managers provided
        if circuit_breaker is not None:
            await circuit_breaker.open_immediately(
                provider, reason=action.reason, duration=action.cooldown_seconds
            )
        # Apply per-credential cooldowns to all affected credentials
        # so they are properly tracked when the circuit recovers
        if cooldown_manager is not None and action.affected_credentials:
            for affected_cred in action.affected_credentials:
                await cooldown_manager.start_cooldown(affected_cred, action.cooldown_seconds)
        return action

    # Step 2: For PROXY_PROVIDERS, skip IP throttle correlation entirely.
    # These providers aggregate multiple backends or use shared quotas;
    # 429 on multiple keys does NOT mean IP-level throttle.
    if provider and provider in PROXY_PROVIDERS:
        cooldown = retry_after or RATE_LIMIT_DEFAULT_COOLDOWN
        lib_logger.debug(
            "Skipping IP throttle correlation for proxy provider '%s'. "
            "Credential-level cooldown: %ss.",
            provider, cooldown,
        )
        action = ThrottleAction(
            action_type=ThrottleActionType.CREDENTIAL_COOLDOWN,
            cooldown_seconds=cooldown,
            open_circuit_breaker=False,
            throttle_scope=ThrottleScope.CREDENTIAL,
            confidence=0.8,
            reason="Credential-level rate limit (proxy provider, IP correlation skipped)",
        )
        if cooldown_manager is not None:
            await cooldown_manager.start_cooldown(credential, action.cooldown_seconds)
        return action

    # Step 3: Record 429 and correlate with other credentials
    assessment = await detector.record_429(
        provider=provider,
        credential=mask_credential(credential),
        error_body=error_body,
        retry_after=retry_after,
    )

    # Step 4: Determine action based on assessment scope
    cooldown = max(retry_after or 0, assessment.suggested_cooldown)
    if cooldown == 0:
        cooldown = RATE_LIMIT_DEFAULT_COOLDOWN

    if assessment.scope == ThrottleScope.IP:
        # Multiple credentials throttled - IP-level
        lib_logger.warning(
            "IP-level throttle detected for provider '%s' via correlation: "
            "%s credentials affected, confidence=%.2f. Blocking provider for %ss.",
            provider, len(assessment.affected_credentials), assessment.confidence, cooldown,
        )
        action = ThrottleAction(
            action_type=ThrottleActionType.PROVIDER_COOLDOWN,
            cooldown_seconds=cooldown,
            open_circuit_breaker=True,
            throttle_scope=ThrottleScope.IP,
            confidence=assessment.confidence,
            affected_credentials=assessment.affected_credentials,
            reason="IP-level throttle detected via correlation",
        )
        # Auto-apply if managers provided
        if circuit_breaker is not None:
            await circuit_breaker.open_immediately(
                provider, reason=action.reason, duration=action.cooldown_seconds
            )
        # Apply per-credential cooldowns to all affected credentials
        # so they are properly tracked when the circuit recovers
        if cooldown_manager is not None and action.affected_credentials:
            for affected_cred in action.affected_credentials:
                await cooldown_manager.start_cooldown(affected_cred, action.cooldown_seconds)
        return action

    # Step 5: Single credential throttle
    lib_logger.debug(
        "Credential-level throttle for %s on provider '%s'. Cooldown: %ss.",
        mask_credential(credential), provider, cooldown,
    )
    action = ThrottleAction(
        action_type=ThrottleActionType.CREDENTIAL_COOLDOWN,
        cooldown_seconds=cooldown,
        open_circuit_breaker=False,
        throttle_scope=ThrottleScope.CREDENTIAL,
        confidence=assessment.confidence,
        reason="Credential-level rate limit",
    )
    # Auto-apply if managers provided
    if cooldown_manager is not None:
        await cooldown_manager.start_cooldown(credential, action.cooldown_seconds)
    return action


def should_rotate_on_error(classified_error: ClassifiedError) -> bool:
    """
    Determines if an error should trigger key rotation.

    Errors that SHOULD rotate (try another key):
    - rate_limit: Current key is throttled
    - quota_exceeded: Current key/account exhausted
    - forbidden: Current credential denied access
    - authentication: Current credential invalid
    - credential_reauth_needed: Credential needs interactive re-auth (queued)
    - server_error: Provider having issues (might work with different endpoint/key)
    - api_connection: Network issues (might be transient)
    - unknown: Safer to try another key

    Errors that should NOT rotate:
    - invalid_request: Client error in request payload (won't help to retry)
    - context_window_exceeded: Request too large (won't help to retry)
    - pre_request_callback_error: Internal proxy error
    - ip_rate_limit: IP-based throttle (rotation won't help, all keys share IP)

    Returns:
        True if should rotate to next key, False if should fail immediately
    """
    return classified_error.error_type not in _NON_ROTATABLE_ERRORS


def should_retry_same_key(classified_error: ClassifiedError) -> bool:
    """
    Determines if an error should retry with the same key (with backoff).

    Server errors, connection issues, and IP-based rate limits should retry
    the same key, as these are often transient or affect all credentials.

    Returns:
        True if should retry same key, False if should rotate immediately
    """
    return classified_error.error_type in _RETRYABLE_SAME_KEY_ERRORS
