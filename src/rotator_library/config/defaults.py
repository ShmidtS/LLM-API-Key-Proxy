# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
Centralized defaults for the rotator library.

This file contains all tunable default values for features like:
- Credential rotation and selection
- Fair Cycle Rotation
- Custom Caps
- Cooldown and backoff timing

Providers can override these by setting class attributes.
Environment variables can override at runtime.

See DOCUMENTATION.md for detailed descriptions of each setting.
"""

import os
from typing import Dict, Optional

import logging

_TRUTHY_VALUES = frozenset({"true", "1", "yes", "on"})

TRACE = 5
logging.addLevelName(TRACE, "TRACE")


def env_bool(key: str, default: bool = False) -> bool:
    """Get a boolean from an environment variable."""
    return os.getenv(key, str(default).lower()).lower() in _TRUTHY_VALUES


def env_int(key: str, default: int) -> int:
    """Get an integer from an environment variable, falling back to default."""
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def env_float(key: str, default: float) -> float:
    """Get a float from an environment variable, falling back to default."""
    try:
        return float(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default

# =============================================================================
# ROTATION & SELECTION DEFAULTS
# =============================================================================

# Default credential rotation mode
# Options: "balanced" (distribute load) or "sequential" (use until exhausted)
# Override per-provider: ROTATION_MODE_{PROVIDER}=balanced/sequential
DEFAULT_ROTATION_MODE: str = "balanced"

# Weight tolerance for weighted random credential selection
# 0.0 = deterministic (always pick least-used)
# 2.0-4.0 = balanced randomness (recommended)
# 5.0+ = high randomness
DEFAULT_ROTATION_TOLERANCE: float = 3.0

# Maximum retries per credential before rotating
DEFAULT_MAX_RETRIES: int = 2

# Maximum total API call attempts across ALL credentials before failing
# Caps N_credentials * max_retries to prevent excessive API calls (e.g. 50 creds * 2 = 100 calls)
# Override via environment variable: MAX_TOTAL_ATTEMPTS=<count>
MAX_TOTAL_ATTEMPTS: int = env_int("MAX_TOTAL_ATTEMPTS", 10)

# Global request timeout in seconds
# This controls how long a request can wait for an available credential.
# If all credentials are on cooldown and the soonest one won't be available
# within this timeout, the request fails fast with a clear message.
# Override via environment variable: GLOBAL_TIMEOUT=<seconds>
DEFAULT_GLOBAL_TIMEOUT: int = env_int("GLOBAL_TIMEOUT", 120)

# Global timeout for media/image generation endpoints.
# Media endpoints need more budget: provider polling, longer cooldowns from 429s.
# Override via environment variable: MEDIA_GLOBAL_TIMEOUT=<seconds>
MEDIA_GLOBAL_TIMEOUT: int = env_int("MEDIA_GLOBAL_TIMEOUT", 300)

# =============================================================================
# TIER & PRIORITY DEFAULTS
# =============================================================================

# Default priority for tiers not in tier_priorities mapping (lower = higher priority)
DEFAULT_TIER_PRIORITY: int = 10

# Fallback concurrency multiplier for sequential mode
# Used when priority not in default_priority_multipliers
DEFAULT_SEQUENTIAL_FALLBACK_MULTIPLIER: int = 1

# =============================================================================
# FAIR CYCLE ROTATION DEFAULTS
# =============================================================================
# Fair cycle ensures each credential exhausts at least once before reuse.

# Enable fair cycle rotation
# None = derive from rotation mode (enabled for sequential only)
# Override: FAIR_CYCLE_{PROVIDER}=true/false
DEFAULT_FAIR_CYCLE_ENABLED: Optional[bool] = None

# Tracking mode for fair cycle
# "model_group" = track per quota group (or per model if ungrouped)
# "credential" = track per credential globally (ignores model)
# Override: FAIR_CYCLE_TRACKING_MODE_{PROVIDER}=model_group/credential
DEFAULT_FAIR_CYCLE_TRACKING_MODE: str = "model_group"

# Cross-tier tracking
# False = each priority tier cycles independently
# True = ALL credentials must exhaust regardless of tier
# Override: FAIR_CYCLE_CROSS_TIER_{PROVIDER}=true/false
DEFAULT_FAIR_CYCLE_CROSS_TIER: bool = False

# Cycle duration in seconds (how long before cycle auto-resets)
# Override: FAIR_CYCLE_DURATION_{PROVIDER}=<seconds>
DEFAULT_FAIR_CYCLE_DURATION: int = 604800  # 7 days

# Exhaustion cooldown threshold in seconds
# Cooldowns longer than this mark credential as "exhausted" for fair cycle
# Override: EXHAUSTION_COOLDOWN_THRESHOLD_{PROVIDER}=<seconds>
# Global fallback: EXHAUSTION_COOLDOWN_THRESHOLD=<seconds>
DEFAULT_EXHAUSTION_COOLDOWN_THRESHOLD: int = 300  # 5 minutes

# =============================================================================
# CUSTOM CAPS DEFAULTS
# =============================================================================
# Custom caps allow setting usage limits more restrictive than actual API limits.

# Default cooldown mode when custom cap is hit
# Options: "quota_reset" | "offset" | "fixed"
DEFAULT_CUSTOM_CAP_COOLDOWN_MODE: str = "quota_reset"

# Default cooldown value in seconds (for offset/fixed modes)
DEFAULT_CUSTOM_CAP_COOLDOWN_VALUE: int = 0

# =============================================================================
# COOLDOWN & BACKOFF DEFAULTS
# =============================================================================
# These control how long credentials are paused after errors.

# Escalating backoff tiers for consecutive failures (seconds)
# Key = failure count, Value = cooldown duration
COOLDOWN_BACKOFF_TIERS: Dict[int, int] = {
    1: 5,  # 1st failure: 5 seconds
    2: 10,  # 2nd failure: 30 seconds
    3: 20,  # 3rd failure: 1 minute
    4: 30,  # 4th failure: 2 minutes
}

# Maximum backoff for 5+ consecutive failures (seconds)
COOLDOWN_BACKOFF_MAX: int = 120  # 5 minutes

# Authentication error lockout duration (seconds)
# Applied when 401/403 received - credential assumed revoked
COOLDOWN_AUTH_ERROR: int = 120  # 5 minutes

# Transient/provider-level error cooldown (seconds)
# Applied for errors that don't count against credential health
COOLDOWN_TRANSIENT_ERROR: int = 30

# Default rate limit cooldown when retry_after not provided (seconds)
COOLDOWN_RATE_LIMIT_DEFAULT: int = 10

# =============================================================================
# HTTP TIMEOUT DEFAULTS
# =============================================================================
# HTTP client timeout configuration with environment variable overrides.

# Connection timeout in seconds
# Reduced from 30s to 10s for faster fail on unreachable hosts
# Override: HTTP_CONNECT_TIMEOUT=<seconds>
HTTP_CONNECT_TIMEOUT: int = env_int("HTTP_CONNECT_TIMEOUT", 10)

# Read timeout in seconds (time to receive response)
# Override: HTTP_READ_TIMEOUT=<seconds>
HTTP_READ_TIMEOUT: int = env_int("HTTP_READ_TIMEOUT", 120)

# Write timeout in seconds (time to send request body for streaming)
# Override: HTTP_WRITE_TIMEOUT=<seconds>
HTTP_WRITE_TIMEOUT: int = env_int("HTTP_WRITE_TIMEOUT", 300)

# =============================================================================
# OAUTH QUEUE TIMEOUT DEFAULTS
# =============================================================================
# OAuth queue and authentication timeouts with environment variable overrides.

# OAuth queue wait timeout in seconds
# Override: OAUTH_QUEUE_TIMEOUT=<seconds>
OAUTH_QUEUE_TIMEOUT: int = env_int("OAUTH_QUEUE_TIMEOUT", 120)

# OAuth authentication timeout in seconds
# Override: OAUTH_AUTH_TIMEOUT=<seconds>
OAUTH_AUTH_TIMEOUT: int = env_int("OAUTH_AUTH_TIMEOUT", 90)

# =============================================================================
# RETRY CONFIGURATION DEFAULTS
# =============================================================================
# Retry timing configuration with jitter for thundering herd prevention.

# Base delay for exponential backoff in seconds
# Override: RETRY_BASE_DELAY=<seconds>
RETRY_BASE_DELAY: float = env_float("RETRY_BASE_DELAY", 1.0)

# Maximum delay cap for exponential backoff in seconds
# Override: RETRY_MAX_DELAY=<seconds>
RETRY_MAX_DELAY: float = env_float("RETRY_MAX_DELAY", 60.0)

# Jitter factor for retry delay randomization (0.0-1.0)
# Helps prevent thundering herd on coordinated retries
# Override: RETRY_JITTER_FACTOR=<factor>
RETRY_JITTER_FACTOR: float = env_float("RETRY_JITTER_FACTOR", 0.1)

# =============================================================================
# DNS CONFIGURATION DEFAULTS
# =============================================================================
# DNS caching and query timeout settings.

# DNS cache TTL in seconds
# Override: DNS_CACHE_TTL=<seconds>
DNS_CACHE_TTL: int = env_int("DNS_CACHE_TTL", 300)

# DNS query timeout in seconds
# Override: DNS_QUERY_TIMEOUT=<seconds>
DNS_QUERY_TIMEOUT: int = env_int("DNS_QUERY_TIMEOUT", 10)

# DNS over HTTPS (DoH) query timeout in seconds
# Override: HTTP_DOH_TIMEOUT=<seconds>
HTTP_DOH_TIMEOUT: int = env_int("HTTP_DOH_TIMEOUT", 5)

# =============================================================================
# HTTP COMPRESSION DEFAULTS
# =============================================================================
# Gzip compression for outgoing requests to bypass WAF payload size limits.

# Minimum request body size in bytes to trigger gzip compression
# Override: HTTP_COMPRESS_MIN_SIZE=<bytes>
HTTP_COMPRESS_MIN_SIZE: int = env_int("HTTP_COMPRESS_MIN_SIZE", 10240)

# Enable gzip compression for outgoing requests
# Override: HTTP_COMPRESS_REQUESTS=true/false
HTTP_COMPRESS_REQUESTS: bool = env_bool("HTTP_COMPRESS_REQUESTS", True)

# Shared gzip request compression worker count
# Override: HTTP_GZIP_MAX_WORKERS=<count>
HTTP_GZIP_MAX_WORKERS: int = env_int("HTTP_GZIP_MAX_WORKERS", 4)

# Minimum compression efficiency for compressed request payloads
# Override: HTTP_COMPRESSION_THRESHOLD=<ratio>
HTTP_COMPRESSION_THRESHOLD: float = env_float("HTTP_COMPRESSION_THRESHOLD", 0.9)

# =============================================================================
# HTTP CLIENT POOL DEFAULTS
# =============================================================================

HTTP_MAX_KEEPALIVE_WINDOWS: int = 32
HTTP_MAX_KEEPALIVE_POSIX: int = 128
HTTP_MAX_CONNECTIONS_WINDOWS: int = 128
HTTP_MAX_CONNECTIONS_POSIX: int = 384
HTTP_KEEPALIVE_EXPIRY: float = env_float("HTTP_KEEPALIVE_EXPIRY", 120.0)
HTTP_WARMUP_CONNECTIONS: int = env_int("HTTP_WARMUP_CONNECTIONS", 5)
HTTP_STREAMING_MAX_CONNECTIONS_WINDOWS: int = 64
HTTP_STREAMING_MAX_CONNECTIONS_POSIX: int = 192
HTTP_STREAMING_MAX_KEEPALIVE_WINDOWS: int = 32
HTTP_STREAMING_MAX_KEEPALIVE_POSIX: int = 64
HTTP_STREAMING_KEEPALIVE_EXPIRY: float = env_float("HTTP_STREAMING_KEEPALIVE_EXPIRY", 180.0)
HTTP_SSL_VERIFY_DEFAULT: bool = True
HTTP_WARMUP_HOST_LIMIT: int = env_int("HTTP_WARMUP_HOST_LIMIT", 5)

# Maximum delay cap for same-key retries in seconds
# Override: RETRY_SAME_KEY_MAX_WAIT=<seconds>
RETRY_SAME_KEY_MAX_WAIT: float = env_float("RETRY_SAME_KEY_MAX_WAIT", 30.0)

# =============================================================================
# OAUTH USER FLOW TIMEOUT DEFAULTS
# =============================================================================

# OAuth user flow timeout in seconds (time for user to complete OAuth)
# Override: OAUTH_USER_TIMEOUT=<seconds>
OAUTH_USER_TIMEOUT: int = env_int("OAUTH_USER_TIMEOUT", 300)

# OAuth callback wait timeout in seconds
# Override: OAUTH_CALLBACK_TIMEOUT=<seconds>
OAUTH_CALLBACK_TIMEOUT: int = env_int("OAUTH_CALLBACK_TIMEOUT", 310)

# =============================================================================
# UI DELAY DEFAULTS
# =============================================================================

# UI refresh delay in seconds (for TUI consistency)
# Override: UI_REFRESH_DELAY=<seconds>
UI_REFRESH_DELAY: float = env_float("UI_REFRESH_DELAY", 0.3)

# UI brief pause delay in seconds
# Override: UI_BRIEF_PAUSE=<seconds>
UI_BRIEF_PAUSE: float = env_float("UI_BRIEF_PAUSE", 0.1)

# =============================================================================
# RETRY DELAY DEFAULTS
# =============================================================================

# Brief retry delay in seconds (for quick retries)
# Override: RETRY_BRIEF_DELAY=<seconds>
RETRY_BRIEF_DELAY: float = env_float("RETRY_BRIEF_DELAY", 1.0)

# OAuth token refresh delay in seconds
# Override: OAUTH_REFRESH_DELAY=<seconds>
OAUTH_REFRESH_DELAY: float = env_float("OAUTH_REFRESH_DELAY", 2.0)

# =============================================================================
# CIRCUIT BREAKER DEFAULTS
# =============================================================================
# Circuit breaker prevents cascade exhaustion during IP-level throttling.

# Number of consecutive failures before opening circuit
# Increased from 3 to 5 for transient rate limit tolerance
# Override: CIRCUIT_BREAKER_FAILURE_THRESHOLD=<count>
CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5

# Seconds to wait before attempting recovery
# Override: CIRCUIT_BREAKER_RECOVERY_TIMEOUT=<seconds>
CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = 60

# Max test requests in half-open state
# Override: CIRCUIT_BREAKER_HALF_OPEN_REQUESTS=<count>
CIRCUIT_BREAKER_HALF_OPEN_REQUESTS: int = 1

# Disable circuit breaker entirely (for debugging)
# Override: CIRCUIT_BREAKER_DISABLED=true
CIRCUIT_BREAKER_DISABLED: bool = False

# Provider-specific circuit breaker overrides
# These providers route to multiple backends and need different settings
# Keys: failure_threshold, recovery_timeout, half_open_requests
CIRCUIT_BREAKER_PROVIDER_OVERRIDES: Dict[str, Dict[str, int]] = {
    "kilocode": {
        "failure_threshold": 5,  # More tolerant (routes to multiple backends)
        "recovery_timeout": 30,  # Faster recovery
        "half_open_requests": 3,  # More test requests
    },
    "openrouter": {
        "failure_threshold": 5,
        "recovery_timeout": 30,
        "half_open_requests": 3,
    },
    "nvidia": {
        "failure_threshold": 10,  # Higher tolerance for NIM endpoints
        "recovery_timeout": 30,  # Faster recovery
        "half_open_requests": 5,  # Increased for concurrent recovery
    },
}

# =============================================================================
# IP THROTTLE DETECTOR DEFAULTS
# =============================================================================
# Detects IP-level throttling via correlation of 429 errors across credentials.

# Time window in seconds to correlate 429 errors
# Override: IP_THROTTLE_WINDOW_SECONDS=<seconds>
IP_THROTTLE_WINDOW_SECONDS: int = 10

# Minimum credentials with 429 to detect IP throttle
# Override: IP_THROTTLE_MIN_CREDENTIALS=<count>
IP_THROTTLE_MIN_CREDENTIALS: int = 2

# Default cooldown for IP-level throttling
# Override: IP_THROTTLE_COOLDOWN=<seconds>
IP_THROTTLE_COOLDOWN: int = 30

# Disable IP throttle detection
# Override: IP_THROTTLE_DETECTION_DISABLED=true
IP_THROTTLE_DETECTION_DISABLED: bool = False

# =============================================================================
# PROVIDER-SPECIFIC BACKOFF DEFAULTS
# =============================================================================
# Tunable retry backoff settings per provider.

# Kilocode provider backoff settings
# Override via environment: KILOCODE_BACKOFF_BASE, KILOCODE_MAX_BACKOFF
KILOCODE_BACKOFF_BASE: float = 1.0  # Base multiplier for server errors
KILOCODE_MAX_BACKOFF: float = 30.0  # Maximum backoff in seconds

# =============================================================================
# ADAPTIVE RATE LIMITER DEFAULTS
# =============================================================================
# Proactive per-provider request pacing with AIMD rate adjustment.
# Disabled by default. Enable via ADAPTIVE_RATE_LIMIT_ENABLED=true.

# Enable adaptive rate limiter
# Override: ADAPTIVE_RATE_LIMIT_ENABLED=true
ADAPTIVE_RATE_LIMIT_ENABLED: bool = env_bool("ADAPTIVE_RATE_LIMIT_ENABLED", False)

# Initial requests per second per provider
ADAPTIVE_RATE_LIMIT_INITIAL_RPS: float = 10.0

# Minimum rps floor (never decrease below this)
ADAPTIVE_RATE_LIMIT_MIN_RPS: float = 1.0

# Maximum rps ceiling
ADAPTIVE_RATE_LIMIT_MAX_RPS: float = 100.0

# Multiplicative decrease factor on 429 (0.5 = halve rate)
ADAPTIVE_RATE_LIMIT_DECREASE_FACTOR: float = 0.5

# Additive increase per interval (rps)
ADAPTIVE_RATE_LIMIT_INCREASE_RPS: float = 1.0

# Seconds between rate increases
ADAPTIVE_RATE_LIMIT_INCREASE_INTERVAL: float = 30.0

# =============================================================================
# GARBAGE RESPONSE DETECTION DEFAULTS
# =============================================================================
# Detects garbage/hallucinated model responses and retries with another credential.

# Enable garbage response quality validation
# Override: GARBAGE_DETECTION_ENABLED=true/false
GARBAGE_DETECTION_ENABLED: bool = env_bool("GARBAGE_DETECTION_ENABLED", True)

# Word repetition threshold (unique/total ratio below this triggers garbage detection)
# Lower = more tolerant (0.1 = almost all repeated), higher = stricter (0.4 = moderate variety required)
# Override: GARBAGE_REPETITION_THRESHOLD=<float>
GARBAGE_REPETITION_THRESHOLD: float = env_float("GARBAGE_REPETITION_THRESHOLD", 0.25)

# =============================================================================
# OAUTH & HOST DEFAULTS
# =============================================================================

# OAuth token refresh retry count
# Override: OAUTH_REFRESH_MAX_RETRIES=<count>
OAUTH_REFRESH_MAX_RETRIES: int = env_int("OAUTH_REFRESH_MAX_RETRIES", 3)

# Default proxy host
# Override: PROXY_DEFAULT_HOST=<host>
PROXY_DEFAULT_HOST: str = os.getenv("PROXY_DEFAULT_HOST", "127.0.0.1")

# Default proxy port
# Override: PROXY_DEFAULT_PORT=<port>
PROXY_DEFAULT_PORT: int = env_int("PROXY_DEFAULT_PORT", 8000)

# =============================================================================
# COOLDOWN DISABLE FLAGS (from theblazehen fork)
# =============================================================================
# Allows disabling cooldowns per-provider for debugging/emergency purposes.


