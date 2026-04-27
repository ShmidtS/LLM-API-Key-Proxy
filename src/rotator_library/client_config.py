# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
Provider configuration builder extracted from RotatingClient.__init__.

Builds rotation modes, priority multipliers, fair cycle config,
exhaustion cooldown thresholds, and custom caps from provider
class defaults + environment variable overrides.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from .config import (
    DEFAULT_EXHAUSTION_COOLDOWN_THRESHOLD,
    DEFAULT_FAIR_CYCLE_DURATION,
    DEFAULT_SEQUENTIAL_FALLBACK_MULTIPLIER,
)
import logging

lib_logger = logging.getLogger("rotator_library")


def _get_env_cache() -> dict:
    """Access the provider env cache from env_cache module."""
    from .env_cache import get_provider_env_cache
    return get_provider_env_cache()


def build_provider_rotation_modes(
    all_credentials: Dict[str, List[str]],
    provider_plugins: Dict[str, Any],
) -> Dict[str, str]:
    """Build per-provider rotation mode map from class defaults + env overrides."""
    env_cache = _get_env_cache()
    modes: Dict[str, str] = {}
    for provider in all_credentials:
        provider_class = provider_plugins.get(provider)
        if provider_class and hasattr(provider_class, "get_rotation_mode"):
            mode = provider_class.get_rotation_mode(provider)
        else:
            env_key = f"ROTATION_MODE_{provider.upper()}"
            mode = env_cache.get(env_key, "balanced")
        modes[provider] = mode
        if mode != "balanced":
            lib_logger.info("Provider '%s' using rotation mode: %s", provider, mode)
    return modes


def build_priority_multipliers(
    all_credentials: Dict[str, List[str]],
    provider_plugins: Dict[str, Any],
) -> Tuple[Dict[str, Dict[int, int]], Dict[str, Dict[str, Dict[int, int]]], Dict[str, int]]:
    """Build priority multipliers, mode-specific multipliers, and sequential fallback maps."""
    env_cache = _get_env_cache()
    priority_multipliers: Dict[str, Dict[int, int]] = {}
    priority_multipliers_by_mode: Dict[str, Dict[str, Dict[int, int]]] = {}
    sequential_fallback_multipliers: Dict[str, int] = {}

    for provider in all_credentials:
        provider_class = provider_plugins.get(provider)

        if provider_class:
            if hasattr(provider_class, "default_priority_multipliers"):
                default_multipliers = provider_class.default_priority_multipliers
                if default_multipliers:
                    priority_multipliers[provider] = dict(default_multipliers)

            if hasattr(provider_class, "default_sequential_fallback_multiplier"):
                fallback = provider_class.default_sequential_fallback_multiplier
                if fallback != DEFAULT_SEQUENTIAL_FALLBACK_MULTIPLIER:
                    sequential_fallback_multipliers[provider] = fallback

        for key, value in env_cache.items():
            prefix = f"CONCURRENCY_MULTIPLIER_{provider.upper()}_PRIORITY_"
            if key.startswith(prefix):
                remainder = key[len(prefix):]
                try:
                    multiplier = int(value)
                    if multiplier < 1:
                        lib_logger.warning(f"Invalid {key}: {value}. Must be >= 1.")
                        continue

                    if "_" in remainder:
                        parts = remainder.rsplit("_", 1)
                        priority = int(parts[0])
                        mode = parts[1].lower()
                        if mode in ("sequential", "balanced"):
                            priority_multipliers_by_mode.setdefault(provider, {})
                            priority_multipliers_by_mode[provider].setdefault(mode, {})
                            priority_multipliers_by_mode[provider][mode][priority] = multiplier
                            lib_logger.info(
                                f"Provider '{provider}' priority {priority} ({mode} mode) multiplier: {multiplier}x"
                            )
                        else:
                            lib_logger.warning(f"Unknown mode in {key}: {mode}")
                    else:
                        priority = int(remainder)
                        priority_multipliers.setdefault(provider, {})
                        priority_multipliers[provider][priority] = multiplier
                        lib_logger.info(
                            f"Provider '{provider}' priority {priority} multiplier: {multiplier}x"
                        )
                except ValueError:
                    lib_logger.warning(
                        f"Invalid {key}: {value}. Could not parse priority or multiplier."
                    )

    for provider, multipliers in priority_multipliers.items():
        if multipliers:
            lib_logger.info(f"Provider '{provider}' priority multipliers: {multipliers}")
    for provider, fallback in sequential_fallback_multipliers.items():
        lib_logger.info(f"Provider '{provider}' sequential fallback multiplier: {fallback}x")

    return priority_multipliers, priority_multipliers_by_mode, sequential_fallback_multipliers


def build_fair_cycle_config(
    all_credentials: Dict[str, List[str]],
    provider_plugins: Dict[str, Any],
    provider_rotation_modes: Dict[str, str],
) -> Tuple[Dict[str, bool], Dict[str, str], Dict[str, bool], Dict[str, int]]:
    """Build fair cycle configuration per provider."""
    env_cache = _get_env_cache()
    enabled: Dict[str, bool] = {}
    tracking_mode: Dict[str, str] = {}
    cross_tier: Dict[str, bool] = {}
    duration: Dict[str, int] = {}

    for provider in all_credentials:
        provider_class = provider_plugins.get(provider)

        env_key = f"FAIR_CYCLE_{provider.upper()}"
        env_val = env_cache.get(env_key)
        if env_val is not None:
            enabled[provider] = env_val.lower() in ("true", "1", "yes")
        elif provider_class and hasattr(provider_class, "default_fair_cycle_enabled"):
            default_val = provider_class.default_fair_cycle_enabled
            if default_val is not None:
                enabled[provider] = default_val

        env_key = f"FAIR_CYCLE_TRACKING_MODE_{provider.upper()}"
        env_val = env_cache.get(env_key)
        if env_val is not None and env_val.lower() in ("model_group", "credential"):
            tracking_mode[provider] = env_val.lower()
        elif provider_class and hasattr(provider_class, "default_fair_cycle_tracking_mode"):
            tracking_mode[provider] = provider_class.default_fair_cycle_tracking_mode

        env_key = f"FAIR_CYCLE_CROSS_TIER_{provider.upper()}"
        env_val = env_cache.get(env_key)
        if env_val is not None:
            cross_tier[provider] = env_val.lower() in ("true", "1", "yes")
        elif provider_class and hasattr(provider_class, "default_fair_cycle_cross_tier"):
            if provider_class.default_fair_cycle_cross_tier:
                cross_tier[provider] = True

        env_key = f"FAIR_CYCLE_DURATION_{provider.upper()}"
        env_val = env_cache.get(env_key)
        if env_val is not None:
            try:
                duration[provider] = int(env_val)
            except ValueError:
                lib_logger.warning(f"Invalid {env_key}: {env_val}. Must be integer.")
        elif provider_class and hasattr(provider_class, "default_fair_cycle_duration"):
            dur = provider_class.default_fair_cycle_duration
            if dur != DEFAULT_FAIR_CYCLE_DURATION:
                duration[provider] = dur

    for provider, is_enabled in enabled.items():
        if not is_enabled:
            lib_logger.info(f"Provider '{provider}' fair cycle: disabled")
    for provider, mode in tracking_mode.items():
        if mode != "model_group":
            lib_logger.info(f"Provider '{provider}' fair cycle tracking mode: {mode}")
    for provider, ct in cross_tier.items():
        if ct:
            lib_logger.info(f"Provider '{provider}' fair cycle cross-tier: enabled")

    return enabled, tracking_mode, cross_tier, duration


def build_exhaustion_cooldown_thresholds(
    all_credentials: Dict[str, List[str]],
    provider_plugins: Dict[str, Any],
) -> Dict[str, int]:
    """Build per-provider exhaustion cooldown thresholds."""
    env_cache = _get_env_cache()
    thresholds: Dict[str, int] = {}

    global_str = env_cache.get("EXHAUSTION_COOLDOWN_THRESHOLD")
    global_threshold = DEFAULT_EXHAUSTION_COOLDOWN_THRESHOLD
    if global_str:
        try:
            global_threshold = int(global_str)
        except ValueError:
            lib_logger.warning(
                f"Invalid EXHAUSTION_COOLDOWN_THRESHOLD: {global_str}. "
                f"Using default {DEFAULT_EXHAUSTION_COOLDOWN_THRESHOLD}."
            )

    for provider in all_credentials:
        provider_class = provider_plugins.get(provider)

        env_key = f"EXHAUSTION_COOLDOWN_THRESHOLD_{provider.upper()}"
        env_val = env_cache.get(env_key)
        if env_val is not None:
            try:
                thresholds[provider] = int(env_val)
            except ValueError:
                lib_logger.warning(f"Invalid {env_key}: {env_val}. Must be integer.")
        elif provider_class and hasattr(provider_class, "default_exhaustion_cooldown_threshold"):
            threshold = provider_class.default_exhaustion_cooldown_threshold
            if threshold != DEFAULT_EXHAUSTION_COOLDOWN_THRESHOLD:
                thresholds[provider] = threshold
        elif global_threshold != DEFAULT_EXHAUSTION_COOLDOWN_THRESHOLD:
            thresholds[provider] = global_threshold

    return thresholds


def parse_custom_cap_env_key(
    remainder: str,
) -> Tuple[Optional[Union[int, Tuple[int, ...], str]], Optional[str]]:
    """Parse tier and model/group from a custom cap env var remainder.

    Args:
        remainder: String after "CUSTOM_CAP_{PROVIDER}_T" prefix
                   e.g., "2_CLAUDE" or "2_3_CLAUDE" or "DEFAULT_CLAUDE"

    Returns:
        (tier_key, model_key) tuple, or (None, None) if parse fails
    """
    if not remainder:
        return None, None

    remaining_parts = remainder.split("_")
    if len(remaining_parts) < 2:
        return None, None

    tier_key: Union[int, Tuple[int, ...], str, None] = None
    model_key: Optional[str] = None
    tier_parts: List[int] = []

    for i, part in enumerate(remaining_parts):
        if part == "DEFAULT":
            tier_key = "default"
            model_key = "_".join(remaining_parts[i + 1:])
            break
        elif part.isdigit():
            tier_parts.append(int(part))
        else:
            if len(tier_parts) == 0:
                return None, None
            elif len(tier_parts) == 1:
                tier_key = tier_parts[0]
            else:
                tier_key = tuple(tier_parts)
            model_key = "_".join(remaining_parts[i:])
            break
    else:
        return None, None

    if model_key:
        model_key = model_key.lower().replace("_", "-")

    return tier_key, model_key


def build_custom_caps(
    all_credentials: Dict[str, List[str]],
    provider_plugins: Dict[str, Any],
) -> Dict[str, Dict[Union[int, Tuple[int, ...], str], Dict[str, Dict[str, Any]]]]:
    """Build per-provider custom caps configuration from class defaults + env overrides."""
    env_cache = _get_env_cache()
    custom_caps: Dict[
        str, Dict[Union[int, Tuple[int, ...], str], Dict[str, Dict[str, Any]]]
    ] = {}

    for provider in all_credentials:
        provider_class = provider_plugins.get(provider)
        provider_upper = provider.upper()

        if provider_class and hasattr(provider_class, "default_custom_caps"):
            default_caps = provider_class.default_custom_caps
            if default_caps:
                custom_caps[provider] = {}
                for tier_key, models_config in default_caps.items():
                    custom_caps[provider][tier_key] = dict(models_config)

        cap_prefix = f"CUSTOM_CAP_{provider_upper}_T"
        cooldown_prefix = f"CUSTOM_CAP_COOLDOWN_{provider_upper}_T"

        for env_key, env_value in env_cache.items():
            if env_key.startswith(cap_prefix) and not env_key.startswith(cooldown_prefix):
                remainder = env_key[len(cap_prefix):]
                tier_key, model_key = parse_custom_cap_env_key(remainder)
                if tier_key is None:
                    continue

                custom_caps.setdefault(provider, {})
                custom_caps[provider].setdefault(tier_key, {})
                custom_caps[provider][tier_key].setdefault(model_key, {})
                custom_caps[provider][tier_key][model_key]["max_requests"] = env_value

            elif env_key.startswith(cooldown_prefix):
                remainder = env_key[len(cooldown_prefix):]
                tier_key, model_key = parse_custom_cap_env_key(remainder)
                if tier_key is None:
                    continue

                if ":" in env_value:
                    mode, value_str = env_value.split(":", 1)
                    try:
                        value = int(value_str)
                    except ValueError:
                        lib_logger.warning(f"Invalid cooldown value in {env_key}: {env_value}")
                        continue
                else:
                    mode = env_value
                    value = 0

                custom_caps.setdefault(provider, {})
                custom_caps[provider].setdefault(tier_key, {})
                custom_caps[provider][tier_key].setdefault(model_key, {})
                custom_caps[provider][tier_key][model_key]["cooldown_mode"] = mode
                custom_caps[provider][tier_key][model_key]["cooldown_value"] = value

    for provider, tier_configs in custom_caps.items():
        for tier_key, models_config in tier_configs.items():
            for model_key, config in models_config.items():
                max_req = config.get("max_requests", "default")
                cooldown = config.get("cooldown_mode", "quota_reset")
                lib_logger.info(
                    f"Custom cap: {provider}/T{tier_key}/{model_key} = {max_req}, cooldown={cooldown}"
                )

    return custom_caps


def build_all_provider_configs(
    all_credentials: Dict[str, List[str]],
    provider_plugins: Dict[str, Any],
) -> dict:
    """Build all provider-specific configuration maps at once.

    Returns dict with keys:
        provider_rotation_modes, priority_multipliers,
        priority_multipliers_by_mode, sequential_fallback_multipliers,
        fair_cycle_enabled, fair_cycle_tracking_mode,
        fair_cycle_cross_tier, fair_cycle_duration,
        exhaustion_cooldown_threshold, custom_caps
    """
    provider_rotation_modes = build_provider_rotation_modes(
        all_credentials, provider_plugins
    )
    priority_multipliers, priority_multipliers_by_mode, sequential_fallback_multipliers = (
        build_priority_multipliers(all_credentials, provider_plugins)
    )
    fair_cycle_enabled, fair_cycle_tracking_mode, fair_cycle_cross_tier, fair_cycle_duration = (
        build_fair_cycle_config(all_credentials, provider_plugins, provider_rotation_modes)
    )
    exhaustion_cooldown_threshold = build_exhaustion_cooldown_thresholds(
        all_credentials, provider_plugins
    )
    custom_caps = build_custom_caps(all_credentials, provider_plugins)

    return {
        "provider_rotation_modes": provider_rotation_modes,
        "priority_multipliers": priority_multipliers,
        "priority_multipliers_by_mode": priority_multipliers_by_mode,
        "sequential_fallback_multipliers": sequential_fallback_multipliers,
        "fair_cycle_enabled": fair_cycle_enabled,
        "fair_cycle_tracking_mode": fair_cycle_tracking_mode,
        "fair_cycle_cross_tier": fair_cycle_cross_tier,
        "fair_cycle_duration": fair_cycle_duration,
        "exhaustion_cooldown_threshold": exhaustion_cooldown_threshold,
        "custom_caps": custom_caps,
    }
