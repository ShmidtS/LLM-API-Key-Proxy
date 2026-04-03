# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Compatibility shim for credential/auth provider lookup."""

from .providers import PROVIDER_AUTH_MAP

PROVIDER_MAP = PROVIDER_AUTH_MAP


def get_provider_auth_class(provider_name: str):
    """
    Returns the authentication class for a given provider.
    """
    provider_class = PROVIDER_MAP.get(provider_name.lower())
    if not provider_class:
        raise ValueError(f"Unknown provider: {provider_name}")
    return provider_class


def get_available_providers():
    """
    Returns a list of available provider names.
    """
    return list(PROVIDER_MAP.keys())
