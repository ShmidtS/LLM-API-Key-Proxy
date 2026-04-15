# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# src/rotator_library/providers/antigravity_auth_base.py

import logging
from typing import Any, Dict, List

from ..utils.ttl_dict import TTLDict
from .google_oauth_base import GoogleOAuthBase
from .utilities.gemini_shared_utils import (
    ANTIGRAVITY_LOAD_ENDPOINT_ORDER,
    ANTIGRAVITY_ENDPOINT_FALLBACKS,
)
from .utilities.google_project_discovery import GoogleProjectDiscoveryMixin

lib_logger = logging.getLogger("rotator_library")

# Headers for Antigravity auth/discovery calls (loadCodeAssist, onboardUser)
# CRITICAL: User-Agent MUST be google-api-nodejs-client/* for standard-tier detection.
# Using antigravity/* UA causes server to return free-tier only (tested via matrix test).
# X-Goog-Api-Client value doesn't affect tier detection.
ANTIGRAVITY_AUTH_HEADERS = {
    "User-Agent": "google-api-nodejs-client/10.3.0",
    "X-Goog-Api-Client": "gl-node/22.18.0",
    "Client-Metadata": '{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}',
}


class AntigravityAuthBase(GoogleProjectDiscoveryMixin, GoogleOAuthBase):
    """
    Antigravity OAuth2 authentication implementation.

    Inherits all OAuth functionality from GoogleOAuthBase with Antigravity-specific configuration.
    Uses Antigravity's OAuth credentials and includes additional scopes for cclog and experimentsandconfigs.

    Also provides project/tier discovery functionality that runs during authentication,
    ensuring credentials have their tier and project_id cached before any API requests.
    """

    CLIENT_ID = (
        "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
    )
    CLIENT_SECRET = "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"
    OAUTH_SCOPES = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/cclog",  # Antigravity-specific
        "https://www.googleapis.com/auth/experimentsandconfigs",  # Antigravity-specific
    ]
    ENV_PREFIX = "ANTIGRAVITY"
    CALLBACK_PORT = 51121
    CALLBACK_PATH = "/oauthcallback"

    _provider_display_name = "Antigravity"
    _auth_headers = ANTIGRAVITY_AUTH_HEADERS
    _project_id_env_var = "ANTIGRAVITY_PROJECT_ID"
    _project_id_extra_env_vars = ["GOOGLE_CLOUD_PROJECT"]
    _load_code_assist_endpoint_order = ANTIGRAVITY_LOAD_ENDPOINT_ORDER
    _onboard_user_endpoint_order = ANTIGRAVITY_ENDPOINT_FALLBACKS
    _onboard_poll_attempts = 30
    _load_code_assist_timeout = 15
    _onboard_user_timeout = 30

    def __init__(self):
        super().__init__()
        # Project and tier caches - shared between auth base and provider
        self.project_id_cache: TTLDict = TTLDict(maxsize=500, default_ttl=86400.0)
        self.project_tier_cache: TTLDict = TTLDict(maxsize=500, default_ttl=86400.0)

    # =========================================================================
    # CREDENTIAL MANAGEMENT OVERRIDES
    # =========================================================================

    def _get_provider_file_prefix(self) -> str:
        """Return the file prefix for Antigravity credentials."""
        return "antigravity"

    # build_env_lines: inherited from GoogleOAuthBase (includes PROJECT_ID and TIER)
