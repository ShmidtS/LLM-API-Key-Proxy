# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/providers/gemini_auth_base.py

import logging

from ..utils.ttl_dict import TTLDict
from .google_oauth_base import GoogleOAuthBase
from .utilities.gemini_shared_utils import CODE_ASSIST_ENDPOINT
from .utilities.google_project_discovery import GoogleProjectDiscoveryMixin

lib_logger = logging.getLogger("rotator_library")

# Headers for Gemini CLI auth/discovery calls (loadCodeAssist, onboardUser, etc.)
#
# For OAuth/Code Assist path, native gemini-cli only sends:
# - Content-Type: application/json
# - Authorization: Bearer <token>
# - User-Agent: GeminiCLI/${version} (${platform}; ${arch})
#
# Headers NOT sent by native CLI (confirmed via explore agent analysis of server.ts):
# - X-Goog-Api-Client: Not used in Code Assist path
# - Client-Metadata: Sent in REQUEST BODY for these endpoints, not as HTTP header
#
# Note: The commented headers below previously worked well for SDK fingerprinting.
# Uncomment if you want to try SDK mimicry for potential rate limit benefits.
#
# Source: gemini-cli/packages/core/src/code_assist/server.ts:284-290
GEMINI_CLI_AUTH_HEADERS = {
    "User-Agent": "GeminiCLI/0.26.0 (win32; x64)",
    # -------------------------------------------------------------------------
    # COMMENTED OUT - Not sent by native gemini-cli for OAuth/Code Assist path
    # -------------------------------------------------------------------------
    # "X-Goog-Api-Client": "gl-node/22.17.0 gdcl/1.30.0",  # SDK mimicry - not used by native CLI
    # "Client-Metadata": (                                  # Sent in body, not as header
    #     "ideType=IDE_UNSPECIFIED,"
    #     "pluginType=GEMINI,"
    #     "ideVersion=0.26.0,"
    #     "platform=WINDOWS_AMD64,"
    #     "updateChannel=stable"
    # ),
}


class GeminiAuthBase(GoogleProjectDiscoveryMixin, GoogleOAuthBase):
    """
    Gemini CLI OAuth2 authentication implementation.

    Inherits all OAuth functionality from GoogleOAuthBase with Gemini-specific configuration.

    Also provides project/tier discovery functionality that runs during authentication,
    ensuring credentials have their tier and project_id cached before any API requests.
    """

    CLIENT_ID = (
        "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com"
    )
    CLIENT_SECRET = "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"
    OAUTH_SCOPES = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
    ]
    ENV_PREFIX = "GEMINI_CLI"
    CALLBACK_PORT = 8085
    CALLBACK_PATH = "/oauth2callback"

    _provider_display_name = "Gemini"
    _auth_headers = GEMINI_CLI_AUTH_HEADERS
    _project_id_env_var = "GEMINI_CLI_PROJECT_ID"
    _project_id_extra_env_vars: list = []
    _load_code_assist_endpoint_order = [CODE_ASSIST_ENDPOINT]
    _onboard_user_endpoint_order = [CODE_ASSIST_ENDPOINT]
    _onboard_poll_attempts = 150
    _load_code_assist_timeout = 20
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
        """Return the file prefix for Gemini CLI credentials."""
        return "gemini_cli"
