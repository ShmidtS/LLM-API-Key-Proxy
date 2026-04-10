# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/providers/utilities/google_project_discovery.py

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ...http_client_pool import get_http_pool
from ...utils.json_utils import json_loads
from ..provider_interface import build_bearer_headers
from .gemini_shared_utils import CODE_ASSIST_ENDPOINT

lib_logger = logging.getLogger("rotator_library")


class GoogleProjectDiscoveryMixin:
    """
    Mixin providing Google Cloud project ID discovery and post-auth discovery.

    Used by GeminiAuthBase and AntigravityAuthBase to deduplicate the shared
    project discovery flow (loadCodeAssist -> onboardUser -> GCP Resource Manager).

    Host class must provide:
        - self.project_id_cache: TTLDict
        - self.project_tier_cache: TTLDict
        - self._credentials_cache: dict
        - self._parse_env_credential_path(path) -> Optional[str]
        - self._extract_project_id_from_response(data, key) -> Optional[str]
        - self._persist_project_metadata(credential_path, project_id, tier)
        - self._save_credentials(path, creds)
        - self.ENV_PREFIX: str

    Class attributes to configure per provider:
        - _provider_display_name: str  (e.g. "Gemini", "Antigravity")
        - _auth_headers: Dict[str, str]  (provider-specific auth headers)
        - _project_id_env_var: str  (primary env var, e.g. "GEMINI_CLI_PROJECT_ID")
        - _project_id_extra_env_vars: List[str]  (additional env vars to check)
        - _load_code_assist_endpoint_order: List[str]  (endpoint list for loadCodeAssist)
        - _onboard_user_endpoint_order: List[str]  (endpoint list for onboardUser)
        - _onboard_poll_attempts: int  (max polling attempts, e.g. 150 for Gemini, 30 for Antigravity)
        - _load_code_assist_timeout: int  (HTTP timeout for loadCodeAssist, seconds)
        - _onboard_user_timeout: int  (HTTP timeout for onboardUser, seconds)
    """

    _provider_display_name: str = "Google"
    _auth_headers: Dict[str, str] = {}
    _project_id_env_var: str = ""
    _project_id_extra_env_vars: List[str] = []
    _load_code_assist_endpoint_order: List[str] = [CODE_ASSIST_ENDPOINT]
    _onboard_user_endpoint_order: List[str] = [CODE_ASSIST_ENDPOINT]
    _onboard_poll_attempts: int = 150
    _load_code_assist_timeout: int = 20
    _onboard_user_timeout: int = 30

    async def _post_auth_discovery(
        self, credential_path: str, access_token: str
    ) -> None:
        """
        Discover and cache tier/project information immediately after OAuth authentication.

        This is called by GoogleOAuthBase._perform_interactive_oauth() after successful auth,
        ensuring tier and project_id are cached during the authentication flow rather than
        waiting for the first API request.

        Args:
            credential_path: Path to the credential file
            access_token: The newly obtained access token
        """
        lib_logger.debug(
            f"Starting post-auth discovery for {self._provider_display_name} credential: {Path(credential_path).name}"
        )

        if (
            credential_path in self.project_id_cache
            and credential_path in self.project_tier_cache
        ):
            lib_logger.debug(
                f"Tier and project already cached for {Path(credential_path).name}, skipping discovery"
            )
            return

        project_id = await self._discover_project_id(
            credential_path, access_token, litellm_params={}
        )

        tier = self.project_tier_cache.get(credential_path, "unknown")
        lib_logger.info(
            f"Post-auth discovery complete for {Path(credential_path).name}: "
            f"tier={tier}, project={project_id}"
        )

    async def _call_load_code_assist(
        self,
        client: httpx.AsyncClient,
        access_token: str,
        configured_project_id: Optional[str],
        headers: Dict[str, str],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Call loadCodeAssist with endpoint fallback chain.

        Returns:
            Tuple of (response_data, successful_endpoint) or (None, None) on failure
        """
        core_client_metadata = {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        }
        if configured_project_id:
            core_client_metadata["duetProject"] = configured_project_id

        load_request = {
            "cloudaicompanionProject": configured_project_id,
            "metadata": core_client_metadata,
        }

        last_error = None
        for endpoint in self._load_code_assist_endpoint_order:
            try:
                lib_logger.debug(f"Trying loadCodeAssist at {endpoint}")
                response = await client.post(
                    f"{endpoint}:loadCodeAssist",
                    headers=headers,
                    json=load_request,
                    timeout=self._load_code_assist_timeout,
                )
                if response.status_code == 200:
                    data = response.json()
                    lib_logger.debug(f"loadCodeAssist succeeded at {endpoint}")
                    return data, endpoint
                lib_logger.debug(
                    f"loadCodeAssist returned {response.status_code} at {endpoint}"
                )
                last_error = f"HTTP {response.status_code}"
            except Exception as e:
                lib_logger.debug(f"loadCodeAssist failed at {endpoint}: {e}")
                last_error = str(e)
                continue

        lib_logger.warning(
            f"All loadCodeAssist endpoints failed. Last error: {last_error}"
        )
        return None, None

    async def _call_onboard_user(
        self,
        client: httpx.AsyncClient,
        headers: Dict[str, str],
        onboard_request: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Call onboardUser with endpoint fallback chain.

        Returns:
            Response data dict or None on failure
        """
        last_error = None
        for endpoint in self._onboard_user_endpoint_order:
            try:
                lib_logger.debug(f"Trying onboardUser at {endpoint}")
                response = await client.post(
                    f"{endpoint}:onboardUser",
                    headers=headers,
                    json=onboard_request,
                    timeout=self._onboard_user_timeout,
                )
                if response.status_code == 200:
                    lib_logger.debug(f"onboardUser succeeded at {endpoint}")
                    return response.json()
                lib_logger.debug(
                    f"onboardUser returned {response.status_code} at {endpoint}"
                )
                last_error = f"HTTP {response.status_code}"
            except Exception as e:
                lib_logger.debug(f"onboardUser failed at {endpoint}: {e}")
                last_error = str(e)
                continue

        lib_logger.warning(
            f"All onboardUser endpoints failed. Last error: {last_error}"
        )
        return None

    async def _discover_project_id(
        self, credential_path: str, access_token: str, litellm_params: Dict[str, Any]
    ) -> str:
        """
        Discovers the Google Cloud Project ID, with caching and onboarding for new accounts.

        This follows the official Gemini CLI discovery flow:
        1. Check in-memory cache
        2. Check configured project_id override (litellm_params or env var)
        3. Check persisted project_id in credential file
        4. Call loadCodeAssist to check if user is already known (has currentTier)
           - If currentTier exists AND cloudaicompanionProject returned: use server's project
           - If currentTier exists but NO cloudaicompanionProject: use configured project_id (paid tier requires this)
           - If no currentTier: user needs onboarding
        5. Onboard user based on tier:
           - FREE tier: pass cloudaicompanionProject=None (server-managed)
           - PAID tier: pass cloudaicompanionProject=configured_project_id
        6. Fallback to GCP Resource Manager project listing
        """
        provider = self._provider_display_name
        lib_logger.debug(
            f"Starting project discovery for credential: {credential_path}"
        )

        if credential_path in self.project_id_cache:
            cached_project = self.project_id_cache[credential_path]
            lib_logger.debug(f"Using cached project ID: {cached_project}")
            return cached_project

        configured_project_id = litellm_params.get("project_id") or os.getenv(
            self._project_id_env_var
        )
        for extra_var in self._project_id_extra_env_vars:
            if configured_project_id:
                break
            configured_project_id = os.getenv(extra_var)
        if configured_project_id:
            lib_logger.debug(
                f"Found configured project_id override: {configured_project_id}"
            )

        credential_index = self._parse_env_credential_path(credential_path)
        if credential_index is None:
            metadata = self._credentials_cache.get(credential_path, {}).get("_proxy_metadata")
            if metadata is None:
                try:
                    with open(credential_path, "r") as f:
                        creds = json_loads(f.read())
                    metadata = creds.get("_proxy_metadata", {})
                except (FileNotFoundError, ValueError, KeyError) as e:
                    lib_logger.debug(f"Could not load persisted project ID from file: {e}")
                    metadata = {}
            persisted_project_id = metadata.get("project_id")
            persisted_tier = metadata.get("tier")

            if persisted_project_id:
                lib_logger.debug(
                    f"Loaded persisted project ID from credential file: {persisted_project_id}"
                )
                self.project_id_cache[credential_path] = persisted_project_id

                if persisted_tier:
                    self.project_tier_cache[credential_path] = persisted_tier
                    lib_logger.debug(f"Loaded persisted tier: {persisted_tier}")

                return persisted_project_id
        else:
            if credential_path in self._credentials_cache:
                creds = self._credentials_cache[credential_path]
                metadata = creds.get("_proxy_metadata", {})
                env_project_id = metadata.get("project_id")
                env_tier = metadata.get("tier")

                if env_project_id:
                    lib_logger.debug(
                        f"Loaded project ID from env credential metadata: {env_project_id}"
                    )
                    self.project_id_cache[credential_path] = env_project_id

                    if env_tier:
                        self.project_tier_cache[credential_path] = env_tier
                        lib_logger.debug(
                            f"Loaded tier from env credential metadata: {env_tier}"
                        )

                    return env_project_id

        lib_logger.debug(
            "No cached or configured project ID found, initiating discovery..."
        )
        headers = {
            **build_bearer_headers(access_token),
            **self._auth_headers,
        }

        discovered_tier = None

        pool = await get_http_pool()
        client = await pool.get_client_async()
        lib_logger.debug(
            "Attempting project discovery via Code Assist loadCodeAssist endpoint..."
        )
        try:
            core_client_metadata = {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
            }
            if configured_project_id:
                core_client_metadata["duetProject"] = configured_project_id

            data, successful_endpoint = await self._call_load_code_assist(
                client, access_token, configured_project_id, headers
            )

            if data is None:
                raise httpx.HTTPStatusError(
                    "All loadCodeAssist endpoints failed",
                    request=None,
                    response=None,
                )

            lib_logger.debug(
                f"loadCodeAssist succeeded at {successful_endpoint}, response keys: {list(data.keys())}"
            )

            allowed_tiers = data.get("allowedTiers", [])
            current_tier = data.get("currentTier")

            lib_logger.debug("=== Tier Information ===")
            lib_logger.debug(f"currentTier: {current_tier}")
            lib_logger.debug(f"allowedTiers count: {len(allowed_tiers)}")
            for i, tier in enumerate(allowed_tiers):
                tier_id = tier.get("id", "unknown")
                is_default = tier.get("isDefault", False)
                user_defined = tier.get("userDefinedCloudaicompanionProject", False)
                lib_logger.debug(
                    f"  Tier {i + 1}: id={tier_id}, isDefault={is_default}, userDefinedProject={user_defined}"
                )
            lib_logger.debug("========================")

            current_tier_id = None
            if current_tier:
                current_tier_id = current_tier.get("id")
                lib_logger.debug(f"User has currentTier: {current_tier_id}")

            if current_tier_id:
                server_project = self._extract_project_id_from_response(data)

                requires_user_project = any(
                    t.get("id") == current_tier_id
                    and t.get("userDefinedCloudaicompanionProject", False)
                    for t in allowed_tiers
                )
                is_free_tier = current_tier_id == "free-tier"

                if server_project:
                    project_id = server_project
                    lib_logger.debug(f"Server returned project: {project_id}")
                elif configured_project_id:
                    project_id = configured_project_id
                    lib_logger.debug(
                        f"No server project, using configured: {project_id}"
                    )
                elif is_free_tier:
                    lib_logger.debug(
                        "Free tier user with currentTier but no project - will try onboarding"
                    )
                    project_id = None
                elif requires_user_project:
                    raise ValueError(
                        f"Paid tier '{current_tier_id}' requires setting {self._project_id_env_var} environment variable. "
                        "See https://goo.gle/gemini-cli-auth-docs#workspace-gca"
                    )
                else:
                    lib_logger.warning(
                        f"Tier '{current_tier_id}' has no project and none configured - will try onboarding"
                    )
                    project_id = None

                if project_id:
                    self.project_tier_cache[credential_path] = current_tier_id
                    discovered_tier = current_tier_id

                    is_paid = current_tier_id and current_tier_id not in [
                        "free-tier",
                        "legacy-tier",
                        "unknown",
                    ]
                    if is_paid:
                        lib_logger.info(
                            f"Using {provider} paid tier '{current_tier_id}' with project: {project_id}"
                        )
                    else:
                        lib_logger.info(
                            f"Discovered {provider} project ID via loadCodeAssist: {project_id}"
                        )

                    self.project_id_cache[credential_path] = project_id

                    await self._persist_project_metadata(
                        credential_path, project_id, discovered_tier
                    )

                    return project_id

            lib_logger.info(
                f"No existing {provider} session found (no currentTier), attempting to onboard user..."
            )

            onboard_tier = None
            for tier in allowed_tiers:
                if tier.get("isDefault"):
                    onboard_tier = tier
                    break

            if not onboard_tier and allowed_tiers:
                for tier in allowed_tiers:
                    if tier.get("id") == "legacy-tier":
                        onboard_tier = tier
                        break
                if not onboard_tier:
                    onboard_tier = allowed_tiers[0]

            if not onboard_tier:
                raise ValueError("No onboarding tiers available from server")

            tier_id = onboard_tier.get("id", "free-tier")
            requires_user_project = onboard_tier.get(
                "userDefinedCloudaicompanionProject", False
            )

            lib_logger.debug(
                f"Onboarding with tier: {tier_id}, requiresUserProject: {requires_user_project}"
            )

            is_free_tier = tier_id == "free-tier"

            if is_free_tier:
                onboard_request = {
                    "tierId": tier_id,
                    "cloudaicompanionProject": None,
                    "metadata": core_client_metadata,
                }
                lib_logger.debug(
                    "Free tier onboarding: using server-managed project"
                )
            else:
                if not configured_project_id and requires_user_project:
                    raise ValueError(
                        f"Tier '{tier_id}' requires setting {self._project_id_env_var} environment variable. "
                        "See https://goo.gle/gemini-cli-auth-docs#workspace-gca"
                    )
                onboard_request = {
                    "tierId": tier_id,
                    "cloudaicompanionProject": configured_project_id,
                    "metadata": {
                        **core_client_metadata,
                        "duetProject": configured_project_id,
                    }
                    if configured_project_id
                    else core_client_metadata,
                }
                lib_logger.debug(
                    f"Paid tier onboarding: using project {configured_project_id}"
                )

            lib_logger.debug(
                "Initiating onboardUser request with endpoint fallback..."
            )
            lro_data = await self._call_onboard_user(
                client, headers, onboard_request
            )

            if lro_data is None:
                raise ValueError(
                    "All onboardUser endpoints failed. Cannot onboard user."
                )

            lib_logger.debug(
                f"Initial onboarding response: done={lro_data.get('done')}"
            )

            for i in range(self._onboard_poll_attempts):
                if lro_data.get("done"):
                    lib_logger.debug(f"Onboarding completed after {i * 2}s")
                    break
                await asyncio.sleep(2)
                poll_interval_log = 30 if self._onboard_poll_attempts > 50 else 20
                if (i + 1) % (poll_interval_log // 2) == 0:
                    lib_logger.info(
                        f"Still waiting for onboarding completion... ({(i + 1) * 2}s elapsed)"
                    )
                lib_logger.debug(
                    f"Polling onboarding status... (Attempt {i + 1}/{self._onboard_poll_attempts})"
                )
                lro_data = await self._call_onboard_user(
                    client, headers, onboard_request
                )
                if lro_data is None:
                    lib_logger.warning("onboardUser endpoint failed during polling")
                    break

            if not lro_data or not lro_data.get("done"):
                timeout_seconds = self._onboard_poll_attempts * 2
                lib_logger.error(f"Onboarding process timed out after {timeout_seconds} seconds")
                raise ValueError(
                    f"Onboarding process timed out after {timeout_seconds} seconds. Please try again or contact support."
                )

            lro_response_data = lro_data.get("response", {})
            project_id = self._extract_project_id_from_response(lro_response_data)

            if not project_id and configured_project_id:
                project_id = configured_project_id
                lib_logger.debug(
                    f"LRO didn't return project, using configured: {project_id}"
                )

            if not project_id:
                lib_logger.error(
                    "Onboarding completed but no project ID in response and none configured"
                )
                raise ValueError(
                    "Onboarding completed, but no project ID was returned. "
                    f"For paid tiers, set {self._project_id_env_var} environment variable."
                )

            lib_logger.debug(
                f"Successfully extracted project ID from onboarding response: {project_id}"
            )

            self.project_tier_cache[credential_path] = tier_id
            discovered_tier = tier_id
            lib_logger.debug(f"Cached tier information: {tier_id}")

            is_paid = tier_id and tier_id not in ["free-tier", "legacy-tier"]
            if is_paid:
                lib_logger.info(
                    f"Using {provider} paid tier '{tier_id}' with project: {project_id}"
                )
            else:
                lib_logger.info(
                    f"Successfully onboarded user and discovered project ID: {project_id}"
                )

            self.project_id_cache[credential_path] = project_id

            await self._persist_project_metadata(
                credential_path, project_id, discovered_tier
            )

            return project_id

        except httpx.HTTPStatusError as e:
            error_body = ""
            if e.response is not None:
                try:
                    error_body = e.response.text
                except Exception:
                    pass
                if e.response.status_code == 403:
                    lib_logger.error(
                        f"{provider} Code Assist API access denied (403). Response: {error_body}"
                    )
                    lib_logger.error(
                        "Possible causes: 1) cloudaicompanion.googleapis.com API not enabled, 2) Wrong project ID for paid tier, 3) Account lacks permissions"
                    )
                elif e.response.status_code == 404:
                    lib_logger.warning(
                        f"{provider} Code Assist endpoint not found (404). Falling back to project listing."
                    )
                elif e.response.status_code == 412:
                    lib_logger.error(
                        f"Precondition failed (412): {error_body}. This may mean the project ID is incompatible with the selected tier."
                    )
                else:
                    lib_logger.warning(
                        f"{provider} onboarding/discovery failed with status {e.response.status_code}: {error_body}. Falling back to project listing."
                    )
            else:
                lib_logger.warning(
                    f"{provider} onboarding/discovery failed with no response. Falling back to project listing."
                )
        except httpx.RequestError as e:
            lib_logger.warning(
                f"{provider} onboarding/discovery network error: {e}. Falling back to project listing."
            )

        lib_logger.debug(
            "Attempting to discover project via GCP Resource Manager API..."
        )
        try:
            pool = await get_http_pool()
            client = await pool.get_client_async()
            lib_logger.debug(
                "Querying Cloud Resource Manager for available projects..."
            )
            response = await client.get(
                "https://cloudresourcemanager.googleapis.com/v1/projects",
                headers=headers,
                timeout=20,
            )
            response.raise_for_status()
            projects = response.json().get("projects", [])
            lib_logger.debug(f"Found {len(projects)} total projects")
            active_projects = [
                p for p in projects if p.get("lifecycleState") == "ACTIVE"
            ]
            lib_logger.debug(f"Found {len(active_projects)} active projects")

            if not projects:
                lib_logger.error(
                    "No GCP projects found for this account. Please create a project in Google Cloud Console."
                )
            elif not active_projects:
                lib_logger.error(
                    "No active GCP projects found. Please activate a project in Google Cloud Console."
                )
            else:
                project_id = active_projects[0]["projectId"]
                lib_logger.info(
                    f"Discovered {provider} project ID from active projects list: {project_id}"
                )
                lib_logger.debug(
                    f"Selected first active project: {project_id} (out of {len(active_projects)} active projects)"
                )
                self.project_id_cache[credential_path] = project_id

                await self._persist_project_metadata(
                    credential_path, project_id, None
                )

                return project_id
        except httpx.HTTPStatusError as e:
            if e.response is not None and e.response.status_code == 403:
                lib_logger.error(
                    "Failed to list GCP projects due to a 403 Forbidden error. The Cloud Resource Manager API may not be enabled, or your account lacks the 'resourcemanager.projects.list' permission."
                )
            else:
                lib_logger.error(
                    f"Failed to list GCP projects: {e}"
                )
        except httpx.RequestError as e:
            lib_logger.error(f"Network error while listing GCP projects: {e}")

        raise ValueError(
            f"Could not auto-discover {provider} project ID. Possible causes:\n"
            "  1. The cloudaicompanion.googleapis.com API is not enabled (enable it in Google Cloud Console)\n"
            "  2. No active GCP projects exist for this account (create one in Google Cloud Console)\n"
            "  3. Account lacks necessary permissions\n"
            f"To manually specify a project, set {self._project_id_env_var} in your .env file."
        )
