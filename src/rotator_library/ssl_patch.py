# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import os
import ssl as _ssl_module
import logging

AZURE_COMPATIBLE_CIPHERS = (
    "ECDH+AESGCM:DH+AESGCM:ECDH+AES256:DH+AES256:ECDH+AES128:DH+AES:"
    "ECDH+3DES:DH+3DES:RSA+AESGCM:RSA+AES:RSA+3DES:!aNULL:!MD5:!DSS"
)


def _safe_set_ciphers(ctx, cipher_string):
    """Set ciphers on an SSL context, silently skipping on Windows/Schannel."""
    try:
        ctx.set_ciphers(cipher_string)
    except Exception as exc:
        if os.name == "nt" and isinstance(exc, _ssl_module.SSLError):
            logging.warning("[SSL-FIX] Schannel does not support set_ciphers, skipping on Windows")
        else:
            raise


def _patch_aiohttp_connector():
    """Patch ssl module and aiohttp.TCPConnector to disable SSL verification."""
    try:
        _ssl_verify = os.environ.get("HTTP_SSL_VERIFY", "true").lower() != "false"

        if not _ssl_verify:
            # Global patch: make ssl.create_default_context() return unverified context
            _force_tls12 = os.environ.get("SSL_FORCE_TLS12", "false").lower() == "true"

            def _patched_create_default(*args, **kwargs):
                ctx = _ssl_module._create_unverified_context()
                if _force_tls12:
                    ctx.maximum_version = _ssl_module.TLSVersion.TLSv1_2
                    _safe_set_ciphers(ctx, AZURE_COMPATIBLE_CIPHERS)
                return ctx

            _ssl_module.create_default_context = _patched_create_default

            # Also patch _create_default_context if it exists
            if hasattr(_ssl_module, "_create_default_context"):
                _ssl_module._create_default_context = _patched_create_default

            logging.info(
                "[SSL-FIX] Global ssl.create_default_context patched to return unverified TLS 1.2 context"
            )

        # Patch aiohttp.TCPConnector
        from aiohttp import TCPConnector as _OriginalTCPConnector

        _original_init = _OriginalTCPConnector.__init__

        def _patched_init(self, *args, **kwargs):
            if not _ssl_verify:
                ssl_context = _ssl_module._create_unverified_context()
                if _force_tls12:
                    ssl_context.maximum_version = _ssl_module.TLSVersion.TLSv1_2
                    _safe_set_ciphers(ssl_context, AZURE_COMPATIBLE_CIPHERS)
                kwargs["ssl"] = ssl_context
            _original_init(self, *args, **kwargs)

        _OriginalTCPConnector.__init__ = _patched_init
        logging.info(f"[SSL-FIX] Patched aiohttp.TCPConnector: SSL_VERIFY={_ssl_verify}")

        # NOTE: ClientSession._request monkey-patch removed — it unconditionally
        # forced ssl=False on every request, overriding per-host skip logic in
        # http_client_pool.py.  The TCPConnector + ssl.create_default_context
        # patches above are sufficient for the aiohttp code path when SSL is
        # disabled globally.  The httpx clients used by the rotator already have
        # fine-grained per-host verify=False via HTTP_SSL_VERIFY_HOSTS.

    except ImportError:
        pass
    except Exception as e:
        logging.error(f"[SSL-FIX] Failed to patch aiohttp connector: {e}")


def _patch_litellm_ssl():
    """Patch litellm to disable SSL verification when HTTP_SSL_VERIFY=false.

    Must be called immediately after litellm import, before any API calls.
    Sets litellm.ssl_verify=False, creates httpx clients with verify=False,
    and sets SSL_VERIFY=False env var so litellm internals respect the flag.
    """
    _ssl_verify_env = os.environ.get("HTTP_SSL_VERIFY", "true").lower()
    if _ssl_verify_env != "false":
        return

    try:
        import litellm
        import httpx
        from rotator_library.timeout_config import TimeoutConfig

        litellm.ssl_verify = False
        logging.info("[SSL-FIX] Set litellm.ssl_verify = False")

        _litellm_timeout = TimeoutConfig.non_streaming()
        litellm.client_session = httpx.Client(verify=False, timeout=_litellm_timeout)
        litellm.aclient_session = httpx.AsyncClient(verify=False, timeout=_litellm_timeout)
        logging.info("[SSL-FIX] Created litellm.client_session and aclient_session with verify=False")

        os.environ["SSL_VERIFY"] = "False"
        logging.info("[SSL-FIX] Set SSL_VERIFY=False environment variable")

    except ImportError:
        pass
    except Exception as e:
        logging.error(f"[SSL-FIX] Failed to patch litellm SSL: {e}")
