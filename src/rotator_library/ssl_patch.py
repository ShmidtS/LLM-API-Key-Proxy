# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import os
import ssl as _ssl_module

_AZURE_COMPATIBLE_CIPHERS = (
    "ECDHE-RSA-AES256-GCM-SHA384:"
    "ECDHE-RSA-AES128-GCM-SHA256:"
    "ECDHE-ECDSA-AES256-GCM-SHA384:"
    "ECDHE-ECDSA-AES128-GCM-SHA256:"
    "AES256-GCM-SHA384:"
    "AES128-GCM-SHA256"
)


def _patch_aiohttp_connector():
    """Patch ssl module and aiohttp.TCPConnector to disable SSL verification."""
    try:
        _ssl_verify = os.environ.get("HTTP_SSL_VERIFY", "true").lower() != "false"

        if not _ssl_verify:
            # Global patch: make ssl.create_default_context() return unverified context
            _original_create_default = _ssl_module.create_default_context

            _force_tls12 = os.environ.get("SSL_FORCE_TLS12", "false").lower() == "true"

            def _patched_create_default(*args, **kwargs):
                ctx = _ssl_module._create_unverified_context()
                if _force_tls12:
                    ctx.maximum_version = _ssl_module.TLSVersion.TLSv1_2
                    try:
                        ctx.set_ciphers(_AZURE_COMPATIBLE_CIPHERS)
                    except _ssl_module.SSLError:
                        pass
                    except Exception as exc:
                        if os.name == "nt" and isinstance(exc, _ssl_module.SSLError):
                            print("[SSL-FIX] Schannel does not support set_ciphers, skipping on Windows")
                        else:
                            raise
                return ctx

            _ssl_module.create_default_context = _patched_create_default

            # Also patch _create_default_context if it exists
            if hasattr(_ssl_module, "_create_default_context"):
                _ssl_module._create_default_context = _patched_create_default

            print(
                "[SSL-FIX] Global ssl.create_default_context patched to return unverified TLS 1.2 context"
            )

        # Patch aiohttp.TCPConnector
        import aiohttp
        from aiohttp import TCPConnector as _OriginalTCPConnector

        _original_init = _OriginalTCPConnector.__init__

        def _patched_init(self, *args, **kwargs):
            if not _ssl_verify:
                ssl_context = _ssl_module._create_unverified_context()
                if _force_tls12:
                    ssl_context.maximum_version = _ssl_module.TLSVersion.TLSv1_2
                    try:
                        ssl_context.set_ciphers(_AZURE_COMPATIBLE_CIPHERS)
                    except _ssl_module.SSLError:
                        pass
                    except Exception as exc:
                        if os.name == "nt" and isinstance(exc, _ssl_module.SSLError):
                            print("[SSL-FIX] Schannel does not support set_ciphers, skipping on Windows")
                        else:
                            raise
                kwargs["ssl"] = ssl_context
            _original_init(self, *args, **kwargs)

        _OriginalTCPConnector.__init__ = _patched_init
        print(f"[SSL-FIX] Patched aiohttp.TCPConnector: SSL_VERIFY={_ssl_verify}")

        # Patch aiohttp.ClientSession to disable SSL verification
        try:
            _original_request = aiohttp.ClientSession._request

            async def _patched_request(self, *args, **kwargs):
                # Force ssl=False to disable SSL verification
                kwargs["ssl"] = False
                return await _original_request(self, *args, **kwargs)

            aiohttp.ClientSession._request = _patched_request
            print("[SSL-FIX] Patched aiohttp.ClientSession._request to use ssl=False")
        except Exception as e:
            print(f"[SSL-FIX] Failed to patch ClientSession: {e}")

    except ImportError:
        pass
    except Exception as e:
        print(f"[SSL-FIX] Failed to patch: {e}")
