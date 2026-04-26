# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import asyncio
import gzip as _gzip
import os


_COMPRESSIBLE_TYPES = frozenset(
    {"text/", "application/json", "application/xml", "application/javascript"}
)


def _hdr_lower(raw):
    return (raw.decode("latin-1") if isinstance(raw, bytes) else raw).lower()


def _hdr_str(raw):
    return raw.decode("latin-1") if isinstance(raw, bytes) else raw


def _filter_content_length(headers):
    return [(hk, hv) for hk, hv in headers if _hdr_lower(hk) != "content-length"]


SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
}


class _NoGzipForSSE:
    """GZip middleware that skips compression for streaming responses.

    Detects streaming via content-type (text/event-stream) or more_body=True
    on the first body chunk. Passes streaming chunks through immediately
    without buffering, avoiding TTFT overhead from GZipMiddleware.
    """

    _ACCEPT_ENCODING = b"accept-encoding"
    _GZIP = b"gzip"

    @staticmethod
    def _env_int(key: str, default: int) -> int:
        try:
            return int(os.getenv(key, str(default)))
        except (ValueError, TypeError):
            return default

    def __init__(self, app, minimum_size=None):
        self._app = app
        self._minimum_size = (
            self._env_int("GZIP_MIN_SIZE", 2048)
            if minimum_size is None
            else minimum_size
        )

    @staticmethod
    def _get_header(message, name):
        """Extract header value (decoded) from a response start message."""
        name_lower = name.lower() if isinstance(name, bytes) else name.encode().lower()
        for hk, hv in message.get("headers", []):
            if hk.lower() == name_lower:
                return hv.decode("latin-1") if isinstance(hv, bytes) else hv
        return None

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        accept_gzip = any(
            _NoGzipForSSE._GZIP in v
            for k, v in scope.get("headers", [])
            if k == _NoGzipForSSE._ACCEPT_ENCODING
        )

        initial_message = None
        compressor = None
        skip = False
        started = False
        passthrough = False

        async def _send(message):
            nonlocal initial_message, compressor, skip, started, passthrough

            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                existing = {_hdr_lower(h[0]) for h in headers}
                for name, value in SECURITY_HEADERS.items():
                    if name.lower() not in existing:
                        headers.append((name.encode(), value.encode()))
                initial_message = {**message, "headers": headers}
                if not accept_gzip:
                    passthrough = True
                    await send(initial_message)
                    return
                for h in headers:
                    hk = _hdr_lower(h[0])
                    hv = _hdr_str(h[1])
                    if hk == "content-type" and "text/event-stream" in hv:
                        skip = True
                        break
                    if hk == "content-length" and int(hv) < self._minimum_size:
                        skip = True
                        break
                if not skip:
                    ct = self._get_header(initial_message, b"content-type")
                    if ct and not any(t in ct for t in _COMPRESSIBLE_TYPES):
                        skip = True
                return

            if message["type"] != "http.response.body":
                await send(message)
                return

            if passthrough:
                await send(message)
                return

            body = message.get("body", b"")
            more = message.get("more_body", False)

            if not started:
                started = True
                if skip:
                    # SSE or too-small response — passthrough without compression
                    filtered_headers = _filter_content_length(initial_message.get("headers", []))
                    await send({**initial_message, "headers": filtered_headers})
                    await send(message)
                    return

                if more:
                    # Chunked non-SSE response — initialize compressor for streaming gzip
                    compressor = _gzip.compressobj(level=3)
                    filtered_headers = _filter_content_length(initial_message.get("headers", []))
                    filtered_headers.append((b"content-encoding", b"gzip"))
                    filtered_headers.append((b"vary", b"accept-encoding"))
                    await send({**initial_message, "headers": filtered_headers})
                    if body:
                        out = compressor.compress(body)
                        await send({"type": "http.response.body", "body": out, "more_body": True})
                    else:
                        await send(message)
                    return

                if len(body) < self._minimum_size:
                    await send(initial_message)
                    await send(message)
                    return

                out = await asyncio.get_running_loop().run_in_executor(
                    None, _gzip.compress, body, 3
                )
                if len(out) >= len(body):
                    await send(initial_message)
                    await send(message)
                    return

                headers = _filter_content_length(initial_message.get("headers", []))
                headers.append((b"content-encoding", b"gzip"))
                headers.append((b"vary", b"accept-encoding"))
                headers.append((b"content-length", str(len(out)).encode()))
                await send({**initial_message, "headers": headers})
                await send({"type": "http.response.body", "body": out, "more_body": False})
                return

            if skip or compressor is None:
                await send(message)
                return

            chunks = []
            if body:
                chunks.append(compressor.compress(body))
            if not more:
                chunks.append(compressor.flush())
                compressor = None
            out = b"".join(chunks)
            await send({"type": "http.response.body", "body": out, "more_body": more})

        await self._app(scope, receive, _send)
