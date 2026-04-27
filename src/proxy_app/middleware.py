# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import asyncio
import gzip as _gzip

from proxy_app.config import DEFAULT_GZIP_COMPRESSION_LEVEL, DEFAULT_GZIP_MIN_SIZE


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
    """Apply gzip compression while passing streaming responses through."""

    _ACCEPT_ENCODING = b"accept-encoding"
    _GZIP = b"gzip"

    def __init__(self, app, minimum_size=None):
        self._app = app
        self._minimum_size = (
            DEFAULT_GZIP_MIN_SIZE
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
        if not self._validate_request(scope):
            await self._app(scope, receive, send)
            return

        await self._process_request(scope, receive, send)

    def _validate_request(self, scope):
        return scope["type"] == "http"

    def _accepts_gzip(self, scope):
        return any(
            _NoGzipForSSE._GZIP in v
            for k, v in scope.get("headers", [])
            if k == _NoGzipForSSE._ACCEPT_ENCODING
        )

    async def _process_request(self, scope, receive, send):
        state = {
            "accept_gzip": self._accepts_gzip(scope),
            "initial_message": None,
            "compressor": None,
            "skip": False,
            "started": False,
            "passthrough": False,
        }

        async def _send(message):
            await self._process_response_message(message, state, send)

        await self._app(scope, receive, _send)

    async def _process_response_message(self, message, state, send):
        if message["type"] == "http.response.start":
            await self._handle_response_start(message, state, send)
            return

        if message["type"] != "http.response.body":
            await send(message)
            return

        await self._handle_response_body(message, state, send)

    async def _handle_response_start(self, message, state, send):
        headers = self._headers_with_security_defaults(message)
        state["initial_message"] = {**message, "headers": headers}
        if not state["accept_gzip"]:
            state["passthrough"] = True
            await send(state["initial_message"])
            return
        state["skip"] = self._should_skip_compression(headers, state["initial_message"])

    def _headers_with_security_defaults(self, message):
        headers = list(message.get("headers", []))
        existing = {_hdr_lower(h[0]) for h in headers}
        for name, value in SECURITY_HEADERS.items():
            if name.lower() not in existing:
                headers.append((name.encode(), value.encode()))
        return headers

    def _should_skip_compression(self, headers, initial_message):
        for h in headers:
            hk = _hdr_lower(h[0])
            hv = _hdr_str(h[1])
            if hk == "content-type" and "text/event-stream" in hv:
                return True
            if hk == "content-length":
                try:
                    if int(hv) < self._minimum_size:
                        return True
                except ValueError:
                    return True
        ct = self._get_header(initial_message, b"content-type")
        return bool(ct and not any(t in ct for t in _COMPRESSIBLE_TYPES))

    async def _handle_response_body(self, message, state, send):
        if state["passthrough"]:
            await send(message)
            return

        body = message.get("body", b"")
        more = message.get("more_body", False)

        if not state["started"]:
            state["started"] = True
            await self._handle_first_body(message, body, more, state, send)
            return

        await self._handle_later_body(message, body, more, state, send)

    async def _handle_first_body(self, message, body, more, state, send):
        initial_message = state["initial_message"]
        if state["skip"]:
            # SSE or too-small response — passthrough without compression
            filtered_headers = _filter_content_length(initial_message.get("headers", []))
            await send({**initial_message, "headers": filtered_headers})
            await send(message)
            return

        if more:
            await self._start_streaming_gzip(message, body, state, send)
            return

        if len(body) < self._minimum_size:
            await send(initial_message)
            await send(message)
            return

        await self._send_compressed_single_body(message, body, initial_message, send)

    async def _start_streaming_gzip(self, message, body, state, send):
        # Chunked non-SSE response — initialize compressor for streaming gzip
        state["compressor"] = _gzip.compressobj(level=DEFAULT_GZIP_COMPRESSION_LEVEL)
        initial_message = state["initial_message"]
        filtered_headers = _filter_content_length(initial_message.get("headers", []))
        filtered_headers.append((b"content-encoding", b"gzip"))
        filtered_headers.append((b"vary", b"accept-encoding"))
        await send({**initial_message, "headers": filtered_headers})
        if body:
            out = state["compressor"].compress(body)
            await send({"type": "http.response.body", "body": out, "more_body": True})
        else:
            await send(message)

    async def _send_compressed_single_body(self, message, body, initial_message, send):
        out = await asyncio.get_running_loop().run_in_executor(
            None, _gzip.compress, body, DEFAULT_GZIP_COMPRESSION_LEVEL
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

    async def _handle_later_body(self, message, body, more, state, send):
        compressor = state["compressor"]
        if state["skip"] or compressor is None:
            await send(message)
            return

        chunks = []
        if body:
            chunks.append(compressor.compress(body))
        if not more:
            chunks.append(compressor.flush())
            state["compressor"] = None
        out = b"".join(chunks)
        await send({"type": "http.response.body", "body": out, "more_body": more})
