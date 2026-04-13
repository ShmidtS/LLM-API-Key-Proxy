# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import logging


class RotatorDebugFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.DEBUG and record.name.startswith(
            "rotator_library"
        )


class NoLiteLLMLogFilter(logging.Filter):
    def filter(self, record):
        return not record.name.startswith("LiteLLM")


class _NoGzipForSSE:
    """GZip middleware that skips compression for streaming responses.

    Detects streaming via content-type (text/event-stream) or more_body=True
    on the first body chunk. Passes streaming chunks through immediately
    without buffering, avoiding TTFT overhead from GZipMiddleware.
    """

    _ACCEPT_ENCODING = b"accept-encoding"
    _GZIP = b"gzip"

    def __init__(self, app, minimum_size=1000):
        self._app = app
        self._minimum_size = minimum_size

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        accept_gzip = any(
            _NoGzipForSSE._GZIP in v
            for k, v in scope.get("headers", [])
            if k == _NoGzipForSSE._ACCEPT_ENCODING
        )

        if not accept_gzip:
            await self._app(scope, receive, send)
            return

        import gzip as _gzip

        initial_message = None
        compressor = None
        skip = False
        started = False

        async def _send(message):
            nonlocal initial_message, compressor, skip, started

            if message["type"] == "http.response.start":
                initial_message = message
                headers = message.get("headers", [])
                for h in headers:
                    hk = (h[0].decode("latin-1") if isinstance(h[0], bytes) else h[0]).lower()
                    hv = h[1].decode("latin-1") if isinstance(h[1], bytes) else h[1]
                    if hk == "content-type" and "text/event-stream" in hv:
                        skip = True
                        break
                    if hk == "content-length" and int(hv) < self._minimum_size:
                        skip = True
                        break
                return

            if message["type"] != "http.response.body":
                await send(message)
                return

            body = message.get("body", b"")
            more = message.get("more_body", False)

            if not started:
                started = True
                if skip or more:
                    skip = True
                    await send(initial_message)
                    await send(message)
                    return

                if len(body) < self._minimum_size:
                    await send(initial_message)
                    await send(message)
                    return

                out = _gzip.compress(body, compresslevel=6)
                if len(out) >= len(body):
                    await send(initial_message)
                    await send(message)
                    return

                headers = [
                    (hk, hv)
                    for hk, hv in initial_message.get("headers", [])
                    if (hk.decode("latin-1") if isinstance(hk, bytes) else hk).lower()
                    != "content-length"
                ]
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
