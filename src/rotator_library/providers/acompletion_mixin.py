# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
ACompletionMixin — template method for providers with identical
acompletion scaffolding (iFlow, Qwen Code).

Subclasses must define:
  - provider_name: str   (display name for log messages, e.g. "iFlow")
  - llm_provider: str    (machine name for RateLimitError, e.g. "iflow")
  - _convert_chunk_to_openai(chunk, model) -> generator

And may override:
  - _get_stream_endpoint(model) -> str   (default: "/chat/completions")
  - _get_extra_headers() -> dict          (default: {})
"""

import httpx
import logging
from typing import Union, AsyncGenerator

from .provider_interface import strip_provider_prefix, build_bearer_headers
from .base_streaming_provider import StreamingResponseMixin, parse_sse_stream
from ..timeout_config import TimeoutConfig
from ..transaction_logger import ProviderLogger
import litellm
from litellm.exceptions import RateLimitError

lib_logger = logging.getLogger("rotator_library")


class ACompletionMixin(StreamingResponseMixin):
    """
    Mixin that provides the full acompletion() template method.

    Requires the host class to also inherit from an auth base that supplies:
      - get_api_details(credential_path) -> (api_base, auth_token)
      - _refresh_token(credential_path, creds=None, force=True)
    and to define _build_request_payload / _convert_chunk_to_openai itself.
    """

    provider_name: str = ""
    llm_provider: str = ""

    def _get_stream_endpoint(self, model: str) -> str:
        return "/chat/completions"

    def _get_extra_headers(self) -> dict:
        return {}

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        credential_path = kwargs.pop("credential_identifier")
        transaction_context = kwargs.pop("transaction_context", None)
        model = kwargs["model"]

        file_logger = ProviderLogger(transaction_context)

        async def make_request():
            api_base, auth_token = await self.get_api_details(credential_path)

            model_name = strip_provider_prefix(model)
            kwargs_with_stripped_model = {**kwargs, "model": model_name}

            payload = self._build_request_payload(**kwargs_with_stripped_model)

            headers = {
                **build_bearer_headers(auth_token),
                "Accept": "text/event-stream",
                **self._get_extra_headers(),
            }

            endpoint = self._get_stream_endpoint(model)
            url = f"{api_base.rstrip('/')}{endpoint}"

            file_logger.log_request(payload)
            lib_logger.debug(f"{self.provider_name} Request URL: {url}")

            return client.stream(
                "POST",
                url,
                headers=headers,
                json=payload,
                timeout=TimeoutConfig.streaming(),
            )

        async def stream_handler(response_stream, attempt=1):
            try:
                async with response_stream as response:
                    if response.status_code >= 400:
                        error_text = await response.aread()
                        error_text = (
                            error_text.decode("utf-8")
                            if isinstance(error_text, bytes)
                            else error_text
                        )

                        if response.status_code == 401 and attempt == 1:
                            lib_logger.warning(
                                f"{self.provider_name} returned 401. Forcing token refresh and retrying once."
                            )
                            await self._refresh_token(credential_path, force=True)
                            retry_stream = await make_request()
                            async for chunk in stream_handler(
                                retry_stream, attempt=2
                            ):
                                yield chunk
                            return

                        elif (
                            response.status_code == 429
                            or "slow_down" in error_text.lower()
                        ):
                            raise RateLimitError(
                                f"{self.provider_name} rate limit exceeded: {error_text}",
                                llm_provider=self.llm_provider,
                                model=model,
                                response=response,
                            )

                        else:
                            error_msg = f"{self.provider_name} HTTP {response.status_code} error: {error_text}"
                            file_logger.log_error(error_msg)
                            raise httpx.HTTPStatusError(
                                f"HTTP {response.status_code}: {error_text}",
                                request=response.request,
                                response=response,
                            )

                    async for chunk in parse_sse_stream(
                        response,
                        provider_name=self.provider_name,
                        on_line=file_logger.log_response_chunk,
                    ):
                        for openai_chunk in self._convert_chunk_to_openai(
                            chunk, model
                        ):
                            yield litellm.ModelResponse(**openai_chunk)

            except httpx.HTTPStatusError:
                raise
            except Exception as e:
                file_logger.log_error(
                    f"Error during {self.provider_name} stream processing: {e}"
                )
                lib_logger.error(
                    f"Error during {self.provider_name} stream processing: {e}",
                    exc_info=True,
                )
                raise

        async def logging_stream_wrapper():
            openai_chunks = []
            try:
                async for chunk in stream_handler(await make_request()):
                    openai_chunks.append(chunk)
                    yield chunk
            finally:
                if openai_chunks:
                    final_response = self._stream_to_completion_response(
                        openai_chunks
                    )
                    file_logger.log_final_response(final_response.dict())

        if kwargs.get("stream"):
            return logging_stream_wrapper()
        else:

            async def non_stream_wrapper():
                chunks = [chunk async for chunk in logging_stream_wrapper()]
                return self._stream_to_completion_response(chunks)

            return await non_stream_wrapper()
