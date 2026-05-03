import logging
import re
from collections.abc import Mapping
from typing import Callable
from typing import Any, Optional

import litellm  # type: ignore[import-untyped]
from litellm.exceptions import BadRequestError as LiteBadRequestError, NotFoundError as LiteNotFoundError, APIError as LiteAPIError  # type: ignore[import-untyped]
from litellm.llms.openai.common_utils import OpenAIError  # type: ignore[import-untyped]

from ..error_types import NoAvailableKeysError
from ..config.defaults import MEDIA_GLOBAL_TIMEOUT
from ..utils.model_utils import extract_provider_from_model, normalize_model_string

lib_logger = logging.getLogger("rotator_library")

_IMAGE_URL_RE = re.compile(r"https?://\S+\.(?:png|jpg|jpeg|webp|gif)\S*")
_DATA_IMAGE_RE = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+")

# Image-only model detection — these models are rejected by /chat/completions upstream
# and must be redirected to /v1/images/generations. Covers flux, z-image, dall-e, sd3, etc.
_IMAGE_ONLY_SUBSTRINGS = (
    "z-image",
    "flux-",
    "gpt-image",
    "dall-e",
    "sd3",
    "stable-diffusion",
    "imagen",
    "firefly",
    "cogview-4",
)
_IMAGE_ONLY_SUFFIXES = (
    "-image",
    "-image-pro",
    "-image-turbo",
    "-image-gen",
)
_IMAGE_ONLY_PATH_FRAGMENTS = ("/flux-", "/image-")

# Allow-list of params accepted by image-generation endpoints.
# Chat-specific params (messages, tools, stream, max_tokens, etc.)
# are deliberately excluded.
_IMAGE_PASSTHROUGH_PARAMS = {
    "n",
    "size",
    "quality",
    "style",
    "response_format",
    "user",
    "extra_headers",
    "extra_body",
    "timeout",
}

_IMAGE_NATIVE_PROVIDERS = {"fireworks", "fireworks_ai", "qwen", "dashscope", "zai", "z.ai"}


def is_image_only_model(model: str) -> bool:
    """Return True if model name matches known image-only patterns.

    Used to redirect chat-completion calls for image models (flux, z-image,
    dall-e, etc.) to the image generation endpoint.
    """
    if not model:
        return False
    m = model.lower()
    if any(s in m for s in _IMAGE_ONLY_SUBSTRINGS):
        return True
    if any(m.endswith(sfx) for sfx in _IMAGE_ONLY_SUFFIXES):
        return True
    if any(frag in m for frag in _IMAGE_ONLY_PATH_FRAGMENTS):
        return True
    return False


class MediaMixin:
    def _is_image_only_model(self, model: str) -> bool:
        """Instance wrapper around module-level is_image_only_model helper."""
        return is_image_only_model(model)

    def _extract_prompt_from_chat_messages(self, messages: list) -> Optional[str]:
        """Extract the last user-message text content to use as image prompt.

        Supports both string content and list-of-parts content (OpenAI vision
        format). Returns None when no user text is found.
        """
        if not messages:
            return None
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                text = content.strip()
                return text or None
            if isinstance(content, list):
                parts = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        text_val = part.get("text")
                        if isinstance(text_val, str) and text_val:
                            parts.append(text_val)
                if parts:
                    return "\n".join(parts)
        return None

    def _media_request(
        self,
        endpoint_fn: Callable,
        request: Optional[Any] = None,
        pre_request_callback: Optional[Callable] = None,
        **kwargs,
    ) -> Any:
        return self._rate_limited_execute(
            endpoint_fn,
            request=request,
            pre_request_callback=pre_request_callback,
            **kwargs,
        )

    async def aimage_generation(
        self,
        request: Optional[Any] = None,
        pre_request_callback: Optional[Callable] = None,
        **kwargs,
    ) -> Any:
        """Generate an image via the image generation endpoint."""
        # Auto-resolve unsupported image sizes — let the model pick the best fit
        size = kwargs.get("size")
        if size and size.lower() not in {
            "1024x1024",
            "1024x1536",
            "1536x1024",
            "1792x1024",
            "1024x1792",
            "auto",
        }:
            kwargs = kwargs.copy()
            kwargs["size"] = "auto"
            lib_logger.info(
                "Remapping unsupported image size %s to auto for model %s",
                size,
                kwargs.get("model", ""),
            )

        response_format = kwargs.get("response_format")
        if isinstance(response_format, Mapping):
            response_format_type = response_format.get("type")
            if response_format_type in {"image", "url"}:
                kwargs = kwargs.copy()
                kwargs["response_format"] = "url"
            elif response_format_type in {"b64_json", "base64"}:
                kwargs = kwargs.copy()
                kwargs["response_format"] = "b64_json"

        model = normalize_model_string(str(kwargs.get("model", "")))
        native_provider = kwargs.pop("_native_provider", None)
        provider = native_provider or extract_provider_from_model(model)
        model_name = model.split("/", 1)[1] if "/" in model else model
        api_base = str(kwargs.get("api_base", "")).lower()
        if provider == "openai" and ("dashscope" in api_base or model_name.startswith("z-image")):
            provider = "qwen"
        if provider in _IMAGE_NATIVE_PROVIDERS:
            kwargs = kwargs.copy()
            kwargs["model"] = model
            kwargs["_native_provider"] = provider
            return await self._rate_limited_execute(
                self._native_image_generation,
                request=request,
                pre_request_callback=pre_request_callback,
                _global_timeout=MEDIA_GLOBAL_TIMEOUT,
                **kwargs,
            )

        # Truncate long prompts — some models reject content > 1000 chars
        # at /images/generations (e.g. Qwen/DashScope returns 400).
        prompt = kwargs.get("prompt")
        if prompt and isinstance(prompt, str) and len(prompt) > 1000:
            kwargs = kwargs.copy()
            kwargs["prompt"] = prompt[:997] + "..."
            lib_logger.info(
                "Truncated image prompt from %d to 1000 chars for model %s",
                len(prompt), kwargs.get("model", ""),
            )

        try:
            response = await self._rate_limited_execute(
                litellm.aimage_generation,
                request=request,
                pre_request_callback=pre_request_callback,
                _global_timeout=MEDIA_GLOBAL_TIMEOUT,
                **kwargs,
            )
            if self._is_image_endpoint_mismatch_response(response):
                lib_logger.info(
                    "Provider doesn't support /images/generations, falling back to /chat/completions for model=%s",
                    kwargs.get("model", ""),
                )
                return await self._image_via_chat_completion(
                    request=request,
                    pre_request_callback=pre_request_callback,
                    **kwargs,
                )
            return response
        except (
            LiteBadRequestError,
            LiteNotFoundError,
            LiteAPIError,
            OpenAIError,
        ) as e:
            err_lower = str(e).lower()
            is_endpoint_mismatch = self._is_image_endpoint_mismatch_text(err_lower)
            is_html_404 = "<!doctype" in err_lower and ("not found" in err_lower or "404" in err_lower)
            if not is_endpoint_mismatch and not is_html_404:
                raise
            if self._is_image_only_model(str(kwargs.get("model", ""))):
                lib_logger.error(
                    "Provider doesn't support /images/generations for image-only model=%s. Failing fast.",
                    kwargs.get("model", ""),
                )
                raise NoAvailableKeysError(
                    f"Model {kwargs.get('model', 'unknown')} is not supported by this provider for image generation"
                ) from e

            lib_logger.info(
                "Provider doesn't support /images/generations, falling back to /chat/completions for model=%s",
                kwargs.get("model", ""),
            )
            try:
                return await self._image_via_chat_completion(
                    request=request,
                    pre_request_callback=pre_request_callback,
                    **kwargs,
                )
            except (LiteNotFoundError, OpenAIError) as chat_e:
                chat_err_lower = str(chat_e).lower()
                if (
                    isinstance(chat_e, OpenAIError)
                    and "not found" not in chat_err_lower
                    and "404" not in chat_err_lower
                ):
                    raise
                lib_logger.error(
                    "Provider doesn't support /images/generations or /chat/completions for image model=%s. Failing fast.",
                    kwargs.get("model", ""),
                )
                raise NoAvailableKeysError(
                    f"Model {kwargs.get('model', 'unknown')} is not supported by this provider for image generation"
                ) from chat_e

    async def _native_image_generation(self, **kwargs) -> Any:
        model = normalize_model_string(str(kwargs.get("model", "")))
        provider = kwargs.pop("_native_provider", None) or extract_provider_from_model(model)
        model_name = model.split("/", 1)[1] if "/" in model else model
        if provider == "openai" and model_name.startswith("z-image"):
            provider = "qwen"
        provider = {"dashscope": "qwen", "fireworks_ai": "fireworks", "z.ai": "zai"}.get(provider, provider)
        api_key = kwargs.pop("api_key")
        provider_plugin = self._get_provider_instance(provider)
        if not provider_plugin or not hasattr(provider_plugin, "native_image_generation"):
            raise ValueError(f"Unsupported native image provider: {provider}")
        http_client = await self._get_http_client_async(streaming=False)
        data = await provider_plugin.native_image_generation(
            http_client, api_key, timeout=kwargs.get("timeout", getattr(self, "global_timeout", 60)), **kwargs
        )
        return self._native_image_response(data)

    def _native_image_response(self, data: Any) -> dict:
        import time as _time

        images = self._extract_native_images(data)
        return {"created": int(_time.time()), "data": images, "object": "list"}

    def _extract_native_images(self, data: Any) -> list[dict]:
        if not isinstance(data, Mapping):
            return []
        images = []
        for key in ("base64", "b64_json"):
            if isinstance(data.get(key), str):
                images.append({"b64_json": data[key]})
        for key in ("url", "image_url", "sample", "image"):
            if isinstance(data.get(key), str):
                value = data[key]
                if key == "image" and not value.startswith("http"):
                    images.append({"b64_json": value})
                else:
                    images.append({"url": value})
        result = data.get("result")
        if isinstance(result, str):
            images.append({"url": result})
        elif isinstance(result, Mapping):
            images.extend(self._extract_native_images(result))
        output = data.get("output")
        if isinstance(output, Mapping):
            for key in ("results", "images"):
                items = output.get(key)
                if isinstance(items, list):
                    for item in items:
                        images.extend(self._extract_native_images(item))
            choices = output.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    images.extend(self._extract_native_images(choice))
            for key in ("url", "image_url"):
                if isinstance(output.get(key), str):
                    images.append({"url": output[key]})
        if isinstance(data.get("data"), list):
            for item in data["data"]:
                images.extend(self._extract_native_images(item))
        for key in ("results", "images"):
            if isinstance(data.get(key), list):
                for item in data[key]:
                    images.extend(self._extract_native_images(item))
        return images

    def _is_image_endpoint_mismatch_response(self, response: Any) -> bool:
        if not isinstance(response, Mapping):
            return False
        error = response.get("error")
        if not isinstance(error, Mapping):
            return False
        error_type = str(error.get("type", "")).lower()
        message = str(error.get("message", "")).lower()
        return error_type == "invalid_request" and self._is_image_endpoint_mismatch_text(message)

    def _is_image_endpoint_mismatch_text(self, text: str) -> bool:
        return any(
            pattern in text
            for pattern in (
                "only accepts the path",
                "invalid_path",
                "path not found",
                "not found: /v1/images/generations",
                "/v1/images/generations",
                "images/generations endpoint",
                "image generation endpoint",
                "endpoint does not support",
                "endpoint not found",
            )
        )

    async def _image_via_chat_completion(
        self,
        request: Optional[Any] = None,
        pre_request_callback: Optional[Callable] = None,
        **kwargs,
    ) -> Any:
        """Route image generation through /chat/completions as fallback."""
        import time as _time

        model = kwargs.get("model", "")
        prompt = kwargs.get("prompt", "")
        n = kwargs.get("n", 1)
        size = kwargs.get("size", "1024x1024")

        # Truncate prompt to avoid exceeding chat API content limits (e.g. Qwen)
        max_prompt_len = 2000
        if prompt and len(prompt) > max_prompt_len:
            prompt = prompt[:max_prompt_len - 3] + "..."
            lib_logger.info(
                "Truncated image prompt from %d to %d chars for chat fallback",
                len(kwargs.get("prompt", "")), max_prompt_len,
            )

        size_hint = f" (size: {size})" if size else ""
        user_content = f"Generate an image: {prompt}{size_hint}"

        chat_kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": user_content}],
            "stream": False,
        }
        # Drop response_format — chat APIs reject 'image' (only accept json_object/text)
        for key in ("temperature", "top_p", "seed"):
            if key in kwargs:
                chat_kwargs[key] = kwargs[key]

        chat_resp = await self._rate_limited_execute(
            litellm.acompletion,
            request=request,
            pre_request_callback=pre_request_callback,
            **chat_kwargs,
        )

        if isinstance(chat_resp, Mapping) and "error" in chat_resp:
            return chat_resp

        images = []
        for choice in chat_resp.get("choices") or []:
            msg = choice.get("message", {})
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)

            if "http" not in content and "data:image/" not in content:
                continue
            url_matches = [m.group() for m in _IMAGE_URL_RE.finditer(content)]
            b64_matches = [m.group() for m in _DATA_IMAGE_RE.finditer(content)]

            for url in url_matches:
                images.append({"url": url})
            for b64 in b64_matches:
                images.append({"b64_json": b64.split(",", 1)[1] if "," in b64 else b64})

            if not url_matches and not b64_matches and content.strip():
                images.append({"url": content.strip()})

        while len(images) < n and images:
            images.append(images[-1])

        return {
            "created": int(_time.time()),
            "data": images[:n] if n else images,
            "object": "list",
        }

    def aimage_edit(
        self,
        request: Optional[Any] = None,
        pre_request_callback: Optional[Callable] = None,
        **kwargs,
    ) -> Any:
        """Edit an image via the image edit endpoint."""
        return self._media_request(
            litellm.aimage_edit,
            request=request,
            pre_request_callback=pre_request_callback,
            **kwargs,
        )

    def aimage_variation(
        self,
        request: Optional[Any] = None,
        pre_request_callback: Optional[Callable] = None,
        **kwargs,
    ) -> Any:
        """Create a variation of an image via the image variation endpoint."""
        return self._media_request(
            litellm.aimage_variation,
            request=request,
            pre_request_callback=pre_request_callback,
            **kwargs,
        )

    def aspeech(
        self,
        request: Optional[Any] = None,
        pre_request_callback: Optional[Callable] = None,
        **kwargs,
    ) -> Any:
        """Generate speech audio via the speech endpoint.

        Gemini TTS models are routed to _native_gemini_tts because
        litellm.aspeech() sends them to generateContent without
        responseModalities, causing 500 errors.
        """
        model = normalize_model_string(str(kwargs.get("model", "")))
        if model.lower().startswith("gemini/") and "tts" in model.lower():
            return self._rate_limited_execute(
                self._native_gemini_tts,
                request=request,
                pre_request_callback=pre_request_callback,
                **kwargs,
            )
        # Mistral TTS models are routed natively because LiteLLM's Mistral
        # provider only exposes /chat/completions and /embeddings, so
        # litellm.aspeech() fails with "Unable to map the custom llm provider".
        if (
            model.lower().startswith("mistral/")
            and "tts" in model.lower()
        ):
            return self._rate_limited_execute(
                self._native_mistral_tts,
                request=request,
                pre_request_callback=pre_request_callback,
                **kwargs,
            )
        return self._media_request(
            litellm.aspeech,
            request=request,
            pre_request_callback=pre_request_callback,
            **kwargs,
        )

    async def _native_gemini_tts(self, **kwargs) -> Any:
        """Call Gemini TTS API directly with responseModalities=['AUDIO'].

        litellm.aspeech() and litellm.acompletion() both fail for Gemini TTS
        because they don't include responseModalities in the API payload.
        """
        import base64 as _b64

        model = normalize_model_string(str(kwargs.pop("model", "")))
        api_key = kwargs.pop("api_key")
        input_text = kwargs.pop("input", "")
        voice = kwargs.pop("voice", "Kore")
        speed = kwargs.pop("speed", None)
        timeout = kwargs.pop("timeout", getattr(self, "global_timeout", 60))

        model_name = model.split("/", 1)[1] if "/" in model else model
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

        payload = {
            "contents": [{"role": "user", "parts": [{"text": input_text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": voice,
                        }
                    }
                },
            },
        }
        if speed is not None:
            payload["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["speed"] = float(speed)

        http_client = await self._get_http_client_async(streaming=False)
        resp = await http_client.post(url, json=payload, timeout=timeout)
        if resp.status_code != 200:
            lib_logger.error(
                "Gemini TTS API error %d: %s | payload: %s",
                resp.status_code, resp.text[:500], str(payload)[:500],
            )
        resp.raise_for_status()
        data = resp.json()

        # Extract audio from response
        candidates = data.get("candidates", [])
        for candidate in candidates:
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                inline_data = part.get("inlineData")
                if inline_data and inline_data.get("data"):
                    return _b64.b64decode(inline_data["data"])

        lib_logger.error("No audio data in Gemini TTS response: %s", list(data.keys()))
        raise ValueError(f"Gemini TTS returned no audio data: {list(data.keys())}")

    async def _native_mistral_tts(self, **kwargs) -> Any:
        """Call Mistral TTS API directly.

        LiteLLM's Mistral provider does not expose /audio/speech, so
        litellm.aspeech() fails with provider-mapping errors.  We hit the
        OpenAI-compatible endpoint directly.
        """
        model = normalize_model_string(str(kwargs.pop("model", "")))
        api_key = kwargs.pop("api_key")
        input_text = kwargs.pop("input", "")
        voice = kwargs.pop("voice", "alloy")
        response_format = kwargs.pop("response_format", "mp3")
        speed = kwargs.pop("speed", None)
        timeout = kwargs.pop("timeout", getattr(self, "global_timeout", 60))

        model_name = model.split("/", 1)[1] if "/" in model else model
        url = "https://api.mistral.ai/v1/audio/speech"

        payload = {
            "model": model_name,
            "input": input_text,
            "voice": voice,
            "response_format": response_format,
        }
        if speed is not None:
            payload["speed"] = float(speed)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        http_client = await self._get_http_client_async(streaming=False)
        resp = await http_client.post(url, json=payload, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            lib_logger.error(
                "Mistral TTS API error %d: %s | payload: %s",
                resp.status_code, resp.text[:500], str(payload)[:500],
            )
        resp.raise_for_status()
        return resp.content

    def atranscription(
        self,
        request: Optional[Any] = None,
        pre_request_callback: Optional[Callable] = None,
        **kwargs,
    ) -> Any:
        """Transcribe audio via the transcription endpoint."""
        return self._media_request(
            litellm.atranscription,
            request=request,
            pre_request_callback=pre_request_callback,
            **kwargs,
        )