# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import logging

from typing import Any

import orjson
from fastapi import APIRouter, Request, Depends, UploadFile, File, Form
from fastapi.responses import StreamingResponse, Response

logger = logging.getLogger(__name__)

from rotator_library import RotatingClient
from proxy_app.dependencies import get_rotating_client, verify_api_key
from proxy_app.routes._helpers import log_request_to_console
from proxy_app.routes.error_handler import handle_route_errors

router = APIRouter(tags=["audio"])


@router.post("/v1/audio/speech", response_model=None)
@handle_route_errors(error_format="openai", log_context="TTS request failed")
async def audio_speech(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
) -> StreamingResponse | Response:
    """
    OpenAI-compatible TTS (text-to-speech) endpoint.

    Accepts model, input, voice, speed, response_format and returns
    streaming audio content.
    """
    request_data = orjson.loads(await request.body())

    log_request_to_console(
        url=str(request.url),
        client_info=(request.client.host if request.client else "unknown", request.client.port if request.client else 0),
        request_data=request_data,
    )

    # Extract response_format for content-type mapping before passing to litellm.
    # litellm.aspeech misroutes response_format through get_optional_params
    # (chat/completions path) for some providers, causing TypeError.
    response_format = request_data.pop("response_format", "mp3")

    response = await client.aspeech(request=request, **request_data)

    content_type_map = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }
    content_type = content_type_map.get(response_format, "audio/mpeg")

    # Native Gemini TTS returns raw bytes directly
    if isinstance(response, (bytes, bytearray)):
        return Response(
            content=bytes(response),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{response_format}",
            },
        )

    # litellm.aspeech may return a dict error payload instead of raising
    if isinstance(response, dict):
        raise ValueError(response)

    # litellm.aspeech returns an httpx.Response with streaming content
    aiter = response.aiter_bytes()
    if hasattr(aiter, "__aiter__"):
        byte_iter = aiter
    else:
        byte_iter = await aiter

    async def _audio_stream():
        chunk_count = 0
        try:
            async for chunk in byte_iter:
                yield chunk
                chunk_count += 1
                if chunk_count % 50 == 0 and await request.is_disconnected():
                    logger.info("Client disconnected during audio streaming")
                    break
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error("Audio streaming error: %s", e)
        except Exception as e:
            logger.exception("Unexpected audio streaming error: %s", e)

    return StreamingResponse(
        _audio_stream(),
        media_type=content_type,
        headers={
            "Content-Disposition": f"attachment; filename=speech.{response_format}",
        },
    )


@router.post("/v1/audio/transcriptions")
@handle_route_errors(error_format="openai", log_context="Transcription request failed")
async def audio_transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
) -> Any:
    """
    OpenAI-compatible STT (speech-to-text) endpoint.

    Accepts multipart form data with an audio file and model name,
    returns transcription text.
    """
    # Use the SpooledTemporaryFile directly for streaming upload
    file_filename = file.filename or "audio.mp3"

    # Collect optional form fields
    kwargs = {"model": model, "file": (file_filename, file.file)}

    # Read additional form fields that may be present
    form = await request.form()
    for key in ("language", "prompt", "response_format", "temperature"):
        value = form.get(key)
        if value is not None:
            kwargs[key] = value if key != "temperature" else float(str(value))  # type: ignore[arg-type]

    log_request_to_console(
        url=str(request.url),
        client_info=(request.client.host if request.client else "unknown", request.client.port if request.client else 0),
        request_data={"model": model, "file": file_filename, **{k: v for k, v in kwargs.items() if k not in ("file",)}},
    )

    response = await client.atranscription(request=request, **kwargs)
    return response
