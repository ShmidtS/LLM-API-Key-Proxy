# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import logging

import orjson
from fastapi import APIRouter, Request, Depends, UploadFile, File, Form
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

from rotator_library import RotatingClient
from proxy_app.dependencies import get_rotating_client, verify_api_key
from proxy_app.routes._helpers import log_request_to_console
from proxy_app.routes.error_handler import handle_route_errors

router = APIRouter(tags=["audio"])


@router.post("/v1/audio/speech")
@handle_route_errors(error_format="openai", log_context="TTS request failed")
async def audio_speech(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """
    OpenAI-compatible TTS (text-to-speech) endpoint.

    Accepts model, input, voice, speed, response_format and returns
    streaming audio content.
    """
    request_data = orjson.loads(await request.body())

    log_request_to_console(
        url=str(request.url),
        client_info=(request.client.host, request.client.port),
        request_data=request_data,
    )

    response = await client.aspeech(request=request, **request_data)

    # litellm.aspeech returns an httpx.Response with streaming content
    # Determine content type from response_format or default to audio/mpeg
    response_format = request_data.get("response_format", "mp3")
    content_type_map = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }
    content_type = content_type_map.get(response_format, "audio/mpeg")

    async def _audio_stream():
        chunk_count = 0
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
                chunk_count += 1
                if chunk_count % 50 == 0 and await request.is_disconnected():
                    logger.info("Client disconnected during audio streaming")
                    break
        except Exception as e:
            logger.exception("Audio streaming error: %s", e)

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
):
    """
    OpenAI-compatible STT (speech-to-text) endpoint.

    Accepts multipart form data with an audio file and model name,
    returns transcription text.
    """
    # Read file bytes and build kwargs for litellm.atranscription
    file_bytes = await file.read()
    file_filename = file.filename or "audio.mp3"

    # Collect optional form fields
    kwargs = {"model": model, "file": (file_filename, file_bytes)}

    # Read additional form fields that may be present
    form = await request.form()
    for key in ("language", "prompt", "response_format", "temperature"):
        value = form.get(key)
        if value is not None:
            kwargs[key] = value if key != "temperature" else float(value)

    log_request_to_console(
        url=str(request.url),
        client_info=(request.client.host, request.client.port),
        request_data={"model": model, "file": file_filename, **{k: v for k, v in kwargs.items() if k not in ("file",)}},
    )

    response = await client.atranscription(request=request, **kwargs)
    return response
