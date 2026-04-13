# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import logging

import orjson
import litellm
from fastapi import APIRouter, Request, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from rotator_library import RotatingClient
from proxy_app.dependencies import get_rotating_client, verify_api_key
from proxy_app.streaming import handle_litellm_error
from proxy_app.request_logger import log_request_to_console

router = APIRouter(tags=["audio"])


@router.post("/v1/audio/speech")
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
    try:
        request_data = orjson.loads(await request.body())

        log_request_to_console(
            url=str(request.url),
            headers=request.headers,
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
            async for chunk in response.aiter_bytes():
                yield chunk

        return StreamingResponse(
            _audio_stream(),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{response_format}",
            },
        )

    except Exception as e:
        if isinstance(e, (litellm.InvalidRequestError, ValueError, litellm.ContextWindowExceededError,
                          litellm.AuthenticationError, litellm.RateLimitError,
                          litellm.ServiceUnavailableError, litellm.APIConnectionError,
                          litellm.Timeout, litellm.InternalServerError, litellm.OpenAIError)):
            raise handle_litellm_error(e, error_format="openai")
        logging.error(f"TTS request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/audio/transcriptions")
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
    try:
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
            headers=request.headers,
            client_info=(request.client.host, request.client.port),
            request_data={"model": model, "file": file_filename, **{k: v for k, v in kwargs.items() if k not in ("file",)}},
        )

        response = await client.atranscription(request=request, **kwargs)
        return response

    except Exception as e:
        if isinstance(e, (litellm.InvalidRequestError, ValueError, litellm.ContextWindowExceededError,
                          litellm.AuthenticationError, litellm.RateLimitError,
                          litellm.ServiceUnavailableError, litellm.APIConnectionError,
                          litellm.Timeout, litellm.InternalServerError, litellm.OpenAIError)):
            raise handle_litellm_error(e, error_format="openai")
        logging.error(f"Transcription request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
