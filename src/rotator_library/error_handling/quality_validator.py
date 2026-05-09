# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import re

from ..config.defaults import env_bool, env_float
from ..error_types import GarbageResponseError

_WORD_SPLIT_RE = re.compile(r'[\s\\/"\']+')
_CODE_FENCE_RE = re.compile(r'```[\s\S]*?```')
_CODE_PATTERN_RES = tuple(re.compile(p, re.IGNORECASE) for p in (
    r'\bimport\s+\w+', r'\bfrom\s+\w+\s+import\b',
    r'\bclass\s+\w+',
    r'\bdef\s+\w+', r'\breturn\s+',
))
_PATH_PATTERN_RES = tuple(re.compile(p) for p in (
    r'[A-Z]:\\[\w\s\\]+\.\w{2,4}',
    r'/home/\w', r'/usr/\w', r'/var/\w', r'/tmp/\w',
    r'C:\\Users\\',
))


def validate_response_quality(response, provider: str = "", model: str = ""):
    """
    Validate that a model response contains meaningful content, not garbage.

    Checks both OpenAI-format (ModelResponse/dict with choices) and
    Anthropic-format (dict with content blocks) responses.

    Raises GarbageResponseError if the response is detected as garbage.
    Returns True if the response appears valid.

    Garbage indicators:
    - High word repetition (unique/total ratio < threshold)
    - Code fragment injection (import, from, extension keywords in non-code context)
    - File path leakage (C:\\..., /home/, /usr/ in response)
    - Repetitive token flooding (same short token repeated many times)
    """
    if not env_bool("GARBAGE_DETECTION_ENABLED", True):
        return True

    repetition_threshold = env_float("GARBAGE_REPETITION_THRESHOLD", 0.25)

    text_parts = []

    if isinstance(response, dict):
        if "content" in response and isinstance(response["content"], list):
            for block in response["content"]:
                if isinstance(block, dict):
                    for key in ("text", "thinking"):
                        val = block.get(key)
                        if isinstance(val, str) and len(val) > 50:
                            text_parts.append(val)
        if "choices" in response and isinstance(response["choices"], list):
            for choice in response["choices"]:
                msg = choice.get("message", {})
                content = msg.get("content")
                if isinstance(content, str) and len(content) > 50:
                    text_parts.append(content)
    elif hasattr(response, "choices"):
        for choice in response.choices:
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                content = choice.message.content
                if isinstance(content, str) and len(content) > 50:
                    text_parts.append(content)

    if not text_parts:
        return True

    full_text = " ".join(text_parts)

    words = _WORD_SPLIT_RE.split(full_text)
    words = [w for w in words if len(w) > 2]
    if len(words) >= 20:
        unique_words = set(w.lower() for w in words)
        ratio = len(unique_words) / len(words)
        if ratio < repetition_threshold:
            raise GarbageResponseError(
                provider=provider, model=model,
                reason=f"High repetition ratio: {ratio:.2f} < {repetition_threshold} "
                       f"(unique {len(unique_words)}/{len(words)} words)"
            )

    text_outside_fences = _CODE_FENCE_RE.sub('', full_text)
    code_hits = sum(len(p.findall(text_outside_fences)) for p in _CODE_PATTERN_RES)
    if code_hits >= 10:
        raise GarbageResponseError(
            provider=provider, model=model,
            reason=f"Code fragment injection: {code_hits} code patterns detected outside fences"
        )

    path_hits = sum(len(p.findall(full_text)) for p in _PATH_PATTERN_RES)
    if path_hits >= 2:
        raise GarbageResponseError(
            provider=provider, model=model,
            reason=f"File path leakage: {path_hits} paths detected in content"
        )

    token_counts = {}
    for w in words:
        wl = w.lower()
        if 2 < len(wl) < 15:
            token_counts[wl] = token_counts.get(wl, 0) + 1
    if token_counts:
        max_count = max(token_counts.values())
        max_token = max(token_counts, key=lambda k: token_counts[k])
        if max_count >= 8 and max_count / max(len(words), 1) > 0.15:
            raise GarbageResponseError(
                provider=provider, model=model,
                reason=f"Token flooding: '{max_token}' repeated {max_count} times "
                       f"({max_count}/{len(words)} = {max_count/max(len(words),1):.1%})"
            )

    return True
