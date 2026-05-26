# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
Translation audit log for protocol translation hardening.

Records which fields were preserved, dropped, or transformed during
bidirectional Anthropic<->OpenAI translation.  Output is at DEBUG level
by default; set ``TRANSLATION_AUDIT_LOG=true`` to elevate to INFO.
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("rotator_library.anthropic_compat")


def _audit_log_enabled_info() -> bool:
    """Check if INFO-level audit logging is enabled via env var."""
    return os.environ.get("TRANSLATION_AUDIT_LOG", "").lower() in ("true", "1", "yes")


class TranslationAuditLog:
    """Structured audit log for protocol translation events."""

    @staticmethod
    def log(
        request_id: str,
        direction: str,
        fields_preserved: Optional[List[str]] = None,
        fields_dropped: Optional[List[str]] = None,
        fields_transformed: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a translation audit event.

        Args:
            request_id: Unique request identifier.
            direction: Translation direction, e.g. ``"anthropic_to_openai"``.
            fields_preserved: Field names passed through unchanged.
            fields_dropped: Field names that had no target equivalent.
            fields_transformed: Mapping of field name to
                ``{"before": ..., "after": ...}`` dicts.
        """
        fields_preserved = fields_preserved or []
        fields_dropped = fields_dropped or []
        fields_transformed = fields_transformed or {}

        audit_entry: Dict[str, Any] = {
            "request_id": request_id,
            "direction": direction,
            "fields_preserved": fields_preserved,
            "fields_dropped": fields_dropped,
            "fields_transformed": fields_transformed,
        }

        msg = "Translation audit: %s preserved=%s dropped=%s transformed=%s"
        args: tuple = (
            direction,
            fields_preserved,
            fields_dropped,
            {k: v for k, v in fields_transformed.items()},
        )

        if _audit_log_enabled_info():
            logger.info(msg, *args)
        else:
            logger.debug(msg, *args)

    @staticmethod
    def log_from_metadata(
        request_id: str,
        direction: str,
        translation_metadata: Dict[str, Any],
    ) -> None:
        """Log audit from translation metadata dict produced by translator.

        The translator embeds ``_translation_audit`` in its output; this
        helper extracts it and delegates to :meth:`log`.
        """
        audit = translation_metadata.get("_translation_audit", {})
        TranslationAuditLog.log(
            request_id=request_id,
            direction=direction,
            fields_preserved=audit.get("fields_preserved", []),
            fields_dropped=audit.get("fields_dropped", []),
            fields_transformed=audit.get("fields_transformed", {}),
        )
