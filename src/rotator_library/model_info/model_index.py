# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
Fast lookup index for model ID resolution.

Extracted from model_info_service for modularity.
"""

import functools
import re
from typing import Dict, List


@functools.lru_cache(maxsize=128)
def _normalize_version_pattern(name: str) -> str:
    """
    Normalize version patterns in model names for fuzzy matching.

    Converts various version formats to a canonical form:
    - claude-opus-4-5 -> claude-opus-4.5
    - claude-opus-4.5 -> claude-opus-4.5
    - gemini-2-0-flash -> gemini-2.0-flash
    - gemini-2-5-pro -> gemini-2.5-pro

    Only applies to patterns that look like versions (digit-digit at end).
    """
    # Pattern matches: -X-Y at end of string or before another dash/segment
    # where X and Y are digits (like -4-5, -2-0, -2-5)
    # This converts 4-5 to 4.5, 2-0 to 2.0, etc.
    normalized = re.sub(r"-(\d+)-(\d+)(?=-|$)", r"-\1.\2", name)
    return normalized


class ModelIndex:
    """Fast lookup structure for model ID resolution."""

    def __init__(self):
        self._by_full_id: Dict[str, str] = {}  # normalized_id -> canonical_id
        self._by_suffix: Dict[str, List[str]] = {}  # short_name -> [canonical_ids]
        self._by_normalized: Dict[
            str, List[str]
        ] = {}  # normalized_name -> [canonical_ids]

    def clear(self):
        """Reset the index."""
        self._by_full_id.clear()
        self._by_suffix.clear()
        self._by_normalized.clear()

    def entry_count(self) -> int:
        """Return total number of suffix index entries."""
        return sum(len(v) for v in self._by_suffix.values())

    def add(self, canonical_id: str):
        """Index a canonical model ID for various lookup patterns."""
        self._by_full_id[canonical_id] = canonical_id

        segments = canonical_id.split("/")
        if len(segments) >= 2:
            # Index by everything after first segment
            partial = "/".join(segments[1:])
            self._by_suffix.setdefault(partial, []).append(canonical_id)

            # Index by final segment only
            if len(segments) >= 3:
                tail = segments[-1]
                self._by_suffix.setdefault(tail, []).append(canonical_id)

            # Index by normalized version pattern (e.g., claude-opus-4.5)
            # This allows 4-5 queries to match 4.5 entries and vice versa
            normalized_partial = _normalize_version_pattern(partial)
            if normalized_partial != partial:
                self._by_normalized.setdefault(normalized_partial, []).append(
                    canonical_id
                )

    def resolve(self, query: str) -> List[str]:
        """Find all canonical IDs matching a query."""
        # Direct match
        if query in self._by_full_id:
            return [self._by_full_id[query]]

        # Try with openrouter prefix
        prefixed = f"openrouter/{query}"
        if prefixed in self._by_full_id:
            return [self._by_full_id[prefixed]]

        # Extract search terms from query
        search_keys = []
        parts = query.split("/")
        if len(parts) >= 2:
            search_keys.append("/".join(parts[1:]))
            search_keys.append(parts[-1])
        else:
            search_keys.append(query)

        # Find matches in suffix index
        matches = []
        seen = set()
        for key in search_keys:
            for cid in self._by_suffix.get(key, []):
                if cid not in seen:
                    seen.add(cid)
                    matches.append(cid)

        # If no matches, try normalized version pattern matching
        # This allows claude-opus-4-5 to match claude-opus-4.5
        if not matches:
            for key in search_keys:
                normalized_key = _normalize_version_pattern(key)
                # Check in normalized index
                for cid in self._by_normalized.get(normalized_key, []):
                    if cid not in seen:
                        seen.add(cid)
                        matches.append(cid)
                # Also check if normalized key matches regular suffix
                # (for when source has 4-5 and query uses 4.5)
                for cid in self._by_suffix.get(normalized_key, []):
                    if cid not in seen:
                        seen.add(cid)
                        matches.append(cid)

        return matches
