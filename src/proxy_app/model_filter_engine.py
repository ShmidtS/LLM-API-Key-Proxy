# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

"""
Filter engine - core filtering logic with rule management.

Handles pattern matching, rule storage, status calculation, and
persistence via .env files. Uses caching for performance with
large model lists. No tkinter dependencies.
"""

import fnmatch
import logging
import os
import traceback
from pathlib import Path
from typing import List, Optional, Set, Tuple

from dotenv import load_dotenv, set_key, unset_key

from proxy_app.model_filter_models import (
    FilterRule,
    IGNORE_COLORS,
    ModelStatus,
    NORMAL_COLOR,
    WHITELIST_COLORS,
)

logger = logging.getLogger(__name__)


class FilterEngine:
    """
    Core filtering logic with rule management.

    Handles pattern matching, rule storage, and status calculation.
    Tracks changes for save/discard functionality.
    Uses caching for performance with large model lists.
    """

    def __init__(self):
        self.ignore_rules: List[FilterRule] = []
        self.whitelist_rules: List[FilterRule] = []
        self._ignore_color_index = 0
        self._whitelist_color_index = 0
        self._original_ignore_patterns: Set[str] = set()
        self._original_whitelist_patterns: Set[str] = set()
        self._current_provider: Optional[str] = None

        # Caching for performance
        self._status_cache: dict[str, ModelStatus] = {}
        self._available_count_cache: Optional[Tuple[int, int]] = None
        self._cache_valid: bool = False

    def _invalidate_cache(self):
        """Mark cache as stale (call when rules change)."""
        self._status_cache.clear()
        self._available_count_cache = None
        self._cache_valid = False

    def reset(self):
        """Clear all rules and reset state."""
        self.ignore_rules.clear()
        self.whitelist_rules.clear()
        self._ignore_color_index = 0
        self._whitelist_color_index = 0
        self._original_ignore_patterns.clear()
        self._original_whitelist_patterns.clear()
        self._invalidate_cache()

    def _get_next_ignore_color(self) -> str:
        """Get next color for ignore rules (cycles through palette)."""
        color = IGNORE_COLORS[self._ignore_color_index % len(IGNORE_COLORS)]
        self._ignore_color_index += 1
        return color

    def _get_next_whitelist_color(self) -> str:
        """Get next color for whitelist rules (cycles through palette)."""
        color = WHITELIST_COLORS[self._whitelist_color_index % len(WHITELIST_COLORS)]
        self._whitelist_color_index += 1
        return color

    def add_ignore_rule(self, pattern: str) -> Optional[FilterRule]:
        """Add a new ignore rule. Returns the rule if added, None if duplicate."""
        pattern = pattern.strip()
        if not pattern:
            return None

        # Check for duplicates
        for rule in self.ignore_rules:
            if rule.pattern == pattern:
                return None

        rule = FilterRule(
            pattern=pattern, color=self._get_next_ignore_color(), rule_type="ignore"
        )
        self.ignore_rules.append(rule)
        self._invalidate_cache()
        return rule

    def add_whitelist_rule(self, pattern: str) -> Optional[FilterRule]:
        """Add a new whitelist rule. Returns the rule if added, None if duplicate."""
        pattern = pattern.strip()
        if not pattern:
            return None

        # Check for duplicates
        for rule in self.whitelist_rules:
            if rule.pattern == pattern:
                return None

        rule = FilterRule(
            pattern=pattern,
            color=self._get_next_whitelist_color(),
            rule_type="whitelist",
        )
        self.whitelist_rules.append(rule)
        self._invalidate_cache()
        return rule

    def remove_ignore_rule(self, pattern: str) -> bool:
        """Remove an ignore rule by pattern. Returns True if removed."""
        for i, rule in enumerate(self.ignore_rules):
            if rule.pattern == pattern:
                self.ignore_rules.pop(i)
                self._invalidate_cache()
                return True
        return False

    def remove_whitelist_rule(self, pattern: str) -> bool:
        """Remove a whitelist rule by pattern. Returns True if removed."""
        for i, rule in enumerate(self.whitelist_rules):
            if rule.pattern == pattern:
                self.whitelist_rules.pop(i)
                self._invalidate_cache()
                return True
        return False

    def _pattern_matches(self, model_id: str, pattern: str) -> bool:
        """
        Check if a pattern matches a model ID.

        Supports full glob/fnmatch syntax:
        - Exact match: "gpt-4" matches only "gpt-4"
        - Prefix wildcard: "gpt-4*" matches "gpt-4", "gpt-4-turbo", etc.
        - Suffix wildcard: "*-preview" matches "gpt-4-preview", "o1-preview", etc.
        - Contains wildcard: "*-preview*" matches anything containing "-preview"
        - Match all: "*" matches everything
        - Single char wildcard: "gpt-?" matches "gpt-4", "gpt-5", etc.
        - Character sets: "gpt-[45]*" matches "gpt-4*", "gpt-5*"
        """
        # Extract model name without provider prefix
        if "/" in model_id:
            provider_model_name = model_id.split("/", 1)[1]
        else:
            provider_model_name = model_id

        # Use fnmatch for full glob pattern support
        # Match against both the provider model name and the full model ID
        return fnmatch.fnmatch(provider_model_name, pattern) or fnmatch.fnmatch(
            model_id, pattern
        )

    def pattern_is_covered_by(self, new_pattern: str, existing_pattern: str) -> bool:
        """
        Check if new_pattern is already covered by existing_pattern.

        A pattern A is covered by pattern B if every model that would match A
        would also match B.

        Examples:
        - "gpt-4" is covered by "gpt-4*" (prefix covers exact)
        - "gpt-4-turbo" is covered by "gpt-4*" (prefix covers longer)
        - "gpt-4*" is covered by "gpt-*" (broader prefix covers narrower)
        - Anything is covered by "*" (match-all covers everything)
        - "gpt-4" is covered by "gpt-4" (exact duplicate)
        """
        # Exact duplicate
        if new_pattern == existing_pattern:
            return True

        # Existing is wildcard-all - covers everything
        if existing_pattern == "*":
            return True

        # If existing is a prefix wildcard
        if existing_pattern.endswith("*"):
            existing_prefix = existing_pattern[:-1]

            # New is exact match - check if it starts with existing prefix
            if not new_pattern.endswith("*"):
                return new_pattern.startswith(existing_prefix)

            # New is also a prefix wildcard - check if new prefix starts with existing
            new_prefix = new_pattern[:-1]
            return new_prefix.startswith(existing_prefix)

        # Existing is exact match - only covers exact duplicate (already handled)
        return False

    def is_pattern_covered(self, new_pattern: str, rule_type: str) -> bool:
        """
        Check if a new pattern is already covered by any existing rule of the same type.
        """
        rules = self.ignore_rules if rule_type == "ignore" else self.whitelist_rules
        for rule in rules:
            if self.pattern_is_covered_by(new_pattern, rule.pattern):
                return True
        return False

    def get_covered_patterns(self, new_pattern: str, rule_type: str) -> List[str]:
        """
        Get list of existing patterns that would be covered (made redundant)
        by adding new_pattern.

        Used for smart merge: when adding a broader pattern, remove the
        narrower patterns it covers.
        """
        rules = self.ignore_rules if rule_type == "ignore" else self.whitelist_rules
        covered = []
        for rule in rules:
            if self.pattern_is_covered_by(rule.pattern, new_pattern):
                # The existing rule would be covered by the new pattern
                covered.append(rule.pattern)
        return covered

    def _compute_status(self, model_id: str) -> ModelStatus:
        """
        Compute the status of a model based on current rules (no caching).

        Priority: Whitelist > Ignore > Normal
        """
        # Check whitelist first (takes priority)
        for rule in self.whitelist_rules:
            if self._pattern_matches(model_id, rule.pattern):
                return ModelStatus(
                    model_id=model_id,
                    status="whitelisted",
                    color=rule.color,
                    affecting_rule=rule,
                )

        # Then check ignore
        for rule in self.ignore_rules:
            if self._pattern_matches(model_id, rule.pattern):
                return ModelStatus(
                    model_id=model_id,
                    status="ignored",
                    color=rule.color,
                    affecting_rule=rule,
                )

        # Default: normal
        return ModelStatus(
            model_id=model_id, status="normal", color=NORMAL_COLOR, affecting_rule=None
        )

    def get_model_status(self, model_id: str) -> ModelStatus:
        """Get status for a model (uses cache if available)."""
        if model_id in self._status_cache:
            return self._status_cache[model_id]
        return self._compute_status(model_id)

    def _rebuild_cache(self, models: List[str]):
        """Rebuild the entire status cache in one efficient pass."""
        self._status_cache.clear()

        # Reset rule counts
        for rule in self.ignore_rules + self.whitelist_rules:
            rule.affected_count = 0
            rule.affected_models = []

        available = 0
        for model_id in models:
            status = self._compute_status(model_id)
            self._status_cache[model_id] = status

            if status.affecting_rule:
                status.affecting_rule.affected_count += 1
                status.affecting_rule.affected_models.append(model_id)

            if status.status != "ignored":
                available += 1

        self._available_count_cache = (available, len(models))
        self._cache_valid = True

    def get_all_statuses(self, models: List[str]) -> List[ModelStatus]:
        """Get status for all models (rebuilds cache if invalid)."""
        if not self._cache_valid:
            self._rebuild_cache(models)
        return [self._status_cache.get(m, self._compute_status(m)) for m in models]

    def update_affected_counts(self, models: List[str]):
        """Update the affected_count and affected_models for all rules."""
        # This now just ensures cache is valid - counts are updated in _rebuild_cache
        if not self._cache_valid:
            self._rebuild_cache(models)

    def get_available_count(self, models: List[str]) -> Tuple[int, int]:
        """Returns (available_count, total_count) from cache."""
        if not self._cache_valid:
            self._rebuild_cache(models)
        return self._available_count_cache or (0, 0)

    def preview_pattern(
        self, pattern: str, rule_type: str, models: List[str]
    ) -> List[str]:
        """
        Preview which models would be affected by a pattern without adding it.
        Returns list of affected model IDs.
        """
        affected = []
        pattern = pattern.strip()
        if not pattern:
            return affected

        for model_id in models:
            if self._pattern_matches(model_id, pattern):
                affected.append(model_id)

        return affected

    def load_from_env(self, provider: str):
        """Load ignore/whitelist rules for a provider from environment."""
        self.reset()
        self._current_provider = provider
        load_dotenv(override=True)

        # Load ignore list
        ignore_key = f"IGNORE_MODELS_{provider.upper()}"
        ignore_value = os.getenv(ignore_key, "")
        if ignore_value:
            patterns = [p.strip() for p in ignore_value.split(",") if p.strip()]
            for pattern in patterns:
                self.add_ignore_rule(pattern)
            self._original_ignore_patterns = set(patterns)

        # Load whitelist
        whitelist_key = f"WHITELIST_MODELS_{provider.upper()}"
        whitelist_value = os.getenv(whitelist_key, "")
        if whitelist_value:
            patterns = [p.strip() for p in whitelist_value.split(",") if p.strip()]
            for pattern in patterns:
                self.add_whitelist_rule(pattern)
            self._original_whitelist_patterns = set(patterns)

    def save_to_env(self, provider: str) -> bool:
        """
        Save current rules to .env file.
        Returns True if successful.
        """
        env_path = Path.cwd() / ".env"

        try:
            ignore_key = f"IGNORE_MODELS_{provider.upper()}"
            whitelist_key = f"WHITELIST_MODELS_{provider.upper()}"

            # Save ignore patterns
            ignore_patterns = [rule.pattern for rule in self.ignore_rules]
            if ignore_patterns:
                set_key(str(env_path), ignore_key, ",".join(ignore_patterns))
            else:
                # Remove the key if no patterns
                unset_key(str(env_path), ignore_key)

            # Save whitelist patterns
            whitelist_patterns = [rule.pattern for rule in self.whitelist_rules]
            if whitelist_patterns:
                set_key(str(env_path), whitelist_key, ",".join(whitelist_patterns))
            else:
                unset_key(str(env_path), whitelist_key)

            # Update original state
            self._original_ignore_patterns = set(ignore_patterns)
            self._original_whitelist_patterns = set(whitelist_patterns)

            return True
        except Exception as e:
            logger.error("Error saving to .env: %s", e)
            traceback.print_exc()
            return False

    def has_unsaved_changes(self) -> bool:
        """Check if current rules differ from saved state."""
        current_ignore = set(rule.pattern for rule in self.ignore_rules)
        current_whitelist = set(rule.pattern for rule in self.whitelist_rules)

        return (
            current_ignore != self._original_ignore_patterns
            or current_whitelist != self._original_whitelist_patterns
        )

    def discard_changes(self):
        """Reload rules from environment, discarding unsaved changes."""
        if self._current_provider:
            self.load_from_env(self._current_provider)
