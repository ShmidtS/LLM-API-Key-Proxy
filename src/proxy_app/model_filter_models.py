# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

"""
Data models and shared constants for the Model Filter GUI.

Contains filter rule dataclasses, model status, cross-platform utilities,
and the color/font/window configuration used across all filter modules.
Zero tkinter dependencies.
"""

import platform
from dataclasses import dataclass, field
from typing import List, Optional


# ════════════════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

# Window settings
WINDOW_TITLE = "Model Filter Configuration"
WINDOW_DEFAULT_SIZE = "1000x750"
WINDOW_MIN_WIDTH = 600
WINDOW_MIN_HEIGHT = 400

# Color scheme (dark mode)
BG_PRIMARY = "#1a1a2e"  # Main background
BG_SECONDARY = "#16213e"  # Card/panel background
BG_TERTIARY = "#0f0f1a"  # Input fields, lists
BG_HOVER = "#1f2b47"  # Hover state
BORDER_COLOR = "#2a2a4a"  # Subtle borders
TEXT_PRIMARY = "#e8e8e8"  # Main text
TEXT_SECONDARY = "#a0a0a0"  # Muted text
TEXT_MUTED = "#666680"  # Very muted text
ACCENT_BLUE = "#4a9eff"  # Primary accent
ACCENT_GREEN = "#2ecc71"  # Success/normal
ACCENT_RED = "#e74c3c"  # Danger/ignore
ACCENT_YELLOW = "#f1c40f"  # Warning

# Status colors
NORMAL_COLOR = "#2ecc71"  # Green - models not affected by any rule
HIGHLIGHT_BG = "#2a3a5a"  # Background for highlighted items

# Ignore rules - warm color progression (reds/oranges)
IGNORE_COLORS = [
    "#e74c3c",  # Bright red
    "#c0392b",  # Dark red
    "#e67e22",  # Orange
    "#d35400",  # Dark orange
    "#f39c12",  # Gold
    "#e91e63",  # Pink
    "#ff5722",  # Deep orange
    "#f44336",  # Material red
    "#ff6b6b",  # Coral
    "#ff8a65",  # Light deep orange
]

# Whitelist rules - cool color progression (blues/teals)
WHITELIST_COLORS = [
    "#3498db",  # Blue
    "#2980b9",  # Dark blue
    "#1abc9c",  # Teal
    "#16a085",  # Dark teal
    "#9b59b6",  # Purple
    "#8e44ad",  # Dark purple
    "#00bcd4",  # Cyan
    "#2196f3",  # Material blue
    "#64b5f6",  # Light blue
    "#4dd0e1",  # Light cyan
]

# Font configuration
FONT_FAMILY = "Segoe UI"
FONT_SIZE_SMALL = 11
FONT_SIZE_NORMAL = 12
FONT_SIZE_LARGE = 14
FONT_SIZE_HEADER = 20


# ════════════════════════════════════════════════════════════════════════════════
# CROSS-PLATFORM UTILITIES
# ════════════════════════════════════════════════════════════════════════════════


def get_scroll_delta(event) -> int:
    """
    Calculate scroll delta in a cross-platform manner.

    On Windows, event.delta is typically +/-120 per notch.
    On macOS, event.delta is typically +/-1 per scroll event.
    On Linux/X11, behavior varies but is usually similar to macOS.

    Returns a normalized scroll direction value (typically +/-1).
    """
    system = platform.system()
    if system == "Darwin":  # macOS
        return -event.delta
    elif system == "Linux":
        # Linux with X11 typically uses +/-1 like macOS
        # but some configurations may use larger values
        if abs(event.delta) >= 120:
            return -1 * (event.delta // 120)
        return -event.delta
    else:  # Windows
        return -1 * (event.delta // 120)


# ════════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ════════════════════════════════════════════════════════════════════════════════


@dataclass
class FilterRule:
    """Represents a single filter rule (ignore or whitelist pattern)."""

    pattern: str
    color: str
    rule_type: str  # 'ignore' or 'whitelist'
    affected_count: int = 0
    affected_models: List[str] = field(default_factory=list)

    def __hash__(self):
        return hash((self.pattern, self.rule_type))

    def __eq__(self, other):
        if not isinstance(other, FilterRule):
            return False
        return self.pattern == other.pattern and self.rule_type == other.rule_type


@dataclass
class ModelStatus:
    """Status information for a single model."""

    model_id: str
    status: str  # 'normal', 'ignored', 'whitelisted'
    color: str
    affecting_rule: Optional[FilterRule] = None

    @property
    def display_name(self) -> str:
        """Get the model name without provider prefix for display."""
        if "/" in self.model_id:
            return self.model_id.split("/", 1)[1]
        return self.model_id

    @property
    def provider(self) -> str:
        """Extract provider from model ID."""
        if "/" in self.model_id:
            return self.model_id.split("/")[0]
        return ""
