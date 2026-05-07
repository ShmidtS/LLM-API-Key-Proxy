# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

"""
CustomTkinter widget components for the Model Filter GUI.

Contains ToolTip, VirtualModelList, VirtualSyncModelLists, and VirtualRuleList.
These are ctk widget classes used by the dialogs and main GUI.
"""

import customtkinter as ctk
from typing import Any, Callable, Dict, List, Optional, Set

from proxy_app.model_filter_models import (
    ACCENT_BLUE,
    ACCENT_RED,
    BG_HOVER,
    BG_SECONDARY,
    BG_TERTIARY,
    BORDER_COLOR,
    FONT_FAMILY,
    FONT_SIZE_LARGE,
    FONT_SIZE_NORMAL,
    FONT_SIZE_SMALL,
    FilterRule,
    HIGHLIGHT_BG,
    ModelStatus,
    NORMAL_COLOR,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    get_scroll_delta,
)


# ════════════════════════════════════════════════════════════════════════════════
# TOOLTIP
# ════════════════════════════════════════════════════════════════════════════════


class ToolTip:
    """Simple tooltip implementation for CustomTkinter widgets."""

    def __init__(self, widget, text: str, delay: int = 500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tooltip_window = None
        self.after_id = None

        widget.bind("<Enter>", self._schedule_show)
        widget.bind("<Leave>", self._hide)
        widget.bind("<Button>", self._hide)

    def _schedule_show(self, event=None):
        self._hide()
        self.after_id = self.widget.after(self.delay, self._show)

    def _show(self):
        if self.tooltip_window:
            return

        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        self.tooltip_window = tw = ctk.CTkToplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.configure(fg_color=BG_SECONDARY)

        # Add border effect
        frame = ctk.CTkFrame(
            tw,
            fg_color=BG_SECONDARY,
            border_width=1,
            border_color=BORDER_COLOR,
            corner_radius=6,
        )
        frame.pack(fill="both", expand=True)

        label = ctk.CTkLabel(
            frame,
            text=self.text,
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_SECONDARY,
            padx=10,
            pady=5,
        )
        label.pack()

        # Ensure tooltip is on top
        tw.lift()

    def _hide(self, event=None):
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

    def update_text(self, text: str):
        """Update tooltip text."""
        self.text = text


# ════════════════════════════════════════════════════════════════════════════════
# VIRTUAL MODEL LIST (Canvas-based for performance)
# ════════════════════════════════════════════════════════════════════════════════

# Constants for virtual list
ITEM_HEIGHT = 24  # Height of each row in pixels
INDICATOR_WIDTH = 18  # Width of status indicator


class VirtualModelList:
    """
    High-performance virtual list that only renders visible items.

    Uses a raw tkinter Canvas to draw text directly rather than
    creating individual widgets per row. This reduces widget count
    from O(n) to O(visible_rows).
    """

    def __init__(
        self,
        parent,
        show_status_indicator: bool = False,
        on_click: Optional[Callable[[str], None]] = None,
        on_right_click: Optional[Callable[[str, Any], None]] = None,
    ):
        self.parent = parent
        self.show_status_indicator = show_status_indicator
        self.on_click = on_click
        self.on_right_click = on_right_click

        # Data
        self.models: List[str] = []
        self.statuses: Dict[str, ModelStatus] = {}
        self.filtered_models: List[str] = []  # Models after search filter
        self.search_query: str = ""
        self.highlighted_models: Set[str] = set()

        # UI state
        self._hover_index: Optional[int] = None

        # Create container frame
        self.frame = ctk.CTkFrame(parent, fg_color=BG_TERTIARY, corner_radius=6)

        # Create canvas (use raw tk.Canvas for performance)
        import tkinter as tk

        self.canvas = tk.Canvas(
            self.frame,
            bg=BG_TERTIARY,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(side="left", fill="both", expand=True)

        # Scrollbar
        self.scrollbar = ctk.CTkScrollbar(self.frame, command=self._on_scroll)
        self.scrollbar.pack(side="right", fill="y")

        # Link canvas to scrollbar
        self.canvas.configure(yscrollcommand=self._on_canvas_scroll)

        # Bind events
        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<Motion>", self._on_mouse_motion)
        self.canvas.bind("<Leave>", self._on_mouse_leave)

    def grid(self, **kwargs):
        """Grid the container frame."""
        self.frame.grid(**kwargs)

    def grid_forget(self):
        """Hide the container frame."""
        self.frame.grid_forget()

    def pack(self, **kwargs):
        """Pack the container frame."""
        self.frame.pack(**kwargs)

    def pack_forget(self):
        """Hide the container frame."""
        self.frame.pack_forget()

    def set_models(self, models: List[str], statuses: Dict[str, ModelStatus]):
        """Set the model list and statuses."""
        self.models = models
        self.statuses = statuses
        self._apply_filter()
        self._update_scroll_region()
        self._render()

    def update_statuses(self, statuses: Dict[str, ModelStatus]):
        """Update just the statuses (no model list change)."""
        self.statuses = statuses
        self._render()

    def filter_by_search(self, query: str):
        """Filter models by search query."""
        self.search_query = query.lower().strip()
        self._apply_filter()
        self._update_scroll_region()
        self._render()

    def _apply_filter(self):
        """Apply current search filter to models."""
        if not self.search_query:
            self.filtered_models = list(self.models)
        else:
            self.filtered_models = [
                m for m in self.models if self.search_query in m.lower()
            ]

    def highlight_models(self, model_ids: Set[str]):
        """Set which models should be highlighted."""
        self.highlighted_models = model_ids
        self._render()

    def clear_highlights(self):
        """Clear all highlights."""
        self.highlighted_models.clear()
        self._render()

    def scroll_to_model(self, model_id: str):
        """Scroll to make a model visible."""
        if model_id not in self.filtered_models:
            return

        index = self.filtered_models.index(model_id)
        total_height = len(self.filtered_models) * ITEM_HEIGHT
        canvas_height = self.canvas.winfo_height()

        if total_height <= canvas_height:
            return

        # Calculate position to center the item
        item_y = index * ITEM_HEIGHT
        target_scroll = (item_y - canvas_height / 2 + ITEM_HEIGHT / 2) / total_height
        target_scroll = max(0, min(1, target_scroll))

        self.canvas.yview_moveto(target_scroll)
        self._render()

    def _update_scroll_region(self):
        """Update the scrollable region based on item count."""
        total_height = max(len(self.filtered_models) * ITEM_HEIGHT, 1)
        self.canvas.configure(scrollregion=(0, 0, 100, total_height))

    def _on_scroll(self, *args):
        """Handle scrollbar command."""
        self.canvas.yview(*args)
        self._render()

    def _on_canvas_scroll(self, first: float, last: float):
        """Handle canvas scroll update - just update scrollbar."""
        self.scrollbar.set(first, last)

    def _on_configure(self, event=None):
        """Handle canvas resize."""
        self._update_scroll_region()
        self._render()

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        delta = get_scroll_delta(event)
        self.canvas.yview_scroll(delta, "units")
        self._render()
        return "break"

    def _get_index_at_y(self, y: int) -> Optional[int]:
        """Get the model index at a y coordinate."""
        if not self.filtered_models:
            return None

        # Convert window y coordinate to canvas (scrollregion) coordinate
        canvas_y = self.canvas.canvasy(y)

        # Calculate index from absolute position
        index = int(canvas_y // ITEM_HEIGHT)

        if 0 <= index < len(self.filtered_models):
            return index
        return None

    def _on_left_click(self, event):
        """Handle left click."""
        index = self._get_index_at_y(event.y)
        if index is not None and self.on_click:
            model_id = self.filtered_models[index]
            self.on_click(model_id)

    def _on_right_click(self, event):
        """Handle right click."""
        index = self._get_index_at_y(event.y)
        if index is not None and self.on_right_click:
            model_id = self.filtered_models[index]
            self.on_right_click(model_id, event)

    def _on_mouse_motion(self, event):
        """Handle mouse motion for hover effect."""
        new_hover = self._get_index_at_y(event.y)
        if new_hover != self._hover_index:
            self._hover_index = new_hover
            self._render()

    def _on_mouse_leave(self, event):
        """Handle mouse leaving canvas."""
        if self._hover_index is not None:
            self._hover_index = None
            self._render()

    def _render(self):
        """Render only the visible items."""
        self.canvas.delete("all")

        if not self.filtered_models:
            # Show empty state
            canvas_height = self.canvas.winfo_height()
            self.canvas.create_text(
                self.canvas.winfo_width() // 2,
                canvas_height // 2,
                text="No models",
                fill=TEXT_MUTED,
                font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            )
            return

        canvas_height = self.canvas.winfo_height()
        canvas_width = self.canvas.winfo_width()
        total_height = len(self.filtered_models) * ITEM_HEIGHT

        # Calculate visible range based on scroll position
        scroll_position = self.canvas.yview()[0]
        scroll_offset = scroll_position * total_height
        first_visible = int(scroll_offset // ITEM_HEIGHT)
        visible_count = int(canvas_height // ITEM_HEIGHT) + 2  # +2 for partial rows

        # Clamp to valid range
        first_visible = max(0, first_visible)
        last_visible = min(len(self.filtered_models), first_visible + visible_count)

        # Draw visible items at ABSOLUTE positions
        # The canvas scrollregion + yview handles showing the correct portion
        for i in range(first_visible, last_visible):
            model_id = self.filtered_models[i]
            status = self.statuses.get(
                model_id,
                ModelStatus(model_id=model_id, status="normal", color=NORMAL_COLOR),
            )

            # Absolute y position in the virtual list
            y = i * ITEM_HEIGHT
            y_center = y + ITEM_HEIGHT // 2

            # Background for hover/highlight
            is_highlighted = model_id in self.highlighted_models
            is_hovered = i == self._hover_index

            if is_highlighted:
                self.canvas.create_rectangle(
                    0, y, canvas_width, y + ITEM_HEIGHT, fill=HIGHLIGHT_BG, outline=""
                )
            elif is_hovered:
                self.canvas.create_rectangle(
                    0, y, canvas_width, y + ITEM_HEIGHT, fill=BG_HOVER, outline=""
                )

            # Status indicator (for right list)
            x_offset = 8
            if self.show_status_indicator:
                indicator_text = {
                    "normal": "●",
                    "ignored": "✗",
                    "whitelisted": "★",
                }.get(status.status, "●")
                self.canvas.create_text(
                    x_offset + INDICATOR_WIDTH // 2,
                    y_center,
                    text=indicator_text,
                    fill=status.color,
                    font=(FONT_FAMILY, FONT_SIZE_SMALL),
                )
                x_offset += INDICATOR_WIDTH

            # Model name
            text_color = status.color if self.show_status_indicator else TEXT_PRIMARY
            display_name = status.display_name

            self.canvas.create_text(
                x_offset,
                y_center,
                text=display_name,
                fill=text_color,
                font=(FONT_FAMILY, FONT_SIZE_NORMAL),
                anchor="w",
            )

    def get_scroll_position(self) -> float:
        """Get current scroll position (0-1) directly from canvas."""
        return self.canvas.yview()[0]

    def set_scroll_position(self, pos: float, render: bool = True):
        """Set scroll position (0-1) and optionally render."""
        self.canvas.yview_moveto(pos)
        if render:
            self._render()


# ════════════════════════════════════════════════════════════════════════════════
# VIRTUAL RULE LIST (Canvas-based for performance)
# ════════════════════════════════════════════════════════════════════════════════

# Constants for virtual rule list
RULE_ITEM_HEIGHT = 32  # Height of each rule row
RULE_DELETE_WIDTH = 24  # Width of delete button area
RULE_COUNT_WIDTH = 40  # Width of count area
RULE_PADDING = 8  # Horizontal padding

class VirtualRuleList:
    """
    High-performance virtual list for filter rules.

    Uses a raw tkinter Canvas to draw rules directly rather than
    creating individual widgets per row.
    """

    def __init__(
        self,
        parent,
        rule_type: str,  # 'ignore' or 'whitelist'
        on_rule_click: Callable[[FilterRule], None],
        on_rule_delete: Callable[[str], None],
    ):
        self.parent = parent
        self.rule_type = rule_type
        self.on_rule_click = on_rule_click
        self.on_rule_delete = on_rule_delete

        # Data
        self.rules: List[FilterRule] = []
        self.highlighted_pattern: Optional[str] = None

        # UI state
        self._hover_index: Optional[int] = None
        self._hover_delete: bool = False  # True if hovering over delete button

        # Tooltip state
        self._tooltip_window = None
        self._tooltip_after_id = None
        self._tooltip_rule_index: Optional[int] = None

        # Create container frame
        self.frame = ctk.CTkFrame(parent, fg_color="transparent")

        # Create canvas
        import tkinter as tk

        self.canvas = tk.Canvas(
            self.frame,
            bg=BG_SECONDARY,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(side="left", fill="both", expand=True)

        # Scrollbar
        self.scrollbar = ctk.CTkScrollbar(self.frame, command=self._on_scroll)
        self.scrollbar.pack(side="right", fill="y")

        # Link canvas to scrollbar
        self.canvas.configure(yscrollcommand=self._on_canvas_scroll)

        # Bind events
        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Motion>", self._on_mouse_motion)
        self.canvas.bind("<Leave>", self._on_mouse_leave)

    def pack(self, **kwargs):
        """Pack the container frame."""
        self.frame.pack(**kwargs)

    def set_rules(self, rules: List[FilterRule]):
        """Set the rules to display."""
        self.rules = rules
        self._update_scroll_region()
        self._render()

    def add_rule(self, rule: FilterRule):
        """Add a rule to the list."""
        # Check for duplicates
        if any(r.pattern == rule.pattern for r in self.rules):
            return
        self.rules.append(rule)
        self._update_scroll_region()
        self._render()

    def remove_rule(self, pattern: str):
        """Remove a rule by pattern."""
        self.rules = [r for r in self.rules if r.pattern != pattern]
        self._update_scroll_region()
        self._render()

    def update_rule_counts(self, rules: List[FilterRule]):
        """Update affected counts from new rule data."""
        rule_map = {r.pattern: r for r in rules}
        for rule in self.rules:
            if rule.pattern in rule_map:
                rule.affected_count = rule_map[rule.pattern].affected_count
                rule.affected_models = rule_map[rule.pattern].affected_models
        self._render()

    def highlight_rule(self, pattern: Optional[str]):
        """Highlight a specific rule."""
        self.highlighted_pattern = pattern
        if pattern:
            self._scroll_to_rule(pattern)
        self._render()

    def clear_highlights(self):
        """Clear all highlights."""
        self.highlighted_pattern = None
        self._render()

    def clear_all(self):
        """Remove all rules."""
        self.rules = []
        self._update_scroll_region()
        self._render()

    def _scroll_to_rule(self, pattern: str):
        """Scroll to make a rule visible."""
        for i, rule in enumerate(self.rules):
            if rule.pattern == pattern:
                total_height = len(self.rules) * RULE_ITEM_HEIGHT
                canvas_height = self.canvas.winfo_height()

                if total_height <= canvas_height:
                    return

                item_y = i * RULE_ITEM_HEIGHT
                target_scroll = (
                    item_y - canvas_height / 2 + RULE_ITEM_HEIGHT / 2
                ) / total_height
                target_scroll = max(0, min(1, target_scroll))

                self.canvas.yview_moveto(target_scroll)
                self._render()
                return

    def _update_scroll_region(self):
        """Update the scrollable region."""
        total_height = max(len(self.rules) * RULE_ITEM_HEIGHT, 1)
        self.canvas.configure(scrollregion=(0, 0, 100, total_height))

    def _on_scroll(self, *args):
        """Handle scrollbar command."""
        self.canvas.yview(*args)
        self._render()

    def _on_canvas_scroll(self, first: float, last: float):
        """Handle canvas scroll update."""
        self.scrollbar.set(first, last)

    def _on_configure(self, event=None):
        """Handle canvas resize."""
        self._update_scroll_region()
        self._render()

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        delta = get_scroll_delta(event)
        self.canvas.yview_scroll(delta, "units")
        self._render()
        return "break"

    def _get_index_at_y(self, y: int) -> Optional[int]:
        """Get the rule index at a y coordinate."""
        if not self.rules:
            return None

        canvas_y = self.canvas.canvasy(y)
        index = int(canvas_y // RULE_ITEM_HEIGHT)

        if 0 <= index < len(self.rules):
            return index
        return None

    def _is_over_delete(self, x: int) -> bool:
        """Check if x coordinate is over the delete button."""
        canvas_width = self.canvas.winfo_width()
        delete_start = canvas_width - RULE_DELETE_WIDTH - RULE_PADDING
        return x >= delete_start

    def _on_left_click(self, event):
        """Handle left click."""
        index = self._get_index_at_y(event.y)
        if index is None:
            return

        rule = self.rules[index]

        if self._is_over_delete(event.x):
            # Click on delete button
            self.on_rule_delete(rule.pattern)
        else:
            # Click on rule
            self.on_rule_click(rule)

    def _on_mouse_motion(self, event):
        """Handle mouse motion for hover effect."""
        new_hover = self._get_index_at_y(event.y)
        new_hover_delete = (
            self._is_over_delete(event.x) if new_hover is not None else False
        )

        if new_hover != self._hover_index or new_hover_delete != self._hover_delete:
            self._hover_index = new_hover
            self._hover_delete = new_hover_delete
            self._render()

        # Handle tooltip
        if new_hover != self._tooltip_rule_index:
            self._hide_tooltip()
            if new_hover is not None and not new_hover_delete:
                self._schedule_tooltip(new_hover)

    def _on_mouse_leave(self, event):
        """Handle mouse leaving canvas."""
        if self._hover_index is not None:
            self._hover_index = None
            self._hover_delete = False
            self._render()
        self._hide_tooltip()

    def _schedule_tooltip(self, index: int):
        """Schedule tooltip to appear."""
        self._tooltip_rule_index = index
        self._tooltip_after_id = self.canvas.after(
            500, lambda: self._show_tooltip(index)
        )

    def _show_tooltip(self, index: int):
        """Show tooltip for a rule."""
        if index != self._tooltip_rule_index or index >= len(self.rules):
            return

        rule = self.rules[index]

        # Build tooltip text
        if rule.affected_models:
            if len(rule.affected_models) <= 5:
                models_text = "\n".join(rule.affected_models)
            else:
                models_text = "\n".join(rule.affected_models[:5])
                models_text += f"\n... and {len(rule.affected_models) - 5} more"
            text = f"Matches:\n{models_text}"
        else:
            text = "No models match this pattern"

        # Position tooltip
        x = self.canvas.winfo_rootx() + 20
        y = (
            self.canvas.winfo_rooty()
            + (index + 1) * RULE_ITEM_HEIGHT
            - int(self.canvas.canvasy(0))
        )

        # Create tooltip window
        self._tooltip_window = tw = ctk.CTkToplevel(self.canvas)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.configure(fg_color=BG_SECONDARY)

        frame = ctk.CTkFrame(
            tw,
            fg_color=BG_SECONDARY,
            border_width=1,
            border_color=BORDER_COLOR,
            corner_radius=6,
        )
        frame.pack(fill="both", expand=True)

        label = ctk.CTkLabel(
            frame,
            text=text,
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_SECONDARY,
            padx=10,
            pady=5,
        )
        label.pack()
        tw.lift()

    def _hide_tooltip(self):
        """Hide the tooltip."""
        if self._tooltip_after_id:
            self.canvas.after_cancel(self._tooltip_after_id)
            self._tooltip_after_id = None
        if self._tooltip_window:
            self._tooltip_window.destroy()
            self._tooltip_window = None
        self._tooltip_rule_index = None

    def _render(self):
        """Render only the visible rules."""
        self.canvas.delete("all")

        if not self.rules:
            # Show empty state
            canvas_height = self.canvas.winfo_height()
            self.canvas.create_text(
                self.canvas.winfo_width() // 2,
                canvas_height // 2,
                text="No rules configured\nAdd patterns below",
                fill=TEXT_MUTED,
                font=(FONT_FAMILY, FONT_SIZE_SMALL),
                justify="center",
            )
            return

        canvas_height = self.canvas.winfo_height()
        canvas_width = self.canvas.winfo_width()
        total_height = len(self.rules) * RULE_ITEM_HEIGHT

        # Calculate visible range
        scroll_position = self.canvas.yview()[0]
        scroll_offset = scroll_position * total_height
        first_visible = int(scroll_offset // RULE_ITEM_HEIGHT)
        visible_count = int(canvas_height // RULE_ITEM_HEIGHT) + 2

        first_visible = max(0, first_visible)
        last_visible = min(len(self.rules), first_visible + visible_count)

        # Draw visible rules
        for i in range(first_visible, last_visible):
            rule = self.rules[i]

            # Absolute y position
            y = i * RULE_ITEM_HEIGHT
            y_center = y + RULE_ITEM_HEIGHT // 2

            # Background
            is_highlighted = rule.pattern == self.highlighted_pattern
            is_hovered = i == self._hover_index

            if is_highlighted:
                # Highlighted - use rule color for border effect
                self.canvas.create_rectangle(
                    2,
                    y + 2,
                    canvas_width - 2,
                    y + RULE_ITEM_HEIGHT - 2,
                    fill=BG_TERTIARY,
                    outline=rule.color,
                    width=2,
                )
            elif is_hovered:
                self.canvas.create_rectangle(
                    2,
                    y + 2,
                    canvas_width - 2,
                    y + RULE_ITEM_HEIGHT - 2,
                    fill=BG_HOVER,
                    outline=BORDER_COLOR,
                    width=1,
                )
            else:
                self.canvas.create_rectangle(
                    2,
                    y + 2,
                    canvas_width - 2,
                    y + RULE_ITEM_HEIGHT - 2,
                    fill=BG_TERTIARY,
                    outline=BORDER_COLOR,
                    width=1,
                )

            # Pattern text (colored)
            self.canvas.create_text(
                RULE_PADDING + 4,
                y_center,
                text=rule.pattern,
                fill=rule.color,
                font=(FONT_FAMILY, FONT_SIZE_NORMAL),
                anchor="w",
            )

            # Count text
            count_x = canvas_width - RULE_DELETE_WIDTH - RULE_COUNT_WIDTH - RULE_PADDING
            self.canvas.create_text(
                count_x,
                y_center,
                text=f"({rule.affected_count})",
                fill=TEXT_MUTED,
                font=(FONT_FAMILY, FONT_SIZE_SMALL),
                anchor="w",
            )

            # Delete button
            delete_x = (
                canvas_width - RULE_DELETE_WIDTH - RULE_PADDING + RULE_DELETE_WIDTH // 2
            )
            delete_color = (
                ACCENT_RED if (is_hovered and self._hover_delete) else TEXT_MUTED
            )
            self.canvas.create_text(
                delete_x,
                y_center,
                text="×",
                fill=delete_color,
                font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            )


# ════════════════════════════════════════════════════════════════════════════════
# SYNCHRONIZED MODEL LISTS
# ════════════════════════════════════════════════════════════════════════════════


class VirtualSyncModelLists(ctk.CTkFrame):
    """
    Container with two synchronized virtual model lists.

    Left list: All fetched models (plain display)
    Right list: Same models with colored status indicators

    Both lists scroll together.
    """

    def __init__(
        self,
        master,
        on_model_click: Callable[[str], None],
        on_model_right_click: Callable[[str, Any], None],
    ):
        super().__init__(master, fg_color="transparent")

        self.on_model_click = on_model_click
        self.on_model_right_click = on_model_right_click

        self.models: List[str] = []
        self.statuses: Dict[str, ModelStatus] = {}
        self._syncing_scroll = False

        self._create_content()

    def _create_content(self):
        """Build the dual list layout."""
        # Don't let content dictate size - let parent grid control height
        self.grid_propagate(False)

        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Left header frame
        left_header_frame = ctk.CTkFrame(self, fg_color="transparent")
        left_header_frame.grid(row=0, column=0, sticky="ew", padx=8, pady=(0, 5))

        left_header = ctk.CTkLabel(
            left_header_frame,
            text="All Fetched Models",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            text_color=TEXT_PRIMARY,
        )
        left_header.pack(side="left")

        self.left_count_label = ctk.CTkLabel(
            left_header_frame,
            text="(0)",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_MUTED,
        )
        self.left_count_label.pack(side="left", padx=(5, 0))

        # Copy button for all models
        self.left_copy_btn = ctk.CTkButton(
            left_header_frame,
            text="Copy",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color=BG_SECONDARY,
            hover_color=BG_HOVER,
            border_width=1,
            border_color=BORDER_COLOR,
            width=50,
            height=20,
            command=self._copy_all_models,
        )
        self.left_copy_btn.pack(side="right")
        ToolTip(self.left_copy_btn, "Copy all model names (comma-separated)")

        # Right header frame
        right_header_frame = ctk.CTkFrame(self, fg_color="transparent")
        right_header_frame.grid(row=0, column=1, sticky="ew", padx=8, pady=(0, 5))

        right_header = ctk.CTkLabel(
            right_header_frame,
            text="Filtered Status",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            text_color=TEXT_PRIMARY,
        )
        right_header.pack(side="left")

        self.right_count_label = ctk.CTkLabel(
            right_header_frame,
            text="",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_MUTED,
        )
        self.right_count_label.pack(side="left", padx=(5, 0))

        # Copy button for filtered models
        self.right_copy_btn = ctk.CTkButton(
            right_header_frame,
            text="Copy",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color=BG_SECONDARY,
            hover_color=BG_HOVER,
            border_width=1,
            border_color=BORDER_COLOR,
            width=50,
            height=20,
            command=self._copy_filtered_models,
        )
        self.right_copy_btn.pack(side="right")
        ToolTip(self.right_copy_btn, "Copy available model names (comma-separated)")

        # Create virtual lists
        self.left_list = VirtualModelList(
            self,
            show_status_indicator=False,
            on_click=self.on_model_click,
            on_right_click=self.on_model_right_click,
        )
        self.left_list.grid(row=1, column=0, sticky="nsew", padx=(0, 5))

        self.right_list = VirtualModelList(
            self,
            show_status_indicator=True,
            on_click=self.on_model_click,
            on_right_click=self.on_model_right_click,
        )
        self.right_list.grid(row=1, column=1, sticky="nsew", padx=(5, 0))

        # Synchronize scrolling
        self._setup_scroll_sync()

        # Loading state
        self.loading_frame = ctk.CTkFrame(self, fg_color=BG_TERTIARY, corner_radius=6)
        self.loading_label = ctk.CTkLabel(
            self.loading_frame,
            text="Loading...",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            text_color=TEXT_MUTED,
        )
        self.loading_label.pack(expand=True)

        # Error state
        self.error_frame = ctk.CTkFrame(self, fg_color=BG_TERTIARY, corner_radius=6)
        self.error_label = ctk.CTkLabel(
            self.error_frame,
            text="",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            text_color=ACCENT_RED,
        )
        self.error_label.pack(expand=True, pady=20)

        self.retry_btn = ctk.CTkButton(
            self.error_frame,
            text="Retry",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg_color="#4a9eff",
            hover_color="#3a8aee",
            width=100,
        )
        self.retry_btn.pack()

    def _setup_scroll_sync(self):
        """Setup synchronized scrolling between both lists."""
        # Override the scroll handlers to sync both lists
        original_left_scroll = self.left_list._on_scroll
        original_right_scroll = self.right_list._on_scroll
        original_left_wheel = self.left_list._on_mousewheel
        original_right_wheel = self.right_list._on_mousewheel

        def sync_scroll_left(*args):
            if self._syncing_scroll:
                return
            self._syncing_scroll = True
            original_left_scroll(*args)
            # Sync to right - get position after scroll completed
            pos = self.left_list.get_scroll_position()
            self.right_list.set_scroll_position(pos)
            self._syncing_scroll = False

        def sync_scroll_right(*args):
            if self._syncing_scroll:
                return
            self._syncing_scroll = True
            original_right_scroll(*args)
            # Sync to left - get position after scroll completed
            pos = self.right_list.get_scroll_position()
            self.left_list.set_scroll_position(pos)
            self._syncing_scroll = False

        def sync_wheel_left(event):
            if self._syncing_scroll:
                return "break"
            self._syncing_scroll = True
            original_left_wheel(event)
            # Sync to right - get position after scroll completed
            pos = self.left_list.get_scroll_position()
            self.right_list.set_scroll_position(pos)
            self._syncing_scroll = False
            return "break"

        def sync_wheel_right(event):
            if self._syncing_scroll:
                return "break"
            self._syncing_scroll = True
            original_right_wheel(event)
            # Sync to left - get position after scroll completed
            pos = self.right_list.get_scroll_position()
            self.left_list.set_scroll_position(pos)
            self._syncing_scroll = False
            return "break"

        # Override the method references
        self.left_list._on_scroll = sync_scroll_left
        self.right_list._on_scroll = sync_scroll_right

        # IMPORTANT: Reconfigure scrollbars to use the new sync handlers
        # The scrollbars were created with command=_on_scroll before we overrode it
        self.left_list.scrollbar.configure(command=sync_scroll_left)
        self.right_list.scrollbar.configure(command=sync_scroll_right)

        # Rebind mouse wheel events
        self.left_list.canvas.bind("<MouseWheel>", sync_wheel_left)
        self.right_list.canvas.bind("<MouseWheel>", sync_wheel_right)

    def show_loading(self, provider: str):
        """Show loading state."""
        self.loading_label.configure(text=f"Fetching models from {provider}...")
        self.loading_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        self.error_frame.grid_forget()

    def show_error(self, message: str, on_retry: Callable):
        """Show error state."""
        self.error_label.configure(text=f"❌ {message}")
        self.retry_btn.configure(command=on_retry)
        self.error_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        self.loading_frame.grid_forget()

    def hide_overlays(self):
        """Hide loading and error overlays."""
        self.loading_frame.grid_forget()
        self.error_frame.grid_forget()

    def set_models(self, models: List[str], statuses: List[ModelStatus]):
        """Set the models to display."""
        self.models = models
        self.statuses = {s.model_id: s for s in statuses}

        self.left_list.set_models(models, self.statuses)
        self.right_list.set_models(models, self.statuses)

        self._update_counts()
        self.hide_overlays()

    def update_statuses(self, statuses: List[ModelStatus]):
        """Update status display for all models."""
        self.statuses = {s.model_id: s for s in statuses}
        self.left_list.update_statuses(self.statuses)
        self.right_list.update_statuses(self.statuses)
        self._update_counts()

    def _update_counts(self):
        """Update the count labels."""
        total = len(self.models)
        available = sum(1 for s in self.statuses.values() if s.status != "ignored")

        self.left_count_label.configure(text=f"({total})")
        self.right_count_label.configure(text=f"{available} available")

    def filter_by_search(self, query: str):
        """Filter models by search query."""
        self.left_list.filter_by_search(query)
        self.right_list.filter_by_search(query)

    def highlight_models_by_rule(self, rule: FilterRule):
        """Highlight all models affected by a rule."""
        model_set = set(rule.affected_models)
        self.left_list.highlight_models(model_set)
        self.right_list.highlight_models(model_set)

        # Scroll to first match
        if rule.affected_models:
            self.left_list.scroll_to_model(rule.affected_models[0])
            # Sync right list scroll
            pos = self.left_list.get_scroll_position()
            self.right_list.set_scroll_position(pos)

    def highlight_model(self, model_id: str):
        """Highlight a specific model."""
        model_set = {model_id}
        self.left_list.highlight_models(model_set)
        self.right_list.highlight_models(model_set)

    def clear_highlights(self):
        """Clear all model highlights."""
        self.left_list.clear_highlights()
        self.right_list.clear_highlights()

    def scroll_to_affected(self, affected_models: List[str]):
        """Scroll to first affected model."""
        if affected_models:
            self.left_list.scroll_to_model(affected_models[0])
            pos = self.left_list.get_scroll_position()
            self.right_list.set_scroll_position(pos)

    def _get_model_display_name(self, model_id: str) -> str:
        """Get model name without provider prefix."""
        if "/" in model_id:
            return model_id.split("/", 1)[1]
        return model_id

    def _copy_all_models(self):
        """Copy all model names to clipboard (comma-separated, without provider prefix)."""
        if not self.models:
            return
        names = [self._get_model_display_name(m) for m in self.models]
        text = ", ".join(names)
        self.clipboard_clear()
        self.clipboard_append(text)

    def _copy_filtered_models(self):
        """Copy filtered/available model names to clipboard (comma-separated)."""
        if not self.models:
            return
        # Get only models that are not ignored (models without status default to available)
        available = [
            self._get_model_display_name(m)
            for m in self.models
            if self.statuses.get(m) is None or self.statuses[m].status != "ignored"
        ]
        text = ", ".join(available)
        self.clipboard_clear()
        self.clipboard_append(text)

    def get_model_at_position(self, model_id: str) -> Optional[ModelStatus]:
        """Get the status of a model."""
        return self.statuses.get(model_id)
