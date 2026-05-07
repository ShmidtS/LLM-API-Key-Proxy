# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

"""
Dialog and panel components for the Model Filter GUI.

Contains HelpWindow, UnsavedChangesDialog, ImportRulesDialog,
ImportResultDialog, and RulePanel.
"""

import customtkinter as ctk
from typing import Any, Callable, List, Optional

from proxy_app.model_filter_models import (
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_RED,
    ACCENT_YELLOW,
    BG_HOVER,
    BG_PRIMARY,
    BG_SECONDARY,
    BG_TERTIARY,
    BORDER_COLOR,
    FONT_FAMILY,
    FONT_SIZE_HEADER,
    FONT_SIZE_LARGE,
    FONT_SIZE_NORMAL,
    FONT_SIZE_SMALL,
    FilterRule,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    get_scroll_delta,
)
from proxy_app.model_filter_widgets import ToolTip, VirtualRuleList


# ════════════════════════════════════════════════════════════════════════════════
# HELP WINDOW
# ════════════════════════════════════════════════════════════════════════════════


class HelpWindow(ctk.CTkToplevel):
    """
    Modal help popup with comprehensive filtering documentation.
    Uses CTkTextbox for proper scrolling with dark theme styling.
    """

    def __init__(self, parent):
        super().__init__(parent)

        self.title("Help - Model Filtering")
        self.geometry("700x600")
        self.minsize(600, 500)

        # Make modal
        self.transient(parent)
        self.grab_set()

        # Configure appearance
        self.configure(fg_color=BG_PRIMARY)

        # Build content
        self._create_content()

        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")

        # Focus
        self.focus_force()

        # Bind escape to close
        self.bind("<Escape>", lambda e: self.destroy())

    def _create_content(self):
        """Build the help content using CTkTextbox for proper scrolling."""
        # Main container
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=20, pady=(20, 10))

        # Use CTkTextbox - CustomTkinter's styled text widget with built-in scrolling
        self.text_box = ctk.CTkTextbox(
            main_frame,
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg_color=BG_SECONDARY,
            text_color=TEXT_SECONDARY,
            corner_radius=8,
            wrap="word",
            activate_scrollbars=True,
        )
        self.text_box.pack(fill="both", expand=True)

        # Configure text tags for formatting
        # Access the underlying tk.Text widget for tag configuration
        text_widget = self.text_box._textbox

        text_widget.tag_configure(
            "title",
            font=(FONT_FAMILY, FONT_SIZE_HEADER, "bold"),
            foreground=TEXT_PRIMARY,
            spacing1=5,
            spacing3=15,
        )
        text_widget.tag_configure(
            "section_title",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            foreground=ACCENT_BLUE,
            spacing1=20,
            spacing3=8,
        )
        text_widget.tag_configure(
            "separator",
            font=(FONT_FAMILY, 6),
            foreground=BORDER_COLOR,
            spacing3=5,
        )
        text_widget.tag_configure(
            "content",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            foreground=TEXT_SECONDARY,
            spacing1=2,
            spacing3=5,
            lmargin1=5,
            lmargin2=5,
        )

        # Insert content
        self._insert_help_content()

        # Make read-only by disabling
        self.text_box.configure(state="disabled")

        # Bind mouse wheel for faster scrolling on the internal canvas
        self.text_box.bind("<MouseWheel>", self._on_mousewheel)
        # Also bind on the textbox's internal widget
        self.text_box._textbox.bind("<MouseWheel>", self._on_mousewheel)

        # Close button at bottom
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=(10, 15))

        close_btn = ctk.CTkButton(
            btn_frame,
            text="Got it!",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            fg_color=ACCENT_BLUE,
            hover_color="#3a8aee",
            height=40,
            width=120,
            command=self.destroy,
        )
        close_btn.pack()

    def _on_mousewheel(self, event):
        """Handle mouse wheel with faster scrolling."""
        # CTkTextbox uses _textbox internally
        # Use larger scroll amount (3 units) for faster scrolling in help window
        delta = get_scroll_delta(event) * 3
        self.text_box._textbox.yview_scroll(delta, "units")
        return "break"

    def _insert_help_content(self):
        """Insert all help text with formatting."""
        # Access internal text widget for inserting with tags
        text_widget = self.text_box._textbox

        # Title
        text_widget.insert("end", "📖 Model Filtering Guide\n", "title")

        # Sections with emojis
        sections = [
            (
                "🎯 Overview",
                """Model filtering allows you to control which models are available through your proxy for each provider.

• Use the IGNORE list to block specific models
• Use the WHITELIST to ensure specific models are always available
• Whitelist ALWAYS takes priority over Ignore""",
            ),
            (
                "⚖️ Filtering Priority",
                """When a model is checked, the following order is used:

1. WHITELIST CHECK
   If the model matches any whitelist pattern → AVAILABLE
   (Whitelist overrides everything else)

2. IGNORE CHECK
   If the model matches any ignore pattern → BLOCKED

3. DEFAULT
   If no patterns match → AVAILABLE""",
            ),
            (
                "✏️ Pattern Syntax",
                """Full glob/wildcard patterns are supported:

EXACT MATCH
  Pattern: gpt-4
  Matches: only "gpt-4", nothing else

PREFIX WILDCARD
  Pattern: gpt-4*
  Matches: "gpt-4", "gpt-4-turbo", "gpt-4-preview", etc.

SUFFIX WILDCARD
  Pattern: *-preview
  Matches: "gpt-4-preview", "o1-preview", etc.

CONTAINS WILDCARD
  Pattern: *-preview*
  Matches: anything containing "-preview"

MATCH ALL
  Pattern: *
  Matches: every model for this provider

SINGLE CHARACTER
  Pattern: gpt-?
  Matches: "gpt-4", "gpt-5", etc. (any single char)

CHARACTER SET
  Pattern: gpt-[45]*
  Matches: "gpt-4", "gpt-4-turbo", "gpt-5", etc.""",
            ),
            (
                "💡 Common Patterns",
                """BLOCK ALL, ALLOW SPECIFIC:
  Ignore:    *
  Whitelist: gpt-4o, gpt-4o-mini
  Result:    Only gpt-4o and gpt-4o-mini available

BLOCK PREVIEW MODELS:
  Ignore:    *-preview, *-preview*
  Result:    All preview variants blocked

BLOCK SPECIFIC SERIES:
  Ignore:    o1*, dall-e*
  Result:    All o1 and DALL-E models blocked

ALLOW ONLY LATEST:
  Ignore:    *
  Whitelist: *-latest
  Result:    Only models ending in "-latest" available""",
            ),
            (
                "🖱️ Interface Guide",
                """PROVIDER DROPDOWN
  Select which provider to configure

MODEL LISTS
  • Left list: All fetched models (unfiltered)
  • Right list: Same models with colored status
  • Green = Available (normal)
  • Red/Orange tones = Blocked (ignored)
  • Blue/Teal tones = Whitelisted

SEARCH BOX
  Filter both lists to find specific models quickly

CLICKING MODELS
  • Left-click: Highlight the rule affecting this model
  • Right-click: Context menu with quick actions

CLICKING RULES
  • Highlights all models affected by that rule
  • Shows which models will be blocked/allowed

RULE INPUT (Merge Mode)
  • Enter patterns separated by commas
  • Only adds patterns not covered by existing rules
  • Press Add or Enter to create rules

IMPORT BUTTON (Replace Mode)
  • Replaces ALL existing rules with imported ones
  • Paste comma-separated patterns

DELETE RULES
  • Click the × button on any rule to remove it""",
            ),
            (
                "⌨️ Keyboard Shortcuts",
                """Ctrl+S     Save changes
Ctrl+R     Refresh models from provider
Ctrl+F     Focus search box
F1         Open this help window
Escape     Clear search / Close dialogs""",
            ),
            (
                "💾 Saving Changes",
                """Changes are saved to your .env file in this format:

  IGNORE_MODELS_OPENAI=pattern1,pattern2*
  WHITELIST_MODELS_OPENAI=specific-model

Click "Save" to persist changes, or "Discard" to revert.
Closing the window with unsaved changes will prompt you.""",
            ),
        ]

        for section_title, content in sections:
            text_widget.insert("end", f"\n{section_title}\n", "section_title")
            text_widget.insert("end", "─" * 50 + "\n", "separator")
            text_widget.insert("end", content.strip() + "\n", "content")


# ════════════════════════════════════════════════════════════════════════════════
# CUSTOM DIALOGS
# ════════════════════════════════════════════════════════════════════════════════


class UnsavedChangesDialog(ctk.CTkToplevel):
    """Modal dialog for unsaved changes confirmation."""

    def __init__(self, parent):
        super().__init__(parent)

        self.result: Optional[str] = None  # 'save', 'discard', 'cancel'

        self.title("Unsaved Changes")
        self.geometry("400x180")
        self.resizable(False, False)

        # Make modal
        self.transient(parent)
        self.grab_set()

        # Configure appearance
        self.configure(fg_color=BG_PRIMARY)

        # Build content
        self._create_content()

        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")

        # Focus
        self.focus_force()

        # Bind escape to cancel
        self.bind("<Escape>", lambda e: self._on_cancel())

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _create_content(self):
        """Build dialog content."""
        # Icon and message
        msg_frame = ctk.CTkFrame(self, fg_color="transparent")
        msg_frame.pack(fill="x", padx=30, pady=(25, 15))

        icon = ctk.CTkLabel(
            msg_frame, text="⚠️", font=(FONT_FAMILY, 32), text_color=ACCENT_YELLOW
        )
        icon.pack(side="left", padx=(0, 15))

        text_frame = ctk.CTkFrame(msg_frame, fg_color="transparent")
        text_frame.pack(side="left", fill="x", expand=True)

        title = ctk.CTkLabel(
            text_frame,
            text="Unsaved Changes",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            text_color=TEXT_PRIMARY,
            anchor="w",
        )
        title.pack(anchor="w")

        subtitle = ctk.CTkLabel(
            text_frame,
            text="You have unsaved filter changes.\nWhat would you like to do?",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            text_color=TEXT_SECONDARY,
            anchor="w",
            justify="left",
        )
        subtitle.pack(anchor="w")

        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=30, pady=(10, 25))

        cancel_btn = ctk.CTkButton(
            btn_frame,
            text="Cancel",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg_color=BG_SECONDARY,
            hover_color=BG_HOVER,
            border_width=1,
            border_color=BORDER_COLOR,
            width=100,
            command=self._on_cancel,
        )
        cancel_btn.pack(side="right", padx=(10, 0))

        discard_btn = ctk.CTkButton(
            btn_frame,
            text="Discard",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg_color=ACCENT_RED,
            hover_color="#c0392b",
            width=100,
            command=self._on_discard,
        )
        discard_btn.pack(side="right", padx=(10, 0))

        save_btn = ctk.CTkButton(
            btn_frame,
            text="Save",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg_color=ACCENT_GREEN,
            hover_color="#27ae60",
            width=100,
            command=self._on_save,
        )
        save_btn.pack(side="right")

    def _on_save(self):
        self.result = "save"
        self.destroy()

    def _on_discard(self):
        self.result = "discard"
        self.destroy()

    def _on_cancel(self):
        self.result = "cancel"
        self.destroy()

    def show(self) -> Optional[str]:
        """Show dialog and return result."""
        self.wait_window()
        return self.result


class ImportRulesDialog(ctk.CTkToplevel):
    """Modal dialog for importing rules from comma-separated text."""

    def __init__(self, parent, rule_type: str):
        super().__init__(parent)

        self.result: Optional[List[str]] = None
        self.rule_type = rule_type

        title_text = (
            "Import Ignore Rules" if rule_type == "ignore" else "Import Whitelist Rules"
        )
        self.title(title_text)
        self.geometry("500x300")
        self.minsize(400, 250)

        # Make modal
        self.transient(parent)
        self.grab_set()

        # Configure appearance
        self.configure(fg_color=BG_PRIMARY)

        # Build content
        self._create_content()

        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")

        # Focus
        self.focus_force()
        self.text_box.focus_set()

        # Bind escape to cancel
        self.bind("<Escape>", lambda e: self._on_cancel())

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _create_content(self):
        """Build dialog content."""
        # Instructions at TOP
        instruction_frame = ctk.CTkFrame(self, fg_color="transparent")
        instruction_frame.pack(fill="x", padx=20, pady=(15, 10))

        instruction = ctk.CTkLabel(
            instruction_frame,
            text="Paste comma-separated patterns below (will REPLACE all existing rules):",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            text_color=TEXT_PRIMARY,
            anchor="w",
        )
        instruction.pack(anchor="w")

        example = ctk.CTkLabel(
            instruction_frame,
            text="Example: gpt-4*, claude-3*, model-name",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_MUTED,
            anchor="w",
        )
        example.pack(anchor="w")

        # Buttons at BOTTOM - pack BEFORE textbox to reserve space
        btn_frame = ctk.CTkFrame(self, fg_color="transparent", height=50)
        btn_frame.pack(side="bottom", fill="x", padx=20, pady=(10, 15))
        btn_frame.pack_propagate(False)

        cancel_btn = ctk.CTkButton(
            btn_frame,
            text="Cancel",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg_color=BG_SECONDARY,
            hover_color=BG_HOVER,
            border_width=1,
            border_color=BORDER_COLOR,
            width=100,
            height=32,
            command=self._on_cancel,
        )
        cancel_btn.pack(side="right", padx=(10, 0))

        import_btn = ctk.CTkButton(
            btn_frame,
            text="Replace All",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            fg_color=ACCENT_BLUE,
            hover_color="#3a8aee",
            width=110,
            height=32,
            command=self._on_import,
        )
        import_btn.pack(side="right")

        # Text box fills MIDDLE space - pack LAST
        self.text_box = ctk.CTkTextbox(
            self,
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg_color=BG_TERTIARY,
            border_color=BORDER_COLOR,
            border_width=1,
            text_color=TEXT_PRIMARY,
            wrap="word",
        )
        self.text_box.pack(fill="both", expand=True, padx=20, pady=(0, 0))

        # Bind Ctrl+Enter to import
        self.text_box.bind("<Control-Return>", lambda e: self._on_import())

    def _on_import(self):
        """Parse and return the patterns."""
        text = self.text_box.get("1.0", "end").strip()
        if text:
            # Parse comma-separated patterns
            patterns = [p.strip() for p in text.split(",") if p.strip()]
            self.result = patterns
        else:
            self.result = []
        self.destroy()

    def _on_cancel(self):
        self.result = None
        self.destroy()

    def show(self) -> Optional[List[str]]:
        """Show dialog and return list of patterns, or None if cancelled."""
        self.wait_window()
        return self.result


class ImportResultDialog(ctk.CTkToplevel):
    """Simple dialog showing import results."""

    def __init__(self, parent, added: int, skipped: int, is_replace: bool = False):
        super().__init__(parent)

        self.title("Import Complete")
        self.geometry("380x160")
        self.resizable(False, False)

        # Make modal
        self.transient(parent)
        self.grab_set()

        # Configure appearance
        self.configure(fg_color=BG_PRIMARY)

        # Build content
        self._create_content(added, skipped, is_replace)

        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")

        # Focus
        self.focus_force()

        # Bind escape and enter to close
        self.bind("<Escape>", lambda e: self.destroy())
        self.bind("<Return>", lambda e: self.destroy())

    def _create_content(self, added: int, skipped: int, is_replace: bool):
        """Build dialog content."""
        # Icon and message
        msg_frame = ctk.CTkFrame(self, fg_color="transparent")
        msg_frame.pack(fill="x", padx=30, pady=(25, 15))

        icon = ctk.CTkLabel(
            msg_frame,
            text="✅" if added > 0 else "ℹ️",
            font=(FONT_FAMILY, 28),
            text_color=ACCENT_GREEN if added > 0 else ACCENT_BLUE,
        )
        icon.pack(side="left", padx=(0, 15))

        text_frame = ctk.CTkFrame(msg_frame, fg_color="transparent")
        text_frame.pack(side="left", fill="x", expand=True)

        # Title text differs based on mode
        if is_replace:
            if added > 0:
                added_text = f"Replaced with {added} rule{'s' if added != 1 else ''}"
            else:
                added_text = "All rules cleared"
        else:
            if added > 0:
                added_text = f"Added {added} rule{'s' if added != 1 else ''}"
            else:
                added_text = "No new rules added"

        title = ctk.CTkLabel(
            text_frame,
            text=added_text,
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            text_color=TEXT_PRIMARY,
            anchor="w",
        )
        title.pack(anchor="w")

        # Subtitle for skipped/duplicates
        if skipped > 0:
            skip_text = f"{skipped} duplicate{'s' if skipped != 1 else ''} skipped"
            subtitle = ctk.CTkLabel(
                text_frame,
                text=skip_text,
                font=(FONT_FAMILY, FONT_SIZE_NORMAL),
                text_color=TEXT_MUTED,
                anchor="w",
            )
            subtitle.pack(anchor="w")

        # OK button
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=30, pady=(0, 20))

        ok_btn = ctk.CTkButton(
            btn_frame,
            text="OK",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg_color=ACCENT_BLUE,
            hover_color="#3a8aee",
            width=80,
            command=self.destroy,
        )
        ok_btn.pack(side="right")


# ════════════════════════════════════════════════════════════════════════════════
# RULE PANEL COMPONENT
# ════════════════════════════════════════════════════════════════════════════════


class RulePanel(ctk.CTkFrame):
    """
    Panel containing rule chips, input field, and add button.

    Uses VirtualRuleList for high-performance rendering of rules.
    """

    def __init__(
        self,
        master,
        title: str,
        rule_type: str,  # 'ignore' or 'whitelist'
        on_rules_changed: Callable[[], None],
        on_rule_clicked: Callable[[FilterRule], None],
        on_input_changed: Callable[[str, str], None],  # (text, rule_type)
    ):
        super().__init__(master, fg_color=BG_SECONDARY, corner_radius=8)

        self.title = title
        self.rule_type = rule_type
        self.on_rules_changed = on_rules_changed
        self.on_rule_clicked = on_rule_clicked
        self.on_input_changed = on_input_changed

        self._create_content()

    def _create_content(self):
        """Build panel content."""
        # Title row at top (compact) with count and buttons
        title_frame = ctk.CTkFrame(self, fg_color="transparent", height=22)
        title_frame.pack(side="top", fill="x", padx=10, pady=(4, 2))
        title_frame.pack_propagate(False)

        # Base title (without count)
        self._base_title = self.title
        self._rule_count = 0

        self.title_label = ctk.CTkLabel(
            title_frame,
            text=f"{self.title}: 0",
            font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"),
            text_color=TEXT_PRIMARY,
        )
        self.title_label.pack(side="left")

        # Import button (right side)
        import_btn = ctk.CTkButton(
            title_frame,
            text="Import",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color=BG_TERTIARY,
            hover_color=BG_HOVER,
            border_width=1,
            border_color=BORDER_COLOR,
            width=50,
            height=18,
            command=self._on_import_clicked,
        )
        import_btn.pack(side="right", padx=(4, 0))
        ToolTip(import_btn, "Import rules from comma-separated text")

        # Copy button
        copy_btn = ctk.CTkButton(
            title_frame,
            text="Copy",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color=BG_TERTIARY,
            hover_color=BG_HOVER,
            border_width=1,
            border_color=BORDER_COLOR,
            width=45,
            height=18,
            command=self._on_copy_clicked,
        )
        copy_btn.pack(side="right")
        ToolTip(copy_btn, "Copy all rules (comma-separated)")

        # Input frame at BOTTOM - pack BEFORE rule_list to reserve space
        input_frame = ctk.CTkFrame(self, fg_color="transparent", height=32)
        input_frame.pack(side="bottom", fill="x", padx=6, pady=(2, 4))
        input_frame.pack_propagate(False)  # Prevent children from changing frame height

        # Pattern input
        self.input_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="pattern1, pattern2*, ...",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color=BG_TERTIARY,
            border_color=BORDER_COLOR,
            text_color=TEXT_PRIMARY,
            placeholder_text_color=TEXT_MUTED,
            height=28,
        )
        self.input_entry.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self.input_entry.bind("<Return>", self._on_add_clicked)
        self.input_entry.bind("<KeyRelease>", self._on_input_key)

        # Add button
        add_btn = ctk.CTkButton(
            input_frame,
            text="+ Add",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color=ACCENT_BLUE,
            hover_color="#3a8aee",
            width=55,
            height=28,
            command=self._on_add_clicked,
        )
        add_btn.pack(side="right")

        # Virtual rule list fills REMAINING middle space - pack LAST
        self.rule_list = VirtualRuleList(
            self,
            rule_type=self.rule_type,
            on_rule_click=self.on_rule_clicked,
            on_rule_delete=self._on_rule_delete,
        )
        self.rule_list.pack(side="top", fill="both", expand=True, padx=6, pady=(0, 2))

    def _on_input_key(self, event=None):
        """Handle key release in input field - for real-time preview."""
        text = self.input_entry.get().strip()
        self.on_input_changed(text, self.rule_type)

    def _on_add_clicked(self, event=None):
        """Handle add button click."""
        text = self.input_entry.get().strip()
        if text:
            # Parse comma-separated patterns
            patterns = [p.strip() for p in text.split(",") if p.strip()]
            if patterns:
                self.input_entry.delete(0, "end")
                for pattern in patterns:
                    self._emit_add_pattern(pattern)

    def _emit_add_pattern(self, pattern: str):
        """Emit request to add a pattern (handled by parent)."""
        if hasattr(self, "_add_pattern_callback"):
            self._add_pattern_callback(pattern)

    def set_add_callback(self, callback: Callable[[str], Any]):
        """Set the callback for adding patterns."""
        self._add_pattern_callback = callback

    def add_rule_chip(self, rule: FilterRule):
        """Add a rule to the panel."""
        self.rule_list.add_rule(rule)

    def remove_rule_chip(self, pattern: str):
        """Remove a rule from the panel."""
        self.rule_list.remove_rule(pattern)

    def _on_rule_delete(self, pattern: str):
        """Handle rule deletion."""
        if hasattr(self, "_delete_pattern_callback"):
            self._delete_pattern_callback(pattern)

    def set_delete_callback(self, callback: Callable[[str], None]):
        """Set the callback for deleting patterns."""
        self._delete_pattern_callback = callback

    def update_rule_counts(self, rules: List[FilterRule], models: List[str]):
        """Update affected counts for all rules."""
        self.rule_list.update_rule_counts(rules)
        self._update_title_count(len(rules))

    def _update_title_count(self, count: int):
        """Update the rule count in the title."""
        self._rule_count = count
        self.title_label.configure(text=f"{self._base_title}: {count}")

    def highlight_rule(self, pattern: str):
        """Highlight a specific rule and scroll to it."""
        self.rule_list.highlight_rule(pattern)

    def clear_highlights(self):
        """Clear all rule highlights."""
        self.rule_list.clear_highlights()

    def clear_all(self):
        """Remove all rules."""
        self.rule_list.clear_all()

    def get_input_text(self) -> str:
        """Get current input text."""
        return self.input_entry.get().strip()

    def clear_input(self):
        """Clear the input field."""
        self.input_entry.delete(0, "end")

    def _on_copy_clicked(self):
        """Copy all rule patterns to clipboard as comma-separated string."""
        patterns = [r.pattern for r in self.rule_list.rules]
        if patterns:
            text = ", ".join(patterns)
            self.clipboard_clear()
            self.clipboard_append(text)

    def _on_import_clicked(self):
        """
        Open import dialog and REPLACE ALL existing rules.

        This is a full replace operation - all existing rules are removed
        and replaced with the imported patterns.
        """
        dialog = ImportRulesDialog(self.winfo_toplevel(), self.rule_type)
        patterns = dialog.show()

        if patterns is None:
            # Cancelled
            return

        if not patterns:
            # Empty input - show message
            ImportResultDialog(self.winfo_toplevel(), 0, 0, is_replace=True)
            return

        # Deduplicate the imported patterns (keep first occurrence)
        seen = set()
        unique_patterns = []
        duplicates_in_import = 0
        for p in patterns:
            if p not in seen:
                seen.add(p)
                unique_patterns.append(p)
            else:
                duplicates_in_import += 1

        # Clear all existing rules first
        if hasattr(self, "_clear_all_callback"):
            self._clear_all_callback()

        # Add all unique patterns (skip coverage check since we're replacing)
        added = 0
        if hasattr(self, "_replace_add_callback"):
            for pattern in unique_patterns:
                if self._replace_add_callback(pattern):
                    added += 1

        # Show result dialog
        ImportResultDialog(
            self.winfo_toplevel(), added, duplicates_in_import, is_replace=True
        )

    def set_clear_all_callback(self, callback: Callable[[], None]):
        """Set the callback for clearing all rules (used by replace import)."""
        self._clear_all_callback = callback

    def set_replace_add_callback(self, callback: Callable[[str], bool]):
        """Set the callback for adding patterns in replace mode (skips coverage check)."""
        self._replace_add_callback = callback

    def get_all_patterns(self) -> List[str]:
        """Get all rule patterns."""
        return [r.pattern for r in self.rule_list.rules]
