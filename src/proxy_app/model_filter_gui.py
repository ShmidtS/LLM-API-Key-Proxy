# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

"""
Model Filter GUI - Visual editor for model ignore/whitelist rules.

A CustomTkinter application that provides a friendly interface for managing
which models are available per provider through ignore lists and whitelists.

This module is the thin orchestrator that imports from the split modules:
- model_filter_models: Data classes and shared constants
- model_filter_engine: Core filtering logic
- model_filter_fetcher: Background model fetching
- model_filter_widgets: UI widget components
- model_filter_dialogs: Dialog and panel components

Features:
- Two synchronized model lists showing all fetched models and their filtered status
- Color-coded rules with visual association to affected models
- Real-time filtering preview as you type patterns
- Click interactions to highlight rule-model relationships
- Right-click context menus for quick actions
- Comprehensive help documentation
"""

import customtkinter as ctk
from tkinter import Menu
from typing import List, Optional

from proxy_app.model_filter_models import (
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_YELLOW,
    BG_HOVER,
    BG_PRIMARY,
    BG_SECONDARY,
    BORDER_COLOR,
    FONT_FAMILY,
    FONT_SIZE_LARGE,
    FONT_SIZE_NORMAL,
    FONT_SIZE_SMALL,
    FilterRule,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    WINDOW_DEFAULT_SIZE,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
    WINDOW_TITLE,
)
from proxy_app.model_filter_engine import FilterEngine
from proxy_app.model_filter_fetcher import ModelFetcher
from proxy_app.model_filter_dialogs import (
    HelpWindow,
    ImportResultDialog,
    RulePanel,
    UnsavedChangesDialog,
)
from proxy_app.model_filter_widgets import (
    ToolTip,
    VirtualSyncModelLists,
)


# ════════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION WINDOW
# ════════════════════════════════════════════════════════════════════════════════


class ModelFilterGUI(ctk.CTk):
    """
    Main application window for model filter configuration.

    Provides a visual interface for managing IGNORE_MODELS_* and WHITELIST_MODELS_*
    environment variables per provider.
    """

    def __init__(self):
        super().__init__()

        # Window configuration
        self.title(WINDOW_TITLE)
        self.geometry(WINDOW_DEFAULT_SIZE)
        self.minsize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.configure(fg_color=BG_PRIMARY)

        # State
        self.current_provider: Optional[str] = None
        self.models: List[str] = []
        self.filter_engine = FilterEngine()
        self.available_providers: List[str] = []
        self._preview_pattern: str = ""
        self._preview_rule_type: str = ""
        self._update_scheduled: bool = False
        self._pending_providers_to_fetch: List[str] = []
        self._fetch_in_progress: bool = False
        self._preview_after_id: Optional[str] = None

        # Build UI with grid layout for responsive sizing
        self._create_main_layout()

        # Context menu
        self._create_context_menu()

        # Load providers and start fetching all models
        self._load_providers()

        # Bind keyboard shortcuts
        self._bind_shortcuts()

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Focus and raise window after it's fully loaded
        self.after(100, self._activate_window)

    def _create_main_layout(self):
        """Create the main layout with grid weights for 3:1 ratio."""
        # Main content frame - regular frame with grid layout
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.pack(fill="both", expand=True, padx=15, pady=(5, 8))

        # Configure grid with proper weights for 3:1 ratio
        self.content_frame.grid_columnconfigure(0, weight=1)

        # Row 0: Header - fixed height
        self.content_frame.grid_rowconfigure(0, weight=0)
        # Row 1: Search - fixed height
        self.content_frame.grid_rowconfigure(1, weight=0)
        # Row 2: Model lists - weight=3 for 3:1 ratio, minimum 100px
        self.content_frame.grid_rowconfigure(2, weight=3, minsize=200)
        # Row 3: Rule panels - weight=1 for 3:1 ratio, minimum 55px
        self.content_frame.grid_rowconfigure(3, weight=1, minsize=55)
        # Row 4: Status bar - fixed height
        self.content_frame.grid_rowconfigure(4, weight=0)

        # Create all sections
        self._create_header()
        self._create_search_bar()
        self._create_model_lists()
        self._create_rule_panels()
        self._create_status_bar()
        self._create_action_buttons()

    def _activate_window(self):
        """Activate and focus the window."""
        self.lift()
        self.focus_force()
        self.attributes("-topmost", True)
        self.after(200, lambda: self.attributes("-topmost", False))

    def _create_header(self):
        """Create the header with provider selector and buttons (compact)."""
        header = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        # Title (smaller font)
        title = ctk.CTkLabel(
            header,
            text="🎯 Model Filter Configuration",
            font=(FONT_FAMILY, FONT_SIZE_LARGE, "bold"),
            text_color=TEXT_PRIMARY,
        )
        title.pack(side="left")

        # Help button (smaller)
        help_btn = ctk.CTkButton(
            header,
            text="?",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, "bold"),
            fg_color=BG_SECONDARY,
            hover_color=BG_HOVER,
            border_width=1,
            border_color=BORDER_COLOR,
            width=26,
            height=26,
            corner_radius=13,
            command=self._show_help,
        )
        help_btn.pack(side="right", padx=(8, 0))
        ToolTip(help_btn, "Help (F1)")

        # Refresh button (smaller)
        refresh_btn = ctk.CTkButton(
            header,
            text="🔄 Refresh",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color=BG_SECONDARY,
            hover_color=BG_HOVER,
            border_width=1,
            border_color=BORDER_COLOR,
            width=80,
            height=26,
            command=self._refresh_models,
        )
        refresh_btn.pack(side="right", padx=(8, 0))
        ToolTip(refresh_btn, "Refresh models (Ctrl+R)")

        # Provider selector (compact)
        provider_frame = ctk.CTkFrame(header, fg_color="transparent")
        provider_frame.pack(side="right")

        provider_label = ctk.CTkLabel(
            provider_frame,
            text="Provider:",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_SECONDARY,
        )
        provider_label.pack(side="left", padx=(0, 6))

        self.provider_dropdown = ctk.CTkComboBox(
            provider_frame,
            values=["Loading..."],
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            dropdown_font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color=BG_SECONDARY,
            border_color=BORDER_COLOR,
            button_color=BORDER_COLOR,
            button_hover_color=BG_HOVER,
            dropdown_fg_color=BG_SECONDARY,
            dropdown_hover_color=BG_HOVER,
            text_color=TEXT_PRIMARY,
            width=160,
            height=26,
            state="readonly",
            command=self._on_provider_changed,
        )
        self.provider_dropdown.pack(side="left")

    def _create_search_bar(self):
        """Create the search bar (compact version)."""
        search_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        search_frame.grid(row=1, column=0, sticky="ew", pady=(0, 5))

        search_icon = ctk.CTkLabel(
            search_frame,
            text="🔍",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_MUTED,
        )
        search_icon.pack(side="left", padx=(0, 6))

        self.search_entry = ctk.CTkEntry(
            search_frame,
            placeholder_text="Search models...",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg_color=BG_SECONDARY,
            border_color=BORDER_COLOR,
            text_color=TEXT_PRIMARY,
            placeholder_text_color=TEXT_MUTED,
            height=28,
        )
        self.search_entry.pack(side="left", fill="x", expand=True)
        self.search_entry.bind("<KeyRelease>", self._on_search_changed)

        # Clear button
        clear_btn = ctk.CTkButton(
            search_frame,
            text="×",
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            fg_color="transparent",
            hover_color=BG_HOVER,
            text_color=TEXT_MUTED,
            width=28,
            height=28,
            command=self._clear_search,
        )
        clear_btn.pack(side="left")

    def _create_model_lists(self):
        """Create the synchronized model list panel."""
        # Use the virtual list implementation for performance
        self.model_list_panel = VirtualSyncModelLists(
            self.content_frame,
            on_model_click=self._on_model_clicked,
            on_model_right_click=self._on_model_right_clicked,
        )
        self.model_list_panel.grid(row=2, column=0, sticky="nsew", pady=(0, 5))

    def _create_rule_panels(self):
        """Create the ignore and whitelist rule panels."""
        self.rules_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.rules_frame.grid(row=3, column=0, sticky="nsew", pady=(0, 5))
        # Don't let content dictate size - let parent grid control height
        self.rules_frame.grid_propagate(False)
        self.rules_frame.grid_columnconfigure(0, weight=1)
        self.rules_frame.grid_columnconfigure(1, weight=1)
        self.rules_frame.grid_rowconfigure(0, weight=1)

        # Ignore panel
        self.ignore_panel = RulePanel(
            self.rules_frame,
            title="🚫 Ignore Rules",
            rule_type="ignore",
            on_rules_changed=self._on_rules_changed,
            on_rule_clicked=self._on_rule_clicked,
            on_input_changed=self._on_rule_input_changed,
        )
        self.ignore_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.ignore_panel.set_add_callback(lambda p: self._add_ignore_pattern(p))
        self.ignore_panel.set_delete_callback(self._remove_ignore_pattern)
        self.ignore_panel.set_clear_all_callback(self._clear_all_ignore_rules)
        self.ignore_panel.set_replace_add_callback(
            lambda p: self._add_ignore_pattern(p, skip_coverage_check=True)
        )

        # Whitelist panel
        self.whitelist_panel = RulePanel(
            self.rules_frame,
            title="✓ Whitelist Rules",
            rule_type="whitelist",
            on_rules_changed=self._on_rules_changed,
            on_rule_clicked=self._on_rule_clicked,
            on_input_changed=self._on_rule_input_changed,
        )
        self.whitelist_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        self.whitelist_panel.set_add_callback(lambda p: self._add_whitelist_pattern(p))
        self.whitelist_panel.set_delete_callback(self._remove_whitelist_pattern)
        self.whitelist_panel.set_clear_all_callback(self._clear_all_whitelist_rules)
        self.whitelist_panel.set_replace_add_callback(
            lambda p: self._add_whitelist_pattern(p, skip_coverage_check=True)
        )

    def _create_status_bar(self):
        """Create the status bar showing available count and action buttons (compact)."""
        # Combined status bar and action buttons in one row
        self.status_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.status_frame.grid(row=4, column=0, sticky="ew", pady=(3, 3))

        # Status label (left side, smaller font)
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Select a provider to begin",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_SECONDARY,
        )
        self.status_label.pack(side="left")

        # Unsaved indicator (after status)
        self.unsaved_label = ctk.CTkLabel(
            self.status_frame,
            text="",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=ACCENT_YELLOW,
        )
        self.unsaved_label.pack(side="left", padx=(10, 0))

        # Buttons (right side, smaller)
        # Discard button
        discard_btn = ctk.CTkButton(
            self.status_frame,
            text="↩️ Discard",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color=BG_SECONDARY,
            hover_color=BG_HOVER,
            border_width=1,
            border_color=BORDER_COLOR,
            width=85,
            height=26,
            command=self._discard_changes,
        )
        discard_btn.pack(side="right", padx=(8, 0))

        # Save button
        save_btn = ctk.CTkButton(
            self.status_frame,
            text="💾 Save",
            font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"),
            fg_color=ACCENT_GREEN,
            hover_color="#27ae60",
            width=75,
            height=26,
            command=self._save_changes,
        )
        save_btn.pack(side="right")
        ToolTip(save_btn, "Save changes (Ctrl+S)")

    def _create_action_buttons(self):
        """Action buttons are now part of status bar - this is a no-op for compatibility."""
        pass

    def _create_context_menu(self):
        """Create the right-click context menu."""
        self.context_menu = Menu(self, tearoff=0, bg=BG_SECONDARY, fg=TEXT_PRIMARY)
        self.context_menu.add_command(
            label="➕ Add to Ignore List",
            command=lambda: self._add_model_to_list("ignore"),
        )
        self.context_menu.add_command(
            label="➕ Add to Whitelist",
            command=lambda: self._add_model_to_list("whitelist"),
        )
        self.context_menu.add_separator()
        self.context_menu.add_command(
            label="🔍 View Affecting Rule", command=self._view_affecting_rule
        )
        self.context_menu.add_command(
            label="📋 Copy Model Name", command=self._copy_model_name
        )

        self._context_model_id: Optional[str] = None

    def _bind_shortcuts(self):
        """Bind keyboard shortcuts."""
        self.bind("<Control-s>", lambda e: self._save_changes())
        self.bind("<Control-r>", lambda e: self._refresh_models())
        self.bind("<Control-f>", lambda e: self.search_entry.focus_set())
        self.bind("<F1>", lambda e: self._show_help())
        self.bind("<Escape>", self._on_escape)

    def _on_escape(self, event=None):
        """Handle escape key."""
        # Clear search if has content
        if self.search_entry.get():
            self._clear_search()
        else:
            # Clear highlights
            self.model_list_panel.clear_highlights()
            self.ignore_panel.clear_highlights()
            self.whitelist_panel.clear_highlights()

    # ─────────────────────────────────────────────────────────────────────────────
    # Provider Management
    # ─────────────────────────────────────────────────────────────────────────────

    def _load_providers(self):
        """Load available providers and start fetching all models in background."""
        self.available_providers = ModelFetcher.get_available_providers()

        if self.available_providers:
            self.provider_dropdown.configure(values=self.available_providers)
            self.provider_dropdown.set(self.available_providers[0])

            # Start fetching all provider models in background
            self._pending_providers_to_fetch = list(self.available_providers)
            self.status_label.configure(text="Loading models for all providers...")
            self._fetch_next_provider()

            # Load the first provider immediately
            self._on_provider_changed(self.available_providers[0])
        else:
            self.provider_dropdown.configure(values=["No providers found"])
            self.provider_dropdown.set("No providers found")
            self.status_label.configure(
                text="No providers with credentials found. Add API keys to .env first."
            )

    def _fetch_next_provider(self):
        """Fetch models for the next provider in the queue (background prefetch)."""
        if not self._pending_providers_to_fetch or self._fetch_in_progress:
            return

        self._fetch_in_progress = True
        provider = self._pending_providers_to_fetch.pop(0)

        # Skip if already cached
        if ModelFetcher.get_cached_models(provider) is not None:
            self._fetch_in_progress = False
            self.after(10, self._fetch_next_provider)
            return

        def on_done(models):
            self._fetch_in_progress = False
            # If this is the current provider, update display
            if provider == self.current_provider:
                self._on_models_loaded(models)
            # Continue with next provider
            self.after(100, self._fetch_next_provider)

        def on_error(error):
            self._fetch_in_progress = False
            # Continue with next provider even on error
            self.after(100, self._fetch_next_provider)

        ModelFetcher.fetch_models(
            provider,
            on_success=on_done,
            on_error=on_error,
            force_refresh=False,
        )

    def _on_provider_changed(self, provider: str):
        """Handle provider selection change."""
        if provider == self.current_provider:
            return

        # Check for unsaved changes
        if self.current_provider and self.filter_engine.has_unsaved_changes():
            result = self._show_unsaved_dialog()
            if result == "cancel":
                # Reset dropdown
                self.provider_dropdown.set(self.current_provider)
                return
            elif result == "save":
                self._save_changes()

        self.current_provider = provider
        self.models = []

        # Clear UI
        self.ignore_panel.clear_all()
        self.whitelist_panel.clear_all()
        self.model_list_panel.clear_highlights()

        # Load rules for this provider
        self.filter_engine.load_from_env(provider)
        self._populate_rule_panels()

        # Try to load from cache first
        cached_models = ModelFetcher.get_cached_models(provider)
        if cached_models is not None:
            self._on_models_loaded(cached_models)
        else:
            # Fetch models (will cache automatically)
            self._fetch_models()

    def _fetch_models(self, force_refresh: bool = False):
        """Fetch models for current provider."""
        if not self.current_provider:
            return

        self.model_list_panel.show_loading(self.current_provider)
        self.status_label.configure(
            text=f"Fetching models from {self.current_provider}..."
        )

        ModelFetcher.fetch_models(
            self.current_provider,
            on_success=self._on_models_loaded,
            on_error=self._on_models_error,
            on_start=None,
            force_refresh=force_refresh,
        )

    def _on_models_loaded(self, models: List[str]):
        """Handle successful model fetch."""
        # Deduplicate while preserving order, then sort
        self.models = sorted(list(dict.fromkeys(models)))

        # Update filter engine counts
        self.filter_engine.update_affected_counts(self.models)

        # Update UI (must be on main thread)
        self.after(0, self._update_model_display)

    def _on_models_error(self, error: str):
        """Handle model fetch error."""
        self.after(
            0,
            lambda: self.model_list_panel.show_error(
                error, on_retry=self._refresh_models
            ),
        )
        self.after(
            0,
            lambda: self.status_label.configure(
                text=f"Failed to fetch models: {error}"
            ),
        )

    def _update_model_display(self):
        """Update the model list display."""
        statuses = self.filter_engine.get_all_statuses(self.models)
        self.model_list_panel.set_models(self.models, statuses)

        # Update rule counts
        self.ignore_panel.update_rule_counts(
            self.filter_engine.ignore_rules, self.models
        )
        self.whitelist_panel.update_rule_counts(
            self.filter_engine.whitelist_rules, self.models
        )

        # Update status
        self._update_status()

    def _refresh_models(self):
        """Refresh models from provider (force bypass cache)."""
        if self.current_provider:
            ModelFetcher.clear_cache(self.current_provider)
            self._fetch_models(force_refresh=True)

    # ─────────────────────────────────────────────────────────────────────────────
    # Rule Management
    # ─────────────────────────────────────────────────────────────────────────────

    def _populate_rule_panels(self):
        """Populate rule panels from filter engine."""
        for rule in self.filter_engine.ignore_rules:
            self.ignore_panel.add_rule_chip(rule)

        for rule in self.filter_engine.whitelist_rules:
            self.whitelist_panel.add_rule_chip(rule)

    def _add_ignore_pattern(self, pattern: str, skip_coverage_check: bool = False):
        """
        Add an ignore pattern with smart merge logic.

        If skip_coverage_check is False (default - from main input):
        - Skip if pattern is already covered by existing rules
        - Remove existing patterns that would be covered by this new pattern

        If skip_coverage_check is True (from replace import):
        - Just add without coverage checks
        """
        if not skip_coverage_check:
            # Check if this pattern is already covered
            if self.filter_engine.is_pattern_covered(pattern, "ignore"):
                return False  # Pattern already covered, skip

            # Remove patterns that this new pattern would cover
            covered = self.filter_engine.get_covered_patterns(pattern, "ignore")
            for covered_pattern in covered:
                self._remove_ignore_pattern(covered_pattern)

        rule = self.filter_engine.add_ignore_rule(pattern)
        if rule:
            self.ignore_panel.add_rule_chip(rule)
            self._on_rules_changed()
            return True
        return False

    def _add_whitelist_pattern(self, pattern: str, skip_coverage_check: bool = False):
        """
        Add a whitelist pattern with smart merge logic.

        If skip_coverage_check is False (default - from main input):
        - Skip if pattern is already covered by existing rules
        - Remove existing patterns that would be covered by this new pattern

        If skip_coverage_check is True (from replace import):
        - Just add without coverage checks
        """
        if not skip_coverage_check:
            # Check if this pattern is already covered
            if self.filter_engine.is_pattern_covered(pattern, "whitelist"):
                return False  # Pattern already covered, skip

            # Remove patterns that this new pattern would cover
            covered = self.filter_engine.get_covered_patterns(pattern, "whitelist")
            for covered_pattern in covered:
                self._remove_whitelist_pattern(covered_pattern)

        rule = self.filter_engine.add_whitelist_rule(pattern)
        if rule:
            self.whitelist_panel.add_rule_chip(rule)
            self._on_rules_changed()
            return True
        return False

    def _remove_ignore_pattern(self, pattern: str):
        """Remove an ignore pattern."""
        self.filter_engine.remove_ignore_rule(pattern)
        self.ignore_panel.remove_rule_chip(pattern)
        self._on_rules_changed()

    def _remove_whitelist_pattern(self, pattern: str):
        """Remove a whitelist pattern."""
        self.filter_engine.remove_whitelist_rule(pattern)
        self.whitelist_panel.remove_rule_chip(pattern)
        self._on_rules_changed()

    def _clear_all_ignore_rules(self):
        """Clear all ignore rules (used by replace import)."""
        # Remove all rules from engine
        patterns = [r.pattern for r in self.filter_engine.ignore_rules]
        for pattern in patterns:
            self.filter_engine.remove_ignore_rule(pattern)
        # Clear the panel
        self.ignore_panel.clear_all()
        self._on_rules_changed()

    def _clear_all_whitelist_rules(self):
        """Clear all whitelist rules (used by replace import)."""
        # Remove all rules from engine
        patterns = [r.pattern for r in self.filter_engine.whitelist_rules]
        for pattern in patterns:
            self.filter_engine.remove_whitelist_rule(pattern)
        # Clear the panel
        self.whitelist_panel.clear_all()
        self._on_rules_changed()

    def _on_rules_changed(self):
        """Handle any rule change - uses debouncing to reduce lag."""
        if self._update_scheduled:
            return

        self._update_scheduled = True
        self.after(50, self._perform_rules_update)

    def _perform_rules_update(self):
        """Actually perform the rules update (called via debounce)."""
        self._update_scheduled = False

        # Update affected counts
        self.filter_engine.update_affected_counts(self.models)

        # Update model statuses
        statuses = self.filter_engine.get_all_statuses(self.models)
        self.model_list_panel.update_statuses(statuses)

        # Update rule counts
        self.ignore_panel.update_rule_counts(
            self.filter_engine.ignore_rules, self.models
        )
        self.whitelist_panel.update_rule_counts(
            self.filter_engine.whitelist_rules, self.models
        )

        # Update status
        self._update_status()

    def _on_rule_input_changed(self, text: str, rule_type: str):
        """Handle real-time input change for preview - debounced."""
        self._preview_pattern = text
        self._preview_rule_type = rule_type

        # Cancel any pending preview update
        if hasattr(self, "_preview_after_id") and self._preview_after_id:
            self.after_cancel(self._preview_after_id)

        # Debounce preview updates
        self._preview_after_id = self.after(
            100, lambda: self._perform_preview_update(text, rule_type)
        )

    def _perform_preview_update(self, text: str, rule_type: str):
        """Actually perform the preview update."""
        if not text or not self.models:
            self.model_list_panel.clear_highlights()
            return

        # Parse comma-separated patterns
        patterns = [p.strip() for p in text.split(",") if p.strip()]

        # Find all affected models
        affected = []
        for pattern in patterns:
            affected.extend(
                self.filter_engine.preview_pattern(pattern, rule_type, self.models)
            )

        # Highlight affected models using new virtual list API
        if affected:
            affected_set = set(affected)
            self.model_list_panel.left_list.highlight_models(affected_set)
            self.model_list_panel.right_list.highlight_models(affected_set)

            # Scroll to first affected
            self.model_list_panel.scroll_to_affected(affected)
        else:
            self.model_list_panel.clear_highlights()

    def _on_rule_clicked(self, rule: FilterRule):
        """Handle click on a rule chip."""
        # Highlight affected models
        self.model_list_panel.highlight_models_by_rule(rule)

        # Highlight the clicked rule
        if rule.rule_type == "ignore":
            self.ignore_panel.highlight_rule(rule.pattern)
            self.whitelist_panel.clear_highlights()
        else:
            self.whitelist_panel.highlight_rule(rule.pattern)
            self.ignore_panel.clear_highlights()

    # ─────────────────────────────────────────────────────────────────────────────
    # Model Interactions
    # ─────────────────────────────────────────────────────────────────────────────

    def _on_model_clicked(self, model_id: str):
        """Handle left-click on a model."""
        status = self.model_list_panel.get_model_at_position(model_id)

        if status and status.affecting_rule:
            # Highlight the affecting rule
            rule = status.affecting_rule
            if rule.rule_type == "ignore":
                self.ignore_panel.highlight_rule(rule.pattern)
                self.whitelist_panel.clear_highlights()
            else:
                self.whitelist_panel.highlight_rule(rule.pattern)
                self.ignore_panel.clear_highlights()

            # Also highlight the model
            self.model_list_panel.highlight_model(model_id)
        else:
            # No affecting rule - just show highlight briefly
            self.model_list_panel.highlight_model(model_id)
            self.ignore_panel.clear_highlights()
            self.whitelist_panel.clear_highlights()

    def _on_model_right_clicked(self, model_id: str, event):
        """Handle right-click on a model."""
        self._context_model_id = model_id

        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()

    def _add_model_to_list(self, list_type: str):
        """Add the context menu model to ignore or whitelist."""
        if not self._context_model_id:
            return

        # Extract model name without provider prefix
        if "/" in self._context_model_id:
            pattern = self._context_model_id.split("/", 1)[1]
        else:
            pattern = self._context_model_id

        if list_type == "ignore":
            self._add_ignore_pattern(pattern)
        else:
            self._add_whitelist_pattern(pattern)

    def _view_affecting_rule(self):
        """View the rule affecting the context menu model."""
        if not self._context_model_id:
            return

        self._on_model_clicked(self._context_model_id)

    def _copy_model_name(self):
        """Copy the context menu model name to clipboard."""
        if self._context_model_id:
            self.clipboard_clear()
            self.clipboard_append(self._context_model_id)

    # ─────────────────────────────────────────────────────────────────────────────
    # Search
    # ─────────────────────────────────────────────────────────────────────────────

    def _on_search_changed(self, event=None):
        """Handle search input change."""
        query = self.search_entry.get()
        self.model_list_panel.filter_by_search(query)

    def _clear_search(self):
        """Clear search field."""
        self.search_entry.delete(0, "end")
        self.model_list_panel.filter_by_search("")

    # ─────────────────────────────────────────────────────────────────────────────
    # Status & UI Updates
    # ─────────────────────────────────────────────────────────────────────────────

    def _update_status(self):
        """Update the status bar."""
        if not self.models:
            self.status_label.configure(text="No models loaded")
            return

        available, total = self.filter_engine.get_available_count(self.models)
        ignored = total - available

        if ignored > 0:
            text = f"✅ {available} of {total} models available ({ignored} ignored)"
        else:
            text = f"✅ All {total} models available"

        self.status_label.configure(text=text)

        # Update unsaved indicator
        if self.filter_engine.has_unsaved_changes():
            self.unsaved_label.configure(text="● Unsaved changes")
        else:
            self.unsaved_label.configure(text="")

    # ─────────────────────────────────────────────────────────────────────────────
    # Dialogs
    # ─────────────────────────────────────────────────────────────────────────────

    def _show_help(self):
        """Show help window."""
        HelpWindow(self)

    def _show_unsaved_dialog(self) -> str:
        """Show unsaved changes dialog. Returns 'save', 'discard', or 'cancel'."""
        dialog = UnsavedChangesDialog(self)
        return dialog.show() or "cancel"

    # ─────────────────────────────────────────────────────────────────────────────
    # Save / Discard
    # ─────────────────────────────────────────────────────────────────────────────

    def _save_changes(self):
        """Save current rules to .env file."""
        if not self.current_provider:
            return

        if self.filter_engine.save_to_env(self.current_provider):
            self.status_label.configure(text="✅ Changes saved successfully!")
            self.unsaved_label.configure(text="")

            # Reset to show normal status after a moment
            self.after(2000, self._update_status)
        else:
            self.status_label.configure(text="❌ Failed to save changes")

    def _discard_changes(self):
        """Discard unsaved changes."""
        if not self.current_provider:
            return

        if not self.filter_engine.has_unsaved_changes():
            return

        # Reload from env
        self.filter_engine.discard_changes()

        # Rebuild rule panels
        self.ignore_panel.clear_all()
        self.whitelist_panel.clear_all()
        self._populate_rule_panels()

        # Update display
        self._on_rules_changed()

        self.status_label.configure(text="Changes discarded")
        self.after(2000, self._update_status)

    # ─────────────────────────────────────────────────────────────────────────────
    # Window Close
    # ─────────────────────────────────────────────────────────────────────────────

    def _on_close(self):
        """Handle window close."""
        if self.filter_engine.has_unsaved_changes():
            result = self._show_unsaved_dialog()
            if result == "cancel":
                return
            elif result == "save":
                self._save_changes()

        self.destroy()


# ════════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════


def run_model_filter_gui():
    """
    Launch the Model Filter GUI application.

    This function configures CustomTkinter for dark mode and starts the
    main application loop. It blocks until the window is closed.
    """
    # Force dark mode
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    # Create and run app
    app = ModelFilterGUI()
    app.mainloop()


if __name__ == "__main__":
    run_model_filter_gui()
