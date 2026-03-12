"""
Export Panel - Export channel selection, event markers, and action buttons.
"""

import tkinter as tk
from utility.user_prefs import set_export_defaults, set_marker_defaults


class ExportPanel:
    """Handles export channel listbox, event marker listbox, and action buttons."""

    def __init__(self, app):
        self.app = app

    def setup(self, parent):
        """Create the export selection and analysis tools panel."""
        self.app.export_panel = tk.Frame(parent)
        self.app.export_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

        compute_frame = tk.LabelFrame(self.app.export_panel, text="Compute", font=self.app.bold_font)
        compute_frame.pack(fill=tk.X, padx=5, pady=(5, 2))

        label_frame = tk.LabelFrame(self.app.export_panel, text="Label", font=self.app.bold_font)
        label_frame.pack(fill=tk.X, padx=5, pady=2)

        utility_frame = tk.LabelFrame(self.app.export_panel, text="Utilities", font=self.app.bold_font)
        utility_frame.pack(fill=tk.X, padx=5, pady=(2, 5))

        self._setup_action_buttons(compute_parent=compute_frame,
                                   label_parent=label_frame,
                                   utility_parent=utility_frame)

        # ---- MARKER EVENTS SECTION ----
        marker_frame = tk.LabelFrame(self.app.export_panel, text="Event Markers", font=self.app.bold_font)
        marker_frame.pack(fill=tk.X, padx=5, pady=5)

        marker_list_frame = tk.Frame(marker_frame)
        marker_list_frame.pack(fill=tk.BOTH, expand=True)

        marker_scrollbar = tk.Scrollbar(marker_list_frame, orient="vertical")
        self.app.marker_listbox = tk.Listbox(
            marker_list_frame,
            selectmode=tk.MULTIPLE,
            height=5,
            yscrollcommand=marker_scrollbar.set,
            exportselection=False
        )
        marker_scrollbar.config(command=self.app.marker_listbox.yview)
        self.app.marker_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        marker_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.app.marker_listbox.bind('<<ListboxSelect>>', self.on_marker_select)

        # ---- EXPORT CHANNELS SECTION ----
        export_frame = tk.LabelFrame(
            self.app.export_panel,
            text="Channels to Export",
            font=self.app.bold_font
        )
        export_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        export_list_frame = tk.Frame(export_frame)
        export_list_frame.pack(fill=tk.BOTH, expand=True)

        self.app.export_scrollbar = tk.Scrollbar(export_list_frame, orient="vertical")
        self.app.export_listbox = tk.Listbox(
            export_list_frame,
            selectmode=tk.MULTIPLE,
            width=40,
            height=20,
            yscrollcommand=self.app.export_scrollbar.set,
            exportselection=False
        )
        self.app.export_scrollbar.config(command=self.app.export_listbox.yview)
        self.app.export_listbox.bind("<<ListboxSelect>>", self.on_export_select)
        self.app.export_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.app.export_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _setup_action_buttons(self, compute_parent, label_parent, utility_parent):
        """Create analysis and utility buttons grouped by function."""
        groups = [
            (compute_parent, [
                ("Gaze Metrics (ρ,θ,φ)", self.app.calculate_gaze_metrics, "gaze_metrics_btn"),
                ("Angular Velocity", self.app.calculate_angular_velocity, "angular_velocity_btn"),
                ("Foveal Visual Radius", self.app.calculate_fvr, "fvr_btn"),
            ]),
            (label_parent, [
                ("Label Events", self.app.label_gaze, "label_gaze_btn"),
            ]),
            (utility_parent, [
                ("Show Parameters", self.app.show_task_protocol, "task_protocol_btn"),
                ("Clear Cache", self.app.clear_cache_dialog, "clear_cache_btn"),
                ("Help", self.app.show_help, "help_btn"),
            ]),
        ]

        for parent, actions in groups:
            for text, cmd, attr_name in actions:
                btn = tk.Button(parent, text=text, command=cmd, width=45)
                btn.pack(side=tk.TOP, pady=2, anchor="w")
                setattr(self.app, attr_name, btn)

    def on_export_select(self, event=None):
        """Remember selected export channels (sticky) and persist to disk."""
        if getattr(self.app, "_populating_export", False):
            return
        try:
            visible = {self.app.export_listbox.get(i) for i in range(self.app.export_listbox.size())}
            selected = {self.app.export_listbox.get(i) for i in self.app.export_listbox.curselection()}
            self.app._sticky_export_selection.difference_update(visible)
            self.app._sticky_export_selection.update(selected)
            set_export_defaults(sorted(self.app._sticky_export_selection))
        except Exception:
            pass

    def on_marker_select(self, event=None):
        """Remember selected marker events (sticky) and persist to disk."""
        if getattr(self.app, "_populating_export", False):
            return
        try:
            selected = {self.app.marker_listbox.get(i) for i in self.app.marker_listbox.curselection()}
            self.app._sticky_marker_selection = set(selected)
            set_marker_defaults(sorted(selected))
        except Exception:
            pass

    def restore_sticky_export_selection(self):
        """Apply sticky selections to the export listbox."""
        if not self.app._sticky_export_selection:
            return
        self.app._populating_export = True
        try:
            self.app.export_listbox.selection_clear(0, tk.END)
            for i in range(self.app.export_listbox.size()):
                if self.app.export_listbox.get(i) in self.app._sticky_export_selection:
                    self.app.export_listbox.selection_set(i)
        finally:
            self.app._populating_export = False