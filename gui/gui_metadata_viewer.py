"""
gui_metadata_viewer.py

Task Protocol Viewer window (popup).

Displays KINARM/Dexterit-E experiment configuration tables in a tabbed UI:
- TARGET_TABLE: workspace targets (position, radii, colors)
- TP_TABLE: task protocol rows (timing, start/end targets, etc.)
- USER_CHANNELS: "Analog Inputs" metadata (ACH label/units/scale/offset/etc.)

This window is read-only: it is meant for quick inspection while the user
browses trials in the main GUI.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import pandas as pd

from utility.kinarm_utils import (
    extract_group,
    find_trial_tp_number,
    infer_tp_rows_from_start,
    infer_used_rows,
    normalize_table,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

DEFAULT_COL_WIDTH_PX = 120


def df_to_tree(tree: ttk.Treeview, df: pd.DataFrame, col_width: int = DEFAULT_COL_WIDTH_PX) -> None:
    """
    Populate a ttk.Treeview widget with a pandas DataFrame.

    Notes
    -----
    - Clears any existing content first.
    - Uses "headings" mode (no first blank column).
    - Converts all cell values to strings for display.

    Parameters
    ----------
    tree : ttk.Treeview
        The treeview widget to populate.
    df : pd.DataFrame
        Data to show.
    col_width : int
        Default column width in pixels.
    """
    # Clear existing rows
    for item in tree.get_children():
        tree.delete(item)

    if df is None or df.empty:
        # Keep the tree empty; nothing to show.
        tree["columns"] = []
        tree["show"] = "headings"
        return

    cols = list(df.columns)
    tree["columns"] = cols
    tree["show"] = "headings"

    # Configure columns + headings
    for col in cols:
        tree.heading(col, text=str(col))
        tree.column(col, width=col_width, anchor="center")

    # Insert rows
    for _, row in df.iterrows():
        tree.insert("", "end", values=[str(val) for val in row.values])


# -----------------------------------------------------------------------------
# Main window
# -----------------------------------------------------------------------------

class TaskProtocolWindow:
    """
    Popup window for displaying Task Protocol and Target tables.

    The main GUI calls:
    - show() to open/raise the window
    - update_selected_trial(...) when the user picks a different trial
    """

    def __init__(self, parent: tk.Tk | tk.Toplevel, exam, current_trial_name: str | None = None):
        self.parent = parent
        self.exam = exam
        self.current_trial_name = current_trial_name

        self.window: tk.Toplevel | None = None
        self.header_label: tk.Label | None = None

        self.tgt_tree: ttk.Treeview | None = None
        self.tp_tree: ttk.Treeview | None = None
        self.analog_tree: ttk.Treeview | None = None

    # -------------------------
    # Data building
    # -------------------------

    def _build_tables(self) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """
        Extract and format TARGET_TABLE and TP_TABLE from the "common" trial.

        Returns
        -------
        (tgt_df, tp_df) or (None, None) if the common trial is missing.
        """
        common = self.exam.trials.get("common")
        if not common:
            return None, None

        # ----------------
        # Target Table
        # ----------------
        tgt_cols = extract_group(common, "TARGET_TABLE")
        tgt_rows = infer_used_rows(tgt_cols)

        desired_tgt_cols = [
            ("X", "X"),
            ("Y", "Y"),
            ("Visual Radius", "Visual Radius"),
            ("Logical Radius", "Logical Radius"),
            ("Initial Color", "Initial Color"),
            ("Reached Color", "Reached Color"),
            ("Gaze Radius", "Gaze Radius"),
        ]
        tgt_df = normalize_table(tgt_cols, desired_tgt_cols, tgt_rows, left_label_prefix="Target")

        # ----------------
        # Task Protocol (TP) Table
        # ----------------
        tp_cols = extract_group(common, "TP_TABLE")
        desired_tp_cols = [
            ("Start Target", "Start Target"),
            ("End Target", "End Target"),
            ("Load", "Load"),
            ("Hand Feedback Delay", "Hand Feedback Delay"),
            ("1st Target Delay (Fixed)", "1st Target Delay (Fixed)"),
            ("1st Target Delay (Random)", "1st Target Delay (Random)"),
            ("Gap Duration Boolean", "Gap Duration Boolean"),
            ("Max Reach Time", "Max Reach Time"),
            ("2nd Target Delay", "2nd Target Delay"),
            ("Post-Trial Delay", "Post-Trial Delay"),
            ("Target Speed", "Target Speed"),
            ("Target Radius", "Target Radius"),
        ]
        tp_rows = infer_tp_rows_from_start(tp_cols, [c[1] for c in desired_tp_cols])
        tp_df = normalize_table(tp_cols, desired_tp_cols, tp_rows, left_label_prefix="TP")

        return tgt_df, tp_df

    def _build_user_channels_table(self) -> pd.DataFrame:
        """
        Build the USER_CHANNELS table (Dexterit-E analog input metadata).

        Returns
        -------
        pd.DataFrame
            Columns: ACH, Label, Units, Scale, Offset, Description
        """
        common = self.exam.trials.get("common")
        if not common:
            return pd.DataFrame()

        uc = extract_group(common, "USER_CHANNELS")

        labels = uc.get("LABEL", [])
        units = uc.get("UNITS", [])
        scales = uc.get("SCALE", [])
        offsets = uc.get("OFFSET", [])
        descriptions = uc.get("DESCRIPTION", [])

        count = len(labels)
        if count == 0:
            return pd.DataFrame()

        def safe(lst, i, default=""):
            """Index into lst with a fallback default if out of range or error."""
            try:
                return lst[i]
            except Exception:
                return default

        rows = []
        for i in range(count):
            rows.append(
                {
                    "ACH": f"ACH {i}",
                    "Label": safe(labels, i),
                    "Units": safe(units, i),
                    "Scale": safe(scales, i, 1.0),
                    "Offset": safe(offsets, i, 0.0),
                    "Description": safe(descriptions, i),
                }
            )

        return pd.DataFrame(rows)

    # -------------------------
    # UI updates
    # -------------------------

    def _rebuild_trees(self) -> None:
        """Recompute dataframes and refresh all three treeviews."""
        if not (self.tgt_tree and self.tp_tree and self.analog_tree):
            return

        tgt_df, tp_df = self._build_tables()
        analog_df = self._build_user_channels_table()

        if tgt_df is None or tp_df is None:
            return

        df_to_tree(self.tgt_tree, tgt_df)
        df_to_tree(self.tp_tree, tp_df)
        df_to_tree(self.analog_tree, analog_df)

    def update_selected_trial(self, trial_name: str, trial) -> None:
        """
        Update the header to show the currently selected trial + its TP number.
        """
        self.current_trial_name = trial_name
        tp_num = find_trial_tp_number(trial)

        if self.header_label:
            if tp_num is not None:
                self.header_label.config(
                    text=f"Task Protocol   •   Selected trial: {trial_name}   •   TP #: {tp_num}"
                )
            else:
                self.header_label.config(
                    text=f"Task Protocol   •   Selected trial: {trial_name}   •   TP #: (unknown)"
                )

    # -------------------------
    # Window lifecycle
    # -------------------------

    def show(self) -> None:
        """
        Create and display the Task Protocol window, or raise it if already open.

        Only one instance is shown at a time. If the window already exists and
        is visible, it is brought to the front instead of creating a duplicate.
        """
        if self.window and self.window.winfo_exists():
            self.window.lift()
            self.window.focus_force()
            return

        self.window = tk.Toplevel(self.parent)
        self.window.title("Task Protocol Viewer")
        self.window.geometry("1000x600")

        # Header (we keep this lightweight; main GUI can update with TP# later)
        header_text = "Task Protocol"
        if self.current_trial_name:
            header_text += f"   •   Selected trial: {self.current_trial_name}"

        self.header_label = tk.Label(
            self.window,
            text=header_text,
            font=("Segoe UI", 12, "bold"),
            bg="lightblue",
            pady=10,
        )
        self.header_label.pack(fill=tk.X)

        notebook = ttk.Notebook(self.window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # -------- Target Table tab --------
        tgt_frame = ttk.Frame(notebook)
        notebook.add(tgt_frame, text="Target Table")

        tgt_scroll_y = ttk.Scrollbar(tgt_frame, orient="vertical")
        tgt_scroll_x = ttk.Scrollbar(tgt_frame, orient="horizontal")

        self.tgt_tree = ttk.Treeview(
            tgt_frame,
            style="Big.Treeview",
            yscrollcommand=tgt_scroll_y.set,
            xscrollcommand=tgt_scroll_x.set,
        )

        tgt_scroll_y.config(command=self.tgt_tree.yview)
        tgt_scroll_x.config(command=self.tgt_tree.xview)

        self.tgt_tree.grid(row=0, column=0, sticky="nsew")
        tgt_scroll_y.grid(row=0, column=1, sticky="ns")
        tgt_scroll_x.grid(row=1, column=0, sticky="ew")

        tgt_frame.grid_rowconfigure(0, weight=1)
        tgt_frame.grid_columnconfigure(0, weight=1)

        # -------- TP Table tab --------
        tp_frame = ttk.Frame(notebook)
        notebook.add(tp_frame, text="TP Table")

        tp_scroll_y = ttk.Scrollbar(tp_frame, orient="vertical")
        tp_scroll_x = ttk.Scrollbar(tp_frame, orient="horizontal")

        self.tp_tree = ttk.Treeview(
            tp_frame,
            style="Big.Treeview",
            yscrollcommand=tp_scroll_y.set,
            xscrollcommand=tp_scroll_x.set,
        )

        tp_scroll_y.config(command=self.tp_tree.yview)
        tp_scroll_x.config(command=self.tp_tree.xview)

        self.tp_tree.grid(row=0, column=0, sticky="nsew")
        tp_scroll_y.grid(row=0, column=1, sticky="ns")
        tp_scroll_x.grid(row=1, column=0, sticky="ew")

        tp_frame.grid_rowconfigure(0, weight=1)
        tp_frame.grid_columnconfigure(0, weight=1)

        # -------- Analog Inputs tab --------
        analog_frame = ttk.Frame(notebook)
        notebook.add(analog_frame, text="Analog Inputs")

        analog_scroll_y = ttk.Scrollbar(analog_frame, orient="vertical")
        analog_scroll_x = ttk.Scrollbar(analog_frame, orient="horizontal")

        self.analog_tree = ttk.Treeview(
            analog_frame,
            style="Big.Treeview",
            yscrollcommand=analog_scroll_y.set,
            xscrollcommand=analog_scroll_x.set,
        )

        analog_scroll_y.config(command=self.analog_tree.yview)
        analog_scroll_x.config(command=self.analog_tree.xview)

        self.analog_tree.grid(row=0, column=0, sticky="nsew")
        analog_scroll_y.grid(row=0, column=1, sticky="ns")
        analog_scroll_x.grid(row=1, column=0, sticky="ew")

        analog_frame.grid_rowconfigure(0, weight=1)
        analog_frame.grid_columnconfigure(0, weight=1)

        # Populate data
        self._rebuild_trees()

        # Center after drawing
        self.window.update_idletasks()
        w = self.window.winfo_width()
        h = self.window.winfo_height()
        screen_w = self.window.winfo_screenwidth()
        screen_h = self.window.winfo_screenheight()
        x = (screen_w // 2) - (w // 2)
        y = (screen_h // 2) - (h // 2)
        self.window.geometry(f"{w}x{h}+{x}+{y}")