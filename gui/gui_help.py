"""
GUI Help Components

Provides tooltip and help window widgets for the KINARM Data Explorer.

Components:
- ToolTip: Hover tooltip that appears after a short delay on any tkinter widget.
- HelpWindow: Scrollable popup window displaying application documentation,
  workflow guidance, and dynamically generated path information.
"""

import tkinter as tk
from tkinter import ttk

class ToolTip:
    """
    Hover tooltip for any tkinter widget.

    Displays a small popup with explanatory text after the cursor hovers over
    the bound widget for ``delay_ms`` milliseconds. The tooltip is dismissed
    on mouse leave or button press.

    Parameters
    ----------
    widget : tk.Widget
        The widget to attach the tooltip to.
    text : str
        Tooltip text to display.
    delay_ms : int
        Milliseconds to wait before showing the tooltip.
    wraplength : int
        Maximum pixel width before text wraps.
    """
    def __init__(self, widget, text, *, delay_ms=450, wraplength=360):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.wraplength = wraplength
        self._after_id = None
        self._tipwin = None

        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self):
        if self._after_id:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self):
        if self._tipwin or not self.text:
            return

        # Position near mouse pointer
        x = self.widget.winfo_pointerx() + 12
        y = self.widget.winfo_pointery() + 14

        self._tipwin = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        # On some platforms, forcing topmost helps
        try:
            tw.attributes("-topmost", True)
        except Exception:
            pass

        frame = tk.Frame(tw, background="#ffffe0", borderwidth=1, relief="solid")
        frame.pack(fill="both", expand=True)

        label = tk.Label(
            frame,
            text=self.text,
            justify="left",
            background="#ffffe0",
            foreground="black",
            wraplength=self.wraplength,
            font=("Segoe UI", 10)
        )
        label.pack(padx=8, pady=6)

    def _hide(self, _=None):
        self._cancel()
        if self._tipwin:
            try:
                self._tipwin.destroy()
            except Exception:
                pass
            self._tipwin = None


def attach_tooltip(widget, text):
    """
    Convenience wrapper to create and bind a ToolTip to a widget.

    The returned ToolTip object is not stored; it persists because tkinter
    keeps a reference through the bound event handlers.
    """
    ToolTip(widget, text)


# ---- Help window ----

class HelpWindow:
    """
    Scrollable popup window displaying application help text.

    The window contains static workflow documentation and optionally appends
    dynamic text (e.g., current file paths) via a callback.

    Parameters
    ----------
    parent : tk.Widget
        Parent widget for the Toplevel window.
    get_dynamic_text : callable or None
        Optional callback returning a string to append after the static help
        content. Called each time the user clicks Refresh.
    """
    def __init__(self, parent, *, get_dynamic_text=None):
        self.parent = parent
        self.get_dynamic_text = get_dynamic_text
        self.win = None

    def show(self):
        """Open the help window, or raise it if already open."""
        if self.win and self.win.winfo_exists():
            self.win.lift()
            self.win.focus_force()
            return

        self.win = tk.Toplevel(self.parent)
        self.win.title("KINARM Data Explorer - Help")
        self.win.resizable(True, True)

        # Layout
        top = tk.Frame(self.win)
        top.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        title = tk.Label(top, text="Help / Notes", font=("Segoe UI", 12, "bold"))
        title.pack(anchor="w", pady=(0, 8))

        text_frame = tk.Frame(top)
        text_frame.pack(fill=tk.BOTH, expand=True)

        yscroll = tk.Scrollbar(text_frame, orient="vertical")
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.text = tk.Text(
            text_frame,
            wrap=tk.WORD,
            yscrollcommand=yscroll.set,
            font=("Segoe UI", 10)
        )
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yscroll.config(command=self.text.yview)

        btn_row = tk.Frame(self.win)
        btn_row.pack(fill=tk.X, padx=10, pady=(0, 10))

        tk.Button(btn_row, text="Refresh", command=self.refresh, width=12).pack(side=tk.LEFT)
        tk.Button(btn_row, text="Close", command=self.win.destroy, width=12).pack(side=tk.RIGHT)

        self.refresh()

        # Size/center
        self.win.update_idletasks()
        w, h = 760, 520
        x = (self.win.winfo_screenwidth() // 2) - (w // 2)
        y = (self.win.winfo_screenheight() // 2) - (h // 2)
        self.win.geometry(f"{w}x{h}+{max(0,x)}+{max(0,y)}")

    def refresh(self):
        """Reload help text content, including dynamic path information."""
        self.text.config(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)

        base = (
            "Recommended Workflow (Read This First)\n"
            "• BEFORE loading a file, set your Save Location. The folder you select will be the parent folder where all subsequent .kinarm folders will be saved to and read from.\n"
            "• Load a .kinarm file.\n"
            "• Select a trial.\n"
            "• Select desired event markers (vertical lines shown in the labeler) and channels to export (.csv file that gets exported will include these channels as columns).\n"
            "• Inspect channels / Label gaze events (interpolation will be offered if large gaps are detected).\n"
            "• Mark trials / add notes as needed.\n\n"

            "IMPORTANT – Save Location Behavior\n"
            "• Trial notes are written to Trial_Notes.csv in the output folder for the loaded file.\n"
            "• The CSV file is created only after at least one trial is marked or noted.\n"
            "• Session state (selected trial, filters, export selections, marker selections) "
            "is saved to session_state.json in the same folder.\n"
            "• If you change the Save Location later, the program will read/write from the new folder.\n"
            "• If previous notes/session files exist in a different location, they will not load "
            "unless you switch back to that location or manually move the folder.\n"
            "• For collaborative work, ensure all users use the same Save Location.\n\n"

            "Resume Behavior\n"
            "• If session_state.json exists for the loaded file, you will be prompted to resume session when you load that file.\n"
            "• Resuming restores trial selection, channel filter, export selections, and marker selections.\n"
            "• Resume does NOT change trial marks or notes (those are stored separately in Trial_Notes.csv).\n\n"

            "Buttons (In UI Order)\n"
            "• Load .kinarm File: Opens a KINARM file and populates trials.\n"
            "• Set Save Location: Choose where trial notes and session files are stored.\n"
            "• Restore Default Save Location: Resets to Desktop/gaze_labels.\n"
            "• Gaze Metrics (ρ, θ, φ): Computes spherical gaze coordinates.\n"
            "• Angular Velocity: Computes angular velocity.\n"
            "• Foveal Visual Radius: Computes foveal visual radius.\n"
            "• Label Events: Launches interactive labeling tool.\n"
            "• Show Parameters: Opens Task Protocol and Target tables for the file.\n"
            "• Clear Cache: Offers ability to clear stored data such as interpolation selections, labelling order, and saved session states. Resetting one of these will prompt the respective option when necessary, like the first time it was set.\n\n"

            "Configuration vs Data\n"
            "• user_prefs.json (per-user): Stores personal application settings such as save location "
            "and default selections.\n"
            "• session_state.json (per .kinarm file): Stores resume information for that dataset. Stored in the .kinarm subfolder of the selected save location.\n"
            "• Trial_Notes.csv (per .kinarm file): Stores trial marks and notes. Stored in the .kinarm subfolder of the selected save location.\n\n"

            "Gaze Event Export Codes\n"
            "The gaze_event column in exported CSV files uses numeric codes:\n"
            "• 0 = Unlabeled/Other (default for frames not labeled as fixation, pursuit, or saccade)\n"
            "• 1 = Saccade (rapid eye movement)\n"
            "• 2 = Pursuit (smooth tracking of moving target)\n"
            "• 3 = Fixation (steady gaze on a target)\n"
            "• 9 = Bad Trial (trial marked as unusable - all frames exported as code 9)\n\n"

            "Notes on File Creation\n"
            "• A subfolder named after the .kinarm file is created in the Save Location.\n"
            "• Trial_Notes.csv is created only after a trial is marked or noted.\n"
            "• session_state.json is created after trial selection or export selections are saved.\n"
            "• If nothing is marked or selected, the folder may exist but remain empty.\n\n"
        )

        extra = ""
        if callable(self.get_dynamic_text):
            try:
                extra = self.get_dynamic_text()
            except Exception:
                extra = ""

        self.text.insert(tk.END, base)
        if extra:
            self.text.insert(tk.END, "\n" + extra.strip() + "\n")

        self.text.config(state=tk.DISABLED)