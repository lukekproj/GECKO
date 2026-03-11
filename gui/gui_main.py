"""
KINARM Data Explorer - Main GUI Application

This is the primary interface for loading KINARM data files, selecting trials,
inspecting channels, and performing gaze analysis.

Dependencies: tkinter, KinarmDataExplorer, matplotlib, numpy, pandas
"""
# TODO: This file exceeds the length I would like (~1729 lines)
# I plan to split into gui_trial_panel.py, gui_channel_panel.py, gui_export_panel.py, and gui_compute.py post-publication.
from pathlib import Path
import sys

# Add src directory to Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

# Imports
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from data.data_loader import KinarmDataExplorer
from gui.gui_help import HelpWindow
from gui.gui_session import SessionManager
from utility.user_prefs import get_label_order, set_label_order, get_export_defaults, get_marker_defaults, set_marker_defaults, get_save_location, set_save_location, MAX_LABELER_CHANNELS, get_labeler_channel_defaults, set_labeler_channel_defaults
import numpy as np
from data.data_loader import KinarmDataExplorer, DerivedChannel

import matplotlib 
from utility.user_prefs import MPL_BACKEND
matplotlib.use(MPL_BACKEND)
import matplotlib.pyplot as plt

from gui.gui_metadata_viewer import TaskProtocolWindow
from utility.kinarm_utils import find_trial_tp_number
from gui.gui_utils import center_window, configure_big_treeview_style, log_crash
sys.excepthook = log_crash

class KinarmDataExplorerGUI:
    """
    Main GUI application for KINARM data exploration and analysis.
    
    This provides the primary interface for loading KINARM data files,
    selecting trials, inspecting channels, and performing various analyses
    including gaze labeling and motion calculations.
    """
    
    def __init__(self, root):
        self.root = root
        configure_big_treeview_style()

        # Set cross-platform fonts
        import platform
        if platform.system() == "Windows":
            self.default_font = ("Segoe UI", 12)
            self.bold_font = ("Segoe UI", 12, "bold")
        else:  # Mac/Linux
            self.default_font = ("Helvetica", 12)
            self.bold_font = ("Helvetica", 12, "bold")

        self.root.title("Data Explorer GUI")
        self.root.withdraw()  # Hide window initially
        self.root.geometry("1200x900")
        
        try:
            self.root.state('zoomed')       # Windows/macOS
        except tk.TclError:
            self.root.attributes('-zoomed', True)  # Linux

        self.root.deiconify() # Show window after centering

        # Core application state
        self.explorer = None                        # KinarmDataExplorer instance
        self.current_trial = None                   # Currently selected trial
        self.taskproto_win = None                   # Task Protocol window reference
        self._sticky_export_selection = set(get_export_defaults())       # Remember selected export channels
        from utility.user_prefs import get_marker_defaults
        self._sticky_marker_selection = set(get_marker_defaults())
        self._populating_export = False
        self._current_file_id = None                # Track which .kinarm file is loaded
        self._populating = False                    # Flag to prevent callback loops
        self._open_figures = []
        self._trial_marks = {}  # Store trial quality marks
        self.bad_trials = set()
        self.available_events = []        
        self._event_channel_map = {}
        self.custom_save_location = get_save_location()
        self._sticky_channel_selection = set()
        self._populating_channels = False
        self.session = SessionManager(
            get_explorer=lambda: self.explorer,
            get_save_location=lambda: self.custom_save_location,
        )

        self._all_channels = []
        self._all_export_channels = []
        self._channel_filter_var = tk.StringVar()
        self._channel_filter_var.trace_add("write", lambda *_: self._apply_channel_filter())
        self._setup_gui()
        if hasattr(self, "channel_listbox"):
            self.channel_listbox.bind('<<ListboxSelect>>', self._on_channel_select)
        self.help_win = HelpWindow(self.root, get_dynamic_text=self._help_dynamic_text)

        self._update_button_states()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _help_dynamic_text(self):
        """Return current file/output paths for display in the Help window."""
        kinarm = Path(self.explorer.filepath) if self.explorer else None
        save_root = Path(self.custom_save_location) if self.custom_save_location else (Path.home() / "Desktop" / "gaze_labels")
        notes = self._get_notes_csv_path()
        state = self._session_state_path()
        
        # Get user_prefs.json location - construct path directly
        prefs_path = Path.home() / ".config" / "KinarmDataExplorer" / "user_prefs.json"

        return (
            "Current Paths (Refresh once file is loaded)\n"
            f"• Loaded file: {kinarm.as_posix() if kinarm else None}\n"
            f"• Output root: {save_root.as_posix()}\n"
            f"• Notes CSV: {notes.as_posix() if notes else None}\n"
            f"• Session JSON: {state.as_posix() if state else None}\n"
            f"• User preferences: {prefs_path.as_posix()}\n"
        )

    def show_help(self):
        self.help_win.show()

    def _session_state_path(self) -> Path | None:
        """Delegate to SessionManager."""
        return self.session.session_state_path()

    def _load_session_state(self) -> dict | None:
        """Delegate to SessionManager."""
        return self.session.load_state()
        
    def _resume_from_state(self, state: dict):
        """
        Restore GUI state from a previously saved session dictionary.

        Restores trial selection, channel/export/marker sticky selections,
        and the channel filter string.
        """
        # restore sticky selections first
        self._sticky_export_selection = set(state.get("export_selection", []))
        self._sticky_marker_selection = set(state.get("marker_selection", []))
        self._sticky_channel_selection = set(state.get("inspect_selection", []))

        # select trial
        trial_name = state.get("trial_name")
        if trial_name in self.explorer.trial_names:
            idx = self.explorer.trial_names.index(trial_name)
            self.trial_listbox.selection_clear(0, tk.END)
            self.trial_listbox.selection_set(idx)
            self.trial_listbox.see(idx)
            self.select_trial()  # refresh lists

        # restore filter AFTER select_trial repopulates channels
        filt = state.get("channel_filter", "")
        self._channel_filter_var.set(filt)
        self._apply_channel_filter()

        # re-apply export + marker selections into listboxes
        self._restore_sticky_export_selection()
        for i in range(self.marker_listbox.size()):
            if self.marker_listbox.get(i) in self._sticky_marker_selection:
                self.marker_listbox.selection_set(i)

    
    def _get_notes_csv_path(self):
        """Delegate to SessionManager."""
        return self.session.notes_csv_path()
    
    def _parse_channel_item(self, item: str) -> str:
        """Extract the channel name from a numbered listbox entry (e.g., '12. Right_HandX' -> 'Right_HandX')."""
        # Listbox items look like "12. Right_HandX"
        return item.split(". ", 1)[1] if ". " in item else item

    def _on_channel_select(self, event=None):
        """Persist channel selections across filtering by tracking by name."""
        if getattr(self, "_populating_channels", False):
            return

        # What is currently visible in the listbox
        visible = {
            self._parse_channel_item(self.channel_listbox.get(i))
            for i in range(self.channel_listbox.size())
        }

        # What is selected among visible entries
        selected = {
            self._parse_channel_item(self.channel_listbox.get(i))
            for i in self.channel_listbox.curselection()
        }

        # Remove any visible channels from the sticky set, then add back selected ones.
        # This makes deselection work correctly.
        self._sticky_channel_selection.difference_update(visible)
        self._sticky_channel_selection.update(selected)
        self.session.save_state(
            current_trial_name=self.current_trial.name if self.current_trial else None,
            trial_names=self.explorer.trial_names if self.explorer else [],
            filepath=self.explorer.filepath if self.explorer else "",
            channel_filter=self._channel_filter_var.get(),
            inspect_selection=self._sticky_channel_selection,
            export_selection=self._sticky_export_selection,
            marker_selection=self._sticky_marker_selection,
        )


    def _load_trial_marks(self):
        """Load trial marks from CSV via SessionManager."""
        self.session.load_trial_marks()
        self._trial_marks = self.session.trial_marks

    def _restore_sticky_export_selection(self):
        """Apply sticky selections to the export listbox."""
        if not self._sticky_export_selection:
            return

        self._populating_export = True

        try:
            # Clear current selection first
            self.export_listbox.selection_clear(0, tk.END)

            # Re-select anything that exists in this listbox
            for i in range(self.export_listbox.size()):
                name = self.export_listbox.get(i)
                if name in self._sticky_export_selection:
                    self.export_listbox.selection_set(i)
        finally:
            self._populating_export = False


    def _save_trial_marks(self):
        """Save trial marks to CSV via SessionManager."""
        self.session.trial_marks = self._trial_marks
        self.session.save_trial_marks(self.explorer.trial_names if self.explorer else [])
            
    def on_closing(self):
        """Cleanup when main window closes to prevent memory leaks."""
        try:
            # Close all matplotlib figures
            plt.close('all')
            
            # Close task protocol window if open
            if self.taskproto_win and hasattr(self.taskproto_win, 'window'):
                try:
                    if self.taskproto_win.window:
                        self.taskproto_win.window.destroy()
                except Exception:
                    pass
            
            # Clear any remaining references
            self._open_figures.clear()
            
        except Exception as e:
            print(f"Cleanup error: {e}")
        finally:
            self.root.destroy()

    def _setup_gui(self):
        """Create and arrange all GUI components."""
        # Create main layout frames
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._setup_file_controls(top_frame)
        self._setup_main_panels(bottom_frame)

    def restore_default_save_location(self):
        """Restore the default save location (Desktop/gaze_labels)."""
        self.custom_save_location = None
        
        # Clear from preferences file
        from utility.user_prefs import set_save_location
        set_save_location(None)
        
        # Update status display
        default_loc = str(Path.home() / "Desktop" / "gaze_labels")
        self.status_var.set(f"Selected Save Location: {default_loc} (Default)")
        
        messagebox.showinfo(
            "Default Location Restored",
            f"Save location has been restored to default:\n{default_loc}\n\n"
            "Gaze labels will be saved here unless you set a custom location again."
        )

    def set_save_location(self):
        """Allow user to select custom save location for gaze labels."""
        directory = filedialog.askdirectory(
            title="Select Save Location for Gaze Labels",
            initialdir=self.custom_save_location or str(Path.home() / "Desktop")
        )
        
        if directory:
            self.custom_save_location = directory
            # Save to preferences file
            from utility.user_prefs import set_save_location
            set_save_location(directory)
            
            self.status_var.set(f"Selected Save Location: {directory}")
            messagebox.showinfo(
                "Save Location Updated", 
                f"Gaze labels will now be saved to:\n{directory}\n\n"
                "A subfolder will be created for each .kinarm file.\n\n"
                "This location will be remembered next time you open the program."
            )
        
        if self.current_trial:
            # Refresh with new save location
            self.select_trial()

    def _save_session_state(self):
        """Delegate session persistence to SessionManager."""
        self.session.save_state(
            current_trial_name=self.current_trial.name if self.current_trial else None,
            trial_names=self.explorer.trial_names if self.explorer else [],
            filepath=self.explorer.filepath if self.explorer else "",
            channel_filter=self._channel_filter_var.get(),
            inspect_selection=self._sticky_channel_selection,
            export_selection=self._sticky_export_selection,
            marker_selection=self._sticky_marker_selection,
        )

    def _setup_file_controls(self, parent):
        """Create file loading and trial selection controls."""
        # ---- File + Output group (only these 3 buttons) ----
        file_frame = tk.LabelFrame(parent, text="File Management", font=self.bold_font)
        file_frame.pack(pady=(8, 4), padx=6)

        file_btn_row = tk.Frame(file_frame)
        file_btn_row.pack(pady=6)

        self.file_button = tk.Button(file_btn_row, text="Load .kinarm File", command=self.load_file)
        self.file_button.pack(side=tk.LEFT, padx=5)

        self.save_location_button = tk.Button(file_btn_row, text="Set Save Location", command=self.set_save_location)
        self.save_location_button.pack(side=tk.LEFT, padx=5)

        self.restore_default_button = tk.Button(
            file_btn_row,
            text="Restore Default Save Location",
            command=self.restore_default_save_location
        )
        self.restore_default_button.pack(side=tk.LEFT, padx=5)
        
        # Trial marking buttons
        mark_frame = tk.Frame(parent)
        mark_frame.pack(pady=5)
        
        self.mark_good_btn = tk.Button(mark_frame, text="Mark Good", command=lambda: self.mark_trial("good"), bg="#90EE90")
        self.mark_good_btn.pack(side=tk.LEFT, padx=2)
        
        self.mark_bad_btn = tk.Button(mark_frame, text="Mark Bad", command=lambda: self.mark_trial("bad"), bg="#FFB6C6")
        self.mark_bad_btn.pack(side=tk.LEFT, padx=2)
        
        self.mark_review_btn = tk.Button(mark_frame, text="Mark Review", command=lambda: self.mark_trial("review"), bg="#FFE4B5")
        self.mark_review_btn.pack(side=tk.LEFT, padx=2)
        
        self.clear_mark_btn = tk.Button(mark_frame, text="Clear Mark", command=lambda: self.mark_trial(None))
        self.clear_mark_btn.pack(side=tk.LEFT, padx=2)

        self.notes_btn = tk.Button(mark_frame, text="Add Notes", command=self.add_trial_notes, bg="#E0E0E0")
        self.notes_btn.pack(side=tk.LEFT, padx=2)

        # Trial list with scrollbar
        trial_frame = tk.Frame(parent)
        trial_frame.pack(pady=10)

        trial_scrollbar = tk.Scrollbar(trial_frame, orient="vertical")
        trial_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.trial_listbox = tk.Listbox(
            trial_frame, 
            width=50, 
            height=6, 
            exportselection=False,
            yscrollcommand=trial_scrollbar.set
        )
        self.trial_listbox.pack(side=tk.LEFT)
        trial_scrollbar.config(command=self.trial_listbox.yview)

        self.trial_listbox.bind("<<ListboxSelect>>", self.select_trial)
        
        # Selected trial info display
        self.selected_trial_var = tk.StringVar(value="Selected Trial: (none)")
        self.selected_trial_label = tk.Label(
            parent,
            textvariable=self.selected_trial_var,
            font=self.bold_font,
            anchor="w"
        )
        self.selected_trial_label.pack(fill=tk.X, padx=6, pady=(0, 8))

        self.status_var = tk.StringVar()
        save_loc = self.custom_save_location if self.custom_save_location else str(Path.home() / "Desktop" / "gaze_labels (Default)")
        self.status_var.set(f"Selected Save Location: {save_loc}")
        self.status_label = tk.Label(
            parent,
            textvariable=self.status_var,
            font=self.bold_font,  # Match first bar
            anchor="w"  # Remove bg, relief, padx, pady
        )
        self.status_label.pack(fill=tk.X, padx=6, pady=(0, 8))

    def mark_trial(self, status):
        """Mark the currently selected trial with a quality status."""
        sel = self.trial_listbox.curselection()
        if not sel:
            messagebox.showwarning("No Selection", "Please select a trial first.")
            return
        
        index = sel[0]
        trial_name = self.explorer.trial_names[index]
        
        # Preserve existing notes if they exist
        existing_notes = ""
        if trial_name in self._trial_marks:
            mark_data = self._trial_marks[trial_name]
            if isinstance(mark_data, dict):
                existing_notes = mark_data.get("notes", "")
        
        # Update marks dictionary
        if status is None and not existing_notes:
            # Only remove if no notes either
            self._trial_marks.pop(trial_name, None)
        else:
            # Store as dictionary to include both mark and notes
            self._trial_marks[trial_name] = {
                "mark": status,
                "notes": existing_notes
            }
        
        # Save to file
        self._save_trial_marks()
        
        # Refresh display
        self._refresh_trial_list()

    def add_trial_notes(self):
        """Open dialog to add/edit notes for the currently selected trial."""
        sel = self.trial_listbox.curselection()
        if not sel:
            messagebox.showwarning("No Selection", "Please select a trial first.")
            return
        
        index = sel[0]
        trial_name = self.explorer.trial_names[index]
        
        # Get existing notes
        existing_notes = ""
        if trial_name in self._trial_marks:
            mark_data = self._trial_marks[trial_name]
            if isinstance(mark_data, dict):
                existing_notes = mark_data.get("notes", "")
        
        # Create notes dialog
        notes_window = tk.Toplevel(self.root)
        notes_window.title(f"Notes for Trial {trial_name}")
        notes_window.resizable(True, True)
        
        # Label
        tk.Label(notes_window, text=f"Notes for {trial_name}:", font=self.bold_font).pack(pady=10)
        
        # Text widget with scrollbar
        text_frame = tk.Frame(notes_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            font=self.default_font
        )
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)
        
        # Insert existing notes
        text_widget.insert("1.0", existing_notes)
        text_widget.focus()

        # Button frame
        btn_frame = tk.Frame(notes_window)
        btn_frame.pack(pady=10)

        # ----- button callbacks -----
        def save_notes():
            notes_text = text_widget.get("1.0", tk.END).strip()
            
            # Get existing mark status
            existing_mark = None
            if trial_name in self._trial_marks:
                mark_data = self._trial_marks[trial_name]
                if isinstance(mark_data, str):
                    existing_mark = mark_data
                elif isinstance(mark_data, dict):
                    existing_mark = mark_data.get("mark")
            
            # Save as dictionary
            if notes_text or existing_mark:
                self._trial_marks[trial_name] = {
                    "mark": existing_mark,
                    "notes": notes_text
                }
            else:
                # Remove entry if both are empty
                self._trial_marks.pop(trial_name, None)
            
            self._save_trial_marks()
            self._refresh_trial_list()
            notes_window.destroy()
        
        def cancel():
            notes_window.destroy()
        
        # ----- buttons -----
        tk.Button(
            btn_frame, text="Save",
            command=save_notes, width=15, bg="#90EE90"
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame, text="Cancel",
            command=cancel, width=15
        ).pack(side=tk.LEFT, padx=5)

        # Size + center AFTER everything is packed
        notes_window.update_idletasks()
        req_w = notes_window.winfo_reqwidth()
        req_h = notes_window.winfo_reqheight()

        width  = max(600, req_w + 40)
        height = max(400, req_h + 40)

        center_window(notes_window, width=width, height=height)
        notes_window.minsize(width, height)

    def _apply_channel_filter(self):
        """Filter both channel and export listboxes using the search text, preserving selections."""
        query = self._channel_filter_var.get().strip().lower()
        tokens = query.split() if query else []

        # Filter channels list
        if self._all_channels:
            filtered_channels = []
            for ch in self._all_channels:
                ch_l = ch.lower()
                if all(t in ch_l for t in tokens):
                    filtered_channels.append(ch)

            self._populating_channels = True
            try:
                self.channel_listbox.delete(0, tk.END)
                for i, ch in enumerate(filtered_channels, 1):
                    self.channel_listbox.insert(tk.END, f"{i}. {ch}")

                # Restore selection
                for i in range(self.channel_listbox.size()):
                    ch_name = self._parse_channel_item(self.channel_listbox.get(i))
                    if ch_name in self._sticky_channel_selection:
                        self.channel_listbox.selection_set(i)
            finally:
                self._populating_channels = False

        # Filter export list (if it exists and has data)
        if hasattr(self, '_all_export_channels') and self._all_export_channels:
            filtered_export = []
            for ch in self._all_export_channels:
                ch_l = ch.lower()
                if all(t in ch_l for t in tokens):
                    filtered_export.append(ch)

            self._populating_export = True
            try:
                self.export_listbox.delete(0, tk.END)
                for ch in filtered_export:
                    self.export_listbox.insert(tk.END, ch)

                # Restore export selection
                for i in range(self.export_listbox.size()):
                    ch_name = self.export_listbox.get(i)
                    if ch_name in self._sticky_export_selection:
                        self.export_listbox.selection_set(i)
            finally:
                self._populating_export = False

    def _refresh_trial_list(self):
        """Rebuild the trial listbox, showing marks and note indicators, and restore previous selection."""
        if not self.explorer:
            return
        
        # Remember current selection
        sel = self.trial_listbox.curselection()
        current_index = sel[0] if sel else None
        
        # Rebuild list
        self.trial_listbox.delete(0, tk.END)
        for idx, name in enumerate(self.explorer.trial_names, start=1):
            mark_data = self._trial_marks.get(name, "")
            
            # Handle both old string format and new dict format
            if isinstance(mark_data, dict):
                mark = mark_data.get("mark", "")
                has_notes = bool(mark_data.get("notes", "").strip())
            else:
                mark = mark_data
                has_notes = False
            
            mark_text = f"  [{mark.title()}]" if mark else ""
            notes_indicator = " [Notes]" if has_notes else ""  # Add note icon if notes exist
            
            # Remove first number from trial name (e.g., "02_11_01" -> "11_01")
            display_name = "_".join(name.split("_")[1:]) if "_" in name else name
            self.trial_listbox.insert(tk.END, f"{idx}. {display_name}{mark_text}{notes_indicator}")
        
        # Restore selection
        if current_index is not None:
            self.trial_listbox.selection_set(current_index)

    def _setup_main_panels(self, parent):
        """Create the main working panels for channels and exports."""

        self.channel_frame = tk.LabelFrame(parent, text="Channels in Trial", font=self.bold_font)
        self.channel_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        # --- Search row ---
        search_row = tk.Frame(self.channel_frame)
        search_row.pack(fill=tk.X, pady=(2, 6))

        tk.Label(search_row, text="Search:", font=self.default_font).pack(side=tk.LEFT, padx=(0, 6))

        self.channel_search_entry = tk.Entry(search_row, textvariable=self._channel_filter_var)
        self.channel_search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.channel_clear_btn = tk.Button(search_row, text="Clear",command=lambda: self._channel_filter_var.set(""))
        self.channel_clear_btn.pack(side=tk.LEFT, padx=(6, 0))

        # --- List area ---
        list_area = tk.Frame(self.channel_frame)
        list_area.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(list_area, orient="vertical")
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.channel_listbox = tk.Listbox(
            list_area,
            selectmode=tk.MULTIPLE,
            yscrollcommand=scrollbar.set,
            exportselection=False
        )
        self.channel_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=self.channel_listbox.yview)

        # --- Inspect button row ---
        button_row = tk.Frame(self.channel_frame)
        button_row.pack(fill=tk.X, pady=(6, 2))

        self.inspect_button = tk.Button(button_row, text="Inspect Selected Channel(s)", command=self.inspect_selected_channels)
        self.inspect_button.pack(pady=(0,2))

        self._setup_export_panel(parent)

    def _on_marker_select(self, event=None):
        """Remember selected marker events (sticky) and persist to disk."""
        if getattr(self, "_populating_export", False):  # Reuse the populate flag
            return
        
        try:
            selected = {self.marker_listbox.get(i) for i in self.marker_listbox.curselection()}
            self._sticky_marker_selection = set(selected)
            
            # Persist to disk
            from utility.user_prefs import set_marker_defaults
            set_marker_defaults(sorted(selected))
            self.session.save_state(
                current_trial_name=self.current_trial.name if self.current_trial else None,
                trial_names=self.explorer.trial_names if self.explorer else [],
                filepath=self.explorer.filepath if self.explorer else "",
                channel_filter=self._channel_filter_var.get(),
                inspect_selection=self._sticky_channel_selection,
                export_selection=self._sticky_export_selection,
                marker_selection=self._sticky_marker_selection,
            )
        except Exception:
            pass

    def _setup_export_panel(self, parent):
        """Create the export selection and analysis tools panel."""
        self.export_panel = tk.Frame(parent)
        self.export_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

        # Compute / Label / Utility groupings
        compute_frame = tk.LabelFrame(self.export_panel, text="Compute", font=self.bold_font)
        compute_frame.pack(fill=tk.X, padx=5, pady=(5, 2))

        label_frame = tk.LabelFrame(self.export_panel, text="Label", font=self.bold_font)
        label_frame.pack(fill=tk.X, padx=5, pady=2)

        utility_frame = tk.LabelFrame(self.export_panel, text="Utilities", font=self.bold_font)
        utility_frame.pack(fill=tk.X, padx=5, pady=(2, 5))

        self._setup_action_buttons(compute_parent=compute_frame,
                                label_parent=label_frame,
                                utility_parent=utility_frame)

        # ---- MARKER EVENTS SECTION ----
        marker_frame = tk.LabelFrame(self.export_panel, text="Event Markers", font=self.bold_font)
        marker_frame.pack(fill=tk.X, padx=5, pady=5)
        
        marker_list_frame = tk.Frame(marker_frame)
        marker_list_frame.pack(fill=tk.BOTH, expand=True)
        
        marker_scrollbar = tk.Scrollbar(marker_list_frame, orient="vertical")
        self.marker_listbox = tk.Listbox(
            marker_list_frame,
            selectmode=tk.MULTIPLE,
            height=5,
            yscrollcommand=marker_scrollbar.set,
            exportselection=False
        )
        marker_scrollbar.config(command=self.marker_listbox.yview)
        
        self.marker_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        marker_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection event
        self.marker_listbox.bind('<<ListboxSelect>>', self._on_marker_select)

        # ---- EXPORT CHANNELS SECTION ----
        export_frame = tk.LabelFrame(
            self.export_panel,
            text="Channels to Export",
            font=self.bold_font
        )
        export_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        export_list_frame = tk.Frame(export_frame)
        export_list_frame.pack(fill=tk.BOTH, expand=True)   

        self.export_scrollbar = tk.Scrollbar(export_list_frame, orient="vertical")
        self.export_listbox = tk.Listbox(
            export_list_frame,
            selectmode=tk.MULTIPLE,
            width=40,
            height=20,
            yscrollcommand=self.export_scrollbar.set,
            exportselection=False
        )
        self.export_scrollbar.config(command=self.export_listbox.yview)
        self.export_listbox.bind("<<ListboxSelect>>", self._on_export_select)

        self.export_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.export_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _setup_action_buttons(self, compute_parent, label_parent, utility_parent):
        """Create analysis and utility buttons grouped by function."""
        groups = [
            (compute_parent, [
                ("Gaze Metrics (ρ,θ,φ)", self.calculate_gaze_metrics, "gaze_metrics_btn"),
                ("Angular Velocity", self.calculate_angular_velocity, "angular_velocity_btn"),
                ("Foveal Visual Radius", self.calculate_fvr, "fvr_btn"),
            ]),
            (label_parent, [
                ("Label Events", self.label_gaze, "label_gaze_btn"),
            ]),
            (utility_parent, [
                ("Show Parameters", self.show_task_protocol, "task_protocol_btn"),
                ("Clear Cache", self.clear_cache_dialog, "clear_cache_btn"),
                ("Help", self.show_help, "help_btn"),
            ]),
        ]

        for parent, actions in groups:
            for text, cmd, attr_name in actions:
                btn = tk.Button(parent, text=text, command=cmd, width=45)
                btn.pack(side=tk.TOP, pady=2, anchor="w")  # anchor left inside the frame
                setattr(self, attr_name, btn)

        self._update_button_states()


    def _update_button_states(self):
        """Enable/disable buttons based on what's loaded."""
        file_loaded = self.explorer is not None
        trial_selected = self.current_trial is not None
        
        # File-level buttons (need file loaded)
        file_buttons = [
            self.task_protocol_btn,
            self.clear_cache_btn
        ]
        
        # Trial-level buttons (need trial selected)
        trial_buttons = [
            self.gaze_metrics_btn,
            self.angular_velocity_btn,
            self.fvr_btn,
            self.label_gaze_btn,
            self.inspect_button,
            self.mark_good_btn,
            self.mark_bad_btn,
            self.mark_review_btn,
            self.clear_mark_btn,
            self.notes_btn
        ]
        
        # Update file-level buttons
        for btn in file_buttons:
            btn.config(state=tk.NORMAL if file_loaded else tk.DISABLED)
        
        # Update trial-level buttons
        for btn in trial_buttons:
            btn.config(state=tk.NORMAL if trial_selected else tk.DISABLED)

        # Search bar enabled only when a trial is selected
        search_state = tk.NORMAL if trial_selected else tk.DISABLED
        self.channel_search_entry.config(state=search_state)
        self.channel_clear_btn.config(state=search_state)

        # If disabling, also clear filter so list isn't stuck filtered
        if not trial_selected:
            try:
                self._channel_filter_var.set("")
            except Exception:
                pass

    def clear_interpolation_cache(self):
        """Clear all cached interpolation decisions for current session."""
        if self.explorer:
            num_cached = len(self.explorer.interpolation_cache)
            self.explorer.interpolation_cache.clear()
            self.explorer.interpolation_methods.clear()
            if num_cached > 0:
                messagebox.showinfo("Cache Cleared", 
                    f"Cleared {num_cached} cached interpolation decision(s).\n\n"
                    "Next time you inspect or export channels,\n"
                    "you'll be prompted for interpolation again.")
            else:
                messagebox.showinfo("Cache Empty", "No cached interpolations to clear.")

    def _on_export_select(self, event=None):
        """Remember selected export channels (sticky) and persist to disk."""
        if getattr(self, "_populating_export", False):
            return

        try:
            # What is currently visible in the export listbox
            visible = {self.export_listbox.get(i) for i in range(self.export_listbox.size())}

            # What is selected among visible entries
            selected = {self.export_listbox.get(i) for i in self.export_listbox.curselection()}

            # Remove any visible items from sticky set, then add back selected ones.
            # This makes deselection work correctly while preserving hidden selections.
            self._sticky_export_selection.difference_update(visible)
            self._sticky_export_selection.update(selected)

            from utility.user_prefs import set_export_defaults
            set_export_defaults(sorted(self._sticky_export_selection))
            self.session.save_state(
                current_trial_name=self.current_trial.name if self.current_trial else None,
                trial_names=self.explorer.trial_names if self.explorer else [],
                filepath=self.explorer.filepath if self.explorer else "",
                channel_filter=self._channel_filter_var.get(),
                inspect_selection=self._sticky_channel_selection,
                export_selection=self._sticky_export_selection,
                marker_selection=self._sticky_marker_selection,
            )

        except Exception:
            pass

    def load_file(self):
        """
        Load a .kinarm file and populate the trial list.
        
        This resets all application state and loads the new file data.
        The trial list preserves the original file ordering to match Dexterit-E.
        """
        filepath = filedialog.askopenfilename(filetypes=[("Kinarm Files", "*.kinarm")])
        if not filepath:
            return
            
        try:
            self._populating = True  # Prevent selection callbacks during loading
            
            # Temporarily disable trial selection callback
            try:
                self.trial_listbox.unbind("<<ListboxSelect>>")
            except Exception:
                pass

            # Initialize data explorer with new file
            self.explorer = KinarmDataExplorer(filepath)
            self._current_file_id = self.explorer.filepath
            self.current_trial = None
            
            # Update status displays
            self.selected_trial_var.set("Selected Trial: (none)")

            # Clear all UI lists
            self.trial_listbox.delete(0, tk.END)
            self.channel_listbox.delete(0, tk.END)
            self.export_listbox.delete(0, tk.END)

            self._load_trial_marks()

            # Get available events from first trial's events
            self.available_events = []
            try:
                # Get unique event labels from any trial that has events
                event_labels_set = set()
                for trial_name, trial in self.explorer.exam.trials.items():
                    if trial_name != "common" and hasattr(trial, 'events'):
                        for event in trial.events:
                            if hasattr(event, 'label'):
                                event_labels_set.add(event.label)
                
                self.available_events = sorted(list(event_labels_set))
                print(f"Found {len(self.available_events)} event types: {self.available_events}")
            except Exception as e:
                print(f"Event load failed: {e}")

            # Populate trial list with numbered entries
            self._refresh_trial_list()
            self._update_button_states()

            state = self._load_session_state()
            if state and messagebox.askyesno(
                    "Resume previous session?",
                    f"Found a previous session for this file.\n\n"
                    f"Resume at trial: {state.get('trial_name')}?\n"
                    f"Note: This will also overwrite currently selected channels with selections from previous session."
                ):
                # Temporarily allow select_trial() to run
                was_populating = self._populating
                self._populating = False
                try:
                    self._resume_from_state(state)
                finally:
                    self._populating = was_populating

        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            # Re-enable trial selection and clear loading flag
            self.trial_listbox.bind("<<ListboxSelect>>", self.select_trial)
            self._populating = False

    def select_trial(self, event=None):
        """
        Handle trial selection from the list.
        
        This loads the selected trial data, computes derived channels,
        and updates all related UI components.
        """
        # Ignore callbacks during list repopulation
        if self._populating:
            return
            
        try:
            # Get selection (may be empty after list refresh)
            sel = self.trial_listbox.curselection()
            if not sel:
                return

            # Extract trial name from numbered list entry
            index = sel[0]
            entry_text = self.trial_listbox.get(index)
            # Use index to get the actual trial name from explorer
            name = self.explorer.trial_names[index]

            if self.current_trial and self.current_trial.name == name:
                return
            
            # Load trial data
            self.explorer.current_trial = self.explorer.exam.trials[name]
            self.explorer._derived_cache.clear()
            self.current_trial = self.explorer.current_trial
                        
            # Build informative header with trial details
            tp_num = find_trial_tp_number(self.current_trial)
            tp_text = f"TP #{tp_num}" if tp_num is not None else "TP #(unknown)"
            frames = self.current_trial.frame_count
            rate = self.current_trial.frame_rate
            duration = frames / rate if rate > 0 else 0
            
            display_name = "_".join(name.split("_")[1:]) if "_" in name else name

            # Update top status bar
            self.selected_trial_var.set(
                f"Selected Trial: {display_name}   •   {tp_text}   •   {frames} frames @ {rate:.2f} Hz (~{duration:.2f}s)"
            )

            # Compute derived kinematic channels
            self._compute_derived_channels()
            
            # Update channel and export lists
            self._refresh_channel_lists()

            # Update Task Protocol window if open
            if self.taskproto_win and hasattr(self.taskproto_win, 'window'):
                try:
                    if self.taskproto_win.window and self.taskproto_win.window.winfo_exists():
                        self.taskproto_win.update_selected_trial(name, self.current_trial)
                    else:
                        self.taskproto_win = None  # Window was closed, clear reference
                except Exception:
                    self.taskproto_win = None  # Window is invalid, clear reference

            self._update_button_states()
            self._restore_sticky_export_selection()
            self.session.save_state(
                current_trial_name=self.current_trial.name if self.current_trial else None,
                trial_names=self.explorer.trial_names if self.explorer else [],
                filepath=self.explorer.filepath if self.explorer else "",
                channel_filter=self._channel_filter_var.get(),
                inspect_selection=self._sticky_channel_selection,
                export_selection=self._sticky_export_selection,
                marker_selection=self._sticky_marker_selection,
            )

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _compute_derived_channels(self):
        """
        Calculate derived kinematic channels from raw position data.

        Iterates over the channel names registered in the data_loader's
        DERIVED_CHANNEL_NAMES list. Channels that cannot be computed for the
        current trial's protocol are silently skipped.
        """
        for ch in self.explorer.DERIVED_CHANNEL_NAMES:
            derived = self.explorer._compute_derived_channel(ch)
            if derived and len(derived[0]) == self.current_trial.frame_count:
                self.current_trial.kinematics[ch] = DerivedChannel(values=derived[0], unit=derived[1])

    def _refresh_channel_lists(self):
        """
        Repopulate the channel inspection and export listboxes for the current trial.

        Adds raw kinematic channels, computed analysis metrics (rho, theta, etc.),
        and event pseudo-channels. Restores sticky selections afterward.
        """
        # Get all available channels
        all_channels = list(self.current_trial.kinematics.keys())
        self._all_channels = all_channels[:] # store raw list for filtering

        # Populate channel listbox using current filter
        self._apply_channel_filter()

        # Add computed analysis options
        calculated_channels = [
            "Rho (Distance)",
            "Theta (Azimuth)", 
            "Phi (Elevation)",
            "Angular_Velocity",
            "FVR (Foveal_Visual_Radius)"
        ]

        # Store all export channels for filtering
        self._all_export_channels = all_channels + calculated_channels
        # Add events too
        for ev in self.available_events:
            self._all_export_channels.append(ev)

        # Update export selection list (initially unfiltered)
        self.export_listbox.delete(0, tk.END)
        for ch in self._all_export_channels:
            self.export_listbox.insert(tk.END, ch)

        # Map event display names to event labels for lookup during export
        self._event_channel_map = {}
        for ev in self.available_events:
            self._event_channel_map[ev] = ev

        # Populate marker event listbox
        self.marker_listbox.delete(0, tk.END)
        for ev in self.available_events:
            self.marker_listbox.insert(tk.END, ev)
        
        # Restore sticky marker selection
        for i in range(self.marker_listbox.size()):
            if self.marker_listbox.get(i) in self._sticky_marker_selection:
                self.marker_listbox.selection_set(i)
            
        self._restore_sticky_export_selection()

    def clear_cache_dialog(self):
        """Dialog to clear specific caches/preferences without wiping everything."""
        win = tk.Toplevel(self.root)
        win.title("Clear Cache")
        win.resizable(False, False)

        tk.Label(win, text="Clear what?", font=self.bold_font).pack(padx=12, pady=(12, 8))

        body = tk.Frame(win)
        body.pack(padx=12, pady=(0, 12), fill=tk.X)

        def clear_interpolation():
            if not self.explorer:
                messagebox.showwarning("No File Loaded", "Load a .kinarm file first.")
                return
            num_cached = len(self.explorer.interpolation_cache)
            self.explorer.interpolation_cache.clear()
            self.explorer.interpolation_methods.clear()
            messagebox.showinfo("Interpolation Cache Cleared", f"Cleared {num_cached} cached interpolation decision(s).")
            win.destroy()

        def reset_label_order():
            try:
                from utility.user_prefs import clear_label_order, set_label_order
                clear_label_order()
                # Also reset in-memory default so the next trial prompts cleanly
                set_label_order(None)
            except Exception:
                pass
            messagebox.showinfo("Label Order Reset", "Label order preference cleared. Next labeling will prompt again.")
            win.destroy()

        def clear_labeler_channels():
            set_labeler_channel_defaults([])
            messagebox.showinfo("Labeler Channels Reset", 
                "Labeler channel selection cleared. Next labeling will default to xT and yT.")
            win.destroy()

        def clear_session_state():
            if self.session.delete_state():
                messagebox.showinfo("Session State Cleared", "session_state.json deleted for this file.")
            else:
                messagebox.showinfo("Session State Cleared", "No session_state.json found for this file.")
            win.destroy()

        btn1 = tk.Button(body, text="Clear Interpolation Cache", command=clear_interpolation, width=35)
        btn2 = tk.Button(body, text="Reset Label Order Preference", command=reset_label_order, width=35)
        btn3 = tk.Button(body, text="Clear Session Resume State", command=clear_session_state, width=35)
        btn4 = tk.Button(body, text="Clear Labeler Channel Selection", command=clear_labeler_channels).pack(fill=tk.X, pady=2)
        btn5 = tk.Button(body, text="Close", command=win.destroy, width=35)

        btn1.pack(pady=4)
        btn2.pack(pady=4)
        btn3.pack(pady=4)
        btn4.pack(pady=4)
        btn5.pack(pady=(10, 0))

        win.update_idletasks()
        center_window(win, width=max(420, win.winfo_reqwidth()+20), height=win.winfo_reqheight()+20)


    def inspect_selected_channels(self):
        """
        Plot the selected channels with smart interpolation applied.
        
        This opens matplotlib plots showing the time series data for
        the selected channels, with automatic handling of missing data
        through interpolation.
        """
        if not self.explorer or not self.current_trial:
            messagebox.showwarning("Warning", "No trial selected.")
            return

        selections = self.channel_listbox.curselection()
        if not selections:
            messagebox.showwarning("Warning", "Please select at least one channel to inspect.")
            return
    
        try:
            for fig in self._open_figures[:]:
                try:
                    plt.close(fig)
                except Exception:
                    pass
            self._open_figures.clear()

            selected_channels = [self.channel_listbox.get(i).split(". ", 1)[1] for i in selections]

            # Check if we have cached interpolation for these channels
            trial_name = self.explorer.current_trial.name
            cache_key = (trial_name, tuple(sorted(selected_channels)))
            has_cached = cache_key in self.explorer.interpolation_cache
            
            force_prompt = False
            if has_cached:
                response = messagebox.askyesnocancel(
                    "Cached Interpolation Found",
                    "You've already interpolated these channels.\n\n"
                    "Yes = Use previous interpolation\n"
                    "No = Re-interpolate (will show preview again)\n"
                    "Cancel = Abort"
                )
                if response is None:
                    return
                elif response:
                    force_prompt = False
                else:
                    force_prompt = True

            from utility.kinarm_utils import find_trial_tp_number
            tp_num = find_trial_tp_number(self.current_trial)
            tp_text = f"TP #{tp_num}" if tp_num is not None else "TP #(unknown)"

            # Find trial number
            trial_number = None
            for idx, name in enumerate(self.explorer.trial_names, start=1):
                if name == self.current_trial.name:
                    trial_number = idx
                    break

            trial_info = f"Trial {trial_number}: {self.current_trial.name}  •  {tp_text}"

            interpolated_data = self.explorer.smart_interpolate_trial_data(
                selected_channels,
                force_prompt=force_prompt,
                trial_info=trial_info
            )
            
            if interpolated_data is None:
                return

            channel_data_dict = {}
            for ch in selected_channels:
                if ch in interpolated_data:
                    processed_data = interpolated_data[ch]
                else:
                    if ch in self.current_trial.kinematics:
                        raw_data = np.array(self.current_trial.kinematics[ch].values, dtype=float)
                    elif ch in self.current_trial.positions:
                        raw_data = np.array(self.current_trial.positions[ch].values, dtype=float)
                    else:
                        derived = self.explorer._compute_derived_channel(ch)
                        if derived:
                            raw_data = np.array(derived[0], dtype=float)
                        else:
                            print(f"Could not get data for channel: {ch}")
                            continue
                    
                    raw_data[np.abs(raw_data) >= 99.9] = np.nan
                    processed_data = raw_data

                channel_data_dict[ch] = (processed_data, processed_data)

            import matplotlib.pyplot as plt

            num_plots = len(selected_channels)
            fig, axs = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots), squeeze=False)
            
            # NEW: Track the figure for cleanup
            self._open_figures.append(fig)

            plot_idx = 0
            for ch, (_, processed) in channel_data_dict.items():
                axs[plot_idx][0].plot(processed, label="Processed", color="blue")
                axs[plot_idx][0].set_title(f"{ch} (Plot)")
                axs[plot_idx][0].legend()
                axs[plot_idx][0].grid(True)
                plot_idx += 1

            fig.tight_layout()
            plt.show(block=False)  # NEW: Changed to non-blocking
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def calculate_gaze_metrics(self):
        """Calculate spherical gaze coordinates (rho, theta, phi) from raw gaze vectors."""
        if self.explorer:
            self.explorer.calculate_gaze_metrics()

    def calculate_angular_velocity(self):
        """Calculate angular velocity of gaze direction changes."""
        if self.explorer:
            self.explorer.calculate_angular_velocity()

    def calculate_fvr(self):
        """Calculate Foveal Visual Radius based on gaze distance from target."""
        if self.explorer:
            self.explorer.calculate_fvr()

    def _pick_labeler_channels(self):
        """
        Open a dialog for the user to select which channels to display
        in the gaze labeler. Gaze_X and Gaze_Y are required and locked.
        Returns a list of channel names or None if cancelled.
        """
        trial = self.current_trial
        if not trial:
            return None

        available = [ch for ch in trial.kinematics.keys()
                     if ch not in ("Gaze_X", "Gaze_Y")]

        result = {"channels": None}

        win = tk.Toplevel(self.root)
        win.title("Select Labeler Channels")
        win.resizable(False, True)
        win.grab_set()

        tk.Label(win, text="Gaze_X and Gaze_Y are always included.",
                 font=self.bold_font).pack(padx=12, pady=(12, 4))
        tk.Label(win, text=f"Select up to {MAX_LABELER_CHANNELS - 2} additional channels to overlay:",
                 ).pack(padx=12, pady=(0, 8))

        frame = tk.Frame(win)
        frame.pack(padx=12, pady=(0, 8), fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, width=40, height=15,
                             yscrollcommand=scrollbar.set)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)

        for ch in available:
            listbox.insert(tk.END, ch)

        saved = get_labeler_channel_defaults()
        defaults = saved if saved else ["xT", "yT"]
        for i in range(listbox.size()):
            if listbox.get(i) in defaults:
                listbox.selection_set(i)

        count_label = tk.Label(win, text="")
        count_label.pack(padx=12)

        max_extra = MAX_LABELER_CHANNELS - 2

        def update_count(event=None):
            n = len(listbox.curselection())
            count_label.config(text=f"{n}/{max_extra} additional channels selected")
            if n > max_extra:
                count_label.config(fg="red")
            else:
                count_label.config(fg="black")

        listbox.bind("<<ListboxSelect>>", update_count)
        update_count()

        btn_frame = tk.Frame(win)
        btn_frame.pack(padx=12, pady=(4, 12))

        def on_ok():
            selected = [listbox.get(i) for i in listbox.curselection()]
            if len(selected) > max_extra:
                messagebox.showwarning("Too Many Channels",
                    f"Please select at most {max_extra} additional channels.",
                    parent=win)
                return
            set_labeler_channel_defaults(selected)
            result["channels"] = ["Gaze_X", "Gaze_Y"] + selected
            win.destroy()

        def on_cancel():
            win.destroy()

        tk.Button(btn_frame, text="OK", width=10, command=on_ok).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="Cancel", width=10, command=on_cancel).pack(side=tk.LEFT, padx=4)

        self.root.wait_window(win)
        return result["channels"]

    def label_gaze(self):
        """
        Launch the interactive gaze labeling tool with multi-trial support.
        
        This opens the GazeLabeler interface where users can manually
        label time periods as fixations, pursuits, or saccades.
        The results are saved along with selected export channels.
        Users can choose to continue to the next trial automatically.
        """
        try:
            trial = self.current_trial
            if not trial:
                messagebox.showwarning("No Trial", "Please select a trial first.")
                return
            
            # Prompt user to pick labeler channels (only on first iteration)
            saved = get_labeler_channel_defaults()
            if saved:
                # Validate saved channels still exist in this trial
                valid = [ch for ch in saved if ch in trial.kinematics]
                labeler_channels = ["Gaze_X", "Gaze_Y"] + valid
            else:
                labeler_channels = self._pick_labeler_channels()
                if labeler_channels is None:
                    return
            
            # Start labeling loop - can continue through multiple trials
            while trial is not None:
                # Get interpolated gaze data for labeling
                interpolated_data = self.explorer.get_interpolated_gaze_data(labeler_channels)
                if interpolated_data is None:
                    return

                gaze_x = interpolated_data["Gaze_X"] 
                gaze_y = interpolated_data["Gaze_Y"]
                overlay_channels = {k: v for k, v in interpolated_data.items()
                                    if k not in ("Gaze_X", "Gaze_Y")}

                # If UI failed to reselect properly, fall back to sticky selection
                # Get all selections from UI
                ui_selections = [self.export_listbox.get(i) for i in self.export_listbox.curselection()]

                # Fall back to sticky selection if UI failed
                if not ui_selections and self._sticky_export_selection:
                    all_selections = list(self._sticky_export_selection)
                else:
                    all_selections = ui_selections

                # Separate channels from events
                selected_export_channels = []
                selected_events = []

                for item in all_selections:
                    # Check if this item is an event (it's in our available_events list)
                    if item in self.available_events:
                        # This is an event - add it directly
                        selected_events.append(item)
                    else:
                        # This is a regular channel
                        selected_export_channels.append(item)

                # Find trial index for numbering (1-based)
                trial_index = None
                for idx, name in enumerate(self.explorer.trial_names):
                    if name == trial.name:
                        trial_index = idx + 1  # Convert to 1-based
                        break

                # Build trial info string for display with number prefix
                tp_num = find_trial_tp_number(trial)
                tp_text = f"TP #{tp_num}" if tp_num is not None else "TP #(unknown)"
                frames = trial.frame_count
                rate = trial.frame_rate
                duration = frames / rate if rate > 0 else 0
                
                # Format: "2. 02_07_01  •  TP #14  •  1538 frames @ 1000.00 Hz (~1.54s)"
                try:
                    display_name = "_".join(trial.name.split("_")[1:])
                except Exception:
                    display_name = trial.name

                # Find total number of trials for progress indicator
                total_trials = len(self.explorer.trial_names)
                trial_info = (
                    f"{trial_index}/{total_trials}. {display_name}  •  {tp_text}  •  "
                    f"{frames} frames @ {rate:.2f} Hz (~{duration:.2f}s)"
                )

                # Launch gaze labeling process with trial info
                from label.gaze_labeler_export import run_labeling_process

                # Get selected marker events
                selected_markers = [self.marker_listbox.get(i) for i in self.marker_listbox.curselection()]
                label_order = get_label_order()
                action, labeler = run_labeling_process(
                    self.explorer,
                    trial.name,
                    gaze_x, gaze_y,
                    overlay_channels,
                    selected_export_channels,
                    kinarm_path=self.explorer.filepath,
                    trial_info=trial_info,
                    output_root=self.custom_save_location,
                    trial_index=trial_index,
                    selected_events=selected_events,
                    marker_events=selected_markers,
                    label_order=label_order,
                )

                # Persist label order if the dialog was used and returned a labeler
                try:
                    if labeler and getattr(labeler, "label_order", None):
                        set_label_order(labeler.label_order)
                except Exception:
                    pass

                # Persist exactly what the user selected in the export listbox (channels + metrics + events)
                self._sticky_export_selection = set(all_selections)
                self._restore_sticky_export_selection()
                self.session.save_state(
                    current_trial_name=self.current_trial.name if self.current_trial else None,
                    trial_names=self.explorer.trial_names if self.explorer else [],
                    filepath=self.explorer.filepath if self.explorer else "",
                    channel_filter=self._channel_filter_var.get(),
                    inspect_selection=self._sticky_channel_selection,
                    export_selection=self._sticky_export_selection,
                    marker_selection=self._sticky_marker_selection,
                )
                
                # Check result
                if action == "next_trial":
                    # User wants to continue to next trial
                    # Find current trial index
                    current_index = None
                    for idx, name in enumerate(self.explorer.trial_names):
                        if name == trial.name:
                            current_index = idx
                            break
                    
                    if current_index is not None and current_index + 1 < len(self.explorer.trial_names):
                        # Move to next trial
                        next_trial_name = self.explorer.trial_names[current_index + 1]
                        
                        # Update GUI selection
                        self.trial_listbox.selection_clear(0, tk.END)
                        self.trial_listbox.selection_set(current_index + 1)
                        self.trial_listbox.see(current_index + 1)
                        
                        self.select_trial()
                        trial = self.current_trial
                        continue
                    else:
                        # No more trials
                        messagebox.showinfo("Complete", "All trials labeled! That was the last trial.")
                        return
                        
                elif action == "accept":
                    # User clicked "Accept & Finish"
                    messagebox.showinfo("Success", "Gaze labeling completed successfully!")
                    return
                else:
                    # User cancelled
                    return

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_task_protocol(self):
        """
        Open the Task Protocol viewer window.
        
        This displays the experimental configuration tables (TP and Target)
        in a separate popup window with tabbed interface.
        """
        if not self.explorer:
            messagebox.showwarning("Warning", "Load a .kinarm file first.")
            return
        
        # Reuse existing window if open
        if self.taskproto_win and self.taskproto_win.window and self.taskproto_win.window.winfo_exists():
            self.taskproto_win.window.lift()
            self.taskproto_win.window.focus_force()
            return
            
        try:
            # Create or reuse Task Protocol window
            self.taskproto_win = TaskProtocolWindow(
                parent=self.root,
                exam=self.explorer.exam,
                current_trial_name=(self.current_trial.name if self.current_trial else None)
            )
            self.taskproto_win.show()
            
            # Update with current trial info if available
            if self.current_trial:
                self.taskproto_win.update_selected_trial(
                    trial_name=self.current_trial.name, 
                    trial=self.current_trial
                )
                
        except Exception as e:
            messagebox.showerror("Error", f"Could not open Task Protocol viewer:\n{e}")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    root = tk.Tk()

    # Fix DPI scaling for high-resolution displays (Windows/macOS/Linux)
    try:
        # macOS Retina
        if root.tk.call("tk", "windowingsystem") == "aqua":
            root.tk.call('tk', 'scaling', 2.0)

        # Windows High-DPI laptops
        elif root.tk.call("tk", "windowingsystem") == "win32":
            # Auto-scale based on system DPI
            import ctypes
            try:
                user32 = ctypes.windll.user32
                user32.SetProcessDPIAware()
                dpi = user32.GetDpiForSystem()
                root.tk.call('tk', 'scaling', dpi / 72.0)
            except Exception:
                # Fallback if DPI read fails
                root.tk.call('tk', 'scaling', 1.5)

        # Linux scaling (optional)
        elif root.tk.call("tk", "windowingsystem") == "x11":
            root.tk.call('tk', 'scaling', 1.5)

    except Exception:
        pass

    app = KinarmDataExplorerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()