"""
KINARM Data Explorer - Main GUI Application

This is the primary interface for loading KINARM data files, selecting trials,
inspecting channels, and performing gaze analysis.

Dependencies: tkinter, KinarmDataExplorer, matplotlib, numpy, pandas
"""
from pathlib import Path
import sys

# Add src directory to Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

# Imports
import tkinter as tk
from tkinter import filedialog, messagebox
from data.data_loader import KinarmDataExplorer
from gui.gui_help import HelpWindow
from gui.gui_session import SessionManager
from utility.user_prefs import get_label_order, set_label_order, get_export_defaults, get_marker_defaults, get_save_location, set_save_location, MAX_LABELER_CHANNELS, get_labeler_channel_defaults, set_labeler_channel_defaults

import matplotlib 
import platform
from utility.user_prefs import MPL_BACKEND
matplotlib.use(MPL_BACKEND)
import matplotlib.pyplot as plt

from gui.gui_metadata_viewer import TaskProtocolWindow
from utility.kinarm_utils import find_trial_tp_number
from gui.gui_utils import center_window, configure_big_treeview_style, log_crash
sys.excepthook = log_crash
from gui.gui_trial_panel import TrialPanel
from gui.gui_channel_panel import ChannelPanel
from gui.gui_export_panel import ExportPanel
from gui.gui_labeler import GazeLabelerController

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
        if platform.system() == "Windows":
            self.default_font = ("Segoe UI", 12)
            self.bold_font = ("Segoe UI", 12, "bold")
        elif platform.system() == "Linux":
            self.default_font = ("DejaVu Sans", 12)
            self.bold_font = ("DejaVu Sans", 12, "bold")
        else:  # Mac
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
        self._sticky_marker_selection = set(get_marker_defaults())
        self._populating_export = False
        self._populating = False                    # Flag to prevent callback loops
        self._open_figures = []
        self._trial_marks = {}  # Store trial quality marks
        self.available_events = []        
        self.custom_save_location = get_save_location()
        self._sticky_channel_selection = set()
        self._populating_channels = False
        self.session = SessionManager(
            get_explorer=lambda: self.explorer,
            get_save_location=lambda: self.custom_save_location,
        )
        self.trial_panel = TrialPanel(self)
        self.channel_panel = ChannelPanel(self)
        self.export_panel_obj = ExportPanel(self)
        self.labeler = GazeLabelerController(self)

        self._all_channels = []
        self._all_export_channels = []
        self._channel_filter_var = tk.StringVar()
        self._channel_filter_var.trace_add("write", lambda *_: self.channel_panel.apply_channel_filter())
        self._setup_gui()
        if hasattr(self, "channel_listbox"):
            self.channel_listbox.bind('<<ListboxSelect>>', self.channel_panel.on_channel_select)
        self.help_win = HelpWindow(self.root)

        self._update_button_states()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def show_help(self):
        self.help_win.show()

    def _load_session_state(self) -> dict | None:
        """Delegate to SessionManager."""
        return self.session.load_state()
        
    def _resume_from_state(self, state: dict, restore_channels: bool = True):
        """
        Restore GUI state from a previously saved session dictionary.

        Restores trial selection, channel/export/marker sticky selections,
        and the channel filter string.
        """
        if restore_channels:
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
        self.channel_panel.apply_channel_filter()

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
            if self.explorer and self.current_trial:
                self.session.save_state(
                    current_trial_name=self.current_trial.name,
                    trial_names=self.explorer.trial_names,
                    filepath=self.explorer.filepath,
                    channel_filter=self._channel_filter_var.get(),
                    inspect_selection=self._sticky_channel_selection,
                )
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
        desktop = Path.home() / "Desktop"
        default_loc = str(desktop / "gaze_labels" if desktop.exists() else Path.home() / "gaze_labels")
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
            initialdir=self.custom_save_location or str(Path.home() / "Desktop" if (Path.home() / "Desktop").exists() else Path.home())
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
        
        self.mark_good_btn = tk.Button(mark_frame, text="Mark Good", command=lambda: self.trial_panel.mark_trial("good"), bg="#90EE90")
        self.mark_bad_btn = tk.Button(mark_frame, text="Mark Bad", command=lambda: self.trial_panel.mark_trial("bad"), bg="#FFB6C6")
        self.mark_review_btn = tk.Button(mark_frame, text="Mark Review", command=lambda: self.trial_panel.mark_trial("review"), bg="#FFE4B5")
        self.clear_mark_btn = tk.Button(mark_frame, text="Clear Mark", command=lambda: self.trial_panel.mark_trial(None))
        self.notes_btn = tk.Button(mark_frame, text="Add Notes", command=self.trial_panel.add_trial_notes, bg="#E0E0E0")

        self.mark_good_btn.pack(side=tk.LEFT, padx=2)
        self.mark_bad_btn.pack(side=tk.LEFT, padx=2)
        self.mark_review_btn.pack(side=tk.LEFT, padx=2)
        self.clear_mark_btn.pack(side=tk.LEFT, padx=2)
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

        self.inspect_button = tk.Button(button_row, text="Inspect Selected Channel(s)", command=self.channel_panel.inspect_selected_channels)
        self.inspect_button.pack(pady=(0,2))

        self.export_panel_obj.setup(parent)

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
            self.current_trial = None
            
            # Update status displays
            self.selected_trial_var.set("Selected Trial: (none)")

            # Clear all UI lists
            self.trial_listbox.delete(0, tk.END)
            self.channel_listbox.delete(0, tk.END)
            self.export_listbox.delete(0, tk.END)

            self.trial_panel.load_trial_marks()

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
            self.trial_panel.refresh_trial_list()
            self._update_button_states()

            state = self._load_session_state()
            if state:
                raw_name = state.get('trial_name', '')
                trial_index = next(
                    (i + 1 for i, n in enumerate(self.explorer.trial_names) if n == raw_name),
                    None
                )
                display_name = "_".join(raw_name.split("_")[1:]) if "_" in raw_name else raw_name
                display_label = f"{trial_index}. {display_name}" if trial_index else display_name
                resume_trial = messagebox.askyesno(
                    "Resume previous session?",
                    f"Found a previous session for this file.\n\n"
                    f"Resume at trial: {display_label}?"
                )
                if resume_trial:
                    resume_channels = messagebox.askyesno(
                        "Restore channel selections?",
                        "Restore previous 'Channels in Trial' selections?\n\n"
                        "Select No to keep your current selections."
                    )
                    was_populating = self._populating
                    self._populating = False
                    try:
                        self._resume_from_state(state, restore_channels=resume_channels)
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
            self.channel_panel.compute_derived_channels()
            
            # Update channel and export lists
            self.channel_panel.refresh_channel_lists()

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
            self.export_panel_obj.restore_sticky_export_selection()
            self.session.save_state(
                current_trial_name=self.current_trial.name if self.current_trial else None,
                trial_names=self.explorer.trial_names if self.explorer else [],
                filepath=self.explorer.filepath if self.explorer else "",
                channel_filter=self._channel_filter_var.get(),
                inspect_selection=self._sticky_channel_selection,
            )

        except Exception as e:
            messagebox.showerror("Error", str(e))

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
        btn4 = tk.Button(body, text="Clear Labeler Channel Selection", command=clear_labeler_channels, width=35)
        btn5 = tk.Button(body, text="Close", command=win.destroy, width=35)

        btn1.pack(pady=4)
        btn2.pack(pady=4)
        btn3.pack(pady=4)
        btn4.pack(pady=4)
        btn5.pack(pady=(10, 0))

        win.update_idletasks()
        center_window(win, width=max(420, win.winfo_reqwidth()+20), height=win.winfo_reqheight()+20)

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