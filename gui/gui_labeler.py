"""
Gaze Labeler Controller

Orchestrates the gaze labeling workflow from the main GUI, including
channel selection and multi-trial iteration.

This module acts as the GUI-side controller for the label subsystem.
It owns the labeler channel picker dialog and the label_gaze loop,
both of which depend heavily on main GUI state (trial selection,
export listbox, sticky selections, session saving).

The underlying labeling engine and export logic live in:
    label/gaze_labeler_ui.py
    label/gaze_labeler_export.py
"""

import tkinter as tk
from tkinter import messagebox

from utility.user_prefs import (
    get_label_order,
    set_label_order,
    get_labeler_channel_defaults,
    set_labeler_channel_defaults,
    MAX_LABELER_CHANNELS,
)
from utility.kinarm_utils import find_trial_tp_number


class GazeLabelerController:
    """
    GUI controller for the gaze labeling workflow.

    Owns the labeler channel picker dialog and the multi-trial
    label_gaze loop. Delegates all labeling and export logic to
    the label subsystem (gaze_labeler_ui, gaze_labeler_export).

    Parameters
    ----------
    app : KinarmDataExplorerGUI
        Reference to the main application instance.
    """

    def __init__(self, app):
        self.app = app

    def _pick_labeler_channels(self):
        """
        Open a dialog for the user to select which channels to display
        in the gaze labeler. Gaze_X and Gaze_Y are required and locked.
        Returns a list of channel names or None if cancelled.
        """
        app = self.app
        trial = app.current_trial
        if not trial:
            return None

        available = [ch for ch in trial.kinematics.keys()
                     if ch not in ("Gaze_X", "Gaze_Y")]

        result = {"channels": None}

        win = tk.Toplevel(app.root)
        win.title("Select Labeler Channels")
        win.resizable(False, True)
        win.grab_set()

        tk.Label(win, text="Gaze_X and Gaze_Y are always included.",
                 font=app.bold_font).pack(padx=12, pady=(12, 4))

        max_extra = MAX_LABELER_CHANNELS - 2
        tk.Label(win, text=f"Select up to {max_extra} additional channels to overlay:",
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

        app.root.wait_window(win)
        return result["channels"]

    def label_gaze(self):
        """
        Launch the interactive gaze labeling tool with multi-trial support.

        This opens the GazeLabeler interface where users can manually
        label time periods as fixations, pursuits, or saccades.
        The results are saved along with selected export channels.
        Users can choose to continue to the next trial automatically.
        """
        from label.gaze_labeler_export import run_labeling_process

        app = self.app

        try:
            trial = app.current_trial
            if not trial:
                messagebox.showwarning("No Trial", "Please select a trial first.")
                return

            saved = get_labeler_channel_defaults()
            if saved:
                valid = [ch for ch in saved if ch in trial.kinematics]
                labeler_channels = ["Gaze_X", "Gaze_Y"] + valid
            else:
                labeler_channels = self._pick_labeler_channels()
                if labeler_channels is None:
                    return

            while trial is not None:
                # Resolve export selections first so they're available for upfront interpolation
                ui_selections = [app.export_listbox.get(i) for i in app.export_listbox.curselection()]
                if not ui_selections and app._sticky_export_selection:
                    all_selections = list(app._sticky_export_selection)
                else:
                    all_selections = ui_selections

                selected_export_channels = []
                selected_events = []
                for item in all_selections:
                    if item in app.available_events:
                        selected_events.append(item)
                    else:
                        selected_export_channels.append(item)

                # Interpolate overlay and export channels upfront so all
                # interpolation prompts appear before the labeler opens.
                all_channels_to_interpolate = list(labeler_channels)
                for ch in selected_export_channels:
                    if ch not in all_channels_to_interpolate and ch in app.current_trial.kinematics:
                        all_channels_to_interpolate.append(ch)

                interpolated_data = app.explorer.get_interpolated_gaze_data(all_channels_to_interpolate)
                if interpolated_data is None:
                    return

                gaze_x = interpolated_data["Gaze_X"]
                gaze_y = interpolated_data["Gaze_Y"]
                # Only include channels selected for overlay display, not export-only channels
                overlay_channels = {k: v for k, v in interpolated_data.items()
                                    if k in labeler_channels and k not in ("Gaze_X", "Gaze_Y")}

                trial_index = None
                for idx, name in enumerate(app.explorer.trial_names):
                    if name == trial.name:
                        trial_index = idx + 1
                        break

                tp_num = find_trial_tp_number(trial)
                tp_text = f"TP #{tp_num}" if tp_num is not None else "TP #(unknown)"
                frames = trial.frame_count
                rate = trial.frame_rate
                duration = frames / rate if rate > 0 else 0
                try:
                    display_name = "_".join(trial.name.split("_")[1:])
                except Exception:
                    display_name = trial.name

                total_trials = len(app.explorer.trial_names)
                trial_info = (
                    f"{trial_index}/{total_trials}. {display_name}  •  {tp_text}  •  "
                    f"{frames} frames @ {rate:.2f} Hz (~{duration:.2f}s)"
                )

                selected_markers = [app.marker_listbox.get(i) for i in app.marker_listbox.curselection()]
                label_order = get_label_order()

                action, labeler = run_labeling_process(
                    app.explorer,
                    trial.name,
                    gaze_x, gaze_y,
                    overlay_channels,
                    selected_export_channels,
                    kinarm_path=app.explorer.filepath,
                    trial_info=trial_info,
                    output_root=app.custom_save_location,
                    trial_index=trial_index,
                    selected_events=selected_events,
                    marker_events=selected_markers,
                    label_order=label_order,
                )

                try:
                    if labeler and getattr(labeler, "label_order", None):
                        set_label_order(labeler.label_order)
                except Exception:
                    pass

                app._sticky_export_selection = set(all_selections)
                app.export_panel_obj.restore_sticky_export_selection()
                app.session.save_state(
                    current_trial_name=app.current_trial.name if app.current_trial else None,
                    trial_names=app.explorer.trial_names if app.explorer else [],
                    filepath=app.explorer.filepath if app.explorer else "",
                    channel_filter=app._channel_filter_var.get(),
                    inspect_selection=app._sticky_channel_selection,
                )

                if action == "next_trial":
                    current_index = None
                    for idx, name in enumerate(app.explorer.trial_names):
                        if name == trial.name:
                            current_index = idx
                            break
                    if current_index is not None and current_index + 1 < len(app.explorer.trial_names):
                        app.trial_listbox.selection_clear(0, tk.END)
                        app.trial_listbox.selection_set(current_index + 1)
                        app.trial_listbox.see(current_index + 1)
                        app.select_trial()
                        trial = app.current_trial
                        continue
                    else:
                        messagebox.showinfo("Complete", "All trials labeled! That was the last trial.")
                        return
                elif action == "accept":
                    messagebox.showinfo("Success", "Gaze labeling completed successfully!")
                    return
                else:
                    return

        except Exception as e:
            messagebox.showerror("Error", str(e))