"""
Channel Panel - Channel inspection, filtering, and plotting.
"""

import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt

from data.data_loader import DerivedChannel
from utility.kinarm_utils import find_trial_tp_number


class ChannelPanel:
    """Handles channel listbox, filtering, inspection, and derived channel computation."""

    def __init__(self, app):
        self.app = app

    def parse_channel_item(self, item: str) -> str:
        """Extract the channel name from a numbered listbox entry (e.g., '12. Right_HandX' -> 'Right_HandX')."""
        return item.split(". ", 1)[1] if ". " in item else item

    def on_channel_select(self, event=None):
        """Persist channel selections across filtering by tracking by name."""
        if getattr(self.app, "_populating_channels", False):
            return

        visible = {
            self.parse_channel_item(self.app.channel_listbox.get(i))
            for i in range(self.app.channel_listbox.size())
        }
        selected = {
            self.parse_channel_item(self.app.channel_listbox.get(i))
            for i in self.app.channel_listbox.curselection()
        }

        self.app._sticky_channel_selection.difference_update(visible)
        self.app._sticky_channel_selection.update(selected)

    def apply_channel_filter(self):
        """Filter both channel and export listboxes using the search text, preserving selections."""
        query = self.app._channel_filter_var.get().strip().lower()
        tokens = query.split() if query else []

        if self.app._all_channels:
            filtered_channels = [
                ch for ch in self.app._all_channels
                if all(t in ch.lower() for t in tokens)
            ]

            self.app._populating_channels = True
            try:
                self.app.channel_listbox.delete(0, tk.END)
                for i, ch in enumerate(filtered_channels, 1):
                    self.app.channel_listbox.insert(tk.END, f"{i}. {ch}")

                for i in range(self.app.channel_listbox.size()):
                    ch_name = self.parse_channel_item(self.app.channel_listbox.get(i))
                    if ch_name in self.app._sticky_channel_selection:
                        self.app.channel_listbox.selection_set(i)
            finally:
                self.app._populating_channels = False

        if hasattr(self.app, '_all_export_channels') and self.app._all_export_channels:
            filtered_export = [
                ch for ch in self.app._all_export_channels
                if all(t in ch.lower() for t in tokens)
            ]

            self.app._populating_export = True
            try:
                self.app.export_listbox.delete(0, tk.END)
                for ch in filtered_export:
                    self.app.export_listbox.insert(tk.END, ch)

                for i in range(self.app.export_listbox.size()):
                    ch_name = self.app.export_listbox.get(i)
                    if ch_name in self.app._sticky_export_selection:
                        self.app.export_listbox.selection_set(i)
            finally:
                self.app._populating_export = False

    def refresh_channel_lists(self):
        """Repopulate the channel inspection and export listboxes for the current trial."""
        all_channels = list(self.app.current_trial.kinematics.keys())
        self.app._all_channels = all_channels[:]

        self.apply_channel_filter()

        calculated_channels = [
            "Rho (Distance)",
            "Theta (Azimuth)",
            "Phi (Elevation)",
            "Angular_Velocity",
            "FVR (Foveal_Visual_Radius)"
        ]

        self.app._all_export_channels = all_channels + calculated_channels
        for ev in self.app.available_events:
            self.app._all_export_channels.append(ev)

        self.app.export_listbox.delete(0, tk.END)
        for ch in self.app._all_export_channels:
            self.app.export_listbox.insert(tk.END, ch)

        self.app._event_channel_map = {ev: ev for ev in self.app.available_events}

        self.app.marker_listbox.delete(0, tk.END)
        for ev in self.app.available_events:
            self.app.marker_listbox.insert(tk.END, ev)

        for i in range(self.app.marker_listbox.size()):
            if self.app.marker_listbox.get(i) in self.app._sticky_marker_selection:
                self.app.marker_listbox.selection_set(i)

        self.app.export_panel_obj.restore_sticky_export_selection()

    def compute_derived_channels(self):
        """Calculate derived kinematic channels from raw position data."""
        for ch in self.app.explorer.DERIVED_CHANNEL_NAMES:
            derived = self.app.explorer._compute_derived_channel(ch)
            if derived and len(derived[0]) == self.app.current_trial.frame_count:
                self.app.current_trial.kinematics[ch] = DerivedChannel(
                    values=derived[0], unit=derived[1]
                )

    def inspect_selected_channels(self):
        """Plot selected channels with smart interpolation applied."""
        if not self.app.explorer or not self.app.current_trial:
            messagebox.showwarning("Warning", "No trial selected.")
            return

        selections = self.app.channel_listbox.curselection()
        if not selections:
            messagebox.showwarning("Warning", "Please select at least one channel to inspect.")
            return

        try:
            for fig in self.app._open_figures[:]:
                try:
                    plt.close(fig)
                except Exception:
                    pass
            self.app._open_figures.clear()

            selected_channels = [
                self.app.channel_listbox.get(i).split(". ", 1)[1] for i in selections
            ]

            trial_name = self.app.explorer.current_trial.name
            cache_key = (trial_name, tuple(sorted(selected_channels)))
            has_cached = cache_key in self.app.explorer.interpolation_cache

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
                force_prompt = not response

            tp_num = find_trial_tp_number(self.app.current_trial)
            tp_text = f"TP #{tp_num}" if tp_num is not None else "TP #(unknown)"

            trial_number = next(
                (idx for idx, name in enumerate(self.app.explorer.trial_names, start=1)
                 if name == self.app.current_trial.name),
                None
            )

            trial_info = f"Trial {trial_number}: {self.app.current_trial.name}  •  {tp_text}"

            interpolated_data = self.app.explorer.smart_interpolate_trial_data(
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
                    if ch in self.app.current_trial.kinematics:
                        raw_data = np.array(self.app.current_trial.kinematics[ch].values, dtype=float)
                    elif ch in self.app.current_trial.positions:
                        raw_data = np.array(self.app.current_trial.positions[ch].values, dtype=float)
                    else:
                        derived = self.app.explorer._compute_derived_channel(ch)
                        if derived:
                            raw_data = np.array(derived[0], dtype=float)
                        else:
                            print(f"Could not get data for channel: {ch}")
                            continue

                    raw_data[np.abs(raw_data) >= 99.9] = np.nan
                    processed_data = raw_data

                channel_data_dict[ch] = processed_data

            num_plots = len(channel_data_dict)

            if num_plots > 4:
                proceed = messagebox.askyesno(
                    "Large Selection",
                    f"You selected {num_plots} channels. Plots may be difficult to read "
                    f"with more than 4 channels. Continue anyway?" 
                )
                if not proceed:
                    return

            fig, axs = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots), squeeze=False)
            try:
                fig.canvas.manager.window.showMaximized()
            except AttributeError:
                pass  # TkAgg or other backends don't support showMaximized
            self.app._open_figures.append(fig)

            for plot_idx, (ch, processed) in enumerate(channel_data_dict.items()):
                axs[plot_idx][0].plot(processed, label="Processed", color="blue")
                axs[plot_idx][0].set_title(f"{ch} (Plot)")
                axs[plot_idx][0].legend()
                axs[plot_idx][0].grid(True)

            fig.tight_layout()
            plt.show(block=False)

        except Exception as e:
            messagebox.showerror("Error", str(e))