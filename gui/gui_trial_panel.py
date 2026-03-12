"""
Trial Panel - Trial marking, notes, and list management.
"""

import tkinter as tk
from tkinter import messagebox
from gui.gui_utils import center_window


class TrialPanel:
    """Handles trial marking, notes, and trial list display."""

    def __init__(self, app):
        self.app = app

    def load_trial_marks(self):
        """Load trial marks from CSV via SessionManager."""
        self.app.session.load_trial_marks()
        self.app._trial_marks = self.app.session.trial_marks

    def save_trial_marks(self):
        """Save trial marks to CSV via SessionManager."""
        self.app.session.trial_marks = self.app._trial_marks
        self.app.session.save_trial_marks(
            self.app.explorer.trial_names if self.app.explorer else []
        )

    def refresh_trial_list(self):
        """Rebuild the trial listbox, showing marks and note indicators, and restore previous selection."""
        if not self.app.explorer:
            return

        sel = self.app.trial_listbox.curselection()
        current_index = sel[0] if sel else None

        self.app.trial_listbox.delete(0, tk.END)
        for idx, name in enumerate(self.app.explorer.trial_names, start=1):
            mark_data = self.app._trial_marks.get(name, "")

            if isinstance(mark_data, dict):
                mark = mark_data.get("mark", "")
                has_notes = bool(mark_data.get("notes", "").strip())
            else:
                mark = mark_data
                has_notes = False

            mark_text = f"  [{mark.title()}]" if mark else ""
            notes_indicator = " [Notes]" if has_notes else ""

            display_name = "_".join(name.split("_")[1:]) if "_" in name else name
            self.app.trial_listbox.insert(tk.END, f"{idx}. {display_name}{mark_text}{notes_indicator}")

        if current_index is not None:
            self.app.trial_listbox.selection_set(current_index)

    def mark_trial(self, status):
        """Mark the currently selected trial with a quality status."""
        sel = self.app.trial_listbox.curselection()
        if not sel:
            messagebox.showwarning("No Selection", "Please select a trial first.")
            return

        index = sel[0]
        trial_name = self.app.explorer.trial_names[index]

        existing_notes = ""
        if trial_name in self.app._trial_marks:
            mark_data = self.app._trial_marks[trial_name]
            if isinstance(mark_data, dict):
                existing_notes = mark_data.get("notes", "")

        if status is None and not existing_notes:
            self.app._trial_marks.pop(trial_name, None)
        else:
            self.app._trial_marks[trial_name] = {
                "mark": status,
                "notes": existing_notes
            }

        self.save_trial_marks()
        self.refresh_trial_list()

    def add_trial_notes(self):
        """Open dialog to add/edit notes for the currently selected trial."""
        sel = self.app.trial_listbox.curselection()
        if not sel:
            messagebox.showwarning("No Selection", "Please select a trial first.")
            return

        index = sel[0]
        trial_name = self.app.explorer.trial_names[index]

        existing_notes = ""
        if trial_name in self.app._trial_marks:
            mark_data = self.app._trial_marks[trial_name]
            if isinstance(mark_data, dict):
                existing_notes = mark_data.get("notes", "")

        notes_window = tk.Toplevel(self.app.root)
        notes_window.title(f"Notes for Trial {trial_name}")
        notes_window.resizable(True, True)

        tk.Label(notes_window, text=f"Notes for {trial_name}:", font=self.app.bold_font).pack(pady=10)

        text_frame = tk.Frame(notes_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            font=self.app.default_font
        )
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)

        text_widget.insert("1.0", existing_notes)
        text_widget.focus()

        btn_frame = tk.Frame(notes_window)
        btn_frame.pack(pady=10)

        def save_notes():
            notes_text = text_widget.get("1.0", tk.END).strip()

            existing_mark = None
            if trial_name in self.app._trial_marks:
                mark_data = self.app._trial_marks[trial_name]
                if isinstance(mark_data, str):
                    existing_mark = mark_data
                elif isinstance(mark_data, dict):
                    existing_mark = mark_data.get("mark")

            if notes_text or existing_mark:
                self.app._trial_marks[trial_name] = {
                    "mark": existing_mark,
                    "notes": notes_text
                }
            else:
                self.app._trial_marks.pop(trial_name, None)

            self.save_trial_marks()
            self.refresh_trial_list()
            notes_window.destroy()

        def cancel():
            notes_window.destroy()

        tk.Button(btn_frame, text="Save", command=save_notes, width=15, bg="#90EE90").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=cancel, width=15).pack(side=tk.LEFT, padx=5)

        notes_window.update_idletasks()
        req_w = notes_window.winfo_reqwidth()
        req_h = notes_window.winfo_reqheight()
        width = max(600, req_w + 40)
        height = max(400, req_h + 40)
        center_window(notes_window, width=width, height=height)
        notes_window.minsize(width, height)