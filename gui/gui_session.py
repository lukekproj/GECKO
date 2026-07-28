"""
Session & Trial Notes Manager

Handles persistence of GUI state and trial quality marks for the KINARM
Data Explorer. All file I/O for session resume and trial notes lives here,
keeping the main GUI class focused on UI logic.

Files managed:
- session_state.json : GUI state (selected trial, channel/export/marker
  selections, filter text) saved per .kinarm file.
- Trial_Notes.csv : Trial quality marks (Good/Bad/Review) and free-text
  notes saved per .kinarm file.

Both files are stored in a subfolder named after the .kinarm file inside
the user's chosen save location (or the default Desktop/gaze_labels).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


class SessionManager:
    """
    Manages session state and trial notes persistence.

    This class owns all file I/O for saving/loading session state and trial
    quality marks. It has no tkinter dependencies — the GUI passes in the
    data it needs saved and receives plain dicts back.

    Parameters
    ----------
    get_explorer : callable
        Returns the current KinarmDataExplorer instance (or None).
    get_save_location : callable
        Returns the current save location string (or None for default).
    """

    _desktop = Path.home() / "Desktop"
    DEFAULT_SAVE_DIR = _desktop / "gaze_labels" if _desktop.exists() else Path.home() / "gaze_labels"

    def __init__(self, get_explorer, get_save_location):
        self._get_explorer = get_explorer
        self._get_save_location = get_save_location
        self.trial_marks: Dict[str, dict] = {}

    # -----------------------------------------------------------------
    # Path helpers
    # -----------------------------------------------------------------

    def _output_dir(self) -> Optional[Path]:
        """
        Build the output directory for the currently loaded .kinarm file.

        Returns
        -------
        Path or None
            ``<save_location>/<kinarm_filename>/``, or None if no file is loaded.
        """
        explorer = self._get_explorer()
        if not explorer or not explorer.filepath:
            return None

        save_root = self._resolve_save_root()
        kinarm_filename = Path(explorer.filepath).name
        out_dir = save_root / kinarm_filename
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _resolve_save_root(self) -> Path:
        """Return the active save root, falling back to the default."""
        loc = self._get_save_location()
        return Path(loc) if loc else self.DEFAULT_SAVE_DIR

    def session_state_path(self) -> Optional[Path]:
        """
        Path to session_state.json for the loaded file.

        Returns None if no file is loaded.
        """
        out = self._output_dir()
        return out / "session_state.json" if out else None

    def notes_csv_path(self) -> Optional[Path]:
        """
        Path to Trial_Notes.csv for the loaded file.

        Returns None if no file is loaded.
        """
        out = self._output_dir()
        return out / "Trial_Notes.csv" if out else None

    # -----------------------------------------------------------------
    # Session state (JSON)
    # -----------------------------------------------------------------

    def save_state(
        self,
        *,
        current_trial_name: Optional[str],
        trial_names: List[str],
        filepath: str,
        channel_filter: str = "",
        inspect_selection: Optional[set] = None,
    ) -> None:
        """
        Write current GUI state to session_state.json.

        Parameters
        ----------
        current_trial_name : str or None
            Name of the currently selected trial.
        trial_names : list[str]
            Ordered list of all trial names (for index lookup).
        filepath : str
            Path to the loaded .kinarm file.
        channel_filter : str
            Current search/filter text.
        inspect_selection : set or None
            Sticky channel inspection selections (by name).
        """
        p = self.session_state_path()
        if not p or not current_trial_name:
            return

        trial_index = None
        try:
            trial_index = trial_names.index(current_trial_name)
        except ValueError:
            pass

        state = {
            "kinarm_filepath": filepath,
            "kinarm_file_id": Path(filepath).name,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "trial_name": current_trial_name,
            "trial_index": trial_index,
            "channel_filter": channel_filter,
            "inspect_selection": sorted(inspect_selection or []),
        }

        try:
            p.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except Exception:
            pass

    def load_state(self) -> Optional[dict]:
        """
        Read session state from disk.

        Returns
        -------
        dict or None
            The saved state dictionary, or None if the file doesn't exist or
            can't be parsed.
        """
        p = self.session_state_path()
        if not p or not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def delete_state(self) -> bool:
        """
        Delete the session_state.json file.

        Returns
        -------
        bool
            True if deleted, False if not found or error.
        """
        p = self.session_state_path()
        if p and p.exists():
            try:
                p.unlink()
                return True
            except Exception:
                return False
        return False

    # -----------------------------------------------------------------
    # Trial marks / notes (CSV)
    # -----------------------------------------------------------------

    def load_trial_marks(self) -> Dict[str, dict]:
        """
        Load trial quality marks and notes from Trial_Notes.csv.

        Returns
        -------
        dict[str, dict]
            Mapping of trial_name -> {"mark": str|None, "notes": str}.
            Returns empty dict if no CSV exists.
        """
        csv_path = self.notes_csv_path()
        if not csv_path or not csv_path.exists():
            self.trial_marks = {}
            return self.trial_marks

        try:
            df = pd.read_csv(csv_path)
            marks: Dict[str, dict] = {}

            for _, row in df.iterrows():
                trial_name = row["Trial_Name"]
                mark = row.get("Mark", "")
                notes = row.get("Notes", "")

                if pd.isna(mark) or mark == "":
                    mark = None
                if pd.isna(notes):
                    notes = ""

                marks[trial_name] = {"mark": mark, "notes": notes}

            self.trial_marks = marks
            print(f"Loaded trial notes from CSV: {csv_path}")

        except Exception as e:
            print(f"Error loading trial notes from CSV: {e}")
            self.trial_marks = {}

        return self.trial_marks

    def save_trial_marks(self, trial_names: List[str]) -> None:
        """
        Write trial quality marks and notes to Trial_Notes.csv.

        Parameters
        ----------
        trial_names : list[str]
            Ordered list of all trial names. Every trial gets a row in the CSV
            even if it has no mark or notes.
        """
        csv_path = self.notes_csv_path()
        if not csv_path:
            return

        try:
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            rows = []
            for trial_name in trial_names:
                mark_data = self.trial_marks.get(trial_name, {"mark": None, "notes": ""})

                # Handle legacy string format
                if isinstance(mark_data, str):
                    mark = mark_data
                    notes = ""
                elif isinstance(mark_data, dict):
                    mark = mark_data.get("mark")
                    notes = mark_data.get("notes", "")
                else:
                    mark = None
                    notes = ""

                if mark:
                    mark = mark.capitalize()

                rows.append({
                    "Trial_Name": trial_name,
                    "Mark": mark if mark else "",
                    "Notes": notes,
                })

            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False, encoding="utf-8")
            print(f"Saved trial notes to: {csv_path}")

        except Exception as e:
            print(f"Error saving trial marks to CSV: {e}")