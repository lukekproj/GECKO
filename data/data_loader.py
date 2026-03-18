"""
KINARM Data Loader / Explorer

This module defines `KinarmDataExplorer`, the core object used by the GUI to:
- Load a .kinarm file (via ExamLoad)
- Track the currently selected trial
- Provide convenience methods for:
    - smart interpolation (with session caching)
    - derived channel computation (with per-trial caching)
    - gaze-related calculations (delegated to data_calculations.py)

Design note:
This class acts as the glue between the GUI and processing modules.
The math-heavy code lives in `data_calculations.py` and interpolation lives in
`data_interpolation.py` to keep responsibilities clean and maintainable.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from tkinter import messagebox
from scipy.signal import butter, filtfilt

from data.exam_load import ExamLoad
from data.data_interpolation import smart_interpolate_trial_data
from data.data_calculations import (
    calculate_gaze_metrics,
    calculate_angular_velocity,
    calculate_fvr,
    GazeCalculator
)
from utility.user_prefs import (
    DEFAULT_GAZE_LOWPASS_ORDER,
    DEFAULT_GAZE_SAMPLING_HZ,
    DEFAULT_GAZE_LOWPASS_CUTOFF_HZ
)
from dataclasses import dataclass

@dataclass
class DerivedChannel:
    """Container for a computed kinematic channel and its unit label."""
    values: np.ndarray
    unit: str

class KinarmDataExplorer:
    """
    Main interface for loading and exploring KINARM experimental data.

    Responsibilities:
    - Load the .kinarm archive into an Exam object (ExamLoad)
    - Maintain a list of trial names in a consistent ordering (Dexterit-E style)
    - Provide interpolation and derived channels for the currently selected trial
    - Delegate gaze calculations (rho/theta/phi, angular velocity, FVR)

    Notes for maintainers:
    - Interpolation results are cached per (trial_name, channel_name) for the current session.
    - Derived channel results are cached per trial so repeated GUI requests are fast.
    """

    # Known derived channel names. Labs can add entries here and in _compute_derived_channel() to register new derivations.
    DERIVED_CHANNEL_NAMES = [
        "Right_HandX", "Right_HandY", "Right_HandSpeed",
        "Right_HandVelX", "Right_HandVelY", "Right_HandAccX", "Right_HandAccY",
        "Right_HandCmdFX", "Right_HandCmdFY",
        "Left_HandX", "Left_HandY", "Left_HandSpeed",
        "Left_HandVelX", "Left_HandVelY", "Left_HandAccX", "Left_HandAccY",
        "Left_HandCmdFX", "Left_HandCmdFY",
        "FP1_CoPX", "FP1_CoPY",
    ]

    def __init__(self, filepath: str):
        """
        Initialize the explorer for a specific .kinarm file.

        Parameters
        ----------
        filepath : str
            Path to the .kinarm file.
        """
        self.filepath: str = filepath

        # Loaded exam container (ExamLoad instance)
        self.exam: Optional[ExamLoad] = None

        # Trial names in the same order as they appear in-file (Dexterit-E style)
        self.trial_names: List[str] = []

        # Currently selected trial object
        self.current_trial = None

        # Interpolation caches for current session
        # (trial_name, channel_name) -> np.ndarray (array of interpolated values)
        self.interpolation_cache: Dict[Tuple[str, str], np.ndarray] = {}

        # Derived channel cache per trial:
        # {trial_name: {derived_channel_name: (data_array, unit_str)}}
        self._derived_cache: Dict[str, Dict[str, Tuple[np.ndarray, str]]] = {}

        # Gaze calculator helper (math lives in data_calculations.py)
        self.gaze_calculator = GazeCalculator()

        self._load_exam()

    def _load_exam(self) -> None:
        """
        Load the KINARM exam file and extract ordered trial names.

        Trials are ordered by their physical layout in the file to match Dexterit-E ordering.

        Implementation detail:
        ExamLoad stores raw file entry order internally. We read that order to preserve the exact trial ordering seen in Dexterit-E.

        If ExamLoad changes internally in the future, this is the first place to check.
        """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")

        self.exam = ExamLoad(self.filepath)
        if not getattr(self.exam, "trials", None):
            raise ValueError("No trials found in the KINARM file")

        seen = set()
        ordered_trial_names: List[str] = []

        # NOTE: This accesses a name-mangled attribute to preserve file ordering.
        # It is intentional, documented, and matches Dexterit-E.
        exam_entries = getattr(self.exam, "_ExamLoad__exam_data", [])

        for entry in exam_entries:
            if entry.startswith("raw/") and not entry.startswith("raw/common/"): # Skip common/ folder
                trial_name = entry.split("/")[1]
                if trial_name in seen: # Skip duplicates
                    continue

                trial = self.exam.trials.get(trial_name)
                if trial and trial.frame_count > 0:
                    ordered_trial_names.append(trial_name)
                    seen.add(trial_name)

        self.trial_names = ordered_trial_names

    def list_channels(self) -> List[str]:
        """
        Returns a combined list of available kinematic and derived channel names.

        Returns
        -------
        list[str]
            Concatenation of raw kinematic channel names and known derived
            channel names.
        """
        if not self.current_trial:
            return []
        return list(self.current_trial.kinematics.keys()) + self.DERIVED_CHANNEL_NAMES

    def get_interpolated_gaze_data(self, channels=None) -> Optional[Dict[str, np.ndarray]]:
        """
        Get core gaze + target channels with smart interpolation applied.

        Channels returned:
        - Gaze_X, Gaze_Y : gaze position channels (used for calculations + labeling)

        Returns
        -------
        dict[str, np.ndarray] | None
            Returns None if:
            - no trial selected
            - required channels missing
            - user cancels interpolation selection
        """
        if not self.current_trial:
            return None

        gaze_channels = channels if channels is not None else ["Gaze_X", "Gaze_Y"]

        missing = [ch for ch in gaze_channels if ch not in self.current_trial.kinematics]
        if missing:
            messagebox.showerror("Missing Data", f"Required channels not found: {missing}")
            return None

        interpolated = smart_interpolate_trial_data(self, gaze_channels)
        return interpolated

    def lowpass_filter(self, data, cutoff: float = DEFAULT_GAZE_LOWPASS_CUTOFF_HZ, fs: float = DEFAULT_GAZE_SAMPLING_HZ, order: int = DEFAULT_GAZE_LOWPASS_ORDER) -> np.ndarray:
        """
        Apply a low-pass Butterworth filter to reduce high-frequency noise.

        Important behavior:
        - NaNs are temporarily interpolated so filtfilt can run.
        - NaNs are restored to original positions after filtering.

        Parameters
        ----------
        data : array-like
            Input signal.
        cutoff : float
            Cutoff frequency (Hz). 
        fs : float
            Sampling frequency (Hz). Defaults to DEFAULT_GAZE_LOWPASS_SAMPLING_HZ -- 
            prefer passing trial.frame_rate directly where possible.
        order : int
            Filter order.

        Returns
        -------
        np.ndarray
            Filtered signal (same length).
        """
        data_clean = np.asarray(data, dtype=float)
        nan_mask = np.isnan(data_clean)

        if np.all(nan_mask):
            return data_clean  # all NaN: nothing to filter

        # Fill NaNs for filtering
        if np.any(nan_mask):
            idx = np.arange(len(data_clean))
            data_clean = data_clean.copy()
            data_clean[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], data_clean[~nan_mask])

        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(int(np.ceil(order / 2)), normal_cutoff, btype="low", analog=False)

        filtered = filtfilt(b, a, data_clean)
        filtered[nan_mask] = np.nan
        return filtered

    def _compute_derived_channel(self, name: str) -> Optional[Tuple[np.ndarray, str]]:
        """
        Compute a derived channel from raw trial data.

        Derived channels are computed on-demand and cached per trial.

        Parameters
        ----------
        name : str
            Derived channel name.

        Returns
        -------
        (np.ndarray, str) | None
            Tuple of (values, unit) or None if not computable.
        """
        t = self.current_trial
        if t is None:
            return None

        trial_name = t.name
        trial_cache = self._derived_cache.setdefault(trial_name, {})
        if name in trial_cache:
            return trial_cache[name]

        fs = t.frame_rate
        dt = 1.0 / fs

        def deriv(signal: np.ndarray) -> np.ndarray:
            """Numerical derivative of a kinematic signal."""
            return np.gradient(signal, dt)

        def magnitude(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            """2D Euclidean vector magnitude."""
            return np.sqrt(x**2 + y**2)

        try:     
            right_available = "Right_Hand" in t.positions
            left_available = "Left_Hand" in t.positions

            if right_available:
                Rx = np.array([pt[0] for pt in t.positions["Right_Hand"].values], dtype=float)
                Ry = np.array([pt[1] for pt in t.positions["Right_Hand"].values], dtype=float)
            if left_available:
                Lx = np.array([pt[0] for pt in t.positions["Left_Hand"].values], dtype=float)
                Ly = np.array([pt[1] for pt in t.positions["Left_Hand"].values], dtype=float)

            derived = {}
            # Lazy evaluation: lambdas defer computation until the requested channel is actually accessed, 
            # avoiding unnecessary derivative and filtering operations for unrequested channels.
            if right_available:
                derived.update({
                "Right_HandX": lambda: (Rx, "mm"),
                "Right_HandY": lambda: (Ry, "mm"),
                "Right_HandVelX": lambda: (deriv(Rx), "mm/s"),
                "Right_HandVelY": lambda: (deriv(Ry), "mm/s"),
                "Right_HandAccX": lambda: (deriv(deriv(Rx)), "mm/s²"),
                "Right_HandAccY": lambda: (deriv(deriv(Ry)), "mm/s²"),
                "Right_HandSpeed": lambda: (magnitude(deriv(Rx), deriv(Ry)), "mm/s"),
                "Right_HandCmdFX": lambda: (np.asarray(t.kinematics["Right_M1TorCMD"].values, dtype=float), "N·m"),
                "Right_HandCmdFY": lambda: (np.asarray(t.kinematics["Right_M2TorCMD"].values, dtype=float), "N·m")
                })

            if left_available:
                derived.update({
                "Left_HandX": lambda: (Lx, "mm"),
                "Left_HandY": lambda: (Ly, "mm"),
                "Left_HandVelX": lambda: (deriv(Lx), "mm/s"),
                "Left_HandVelY": lambda: (deriv(Ly), "mm/s"),
                "Left_HandAccX": lambda: (deriv(deriv(Lx)), "mm/s²"),
                "Left_HandAccY": lambda: (deriv(deriv(Ly)), "mm/s²"),
                "Left_HandSpeed": lambda: (magnitude(deriv(Lx), deriv(Ly)), "mm/s"),
                "Left_HandCmdFX": lambda: (np.asarray(t.kinematics["Left_M1TorCMD"].values, dtype=float), "N·m"),
                "Left_HandCmdFY": lambda: (np.asarray(t.kinematics["Left_M2TorCMD"].values, dtype=float), "N·m")
                })

            # Force plate center-of-pressure (if available)
            derived.update({
                "FP1_CoPX": lambda: (np.asarray(t.kinematics["FP1_MX"].values, dtype=float), "mm"),
                "FP1_CoPY": lambda: (np.asarray(t.kinematics["FP1_MY"].values, dtype=float), "mm")
            })

            factory = derived.get(name)
            if factory is not None:
                result = factory()
                trial_cache[name] = result
                return result
            return None

        except KeyError:
            # Missing kinematic keys for this specific trial/protocol
            return None
        except Exception as e:
            print(f"Error computing derived channel {name}: {e}")
            return None

    def calculate_gaze_metrics(self):
        """Wrapper so GUI can call explorer.calculate_gaze_metrics()."""
        return calculate_gaze_metrics(self)

    def calculate_angular_velocity(self):
        """Wrapper so GUI can call explorer.calculate_angular_velocity()."""
        return calculate_angular_velocity(self)

    def calculate_fvr(self):
        """Wrapper so GUI can call explorer.calculate_fvr()."""
        return calculate_fvr(self)

    def smart_interpolate_trial_data(self, channel_names, auto_threshold: int = 50, force_prompt: bool = False, trial_info=None):
        """
        Wrapper so GUI can call explorer.smart_interpolate_trial_data().
        """
        return smart_interpolate_trial_data(self, channel_names, auto_threshold, force_prompt, trial_info)