"""
Gaze Labeling and Export Pipeline

This module orchestrates the complete workflow for manual gaze event labeling and data export.

Workflow
--------
1. Clean raw gaze/target channels (convert KINARM invalid sentinel values to NaN)
2. Launch the manual labeling UI (GazeLabeler) 
3. Convert labeled time ranges into a per-frame event vector
4. Compute gaze metrics (spherical coordinates, angular velocity, FVR)
5. Export to CSV (human-readable) and NPZ (compressed) formats

Output Files
------------
For each trial:
- Trial{index}.TP{protocol}.C{count}.csv : Human-readable data with selected channels
- Trial{index}.TP{protocol}.C{count}.npz : Compressed NumPy format
- Target_Table.csv : Reference file with workspace target definitions
- TP_Table.csv : Reference file with task protocol parameters
- Trial_Marks_and_Notes.csv : Trial quality marks and researcher notes  

Directory Structure
-------------------
<output_root>/
    <kinarm_filename>/
        Trial1.TP1.C1.csv
        Trial1.TP1.C1.npz
        Target_Table.csv
        TP_Table.csv

Notes for Developers
--------------------
- GUI-independent: receives parameters (trial_index, selected_export_channels) from caller
- Gaze calculations use Gaze_X and Gaze_Y channels (professor-confirmed)
- Computed metrics (rho, theta, phi, angular velocity, FVR) are never interpolated
- Only raw kinematic channels undergo interpolation when needed
- Eye height is fixed at 0.20 meters (lab standard)

Integration Points
------------------
- Called by: gui_task_protocol.py (main GUI)
- Uses: GazeLabeler (label.gaze_labeler_ui)
- Uses: KinarmDataExplorer.gaze_calculator (data.data_calculations)
- Uses: smart_interpolate_trial_data (data.data_interpolation)
"""

from __future__ import annotations

from pathlib import Path
import csv
import pickle
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from data.exam_load import ExamLoad
from label.gaze_labeler_ui import GazeLabeler


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# KINARM uses sentinel values (e.g., ±99.9) to mark invalid samples (blinks, tracking loss)
KINARM_INVALID_ABS_THRESHOLD = 99.9

# Alias mapping for backward compatibility and user-friendly names
GAZE_METRIC_ALIASES = {
        "Angular Velocity": "Angular_Velocity",
        "Angular Velocity (deg/s)": "Angular_Velocity",
        "Angular_Velocity": "Angular_Velocity",
        "FVR": "FVR (Foveal_Visual_Radius)",
        "FVR (mm)": "FVR (Foveal_Visual_Radius)",
        "FVR (Foveal_Visual_Radius)": "FVR (Foveal_Visual_Radius)",
        "Rho": "Rho (Distance)",
        "Rho (Distance)": "Rho (Distance)",
        "Theta": "Theta (Azimuth)",
        "Theta (Azimuth)": "Theta (Azimuth)",
        "Phi": "Phi (Elevation)",
        "Phi (Elevation)": "Phi (Elevation)",
    }


# -----------------------------------------------------------------------------
# Data Cleaning Utilities
# -----------------------------------------------------------------------------

def clean_kinarm_signal(values: Any) -> np.ndarray:
    """
    Convert input to float array and replace KINARM invalid sentinel values with NaN.
    
    KINARM systems use large sentinel values (e.g., ±99.9) to indicate invalid samples
    rather than NaN. This standardizes the representation for easier numerical processing.
    
    Parameters
    ----------
    values : array-like
        Raw channel data from KINARM system.
    
    Returns
    -------
    np.ndarray
        Float array with sentinel values replaced by NaN.
    
    Examples
    --------
    >>> data = [1.0, 2.0, 99.9, 3.0, -99.9]
    >>> clean_kinarm_signal(data)
    array([1., 2., nan, 3., nan])
    """
    arr = np.asarray(values, dtype=float).copy()
    arr[np.abs(arr) >= KINARM_INVALID_ABS_THRESHOLD] = np.nan
    return arr


def _fill_nans_linear(arr: np.ndarray) -> np.ndarray:
    """
    Fill NaN gaps via linear interpolation (flat extrapolation at boundaries).
    
    Used before derivative-based calculations to avoid propagating NaNs through
    computations. This is purely for computational purposes and does not modify
    the original data.
    
    Parameters
    ----------
    arr : np.ndarray
        Array potentially containing NaN values.
    
    Returns
    -------
    np.ndarray
        Array with NaNs filled by linear interpolation.
    
    Notes
    -----
    - Interior NaNs: linearly interpolated between valid neighbors
    - Leading/trailing NaNs: filled with nearest valid value (flat extrapolation)
    - All NaN or empty arrays: returned unchanged
    """
    arr = np.asarray(arr, dtype=float).copy().ravel()
    n = len(arr)
    
    if n == 0:
        return arr
    
    idx = np.arange(n)
    ok = np.isfinite(arr)
    
    # If all valid or all NaN, no interpolation needed
    if ok.all() or not ok.any():
        return arr
    
    # Linear interpolation (numpy handles flat extrapolation at boundaries)
    arr[~ok] = np.interp(idx[~ok], idx[ok], arr[ok])
    return arr


def _force_len(values: Any, N: int) -> np.ndarray:
    """
    Coerce input to a 1D float array of exact length N.
    
    Ensures all channel data matches the trial frame count, handling both
    raw arrays and KINARM Channel objects with a .values attribute.
    
    Parameters
    ----------
    values : array-like or Channel object
        Input data (may have .values attribute).
    N : int
        Required output length.
    
    Returns
    -------
    np.ndarray
        1D float array of length N.
    
    Behavior
    --------
    - Shorter than N: pad with NaN
    - Longer than N: truncate to N
    - Extraction fails: return array of N NaNs
    """
    try:
        # Handle Channel objects with .values attribute
        vals = getattr(values, "values", values)
        arr = np.asarray(vals, dtype=float).ravel()
    except Exception:
        arr = np.full(N, np.nan, dtype=float)

    # Adjust length to match N
    if len(arr) < N:
        arr = np.concatenate([arr, np.full(N - len(arr), np.nan, dtype=float)])
    elif len(arr) > N:
        arr = arr[:N]
    
    return arr

# -----------------------------------------------------------------------------
# Gaze Metric Computation
# -----------------------------------------------------------------------------

def compute_metrics_for_export(
    explorer,
    gaze_x: np.ndarray,
    gaze_y: np.ndarray,
    frame_rate: float,
) -> Dict[str, np.ndarray]:
    """
    Compute gaze metrics using the same methods as GUI calculation buttons.
    
    This ensures exported metrics match what users see in the GUI, maintaining
    consistency across the application.
    
    Metrics Computed
    ----------------
    - Rho (Distance): Eye-to-gaze radial distance (meters)
    - Theta (Azimuth): Horizontal angle in spherical coordinates (radians)
    - Phi (Elevation): Vertical angle in spherical coordinates (radians)
    - Angular_Velocity: Gaze angular speed magnitude (deg/s)
    - FVR (Foveal_Visual_Radius): Foveal coverage area (meters)
    
    Parameters
    ----------
    explorer : KinarmDataExplorer
        Explorer instance providing the gaze_calculator.
    gaze_x : np.ndarray
        Gaze_X channel (point-of-regard x-coordinate on stimulus plane).
    gaze_y : np.ndarray
        Gaze_Y channel (point-of-regard y-coordinate on stimulus plane).
    frame_rate : float
        Sampling rate in Hz (typically ~1000 Hz for KINARM).
    
    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping metric names to 1D arrays, or empty dict on failure.
    
    Notes
    -----
    - NaNs are linearly interpolated before computation to avoid derivative issues
    - Eye height is fixed at 0.20 meters (confirmed with professor)
    - Returns empty dict on failure (caller handles gracefully)
    """
    try:
        # Clean and interpolate input data for computation
        gx = _fill_nans_linear(clean_kinarm_signal(gaze_x))
        gy = _fill_nans_linear(clean_kinarm_signal(gaze_y))

        # Apply same 20 Hz low-pass filter used in GUI calculations
        gx = explorer.lowpass_filter(gx, cutoff=20, fs=frame_rate)
        gy = explorer.lowpass_filter(gy, cutoff=20, fs=frame_rate)

        calc = explorer.gaze_calculator

        # Step 1: Compute spherical coordinates (rho, theta, phi)
        rho, theta, phi = calc.compute_spherical_coords(gx, gy)
        
        # Step 2: Compute angular velocity magnitude
        v_deg_s, _, _, _ = calc.compute_angular_velocity(
            gx, gy, rho, phi, frame_rate_hz=frame_rate
        )
        
        # Step 3: Compute epsilon (angle between gaze direction and stimulus normal)
        epsilon_rad = calc.compute_epsilon_from_gaze_direction(gx, gy)
        
        # Step 4: Compute Foveal Visual Radius (FVR)
        fvr = calc.compute_fvr(rho, epsilon_rad)

        # Return with standardized naming
        return {
            "Rho (Distance)": np.asarray(rho, dtype=float).ravel(),
            "Theta (Azimuth)": np.asarray(theta, dtype=float).ravel(),
            "Phi (Elevation)": np.asarray(phi, dtype=float).ravel(),
            "Angular_Velocity": np.asarray(v_deg_s, dtype=float).ravel(),
            "FVR (Foveal_Visual_Radius)": np.asarray(fvr, dtype=float).ravel(),
        }
    
    except Exception as e:
        print(f"[export] Gaze metric computation failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


# -----------------------------------------------------------------------------
# File Path Generation and Reference Table Export
# -----------------------------------------------------------------------------

def get_export_path_for_trial(
    kinarm_path: str,
    trial_id: int,
    tp: Any,
    count: int,
    ext: str,
    output_root: str | Path | None = None,
) -> str:
    """
    Build standardized output file path.
    
    Output structure:
        <output_root>/<kinarm_filename>/Trial{X}.TP{Y}.C{Z}.{ext}
        NPZ files go in: <output_root>/<kinarm_filename>/npz/Trial{X}.TP{Y}.C{Z}.npz
    
    Parameters
    ----------
    kinarm_path : str
        Path to original .kinarm file.
    trial_id : int
        Trial number (1-based).
    tp : Any
        Task Protocol number.
    count : int
        Repeat count for this trial.
    ext : str
        File extension (e.g., 'csv', 'npz').
    output_root : str | Path | None
        Root directory for outputs (defaults to ~/Desktop/gaze_labels).
    
    Returns
    -------
    str
        Full path to output file.
    
    Examples
    --------
    >>> get_export_path_for_trial("data.kinarm", 3, 7, 1, "csv")
    '/home/user/Desktop/gaze_labels/data.kinarm/Trial3.TP7.C1.csv'
    >>> get_export_path_for_trial("data.kinarm", 3, 7, 1, "npz")
    '/home/user/Desktop/gaze_labels/data.kinarm/npz/Trial3.TP7.C1.npz'
    """
    # Default output location: Desktop/gaze_labels
    output_root = Path(output_root) if output_root is not None else (Path.home() / "Desktop" / "gaze_labels")
    
    # Create subdirectory named after the KINARM file
    kinarm_name_with_ext = Path(kinarm_path).name
    out_dir = output_root / kinarm_name_with_ext
    
    # Build filename
    ext = ext.lstrip(".")
    filename = f"Trial{trial_id}.TP{tp}.C{count}.{ext}"
    
    # NPZ files go in a subfolder to reduce clutter
    if ext == "npz":
        npz_dir = out_dir / "npz"
        npz_dir.mkdir(parents=True, exist_ok=True)
        return str(npz_dir / filename)
    else:
        # CSV and other files stay in root
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir / filename)


def write_tp_and_target_tables(out_dir: Path, exam: ExamLoad) -> None:
    """
    Extract and save experimental configuration reference tables.
    
    KINARM experiments use Task Protocol (TP) and Target tables to define
    experimental parameters. This function extracts those tables from the
    'common' trial and saves them as CSV files for reference.
    
    Parameters
    ----------
    out_dir : Path
        Directory where CSV files should be saved.
    exam : ExamLoad
        Loaded KINARM exam data.
    
    Output Files
    ------------
    - Target_Table.csv: Workspace target definitions (position, radii, colors)
    - TP_Table.csv: Task protocol parameters (timing, delays, targets)
    
    Notes
    -----
    - These are reference files shared across all trials
    - Only created once per export directory
    - Skipped if 'common' trial is missing or malformed
    """
    common = exam.trials.get("common")
    if common is None or not hasattr(common, "parameters"):
        return

    # Helper functions for data extraction and formatting
    def to_list(v):
        """Convert various data types to lists for consistent processing."""
        if isinstance(v, (list, tuple)):
            return list(v)
        if hasattr(v, "tolist"):
            x = v.tolist()
            return x if isinstance(x, list) else [x]
        return [v]

    def flatten(x):
        """Convert nested data structures to readable strings."""
        if hasattr(x, "tolist") and not isinstance(x, (list, tuple)):
            return x.tolist()
        if isinstance(x, (list, tuple)):
            if len(x) == 1:
                return flatten(x[0])
            return ", ".join(str(flatten(e)) for e in x)
        return x

    def extract(prefix: str) -> Dict[str, List[Any]]:
        """Extract parameters with a specific prefix from common trial."""
        pref = prefix + ":"
        out: Dict[str, List[Any]] = {}
        for k, v in common.parameters.items():
            if k.startswith(pref):
                col = k.split(":", 1)[1]
                out[col] = [flatten(e) for e in to_list(v)]
        return out

    def normalize(raw: Dict[str, List[Any]], desired: List[Tuple[str, str]], label: str) -> pd.DataFrame:
        """Convert raw parameter data into a properly formatted DataFrame."""
        if not raw:
            return pd.DataFrame()

        # Determine number of rows needed
        rows = max(len(v) for v in raw.values())
        
        # Create index column (e.g., "Target 1", "Target 2", ...)
        table = {label: [f"{label} {i+1}" for i in range(rows)]}

        # Add data columns
        for disp, src in desired:
            vals = raw.get(src, [])
            # Broadcast single values or pad with empty strings
            if len(vals) == 1 and rows > 1:
                vals = vals * rows
            elif len(vals) < rows:
                vals = vals + [""] * (rows - len(vals))
            table[disp] = vals[:rows]

        return pd.DataFrame(table)

    # Extract and format Target Table
    tgt_raw = extract("TARGET_TABLE")
    tgt_cols = [
        ("X", "X"), ("Y", "Y"),
        ("Visual Radius", "Visual Radius"),
        ("Logical Radius", "Logical Radius"),
        ("Initial Color", "Initial Color"),
        ("Reached Color", "Reached Color"),
        ("Gaze Radius", "Gaze Radius"),
    ]
    tgt_df = normalize(tgt_raw, tgt_cols, "Target")

    # Extract and format TP (Task Protocol) Table
    tp_raw = extract("TP_TABLE")
    tp_cols = [
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
    tp_df = normalize(tp_raw, tp_cols, "TP")

    # Save tables as CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    if not tgt_df.empty:
        tgt_df.to_csv(out_dir / "Target_Table.csv", index=False)
    if not tp_df.empty:
        tp_df.to_csv(out_dir / "TP_Table.csv", index=False)

def write_trial_marks_and_notes(out_dir: Path, kinarm_path: str) -> None:
    """
    Export trial quality marks and notes to a CSV summary file.
    
    This creates a human-readable CSV file containing all trial marks (good/bad/review)
    and associated notes that were added during the labeling process. This allows
    researchers to share quality assessments along with exported data.
    
    Parameters
    ----------
    out_dir : Path
        Directory where CSV file should be saved (same as trial CSVs).
    kinarm_path : str
        Path to original .kinarm file (used to locate the .notes.json file).
    
    Output File
    -----------
    - Trial_Marks_and_Notes.csv: Summary of all trial quality assessments
    
    CSV Format
    ----------
    Trial,Mark,Notes
    02_11_01,good,""
    02_11_02,bad,"Subject blinked during target presentation"
    02_11_03,review,"Check angular velocity - looks unusual"
    
    Notes
    -----
    - Only trials with marks or notes are included (empty trials omitted)
    - Silently skips if .kinarm.notes.json doesn't exist (first export scenario)
    - Notes text is properly escaped for CSV format
    - Empty notes are shown as empty strings for clarity
    """
    import json
    
    # Find the .kinarm.notes.json file
    notes_file = Path(kinarm_path).with_suffix('.kinarm.notes.json')
    
    # Skip if no marks/notes file exists yet
    if not notes_file.exists():
        return
    
    try:
        # Load marks and notes
        with open(notes_file, 'r') as f:
            trial_marks = json.load(f)
        
        # Skip if empty
        if not trial_marks:
            return
        
        # Prepare CSV data
        csv_path = out_dir / "Trial_Marks_and_Notes.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['Trial', 'Mark', 'Notes'])
            
            # Write each trial's marks/notes
            for trial_name in sorted(trial_marks.keys()):
                mark_data = trial_marks[trial_name]
                
                # Handle both old string format and new dict format
                if isinstance(mark_data, str):
                    mark = mark_data
                    notes = ""
                elif isinstance(mark_data, dict):
                    mark = mark_data.get("mark", "")
                    notes = mark_data.get("notes", "")
                else:
                    continue  # Skip malformed entries
                
                # Only include trials that have a mark or notes
                if mark or notes:
                    writer.writerow([trial_name, mark, notes])
        
        print(f"✓ Trial marks and notes saved to: {csv_path}")
        
    except Exception as e:
        # Silently fail - don't block export if marks file is corrupted
        print(f"Note: Could not export trial marks/notes: {e}")


# -----------------------------------------------------------------------------
# Main Export Pipeline
# -----------------------------------------------------------------------------

def run_labeling_process(
    explorer,
    trial_name: str,
    gaze_x,
    gaze_y,
    overlay_channels: Dict[str, np.ndarray],
    selected_export_channels: List[str],
    kinarm_path: str | None = None,
    trial_info: str | None = None,
    output_root: str | Path | None = None,
    trial_index: int | None = None,
    selected_events: List[str] | None = None,
    marker_events: List[str] | None = None,
    label_order: List[str] | None = None,
):
    """
    Complete pipeline: clean → label → compute metrics → export CSV/NPZ.
    
    This is the main entry point for the gaze labeling and export workflow.
    It coordinates all steps from data cleaning through file export.
    
    Parameters
    ----------
    explorer : KinarmDataExplorer
        Data explorer instance with loaded exam.
    trial_name : str
        Name/ID of the trial to process.
    gaze_x, gaze_y : array-like
        Raw gaze position data.
    xT, yT : array-like
        Target position data.
    selected_export_channels : list[str]
        List of data channels to include in output files.
    kinarm_path : str, optional
        Path to original .kinarm file (auto-detected from explorer if not provided).
    trial_info : str, optional
        Formatted string with trial details for UI display.
    output_root : str | Path, optional
        Root directory for output files (defaults to ~/Desktop/gaze_labels).
    trial_index : int
        Trial number for file naming (required from GUI).
    selected_events : list[str], optional
        Event labels to export as columns (e.g., ["TARGET_ON", "BLINK_START"]).
    
    Returns
    -------
    (action, labeler) : tuple
        - action: "accept" or "next_trial" (from UI), or False on cancel/failure
        - labeler: GazeLabeler instance (contains flags like bad_trial)
    
    Workflow Steps
    --------------
    1. Clean raw channels (convert sentinel values to NaN)
    2. Extract trial metadata (TP number, event frames)
    3. Compute gaze metrics (rho, theta, phi, angular velocity, FVR)
    4. Launch manual labeling UI
    5. Convert labeled time ranges to per-frame event vector
    6. Interpolate raw channels (not computed metrics)
    7. Export to CSV and NPZ formats
    8. Save reference tables (Target_Table.csv, TP_Table.csv)
    
    Gaze Event Codes
    ----------------
    - 0: Unlabeled / Target appearance instant
    - 1: Saccade
    - 2: Pursuit
    - 3: Fixation
    - 9: Bad trial (entire trial marked as unusable)
    
    Notes for Future Developers
    ----------------------------
    - Computed metrics are never interpolated (only raw channels)
    - Event columns store frame numbers in row 0 as semicolon-separated lists
    - Trial marked "bad" exports all frames as gaze_event=9
    """
    try:
        # ---------------------------------------------------------------------
        # Validate inputs and extract metadata
        # ---------------------------------------------------------------------
        
        if kinarm_path is None:
            kinarm_path = getattr(explorer, "filepath", None)
            if not kinarm_path:
                raise ValueError("kinarm_path not provided and explorer has no .filepath")

        if trial_index is None:
            raise ValueError("run_labeling_process() requires trial_index from GUI")

        selected_events = selected_events or []
        selected_export_channels = selected_export_channels or []

        # Extract repeat count from trial name (e.g., "TP1_2" → count=2)
        try:
            count = int(trial_name.split("_")[-1])
        except Exception:
            count = 1

        trial = explorer.exam.trials[trial_name]
        explorer.current_trial = trial
        exam = explorer.exam
        N = int(trial.frame_count)

        # ---------------------------------------------------------------------
        # Compute gaze metrics
        # ---------------------------------------------------------------------
        
        # Only compute gaze metrics if at least one is selected for export
        metric_names = set(GAZE_METRIC_ALIASES.values())
        needs_metrics = any(
            GAZE_METRIC_ALIASES.get(ch, ch) in metric_names
            for ch in selected_export_channels
        )

        if needs_metrics:
            gaze_metrics = compute_metrics_for_export(explorer, gaze_x, gaze_y, float(trial.frame_rate))
            for k in list(gaze_metrics.keys()):
                gaze_metrics[k] = _force_len(gaze_metrics[k], N)
        else:
            gaze_metrics = {}

        # ---------------------------------------------------------------------
        # Extract TP number for file naming
        # ---------------------------------------------------------------------
        
        tp_num = "NA"
        try:
            for k, v in trial.parameters.items():
                if "TP" in k.upper():
                    if isinstance(v, (list, tuple)) and len(v) > 0:
                        tp_num = v[0]
                    elif hasattr(v, "tolist"):
                        tv = v.tolist()
                        tp_num = tv[0] if isinstance(tv, list) else tv
                    elif isinstance(v, int):
                        tp_num = v
                    else:
                        tp_num = int(v[0])
                    break
        except Exception:
            tp_num = "NA"

        # ---------------------------------------------------------------------
        # Extract event frames for optional export columns
        # ---------------------------------------------------------------------
        
        event_frames: Dict[str, List[int]] = {}
        for ev in selected_events:
            matches = [e for e in trial.events if e.label.upper() == ev.upper()]
            if matches:
                frames = sorted([int(round(e.time * trial.frame_rate)) for e in matches])
                event_frames[ev] = frames

        # ---------------------------------------------------------------------
        # Build marker frames for plotting (vertical lines) - USER SELECTED
        # ---------------------------------------------------------------------
        marker_events = marker_events or []
        marker_frames: Dict[str, List[int]] = {}

        for ev in marker_events:
            matches = [e for e in trial.events if e.label.upper() == ev.upper()]
            if matches:
                marker_frames[ev] = sorted([int(round(e.time * trial.frame_rate)) for e in matches])

        # ---------------------------------------------------------------------
        # Launch manual labeling UI
        # ---------------------------------------------------------------------
        
        labeler = GazeLabeler(
            trial_name,
            gaze_x,
            gaze_y,
            overlay_channels=overlay_channels,
            trial_info=trial_info,
            marker_frames=marker_frames,
            label_order=label_order,
        )
        label_ranges, action = labeler.plot_and_select_range()

        # Check for user cancellation
        if label_ranges is None or getattr(labeler, "cancel_all", False):
            print("Labeling cancelled by user. No files were written.")
            return False, None

        # ---------------------------------------------------------------------
        # Convert labeled time ranges to per-frame event vector
        # ---------------------------------------------------------------------
        
        if labeler.bad_trial:
            # Trial marked as "bad" - export all frames as 9
            gaze_events = np.full(N, 9, dtype=int)
        else:
            # Normal labeling: assign codes 1, 2, 3
            gaze_events = np.zeros(N, dtype=int)
            label_code = {"saccade": 1, "pursuit": 2, "fixation": 3, "other":0}

            for name, time_ranges in label_ranges.items():
                code = label_code.get(name, 0)
                for start_frame, end_frame in time_ranges:
                    start_frame = max(0, int(start_frame))
                    end_frame = min(N - 1, int(end_frame))
                    if start_frame <= end_frame:
                        gaze_events[start_frame:end_frame + 1] = code

        # Build set of computed metric keys (never interpolate these)
        metric_key_set = set(gaze_metrics.keys())
        metric_key_set.update(GAZE_METRIC_ALIASES.values())

        # Identify which channels need interpolation (raw data only)
        channels_to_interpolate = []
        for ch in selected_export_channels:
            if ch in ("Frame", "Gaze_Events"):
                continue  # Skip special columns
            key = GAZE_METRIC_ALIASES.get(ch, ch)
            if key in metric_key_set:
                continue  # Skip computed metrics
            if ch in trial.kinematics:
                channels_to_interpolate.append(ch)

        # Perform interpolation if needed
        if channels_to_interpolate:
            interpolated_export_data = explorer.smart_interpolate_trial_data(channels_to_interpolate)
            if interpolated_export_data is None:
                return False, None  # User cancelled interpolation
        else:
            interpolated_export_data = {}

        def get_channel_series(ch_name: str) -> np.ndarray:
            """
            Resolve a requested export channel into a 1D array of length N.
            
            Resolution Priority
            -------------------
            1. Interpolated export data (raw channels)
            2. Computed gaze metrics (with aliasing)
            3. Raw trial.kinematics channel
            4. Derived channel via explorer (if supported)
            5. Fallback to NaNs
            """
            # Priority 1: Interpolated export data
            if ch_name in interpolated_export_data:
                return _force_len(interpolated_export_data[ch_name], N)

            # Priority 2: Computed gaze metrics
            key = GAZE_METRIC_ALIASES.get(ch_name, ch_name)
            if key in gaze_metrics:
                return _force_len(gaze_metrics[key], N)

            # Priority 3: Raw kinematics
            if ch_name in trial.kinematics:
                return _force_len(trial.kinematics[ch_name], N)

            # Priority 4: Derived channels
            try:
                derived = explorer._compute_derived_channel(ch_name)
                if derived and len(derived[0]) == N:
                    return _force_len(derived[0], N)
            except Exception:
                pass

            # Priority 5: Fallback
            return np.full(N, np.nan, dtype=float)

        # ---------------------------------------------------------------------
        # Generate output file paths
        # ---------------------------------------------------------------------
        
        csv_filename = get_export_path_for_trial(
            kinarm_path, trial_index, tp_num, count, "csv", output_root=output_root
        )
        npz_filename = get_export_path_for_trial(
            kinarm_path, trial_index, tp_num, count, "npz", output_root=output_root
        )

        out_dir = Path(csv_filename).parent
        
        # Write reference tables (once per output directory)
        write_tp_and_target_tables(out_dir, exam)
        # Write trial marks and notes summary
        write_trial_marks_and_notes(out_dir, kinarm_path)

        # ---------------------------------------------------------------------
        # Export to CSV (human-readable format)
        # ---------------------------------------------------------------------
        
        # Build CSV headers
        headers = (
            ["Frame", "Gaze_Events"]
            + [ch for ch in selected_export_channels if ch not in ("Frame", "Gaze_Events")]
            + list(event_frames.keys())
        )

        # Precompute all channel arrays for performance
        channel_arrays = {
            ch: get_channel_series(ch)
            for ch in selected_export_channels
            if ch not in ("Frame", "Gaze_Events")
        }

        # Write CSV
        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for i in range(N):
                row = [i, int(gaze_events[i])]

                # Add selected channel values
                for ch in selected_export_channels:
                    if ch in ("Frame", "Gaze_Events"):
                        continue
                    row.append(channel_arrays[ch][i])

                # Add event columns (frame lists in row 0 only)
                if i == 0:
                    for ev in event_frames:
                        row.append(";".join(str(fr) for fr in event_frames[ev]))
                else:
                    for _ in event_frames:
                        row.append("")

                writer.writerow(row)

        # ---------------------------------------------------------------------
        # Export to NPZ (compressed NumPy format)
        # ---------------------------------------------------------------------
        
        save_dict: Dict[str, np.ndarray] = {
            "Frame": np.arange(N, dtype=int),
            "Gaze_Events": gaze_events.astype(int),
        }
        
        # Add event frame arrays
        for ev, frames_list in event_frames.items():
            save_dict[ev] = np.asarray(frames_list, dtype=int)

        np.savez_compressed(npz_filename, **save_dict)

        # Report success
        print(f"CSV saved to: {csv_filename}")
        print(f"Compressed .npz saved to: {npz_filename}")
        print(f"All files saved to: {out_dir}")

        return action, labeler

    except Exception:
        import traceback
        print("\nExport pipeline failed with exception:")
        traceback.print_exc()
        return False, None


# -----------------------------------------------------------------------------
# Command-Line Interface (for backwards compatibility)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Command-line entry point for standalone usage.
    
    Usage
    -----
    python gaze_labeler_export.py <data.pkl> <path_to_kinarm_file>
    
    The data.pkl file should contain:
        (trial_name, gaze_x, gaze_y, xT, yT, selected_export_channels)
    
    This interface is maintained for backwards compatibility with older scripts
    and automated workflows. Most users should use the GUI (gui_task_protocol.py).
    """
    try:
        if len(sys.argv) < 3:
            raise ValueError("Usage: gaze_labeler_export.py <data.pkl> <path_to_kinarm_file>")

        data_path = sys.argv[1]
        kinarm_path = sys.argv[2]

        # Load pickled data
        with open(data_path, "rb") as f:
            trial_name, gaze_x, gaze_y, xT, yT, selected_export_channels = pickle.load(f)

        # Create explorer instance
        from data.data_loader import KinarmDataExplorer
        explorer = KinarmDataExplorer(kinarm_path)

        # Run export pipeline
        result = run_labeling_process(
            explorer,
            trial_name,
            gaze_x,
            gaze_y,
            xT,
            yT,
            selected_export_channels,
            kinarm_path=kinarm_path,
            trial_index=1,  # CLI default
        )

        sys.exit(0 if result and result[0] else 1)

    except Exception:
        import traceback
        print("\nCLI runner failed with exception:")
        traceback.print_exc()
        sys.exit(1)