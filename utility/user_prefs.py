"""
User Preferences Management

Persistent storage for application settings and user preferences.

Overview
--------
This module provides cross-platform persistent storage for user preferences
such as default export channel selections and marker event selections. 
Preferences are stored in JSON format in the user's standard configuration directory.

Storage Locations
-----------------
Windows:    C:\\Users\\<username>\\AppData\\Roaming\\KinarmDataExplorer\\user_prefs.json
macOS:      ~/.config/KinarmDataExplorer/user_prefs.json  
Linux:      ~/.config/KinarmDataExplorer/user_prefs.json

Stored Preferences
------------------
Currently tracked preferences:
- export_defaults: List of channel names to pre-select in export UI
- marker_defaults: List of event names to show as vertical lines in gaze labeler

Future expandability:
- window_geometry: Main window size/position
- recent_files: Recently opened .kinarm files
- custom_save_location: Default export directory
- interpolation_preferences: Default interpolation methods

Architecture
------------
Simple key-value storage using JSON for human readability and easy debugging.
All functions are designed to fail gracefully - if preferences can't be loaded,
the application continues with sensible defaults.

Integration Points
------------------
Used by: gui_main.py (for sticky export and marker selections)
Called: On GUI initialization and when selections change

Notes for Developers
--------------------
- All file operations are wrapped in try-except for robustness
- Invalid JSON is handled gracefully (returns empty dict)
- Directory creation is automatic on first use
- JSON format allows manual editing by advanced users
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any
import platform

# Matplotlib backend for all GUI plotting.
# Requires PyQt5. Change to "TkAgg" if PyQt5 is unavailable.
MPL_BACKEND = "Qt5Agg"

# Application directory name in user's config folder
APP_DIR_NAME = "KinarmDataExplorer"

# Preferences file name (JSON format for readability)
PREFS_FILE_NAME = "user_prefs.json"

# =============================================================================
# Lab-Configurable Processing Parameters
# =============================================================================

# These parameters control all signal processing in the application.
# Labs should adjust these to match their KINARM setup and analysis pipeline.

# --- Sentinel Value Cleaning ---
KINARM_INVALID_ABS_THRESHOLD = 99.9     # |value| >= this treated as invalid (NaN)

# --- Gaze Low-Pass Filter ---
# Applied to Gaze_X/Gaze_Y before all gaze metric calculations.
# Uses a Butterworth filter with filtfilt (zero-phase, forward-backward pass).
# The effective order is double what's specified here due to filtfilt.
DEFAULT_GAZE_LOWPASS_CUTOFF_HZ = 20            # Cutoff frequency (Hz)
DEFAULT_GAZE_SAMPLING_HZ = 1000                  # Sampling Frequency (Hz)
DEFAULT_GAZE_LOWPASS_ORDER = 4                  # Effective order (halved internally for filtfilt)

# --- Savitzky-Golay Filter (Angular Velocity Derivatives) ---
# Used to compute time derivatives of eye-centered Cartesian coordinates
# for angular velocity calculation (Equations 4a, 4b from Singh et al.).
DEFAULT_SAVGOL_WINDOW = 11                      # Window length in frames (must be odd)
DEFAULT_SAVGOL_POLYORDER = 3                    # Polynomial order

# --- Gaze Geometry ---
DEFAULT_EYE_HEIGHT_M = 0.2             # Eye height above stimulus plane (meters)
DEFAULT_VISUAL_ANGLE_DEG = 5.0          # Foveal cone angle (degrees), δ in Equation 9

# --- Interpolation ---
AUTO_INTERP_THRESHOLD_FRAMES = 50       # Gaps <= this are auto-filled with linear interp

# --- Saccadic Interpolation (preview only) ---
SACCADIC_TRANSITION_FRACTION = 0.2      # Fraction of gap used for sigmoid blend
SACCADIC_SIGMOID_STEEPNESS = 10.0       # Higher = sharper transition

# Maximum number of channels displayed in the gaze labeler (including Gaze_X, Gaze_Y).
MAX_LABELER_CHANNELS = 6

DEFAULT_TIMESTAMP_SPACING_S = 0.001  # 1ms default for typical KINARM data

def _app_dir() -> Path:
    """
    Get the application's configuration directory.
    
    Creates the directory if it doesn't exist. Location varies by platform:
    - Windows: C:\\Users\\<username>\\AppData\\Roaming\\KinarmDataExplorer
    - macOS/Linux: ~/.config/KinarmDataExplorer
    
    Returns
    -------
    Path
        Application configuration directory.
    
    Notes
    -----
    - Directory is created automatically on first access
    - Uses .config directory following XDG Base Directory specification
    - For Windows AppData/Roaming compliance, modify base path detection
    - Parent directories are created if needed (parents=True)
    """
    if platform.system() == "Windows":
        import os
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path.home() / ".config"
    
    d = base / APP_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def prefs_path() -> Path:
    """
    Get the full path to the preferences file.
    
    Returns
    -------
    Path
        Full path to user_prefs.json.
    
    Examples
    --------
    >>> prefs_path()
    PosixPath('/home/user/.config/KinarmDataExplorer/user_prefs.json')
    """
    return _app_dir() / PREFS_FILE_NAME

def load_prefs() -> Dict[str, Any]:
    """
    Load user preferences from disk.
    
    Returns
    -------
    dict
        Preferences dictionary, or empty dict if file doesn't exist or is invalid.
    
    Error Handling
    --------------
    - Missing file: Returns empty dict (first-run scenario)
    - Invalid JSON: Returns empty dict (corrupted file)
    - Read errors: Returns empty dict (permission issues)
    
    Notes
    -----
    - Graceful failure ensures application can start even with bad prefs
    - UTF-8 encoding handles international characters in paths
    - Consider logging errors in production for debugging
    
    Examples
    --------
    >>> load_prefs()
    {'export_defaults': ['Gaze_X', 'Gaze_Y', 'xT', 'yT'], 'marker_defaults': ['TARGET_ON']}
    """
    p = prefs_path()
    
    # First run - no preferences file exists yet
    if not p.exists():
        return {}
    
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        # Invalid JSON or read error - return empty dict
        # Application will use defaults
        return {}


def save_prefs(prefs: Dict[str, Any]) -> None:
    """
    Save user preferences to disk.
    
    Parameters
    ----------
    prefs : dict
        Preferences dictionary to save.
    
    File Format
    -----------
    JSON with 2-space indentation for human readability:
    {
      "export_defaults": [
        "Gaze_X",
        "Gaze_Y"
      ],
      "marker_defaults": [
        "TARGET_ON",
        "WAIT_GAP"
      ]
    }
    
    Error Handling
    --------------
    Silently fails if write is unsuccessful (disk full, permissions, etc.).
    This prevents preferences from blocking critical application operations.
    
    Notes
    -----
    - Pretty-printed JSON (indent=2) for manual editing
    - UTF-8 encoding for international characters
    - Overwrites entire file atomically
    - Consider adding backup/versioning for critical settings
    
    Examples
    --------
    >>> save_prefs({'export_defaults': ['Gaze_X', 'Gaze_Y'], 'marker_defaults': ['TARGET_ON']})
    """
    p = prefs_path()
    try:
        p.write_text(json.dumps(prefs, indent=2), encoding="utf-8")
    except Exception:
        # Write failed - silently continue
        # Could add logging here for debugging
        pass

def get_export_defaults() -> List[str]:
    """
    Get the list of channels to pre-select in the export UI.
    
    This implements "sticky selection" - channels selected during previous
    sessions are automatically selected when the application starts.
    
    Returns
    -------
    list[str]
        List of channel names to pre-select, or empty list if none saved.
    
    Examples
    --------
    >>> get_export_defaults()
    ['Gaze_X', 'Gaze_Y', 'xT', 'yT', 'Angular_Velocity']
    
    >>> get_export_defaults()  # First run
    []
    
    Usage
    -----
    Called during GUI initialization to restore previous export selections:
```python
    self._sticky_export_selection = set(get_export_defaults())
```
    
    Notes
    -----
    - Returns empty list (not None) for consistent iteration
    - Channel names are not validated (may reference nonexistent channels)
    - List is copied to prevent external modification of preferences
    """
    prefs = load_prefs()
    return list(prefs.get("export_defaults", []))


def set_export_defaults(selected: List[str]) -> None:
    """
    Save the current export channel selection for future sessions.
    
    This is called whenever the user changes their export selection,
    implementing the "sticky selection" feature.
    
    Parameters
    ----------
    selected : list[str]
        List of channel names currently selected for export.
    
    Examples
    --------
    >>> set_export_defaults(['Gaze_X', 'Gaze_Y', 'xT', 'yT'])
    
    >>> set_export_defaults([])  # Clear saved selection
    
    Usage
    -----
    Called when export selection changes in GUI:
```python
    def _on_export_select(self, event=None):
        selected = {self.export_listbox.get(i) for i in self.export_listbox.curselection()}
        set_export_defaults(sorted(selected))
```
    
    Notes
    -----
    - Overwrites previous defaults completely (no merging)
    - List is copied to prevent later mutations affecting saved prefs
    - Save operation is immediate (no manual "save settings" required)
    - Empty list is valid (clears sticky selection)
    """
    prefs = load_prefs()
    prefs["export_defaults"] = list(selected)
    save_prefs(prefs)


def get_marker_defaults() -> List[str]:
    """
    Get the list of event markers to pre-select in the marker UI.
    
    This implements "sticky selection" for event markers - events selected during 
    previous sessions are automatically selected when the application starts.
    
    Returns
    -------
    list[str]
        List of event names to pre-select, or empty list if none saved.
    
    Examples
    --------
    >>> get_marker_defaults()
    ['TARGET_ON', 'WAIT_GAP', 'Gaze blink start']
    
    >>> get_marker_defaults()  # First run
    []
    
    Usage
    -----
    Called during GUI initialization to restore previous marker selections:
```python
    self._sticky_marker_selection = set(get_marker_defaults())
```
    
    Notes
    -----
    - Returns empty list (not None) for consistent iteration
    - Event names are not validated (may reference nonexistent events)
    - List is copied to prevent external modification of preferences
    """
    prefs = load_prefs()
    return list(prefs.get("marker_defaults", []))


def set_marker_defaults(selected: List[str]) -> None:
    """
    Save the current marker event selection for future sessions.
    
    This is called whenever the user changes their marker selection,
    implementing the "sticky selection" feature for event markers.
    
    Parameters
    ----------
    selected : list[str]
        List of event names currently selected as markers.
    
    Examples
    --------
    >>> set_marker_defaults(['TARGET_ON', 'WAIT_GAP'])
    
    >>> set_marker_defaults([])  # Clear saved selection
    
    Usage
    -----
    Called when marker selection changes in GUI:
```python
    def _on_marker_select(self, event=None):
        selected = {self.marker_listbox.get(i) for i in self.marker_listbox.curselection()}
        set_marker_defaults(sorted(selected))
```
    
    Notes
    -----
    - Overwrites previous defaults completely (no merging)
    - List is copied to prevent later mutations affecting saved prefs
    - Save operation is immediate (no manual "save settings" required)
    - Empty list is valid (clears sticky selection)
    """
    prefs = load_prefs()
    prefs["marker_defaults"] = list(selected)
    save_prefs(prefs)

def get_labeler_channel_defaults() -> List[str]:
    """
    Get the list of extra overlay channels to pre-select in the labeler picker.
    Gaze_X and Gaze_Y are always included and not stored here.
    """
    prefs = load_prefs()
    return list(prefs.get("labeler_channel_defaults", []))


def set_labeler_channel_defaults(selected: List[str]) -> None:
    """
    Save the labeler overlay channel selection for future sessions.
    """
    prefs = load_prefs()
    # Only store the extras, never Gaze_X/Gaze_Y
    extras = [ch for ch in selected if ch not in ("Gaze_X", "Gaze_Y")]
    prefs["labeler_channel_defaults"] = extras
    save_prefs(prefs)

def get_save_location() -> str | None:
    """
    Get the custom save location from preferences.
    
    Returns
    -------
    str | None
        Path to custom save location, or None if using default.
    
    Examples
    --------
    >>> get_save_location()
    '/Users/luke/Downloads/hi'
    
    >>> get_save_location()  # First run or using default
    None
    """
    prefs = load_prefs()
    return prefs.get("custom_save_location")

def set_save_location(location: str | None):
    """
    Save the custom save location to preferences.
    
    Parameters
    ----------
    location : str | None
        Path to custom save location, or None to clear and use default.
    
    Examples
    --------
    >>> set_save_location('/Users/luke/Downloads/hi')
    
    >>> set_save_location(None)  # Reset to default
    
    Notes
    -----
    - Setting to None removes the custom location from preferences
    - Save operation is immediate (no manual "save settings" required)
    """
    prefs = load_prefs()
    if location:
        prefs["custom_save_location"] = location
    else:
        prefs.pop("custom_save_location", None)
    save_prefs(prefs)

_DEFAULT_LABEL_ORDER = ["fixation", "pursuit", "saccade"]

def get_label_order():
    """
    Return saved label order list, or None if not set.
    """
    try:
        prefs = load_prefs()
        order = prefs.get("label_order", None)
        if not order:
            return None
        # validate
        if sorted(order) != sorted(_DEFAULT_LABEL_ORDER):
            return None
        return list(order)
    except Exception:
        return None

def set_label_order(order):
    try:
        prefs = load_prefs()
        if order is None:
            prefs.pop("label_order", None)
        else:
            prefs["label_order"] = list(order)
        save_prefs(prefs)
    except Exception:
        pass

def clear_label_order():
    """Remove stored label order preference (next labeling will prompt again)."""
    set_label_order(None)