"""
KINARM Utility Functions

Helper functions for processing KINARM experimental data files and parameters.

Overview
--------
KINARM/Dexterit-E systems store experimental parameters in a structured but sometimes
inconsistent format. These utilities handle common data processing tasks:

- Extracting parameter groups (TARGET_TABLE, TP_TABLE, etc.)
- Determining valid data lengths in pre-allocated tables
- Normalizing tables for display and export
- Finding Task Protocol (TP) numbers across different software versions

Key Concepts
------------
Parameter Groups:
    KINARM stores related parameters with prefixes like "TARGET_TABLE:X",
    "TARGET_TABLE:Y", etc. These utilities extract and organize such groups.

Pre-allocated Tables:
    Many KINARM tables allocate more rows than needed. Functions here detect
    which rows contain actual experimental data vs. empty pre-allocation.

Case Insensitivity:
    Parameter names can vary in capitalization across KINARM versions.
    Utilities handle this with case-insensitive lookups.

Integration Points
------------------
Used by: gui_metadata_viewer.py, gaze_labeler_export.py, gui_main.py
Purpose: Standardize parameter access across different KINARM file versions

Notes for Developers
--------------------
- Always use case-insensitive helpers (get_ci_key) for parameter lookups
- Normalize tables before display to ensure consistent row counts
- TP number extraction has multiple fallback strategies for robustness
"""

from collections import OrderedDict
from array import array
import pandas as pd


# -----------------------------------------------------------------------------
# Type Conversion
# -----------------------------------------------------------------------------

def to_list(v):
    """
    Convert various data types to standard Python lists.
    
    KINARM data can come in many formats (Python arrays, tuples, lists,
    scalars). This normalizes everything to standard Python lists for
    consistent processing.
    
    Parameters
    ----------
    v : any
        Value to convert (can be array, list, tuple, or scalar).
    
    Returns
    -------
    list
        Standard Python list representation.
    
    Examples
    --------
    >>> to_list(array('d', [1.0, 2.0]))
    [1.0, 2.0]
    
    >>> to_list(5)
    [5]
    
    >>> to_list([1, 2, 3])
    [1, 2, 3]
    """
    if isinstance(v, array):
        return list(v)
    elif isinstance(v, (list, tuple)):
        return list(v)
    else:
        return [v]


# -----------------------------------------------------------------------------
# Parameter Extraction
# -----------------------------------------------------------------------------

def extract_group(trial, group_name):
    """
    Extract parameter groups from trial data with prefix removal.
    
    KINARM trials store experimental parameters with prefixes like
    "TARGET_TABLE:X" or "TP_TABLE:Start Target". This function extracts
    all parameters for a specific group and removes the prefix, leaving
    clean column names for table display.
    
    Parameters
    ----------
    trial : Trial object
        Trial containing parameter data.
    group_name : str
        Parameter group prefix (e.g., "TARGET_TABLE", "TP_TABLE").
    
    Returns
    -------
    OrderedDict
        Parameters with prefixes removed and values converted to lists.
    
    Examples
    --------
    >>> extract_group(trial, "TARGET_TABLE")
    OrderedDict([('X', [0.1, 0.2]), ('Y', [0.3, 0.4])])
    
    Notes
    -----
    - Returns OrderedDict to preserve parameter order from file
    - All values are converted to lists via to_list() for consistency
    """
    out = OrderedDict()
    prefix = f"{group_name}:"
    
    for k, v in trial.parameters.items():
        if k.startswith(prefix):
            label = k.split(":", 1)[1]  # Remove prefix to get clean column name
            out[label] = to_list(v)
    
    return out


def get_ci_key(d, key):
    """
    Find a key in dictionary using case-insensitive matching.
    
    KINARM parameter names sometimes vary in capitalization between
    different software versions and file formats. This function handles
    those inconsistencies by performing case-insensitive lookups.
    
    Parameters
    ----------
    d : dict
        Dictionary to search.
    key : str
        Key to find (case-insensitive).
    
    Returns
    -------
    str or None
        Actual key name from dictionary, or None if not found.
    
    Examples
    --------
    >>> d = {"Start Target": 1, "End Target": 2}
    >>> get_ci_key(d, "start target")
    'Start Target'
    
    >>> get_ci_key(d, "missing_key")
    None
    
    Notes
    -----
    - Exact match is checked first (fastest path)
    - Falls back to case-insensitive search if no exact match
    - Returns the actual key from the dictionary, not the input key
    """
    # Fast path: exact match
    if key in d:
        return key
    
    # Slow path: case-insensitive lookup
    lk = key.lower()
    for k in d.keys():
        if k.lower() == lk:
            return k
    
    return None


# -----------------------------------------------------------------------------
# Table Row Detection
# -----------------------------------------------------------------------------

def infer_used_rows(cols):
    """
    Determine how many rows of a parameter table are actually used.
    
    Some KINARM tables have a "USED" parameter that explicitly states
    how many rows contain valid data. Otherwise, we determine the count
    from the maximum length across all columns.
    
    This is necessary because KINARM often pre-allocates large tables
    but only fills the first several rows with actual experimental data.
    
    Parameters
    ----------
    cols : dict
        Parameter dictionary (typically from extract_group).
    
    Returns
    -------
    int
        Number of valid rows in the table.
    
    Algorithm
    ---------
    1. Check for explicit "USED" parameter
    2. If not found, return maximum length across all list columns
    3. If no lists found, return 0
    
    Examples
    --------
    >>> cols = {"USED": [5], "X": [1,2,3,4,5,0,0], "Y": [1,2,3,4,5,0,0]}
    >>> infer_used_rows(cols)
    5
    
    >>> cols = {"X": [1,2,3], "Y": [1,2]}
    >>> infer_used_rows(cols)
    3
    
    Notes
    -----
    - The "USED" parameter is checked first as the authoritative source
    - Fallback method prevents display of empty pre-allocated rows
    """
    # Check for explicit "USED" parameter
    if "USED" in cols and len(cols["USED"]) == 1:
        try:
            n = int(float(cols["USED"][0]))
            if n > 0:
                return n
        except Exception:
            pass
    
    # Fallback: find maximum length across all list columns
    max_len = 0
    for v in cols.values():
        if isinstance(v, list):
            max_len = max(max_len, len(v))
    
    return max_len or 0


def infer_tp_rows_from_start(cols, fallback_keys):
    """
    Determine Task Protocol table length by examining Start Target column.
    
    The TP table often contains many pre-allocated rows, but only the first
    several are actually used in the experiment. We look for non-zero
    "Start Target" values to find where the valid data ends, since unused
    rows typically have zero values.
    
    Parameters
    ----------
    cols : dict
        Parameter dictionary for TP_TABLE.
    fallback_keys : list[str]
        Column names to use for length if Start Target method fails.
    
    Returns
    -------
    int
        Number of valid TP table rows.
    
    Algorithm
    ---------
    1. Find "Start Target" column (case-insensitive)
    2. Count consecutive non-zero values from beginning
    3. If that fails, try "End Target" column
    4. Last resort: use maximum length from fallback_keys
    
    Examples
    --------
    >>> cols = {"Start Target": [1, 2, 3, 0, 0, 0]}
    >>> infer_tp_rows_from_start(cols, [])
    3
    
    Notes
    -----
    - Stops counting at first zero (assumes valid TPs are contiguous)
    - This heuristic works for standard KINARM experiments
    - Fallback ensures we always return a reasonable value
    """
    # Try to find Start Target column (case insensitive)
    k = get_ci_key(cols, "Start Target")
    if k and isinstance(cols[k], list) and cols[k]:
        n = 0
        # Count consecutive non-zero start target values
        for val in cols[k]:
            try:
                if float(val) != 0.0:
                    n += 1
                else:
                    break  # Stop at first zero
            except Exception:
                break
        if n > 0:
            return n
    
    # Fallback: try End Target column
    for key in ("Start Target", "End Target"):
        kk = get_ci_key(cols, key)
        if kk and isinstance(cols[kk], list) and len(cols[kk]) > 0:
            return len(cols[kk])
    
    # Last resort: maximum length from fallback keys
    lengths = []
    for key in fallback_keys:
        kk = get_ci_key(cols, key)
        if kk and isinstance(cols[kk], list):
            lengths.append(len(cols[kk]))
    
    return max(lengths) if lengths else 0


# -----------------------------------------------------------------------------
# Table Normalization
# -----------------------------------------------------------------------------

def normalize_table(cols, desired_cols, rows, left_label_prefix=None):
    """
    Convert raw parameter data into a properly formatted pandas DataFrame.
    
    This ensures all columns have the same number of rows, handles missing
    data appropriately, and optionally adds row labels for display.
    
    Parameters
    ----------
    cols : dict
        Raw parameter dictionary (from extract_group).
    desired_cols : list of tuples
        List of (display_name, source_key) pairs defining columns to include.
    rows : int
        Number of rows the final table should have.
    left_label_prefix : str, optional
        If provided, adds a label column (e.g., "Target 1", "Target 2").
    
    Returns
    -------
    pandas.DataFrame
        Normalized table ready for display or export.
    
    Normalization Rules
    -------------------
    - Single values are broadcast to all rows if rows > 1
    - Short columns are padded with empty strings
    - Long columns are truncated to match row count
    - Missing columns are filled with empty strings
    
    Examples
    --------
    >>> cols = {"X": [0.1, 0.2], "Y": [0.3, 0.4]}
    >>> desired = [("X Position", "X"), ("Y Position", "Y")]
    >>> normalize_table(cols, desired, 2, "Target")
           Target  X Position  Y Position
    0   Target 1         0.1         0.3
    1   Target 2         0.2         0.4
    
    Notes
    -----
    - Uses case-insensitive key lookup via get_ci_key
    - Returns empty DataFrame if cols is empty or rows is 0
    - Label column (if requested) is always the leftmost column
    """
    table = {}
    
    # Add row labels if requested
    if left_label_prefix:
        table[left_label_prefix] = [f"{left_label_prefix} {i+1}" for i in range(rows)]
    
    # Process each requested column
    for display_name, source_key in desired_cols:
        kk = get_ci_key(cols, source_key)
        vals = cols.get(kk, [])
        
        # Ensure it's a list
        vals = list(vals) if isinstance(vals, list) else [vals]
        
        # Handle single values that should be repeated (broadcast)
        if len(vals) == 1 and rows > 1:
            vals = [vals[0]] * rows
        
        # Pad with empty strings if too short
        if len(vals) < rows:
            vals = vals + [""] * (rows - len(vals))
        
        # Truncate if too long
        table[display_name] = vals[:rows]
    
    return pd.DataFrame(table)


# -----------------------------------------------------------------------------
# Trial Metadata Extraction
# -----------------------------------------------------------------------------

def find_trial_tp_number(trial):
    """
    Extract the Task Protocol (TP) number from a trial.
    
    Different KINARM software versions and experimental protocols store
    the TP number under different parameter names. This function checks
    multiple common locations and returns the TP number if found.
    
    Parameters
    ----------
    trial : Trial object
        Trial to examine for TP number.
    
    Returns
    -------
    int or None
        TP number if found, None otherwise.
    
    Search Strategy
    ---------------
    1. Check exact parameter name matches (common locations)
    2. Check for any parameter containing 'TP' with scalar-like value
    3. Return None if no valid TP number found
    
    Examples
    --------
    >>> trial.parameters = {"TRIAL:TP": [5]}
    >>> find_trial_tp_number(trial)
    5
    
    >>> trial.parameters = {"TP_NUM": 3}
    >>> find_trial_tp_number(trial)
    3
    
    >>> trial.parameters = {"OTHER_PARAM": 1}
    >>> find_trial_tp_number(trial)
    None
    
    Notes
    -----
    - Handles both scalar and array-like values
    - Converts to int via float (handles string numbers)
    - Returns None rather than raising exception if not found
    - Robust across different KINARM software versions (3.x, 4.x, 5.x)
    """
    # Common parameter names that might contain TP number
    # Ordered by likelihood for performance
    candidates = [
        "TRIAL:TP", "TRIAL:TP_NUM", "TRIAL:TP_NUMBER",
        "TP_TABLE:TP", "TP_TABLE:TP_NUM", 
        "TP", "TP NUM", "TP NUMBER", "TP_IDX", "TP_INDEX"
    ]
    
    # Check exact parameter names first (fast path)
    for key in candidates:
        if key in trial.parameters:
            try:
                val = trial.parameters[key]
                # Handle both scalar and array values
                if isinstance(val, (list, tuple, array)):
                    if len(val) >= 1:
                        return int(float(val[0]))
                else:
                    return int(float(val))
            except Exception:
                continue
    
    # Fallback: look for any parameter containing 'TP' (slow path)
    # This catches uncommon naming conventions
    for key, val in trial.parameters.items():
        if "TP" in key.upper():
            try:
                if isinstance(val, (list, tuple, array)):
                    if len(val) == 1:
                        return int(float(val[0]))
                else:
                    return int(float(val))
            except Exception:
                continue
    
    return None