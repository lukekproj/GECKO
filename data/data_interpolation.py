"""
Data Interpolation Module

Interactive interpolation for KINARM channels with missing samples.

Key features:
- Automatically fills small NaN gaps (<= auto_threshold) using linear interpolation.
- For large NaN gaps, shows a 3-panel preview (original / linear / saccadic) and
  lets the user pick an interpolation strategy.
- Caches interpolation results per (trial_name, channel_name) for the current session
  to avoid re-prompting when the same channel is requested again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.ndimage import label

from utility.user_prefs import KINARM_INVALID_ABS_THRESHOLD
from utility.user_prefs import (
    SACCADIC_TRANSITION_FRACTION, 
    SACCADIC_SIGMOID_STEEPNESS,
    DEFAULT_TIMESTAMP_SPACING_S
)

@dataclass(frozen=True)
class Gap:
    """Represents a contiguous missing-data region in a 1D signal."""
    gap_id: int
    start: int
    end: int
    length: int
    indices: np.ndarray

def _sanitize_kinarm_signal(values: np.ndarray) -> np.ndarray:
    """
    Convert to float array and replace KINARM sentinel invalid values with NaN.
    
    Note: This should NOT be applied to timestamp channels, acceleration channels,
    or other channels where values > 99.9 are legitimate.
    """
    data = np.asarray(values, dtype=float).copy()
    data[np.abs(data) >= KINARM_INVALID_ABS_THRESHOLD] = np.nan
    return data


def _should_sanitize_channel(channel_name: str) -> bool:
    """
    Determine if a channel should have sentinel values (±99.9) replaced with NaN.

    Only gaze position and pupil channels use KINARM's ±99.9 sentinel to mark
    invalid samples (blinks, tracking loss). All other channel types — hand
    kinematics, force plate, timestamps, acceleration, status bits — either
    use different representations for invalid data or can legitimately exceed
    the 99.9 threshold.

    Parameters
    ----------
    channel_name : str
        Name of the channel to check.

    Returns
    -------
    bool
        True only for Gaze_* channels, excluding Gaze_TimeStamp.
    """
    channel_lower = channel_name.lower()

    # Only gaze channels use the ±99.9 sentinel value convention.
    if not channel_lower.startswith("gaze_"):
        return False

    # Gaze_TimeStamp contains elapsed time in seconds and can legitimately
    # exceed 99.9 for long trials.
    if "timestamp" in channel_lower:
        return False

    return True

def _find_nan_gaps(data: np.ndarray) -> list[Gap]:
    """
    Find contiguous NaN segments in a 1D array.

    Uses scipy.ndimage.label for connected-component labeling of NaN regions.

    Parameters
    ----------
    data : np.ndarray
        1D signal array.

    Returns
    -------
    list[Gap]
        Each Gap contains absolute array indices (not relative to gap start),
        ordered by position in the signal.
    """
    nan_mask = np.isnan(data)
    if not np.any(nan_mask):
        return []

    labeled_gaps, num_gaps = label(nan_mask)
    gaps: list[Gap] = []

    for gap_id in range(1, num_gaps + 1):
        gap_indices = np.where(labeled_gaps == gap_id)[0]
        gaps.append(
            Gap(
                gap_id=gap_id,
                start=int(gap_indices[0]),
                end=int(gap_indices[-1]),
                length=int(gap_indices.size),
                indices=gap_indices,
            )
        )

    return gaps


def _linear_interpolate_gap(data: np.ndarray, gap_indices: np.ndarray) -> np.ndarray:
    """
    Fill a NaN gap by linear interpolation between the closest valid boundary samples.

    If only one boundary exists (gap touches start or end), the gap is filled with the
    nearest available value (flat extrapolation).
    """
    data_copy = data.copy()

    start_idx = int(gap_indices[0])
    end_idx = int(gap_indices[-1])

    before_idx = start_idx - 1
    while before_idx >= 0 and np.isnan(data_copy[before_idx]):
        before_idx -= 1

    after_idx = end_idx + 1
    while after_idx < len(data_copy) and np.isnan(data_copy[after_idx]):
        after_idx += 1

    if before_idx >= 0 and after_idx < len(data_copy):
        x_vals = np.array([before_idx, after_idx], dtype=float)
        y_vals = np.array([data_copy[before_idx], data_copy[after_idx]], dtype=float)
        data_copy[gap_indices] = np.interp(gap_indices, x_vals, y_vals)
    elif before_idx >= 0:
        data_copy[gap_indices] = data_copy[before_idx]
    elif after_idx < len(data_copy):
        data_copy[gap_indices] = data_copy[after_idx]
    # else: gap spans entire array → leave as NaN

    return data_copy


def saccadic_interpolate_gap(data: np.ndarray, gap_indices: np.ndarray) -> np.ndarray:
    """
    Fill a NaN gap using a "saccade-like" pattern.

    Models a rapid transition during the first ~20% of the gap followed by a
    plateau near the final value, mimicking the velocity profile of a saccadic
    eye movement.

    Tunable heuristic parameters (currently hard-coded):
    - Transition fraction: 0.2 (first 20% of gap is the rapid jump)
    - Sigmoid steepness: 10.0 (controls sharpness of the transition curve)

    This is used only for visualization in the interpolation preview UI to help
    the user decide on an interpolation strategy. It is NOT used in downstream
    calculations unless the user explicitly selects saccadic interpolation.

    Parameters
    ----------
    data : np.ndarray
        1D signal with NaN gap(s).
    gap_indices : np.ndarray
        Indices of the gap to fill (from a Gap object).

    Returns
    -------
    np.ndarray
        Copy of data with the specified gap filled.
    """
    data_copy = data.copy()

    start_idx = int(gap_indices[0])
    end_idx = int(gap_indices[-1])

    before_idx = start_idx - 1
    while before_idx >= 0 and np.isnan(data_copy[before_idx]):
        before_idx -= 1

    after_idx = end_idx + 1
    while after_idx < len(data_copy) and np.isnan(data_copy[after_idx]):
        after_idx += 1

    if before_idx >= 0 and after_idx < len(data_copy):
        gap_length = int(gap_indices.size)
        transition_length = max(1, int(gap_length * SACCADIC_TRANSITION_FRACTION))

        start_val = float(data_copy[before_idx])
        end_val = float(data_copy[after_idx])

        for i, idx in enumerate(gap_indices):
            if i < transition_length:
                progress = i / transition_length
                sigmoid = 1.0 / (1.0 + np.exp(-SACCADIC_SIGMOID_STEEPNESS * (progress - 0.5)))
                data_copy[idx] = start_val + (end_val - start_val) * sigmoid
            else:
                data_copy[idx] = end_val

    elif before_idx >= 0:
        data_copy[gap_indices] = data_copy[before_idx]
    elif after_idx < len(data_copy):
        data_copy[gap_indices] = data_copy[after_idx]

    return data_copy


def upsample_timestamps(timestamps: np.ndarray) -> np.ndarray:
    """
    Upsample repeated timestamp values to linearly spaced values.

    KINARM systems sometimes record gaze timestamps at half the gaze sampling
    rate (e.g., 500 Hz timestamps for 1000 Hz gaze data), resulting in
    consecutive repeated values. This function distributes repeated timestamps
    into evenly spaced intervals to restore a monotonically increasing sequence
    that matches the actual sample rate.

    Example::

        [1, 1, 2, 2, 3, 3] -> [1, 1.5, 2, 2.5, 3, 3.5]

    Parameters
    ----------
    timestamps : np.ndarray
        1D array of timestamp values, possibly with consecutive repeats.

    Returns
    -------
    np.ndarray
        Upsampled timestamps with the same length as the input.
    """
    ts = np.asarray(timestamps, dtype=float)
    upsampled: list[float] = []

    i = 0
    n = len(ts)
    last_spacing = None  # Track the most recent spacing we used
    
    while i < n:
        current_val = ts[i]
        count = 1
        while i + count < n and ts[i + count] == current_val:
            count += 1

        if count == 1:
            upsampled.append(float(current_val))
            i += 1
            continue

        # Determine spacing based on the next distinct timestamp
        if (i + count) < n:
            next_val = ts[i + count]
            spacing = (next_val - current_val) / count
            last_spacing = spacing  # Remember this spacing
        elif last_spacing is not None:
            # Use the last known spacing for the final group
            spacing = last_spacing
        else:
            # Fallback: estimate from overall data
            if i > 0:
                # Use average spacing from the data we've seen
                spacing = (current_val - ts[0]) / i / 2.0
            else:
                # Only one group in entire array
                spacing = DEFAULT_TIMESTAMP_SPACING_S

        for j in range(count):
            upsampled.append(float(current_val + j * spacing))

        i += count

    return np.asarray(upsampled, dtype=float)

def _overlay_target_xt_yt(ax, explorer) -> None:
    """
    Overlay target xT/yT on a secondary y-axis (if present).

    This is purely for helping users choose the correct interpolation method.
    If xT/yT are missing, this function does nothing.
    """
    try:
        xt = _sanitize_kinarm_signal(explorer.current_trial.kinematics["xT"].values)
        yt = _sanitize_kinarm_signal(explorer.current_trial.kinematics["yT"].values)

        ax2 = ax.twinx()
        ax2.plot(xt, linestyle="--", alpha=0.25, label="xT")
        ax2.plot(yt, linestyle="--", alpha=0.25, label="yT")
        ax2.set_ylabel("Target (xT/yT)")
        ax2.grid(False)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right")
    except Exception:
        return


def _choose_large_gap_strategy(
    explorer,
    name: str,
    trial_info: Optional[str],
    original: np.ndarray,
    base: np.ndarray,
    large_gaps: list[Gap],
    gap_index: int = 1,
    gap_total: int = 1,
    fig=None,
) -> Optional[str]:
    """
    Show preview plots for large gaps and return the user's chosen strategy.

    Returns one of: "linear", "saccadic", "nan", or None if user cancels.

    NOTE: This uses matplotlib's interactive window and blocks execution
    until the user selects a button or closes the figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    reusing_fig = fig is not None

    if reusing_fig:
        fig.clear()
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3)
        fig.subplots_adjust(bottom=0.15, hspace=0.3)
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        fig.canvas.manager.window.showMaximized()
        plt.subplots_adjust(bottom=0.15, hspace=0.3)

    gap = large_gaps[0]
    title_text = f"{name} - Gap {gap_index} of {gap_total}: frames {gap.start}-{gap.end} ({gap.length} frames)"
    if trial_info:
        title_text = f"{trial_info}  •  {title_text}"
    ax1.set_title(title_text, fontsize=12, fontweight="bold")

    # Original
    ax1.plot(original, "b-", linewidth=1.5, label="Original (with gaps)")
    ax1.set_ylabel("Value")
    ax1.grid(True, alpha=0.3)
    for g in large_gaps:
        ax1.axvspan(g.start, g.end, color="red", alpha=0.3)
    ax1.legend()
    _overlay_target_xt_yt(ax1, explorer)

    # Linear preview
    linear_preview = base.copy()
    for g in large_gaps:
        linear_preview = _linear_interpolate_gap(linear_preview, g.indices)
    ax2.plot(linear_preview, "g-", linewidth=1.5, label="Linear Interpolation")
    ax2.set_title("Preview: Linear Interpolation", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Value")
    ax2.grid(True, alpha=0.3)
    for g in large_gaps:
        ax2.axvspan(g.start, g.end, color="green", alpha=0.2)
    ax2.legend()
    _overlay_target_xt_yt(ax2, explorer)

    # Saccadic preview
    sacc_preview = base.copy()
    for g in large_gaps:
        sacc_preview = saccadic_interpolate_gap(sacc_preview, g.indices)
    ax3.plot(sacc_preview, "orange", linewidth=1.5, label="Saccadic Interpolation")
    ax3.set_title("Preview: Saccadic Interpolation (Fast Jump)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Frame")
    ax3.set_ylabel("Value")
    ax3.grid(True, alpha=0.3)
    for g in large_gaps:
        ax3.axvspan(g.start, g.end, color="orange", alpha=0.2)
    ax3.legend()
    _overlay_target_xt_yt(ax3, explorer)

    gap_text = f"Gap {gap_index} of {gap_total}\nFrames {gap.start}-{gap.end} ({gap.length} frames)"
    fig.text(
        0.02,
        0.98,
        gap_text,
        transform=fig.transFigure,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
    )

    # Decision buttons
    user_decision = {"action": None}

    def on_linear(event):
        user_decision["action"] = "linear"
        if reusing_fig:
            fig.canvas.stop_event_loop()
        else:
            plt.close(fig)

    def on_saccadic(event):
        user_decision["action"] = "saccadic"
        if reusing_fig:
            fig.canvas.stop_event_loop()
        else:
            plt.close(fig)

    def on_nan(event):
        user_decision["action"] = "nan"
        if reusing_fig:
            fig.canvas.stop_event_loop()
        else:
            plt.close(fig)

    def on_cancel(event):
        user_decision["action"] = "cancel"
        if reusing_fig:
            fig.canvas.stop_event_loop()
        else:
            plt.close(fig)

    # Create four buttons
    ax_linear   = fig.add_axes([0.05, 0.05, 0.2, 0.06])
    ax_saccadic = fig.add_axes([0.28, 0.05, 0.2, 0.06])
    ax_nan      = fig.add_axes([0.51, 0.05, 0.2, 0.06])
    ax_cancel   = fig.add_axes([0.74, 0.05, 0.2, 0.06])

    btn_linear   = Button(ax_linear,   "Linear\nInterpolation")
    btn_saccadic = Button(ax_saccadic, "Saccadic\nInterpolation")
    btn_nan      = Button(ax_nan,      "Leave as NaN")
    btn_cancel   = Button(ax_cancel,   "Cancel")

    btn_linear.on_clicked(on_linear)
    btn_saccadic.on_clicked(on_saccadic)
    btn_nan.on_clicked(on_nan)
    btn_cancel.on_clicked(on_cancel)

    # Keep refs alive so callbacks keep working
    fig._interp_ui_refs = {
        "btn_linear": btn_linear,
        "btn_saccadic": btn_saccadic,
        "btn_nan": btn_nan,
        "btn_cancel": btn_cancel,
        "on_linear": on_linear,
        "on_saccadic": on_saccadic,
        "on_nan": on_nan,
        "on_cancel": on_cancel,
    }

    if reusing_fig:
        fig.canvas.draw()
        fig.canvas.start_event_loop(timeout=0)
    else:
        plt.show(block=True)

    # After window closes, user_decision["action"] is set (or still None)
    if user_decision["action"] is None:
        # User clicked the X on the window
        user_decision["action"] = "cancel"

    # Translate decision into return value
    if user_decision["action"] in ("cancel", None):
        return None
    return user_decision["action"]

def smart_interpolate_trial_data(explorer, channel_names, auto_threshold: int = 50, force_prompt: bool = False, trial_info: Optional[str] = None):
    """
    Interpolate missing samples for multiple channels in the current trial.

    Parameters
    ----------
    explorer : KinarmDataExplorer
        The active explorer (must have current_trial set).
    channel_names : list[str]
        Channel names to interpolate from trial.kinematics.
    auto_threshold : int
        Gaps of length <= auto_threshold are automatically linearly interpolated.
    force_prompt : bool
        If True, ignores cached results and forces the preview dialog again.
    trial_info : str | None
        Optional text shown at the top of the preview window (e.g., trial name / TP #).

    Returns
    -------
    dict[str, np.ndarray] | None
        Mapping of channel name -> interpolated 1D signal.
        Returns None if the user cancels on any channel with large gaps.
    """
    if not explorer.current_trial:
        return None

    trial_name = explorer.current_trial.name
    
    interpolated_channels: Dict[str, np.ndarray] = {}
    channels_needing_processing: list[str] = []

    for ch in channel_names:
        key = (trial_name, ch)
        if not force_prompt and key in explorer.interpolation_cache:
            interpolated_channels[ch] = explorer.interpolation_cache[key]
        else:
            channels_needing_processing.append(ch)

    if not channels_needing_processing:
        return interpolated_channels

    for channel_name in channels_needing_processing:
        if channel_name not in explorer.current_trial.kinematics:
            continue

        raw = explorer.current_trial.kinematics[channel_name].values
        
        # Only sanitize if appropriate for this channel type
        if _should_sanitize_channel(channel_name):
            data = _sanitize_kinarm_signal(raw)
        else:
            data = np.asarray(raw, dtype=float).copy()

        # Special handling: timestamps are upsampled (not interpolated like signals)
        if channel_name == "Gaze_TimeStamp":
            upsampled = upsample_timestamps(data)
            interpolated_channels[channel_name] = upsampled
            explorer.interpolation_cache[(trial_name, channel_name)] = upsampled
            continue

        gaps = _find_nan_gaps(data)
        if not gaps:
            interpolated_channels[channel_name] = data
            explorer.interpolation_cache[(trial_name, channel_name)] = data
            continue

        small = [g for g in gaps if g.length <= auto_threshold]
        large = [g for g in gaps if g.length > auto_threshold]

        interpolated = data.copy()

        if small:
            for g in small:
                interpolated = _linear_interpolate_gap(interpolated, g.indices)

        if large:
            import matplotlib.pyplot as plt
            gap_fig = plt.figure(figsize=(12, 10))
            try:
                gap_fig.canvas.manager.window.showMaximized()
            except Exception:
                pass
            plt.show(block=False)

            for gap_idx, g in enumerate(large, start=1):
                action = _choose_large_gap_strategy(
                    explorer, channel_name, trial_info, data, interpolated, [g],
                    gap_index=gap_idx, gap_total=len(large), fig=gap_fig
                )
                if action is None:
                    try:
                        plt.close(gap_fig)
                    except Exception:
                        pass
                    return None
                if action == "linear":
                    interpolated = _linear_interpolate_gap(interpolated, g.indices)
                elif action == "saccadic":
                    interpolated = saccadic_interpolate_gap(interpolated, g.indices)
                # else "nan": leave gap as-is

            try:
                plt.close(gap_fig)
            except Exception:
                pass

        interpolated_channels[channel_name] = interpolated
        explorer.interpolation_cache[(trial_name, channel_name)] = interpolated
    return interpolated_channels