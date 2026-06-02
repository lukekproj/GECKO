"""
Interactive Gaze Event Labeler

Visual interface for manually labeling eye movement events in KINARM gaze tracking data.

Overview
--------
This module provides an interactive matplotlib-based interface where researchers can
manually annotate eye movement data by clicking directly on plots. Users mark time
periods as fixations (steady gaze), pursuits (smooth tracking), or saccades (rapid jumps).

Workflow
--------
1. Initial labeling: User labels fixations, then pursuits, then saccades in sequence
2. Summary review: Visual overview showing all labels overlaid on data
3. Editing (optional): Refine any label type without restarting
4. Export: Accept results or save and continue to next trial

Features
--------
- Auto-trimming: Prevents overlapping labels (new labels skip already-labeled frames)
- Visual feedback: Real-time highlighting as selections are made
- Batch marking: "Mark All" button for trials that are entirely one type
- Multi-trial workflow: "Save & Next Trial" for efficient batch processing
- Bad trial flagging: Mark unusable trials (exports all frames as code 9)
- Performance optimization: Intelligent downsampling for large datasets

Integration
-----------
Called by: gui_main.py (via gaze_labeler_export.py)
Returns to: gaze_labeler_export.py for file writing

Notes for Developers
--------------------
- Uses matplotlib event system for click detection and rendering
- Blitting optimization for smooth real-time feedback
- Range merging prevents fragmentation and ensures clean output
- All button references must be stored to prevent garbage collection
"""

import matplotlib
from utility.user_prefs import MPL_BACKEND
matplotlib.use(MPL_BACKEND)
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
import numpy as np
from tkinter import messagebox
import tkinter as tk


# -----------------------------------------------------------------------------
# Main Labeler Class
# -----------------------------------------------------------------------------

class GazeLabeler:
    """
    Interactive tool for manually labeling eye movement events in gaze tracking data.
    
    This class creates a visual interface where researchers can click to mark
    start and end points of different types of eye movements (fixations, pursuits,
    saccades) in recorded eye tracking data.
    
    The labeling process is designed to be efficient and error-resistant:
    - Auto-skip already labeled frames to prevent overlaps
    - Visual feedback shows selections in real-time
    - Undo/clear functions for quick corrections
    - Summary view for final review before accepting
    
    Attributes
    ----------
    trial_name : str
        Identifier for the current trial.
    trial_info : str
        Formatted display string with full trial details.
    gaze_x, gaze_y : np.ndarray
        Optimized gaze position data (may be downsampled for performance).
    xT, yT : np.ndarray
        Target position data (matched to gaze arrays).
    label_ranges : dict
        Dictionary mapping label types to lists of (start, end) frame ranges.
    cancel_all : bool
        Flag indicating user cancelled entire labeling process.
    bad_trial : bool
        Flag indicating trial is marked as unusable (exports as code 9).
    """
    
    def __init__(self, trial_name, gaze_x, gaze_y, overlay_channels=None, trial_info=None, marker_frames=None, label_order=None):
        """
        Initialize the labeling tool with trial data.
        
        Parameters
        ----------
        trial_name : str
            Name/ID of the experimental trial being analyzed.
        gaze_x, gaze_y : array-like
            Arrays showing where the person was looking (x,y coordinates in meters).
        xT, yT : array-like
            Arrays showing where targets appeared (x,y coordinates in meters).
        trial_info : str, optional
            Formatted string with full trial details for display banner.
        marker_frames : dict, optional
            Dictionary mapping event labels to lists of frame indices.
            Example: {"TARGET_ON": [150, 300], "BLINK_START": [450]}
        
        Notes
        -----
        - Input arrays are automatically optimized (downsampled) if too large for smooth rendering
        - Target event frames are preserved during optimization for accurate display
        """
        self.trial_name = trial_name
        
        # Format trial info with trial number prefix (e.g., "2. 02_07_01  •  TP #14")
        if trial_info:
            self.trial_info = trial_info
        else:
            # Fallback: extract trial number from trial_name
            trial_num = trial_name.replace("Trial", "") if "Trial" in trial_name else trial_name
            self.trial_info = f"{trial_num}. {trial_name}"

        # Store arbitrary marker event frames (e.g., BLINK_START, WAIT_GAP, etc.)
        self.marker_frames = marker_frames if marker_frames is not None else {}
        
        # Optimize data display for better performance
        self.overlay_channels = overlay_channels if overlay_channels is not None else {}
        self.gaze_x, self.gaze_y, self.overlay_channels = self.optimize_plot_data(
            gaze_x, gaze_y, self.overlay_channels
        )

        # Initialize storage for labeled eye movement events
        self.label_ranges = {'fixation': [], 'pursuit': [], 'saccade': [], 'other': []} # 'other' can be used for general eye movements that dont fit under other categories
        
        # Control flags
        self.cancel_all = False      # User cancelled entire process
        self.bad_trial = False        # Trial marked as unusable

        self._prompt_for_order = (label_order is None)
        self.label_order = list(label_order) if label_order else None

    # -------------------------------------------------------------------------
    # Data Optimization
    # -------------------------------------------------------------------------

    def optimize_plot_data(self, gaze_x, gaze_y, overlay_channels, max_points=None):
        """
        Reduce data density for faster, more responsive plotting.
        
        Eye tracking data often contains thousands of points per second (1000+ Hz sampling).
        Displaying all points can make the interface slow and unresponsive. This function
        intelligently reduces the data while preserving important features and maintaining
        the overall pattern.
        
        Parameters
        ----------
        gaze_x, gaze_y, xT, yT : array-like
            Full arrays of gaze and target data.
        max_points : int, optional
            Maximum number of points to display (default 5000).
        
        Returns
        -------
        tuple of np.ndarray
            Reduced arrays (gaze_x, gaze_y, xT, yT) that maintain data fidelity
            while improving rendering performance.
        
        Algorithm
        ---------
        1. Skip optimization if data is already small enough
        2. Uniform sampling: Take every Nth point based on data length
        3. Preserve critical points: Always include target appearance frames
        4. Sort and deduplicate the final index set
        
        Notes
        -----
        - Important time points (target appearances) are always preserved
        - This only affects display; full data is used for export
        - Typical trials: 1000-5000 frames → display ~5000 points
        """
        if max_points is None:
            max_points = max(5000, len(gaze_x))

        # Skip optimization if data is already small enough
        if len(gaze_x) <= max_points:
            return gaze_x, gaze_y, overlay_channels
        
        # Calculate step size for uniform sampling
        step = max(1, len(gaze_x) // max_points)
        
        # Create base index set with uniform sampling
        idx = list(range(0, len(gaze_x), step))
        
        # Always preserve important time points (target + marker events)
        for frame in getattr(self, "target_on_frames", []):
            if 0 <= frame < len(gaze_x):
                idx.append(frame)

        for frame in getattr(self, "hold_at_target_frames", []):
            if 0 <= frame < len(gaze_x):
                idx.append(frame)

        for frames in getattr(self, "marker_frames", {}).values():
            for frame in frames:
                if 0 <= frame < len(gaze_x):
                    idx.append(frame)
        
        # Remove duplicates and sort
        idx = sorted(set(idx))
        
        optimized_overlays = {k: np.array(v)[idx] for k, v in overlay_channels.items()}
        return (np.array(gaze_x)[idx], np.array(gaze_y)[idx], optimized_overlays)

    # -------------------------------------------------------------------------
    # Range Manipulation
    # -------------------------------------------------------------------------

    @staticmethod
    def _merge_ranges(ranges, pad=0):
        """
        Combine overlapping or touching time periods into single segments.
        
        When users select multiple time periods that overlap or are very close,
        this function merges them into continuous periods. This prevents visual
        artifacts and ensures clean data output.
        
        Parameters
        ----------
        ranges : list of tuples
            List of (start_frame, end_frame) pairs.
        pad : int, optional
            Minimum gap between ranges to keep them separate (default 0).
            If pad=1, ranges separated by 1 frame or less will merge.
        
        Returns
        -------
        list of tuples
            Merged (start_frame, end_frame) pairs with no overlaps.
        
        Algorithm
        ---------
        1. Sort ranges by start time
        2. Iterate through sorted ranges
        3. If current range overlaps with last merged range, extend the merged range
        4. Otherwise, start a new merged range
        
        Examples
        --------
        >>> _merge_ranges([(10, 20), (15, 25), (30, 40)], pad=0)
        [(10, 25), (30, 40)]
        
        >>> _merge_ranges([(10, 20), (22, 30)], pad=1)
        [(10, 30)]  # Merged because gap ≤ 1
        """
        if not ranges:
            return []
        
        # Sort ranges by start time
        ranges = sorted(ranges, key=lambda r: (r[0], r[1]))
        merged = [list(ranges[0])]  # Start with first range
        
        # Check each subsequent range for overlap with the last merged range
        for start, end in ranges[1:]:
            last_start, last_end = merged[-1]
            # Merge if ranges overlap or are within padding distance
            if start <= last_end + pad:
                merged[-1][1] = max(last_end, end)  # Extend to later end time
            else:
                merged.append([start, end])  # Keep as separate range
        
        return [tuple(r) for r in merged]

    def _trim_to_unlabeled(self, start, end, current_label_type):
        """
        Trim a selection range to only include unlabeled frames.
        
        This prevents overlaps by automatically excluding frames that are
        already labeled with OTHER label types. Returns a list of sub-ranges
        that are safe to label.
        
        This is the core auto-skip functionality: when a user selects a time
        period, the system automatically skips over frames that are already
        labeled with a different type.
        
        Parameters
        ----------
        start, end : int
            The user's selected frame range (inclusive).
        current_label_type : str
            The type being labeled ('fixation', 'pursuit', or 'saccade').
        
        Returns
        -------
        list of tuples
            List of (start, end) frame ranges that are unlabeled and safe to mark.
        
        Algorithm
        ---------
        1. Collect all frames already labeled with OTHER types (not current type)
        2. Scan through user's selected range [start, end]
        3. Build continuous sub-ranges of unlabeled frames
        4. Return list of these safe-to-label sub-ranges
        
        Examples
        --------
        User selects frames 100-200 for pursuit labeling
        Frames 120-150 already labeled as fixation
        Returns: [(100, 119), (151, 200)]  # Automatically skips 120-150
        
        Notes
        -----
        - Allows overlapping labels of the SAME type (editing mode)
        - Only prevents overlaps with DIFFERENT types
        - Returns empty list if entire range is already labeled
        """
        # Get all already-labeled frames from OTHER label types
        labeled_frames = set()
        for label_name, ranges in self.label_ranges.items():
            if label_name != current_label_type:  # Don't check against own type
                for range_start, range_end in ranges:
                    labeled_frames.update(range(range_start, range_end + 1))
        
        # Find continuous unlabeled sub-ranges within [start, end]
        unlabeled_ranges = []
        current_start = None
        
        for frame in range(start, end + 1):
            if frame not in labeled_frames:
                # Frame is unlabeled
                if current_start is None:
                    current_start = frame  # Start new range
            else:
                # Frame is already labeled - close current range if any
                if current_start is not None:
                    unlabeled_ranges.append((current_start, frame - 1))
                    current_start = None
        
        # Close final range if still open
        if current_start is not None:
            unlabeled_ranges.append((current_start, end))
        
        return unlabeled_ranges

    # -------------------------------------------------------------------------
    # Main Workflow
    # -------------------------------------------------------------------------

    def plot_and_select_range(self):
        """
        Launch the interactive labeling interface.
        
        This is the main entry point that coordinates the entire labeling workflow:
        1. Initial sequential labeling (fixation → pursuit → saccade)
        2. Summary review with editing options
        3. Optional editing of specific label types
        4. Return results or continue to next trial
        
        Returns
        -------
        tuple or (None, None)
            Success: (label_ranges_dict, action_string)
                - label_ranges_dict: {'fixation': [(s,e), ...], 'pursuit': [...], 'saccade': [...]}
                - action_string: 'accept' (finish) or 'next_trial' (continue to next)
            Cancelled: (None, None)
        
        Workflow States
        ---------------
        - initial_labeling_done: Tracks whether initial sequence is complete
        - After initial labeling: Loop between summary and editing until accept/cancel
        - cancel_all flag: Propagates cancellation through the entire process
        
        Notes
        -----
        - User can restart entire labeling from summary screen
        - Editing mode preserves other label types (shows as background)
        - "Next Trial" action automatically advances to next trial in GUI
        """
        label_colors = {'fixation': 'green', 'pursuit': 'cornflowerblue', 'saccade': 'red', 'other': 'orange'}
        total_frames = len(self.gaze_x)
        initial_labeling_done = False

        # Main workflow loop
        while True:
            # Phase 1: Initial labeling sequence
            if not initial_labeling_done:
                if self._prompt_for_order:
                    picked = self._choose_label_order()
                    if picked is None:
                        self.cancel_all = True
                        return (None, None)
                    self.label_order = picked
                    try:
                        from utility.user_prefs import set_label_order
                        set_label_order(self.label_order)
                    except Exception:
                        pass

                label_order = self.label_order

                for li, label_type in enumerate(label_order):
                    if self.cancel_all:
                        break
                    
                    # Label this type (shows previously labeled types as background)
                    self._label_single_type(
                        label_type=label_type,
                        label_colors=label_colors,
                        total_frames=total_frames,
                        show_previous=label_order[:li],  # Show earlier types as background
                        is_editing=False
                    )
                    
                    if self.cancel_all:
                        break
                
                initial_labeling_done = True
                
                if self.cancel_all:
                    return (None, None)

            # Phase 2: Summary review
            summary_action = self._create_summary_plot(label_colors)
            
            # Phase 3: Handle user's choice
            if summary_action == "accept":
                return (self.label_ranges, "accept")
            elif summary_action == "next_trial":
                return (self.label_ranges, "next_trial")
            elif summary_action in ["edit_fixation", "edit_pursuit", "edit_saccade", "edit_other"]:
                # Edit a specific label type
                label_to_edit = summary_action.replace("edit_", "")
                self._label_single_type(
                    label_type=label_to_edit,
                    label_colors=label_colors,
                    total_frames=total_frames,
                    show_previous=['fixation', 'pursuit', 'saccade', 'other'],  # Show all as background
                    is_editing=True
                )
                continue  # Return to summary after editing
            elif summary_action == "restart_all":
                # Clear all labels and restart from beginning
                self.label_ranges = {'fixation': [], 'pursuit': [], 'saccade': []}
                initial_labeling_done = False
                continue
            else:
                # User cancelled
                self.cancel_all = True
                return (None, None)

    # -------------------------------------------------------------------------
    # Single Label Type Interface
    # -------------------------------------------------------------------------

    def _label_single_type(self, label_type, label_colors, total_frames, show_previous, is_editing):
        """
        Interactive interface for labeling a single eye movement type.
        
        This creates a matplotlib window where users click to select time ranges.
        The interface handles both initial labeling and editing modes with
        appropriate button layouts and instructions.
        
        Parameters
        ----------
        label_type : str
            Type being labeled ('fixation', 'pursuit', or 'saccade').
        label_colors : dict
            Color mapping for each label type.
        total_frames : int
            Total number of frames in trial.
        show_previous : list[str]
            Label types to show as background (for context).
        is_editing : bool
            If True, shows edit UI (Cancel button); if False, shows initial UI (Skip/Quit buttons).
        
        UI Elements
        -----------
        - Plot: Gaze X/Y and target X/Y with event markers
        - Colored spans: Previously labeled periods (background)
        - Black dashed line: Current selection start point
        - Colored highlighting: Current label type selections
        - Status bar: Shows current frame on hover
        - Buttons: Undo, Mark All, Clear, Finish, Skip/Cancel, Quit
        
        Click Behavior
        --------------
        - First click: Sets start point (shown as dashed line)
        - Second click: Creates range from start to click point
        - Auto-trimming: Range is automatically split to skip already-labeled frames
        - Ranges are merged automatically to prevent fragmentation
        
        Performance
        -----------
        - Uses blitting for fast rendering (fallback to traditional redraw if needed)
        - Background is cached for efficient updates
        - Only selection overlays are redrawn during interaction
        
        Notes for Developers
        --------------------
        - Button references MUST be stored in self._btn_refs to prevent garbage collection
        - close_event handler sets action["kind"] if user closes window
        - action["kind"] is checked after plt.show() to determine next step
        """
        # Initialize state - save original for cancel restore
        selected_ranges = list(self.label_ranges.get(label_type, []))
        original_ranges = list(selected_ranges)  # Backup for cancel
        click_state = {'start': None}
        last_hover_frame = [None]
        action = {"kind": None}
        erase_mode = {'active': False}

        # Create plot window
        fig, ax = plt.subplots(figsize=(12, 6))
        try:
            fig.canvas.manager.window.showMaximized()
        except AttributeError:
            pass  # TkAgg or other backends don't support showMaximized
        plt.subplots_adjust(bottom=0.25, top=0.88)

        # Set banner text and color based on mode
        if is_editing:
            banner_text = f"Selected trial: {self.trial_info} - EDITING {label_type.upper()}"
            banner_color = 'lightyellow'
            title_text = f"EDITING {label_type.upper()} — Click start & end (auto-skips labeled frames) | Undo • Mark All • Clear • Finish • Cancel"
        else:
            banner_text = f"Selected trial: {self.trial_info}"
            banner_color = 'lightblue'
            title_text = f"{label_type.upper()} — Click start & end (auto-skips labeled frames) | Undo • Mark All • Clear • Finish • Skip • Quit"

        # Add trial info banner at top
        fig.text(0.5, 0.96, banner_text, 
                ha='center', va='top', fontsize=10, weight='bold',
                bbox=dict(boxstyle="round,pad=0.4", facecolor=banner_color, alpha=0.9))

        # Add status display at bottom
        status_text = fig.text(
            0.02, 0.02, "Ready - Move mouse over plot",
            fontsize=10, ha='left', va='bottom',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8)
        )

        # Plot gaze and target data, * 100 to convert from m to cm.
        ax.plot(self.gaze_x * 100, label='Gaze X')
        ax.plot(self.gaze_y * 100, label='Gaze Y')
        overlay_styles = ['--', '-.', ':', '--', '-.', ':']
        for i, (name, data) in enumerate(self.overlay_channels.items()):
            ax.plot(data * 100, label=name, linestyle=overlay_styles[i % len(overlay_styles)])
        
        # Draw any user-selected marker events with unique colors
        marker_colors = ['purple', 'orange', 'cyan', 'magenta', 'lime', 'pink', 'brown', 'gray']
        for marker_idx, (ev_label, frames) in enumerate(self.marker_frames.items()):
            color = marker_colors[marker_idx % len(marker_colors)]  # Cycle through colors
            for i, frame in enumerate(frames):
                if 0 <= frame < total_frames:
                    ax.axvline(
                        x=frame,
                        color=color,
                        linestyle="-",
                        linewidth=1.8,
                        alpha=0.8,
                        label=(ev_label if i == 0 else "")
                    )

        # Show other labeled types as background (low alpha for context)
        for prev_label in show_previous:
            if prev_label != label_type:  # Don't show current type as background
                for start, end in self._merge_ranges(self.label_ranges.get(prev_label, []), pad=1):
                    ax.axvspan(start, end, color=label_colors[prev_label], alpha=0.15, zorder=0.1)

        # Configure plot appearance
        ax.set_xlim(-20, total_frames - 1 + 20)
        ax.set_title(title_text, pad=15)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Distance (cm)")
        ax.grid(True)
        ax.legend(loc='lower left')

        # Set up efficient rendering with blitting
        fig.canvas.draw()
        fig.canvas.flush_events()  # Process maximize event
        background = fig.canvas.copy_from_bbox(ax.bbox)
        selection_artists = []
        current_line_artist = None

        # ---------------------------------------------------------------------
        # Helper Functions for Rendering
        # ---------------------------------------------------------------------

        def _draw_current_spans(merged_ranges):
            """Render colored highlighting for selected time periods."""
            nonlocal selection_artists
            
            # Remove old span highlights
            for artist in selection_artists:
                try:
                    artist.remove()
                except Exception:
                    pass
            selection_artists.clear()

            # Draw new spans
            for start, end in merged_ranges:
                span = ax.axvspan(start, end, color=label_colors[label_type], alpha=0.35, zorder=1.0)
                selection_artists.append(span)

        def redraw_fast():
            """Efficient plot update using blitting."""
            nonlocal background, current_line_artist
            
            if background is None:
                redraw_traditional()
                return
            
            try:
                # Restore background
                fig.canvas.restore_region(background)
                
                # Draw selection spans
                merged = self._merge_ranges(selected_ranges, pad=1)
                _draw_current_spans(merged)
                for artist in selection_artists:
                    ax.draw_artist(artist)

                # Draw current selection start line
                if current_line_artist:
                    try:
                        current_line_artist.remove()
                    except Exception:
                        pass
                    current_line_artist = None
                
                if click_state['start'] is not None:
                    current_line_artist = ax.axvline(
                        click_state['start'], color='black', linestyle='--', linewidth=1, zorder=2.0
                    )
                    ax.draw_artist(current_line_artist)

                # Update display
                fig.canvas.blit(ax.bbox)
                fig.canvas.flush_events()
                
            except Exception:
                # Fallback to traditional redraw if blitting fails
                redraw_traditional()

        def redraw_traditional():
            """Complete plot redraw (fallback for when blitting fails)."""
            nonlocal current_line_artist
            
            merged = self._merge_ranges(selected_ranges, pad=1)
            _draw_current_spans(merged)

            if current_line_artist:
                try:
                    current_line_artist.remove()
                except Exception:
                    pass
                current_line_artist = None
            
            if click_state['start'] is not None:
                current_line_artist = ax.axvline(
                    click_state['start'], color='black', linestyle='--', linewidth=1, zorder=2.0
                )
            
            fig.canvas.draw_idle()

        # Use fast redraw by default
        redraw = redraw_fast

        # ---------------------------------------------------------------------
        # Event Handlers
        # ---------------------------------------------------------------------

        def on_click(event):
            """Handle mouse clicks for time period selection."""
            if event.inaxes != ax or event.xdata is None:
                return
            
            idx = int(round(event.xdata))
            # Clamp to valid frame range (allows clicking in padding to select edge frames)
            idx = max(0, min(total_frames - 1, idx))
            
            if click_state['start'] is None:
                # First click: Set start point
                click_state['start'] = idx
                redraw()
            else:
                # Second click: Create range
                start, end = sorted([click_state['start'], idx])

                if erase_mode['active']:
                    # ERASE MODE: Remove labels from this range
                    # Remove any ranges that overlap with [start, end]
                    new_ranges = []
                    for r_start, r_end in selected_ranges:
                        # Check if this range overlaps with erase range
                        if r_end < start or r_start > end:
                            # No overlap - keep it
                            new_ranges.append((r_start, r_end))
                        else:
                            # Overlap - split or remove
                            if r_start < start:
                                # Keep part before erase range
                                new_ranges.append((r_start, start - 1))
                            if r_end > end:
                                # Keep part after erase range
                                new_ranges.append((end + 1, r_end))
                    
                    selected_ranges[:] = new_ranges
                else:
                    # NORMAL MODE: Add labels (existing behavior)
                    # Auto-trim to only unlabeled frames
                    trimmed_ranges = self._trim_to_unlabeled(start, end, label_type)
                    
                    # Add all trimmed ranges
                    for trimmed_start, trimmed_end in trimmed_ranges:
                        selected_ranges.append((trimmed_start, trimmed_end))
                
                # Update and merge
                self.label_ranges[label_type] = self._merge_ranges(selected_ranges, pad=1)
                click_state['start'] = None
                redraw()

        def on_mark_all(event):
            """Mark the entire trial range with the current label type."""
            # Get unlabeled ranges for the entire trial
            unlabeled = self._trim_to_unlabeled(0, total_frames - 1, label_type)
            if unlabeled:
                selected_ranges.extend(unlabeled)
                redraw()

        def on_undo(event):
            """Remove most recent selection."""
            if selected_ranges:
                selected_ranges.pop()
                self.label_ranges[label_type] = self._merge_ranges(selected_ranges, pad=1)
                redraw()

        def on_clear(event):
            """Remove all selections for this label type."""
            selected_ranges.clear()
            click_state['start'] = None
            self.label_ranges[label_type] = []
            redraw()

        def on_erase_toggle(event):
            """Toggle erase mode on/off."""
            erase_mode['active'] = not erase_mode['active']
            if erase_mode['active']:
                btn_erase.label.set_text('Erase: ON')
                btn_erase.color = '#FFB6C6'  # Light red
                btn_erase.hovercolor = '#FF8A9A'
                status_text.set_text("ERASE MODE: Click start & end to REMOVE labels")
            else:
                btn_erase.label.set_text('Erase: OFF')
                btn_erase.color = 'lightgray'
                btn_erase.hovercolor = 'gray'
                status_text.set_text("Ready - Move mouse over plot")
            fig.canvas.draw_idle()

        def on_finish(event):
            """Complete labeling for this type and close window."""
            action["kind"] = "finish"
            self.label_ranges[label_type] = self._merge_ranges(selected_ranges, pad=1)
            plt.close(fig)

        def on_skip_or_cancel(event):
            """Skip (initial mode) or Cancel (editing mode)."""
            if is_editing:
                action["kind"] = "cancel"
                self.label_ranges[label_type] = original_ranges  # Restore original ranges
            else:
                action["kind"] = "skip"
                self.label_ranges[label_type] = []
            plt.close(fig)

        def on_quit(event):
            """Cancel entire labeling process (only shown in initial mode)."""
            action["kind"] = "quit"
            self.cancel_all = True
            plt.close(fig)

        def on_hover(event):
            """Update status bar with current frame number."""
            if event.inaxes == ax and event.xdata is not None:
                frame = int(round(event.xdata))
                if frame != last_hover_frame[0]:
                    status_text.set_text(f"Frame: {frame} | Click to select range start/end")
                    last_hover_frame[0] = frame
                    fig.canvas.draw_idle()
            else:
                if last_hover_frame[0] is not None:
                    status_text.set_text("Move mouse over plot to see frame numbers")
                    last_hover_frame[0] = None
                    fig.canvas.draw_idle()

        def on_close(evt):
            """Handle window close (X button)."""
            if action["kind"] is None:
                if is_editing:
                    action["kind"] = "cancel"
                else:
                    self.cancel_all = True

        # Connect event handlers
        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('motion_notify_event', on_hover)
        fig.canvas.mpl_connect('close_event', on_close)

        # ---------------------------------------------------------------------
        # Create Buttons
        # ---------------------------------------------------------------------

        if is_editing:
            # Edit mode: Undo, Mark All, Erase, Clear, Finish, Cancel (centered - 6 buttons)
            ax_undo = plt.axes([0.175, 0.05, 0.10, 0.075])
            ax_mark_all = plt.axes([0.285, 0.05, 0.10, 0.075])
            ax_erase = plt.axes([0.395, 0.05, 0.10, 0.075])
            ax_clear = plt.axes([0.505, 0.05, 0.10, 0.075])
            ax_finish = plt.axes([0.615, 0.05, 0.10, 0.075])
            ax_skip = plt.axes([0.725, 0.05, 0.10, 0.075])

            btn_undo = Button(ax_undo, 'Undo')
            btn_mark_all = Button(ax_mark_all, 'Mark All')
            btn_erase = Button(ax_erase, 'Erase: OFF', color='lightgray', hovercolor='gray')
            btn_clear = Button(ax_clear, 'Clear')
            btn_finish = Button(ax_finish, 'Finish')
            btn_skip = Button(ax_skip, 'Cancel')

            btn_undo.on_clicked(on_undo)
            btn_mark_all.on_clicked(on_mark_all)
            btn_erase.on_clicked(on_erase_toggle)
            btn_clear.on_clicked(on_clear)
            btn_finish.on_clicked(on_finish)
            btn_skip.on_clicked(on_skip_or_cancel)

            # CRITICAL: Store button references to prevent garbage collection
            self._btn_refs = [btn_undo, btn_mark_all, btn_erase, btn_clear, btn_finish, btn_skip]
        else:
            # Initial mode: Undo, Mark All, Clear, Finish, Skip, Quit (centered)
            ax_undo = plt.axes([0.12, 0.05, 0.10, 0.075])
            ax_mark_all = plt.axes([0.23, 0.05, 0.10, 0.075])
            ax_erase = plt.axes([0.34, 0.05, 0.10, 0.075])
            ax_clear = plt.axes([0.45, 0.05, 0.10, 0.075])
            ax_finish = plt.axes([0.56, 0.05, 0.10, 0.075])
            ax_skip = plt.axes([0.67, 0.05, 0.10, 0.075])
            ax_quit = plt.axes([0.78, 0.05, 0.10, 0.075])

            btn_undo = Button(ax_undo, 'Undo')
            btn_clear = Button(ax_clear, 'Clear')
            btn_finish = Button(ax_finish, 'Finish')
            btn_skip = Button(ax_skip, 'Skip')
            btn_quit = Button(ax_quit, 'Quit')
            btn_mark_all = Button(ax_mark_all, 'Mark All')
            btn_erase = Button(ax_erase, 'Erase: OFF', color='lightgray', hovercolor='gray')

            btn_undo.on_clicked(on_undo)
            btn_clear.on_clicked(on_clear)
            btn_finish.on_clicked(on_finish)
            btn_skip.on_clicked(on_skip_or_cancel)
            btn_quit.on_clicked(on_quit)
            btn_mark_all.on_clicked(on_mark_all)
            btn_erase.on_clicked(on_erase_toggle)

            # CRITICAL: Store button references to prevent garbage collection
            self._btn_refs = [btn_undo, btn_mark_all, btn_erase, btn_clear, btn_finish, btn_skip, btn_quit]

        # Initial draw and show
        redraw()

        # Force button rendering (fixes multi-monitor display issue)
        fig.canvas.draw
        fig.canvas.flush_events()
        fig.canvas.draw()

        plt.show()

    # -------------------------------------------------------------------------
    # Summary Interface
    # -------------------------------------------------------------------------

    def _create_summary_plot(self, label_colors):
        """
        Generate final summary visualization with editing and export options.
        
        This creates a comprehensive view showing all eye movement labels
        overlaid on the original data. Users can review the results and choose
        to accept, edit specific label types, restart, mark as bad trial, or
        save and continue to the next trial.
        
        Parameters
        ----------
        label_colors : dict
            Dictionary mapping label types to display colors.
        
        Returns
        -------
        str or None
            User's choice: 'accept', 'next_trial', 'edit_fixation', 'edit_pursuit',
            'edit_saccade', 'restart_all', or None if cancelled.
        
        UI Elements
        -----------
        - Complete data plot with all labels overlaid
        - Color-coded buttons for editing each label type
        - "Mark as Bad Trial" toggle (changes export behavior)
        - "Save & Next Trial" for batch processing workflow
        - "Save & Finish" to complete labeling
        
        Button Behaviors
        ----------------
        - Edit buttons: Return to editing mode for that specific type
        - Restart All: Clear all labels and start over from fixation
        - Mark as Bad Trial: Toggle flag (affects export - all frames become code 9)
        - Save & Next: Accept current labels and automatically load next trial
        - Save & Finish: Accept current labels and return to main GUI
        
        Notes
        -----
        - Bad trial toggle updates button appearance to show current state
        - All button references must be stored to prevent garbage collection
        - Close event returns None (treated as cancel)
        """
        # Create summary plot with space for buttons
        fig, ax = plt.subplots(figsize=(12, 7))
        try:
            fig.canvas.manager.window.showMaximized()
        except AttributeError:
            pass  # TkAgg or other backends don't support showMaximized
        plt.subplots_adjust(bottom=0.15, top=0.88)
        
        # Add trial info banner at top
        fig.text(0.5, 0.96, f"Selected trial: {self.trial_info}", 
                ha='center', va='top', fontsize=10, weight='bold',
                bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.9))
        
        # Plot original gaze and target data, * 100 to convert from m to cm
        ax.plot(self.gaze_x * 100, label='Gaze X')
        ax.plot(self.gaze_y * 100, label='Gaze Y')
        overlay_styles = ['--', '-.', ':', '--', '-.', ':']
        for i, (name, data) in enumerate(self.overlay_channels.items()):
            ax.plot(data * 100, label=name, linestyle=overlay_styles[i % len(overlay_styles)])
        
        # Draw any user-selected marker events with unique colors
        marker_colors = ['purple', 'orange', 'cyan', 'magenta', 'lime', 'pink', 'brown', 'gray']
        for marker_idx, (ev_label, frames) in enumerate(self.marker_frames.items()):
            color = marker_colors[marker_idx % len(marker_colors)]  # Cycle through colors
            for i, frame in enumerate(frames):
                if 0 <= frame < len(self.gaze_x):
                    ax.axvline(
                        x=frame,
                        color=color,
                        linestyle="-",
                        linewidth=1.8,
                        alpha=0.8,
                        label=(ev_label if i == 0 else "")
                    )

        # Overlay all labeled events with transparency
        for label_type, ranges in self.label_ranges.items():
            for start, end in self._merge_ranges(ranges, pad=1):
                ax.axvspan(start, end, color=label_colors[label_type], alpha=0.25)

        # Clean up legend (remove duplicate labels)
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        unique_handles, unique_labels = [], []
        for handle, label in zip(handles, labels):
            if label not in seen:
                unique_handles.append(handle)
                unique_labels.append(label)
                seen.add(label)

        # Configure plot appearance
        ax.set_xlim(0, len(self.gaze_x) - 1)
        try:
            # Remove trial number prefix for cleaner display
            display_name = "_".join(self.trial_name.split("_")[1:])
        except:
            display_name = self.trial_name

        ax.set_title(f"Summary — {display_name}", pad=15)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Distance (cm)")
        ax.grid(True)
        ax.legend(unique_handles, unique_labels)
        
        # Track user's choice
        choice = {"method": None}
        
        # ---------------------------------------------------------------------
        # Button Handlers
        # ---------------------------------------------------------------------

        def on_bad_trial_toggle(event):
            """Toggle the bad trial flag and update button appearance."""
            self.bad_trial = not self.bad_trial
            if self.bad_trial:
                btn_bad_trial.label.set_text('✓ Bad Trial\n(exports 9)')
                btn_bad_trial.color = 'lightcoral'
                btn_bad_trial.hovercolor = 'red'
            else:
                btn_bad_trial.label.set_text('Mark as\nBad Trial')
                btn_bad_trial.color = 'lightgray'
                btn_bad_trial.hovercolor = 'gray'
            fig.canvas.draw_idle()

        def on_edit_fixation(event):
            choice["method"] = "edit_fixation"
            plt.close(fig)
        
        def on_edit_pursuit(event):
            choice["method"] = "edit_pursuit"
            plt.close(fig)
        
        def on_edit_saccade(event):
            choice["method"] = "edit_saccade"
            plt.close(fig)

        def on_edit_other(event):
            choice["method"] = "edit_other"
            plt.close(fig)
        
        def on_restart(event):
            confirm_root = tk.Tk()
            confirm_root.withdraw()
            confirm_root.attributes("-topmost", True)
            result = messagebox.askyesno("Confirm Restart", "Are you sure you want to restart? All labels will be cleared.", parent=confirm_root)
            confirm_root.destroy()
            if result:
                choice["method"] = "restart_all"
                plt.close(fig)
            else:
                fig.canvas.manager.window.activateWindow()
        
        def on_next_trial(event):
            choice["method"] = "next_trial"
            plt.close(fig)
        
        def on_accept(event):
            choice["method"] = "accept"
            plt.close(fig)
        
        def on_close(event):
            if choice["method"] is None:
                choice["method"] = None
        
        # ---------------------------------------------------------------------
        # Create Buttons
        # ---------------------------------------------------------------------

        button_y = 0.02
        button_height = 0.06
        
        # Button layout: Edit buttons, Restart, Next Trial, Accept, Bad Trial (top row)
        ax_bad_trial = plt.axes([0.105, button_y, 0.09, button_height])
        ax_edit_fix = plt.axes([0.205, button_y, 0.09, button_height])
        ax_edit_pur = plt.axes([0.305, button_y, 0.09, button_height])
        ax_edit_sac = plt.axes([0.405, button_y, 0.09, button_height])
        ax_edit_other = plt.axes([0.505, button_y, 0.09, button_height])
        ax_restart = plt.axes([0.605, button_y, 0.09, button_height])
        ax_next = plt.axes([0.705, button_y, 0.09, button_height])
        ax_accept = plt.axes([0.805, button_y, 0.09, button_height])
        
        # Create buttons with colors matching label types
        btn_fix = Button(ax_edit_fix, 'Edit\nFixation', color=label_colors['fixation'], hovercolor='lightgreen')
        btn_pur = Button(ax_edit_pur, 'Edit\nPursuit', color=label_colors['pursuit'], hovercolor='cornflowerblue')
        btn_sac = Button(ax_edit_sac, 'Edit\nSaccade', color=label_colors['saccade'], hovercolor='lightcoral')
        btn_other = Button(ax_edit_other, 'Edit\nOther', color=label_colors['other'], hovercolor='lightyellow')
        btn_restart = Button(ax_restart, 'Restart\nAll', color='lightgray', hovercolor='gray')
        btn_next = Button(ax_next, 'Save &\nNext Trial', color='lightyellow', hovercolor='yellow')
        btn_accept = Button(ax_accept, 'Save &\nFinish', color='lightgreen', hovercolor='green')
        
        # Bad trial button appearance depends on current state
        if self.bad_trial:
            btn_bad_trial = Button(ax_bad_trial, '✓ Bad Trial\n(exports 9)', color='lightcoral', hovercolor='red')
        else:
            btn_bad_trial = Button(ax_bad_trial, 'Mark as\nBad Trial', color='lightgray', hovercolor='gray')
        
        # Connect button handlers
        btn_fix.on_clicked(on_edit_fixation)
        btn_pur.on_clicked(on_edit_pursuit)
        btn_sac.on_clicked(on_edit_saccade)
        btn_restart.on_clicked(on_restart)
        btn_other.on_clicked(on_edit_other)
        btn_next.on_clicked(on_next_trial)
        btn_accept.on_clicked(on_accept)
        btn_bad_trial.on_clicked(on_bad_trial_toggle)
        
        # Connect close event
        fig.canvas.mpl_connect('close_event', on_close)
        
        # CRITICAL: Keep button references to prevent garbage collection
        self._summary_btn_refs = [btn_fix, btn_pur, btn_sac, btn_other, btn_restart, btn_next, btn_accept, btn_bad_trial]
        
        plt.show()
        
        return choice["method"]
    

    def _choose_label_order(self):
        """
        Matplotlib dialog to choose label order using a clickable list + Up/Down.
        Returns: list[str] order, or None if user closes/cancels.
        """
        labels = ['fixation', 'pursuit', 'saccade']

        existing = getattr(self, "label_order", None)
        if not existing:
            order = labels[:]
        else:
            order = list(existing)
            if sorted(order) != sorted(labels):
                order = labels[:]

        fig = plt.figure(figsize=(6.2, 3.2))
        try:
            fig.canvas.manager.set_window_title("Choose Label Order")
        except Exception:
            pass

        fig.text(
            0.5, 0.92,
            "Choose labeling order (used for the initial pass)",
            ha="center", va="top", fontsize=10, weight="bold"
        )

        order_text = fig.text(0.5, 0.82, " → ".join(order), ha="center", va="top", fontsize=10)

        # Clickable list area (this is what visibly reorders)
        ax_list = fig.add_axes([0.08, 0.18, 0.36, 0.50])
        ax_list.set_axis_off()

        chosen = {"idx": 0, "done": False, "cancel": False}

        list_text_artists = []

        def redraw_list():
            nonlocal list_text_artists
            for t in list_text_artists:
                try:
                    t.remove()
                except Exception:
                    pass
            list_text_artists = []

            for i, name in enumerate(order):
                is_sel = (i == chosen["idx"])
                txt = ax_list.text(
                    0.05, 0.85 - i * 0.30,
                    f"{i+1}. {name}",
                    transform=ax_list.transAxes,
                    fontsize=10,
                    weight="bold" if is_sel else "normal",
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor="lightgray" if is_sel else "white",
                        edgecolor="black" if is_sel else "none",
                        alpha=0.9 if is_sel else 0.6
                    )
                )
                list_text_artists.append(txt)

            order_text.set_text(" → ".join(order))
            fig.canvas.draw_idle()

        def on_click(event):
            if event.inaxes != ax_list:
                return
            if event.ydata is None:
                return
            # Map y position to item index (3 items)
            # Axes coords: top ~1, bottom ~0
            y_ax = event.ydata
            # Convert to approximate index based on our placement
            # y positions are at 0.85, 0.55, 0.25
            candidates = [0.85, 0.55, 0.25]
            diffs = [abs(y_ax - c) for c in candidates]
            idx = int(np.argmin(diffs))
            chosen["idx"] = max(0, min(len(order)-1, idx))
            redraw_list()

        fig.canvas.mpl_connect("button_press_event", on_click)

        # Buttons
        ax_up     = fig.add_axes([0.50, 0.58, 0.18, 0.12])
        ax_down   = fig.add_axes([0.72, 0.58, 0.18, 0.12])
        ax_reset  = fig.add_axes([0.50, 0.42, 0.40, 0.12])
        ax_start  = fig.add_axes([0.50, 0.22, 0.40, 0.14])
        ax_cancel = fig.add_axes([0.50, 0.06, 0.40, 0.12])

        btn_up     = Button(ax_up, "Up")
        btn_down   = Button(ax_down, "Down")
        btn_reset  = Button(ax_reset, "Reset (Fix→Pur→Sac)")
        btn_start  = Button(ax_start, "Start Labeling")
        btn_cancel = Button(ax_cancel, "Cancel")

        def move_up(_):
            i = chosen["idx"]
            if i <= 0:
                return
            order[i-1], order[i] = order[i], order[i-1]
            chosen["idx"] = i - 1
            redraw_list()

        def move_down(_):
            i = chosen["idx"]
            if i >= len(order) - 1:
                return
            order[i+1], order[i] = order[i], order[i+1]
            chosen["idx"] = i + 1
            redraw_list()

        def reset(_):
            order[:] = labels[:]
            chosen["idx"] = 0
            redraw_list()

        def start(_):
            chosen["done"] = True
            plt.close(fig)

        def cancel(_):
            chosen["cancel"] = True
            plt.close(fig)

        def on_close(_evt):
            if not chosen["done"]:
                chosen["cancel"] = True

        btn_up.on_clicked(move_up)
        btn_down.on_clicked(move_down)
        btn_reset.on_clicked(reset)
        btn_start.on_clicked(start)
        btn_cancel.on_clicked(cancel)
        fig.canvas.mpl_connect("close_event", on_close)

        self._order_btn_refs = [btn_up, btn_down, btn_reset, btn_start, btn_cancel]

        redraw_list()
        plt.show()

        if chosen["cancel"]:
            return None
        return order
