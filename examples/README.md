# Example Data

## demo_1.kinarm

A sample KINARM data file included so new users and reviewers can try GECKO's full workflow (loading, inspecting, labeling, and exporting) without needing their own lab's data.

### Usage

1. Launch GECKO (`gui_main.py` or the packaged `.exe`)
2. Click **Load .kinarm File** and select `examples/demo_1.kinarm`
3. Explore the loaded trials:
   - **Inspect Channels** to view raw kinematic and gaze data
   - **Label Gaze** to try the manual gaze event labeling workflow
   - **Export** to see the CSV/NPZ output format

### Data Gaps (Trials 4-6)

Trials 4-6 contain intentionally placed data gaps, useful for testing GECKO's interpolation features:
- Auto-interpolation for small gaps (below the configurable `AUTO_INTERP_THRESHOLD_FRAMES` threshold)
- Manual interpolation method selection (linear vs. saccadic) for larger gaps

### Notes

- This file is provided for demonstration and testing purposes only.
- Exported output will be written to your configured save location (defaults to `Desktop/gaze_labels`), not this `examples/` folder.