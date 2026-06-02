# Changelog

## v1.5.2 - 2026-06-02
### Configuration
- Add external config.json for lab-configurable parameters (filter cutoffs, eye height, FVR threshold, etc.)
- Edit values next to GECKO.exe to override defaults without touching source code

### Bug Fixes
- Fix hardcoded filter cutoff (20 Hz) in gaze labeler export to respect config
- Fix hardcoded KINARM sentinel threshold in gaze labeler export to respect config
- Fix lowpass_filter() to use trial frame rate instead of hardcoded 1000 Hz default
- Fix labeler channel pre-selection exceeding MAX_LABELER_CHANNELS cap
- - Fix labeler truncating trials longer than 5000 frames (closes #21)

### Build
- Switch PyInstaller build to onedir for faster startup and config file accessibility

### Internal
- Remove unused DEFAULT_GAZE_SAMPLING_HZ constant
- Update help text to list all 4 Clear Cache options
- Add x/y axis labels to channel inspect plot

## v1.5.1 - 2026-05-07
- Fixed interpolation window crash
- Added warning when selecting 4+ channels to inspect

## v1.5.0 - 2026-03-19
### Per-Gap Interpolation
- Each large data gap is now handled individually, allowing different strategies per gap within the same trial

### Gaze Labeler
- Additional overlay channels can now be selected for display (e.g. xT, yT). Gaze_X and Gaze_Y always included
- 'Restart All' now requires confirmation dialog

### Session State
- Resume prompt now shows trial number and display name instead of raw internal identifier
- Split resume prompt into two separate questions (jump to trial vs. restore channel selections)
- Session state saved on close instead of on every click
- Removed overlap between session_state.json and user_prefs.json

### Help Window
- Simplified to static content, removed refresh button and dynamic file path display

### Performance
- Interpolation window no longer flashes or reloads between gaps; figure is reused across all gaps

### Internal
- Codebase restructured into smaller modules
- Lab-configurable constants centralized in user_prefs.py

## v1.4.1 - 2026-04-30
- Fixed interpolation window crash
- Added warning when selecting 4+ channels to view

## v1.4.0 - 2026-02-03
### Main Menu
- UI consistency improvements: scrollbars, clearly labeled button groups
- Channel search bar to filter inspect and export channels
- Session state saved as .json, offers resume on reopen
- "Clear Cache" replaces "Clear Interpolation Cache" with options for interpolation, labeling order, or session state
- Added Help button under Utilities

### Gaze Labeler
- Fixed visual glitch where smaller plot would flash or remain stuck
- Labeler and interpolation windows now open maximized
- Fixed legend flickering during labeling
- Centered buttons at bottom of interface
- Trial progress indicator (e.g. "2/30. 05_01")
- Custom labeling order selection on first use, persistent until cleared
- "Edit Other" button for general/unknown gaze events (exports as 0)
- Y-axis label changed to "Distance (cm)"
- Toggleable Erase mode for removing labeled segments
- ±20 frame buffer for selecting near trial boundaries

## v1.3.2 - 2026-01-21
- Fixed issue where timestamp channels with values > 99.9 would incorrectly become NaN
- Notes and marks now save to separate .csv file (along with Target_Table, TP_Table)
- Custom save location now saved across sessions
- Added button to restore default save location
- Added npz subfolder within export folder to reduce clutter

## v1.3.1 - 2026-01-20
- Removed hardcoded TARGET_ON and TP_ON vertical lines in gaze labeler
- Added event marker selection box in main menu for choosing labeler overlay channels
- Fixed issue where populated export channels wouldn't save and export properly
- Bolded section headings on main menu

## v1.3.0 - 2026-01 (approx.)
- Restructured and organized code, added initial source code comments
- Fixed incorrect units in gaze metrics, velocity, and FVR calculations
- Export channel selections now save across sessions
- Added xT and yT to interpolation selection plot backgrounds
- Fixed interpolation window appearing for previous trial after closing or advancing

## v1.2.1 - 2025-12 (approx.)
- Fixed exported filename format for .csv/.npz
- Fixed gaze labeler information bar format
- Fixed bug where export channels would deselect after quitting gaze labeler
- Added dynamic event channel export (no longer hardcoded; populates from EVENT_DEFINITIONS in .kinarm file)

## v1.2.0 - 2025-12-08
- "Mark as bad" option in labeler summary screen (exports 9s instead of 1,2,3)
- "Mark all" button to fill empty gaps with current gaze event selection
- Buttons greyed out until prerequisites are met (e.g. trial selected, file loaded)
- Second status bar displaying current save location
- Simplified trial naming (removed first two numbers)
- Centered most GUI windows
- Fixed crash when selecting new trial after closing Task Protocol window
- Simple note feature for attaching notes to trials
- Moved Analog Inputs into "Show Parameters" with tabbed layout

## v1.1.0 - 2025-11-03
- Added HOLD_AT_TARGET marker along with TARGET_ON in Label Gaze
- "Set Save Location" to choose export folder (defaults to Desktop)
- Matplotlib backend closing improvements for crash resistance
- Fixed export options deselecting after labeling and saving
- "Mark Good/Bad/Review" for trials, saves persistently
- Auto-trimming in label gaze (overlapping selections automatically clip to fill gaps)
- Trial # and TP # added to interpolation previews
- Added crash report (sends crash file to desktop)
- Fixed inability to select different trial with export channels selected

## v1.0.0
- Initial GUI release
