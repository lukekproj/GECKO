# GECKO

**Gaze Event Classification in the KINARM, Open-access**

<img width="514" height="514" alt="Untitled" src="https://github.com/user-attachments/assets/5f6462b1-7cd8-4f49-a42d-c4771e202c9a" />

GECKO is an open-source Python desktop application for visualizing, annotating, and exporting gaze and kinematic data from KINARM robotic exoskeleton experiments. It provides a researcher-in-the-loop workflow for classifying gaze events (fixations, saccades, and smooth pursuits) with frame-level precision, reducing manual analysis time per trial to seconds.

Developed by the [Sensorimotor Neuroscience and Learning Laboratory](https://sites.psu.edu/singhlab/) at Pennsylvania State University.

---

## Repository Structure

```
GECKO/
│
├── gui/                          # Main application (tkinter GUI)
│   ├── gui_main.py                # Entry point — main window, file loading, session state
│   ├── gui_labeler.py              # Labeler controller (launches GazeLabeler)
│   ├── gui_channel_panel.py        # Channel list, filtering, inspection plots
│   ├── gui_export_panel.py         # Export channel selection panel
│   ├── gui_trial_panel.py          # Trial list, marks, notes
│   ├── gui_session.py              # Session state persistence (resume, Trial_Notes.csv)
│   ├── gui_metadata_viewer.py      # Task Protocol / metadata viewer
│   ├── gui_help.py                 # In-app Help/Info window
│   └── gui_utils.py                # Shared GUI helpers
│
├── label/                        # Gaze labeling & export pipeline
│   ├── gaze_labeler_ui.py          # Interactive labeling interface (click-to-label, blitting)
│   └── gaze_labeler_export.py      # Label → gaze_events → CSV/NPZ export
│
├── data/                         # Core data pipeline (GUI-independent)
│   ├── exam_load.py                # .kinarm binary file parser
│   ├── data_loader.py              # KinarmDataExplorer — trial/channel access, interpolation entrypoint
│   ├── data_calculations.py        # GazeCalculator — spherical coords, angular velocity, FVR
│   └── data_interpolation.py       # Sentinel cleaning, gap detection, linear/saccadic interpolation
│
├── utility/                      # Shared utilities
│   ├── user_prefs.py               # Config constants, config.json loading, preferences
│   └── kinarm_utils.py             # Trial parameter table helpers
│
├── tests/                        # pytest test suite
│   ├── test_data_interpolation.py  # Sentinel cleaning, gap detection, buffer expansion
│   ├── test_data_calculations.py   # Spherical coordinate / angular velocity / FVR math
│   └── test_user_prefs.py          # config.json loading behavior
│
├── examples/                     # Demo dataset for new users and reviewers
│   ├── demo_1.kinarm               # Sample KINARM file (includes intentional data gaps)
│   └── README.md
│
├── docs/                         # User-facing documentation
│   ├── GECKO_User_Manual.pdf
│   └── GECKO_Technical_Reference.pdf
│
├── paper/                        # JOSS manuscript
│   ├── paper.md
│   ├── paper.bib
│   └── kinarm_gaze_geometry.jpg
│
├── .github/workflows/            # CI
│   ├── draft-pdf.yml               # Builds JOSS paper PDF preview
│   ├── build-mac.yml               # Builds macOS executable
│   └── tests.yml                   # Runs pytest suite on push/PR
│
├── config.json                   # Lab-configurable processing parameters
├── requirements.txt                # Runtime dependencies
├── requirements-dev.txt            # Dev dependencies (pytest)
├── pytest.ini                      # pytest configuration
├── CHANGELOG.md
├── CONTRIBUTING.md
├── GECKO_mac.spec                  # PyInstaller spec (macOS build)
├── .gitattributes                  # Ensures binary files (PDFs) aren't corrupted by line-ending conversion
├── .gitignore
└── LICENSE
```

---

## Documentation

- [User Manual](docs/GECKO_User_Manual.pdf)
- [Technical Reference](docs/GECKO_Technical_Reference.pdf)

---

## Features

- Load and browse `.kinarm` files trial-by-trial
- Visualize any data channel with smart interpolation for missing samples
- Manually label gaze events (fixation, saccade, smooth pursuit) with click-based precision
- Export labeled trials to `.csv` and `.npz` with any combination of kinematic, gaze, and derived channels
- Compute and export gaze metrics: spherical coordinates (ρ, θ, φ), angular velocity, and foveal visual radius (FVR)
- Event marker overlays during labeling (e.g. TARGET_ON, TGT_REACHED)
- Trial quality marking (Good / Bad / Review) with notes
- Session resume — pick up where you left off across sessions
- Lab-configurable processing parameters via `config.json`

---

## Installation

### Executable (Recommended)
1. Download the latest release from the [Releases](../../releases) page
2. Extract the zip file
3. Double-click `GECKO.exe` to launch

No installation or Python required.

### From Source
Requires Python 3.10+

```bash
git clone https://github.com/lukekproj/GECKO.git
cd GECKO
pip install -r requirements.txt
python gui/gui_main.py
```

---

## Quick Start

1. **Set Save Location** — choose where exported files will be saved before loading any data
2. **Load `.kinarm` File** — populates the trial list
3. **Select a Trial** — channels populate automatically
4. **Select Event Markers** — choose events to display as vertical lines during labeling (e.g. TARGET_ON)
5. **Select Channels to Export** — choose which data channels appear in the output CSV
6. **Label Events** — opens the gaze labeling interface; interpolation is offered automatically if large gaps are detected
7. **Save & Finish / Save & Next Trial** — exports CSV and NPZ for the trial

Repeat steps 2–7 for each trial. All export and marker selections persist across trials.

New to GECKO? Watch the [video walkthrough](https://psu.mediaspace.kaltura.com/media/t/1_04zjsojj) using the included demo dataset.

---

## Output Files

For each labeled trial, GECKO saves:

| File | Description |
|------|-------------|
| `Trial{N}.TP{X}.C{Y}.csv` | Frame-by-frame gaze events and selected channels |
| `npz/Trial{N}.TP{X}.C{Y}.npz` | Compressed NumPy equivalent of the CSV |
| `Target_Table.csv` | Target workspace definitions from the `.kinarm` file |
| `TP_Table.csv` | Task protocol parameters |
| `Trial_Notes.csv` | Trial quality marks and researcher notes |

### Gaze Event Codes

| Code | Event |
|------|-------|
| 0 | Unlabeled / Other |
| 1 | Saccade |
| 2 | Smooth Pursuit |
| 3 | Fixation |
| 9 | Bad Trial |

---

## Configuration

A `config.json` file is included alongside `GECKO.exe`. Edit any value to override built-in defaults (filter cutoffs, eye height, FVR threshold, labeler channel limit, etc.) without touching source code. Delete the file to restore all defaults.

> **When updating GECKO:** note any custom values before updating — new releases ship with default values and will not preserve your changes.

---

## Requirements

### Executable
- Windows 10 or 11
- No Python required — all dependencies are bundled

### From Source
- Python 3.10+
- Tested on Windows; macOS and Linux not officially supported
- `.kinarm` files generated from KINARM Endpoint or Exoskeleton experiments with gaze data

Python dependencies:
```
numpy
scipy
matplotlib
PyQt5
pandas
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on reporting bugs, requesting features, and submitting pull requests.

---

## Persistent Storage

GECKO stores user preferences and session state outside the application folder so they survive updates:

| File | Location | Purpose |
|------|----------|---------|
| `user_prefs.json` | `AppData\Roaming\KinarmDataExplorer\` (Windows) | Export defaults, marker selections, save location |
| `session_state.json` | Inside your save location folder | Session resume state per `.kinarm` file |
| `Trial_Notes.csv` | Inside your save location folder | Trial marks and notes per `.kinarm` file |
