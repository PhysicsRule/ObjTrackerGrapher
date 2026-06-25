# ObjTracker — CLAUDE.md

## Purpose

A desktop GUI application for 3D object tracking using an Intel RealSense D435i depth camera. It tracks objects by color or infrared bounding-box methods, records their 3D coordinates over time, and graphs position, velocity, acceleration, momentum, and energy. Data is saved as CSV files and can be replayed from `.bag` recordings.

Designed for physics education — students track real objects (balls, etc.) and analyze their motion in 3D.

---

## Repository Structure

```
ObjTracker_py311/
├── src/
│   ├── GUI_main_program.py        # Main window class; loads .ui, wires signals, owns the matplotlib canvas
│   └── tracker/
│       └── lib/
│           ├── GUI_Base.ui        # Qt Designer layout (PyQt5); loaded at runtime via uic.loadUi()
│           ├── GUI_tracker.py     # Real-time tracking loops (color + infrared/obj_tracker modes)
│           ├── GUI_tables.py      # Loads/reloads color-definition tables; reload_table() modifies in-place
│           ├── color.py           # HSV mask helpers, GUI color-picker (find_object_by_color, GUI_find_hsv_bounds)
│           ├── object_tracking.py # OpenCV bounding-box tracker (CSRT/etc.), bbox helpers
│           ├── intel_realsense_D435i.py  # RealSense pipeline setup, frame capture, depth → meters
│           ├── setup_files.py     # Creates output folder structure and CSV files per tracked object
│           └── general.py        # save_video_file() and other shared utilities
└── CLAUDE.md
```

### Key runtime relationships

- `GUI_main_program.py` calls `uic.loadUi('...GUI_Base.ui', self)` — every widget `name=` in the .ui becomes `self.<name>` immediately after that call. **Widget names must never be renamed without updating Python references.**
- `self.graph_widget` is replaced in Python with a matplotlib canvas (`mlpcanvas()`), then added to the pre-existing `grid_layout` (QGridLayout) via `self.grid_layout.addWidget(self.graph_widget, 0, 0)`.
- `help_menu_2` (QFrame, zero size) is kept as a dead widget so no AttributeError occurs if old code paths reference it.
- `combo_test` (QComboBox, zero size) is a legacy hidden widget — do not remove.
- The three color tables (`table_widget_color`, `table_widget_color_2`, `table_widget_objects`) intentionally **overlap at the same position** inside `step3_group`; Python shows/hides them based on radio-button selection.
- `reload_table()` modifies `table_widget_color_2` in-place and returns the same object — no reparenting.

---

## Coding Conventions

- NEVER use global despite some existing code using it
- type annotations everywhere (existing code is hit or miss, new code SHOULD have type annotations)
- use existing functions/patterns when helpful



---

## UI Layout (GUI_Base.ui)

The window is **1637 × 900 px**, absolute-positioned throughout (no Qt layout managers on `centralwidget`).

### Left panel — `left_scroll_area` (x=0, w=540)
A `QScrollArea` containing steps 1–5. Each `QGroupBox` has exactly **6 px bottom padding** after its last widget.

| Group | Step | Contents |
|---|---|---|
| `step1_group` | Camera / Data Source | 5 camera-mode buttons + "data from another source" checkbox |
| `step2_group` | Output Folder | Folder name input (`folder_name`) + existing-folders list |
| `step3_group` | Color Setup | 3 radio buttons + color-selector controls + 3 overlapping tables |
| `step4_group` | Display & Save | Show/Save checkboxes (2 columns); Save Images Video in right column |
| `step5_group` | D435i + Run Tracker | D435i note, 4 run buttons, bag-file warning |

### Right panel — direct children of `centralwidget`
Two-column layout:

| Widget | x | width | Purpose |
|---|---|---|---|
| `step6Group` | 540 | 440 | Variable selection (accel/momentum/energy), axis/height, Graph/3D/Trendlines buttons |
| `gridLayoutWidget` | 540 | 756 | Graph area; contains `grid_layout` → `graph_widget` → `DataGraph` (PlotWidget) |
| `step7Group` | 1300 | 329 | Trendline analysis label + `trendline_table_widget` |

`step7Group` is a full-height right column (h=870) that runs alongside the graph.

---

## Design Constraints

### CPU only — no GPU
- Do **not** recommend or introduce GPU-dependent libraries (CUDA, PyTorch/GPU, TensorFlow/GPU, OpenCL).
- OpenCV is used in CPU mode only.
- The application must run on any laptop, including older machines with integrated graphics.
- Tracking algorithms must remain lightweight enough for real-time use on CPU.

### Portable distribution via PyInstaller
- The app is packaged with **PyInstaller** into a standalone executable for portable USB deployment.
- All dependencies must be PyInstaller-compatible (no packages that rely on dynamic loading tricks that PyInstaller can't bundle).
- Avoid adding dependencies that are large or that PyInstaller routinely fails to bundle (e.g. packages with heavy C extensions needing special hooks).
- Data files (`.ui`, config, default color CSVs) must be referenced via PyInstaller's `sys._MEIPASS` path helper when accessed at runtime inside a frozen build.
- Poetry was explored but abandoned; use standard `pip`/`requirements.txt` for dependency management.

### Python version
Python **3.11** (reflected in the repo name `ObjTracker_py311`).

### Key dependencies
| Package | Purpose |
|---|---|
| `PyQt5` | GUI framework; `.ui` loaded via `uic.loadUi()` |
| `pyqtgraph` | `PlotWidget` / `DataGraph` live plot widget |
| `opencv-python` | Camera capture, color masking, bounding-box tracking, video saving |
| `pyrealsense2` | Intel RealSense D435i pipeline (depth + color + infrared frames) |
| `numpy` | Array operations on frame data and tracking results |
| `matplotlib` | Graph canvas embedded in `graph_widget` via `mlpcanvas` |

---

## Conventions

- **Widget names are sacred.** Python accesses every widget as `self.<name>` after `uic.loadUi()`. Renaming a widget in the `.ui` without updating every Python reference will cause an `AttributeError` at startup.
- Output data folders live under `data/color_o`, `data/infrared_o`, `data/other_o` by convention.
- Tracked object coordinates are written as `time,x,y,z` CSV lines, one file per tracked color/object.
- `.bag` files are RealSense recordings; new bag files must have unique names or the RealSense SDK will error.
