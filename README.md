# 3D Scene Analysis

Visualise and tabulate the world-space layout of objects reconstructed by the
[SAM3D API](https://github.com/markhorsfield/sam3d-api).  Given a `scene.glb`
and a `_labels.txt` file produced by a reconstruction job, the tool renders
three orthographic 3-D views (front, side, top) and prints a summary table of
every object's position and size.

---

## Requirements

- Python 3.9+
- [numpy](https://numpy.org/)
- [trimesh](https://trimesh.org/)
- [matplotlib](https://matplotlib.org/)

Install everything in one step:

```bash
pip install -r requirements.txt
```

> **Tip — Apple Silicon / miniforge:**  
> If the system Python lacks these packages, prefix every command with the
> full interpreter path, e.g. `/Users/mark/miniforge3/bin/python3 analyse_scene.py …`

---

## Quick start

```bash
# All outputs (table + three views) for a SAM3D results directory
python analyse_scene.py path/to/results_dir

# Table only
python analyse_scene.py path/to/results_dir --table

# Single view
python analyse_scene.py path/to/results_dir --front
python analyse_scene.py path/to/results_dir --side
python analyse_scene.py path/to/results_dir --top

# Combine selectively
python analyse_scene.py path/to/results_dir --front --side --table

# Custom file paths
python analyse_scene.py --glb path/to/scene.glb --labels path/to/_labels.txt

# Write images to a different directory
python analyse_scene.py path/to/results_dir --out-dir /tmp/plots
```

---

## Output

### Table (stdout)

```
Idx  Label                                       Centroid (x, y, z)       W (X)   H (Y)   D (Z)    Bot Y   Top Y
-------------------------------------------------------------------------------------------------------------------
0    victorian corner building                (  0.012,  0.048, -0.023)   1.243   1.102   0.981    -0.503   0.599
2    remfry bird seed delivery wagon          ( -0.231, -0.198,  0.114)   0.412   0.287   0.330    -0.342  -0.055
…
```

Columns:

| Column | Description |
|--------|-------------|
| `Idx` | Object index (matches the label file) |
| `Label` | Human-readable label from `_labels.txt` |
| `Centroid (x, y, z)` | World-space centre of the bounding box |
| `W (X)` | Width along the X axis |
| `H (Y)` | Height along the Y axis (up = positive) |
| `D (Z)` | Depth along the Z axis |
| `Bot Y` | Y coordinate of the bottom face |
| `Top Y` | Y coordinate of the top face |

### Images

Three PNG files are written to the output directory:

| File | View |
|------|------|
| `scene_layout_front.png` | Front perspective (elev 10°, azim −60°) |
| `scene_layout_side.png` | Side view — reveals vertical (Y) misalignment |
| `scene_layout_top.png` | Bird's-eye view (elev 85°) |

Each image shows:
- Colour-coded semi-transparent bounding boxes for every object
- A centroid marker and index label
- A semi-transparent ground plane
- A legend mapping each index to its label

---

## File conventions

The script expects the following layout inside a SAM3D results directory:

```
results_dir/
├── scene.glb        # trimesh scene — one geometry per reconstructed object
└── _labels.txt      # tab-separated: index <TAB> label <TAB> score <TAB> area <TAB> box
```

Override either path with `--glb` / `--labels` if your files live elsewhere.

---

## Usage reference

```
usage: analyse_scene.py [-h] [--glb GLB] [--labels LABELS] [--out-dir OUT_DIR]
                        [--table] [--front] [--side] [--top] [--all]
                        [results_dir]

positional arguments:
  results_dir        Job results directory (contains scene.glb and _labels.txt)

optional arguments:
  --glb GLB          Explicit path to scene.glb
  --labels LABELS    Explicit path to _labels.txt
  --out-dir OUT_DIR  Directory to write output images (default: results_dir)

output selection (default: all):
  --table            Print table to stdout
  --front            Render front view
  --side             Render side view (shows Y misalignment)
  --top              Render top/bird's-eye view
  --all              All outputs (default if none specified)
```
