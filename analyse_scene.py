#!/usr/bin/env python3
"""
analyse_scene.py — Visualise and tabulate the world-space layout of objects
reconstructed by the SAM3D API.

Usage:
    python analyse_scene.py <results_dir>              # all outputs
    python analyse_scene.py <results_dir> --table      # table only
    python analyse_scene.py <results_dir> --front      # front view only
    python analyse_scene.py <results_dir> --side       # side view only
    python analyse_scene.py <results_dir> --top        # top view only
    python analyse_scene.py <results_dir> --front --side --table

    # Custom paths:
    python analyse_scene.py --glb path/to/scene.glb --labels path/to/_labels.txt

Outputs are saved to <results_dir>/ (or --out-dir if specified).
"""

import argparse
import os
import sys

import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ── Data loading ──────────────────────────────────────────────────────────────

def load_labels(path):
    """Parse _labels.txt → {index: label_string}"""
    labels = {}
    if not os.path.exists(path):
        return labels
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("index"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                idx   = int(parts[0])
                label = parts[1]
                labels[idx] = label
    return labels


def load_scene_boxes(glb_path):
    """
    Load scene.glb and return per-object bounding boxes in world space.
    Returns {index: {name, min, max, centroid, extents}}
    """
    scene = trimesh.load(glb_path, force="scene")
    boxes = {}

    for name, geom in scene.geometry.items():
        # Apply the node transform if present
        try:
            node_names = [n for n in scene.graph.nodes if scene.graph[n][1] == name]
            if node_names:
                transform, _ = scene.graph[node_names[0]]
                mesh = geom.copy()
                mesh.apply_transform(transform)
            else:
                mesh = geom
        except Exception:
            mesh = geom

        if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
            continue

        verts    = np.array(mesh.vertices)
        min_pt   = verts.min(axis=0)
        max_pt   = verts.max(axis=0)
        centroid = (min_pt + max_pt) / 2
        extents  = max_pt - min_pt

        try:
            idx = int(name.split("_")[-1])
        except ValueError:
            idx = len(boxes)

        boxes[idx] = {
            "name":     name,
            "min":      min_pt,
            "max":      max_pt,
            "centroid": centroid,
            "extents":  extents,
        }

    return boxes


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _color_palette(n):
    cmap = plt.get_cmap("tab10")
    return {idx: cmap(i % 10) for i, idx in enumerate(sorted(range(n)))}


def _draw_box(ax, min_pt, max_pt, color, alpha=0.18):
    x0, y0, z0 = min_pt
    x1, y1, z1 = max_pt
    faces = [
        [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]],
        [[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]],
        [[x0,y0,z0],[x0,y0,z1],[x0,y1,z1],[x0,y1,z0]],
        [[x1,y0,z0],[x1,y0,z1],[x1,y1,z1],[x1,y1,z0]],
        [[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1]],
        [[x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1]],
    ]
    poly = Poly3DCollection(
        faces, alpha=alpha, facecolor=color, edgecolor=color, linewidth=0.5
    )
    ax.add_collection3d(poly)


def _setup_axes(ax, boxes, mid, max_r):
    ax.set_xlim(mid[0]-max_r, mid[0]+max_r)
    ax.set_ylim(mid[1]-max_r, mid[1]+max_r)
    ax.set_zlim(mid[2]-max_r, mid[2]+max_r)
    ax.set_xlabel("X  (left ↔ right)", labelpad=10)
    ax.set_ylabel("Y  (down ↕ up)",    labelpad=10)
    ax.set_zlabel("Z  (depth)",         labelpad=10)

    # Ground plane at Y = 0 (pipeline aligns every object's bottom to zero)
    ground_y = 0.0
    x0, x1   = mid[0]-max_r, mid[0]+max_r
    z0, z1   = mid[2]-max_r, mid[2]+max_r
    xx, zz   = np.meshgrid([x0, x1], [z0, z1])
    yy       = np.full_like(xx, ground_y)
    ax.plot_surface(xx, yy, zz, alpha=0.07, color="green")
    ax.text(x1, ground_y, z1, "  ground plane",
            fontsize=8, color="green", style="italic")


def _render_view(boxes, labels, colors, legend_patches,
                 mid, max_r, elev, azim, title, out_path):
    fig = plt.figure(figsize=(15, 9))
    ax  = fig.add_subplot(111, projection="3d")

    for idx in sorted(boxes):
        b     = boxes[idx]
        color = colors[idx]
        _draw_box(ax, b["min"], b["max"], color=color)
        cx, cy, cz = b["centroid"]
        ax.scatter(cx, cy, cz, color=color, s=40, zorder=5)
        ax.text(cx, cy, cz, f"  {idx}", fontsize=8,
                color=color, fontweight="bold")

    _setup_axes(ax, boxes, mid, max_r)
    ax.set_title(title, fontsize=13)
    ax.legend(handles=legend_patches, fontsize=7,
              bbox_to_anchor=(1.02, 1), loc="upper left", framealpha=0.8)
    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Table output ──────────────────────────────────────────────────────────────

def print_table(boxes, labels):
    print()
    print(f"{'Idx':<4} {'Label':<38} {'Centroid (x, y, z)':>32}  "
          f"{'W (X)':>7} {'H (Y)':>7} {'D (Z)':>7}  "
          f"{'Bot Y':>7} {'Top Y':>7}")
    print("-" * 115)
    for idx in sorted(boxes):
        b      = boxes[idx]
        label  = labels.get(idx, f"object_{idx}")
        if len(label) > 37:
            label = label[:34] + "…"
        cx, cy, cz = b["centroid"]
        ex, ey, ez = b["extents"]
        bot_y = b["min"][1]
        top_y = b["max"][1]
        print(
            f"{idx:<4} {label:<38} ({cx:7.3f}, {cy:7.3f}, {cz:7.3f})  "
            f"{ex:7.3f} {ey:7.3f} {ez:7.3f}  "
            f"{bot_y:7.3f} {top_y:7.3f}"
        )
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Analyse and visualise a SAM3D scene reconstruction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "results_dir", nargs="?", default=None,
        help="Job results directory (contains scene.glb and _labels.txt).",
    )
    p.add_argument("--glb",    default=None, help="Explicit path to scene.glb")
    p.add_argument("--labels", default=None, help="Explicit path to _labels.txt")
    p.add_argument("--out-dir", default=None,
                   help="Directory to write output images (default: results_dir).")

    views = p.add_argument_group("output selection (default: all)")
    views.add_argument("--table", action="store_true", help="Print table to stdout")
    views.add_argument("--front", action="store_true", help="Render front view")
    views.add_argument("--side",  action="store_true", help="Render side view (shows Y misalignment)")
    views.add_argument("--top",   action="store_true", help="Render top/bird's-eye view")
    views.add_argument("--all",   action="store_true", help="All outputs (default if none specified)")

    return p.parse_args()


def main():
    args = parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────────
    if args.results_dir:
        glb_path    = args.glb    or os.path.join(args.results_dir, "scene.glb")
        labels_path = args.labels or os.path.join(args.results_dir, "_labels.txt")
        out_dir     = args.out_dir or args.results_dir
    elif args.glb:
        glb_path    = args.glb
        labels_path = args.labels or os.path.join(os.path.dirname(args.glb), "_labels.txt")
        out_dir     = args.out_dir or os.path.dirname(args.glb)
    else:
        print("ERROR: provide a results_dir or --glb path.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(glb_path):
        print(f"ERROR: scene.glb not found at {glb_path}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    # Default: all outputs if nothing specified
    do_all   = args.all or not any([args.table, args.front, args.side, args.top])
    do_table = do_all or args.table
    do_front = do_all or args.front
    do_side  = do_all or args.side
    do_top   = do_all or args.top

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading {glb_path} …")
    boxes  = load_scene_boxes(glb_path)
    labels = load_labels(labels_path)

    if not boxes:
        print("ERROR: no geometry found in scene.glb", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(boxes)} objects.")

    # ── Shared geometry ───────────────────────────────────────────────────────
    colors         = _color_palette(max(boxes.keys()) + 1)
    legend_patches = [
        mpatches.Patch(
            color=colors[idx],
            label=f"{idx}: {labels.get(idx, f'object_{idx}')[:40]}",
        )
        for idx in sorted(boxes)
    ]

    all_pts = np.vstack([np.array([b["min"], b["max"]]) for b in boxes.values()])
    mid     = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
    max_r   = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2

    # ── Outputs ───────────────────────────────────────────────────────────────
    if do_table:
        print_table(boxes, labels)

    view_configs = [
        ("front", do_front,  5,  -90, "World-space object layout — front view  (X = left/right, Y = up/down)"),
        ("side",  do_side,   5,    0, "World-space object layout — side view   (Y = up/down, Z = depth)"),
        ("top",   do_top,   85,  -90, "World-space object layout — top view"),
    ]

    rendered = 0
    for name, enabled, elev, azim, title in view_configs:
        if not enabled:
            continue
        out_path = os.path.join(out_dir, f"scene_layout_{name}.png")
        _render_view(boxes, labels, colors, legend_patches,
                     mid, max_r, elev, azim, title, out_path)
        rendered += 1

    if rendered == 0 and not do_table:
        print("Nothing to output — use --table, --front, --side, --top, or --all.")


if __name__ == "__main__":
    main()
