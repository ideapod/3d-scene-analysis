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

    # Composite GIF frame + layout side-by-side (auto-detected from results_dir):
    python analyse_scene.py <results_dir> --composite
    python analyse_scene.py <results_dir> --gif path/to/scene.gif --composite
    python analyse_scene.py <results_dir> --composite --front-frame 40 --side-frame 120

Outputs are saved to <results_dir>/ (or --out-dir if specified).
"""

import argparse
import glob
import os
import sys

import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Default GIF frame indices that align with front and side layout views.
# The SAM3D render_video() spins 300 frames from yaw=-90° → +270°.
#   frame 50  ≈ yaw -30°  → building facade head-on   (front)
#   frame 125 ≈ yaw  60°  → building in right profile  (side)
_DEFAULT_FRONT_FRAME = 50
_DEFAULT_SIDE_FRAME  = 125


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
            # Geometry name format: "object_{idx}" or "object_{idx}_{label_slug}"
            idx = int(name.split("_")[1])
        except (ValueError, IndexError):
            idx = len(boxes)

        boxes[idx] = {
            "name":     name,
            "min":      min_pt,
            "max":      max_pt,
            "centroid": centroid,
            "extents":  extents,
        }

    return boxes


# ── GIF helpers ───────────────────────────────────────────────────────────────

def find_gif(results_dir):
    """Return the first .gif found in results_dir, or None."""
    hits = sorted(glob.glob(os.path.join(results_dir, "*.gif")))
    return hits[0] if hits else None


def extract_gif_frame(gif_path, frame_idx):
    """Load a GIF and return a single frame as a PIL Image."""
    try:
        import imageio
        from PIL import Image
    except ImportError:
        print("  WARNING: imageio / Pillow not installed — skipping GIF composite.",
              file=sys.stderr)
        return None

    frames = imageio.mimread(gif_path)
    frame_idx = frame_idx % len(frames)
    return Image.fromarray(np.array(frames[frame_idx]).astype(np.uint8))


def make_composite(layout_path, gif_frame, label, out_path):
    """
    Place the layout PNG and the GIF frame side by side and save.
    A small label strip is added above the GIF frame identifying the frame.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("  WARNING: Pillow not installed — skipping composite.", file=sys.stderr)
        return

    layout = Image.open(layout_path).convert("RGB")
    lw, lh = layout.size

    # Scale GIF frame to match layout height
    gw, gh = gif_frame.size
    scale      = lh / gh
    gif_scaled = gif_frame.resize((int(gw * scale), lh), Image.LANCZOS)
    sw         = gif_scaled.width

    # Add a label bar above the GIF frame
    bar_h  = 36
    canvas = Image.new("RGB", (lw + sw, lh + bar_h), (245, 245, 245))

    # Layout plot (top-left, shifted down by bar)
    canvas.paste(layout, (0, bar_h))

    # GIF frame (top-right, shifted down by bar)
    canvas.paste(gif_scaled.convert("RGB"), (lw, bar_h))

    # Label bar
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([lw, 0, lw + sw, bar_h], fill=(30, 30, 30))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font = ImageFont.load_default()
    draw.text((lw + 8, 8), label, fill=(220, 220, 220), font=font)

    # Dividing line between layout and GIF
    draw.line([(lw, 0), (lw, lh + bar_h)], fill=(180, 180, 180), width=2)

    canvas.save(out_path)
    print(f"  Saved: {out_path}")


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _color_palette(n):
    cmap = plt.get_cmap("tab10")
    return {idx: cmap(i % 10) for i, idx in enumerate(sorted(range(n)))}


def _render_2d_projection(boxes, labels, colors, legend_patches,
                           h_axis, v_axis,
                           h_label, v_label, depth_label,
                           title, out_path,
                           invert_h=False, invert_v=False):
    """
    True 2D projection: each object drawn as a flat rectangle in the
    (h_axis, v_axis) plane.  The third axis is 'depth' — objects are
    sorted so nearer ones (smaller depth centroid) are drawn on top, and
    each box is annotated with its depth centroid value.
    """
    depth_axis = ({0, 1, 2} - {h_axis, v_axis}).pop()

    fig, ax = plt.subplots(figsize=(14, 8))

    # Draw farther objects first so nearer ones appear on top
    sorted_idxs = sorted(boxes.keys(),
                         key=lambda i: -boxes[i]["centroid"][depth_axis])

    for idx in sorted_idxs:
        b     = boxes[idx]
        color = colors[idx]
        h0, h1    = b["min"][h_axis], b["max"][h_axis]
        v0, v1    = b["min"][v_axis], b["max"][v_axis]
        depth_c   = b["centroid"][depth_axis]

        rect = plt.Rectangle(
            (h0, v0), h1 - h0, v1 - v0,
            linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.25,
        )
        ax.add_patch(rect)

        ch, cv = (h0 + h1) / 2, (v0 + v1) / 2
        ax.text(ch, cv, f"{idx}", ha="center", va="center",
                fontsize=9, color=color, fontweight="bold")
        ax.text(ch, v1, f" {depth_label}={depth_c:.2f}",
                ha="center", va="bottom", fontsize=7, color=color, alpha=0.8)

    # Ground line at Y = 0 when the vertical axis is Y
    if v_axis == 1:
        all_h = [b["min"][h_axis] for b in boxes.values()] + \
                [b["max"][h_axis] for b in boxes.values()]
        ax.axhline(0, color="green", linewidth=1, linestyle="--", alpha=0.6)
        ax.text(min(all_h), 0, "  ground (Y=0)", color="green",
                fontsize=8, va="bottom")

    ax.autoscale()
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(h_label, fontsize=11)
    ax.set_ylabel(v_label, fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(handles=legend_patches, fontsize=7,
              bbox_to_anchor=(1.02, 1), loc="upper left", framealpha=0.8)
    ax.grid(True, alpha=0.3)

    if invert_h:
        ax.invert_xaxis()
    if invert_v:
        ax.invert_yaxis()

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
    p.add_argument("--gif",    default=None, help="Explicit path to reconstruction GIF")
    p.add_argument("--out-dir", default=None,
                   help="Directory to write output images (default: results_dir).")

    views = p.add_argument_group("output selection (default: all)")
    views.add_argument("--table",     action="store_true", help="Print table to stdout")
    views.add_argument("--front",     action="store_true", help="Render front view")
    views.add_argument("--side",      action="store_true", help="Render side view")
    views.add_argument("--top",       action="store_true", help="Render top/bird's-eye view")
    views.add_argument("--composite", action="store_true",
                       help="Save layout+GIF composite images for front and side views")
    views.add_argument("--all",       action="store_true", help="All outputs (default if none specified)")

    gif_grp = p.add_argument_group("GIF frame selection")
    gif_grp.add_argument("--front-frame", type=int, default=_DEFAULT_FRONT_FRAME,
                         help=f"GIF frame index for front composite (default: {_DEFAULT_FRONT_FRAME})")
    gif_grp.add_argument("--side-frame",  type=int, default=_DEFAULT_SIDE_FRAME,
                         help=f"GIF frame index for side composite (default: {_DEFAULT_SIDE_FRAME})")

    return p.parse_args()


def main():
    args = parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────────
    if args.results_dir:
        glb_path    = args.glb    or os.path.join(args.results_dir, "scene.glb")
        labels_path = args.labels or os.path.join(args.results_dir, "_labels.txt")
        gif_path    = args.gif    or find_gif(args.results_dir)
        out_dir     = args.out_dir or args.results_dir
    elif args.glb:
        glb_path    = args.glb
        labels_path = args.labels or os.path.join(os.path.dirname(args.glb), "_labels.txt")
        gif_path    = args.gif
        out_dir     = args.out_dir or os.path.dirname(args.glb)
    else:
        print("ERROR: provide a results_dir or --glb path.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(glb_path):
        print(f"ERROR: scene.glb not found at {glb_path}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    # Default: all outputs if nothing specified
    do_all       = args.all or not any([args.table, args.front, args.side,
                                        args.top, args.composite])
    do_table     = do_all or args.table
    do_front     = do_all or args.front
    do_side      = do_all or args.side
    do_top       = do_all or args.top
    do_composite = do_all or args.composite

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

    # (mid/max_r not needed for 2D projections)

    # ── Outputs ───────────────────────────────────────────────────────────────
    if do_table:
        print_table(boxes, labels)

    # h_axis, v_axis, h_label, v_label, depth_label, invert_h, invert_v
    view_configs = [
        ("front", do_front,
         0, 1, "X  (left ↔ right)", "Y  (up ↓ down)", "Z",
         False, False,
         "Front elevation  —  looking along Z  (annotated: Z = depth)"),
        ("side",  do_side,
         2, 1, "Z  (depth →)", "Y  (up ↓ down)", "X",
         False, False,
         "Side elevation  —  looking along X  (annotated: X = left/right)"),
        ("top",   do_top,
         0, 2, "X  (left ↔ right)", "Z  (depth →)", "Y",
         False, False,
         "Top / plan view  —  looking down Y  (annotated: Y = height)"),
    ]

    rendered = 0
    for name, enabled, h_ax, v_ax, h_lbl, v_lbl, d_lbl, inv_h, inv_v, title \
            in view_configs:
        if not enabled:
            continue
        out_path = os.path.join(out_dir, f"scene_layout_{name}.png")
        _render_2d_projection(boxes, labels, colors, legend_patches,
                              h_ax, v_ax, h_lbl, v_lbl, d_lbl,
                              title, out_path, inv_h, inv_v)
        rendered += 1

    # ── GIF composites ────────────────────────────────────────────────────────
    if do_composite:
        if not gif_path or not os.path.exists(gif_path):
            print("  WARNING: no GIF found — skipping composites.", file=sys.stderr)
        else:
            print(f"  GIF: {gif_path}")
            composite_configs = [
                ("front", do_front, args.front_frame,
                 f"GIF frame {args.front_frame}  (front view)"),
                ("side",  do_side,  args.side_frame,
                 f"GIF frame {args.side_frame}  (side view)"),
            ]
            for name, layout_enabled, frame_idx, label in composite_configs:
                layout_path = os.path.join(out_dir, f"scene_layout_{name}.png")
                if not os.path.exists(layout_path):
                    # Layout wasn't rendered yet — render it now
                    cfg = {c[0]: c for c in view_configs}[name]
                    _, _, elev, azim, title = cfg
                    _render_view(boxes, labels, colors, legend_patches,
                                 mid, max_r, elev, azim, title, layout_path)

                gif_frame = extract_gif_frame(gif_path, frame_idx)
                if gif_frame is None:
                    continue
                out_path = os.path.join(out_dir, f"scene_composite_{name}.png")
                make_composite(layout_path, gif_frame, label, out_path)

    if rendered == 0 and not do_table and not do_composite:
        print("Nothing to output — use --table, --front, --side, --top, --composite, or --all.")


if __name__ == "__main__":
    main()
