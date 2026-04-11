#!/usr/bin/env python3
"""
analyse_scene.py — Visualise and tabulate the world-space layout of objects
reconstructed by the SAM3D API.

Usage:
    python analyse_scene.py <results_dir>              # all outputs
    python analyse_scene.py <results_dir> --table      # table only
    python analyse_scene.py <results_dir> --front      # 2D front elevation
    python analyse_scene.py <results_dir> --front-3d   # 3D front view
    python analyse_scene.py <results_dir> --side        # 2D side elevation
    python analyse_scene.py <results_dir> --side-3d    # 3D side view
    python analyse_scene.py <results_dir> --top         # 2D top/plan view
    python analyse_scene.py <results_dir> --top-3d     # 3D top view

    # Custom paths:
    python analyse_scene.py --glb path/to/scene.glb --labels path/to/_labels.txt

    # Composite GIF frame + layout side-by-side (auto-detected from results_dir):
    python analyse_scene.py <results_dir> --composite
    python analyse_scene.py <results_dir> --gif path/to/scene.gif --composite
    python analyse_scene.py <results_dir> --composite --front-frame 40 --side-frame 120

Outputs are saved to <results_dir>/ (or --out-dir if specified).
  2D views  →  scene_layout_{front|side|top}.png
  3D views  →  scene_layout_{front|side|top}_3d.png
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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


# ── GLB rendering helpers ─────────────────────────────────────────────────────

def render_glb_cardinal_frames(glb_path, out_dir, resolution=512):
    """
    Render scene.glb from 0°, 90°, 180°, 270° orbit positions using pyrender,
    matching the camera convention used for the GIF (yaw_start=-90°, pitch=0,
    fov=60°, Y-up, looks at scene centroid).

    Requires pyrender.  On headless Linux (EC2) pyrender uses the EGL backend
    automatically when PYOPENGL_PLATFORM=egl is set before import.

    Outputs: glb_frame_000deg.png … glb_frame_270deg.png
    """
    # Must be set before pyrender is imported on headless systems
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    try:
        import pyrender
        from PIL import Image as PILImage
    except ImportError:
        print("  WARNING: pyrender / Pillow not installed — skipping GLB renders.",
              file=sys.stderr)
        return

    # ── Load scene ────────────────────────────────────────────────────────────
    try:
        scene_tm = trimesh.load(glb_path, force="scene")
    except Exception as exc:
        print(f"  WARNING: could not load {glb_path}: {exc}", file=sys.stderr)
        return

    # ── Scene bounds → camera distance ────────────────────────────────────────
    try:
        bounds = scene_tm.bounds          # (2, 3)
        center = bounds.mean(axis=0)
        radius = float(np.linalg.norm(bounds[1] - bounds[0])) / 2
    except Exception:
        center = np.zeros(3)
        radius = 1.0
    cam_distance = max(radius * 2.5, 0.1)

    # ── Build pyrender scene ──────────────────────────────────────────────────
    try:
        pr_scene = pyrender.Scene.from_trimesh_scene(
            scene_tm, ambient_light=[0.3, 0.3, 0.3, 1.0]
        )
    except Exception as exc:
        print(f"  WARNING: could not build pyrender scene: {exc}", file=sys.stderr)
        return

    camera = pyrender.PerspectiveCamera(yfov=np.radians(60), aspectRatio=1.0)
    light  = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)

    try:
        renderer = pyrender.OffscreenRenderer(resolution, resolution)
    except Exception as exc:
        print(f"  WARNING: could not create offscreen renderer: {exc}", file=sys.stderr)
        return

    # ── Render each cardinal position ─────────────────────────────────────────
    # GIF orbit: yaw_start = -90°, so the Nth 90° step lands at:
    #   0°  → absolute yaw = -90° → camera at (-1,  0,  0)·r
    #   90° → absolute yaw =   0° → camera at ( 0,  0,  1)·r
    #  180° → absolute yaw =  90° → camera at ( 1,  0,  0)·r
    #  270° → absolute yaw = 180° → camera at ( 0,  0, -1)·r
    up_vec = np.array([0.0, 1.0, 0.0])

    for orbit_deg in [0, 90, 180, 270]:
        yaw = np.radians(orbit_deg - 90)          # absolute yaw (radians)
        cam_pos = center + cam_distance * np.array([
            np.sin(yaw), 0.0, np.cos(yaw)
        ])

        # Look-at → camera pose (OpenGL convention: camera looks along -Z)
        fwd = center - cam_pos
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up_vec)
        if np.linalg.norm(right) < 1e-6:          # degenerate: camera on Y axis
            right = np.array([1.0, 0.0, 0.0])
        else:
            right /= np.linalg.norm(right)
        cam_up = np.cross(right, fwd)

        cam_pose = np.eye(4)
        cam_pose[:3, 0] = right
        cam_pose[:3, 1] = cam_up
        cam_pose[:3, 2] = -fwd          # -Z = forward in OpenGL/pyrender
        cam_pose[:3, 3] = cam_pos

        cam_node   = pr_scene.add(camera, pose=cam_pose)
        light_node = pr_scene.add(light,  pose=cam_pose)

        try:
            color, _ = renderer.render(pr_scene)
            out_path  = os.path.join(out_dir, f"glb_frame_{orbit_deg:03d}deg.png")
            PILImage.fromarray(color).save(out_path)
            print(f"  Saved: {out_path}  (yaw={np.degrees(yaw):.0f}°)")
        except Exception as exc:
            print(f"  WARNING: render failed at {orbit_deg}°: {exc}", file=sys.stderr)

        pr_scene.remove_node(cam_node)
        pr_scene.remove_node(light_node)

    renderer.delete()


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


def save_gif_cardinal_frames(gif_path, out_dir):
    """
    Extract frames at 0°, 90°, 180°, 270° of the 360° orbit and save as PNGs.
    Frame index = round(degrees / 360 * total_frames).
    """
    try:
        import imageio
        from PIL import Image
    except ImportError:
        print("  WARNING: imageio/Pillow not installed — skipping GIF frames.",
              file=sys.stderr)
        return

    frames = imageio.mimread(gif_path)
    n = len(frames)
    for deg in [0, 90, 180, 270]:
        idx = round(deg / 360 * n) % n
        img = Image.fromarray(np.array(frames[idx]).astype(np.uint8))
        out_path = os.path.join(out_dir, f"gif_frame_{deg:03d}deg.png")
        img.save(out_path)
        print(f"  Saved: {out_path}  (frame {idx}/{n})")


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

    canvas.paste(layout, (0, bar_h))
    canvas.paste(gif_scaled.convert("RGB"), (lw, bar_h))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle([lw, 0, lw + sw, bar_h], fill=(30, 30, 30))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font = ImageFont.load_default()
    draw.text((lw + 8, 8), label, fill=(220, 220, 220), font=font)
    draw.line([(lw, 0), (lw, lh + bar_h)], fill=(180, 180, 180), width=2)

    canvas.save(out_path)
    print(f"  Saved: {out_path}")


# ── 2D projection renderer ────────────────────────────────────────────────────

def _color_palette(n):
    cmap = plt.get_cmap("tab10")
    return {idx: cmap(i % 10) for i, idx in enumerate(sorted(range(n)))}


# ── Oblique projection renderer ───────────────────────────────────────────────

def _oblique(x, y, z, angle_deg=30, scale=0.35):
    """Cabinet oblique: X→right, Y→up, Z→diagonal at angle_deg."""
    import math
    a = math.radians(angle_deg)
    return x + z * scale * math.cos(a), y + z * scale * math.sin(a)


def _render_oblique_view(boxes, labels, colors, legend_patches, title, out_path,
                          angle_deg=30, depth_scale=0.35):
    """
    Oblique (cabinet) projection of all bounding boxes.
      X  = left / right  (horizontal)
      Y  = up / down     (vertical)
      Z  = depth, shown as a diagonal offset at angle_deg

    Objects further away (larger Z centroid) are shifted up-right, so gaps
    between the front face of one object and the back face of another are
    visible as real space — unlike the flat 2D projection which collapses Z.
    """
    from matplotlib.patches import Polygon as MplPolygon

    def p(x, y, z):
        return _oblique(x, y, z, angle_deg, depth_scale)

    fig, ax = plt.subplots(figsize=(15, 9))

    # Draw back-to-front so nearer objects paint over farther ones
    sorted_idxs = sorted(boxes.keys(),
                         key=lambda i: -boxes[i]["centroid"][2])

    for idx in sorted_idxs:
        b   = boxes[idx]
        col = colors[idx]
        x0, y0, z0 = b["min"]
        x1, y1, z1 = b["max"]

        # Three visible faces: front (z0), top (y1), right side (x1)
        faces = [
            # (corners,              face alpha)
            ([p(x0,y0,z0), p(x1,y0,z0), p(x1,y1,z0), p(x0,y1,z0)], 0.30),  # front
            ([p(x0,y1,z0), p(x1,y1,z0), p(x1,y1,z1), p(x0,y1,z1)], 0.18),  # top
            ([p(x1,y0,z0), p(x1,y1,z0), p(x1,y1,z1), p(x1,y0,z1)], 0.22),  # right
        ]
        for corners, alpha in faces:
            patch = MplPolygon(corners, closed=True,
                               facecolor=col, edgecolor=col,
                               alpha=alpha, linewidth=1.2)
            ax.add_patch(patch)

        # Draw back edges as thin dashed lines for context
        back_edges = [
            [p(x0,y0,z1), p(x1,y0,z1)],
            [p(x0,y1,z1), p(x1,y1,z1)],
            [p(x0,y0,z0), p(x0,y0,z1)],
            [p(x0,y1,z0), p(x0,y1,z1)],
            [p(x1,y0,z1), p(x1,y1,z1)],
        ]
        for e in back_edges:
            ax.plot([e[0][0], e[1][0]], [e[0][1], e[1][1]],
                    color=col, alpha=0.25, linewidth=0.7, linestyle="--")

        # Label at projected centroid
        cx, cy, cz = b["centroid"]
        sx, sy = p(cx, cy, cz)
        ax.text(sx, sy, f"{idx}", ha="center", va="center",
                fontsize=9, color=col, fontweight="bold")

    # Ground plane at Y = 0
    all_x = [v for b in boxes.values() for v in [b["min"][0], b["max"][0]]]
    all_z = [v for b in boxes.values() for v in [b["min"][2], b["max"][2]]]
    gx0, gx1 = min(all_x), max(all_x)
    gz0, gz1 = min(all_z), max(all_z)
    ground_corners = [p(gx0, 0, gz0), p(gx1, 0, gz0),
                      p(gx1, 0, gz1), p(gx0, 0, gz1)]
    ground = MplPolygon(ground_corners, closed=True,
                        facecolor="green", edgecolor="green",
                        alpha=0.08, linewidth=1, linestyle="--")
    ax.add_patch(ground)
    ax.text(*p(gx0, 0, gz0), "  ground (Y=0)", color="green",
            fontsize=8, va="top", alpha=0.8)

    ax.autoscale()
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"X  (left ↔ right)  +  Z×{depth_scale:.2f}·cos({angle_deg}°)",
                  fontsize=10)
    ax.set_ylabel(f"Y  (up/down)  +  Z×{depth_scale:.2f}·sin({angle_deg}°)",
                  fontsize=10)
    ax.set_title(title, fontsize=13)
    ax.legend(handles=legend_patches, fontsize=7,
              bbox_to_anchor=(1.02, 1), loc="upper left", framealpha=0.8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


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
        h0, h1  = b["min"][h_axis], b["max"][h_axis]
        v0, v1  = b["min"][v_axis], b["max"][v_axis]
        depth_c = b["centroid"][depth_axis]

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


# ── 3D view renderer ──────────────────────────────────────────────────────────

def _draw_box_3d(ax, min_pt, max_pt, color, alpha=0.18):
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


def _render_3d_view(boxes, labels, colors, legend_patches,
                    mid, max_r, elev, azim, title, out_path):
    fig = plt.figure(figsize=(15, 9))
    ax  = fig.add_subplot(111, projection="3d")

    for idx in sorted(boxes):
        b     = boxes[idx]
        color = colors[idx]
        _draw_box_3d(ax, b["min"], b["max"], color=color)
        cx, cy, cz = b["centroid"]
        ax.scatter(cx, cy, cz, color=color, s=40, zorder=5)
        ax.text(cx, cy, cz, f"  {idx}", fontsize=8,
                color=color, fontweight="bold")

    ax.set_xlim(mid[0]-max_r, mid[0]+max_r)
    ax.set_ylim(mid[1]-max_r, mid[1]+max_r)
    ax.set_zlim(mid[2]-max_r, mid[2]+max_r)
    ax.set_xlabel("X  (left ↔ right)", labelpad=10)
    ax.set_ylabel("Y  (down ↕ up)",    labelpad=10)
    ax.set_zlabel("Z  (depth)",         labelpad=10)

    # Ground plane at Y = 0
    x0, x1 = mid[0]-max_r, mid[0]+max_r
    z0, z1 = mid[2]-max_r, mid[2]+max_r
    xx, zz = np.meshgrid([x0, x1], [z0, z1])
    yy     = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.07, color="green")
    ax.text(x1, 0, z1, "  ground plane", fontsize=8,
            color="green", style="italic")

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
    p.add_argument("--gif",    default=None, help="Explicit path to reconstruction GIF")
    p.add_argument("--out-dir", default=None,
                   help="Directory to write output images (default: results_dir).")

    views = p.add_argument_group("output selection (default: all)")
    views.add_argument("--table",     action="store_true", help="Print table to stdout")
    views.add_argument("--front",     action="store_true", help="2D front elevation")
    views.add_argument("--side",      action="store_true", help="2D side elevation")
    views.add_argument("--top",       action="store_true", help="2D top / plan view")
    views.add_argument("--front-3d",  action="store_true", help="3D front view (scene_layout_front_3d.png)")
    views.add_argument("--side-3d",   action="store_true", help="3D side view  (scene_layout_side_3d.png)")
    views.add_argument("--top-3d",    action="store_true", help="3D top view   (scene_layout_top_3d.png)")
    views.add_argument("--oblique",   action="store_true",
                       help="Oblique (cabinet) projection — shows X, Y and Z depth as diagonal offset "
                            "(scene_layout_oblique.png)")
    views.add_argument("--composite", action="store_true",
                       help="Save layout+GIF composite images for front and side views")
    views.add_argument("--gif-frames", action="store_true",
                       help="Save individual frames at 0°, 90°, 180°, 270° of the GIF orbit "
                            "(gif_frame_000deg.png … gif_frame_270deg.png)")
    views.add_argument("--glb-frames", action="store_true",
                       help="Render scene.glb from 0°, 90°, 180°, 270° using pyrender "
                            "(glb_frame_000deg.png … glb_frame_270deg.png)")
    views.add_argument("--all",       action="store_true",
                       help="All outputs — 2D + 3D + table + composite + gif-frames + glb-frames "
                            "(default if none specified)")

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

    explicit = [args.table, args.front, args.side, args.top,
                args.front_3d, args.side_3d, args.top_3d,
                args.oblique, args.composite, args.gif_frames, args.glb_frames]
    do_all        = args.all or not any(explicit)
    do_table      = do_all or args.table
    do_front      = do_all or args.front
    do_side       = do_all or args.side
    do_top        = do_all or args.top
    do_front_3d   = do_all or args.front_3d
    do_side_3d    = do_all or args.side_3d
    do_top_3d     = do_all or args.top_3d
    do_oblique    = do_all or args.oblique
    do_composite  = do_all or args.composite
    do_gif_frames = do_all or args.gif_frames
    do_glb_frames = do_all or args.glb_frames

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

    # Bounding sphere centre/radius — used by 3D views
    all_pts = np.vstack([np.array([b["min"], b["max"]]) for b in boxes.values()])
    mid     = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
    max_r   = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2

    # ── 2D projection outputs ─────────────────────────────────────────────────
    if do_table:
        print_table(boxes, labels)

    # (name, enabled, h_axis, v_axis, h_label, v_label, depth_label, inv_h, inv_v, title)
    view_2d_configs = [
        ("front", do_front,
         0, 1, "X  (left ↔ right)", "Y  (up ↓ down)", "Z",
         False, False,
         "Front elevation  —  looking along Z  (annotated: Z = depth)"),
        ("side", do_side,
         2, 1, "Z  (depth →)", "Y  (up ↓ down)", "X",
         False, False,
         "Side elevation  —  looking along X  (annotated: X = left/right)"),
        ("top", do_top,
         0, 2, "X  (left ↔ right)", "Z  (depth →)", "Y",
         False, False,
         "Top / plan view  —  looking down Y  (annotated: Y = height)"),
    ]

    rendered = 0
    for name, enabled, h_ax, v_ax, h_lbl, v_lbl, d_lbl, inv_h, inv_v, title \
            in view_2d_configs:
        if not enabled:
            continue
        out_path = os.path.join(out_dir, f"scene_layout_{name}.png")
        _render_2d_projection(boxes, labels, colors, legend_patches,
                              h_ax, v_ax, h_lbl, v_lbl, d_lbl,
                              title, out_path, inv_h, inv_v)
        rendered += 1

    # ── 3D view outputs ───────────────────────────────────────────────────────
    # (name, enabled, elev, azim, title)
    view_3d_configs = [
        ("front", do_front_3d,  5, -90,
         "World-space layout — front 3D view  (X = left/right, Y = up/down)"),
        ("side",  do_side_3d,   5,   0,
         "World-space layout — side 3D view   (Y = up/down, Z = depth)"),
        ("top",   do_top_3d,   85, -90,
         "World-space layout — top 3D view"),
    ]

    for name, enabled, elev, azim, title in view_3d_configs:
        if not enabled:
            continue
        out_path = os.path.join(out_dir, f"scene_layout_{name}_3d.png")
        _render_3d_view(boxes, labels, colors, legend_patches,
                        mid, max_r, elev, azim, title, out_path)
        rendered += 1

    # ── Oblique projection ────────────────────────────────────────────────────
    if do_oblique:
        out_path = os.path.join(out_dir, "scene_layout_oblique.png")
        _render_oblique_view(
            boxes, labels, colors, legend_patches,
            "Oblique (cabinet) projection  —  Z depth shown as diagonal offset",
            out_path,
        )
        rendered += 1

    # ── GIF composites ────────────────────────────────────────────────────────
    if do_composite:
        if not gif_path or not os.path.exists(gif_path):
            print("  WARNING: no GIF found — skipping composites.", file=sys.stderr)
        else:
            print(f"  GIF: {gif_path}")
            composite_configs = [
                ("front", args.front_frame, f"GIF frame {args.front_frame}  (front view)"),
                ("side",  args.side_frame,  f"GIF frame {args.side_frame}  (side view)"),
            ]
            for name, frame_idx, label in composite_configs:
                layout_path = os.path.join(out_dir, f"scene_layout_{name}.png")
                if not os.path.exists(layout_path):
                    # 2D layout wasn't rendered yet — render it now
                    cfg = {c[0]: c for c in view_2d_configs}[name]
                    _, _, h_ax, v_ax, h_lbl, v_lbl, d_lbl, inv_h, inv_v, ttl = cfg
                    _render_2d_projection(boxes, labels, colors, legend_patches,
                                          h_ax, v_ax, h_lbl, v_lbl, d_lbl,
                                          ttl, layout_path, inv_h, inv_v)

                gif_frame = extract_gif_frame(gif_path, frame_idx)
                if gif_frame is None:
                    continue
                out_path = os.path.join(out_dir, f"scene_composite_{name}.png")
                make_composite(layout_path, gif_frame, label, out_path)

    # ── GLB cardinal renders ──────────────────────────────────────────────────
    if do_glb_frames:
        print(f"  Rendering GLB cardinal frames from {glb_path} …")
        render_glb_cardinal_frames(glb_path, out_dir)

    # ── GIF cardinal frames ───────────────────────────────────────────────────
    if do_gif_frames:
        if not gif_path or not os.path.exists(gif_path):
            print("  WARNING: no GIF found — skipping cardinal frames.", file=sys.stderr)
        else:
            print(f"  Extracting cardinal GIF frames from {gif_path} …")
            save_gif_cardinal_frames(gif_path, out_dir)

    if rendered == 0 and not do_table and not do_composite and not do_gif_frames \
            and not do_glb_frames:
        print("Nothing to output — use --front/--side/--top, --front-3d/--side-3d/--top-3d, "
              "--table, --composite, --gif-frames, --glb-frames, or --all.")


if __name__ == "__main__":
    main()
