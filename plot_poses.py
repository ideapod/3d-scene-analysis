#!/usr/bin/env python3
"""
Visualise per-object poses (translation origin + orientation axes) as a 3D plot.

Data sources (tried in order):
  1. glb/poses.json — saved by save_intermediates() on new pipeline runs
  2. Mesh-pair SVD  — recovers R, t, scale by comparing per-object GLBs
                      (local space) against the named geometries in scene.glb
                      (world space) using an Umeyama-style decomposition

Output
------
  <results_dir>/pose_plot.png   — static 3D figure

Usage
-----
  python plot_poses.py <results_dir>
  python plot_poses.py <results_dir> --show          # also open interactive window
  python plot_poses.py <results_dir> --axis-scale 0.2
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless by default; overridden to TkAgg/macosx if --show
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


# ── helpers ──────────────────────────────────────────────────────────────────

def _slug_to_label(slug: str) -> str:
    """'object_2_remfry_bird_seed_delivery_wagon' → '2 remfry bird seed delivery wagon'"""
    m = re.match(r"object_(\d+)_(.*)", slug)
    if m:
        return f"{m.group(1)} {m.group(2).replace('_', ' ')}"
    return slug.replace("_", " ")


def _recover_pose_svd(local_mesh, world_mesh):
    """
    Recover R (3×3), t (3,), scale (float) from a mesh pair.

    Uses Umeyama-style SVD:
      world ≈ scale * (R @ local.T).T + t
    Residual is effectively zero for a rigid+isotropic-scale transform.
    """
    c_loc = local_mesh.vertices.mean(axis=0)
    c_wld = world_mesh.vertices.mean(axis=0)
    X_loc = local_mesh.vertices - c_loc
    X_wld = world_mesh.vertices - c_wld

    scale = float(np.sqrt((X_wld ** 2).sum() / (X_loc ** 2).sum()))

    H = X_loc.T @ X_wld
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:         # fix reflection
        Vt[-1] *= -1
        R = Vt.T @ U.T

    t = c_wld - scale * (R @ c_loc)
    return R, t, scale


def _poses_from_json(poses_path: str):
    """Load poses from poses.json → list of dicts with keys: label, R, t, scale."""
    with open(poses_path) as f:
        data = json.load(f)
    poses = []
    for obj in data:
        quat = obj["rotation"]          # [w, x, y, z]
        trans = np.array(obj["translation"], dtype=float)
        scale = float(np.array(obj["scale"]).ravel()[0])
        R = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        poses.append(dict(label=obj["label"], R=R, t=trans, scale=scale))
    return poses


def _poses_from_meshes(results_dir: str, scene):
    """
    Recover poses by comparing per-object GLBs (local space) with the
    named geometries inside scene.glb (world space).
    """
    poses = []
    # Find per-object GLBs in the results dir (may be root or glb/ subdir)
    for candidate_dir in [results_dir, os.path.join(results_dir, "glb")]:
        glbs = sorted(Path(candidate_dir).glob("object_*.glb"))
        if glbs:
            break
    else:
        print("WARNING: no per-object GLBs found — cannot recover poses", file=sys.stderr)
        return poses

    for glb_path in glbs:
        slug = glb_path.stem  # e.g. "object_0_main_two_storey_victorian_building"
        if slug not in scene.geometry:
            print(f"  skipping {slug}: not found in scene.glb geometry", file=sys.stderr)
            continue

        local_mesh = trimesh.load(str(glb_path), force="mesh")
        world_mesh = scene.geometry[slug]

        if local_mesh.vertices.shape != world_mesh.vertices.shape:
            print(
                f"  skipping {slug}: vertex count mismatch "
                f"({local_mesh.vertices.shape[0]} vs {world_mesh.vertices.shape[0]})",
                file=sys.stderr,
            )
            continue

        R, t, scale = _recover_pose_svd(local_mesh, world_mesh)
        poses.append(dict(label=_slug_to_label(slug), R=R, t=t, scale=scale))

    return poses


# ── drawing ───────────────────────────────────────────────────────────────────

# Colour palette for objects (cycles if >10)
_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#aaffc3",
]

_AXIS_COLORS = ["#ff3333", "#33cc33", "#3399ff"]   # local X, Y, Z
_AXIS_LABELS = ["X", "Y", "Z"]


def _p(t):
    """
    Map model coordinates → matplotlib 3D axes so the plot reads like a scene:
      matplotlib X  ←  model X  (left/right in scene)
      matplotlib Y  ←  model Z  (depth into scene)   [matplotlib Y = into page]
      matplotlib Z  ←  model Y  (height)              [matplotlib Z = vertical]
    """
    return t[0], t[2], t[1]


def _v(v):
    """Same axis-swap for a direction vector."""
    return v[0], v[2], v[1]


def draw_poses(poses, out_path: str, axis_scale: float = 0.15, show: bool = False):
    if show:
        matplotlib.use("macosx" if sys.platform == "darwin" else "TkAgg")
        plt.switch_backend(matplotlib.get_backend())

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # ── World axes reference at origin ──
    # Draw world X (red), Z/depth (blue), Y/height (green) as reference arrows.
    # We use the same _p/_v mapping so they land on the right matplotlib axes.
    alen = axis_scale * 0.8
    world_axes = [
        (np.array([alen, 0, 0]),  _AXIS_COLORS[0], "W-X"),   # model X → matplotlib X
        (np.array([0, 0, alen]),  _AXIS_COLORS[1], "W-Y↑"),  # model Y (height) → matplotlib Z
        (np.array([0, alen, 0]),  _AXIS_COLORS[2], "W-Z→"),  # model Z (depth) → matplotlib Y
    ]
    for d_model, col, lbl in world_axes:
        px, py, pz = _v(d_model)
        ax.quiver(0, 0, 0, px, py, pz,
                  color=col, linewidth=2, arrow_length_ratio=0.25, alpha=0.5)
        ax.text(px * 1.3, py * 1.3, pz * 1.3, lbl, color=col, fontsize=7, alpha=0.6)

    ax.scatter([0], [0], [0], color="black", s=40, zorder=5)
    ax.text(0, 0, 0, " origin", fontsize=7, color="black")

    # ── Per-object poses ──
    all_t = np.array([p["t"] for p in poses])

    for i, pose in enumerate(poses):
        col = _PALETTE[i % len(_PALETTE)]
        t   = pose["t"]
        R   = pose["R"]
        sc  = pose["scale"]
        lbl = pose["label"]

        px, py, pz = _p(t)

        # Scale arrow length by object scale, clamped to a readable range
        arrow_len = np.clip(sc * axis_scale, axis_scale * 0.4, axis_scale * 2.0)

        # Local XYZ axes (columns of R) — apply same axis swap
        for j in range(3):
            axis_model = R[:, j] * arrow_len
            ax_x, ax_y, ax_z = _v(axis_model)
            ax.quiver(
                px, py, pz,
                ax_x, ax_y, ax_z,
                color=_AXIS_COLORS[j],
                linewidth=1.5,
                arrow_length_ratio=0.3,
            )

        # Origin sphere
        ax.scatter([px], [py], [pz], color=col, s=80, zorder=6, depthshade=True)

        # Label — offset upward (matplotlib Z = model Y = height)
        ax.text(px, py, pz + arrow_len * 0.3,
                f" {lbl}\n  x={t[0]:.2f} y={t[1]:.2f} z={t[2]:.2f}\n  s={sc:.3f}",
                fontsize=6.5, color=col, zorder=7)

    # ── Formatting ──────────────────────────────────────────────────────────
    if len(all_t):
        pad = axis_scale * 2
        # matplotlib X = model X
        ax.set_xlim(all_t[:, 0].min() - pad, all_t[:, 0].max() + pad)
        # matplotlib Y = model Z (depth) — put near depth at front (smaller Y)
        ax.set_ylim(all_t[:, 2].min() - pad, all_t[:, 2].max() + pad)
        # matplotlib Z = model Y (height)
        ax.set_zlim(0, all_t[:, 1].max() + pad)

    ax.set_xlabel("X  (left ← | → right)")
    ax.set_ylabel("Z  (depth into scene →)")
    ax.set_zlabel("Y  (height ↑)")
    ax.set_title("Object poses — translation origins + local XYZ axes\n"
                 "Red=local-X  Green=local-Y  Blue=local-Z      Faded=world axes\n"
                 "Plot axes: horizontal=scene-X, into-page=depth(Z), vertical=height(Y)")

    # Small legend for axis colours
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color=_AXIS_COLORS[0], label="Local X"),
        mpatches.Patch(color=_AXIS_COLORS[1], label="Local Y"),
        mpatches.Patch(color=_AXIS_COLORS[2], label="Local Z"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    if show:
        plt.show()

    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot per-object pose vectors from a scene assembly results dir."
    )
    parser.add_argument("results_dir", help="Job result directory")
    parser.add_argument(
        "--show", action="store_true",
        help="Open interactive 3D plot window after saving PNG",
    )
    parser.add_argument(
        "--axis-scale", type=float, default=0.15,
        help="Base length of orientation axis arrows (default 0.15)",
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    scene_glb   = os.path.join(results_dir, "scene.glb")
    poses_json  = os.path.join(results_dir, "glb", "poses.json")
    out_path    = os.path.join(results_dir, "pose_plot.png")

    if not os.path.exists(scene_glb):
        print(f"ERROR: scene.glb not found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    print("Loading scene.glb …")
    scene = trimesh.load(scene_glb, force="scene")

    if os.path.exists(poses_json):
        print(f"Loading poses from {poses_json} …")
        poses = _poses_from_json(poses_json)
    else:
        print("poses.json not found — recovering poses from mesh pairs …")
        poses = _poses_from_meshes(results_dir, scene)

    if not poses:
        print("ERROR: no poses found", file=sys.stderr)
        sys.exit(1)

    print(f"Plotting {len(poses)} objects …")
    for p in poses:
        euler = Rotation.from_matrix(p["R"]).as_euler("xyz", degrees=True)
        print(f"  {p['label']}")
        print(f"    t = {p['t'].round(4)}")
        print(f"    euler(xyz) = {euler.round(1)} deg")
        print(f"    scale = {p['scale']:.4f}")

    draw_poses(poses, out_path, axis_scale=args.axis_scale, show=args.show)


if __name__ == "__main__":
    main()
