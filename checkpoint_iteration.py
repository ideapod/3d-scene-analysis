#!/usr/bin/env python3
"""
Checkpoint the current scene assembly + analysis outputs into a versioned
subdirectory so previous iterations are preserved before the next run.

Usage
-----
  # Snapshot current outputs (auto-increments iteration number):
  python checkpoint_iteration.py <results_dir>

  # Preview what would be copied without doing it:
  python checkpoint_iteration.py <results_dir> --dry-run

What gets copied
----------------
All recognised output files in <results_dir>:
  *.glb, *.ply, *.gif, *.png, status.json, *.txt

What is intentionally skipped
------------------------------
  glb/          — raw intermediates (large, reusable, never change per-object)
  iteration_*/  — previous snapshots

Output
------
  <results_dir>/
    iteration_1/   ← first snapshot
    iteration_2/   ← second snapshot
    ...
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# File extensions that are worth keeping per iteration
_KEEP_EXTS = {".glb", ".ply", ".gif", ".png", ".json", ".txt"}

# Subdirectory names that should never be snapshotted
_SKIP_DIRS = {"glb"}


def _next_iteration_dir(results_dir: str) -> str:
    """Return the next iteration_N path (does not create it)."""
    n = 1
    while True:
        candidate = os.path.join(results_dir, f"iteration_{n}")
        if not os.path.exists(candidate):
            return candidate
        n += 1


def _collect_files(results_dir: str) -> list[tuple[str, str]]:
    """
    Return list of (src_abs, dest_rel) pairs for files to snapshot.
    dest_rel is relative to the iteration dir root.
    """
    pairs = []
    results_path = Path(results_dir)

    for entry in sorted(results_path.iterdir()):
        # Skip iteration dirs and the glb intermediates dir
        if entry.is_dir():
            if entry.name in _SKIP_DIRS or entry.name.startswith("iteration_"):
                continue
            # Include other subdirs recursively (unlikely but safe)
            for sub in sorted(entry.rglob("*")):
                if sub.is_file() and sub.suffix.lower() in _KEEP_EXTS:
                    pairs.append((str(sub), str(sub.relative_to(results_path))))
        elif entry.is_file() and entry.suffix.lower() in _KEEP_EXTS:
            pairs.append((str(entry), entry.name))

    return pairs


def checkpoint(results_dir: str, dry_run: bool = False) -> str:
    results_dir = os.path.abspath(results_dir)

    if not os.path.isdir(results_dir):
        print(f"ERROR: {results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    files = _collect_files(results_dir)
    if not files:
        print("Nothing to checkpoint — no recognised output files found.")
        return ""

    iteration_dir = _next_iteration_dir(results_dir)
    n = Path(iteration_dir).name  # e.g. "iteration_3"

    print(f"Checkpointing {len(files)} file(s) → {iteration_dir}")
    if dry_run:
        for src, rel in files:
            size_kb = os.path.getsize(src) / 1024
            print(f"  [{size_kb:7.1f} KB]  {rel}")
        print("(dry run — nothing copied)")
        return iteration_dir

    os.makedirs(iteration_dir, exist_ok=True)
    for src, rel in files:
        dest = os.path.join(iteration_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(src, dest)
        print(f"  copied  {rel}")

    print(f"\nSaved as {n}  →  {iteration_dir}")
    return iteration_dir


def main():
    parser = argparse.ArgumentParser(
        description="Snapshot scene assembly + analysis outputs into iteration_N/ subdir."
    )
    parser.add_argument("results_dir", help="Job result directory to checkpoint")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be copied without copying them",
    )
    args = parser.parse_args()

    checkpoint(args.results_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
