#!/usr/bin/env python3
"""Finish Phase 0 raw CALVIN inspection outputs.

This script reads the configured raw CALVIN task_ABC_D split, uses
training/scene_info.npy to split A/B/C, treats validation as D, writes
LaTeX-ready CSV tables, and creates a compact visual sample grid.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from calvin_phase0_common import (
    clipped_episode_ranges,
    configured_raw_root,
    count_language_tasks,
    frame_file,
    load_environment_ranges,
    load_episode_ranges,
    load_language_intervals,
    split_dir,
)


def evenly_spaced_ids(ranges: list[tuple[int, int]], count: int) -> list[int]:
    if not ranges:
        return []
    total = sum(end - start + 1 for start, end in ranges)
    if total <= count:
        ids = []
        for start, end in ranges:
            ids.extend(range(start, end + 1))
        return ids[:count]
    targets = np.linspace(0, total - 1, count, dtype=int)
    ids = []
    cursor_base = 0
    range_idx = 0
    start, end = ranges[range_idx]
    for target in targets:
        while target >= cursor_base + (end - start + 1):
            cursor_base += end - start + 1
            range_idx += 1
            start, end = ranges[range_idx]
        ids.append(start + int(target - cursor_base))
    return ids


def write_stats(raw_root: Path, output_dir: Path) -> None:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    env_ranges = load_environment_ranges(raw_root)
    rows = []
    task_rows = []
    for env in ["A", "B", "C", "D"]:
        frame_range = env_ranges[env]
        directory = split_dir(raw_root, frame_range.split)
        episode_ranges = clipped_episode_ranges(load_episode_ranges(directory), frame_range)
        intervals = load_language_intervals(directory)
        task_counts = count_language_tasks(intervals, frame_range)
        rows.append(
            {
                "environment": env,
                "split": frame_range.split,
                "scene": frame_range.scene,
                "path": str(directory),
                "frame_start": frame_range.start,
                "frame_end": frame_range.end,
                "num_raw_episodes_or_segments": len(episode_ranges),
                "num_frames": sum(end - start + 1 for start, end in episode_ranges),
                "num_language_sequences": sum(task_counts.values()),
                "num_language_tasks": len(task_counts),
                "status": "raw_present",
                "notes": "training scene_info split" if env != "D" else "validation split treated as D",
            }
        )
        for task, count in sorted(task_counts.items()):
            task_rows.append({"environment": env, "task": task, "language_sequences": count})

    with (table_dir / "dataset_stats_ABC.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (table_dir / "task_counts_ABCD.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["environment", "task", "language_sequences"])
        writer.writeheader()
        writer.writerows(task_rows)


def add_label(img: Image.Image, label: str) -> Image.Image:
    label_h = 22
    canvas = Image.new("RGB", (img.width, img.height + label_h), "white")
    canvas.paste(img, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), label, fill=(0, 0, 0))
    return canvas


def make_samples(raw_root: Path, output_dir: Path, samples_per_env: int) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    env_ranges = load_environment_ranges(raw_root)
    rows = []
    cell_w, cell_h = 200, 222
    for env in ["A", "B", "C", "D"]:
        frame_range = env_ranges[env]
        directory = split_dir(raw_root, frame_range.split)
        episode_ranges = clipped_episode_ranges(load_episode_ranges(directory), frame_range)
        ids = evenly_spaced_ids(episode_ranges, samples_per_env)
        cells = []
        for frame_id in ids:
            data = np.load(frame_file(directory, frame_id))
            img = Image.fromarray(data["rgb_static"]).resize((cell_w, 200))
            cells.append(add_label(img, f"{env} frame {frame_id}"))
        while len(cells) < samples_per_env:
            cells.append(add_label(Image.new("RGB", (cell_w, 200), "white"), f"{env} missing"))
        rows.append(cells)

    grid = Image.new("RGB", (samples_per_env * cell_w, 4 * cell_h), "white")
    for row_idx, cells in enumerate(rows):
        for col_idx, cell in enumerate(cells):
            grid.paste(cell, (col_idx * cell_w, row_idx * cell_h))
    grid.save(figures_dir / "env_samples_ABCD.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("project"))
    parser.add_argument("--samples-per-env", type=int, default=4)
    args = parser.parse_args()

    raw_root = configured_raw_root(args.raw_root)
    write_stats(raw_root, args.output_dir)
    make_samples(raw_root, args.output_dir, args.samples_per_env)
    print(f"Wrote Phase 0 stats and samples under {args.output_dir}")


if __name__ == "__main__":
    main()
