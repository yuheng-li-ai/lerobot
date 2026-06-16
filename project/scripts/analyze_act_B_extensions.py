#!/usr/bin/env python3
"""Task 1 ACT-B non-training extensions.

Produces loss/overfitting analysis, environment-B visual statistics/samples,
action smoothness/chunk diagnostics, and presentation figures from existing
full-training metrics and raw CALVIN B data. This script does not train.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    total = sum(end - start + 1 for start, end in ranges)
    if total <= 0:
        return []
    targets = np.linspace(0, total - 1, min(count, total), dtype=int)
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


def load_metrics(metrics_path: Path) -> list[dict[str, float]]:
    rows = []
    with metrics_path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({key: float(value) for key, value in row.items()})
    return rows


def finite_metric_points(rows: list[dict[str, float]], key: str) -> tuple[np.ndarray, np.ndarray]:
    points = [(int(row["step"]), row[key]) for row in rows if math.isfinite(row[key])]
    return np.array([p[0] for p in points]), np.array([p[1] for p in points])


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    window = max(1, int(window))
    cumsum = np.cumsum(np.insert(values.astype(np.float64), 0, 0.0))
    out = np.empty_like(values, dtype=np.float64)
    for idx in range(len(values)):
        start = max(0, idx + 1 - window)
        out[idx] = (cumsum[idx + 1] - cumsum[start]) / (idx + 1 - start)
    return out


def write_loss_summary(rows: list[dict[str, float]], table_dir: Path) -> dict[str, float]:
    val_steps, val_l1 = finite_metric_points(rows, "val_action_l1")
    train_steps, train_l1 = finite_metric_points(rows, "train_action_l1")
    best_idx = int(np.argmin(val_l1))
    best_step = int(val_steps[best_idx])
    best_val_l1 = float(val_l1[best_idx])
    final = rows[-1]
    first = rows[0]
    summary = {
        "first_train_action_l1": first["train_action_l1"],
        "final_train_action_l1": final["train_action_l1"],
        "best_val_step": best_step,
        "best_val_action_l1": best_val_l1,
        "final_val_action_l1": final["val_action_l1"],
        "final_minus_best_val_action_l1": final["val_action_l1"] - best_val_l1,
        "num_steps": int(final["step"]),
        "num_val_points": len(val_steps),
    }
    with (table_dir / "act_B_overfitting_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    return summary


def write_train_l1_smoothed_table(
    train_steps: np.ndarray,
    train_l1: np.ndarray,
    train_l1_smooth: np.ndarray,
    table_dir: Path,
    smoothing_window: int,
    sample_every: int = 1000,
) -> None:
    rows = []
    for step, raw, smooth in zip(train_steps, train_l1, train_l1_smooth, strict=True):
        if int(step) % sample_every != 0 and int(step) != int(train_steps[-1]):
            continue
        rows.append(
            {
                "step": int(step),
                "train_action_l1_raw": float(raw),
                "train_action_l1_trailing_mean": float(smooth),
                "train_smoothing_window_steps": min(int(step), smoothing_window),
            }
        )
    write_rows(table_dir / "act_B_train_l1_smoothed.csv", rows)


def plot_loss_curve(
    rows: list[dict[str, float]],
    summary: dict[str, float],
    figure_dir: Path,
    table_dir: Path,
    smoothing_window: int = 1000,
) -> None:
    train_steps, train_l1 = finite_metric_points(rows, "train_action_l1")
    val_steps, val_l1 = finite_metric_points(rows, "val_action_l1")
    train_l1_smooth = trailing_mean(train_l1, smoothing_window)
    write_train_l1_smoothed_table(train_steps, train_l1, train_l1_smooth, table_dir, smoothing_window)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=180)
    ax.plot(
        train_steps,
        train_l1,
        color="#9ecae1",
        linewidth=0.35,
        alpha=0.22,
        label="Train Action L1, raw mini-batch",
    )
    ax.plot(
        train_steps,
        train_l1_smooth,
        color="#2f6f9f",
        linewidth=1.35,
        alpha=0.95,
        label=f"Train Action L1, {smoothing_window}-step mean",
    )
    ax.plot(val_steps, val_l1, color="#b23a48", linewidth=1.2, label="Validation Action L1")
    ax.axvline(summary["best_val_step"], color="#2f9f6f", linestyle="--", linewidth=1.0, label="Best val")
    ax.scatter(
        [summary["best_val_step"]],
        [summary["best_val_action_l1"]],
        color="#2f9f6f",
        s=28,
        zorder=3,
    )
    ax.scatter(
        [summary["num_steps"]],
        [summary["final_val_action_l1"]],
        color="#1f1f1f",
        s=24,
        zorder=3,
        label="Final val",
    )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Action L1")
    ax.set_title("ACT-B Training and Validation Action L1")
    ax.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "loss_curve_act_B.png")
    plt.close(fig)


def plot_train_val_gap(
    rows: list[dict[str, float]],
    figure_dir: Path,
    table_dir: Path,
    smoothing_window: int = 1000,
) -> None:
    """Plot validation gap using a smoothed train baseline.

    ``train_action_l1`` is logged per mini-batch and is therefore noisy. For a
    gap diagnostic, compare validation against the trailing rolling mean of
    train L1 at the validation step, while keeping the raw gap in the table for
    auditability.
    """
    train_by_step = {
        int(row["step"]): row["train_action_l1"]
        for row in rows
        if math.isfinite(row["train_action_l1"])
    }
    gap_rows = []
    steps = []
    raw_gaps = []
    smooth_gaps = []
    for row in rows:
        train_l1 = row["train_action_l1"]
        val_l1 = row["val_action_l1"]
        if not (math.isfinite(train_l1) and math.isfinite(val_l1)):
            continue
        step = int(row["step"])
        window_start = max(1, step - smoothing_window + 1)
        window_values = [
            train_by_step[idx]
            for idx in range(window_start, step + 1)
            if idx in train_by_step and math.isfinite(train_by_step[idx])
        ]
        train_l1_smooth = float(np.mean(window_values)) if window_values else train_l1
        raw_gap = val_l1 - train_l1
        smooth_gap = val_l1 - train_l1_smooth
        steps.append(step)
        raw_gaps.append(raw_gap)
        smooth_gaps.append(smooth_gap)
        if step % 5000 == 0 or step in {int(rows[-1]["step"])}:
            gap_rows.append(
                {
                    "step": step,
                    "train_action_l1_raw": train_l1,
                    "train_action_l1_trailing_mean": train_l1_smooth,
                    "train_smoothing_window_steps": len(window_values),
                    "val_action_l1": val_l1,
                    "val_minus_train_action_l1_raw": raw_gap,
                    "val_minus_train_action_l1_smoothed": smooth_gap,
                }
            )

    if gap_rows:
        write_rows(table_dir / "act_B_train_val_gap.csv", gap_rows)

    fig, ax = plt.subplots(figsize=(7.0, 3.8), dpi=180)
    ax.plot(steps, raw_gaps, color="#b7a4d9", linewidth=0.6, alpha=0.45, label="Raw mini-batch gap")
    ax.plot(steps, smooth_gaps, color="#6d4c9f", linewidth=1.3, alpha=0.95, label="Gap vs 1000-step train mean")
    ax.axhline(0, color="#222222", linewidth=0.8)
    smooth_gap_array = np.array(smooth_gaps)
    ax.fill_between(
        steps,
        0,
        smooth_gap_array,
        where=smooth_gap_array >= 0,
        color="#b7a4d9",
        alpha=0.28,
    )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation L1 - train L1")
    ax.set_title("ACT-B Train-Val Generalization Gap")
    ax.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "train_val_gap_act_B.png")
    plt.close(fig)


def add_label(img: Image.Image, label: str) -> Image.Image:
    label_h = 24
    canvas = Image.new("RGB", (img.width, img.height + label_h), "white")
    canvas.paste(img, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 5), label, fill=(0, 0, 0))
    return canvas


def make_env_b_samples(split_path: Path, ranges: list[tuple[int, int]], figure_dir: Path, count: int) -> None:
    ids = evenly_spaced_ids(ranges, count)
    cell_w = 200
    static_cells = []
    gripper_cells = []
    for frame_id in ids:
        data = np.load(frame_file(split_path, frame_id))
        static = Image.fromarray(data["rgb_static"]).resize((cell_w, 200))
        gripper = Image.fromarray(data["rgb_gripper"]).resize((cell_w, 200))
        static_cells.append(add_label(static, f"static {frame_id}"))
        gripper_cells.append(add_label(gripper, f"gripper {frame_id}"))

    grid = Image.new("RGB", (len(ids) * cell_w, 2 * 224), "white")
    for col, cell in enumerate(static_cells):
        grid.paste(cell, (col * cell_w, 0))
    for col, cell in enumerate(gripper_cells):
        grid.paste(cell, (col * cell_w, 224))
    grid.save(figure_dir / "env_B_samples.png")


def collect_visual_and_action_stats(
    split_path: Path,
    ranges: list[tuple[int, int]],
    visual_sample_count: int,
) -> tuple[list[dict[str, float | str | int]], list[dict[str, float | str | int]], dict[str, np.ndarray]]:
    visual_ids = set(evenly_spaced_ids(ranges, visual_sample_count))
    visual_acc = {
        "static": {"sum": np.zeros(3), "sumsq": np.zeros(3), "brightness": [], "contrast": [], "pixels": 0},
        "gripper": {"sum": np.zeros(3), "sumsq": np.zeros(3), "brightness": [], "contrast": [], "pixels": 0},
    }
    actions = []
    frame_ids = []
    range_lengths = []

    for start, end in ranges:
        range_lengths.append(end - start + 1)
        for frame_id in range(start, end + 1):
            data = np.load(frame_file(split_path, frame_id))
            actions.append(np.asarray(data["rel_actions"], dtype=np.float64))
            frame_ids.append(frame_id)
            if frame_id not in visual_ids:
                continue
            for key, raw_key in [("static", "rgb_static"), ("gripper", "rgb_gripper")]:
                img = np.asarray(data[raw_key], dtype=np.float64) / 255.0
                flat = img.reshape(-1, 3)
                visual_acc[key]["sum"] += flat.sum(axis=0)
                visual_acc[key]["sumsq"] += (flat * flat).sum(axis=0)
                visual_acc[key]["pixels"] += flat.shape[0]
                gray = flat.mean(axis=1)
                visual_acc[key]["brightness"].append(float(gray.mean()))
                visual_acc[key]["contrast"].append(float(gray.std()))

    visual_rows = []
    for camera, acc in visual_acc.items():
        pixels = int(acc["pixels"])
        mean = acc["sum"] / pixels
        std = np.sqrt(np.maximum(acc["sumsq"] / pixels - mean * mean, 0))
        brightness = np.array(acc["brightness"])
        contrast = np.array(acc["contrast"])
        row = {
            "environment": "B",
            "camera": camera,
            "sampled_frames": len(brightness),
            "rgb_mean_r": mean[0],
            "rgb_mean_g": mean[1],
            "rgb_mean_b": mean[2],
            "rgb_std_r": std[0],
            "rgb_std_g": std[1],
            "rgb_std_b": std[2],
            "brightness_mean": float(brightness.mean()),
            "brightness_std": float(brightness.std()),
            "contrast_mean": float(contrast.mean()),
            "contrast_std": float(contrast.std()),
        }
        visual_rows.append(row)

    action_array = np.stack(actions)
    action_rows = []
    names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
    for dim, name in enumerate(names):
        vals = action_array[:, dim]
        action_rows.append(
            {
                "environment": "B",
                "action_dim": dim,
                "action_name": name,
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "q01": float(np.quantile(vals, 0.01)),
                "q50": float(np.quantile(vals, 0.50)),
                "q99": float(np.quantile(vals, 0.99)),
            }
        )
    return visual_rows, action_rows, {
        "actions": action_array,
        "frame_ids": np.asarray(frame_ids, dtype=np.int64),
        "range_lengths": np.asarray(range_lengths, dtype=np.int64),
        "static_brightness": np.asarray(visual_acc["static"]["brightness"], dtype=np.float64),
        "static_contrast": np.asarray(visual_acc["static"]["contrast"], dtype=np.float64),
        "gripper_brightness": np.asarray(visual_acc["gripper"]["brightness"], dtype=np.float64),
        "gripper_contrast": np.asarray(visual_acc["gripper"]["contrast"], dtype=np.float64),
    }


def write_rows(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def episode_diffs(action_array: np.ndarray, range_lengths: np.ndarray) -> np.ndarray:
    diffs = []
    cursor = 0
    for length in range_lengths:
        segment = action_array[cursor : cursor + int(length)]
        cursor += int(length)
        if len(segment) > 1:
            diffs.append(np.diff(segment, axis=0))
    if not diffs:
        return np.empty((0, action_array.shape[1]))
    return np.concatenate(diffs, axis=0)


def write_action_summary(action_array: np.ndarray, range_lengths: np.ndarray, table_dir: Path) -> dict[str, float]:
    diffs = episode_diffs(action_array, range_lengths)
    trans = action_array[:, :6]
    gripper = action_array[:, 6]
    summary = {
        "environment": "B",
        "num_frames": int(action_array.shape[0]),
        "mean_action_l2_first6": float(np.linalg.norm(trans, axis=1).mean()),
        "std_action_l2_first6": float(np.linalg.norm(trans, axis=1).std()),
        "mean_step_delta_l2": float(np.linalg.norm(diffs, axis=1).mean()),
        "std_step_delta_l2": float(np.linalg.norm(diffs, axis=1).std()),
        "gripper_close_fraction": float((gripper < 0).mean()),
        "gripper_open_fraction": float((gripper > 0).mean()),
        "gripper_switch_count": int(np.count_nonzero(np.diff(np.sign(gripper)))),
    }
    with (table_dir / "env_B_action_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    return summary


def plot_action_distribution(action_rows: list[dict[str, float | str | int]], figure_dir: Path) -> None:
    names = [str(row["action_name"]) for row in action_rows]
    means = np.array([float(row["mean"]) for row in action_rows])
    stds = np.array([float(row["std"]) for row in action_rows])
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=180)
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, color="#4c78a8", edgecolor="#222222", linewidth=0.5, capsize=3)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Relative action value")
    ax.set_title("Environment B Action Distribution")
    ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(figure_dir / "env_B_action_distribution.png")
    plt.close(fig)


def downsample_rows(array: np.ndarray, max_rows: int) -> np.ndarray:
    if len(array) <= max_rows:
        return array
    ids = np.linspace(0, len(array) - 1, max_rows, dtype=int)
    return array[ids]


def plot_action_violin(action_array: np.ndarray, figure_dir: Path) -> None:
    names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
    sampled = downsample_rows(action_array, 50000)
    data = [sampled[:, dim] for dim in range(sampled.shape[1])]
    fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=180)
    parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#4c78a8")
        body.set_edgecolor("#1f1f1f")
        body.set_alpha(0.72)
    parts["cmedians"].set_color("#1f1f1f")
    parts["cmedians"].set_linewidth(1.1)
    ax.axhline(0, color="#222222", linewidth=0.7)
    ax.set_xticks(np.arange(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Relative action value")
    ax.set_title("Environment B Per-Dimension Action Distribution")
    ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(figure_dir / "env_B_action_violin.png")
    plt.close(fig)


def write_and_plot_action_smoothness(
    action_array: np.ndarray,
    range_lengths: np.ndarray,
    table_dir: Path,
    figure_dir: Path,
) -> None:
    names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
    diffs = episode_diffs(action_array, range_lengths)
    abs_diffs = np.abs(diffs)
    delta_l2_first6 = np.linalg.norm(diffs[:, :6], axis=1)
    rows = []
    for dim, name in enumerate(names):
        vals = abs_diffs[:, dim]
        rows.append(
            {
                "environment": "B",
                "action_dim": dim,
                "action_name": name,
                "mean_abs_delta": float(vals.mean()),
                "std_abs_delta": float(vals.std()),
                "q50_abs_delta": float(np.quantile(vals, 0.50)),
                "q90_abs_delta": float(np.quantile(vals, 0.90)),
                "q99_abs_delta": float(np.quantile(vals, 0.99)),
            }
        )
    rows.append(
        {
            "environment": "B",
            "action_dim": -1,
            "action_name": "l2_first6",
            "mean_abs_delta": float(delta_l2_first6.mean()),
            "std_abs_delta": float(delta_l2_first6.std()),
            "q50_abs_delta": float(np.quantile(delta_l2_first6, 0.50)),
            "q90_abs_delta": float(np.quantile(delta_l2_first6, 0.90)),
            "q99_abs_delta": float(np.quantile(delta_l2_first6, 0.99)),
        }
    )
    write_rows(table_dir / "env_B_action_smoothness.csv", rows)

    sampled = downsample_rows(delta_l2_first6[:, None], 100000).ravel()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), dpi=180)
    axes[0].hist(sampled, bins=80, color="#4c78a8", alpha=0.88)
    axes[0].set_xlabel("||a_t - a_{t-1}||, first 6 dims")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Step Delta L2")
    means = np.array([float(row["mean_abs_delta"]) for row in rows[:-1]])
    axes[1].bar(np.arange(len(names)), means, color="#72b7b2", edgecolor="#222222", linewidth=0.4)
    axes[1].set_xticks(np.arange(len(names)))
    axes[1].set_xticklabels(names, rotation=35, ha="right")
    axes[1].set_ylabel("Mean absolute delta")
    axes[1].set_title("Per-Dimension Smoothness")
    for ax in axes:
        ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(figure_dir / "env_B_action_smoothness.png")
    plt.close(fig)


def write_and_plot_gripper_diagnostics(
    action_array: np.ndarray,
    range_lengths: np.ndarray,
    table_dir: Path,
    figure_dir: Path,
    timeline_window_frames: int = 1200,
) -> None:
    gripper = action_array[:, 6]
    close_mask = gripper < 0
    open_mask = gripper > 0
    switches = np.flatnonzero(np.diff(np.sign(gripper)) != 0) + 1

    run_lengths = []
    run_states = []
    start = 0
    for idx in range(1, len(gripper)):
        if np.sign(gripper[idx]) != np.sign(gripper[idx - 1]):
            run_lengths.append(idx - start)
            run_states.append(gripper[start])
            start = idx
    run_lengths.append(len(gripper) - start)
    run_states.append(gripper[start])
    run_lengths_arr = np.asarray(run_lengths, dtype=np.float64)
    run_states_arr = np.asarray(run_states, dtype=np.float64)

    summary = [
        {
            "environment": "B",
            "num_frames": int(len(gripper)),
            "close_count": int(close_mask.sum()),
            "open_count": int(open_mask.sum()),
            "close_fraction": float(close_mask.mean()),
            "open_fraction": float(open_mask.mean()),
            "switch_count": int(len(switches)),
            "switch_rate_per_1000_frames": float(len(switches) / len(gripper) * 1000.0),
            "mean_run_length_frames": float(run_lengths_arr.mean()),
            "median_run_length_frames": float(np.median(run_lengths_arr)),
            "q90_run_length_frames": float(np.quantile(run_lengths_arr, 0.90)),
            "q99_run_length_frames": float(np.quantile(run_lengths_arr, 0.99)),
        }
    ]
    write_rows(table_dir / "env_B_gripper_summary.csv", summary)

    run_rows = []
    for run_idx, (length, state) in enumerate(zip(run_lengths_arr, run_states_arr, strict=True)):
        run_rows.append(
            {
                "environment": "B",
                "run_index": run_idx,
                "gripper_state": "open" if state > 0 else "close",
                "state_value": float(state),
                "run_length_frames": int(length),
            }
        )
    write_rows(table_dir / "env_B_gripper_runs.csv", run_rows)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), dpi=180, constrained_layout=True)
    axes[0].bar(
        ["close (-1)", "open (+1)"],
        [float(close_mask.mean()), float(open_mask.mean())],
        color=["#b23a48", "#2f6f9f"],
        edgecolor="#222222",
        linewidth=0.5,
    )
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Frame fraction")
    axes[0].set_title("Gripper State Balance")
    clipped_runs = np.clip(run_lengths_arr, 0, np.quantile(run_lengths_arr, 0.99))
    axes[1].hist(clipped_runs, bins=60, color="#6d4c9f", alpha=0.84)
    axes[1].set_xlabel("Run length, frames")
    axes[1].set_ylabel("Run count")
    axes[1].set_title("Open/Close Run Lengths")
    for ax in axes:
        ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    fig.suptitle("Environment B Gripper Diagnostics")
    fig.savefig(figure_dir / "env_B_gripper_diagnostics.png", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    longest_idx = int(np.argmax(range_lengths))
    cursor = int(range_lengths[:longest_idx].sum())
    long_len = int(range_lengths[longest_idx])
    window_len = min(timeline_window_frames, long_len)
    local_start = max(0, long_len // 2 - window_len // 2)
    local_end = local_start + window_len
    segment = gripper[cursor + local_start : cursor + local_end]
    x = np.arange(len(segment))
    switch_local = np.flatnonzero(np.diff(np.sign(segment)) != 0) + 1

    fig, ax = plt.subplots(figsize=(8.0, 2.8), dpi=180)
    ax.step(x, segment, where="post", color="#2f6f9f", linewidth=1.2)
    for switch_idx in switch_local:
        ax.axvline(int(switch_idx), color="#b23a48", linewidth=0.55, alpha=0.45)
    ax.set_yticks([-1, 1])
    ax.set_yticklabels(["close", "open"])
    ax.set_xlabel("Frame in representative 1,200-frame window")
    ax.set_title("Environment B Gripper State Timeline")
    ax.grid(True, axis="x", color="#d0d0d0", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(figure_dir / "env_B_gripper_timeline.png")
    plt.close(fig)


def write_and_plot_chunk_baseline(
    action_array: np.ndarray,
    range_lengths: np.ndarray,
    table_dir: Path,
    figure_dir: Path,
    chunk_size: int,
) -> None:
    chunk_rows = []
    boundary_jumps = []
    within_delta = []
    first_last = []
    cursor = 0
    chunk_index = 0
    for episode_index, length in enumerate(range_lengths):
        segment = action_array[cursor : cursor + int(length)]
        cursor += int(length)
        prev_last = None
        for chunk_start in range(0, len(segment), chunk_size):
            chunk = segment[chunk_start : chunk_start + chunk_size]
            if len(chunk) < 2:
                continue
            diffs = np.diff(chunk, axis=0)
            chunk_within = float(np.linalg.norm(diffs[:, :6], axis=1).mean())
            chunk_first_last = float(np.linalg.norm(chunk[-1, :6] - chunk[0, :6]))
            boundary = float(np.linalg.norm(chunk[0, :6] - prev_last[:6])) if prev_last is not None else np.nan
            if math.isfinite(boundary):
                boundary_jumps.append(boundary)
            within_delta.append(chunk_within)
            first_last.append(chunk_first_last)
            chunk_rows.append(
                {
                    "environment": "B",
                    "episode_index": episode_index,
                    "chunk_index": chunk_index,
                    "chunk_size": chunk_size,
                    "chunk_length": len(chunk),
                    "mean_within_chunk_delta_l2_first6": chunk_within,
                    "first_last_delta_l2_first6": chunk_first_last,
                    "boundary_jump_l2_first6": boundary,
                }
            )
            prev_last = chunk[-1]
            chunk_index += 1
    write_rows(table_dir / "env_B_chunk_baseline.csv", chunk_rows)

    summary = [
        {
            "environment": "B",
            "chunk_size": chunk_size,
            "num_chunks": len(chunk_rows),
            "mean_within_chunk_delta_l2_first6": float(np.mean(within_delta)),
            "mean_first_last_delta_l2_first6": float(np.mean(first_last)),
            "mean_boundary_jump_l2_first6": float(np.mean(boundary_jumps)),
            "q90_boundary_jump_l2_first6": float(np.quantile(boundary_jumps, 0.90)),
            "q99_boundary_jump_l2_first6": float(np.quantile(boundary_jumps, 0.99)),
        }
    ]
    write_rows(table_dir / "env_B_chunk_baseline_summary.csv", summary)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), dpi=180)
    axes[0].hist(downsample_rows(np.asarray(within_delta)[:, None], 50000).ravel(), bins=60, color="#54a24b", alpha=0.86)
    axes[0].set_xlabel("Mean within-chunk delta L2")
    axes[0].set_ylabel("Chunk count")
    axes[0].set_title(f"Within-Chunk Variation, c={chunk_size}")
    axes[1].hist(downsample_rows(np.asarray(boundary_jumps)[:, None], 50000).ravel(), bins=60, color="#e45756", alpha=0.86)
    axes[1].set_xlabel("Boundary jump L2")
    axes[1].set_ylabel("Boundary count")
    axes[1].set_title("Boundary-Style Jumps")
    for ax in axes:
        ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(figure_dir / "env_B_chunk_baseline.png")
    plt.close(fig)


def write_and_plot_action_delta_heatmap(
    action_array: np.ndarray,
    range_lengths: np.ndarray,
    table_dir: Path,
    figure_dir: Path,
    bins: int,
) -> None:
    names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
    diffs = np.abs(episode_diffs(action_array, range_lengths))
    ids = np.linspace(0, len(diffs), bins + 1, dtype=int)
    heat = np.zeros((bins, diffs.shape[1]), dtype=np.float64)
    table_rows = []
    for idx in range(bins):
        block = diffs[ids[idx] : ids[idx + 1]]
        if len(block) == 0:
            continue
        heat[idx] = block.mean(axis=0)
        row = {"bin": idx, "start_delta_index": int(ids[idx]), "end_delta_index": int(ids[idx + 1])}
        for dim, name in enumerate(names):
            row[f"{name}_mean_abs_delta"] = heat[idx, dim]
        table_rows.append(row)
    write_rows(table_dir / "env_B_action_delta_heatmap.csv", table_rows)

    fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=180)
    im = ax.imshow(heat.T, aspect="auto", interpolation="nearest", cmap="magma")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Time bin across environment B")
    ax.set_title("Environment B Action Delta Heatmap")
    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
    cbar.set_label("Mean absolute delta")
    fig.tight_layout()
    fig.savefig(figure_dir / "env_B_action_delta_heatmap.png")
    plt.close(fig)


def plot_visual_color_profile(visual_rows: list[dict[str, float | str | int]], figure_dir: Path) -> None:
    channels = ["r", "g", "b"]
    colors = ["#c44e52", "#55a868", "#4c72b0"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), dpi=180, sharey=True, constrained_layout=True)
    for ax, row in zip(axes, visual_rows):
        means = np.array([float(row[f"rgb_mean_{channel}"]) for channel in channels])
        stds = np.array([float(row[f"rgb_std_{channel}"]) for channel in channels])
        x = np.arange(len(channels))
        ax.bar(x, means, yerr=stds, color=colors, edgecolor="#222222", linewidth=0.4, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(["R", "G", "B"])
        ax.set_title(f"{row['camera']} camera")
        ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel("Pixel value, normalized")
    fig.suptitle("Environment B Visual Color Profile")
    fig.savefig(figure_dir / "env_B_visual_color_profile.png", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def plot_brightness_contrast_hist(arrays: dict[str, np.ndarray], figure_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), dpi=180, constrained_layout=True)
    axes[0].hist(arrays["static_brightness"], bins=45, color="#4c78a8", alpha=0.65, label="static")
    axes[0].hist(arrays["gripper_brightness"], bins=45, color="#f58518", alpha=0.65, label="gripper")
    axes[0].set_xlabel("Brightness")
    axes[0].set_ylabel("Sampled frame count")
    axes[0].set_title("Brightness")
    axes[1].hist(arrays["static_contrast"], bins=45, color="#4c78a8", alpha=0.65, label="static")
    axes[1].hist(arrays["gripper_contrast"], bins=45, color="#f58518", alpha=0.65, label="gripper")
    axes[1].set_xlabel("Contrast")
    axes[1].set_ylabel("Sampled frame count")
    axes[1].set_title("Contrast")
    for ax in axes:
        ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
        ax.legend(frameon=False)
    fig.suptitle("Environment B Brightness and Contrast")
    fig.savefig(figure_dir / "env_B_brightness_contrast_hist.png", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def plot_task_frequency(task_rows: list[dict[str, float | str | int]], figure_dir: Path) -> None:
    sorted_rows = sorted(task_rows, key=lambda row: int(row["language_sequences"]), reverse=True)
    tasks = [str(row["task"]).replace("_", " ") for row in sorted_rows]
    counts = np.array([int(row["language_sequences"]) for row in sorted_rows])
    fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=180)
    y = np.arange(len(tasks))
    ax.barh(y, counts, color="#4c78a8", edgecolor="#222222", linewidth=0.35)
    ax.set_yticks(y)
    ax.set_yticklabels(tasks, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Language sequences")
    ax.set_title("Environment B Task Frequency")
    ax.grid(True, axis="x", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(figure_dir / "env_B_task_frequency.png")
    plt.close(fig)


def make_representative_trajectory_strip(
    split_path: Path,
    ranges: list[tuple[int, int]],
    figure_dir: Path,
    table_dir: Path,
    sample_count: int,
    window_frames: int,
    smoothing_window: int,
) -> None:
    long_start, long_end = max(ranges, key=lambda item: item[1] - item[0])
    long_len = long_end - long_start + 1
    window_len = min(window_frames, long_len)
    center = (long_start + long_end) // 2
    start = max(long_start, center - window_len // 2)
    end = min(long_end, start + window_len - 1)
    start = max(long_start, end - window_len + 1)
    frame_ids = np.linspace(start, end, sample_count, dtype=int)
    action_ids = np.arange(start, end + 1, dtype=int)
    action_norm = []
    for frame_id in action_ids:
        data = np.load(frame_file(split_path, int(frame_id)))
        action_norm.append(float(np.linalg.norm(np.asarray(data["rel_actions"], dtype=np.float64)[:6])))
    action_norm = np.asarray(action_norm, dtype=np.float64)
    action_norm_smooth = trailing_mean(action_norm, smoothing_window)

    images = []
    rows = []
    for frame_id in frame_ids:
        data = np.load(frame_file(split_path, int(frame_id)))
        action_idx = int(frame_id - start)
        images.append(Image.fromarray(data["rgb_static"]).resize((160, 160)))
        rows.append(
            {
                "environment": "B",
                "source_long_segment_start": int(long_start),
                "source_long_segment_end": int(long_end),
                "window_start": int(start),
                "window_end": int(end),
                "frame_id": int(frame_id),
                "action_l2_first6_raw": float(action_norm[action_idx]),
                "action_l2_first6_trailing_mean": float(action_norm_smooth[action_idx]),
                "trajectory_smoothing_window_frames": int(smoothing_window),
            }
        )
    write_rows(table_dir / "env_B_representative_trajectory.csv", rows)

    fig = plt.figure(figsize=(10.0, 4.6), dpi=180)
    gs = fig.add_gridspec(2, sample_count, height_ratios=[1.0, 0.75], hspace=0.18, wspace=0.02)
    for idx, img in enumerate(images):
        ax_img = fig.add_subplot(gs[0, idx])
        ax_img.imshow(img)
        ax_img.set_title(str(int(frame_ids[idx])), fontsize=7)
        ax_img.axis("off")
    ax = fig.add_subplot(gs[1, :])
    rel_x = np.linspace(0, 1, len(action_norm))
    ax.plot(rel_x, action_norm, color="#9ecae1", linewidth=0.6, alpha=0.45, label="Raw action norm")
    ax.plot(
        rel_x,
        action_norm_smooth,
        color="#2f6f9f",
        linewidth=1.5,
        alpha=0.96,
        label=f"{smoothing_window}-frame trailing mean",
    )
    sample_x = (frame_ids - start) / max(end - start, 1)
    for sx in sample_x:
        ax.axvline(float(sx), color="#e45756", linewidth=0.7, alpha=0.45)
    ax.set_xlabel("Relative time in trajectory")
    ax.set_ylabel("Action L2, first 6 dims")
    ax.set_title(f"Representative Environment B Trajectory Window ({end - start + 1} frames)")
    ax.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(figure_dir / "env_B_representative_trajectory_strip.png")
    plt.close(fig)


def write_checkpoint_selection(summary: dict[str, float], run_dir: Path, table_dir: Path) -> None:
    rows = [
        {
            "experiment": "act_B",
            "selection": "best_val",
            "step": int(summary["best_val_step"]),
            "val_action_l1": summary["best_val_action_l1"],
            "checkpoint": str(run_dir / "checkpoints" / f"step_{int(summary['best_val_step']):08d}"),
            "notes": "candidate early-stopped checkpoint",
        },
        {
            "experiment": "act_B",
            "selection": "final",
            "step": int(summary["num_steps"]),
            "val_action_l1": summary["final_val_action_l1"],
            "checkpoint": str(run_dir / "checkpoint"),
            "notes": "full 100k-step endpoint",
        },
    ]
    write_rows(table_dir / "act_B_checkpoint_selection.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("project"))
    parser.add_argument("--visual-sample-count", type=int, default=2000)
    parser.add_argument("--image-sample-count", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--heatmap-bins", type=int, default=100)
    parser.add_argument("--trajectory-sample-count", type=int, default=10)
    parser.add_argument("--trajectory-window-frames", type=int, default=1200)
    parser.add_argument("--trajectory-smoothing-window", type=int, default=31)
    args = parser.parse_args()

    output_dir = args.output_dir
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    raw_root = configured_raw_root(args.raw_root)
    env_range = load_environment_ranges(raw_root)["B"]
    b_split = split_dir(raw_root, env_range.split)
    b_ranges = clipped_episode_ranges(load_episode_ranges(b_split), env_range)

    metrics_path = args.run_dir / "metrics.csv"
    rows = load_metrics(metrics_path)
    summary = write_loss_summary(rows, table_dir)
    plot_loss_curve(rows, summary, figure_dir, table_dir)
    plot_train_val_gap(rows, figure_dir, table_dir)
    write_checkpoint_selection(summary, args.run_dir, table_dir)

    make_env_b_samples(b_split, b_ranges, figure_dir, args.image_sample_count)
    visual_rows, action_rows, arrays = collect_visual_and_action_stats(
        b_split,
        b_ranges,
        args.visual_sample_count,
    )
    write_rows(table_dir / "env_B_visual_stats.csv", visual_rows)
    write_rows(table_dir / "env_B_action_stats.csv", action_rows)
    write_action_summary(arrays["actions"], arrays["range_lengths"], table_dir)
    plot_action_distribution(action_rows, figure_dir)
    plot_action_violin(arrays["actions"], figure_dir)
    write_and_plot_action_smoothness(arrays["actions"], arrays["range_lengths"], table_dir, figure_dir)
    write_and_plot_gripper_diagnostics(arrays["actions"], arrays["range_lengths"], table_dir, figure_dir)
    write_and_plot_chunk_baseline(
        arrays["actions"],
        arrays["range_lengths"],
        table_dir,
        figure_dir,
        args.chunk_size,
    )
    write_and_plot_action_delta_heatmap(
        arrays["actions"],
        arrays["range_lengths"],
        table_dir,
        figure_dir,
        args.heatmap_bins,
    )
    plot_visual_color_profile(visual_rows, figure_dir)
    plot_brightness_contrast_hist(arrays, figure_dir)

    task_counts = count_language_tasks(load_language_intervals(b_split), env_range)
    task_rows = [
        {"environment": "B", "task": task, "language_sequences": count}
        for task, count in sorted(task_counts.items())
    ]
    write_rows(table_dir / "env_B_task_counts.csv", task_rows)
    plot_task_frequency(task_rows, figure_dir)
    make_representative_trajectory_strip(
        b_split,
        b_ranges,
        figure_dir,
        table_dir,
        args.trajectory_sample_count,
        args.trajectory_window_frames,
        args.trajectory_smoothing_window,
    )
    print(f"Wrote ACT-B extension tables to {table_dir}")
    print(f"Wrote ACT-B extension figures to {figure_dir}")


if __name__ == "__main__":
    main()
