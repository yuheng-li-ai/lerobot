#!/usr/bin/env python3
"""Per-model Task 2 visualizations from completed ACT full runs.

This script intentionally does not create cross-model comparison figures.
Each experiment gets its own figure/table folder.
"""

from __future__ import annotations

import argparse
import csv
import math
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TASK2_EXPERIMENTS = [
    "act_ABC",
    "act_ABC_size_matched",
    "act_ABC_aug",
    "act_ABC_size_matched_aug",
]

DISPLAY_NAMES = {
    "act_ABC": "ACT-ABC",
    "act_ABC_size_matched": "ACT-ABC Size-Matched",
    "act_ABC_aug": "ACT-ABC Aug",
    "act_ABC_size_matched_aug": "ACT-ABC Size-Matched Aug",
}

SPLIT_BY_EXPERIMENT = {
    "act_ABC": "full",
    "act_ABC_aug": "full",
    "act_ABC_size_matched": "size_matched",
    "act_ABC_size_matched_aug": "size_matched",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fval(value: str | float) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def finite_points(rows: list[dict[str, str]], key: str) -> tuple[np.ndarray, np.ndarray]:
    points = []
    for row in rows:
        value = fval(row[key])
        if math.isfinite(value):
            points.append((int(row["step"]), value))
    return np.array([p[0] for p in points]), np.array([p[1] for p in points], dtype=np.float64)


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


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)


def title(ax: plt.Axes, text: str) -> None:
    ax.set_title(textwrap.fill(text, width=70), pad=18, fontsize=11)


def load_summary(summary_rows: list[dict[str, str]], experiment: str) -> dict[str, str]:
    for row in summary_rows:
        if row["experiment"] == experiment:
            return row
    raise KeyError(experiment)


def plot_loss_curve(
    experiment: str,
    metrics: list[dict[str, str]],
    summary: dict[str, str],
    figure_dir: Path,
    table_dir: Path,
    smoothing_window: int,
) -> None:
    train_steps, train_l1 = finite_points(metrics, "train_action_l1")
    val_steps, val_l1 = finite_points(metrics, "val_action_l1")
    train_smooth = trailing_mean(train_l1, smoothing_window)

    smoothed_rows = []
    for step, raw, smooth in zip(train_steps, train_l1, train_smooth, strict=True):
        if int(step) % 1000 == 0 or int(step) == int(train_steps[-1]):
            smoothed_rows.append(
                {
                    "step": int(step),
                    "train_action_l1_raw": float(raw),
                    "train_action_l1_trailing_mean": float(smooth),
                    "train_smoothing_window_steps": min(int(step), smoothing_window),
                }
            )
    write_rows(table_dir / "train_l1_smoothed.csv", smoothed_rows)

    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=180, constrained_layout=True)
    ax.plot(train_steps, train_l1, color="#b7d7ea", linewidth=0.35, alpha=0.24, label="Train Action L1, raw")
    ax.plot(
        train_steps,
        train_smooth,
        color="#2f6f9f",
        linewidth=1.35,
        label=f"Train Action L1, {smoothing_window}-step mean",
    )
    ax.plot(val_steps, val_l1, color="#b23a48", linewidth=1.25, label="Validation Action L1")
    best_step = int(summary["best_val_step"])
    best_val = fval(summary["best_val_action_l1"])
    final_step = int(summary["final_step"])
    final_val = fval(summary["final_val_action_l1"])
    selected_step = int(summary["selected_available_checkpoint_step"])
    selected_val = fval(summary["selected_available_checkpoint_val_action_l1"])
    ax.axvline(best_step, color="#2f9f6f", linestyle="--", linewidth=1.0, label="Best validation step")
    ax.scatter([best_step], [best_val], color="#2f9f6f", s=34, zorder=4)
    ax.scatter([selected_step], [selected_val], color="#f2a541", s=42, marker="D", zorder=4, label="Selected checkpoint")
    ax.scatter([final_step], [final_val], color="#1f1f1f", s=30, zorder=4, label="Final validation")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Action L1")
    title(ax, f"{DISPLAY_NAMES[experiment]} Training and Validation Action L1")
    ax.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=False, fontsize=8, loc="best")
    save_fig(fig, figure_dir / "loss_curve.png")


def plot_train_val_gap(
    experiment: str,
    metrics: list[dict[str, str]],
    figure_dir: Path,
    table_dir: Path,
    smoothing_window: int,
) -> None:
    train_steps, train_l1 = finite_points(metrics, "train_action_l1")
    train_smooth = trailing_mean(train_l1, smoothing_window)
    smooth_by_step = {int(step): float(value) for step, value in zip(train_steps, train_smooth, strict=True)}
    raw_by_step = {int(row["step"]): fval(row["train_action_l1"]) for row in metrics}

    rows = []
    steps = []
    raw_gaps = []
    smooth_gaps = []
    for row in metrics:
        step = int(row["step"])
        val = fval(row["val_action_l1"])
        if not math.isfinite(val):
            continue
        raw_train = raw_by_step[step]
        smooth_train = smooth_by_step[step]
        raw_gap = val - raw_train
        smooth_gap = val - smooth_train
        steps.append(step)
        raw_gaps.append(raw_gap)
        smooth_gaps.append(smooth_gap)
        rows.append(
            {
                "step": step,
                "train_action_l1_raw": raw_train,
                "train_action_l1_trailing_mean": smooth_train,
                "val_action_l1": val,
                "val_minus_train_action_l1_raw": raw_gap,
                "val_minus_train_action_l1_smoothed": smooth_gap,
            }
        )
    write_rows(table_dir / "train_val_gap.csv", rows)

    fig, ax = plt.subplots(figsize=(8.2, 4.3), dpi=180, constrained_layout=True)
    ax.plot(steps, raw_gaps, color="#b7a4d9", linewidth=0.65, alpha=0.42, label="Raw mini-batch gap")
    ax.plot(steps, smooth_gaps, color="#6d4c9f", linewidth=1.35, label="Gap vs smoothed train L1")
    ax.axhline(0, color="#222222", linewidth=0.8)
    smooth_gap_array = np.array(smooth_gaps, dtype=np.float64)
    ax.fill_between(steps, 0, smooth_gap_array, where=smooth_gap_array >= 0, color="#b7a4d9", alpha=0.26)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation L1 - train L1")
    title(ax, f"{DISPLAY_NAMES[experiment]} Train-Val Generalization Gap")
    ax.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=False, fontsize=8, loc="best")
    save_fig(fig, figure_dir / "train_val_gap.png")


def plot_checkpoint_selection(
    experiment: str,
    metrics: list[dict[str, str]],
    selection: dict[str, str],
    figure_dir: Path,
    table_dir: Path,
) -> None:
    val_steps, val_l1 = finite_points(metrics, "val_action_l1")
    checkpoint_steps = []
    checkpoint_l1 = []
    for step, value in zip(val_steps, val_l1, strict=True):
        if int(step) % 10000 == 0 or int(step) == int(val_steps[-1]):
            checkpoint_steps.append(int(step))
            checkpoint_l1.append(float(value))
    best_step = int(selection["best_val_step"])
    best_val = fval(selection["best_val_action_l1"])
    prev_step = int(selection["previous_checkpoint_step"])
    prev_val = fval(selection["previous_checkpoint_val_action_l1"])
    next_step = int(selection["next_checkpoint_step"])
    next_val = fval(selection["next_checkpoint_val_action_l1"])
    selected_step = int(selection["selected_available_checkpoint_step"])
    selected_val = fval(selection["selected_available_checkpoint_val_action_l1"])

    rows = []
    for step, val in zip(checkpoint_steps, checkpoint_l1, strict=True):
        rows.append(
            {
                "step": step,
                "val_action_l1": val,
                "is_previous_neighbor": step == prev_step,
                "is_next_neighbor": step == next_step,
                "is_selected": step == selected_step,
            }
        )
    write_rows(table_dir / "checkpoint_selection.csv", rows)

    fig, ax = plt.subplots(figsize=(8.2, 4.5), dpi=180, constrained_layout=True)
    ax.plot(checkpoint_steps, checkpoint_l1, color="#4d6a7d", linewidth=1.2, marker="o", markersize=3.5)
    ax.scatter([best_step], [best_val], color="#2f9f6f", s=48, marker="*", zorder=5, label="Best validation point")
    ax.scatter([prev_step], [prev_val], color="#a6a6a6", s=42, marker="s", zorder=4, label="Previous checkpoint")
    ax.scatter([next_step], [next_val], color="#c7a66b", s=42, marker="s", zorder=4, label="Next checkpoint")
    ax.scatter([selected_step], [selected_val], color="#d95f02", s=58, marker="D", zorder=6, label="Selected checkpoint")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation Action L1")
    title(ax, f"{DISPLAY_NAMES[experiment]} Checkpoint Selection")
    ax.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=False, fontsize=8, loc="best")
    save_fig(fig, figure_dir / "checkpoint_selection.png")


def plot_step_time(
    experiment: str,
    metrics: list[dict[str, str]],
    figure_dir: Path,
    table_dir: Path,
    smoothing_window: int,
) -> None:
    steps, step_s = finite_points(metrics, "step_s")
    step_s_smooth = trailing_mean(step_s, smoothing_window)
    rows = []
    for step, raw, smooth in zip(steps, step_s, step_s_smooth, strict=True):
        if int(step) % 1000 == 0 or int(step) == int(steps[-1]):
            rows.append({"step": int(step), "step_s_raw": float(raw), "step_s_trailing_mean": float(smooth)})
    write_rows(table_dir / "step_time_profile.csv", rows)

    fig, ax = plt.subplots(figsize=(8.2, 4.3), dpi=180, constrained_layout=True)
    ax.plot(steps, step_s, color="#c9c9c9", linewidth=0.35, alpha=0.35, label="Raw step time")
    ax.plot(steps, step_s_smooth, color="#3c6e71", linewidth=1.25, label=f"{smoothing_window}-step mean")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Seconds per logged step")
    title(ax, f"{DISPLAY_NAMES[experiment]} Step-Time Profile")
    ax.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=False, fontsize=8, loc="best")
    save_fig(fig, figure_dir / "step_time_profile.png")


def plot_dataset_split(
    experiment: str,
    split_rows: list[dict[str, str]],
    figure_dir: Path,
    table_dir: Path,
) -> None:
    split = SPLIT_BY_EXPERIMENT[experiment]
    rows = [r for r in split_rows if r["split"] == split and r["environment"] in {"A", "B", "C", "ABC_total"}]
    write_rows(table_dir / "dataset_split.csv", rows)

    envs = ["A", "B", "C"]
    train = [int(next(r for r in rows if r["subset"] == "train" and r["environment"] == env)["num_frames"]) for env in envs]
    val = [int(next(r for r in rows if r["subset"] == "val" and r["environment"] == env)["num_frames"]) for env in envs]
    x = np.arange(len(envs))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=180, constrained_layout=True)
    ax.bar(x - width / 2, train, width=width, color="#4c78a8", label="Train frames")
    ax.bar(x + width / 2, val, width=width, color="#f58518", label="Validation frames")
    ax.set_xticks(x, envs)
    ax.set_xlabel("CALVIN environment")
    ax.set_ylabel("Frames")
    title(ax, f"{DISPLAY_NAMES[experiment]} Dataset Split ({split.replace('_', ' ')})")
    ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=False, fontsize=8, loc="best")
    save_fig(fig, figure_dir / "dataset_split.png")


def copy_summary_tables(
    experiment: str,
    summary: dict[str, str],
    selection: dict[str, str],
    table_dir: Path,
) -> None:
    write_rows(table_dir / "full_training_summary.csv", [summary])
    write_rows(table_dir / "model_selection.csv", [selection])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-root", type=Path, default=Path("project/figures/task2"))
    parser.add_argument("--table-root", type=Path, default=Path("project/tables/task2"))
    parser.add_argument("--smoothing-window", type=int, default=1000)
    args = parser.parse_args()

    summary_rows = read_csv(Path("project/tables/task2_full_training_summary.csv"))
    selection_rows = read_csv(Path("project/tables/model_selection_checkpoints.csv"))
    split_rows = read_csv(Path("project/tables/task2_episode_splits.csv"))
    manifest = []

    for experiment in TASK2_EXPERIMENTS:
        summary = load_summary(summary_rows, experiment)
        selection = load_summary(selection_rows, experiment)
        metrics_path = Path(summary["run_dir"]) / "metrics.csv"
        metrics = read_csv(metrics_path)
        figure_dir = args.figure_root / experiment
        table_dir = args.table_root / experiment
        figure_dir.mkdir(parents=True, exist_ok=True)
        table_dir.mkdir(parents=True, exist_ok=True)

        copy_summary_tables(experiment, summary, selection, table_dir)
        plot_loss_curve(experiment, metrics, summary | selection, figure_dir, table_dir, args.smoothing_window)
        plot_train_val_gap(experiment, metrics, figure_dir, table_dir, args.smoothing_window)
        plot_checkpoint_selection(experiment, metrics, selection, figure_dir, table_dir)
        plot_step_time(experiment, metrics, figure_dir, table_dir, args.smoothing_window)
        plot_dataset_split(experiment, split_rows, figure_dir, table_dir)

        for fig_path in sorted(figure_dir.glob("*.png")):
            manifest.append(
                {
                    "experiment": experiment,
                    "figure": str(fig_path),
                    "kind": fig_path.stem,
                    "note": "single-model diagnostic; no cross-model comparison",
                }
            )

    write_rows(Path("project/tables/task2_visualization_manifest.csv"), manifest)
    print(f"Wrote {len(manifest)} Task 2 figures under {args.figure_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
