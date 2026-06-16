#!/usr/bin/env python3
"""Task 2 comparison tables and figures.

Requested pairwise comparisons:
- ACT-ABC vs ACT-ABC-Aug
- ACT-ABC vs ACT-ABC-Size-Matched
- ACT-ABC vs ACT-B
- ACT-ABC-Size-Matched vs ACT-B
- ACT-ABC-Size-Matched-Aug vs ACT-B-Aug

The B-vs-ABC validation comparisons are marked as different-split diagnostics;
they should not be used as the final generalization claim.
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
import pandas as pd


MODELS = ["act_B", "act_B_aug", "act_ABC", "act_ABC_aug", "act_ABC_size_matched", "act_ABC_size_matched_aug"]

DISPLAY = {
    "act_B": "ACT-B",
    "act_B_aug": "ACT-B Aug",
    "act_ABC": "ACT-ABC",
    "act_ABC_aug": "ACT-ABC Aug",
    "act_ABC_size_matched": "ACT-ABC Size-Matched",
    "act_ABC_size_matched_aug": "ACT-ABC Size-Matched Aug",
}

PAIRS = [
    ("ABC_vs_ABC_aug", "act_ABC", "act_ABC_aug", "same_ABC_validation_split"),
    ("ABC_vs_ABC_size_matched", "act_ABC", "act_ABC_size_matched", "same_ABC_validation_split"),
    ("ABC_vs_B", "act_ABC", "act_B", "different_validation_split"),
    ("ABC_size_matched_vs_B", "act_ABC_size_matched", "act_B", "different_validation_split"),
    ("ABC_size_matched_aug_vs_B_aug", "act_ABC_size_matched_aug", "act_B_aug", "different_validation_split"),
]


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


def save_latex(path: Path, rows: list[dict[str, Any]], float_format: str = "%.4f") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    df = pd.DataFrame(rows)
    path.write_text(df.to_latex(index=False, escape=True, float_format=float_format), encoding="utf-8")


def fval(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    try:
        return float(row[key])
    except Exception:
        return default


def model_run_dirs(main_rows: list[dict[str, str]], task2_rows: list[dict[str, str]]) -> dict[str, Path]:
    run_dirs: dict[str, Path] = {}
    for row in main_rows:
        if row["phase"] != "full_training":
            continue
        model = row["experiment"]
        checkpoint = Path(row["checkpoint"])
        run_dirs[model] = checkpoint.parent
    for row in task2_rows:
        run_dirs[row["experiment"]] = Path(row["run_dir"])
    return run_dirs


def load_metrics(run_dirs: dict[str, Path]) -> dict[str, list[dict[str, float]]]:
    metrics = {}
    for model, run_dir in run_dirs.items():
        path = run_dir / "metrics.csv"
        if not path.exists():
            continue
        rows = []
        for row in read_csv(path):
            rows.append({key: float(value) for key, value in row.items()})
        metrics[model] = rows
    return metrics


def finite_points(rows: list[dict[str, float]], key: str) -> tuple[np.ndarray, np.ndarray]:
    points = [(int(row["step"]), row[key]) for row in rows if math.isfinite(row[key])]
    return np.asarray([step for step, _ in points]), np.asarray([value for _, value in points], dtype=np.float64)


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    cumsum = np.cumsum(np.insert(values.astype(np.float64), 0, 0.0))
    out = np.empty_like(values, dtype=np.float64)
    for idx in range(len(values)):
        start = max(0, idx + 1 - window)
        out[idx] = (cumsum[idx + 1] - cumsum[start]) / (idx + 1 - start)
    return out


def title(ax: plt.Axes, text: str, fontsize: int = 10) -> None:
    ax.set_title(textwrap.fill(text, width=68), pad=14, fontsize=fontsize)


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)


def training_summary_rows(main_rows: list[dict[str, str]], selection_rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    selections = {row["experiment"]: row for row in selection_rows}
    for row in main_rows:
        if row["phase"] != "full_training" or row["experiment"] not in MODELS:
            continue
        model = row["experiment"]
        sel = selections.get(model, {})
        best = fval(sel, "best_val_action_l1")
        final_val = fval(row, "final_val_action_l1")
        final_train = fval(row, "final_train_action_l1")
        summaries[model] = {
            "experiment": model,
            "display_name": DISPLAY[model],
            "dataset": row["dataset"],
            "train_episode_spec": row["train_episodes"],
            "val_episode_spec": row["val_episodes"],
            "final_step": int(float(row["steps"])),
            "final_train_action_l1": final_train,
            "final_val_action_l1": final_val,
            "best_val_step": int(float(sel.get("best_val_step", math.nan))),
            "best_val_action_l1": best,
            "final_minus_best_val_action_l1": final_val - best,
            "selected_checkpoint_step": int(float(sel.get("selected_available_checkpoint_step", math.nan))),
            "selected_checkpoint_val_action_l1": fval(sel, "selected_available_checkpoint_val_action_l1"),
            "selected_checkpoint": sel.get("selected_available_checkpoint", ""),
            "validation_split_group": "B_val" if model.startswith("act_B") else "ABC_val",
        }
    return summaries


def convergence_rows(metrics: dict[str, list[dict[str, float]]], summaries: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for model, mrows in metrics.items():
        val_steps, val_l1 = finite_points(mrows, "val_action_l1")
        train_steps, train_l1 = finite_points(mrows, "train_action_l1")
        train_smooth = trailing_mean(train_l1, 1000)
        best = summaries[model]["best_val_action_l1"]
        threshold = best * 1.01
        first_within = math.nan
        for step, val in zip(val_steps, val_l1, strict=True):
            if val <= threshold:
                first_within = int(step)
                break
        post_best = val_l1[val_steps >= summaries[model]["best_val_step"]]
        rows[model] = {
            "experiment": model,
            "first_step_within_1pct_best_val": first_within,
            "num_val_points": int(len(val_steps)),
            "val_action_l1_std_all": float(np.std(val_l1)),
            "val_action_l1_std_after_best": float(np.std(post_best)) if len(post_best) else math.nan,
            "final_train_l1_1000step_mean": float(train_smooth[-1]) if len(train_smooth) else math.nan,
        }
    return rows


def one_row(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    if not rows:
        raise RuntimeError(f"Empty table: {path}")
    return rows[0]


def scalar_diagnostics(project_dir: Path) -> dict[str, dict[str, Any]]:
    diagnostics: dict[str, dict[str, Any]] = {}
    mapping = {
        "act_B": project_dir / "tables",
        "act_B_aug": project_dir / "tables",
        "act_ABC": project_dir / "tables" / "task2" / "act_ABC",
        "act_ABC_aug": project_dir / "tables" / "task2" / "act_ABC_aug",
        "act_ABC_size_matched": project_dir / "tables" / "task2" / "act_ABC_size_matched",
        "act_ABC_size_matched_aug": project_dir / "tables" / "task2" / "act_ABC_size_matched_aug",
    }
    for model, base in mapping.items():
        if model == "act_B":
            action = one_row(base / "env_B_action_summary.csv")
            chunk = one_row(base / "env_B_chunk_baseline_summary.csv")
            gripper = one_row(base / "env_B_gripper_summary.csv")
            visual_rows = read_csv(base / "env_B_visual_stats.csv")
            train_frames = 535403
            train_episodes = 212
            diagnostic_scope = "Task1 environment-B diagnostic split"
        elif model == "act_B_aug":
            action = one_row(base / "act_B_aug_action_summary.csv")
            chunk = one_row(base / "act_B_aug_chunk_baseline_summary.csv")
            gripper = one_row(base / "act_B_aug_gripper_summary.csv")
            visual_rows = read_csv(base / "act_B_aug_visual_stats.csv")
            train_frames = 535403
            train_episodes = 212
            diagnostic_scope = "Task1 environment-B augmented diagnostic split"
        else:
            action = one_row(base / "action_summary.csv")
            chunk = one_row(base / "chunk_baseline_summary.csv")
            gripper = one_row(base / "gripper_summary.csv")
            visual_rows = read_csv(base / "visual_stats.csv")
            selected = one_row(base / "selected_episode_summary.csv")
            train_frames = int(float(selected["num_frames"]))
            train_episodes = int(float(selected["num_episodes"]))
            diagnostic_scope = "Task2 selected training split"

        static_raw = visual_row(visual_rows, "static", prefer_augmented=False)
        gripper_raw = visual_row(visual_rows, "gripper", prefer_augmented=False)
        static_train_input = visual_row(visual_rows, "static", prefer_augmented=model.endswith("_aug"))
        gripper_train_input = visual_row(visual_rows, "gripper", prefer_augmented=model.endswith("_aug"))
        step_delta_key = "mean_step_delta_l2_first6" if "mean_step_delta_l2_first6" in action else "mean_step_delta_l2"
        diagnostics[model] = {
            "experiment": model,
            "display_name": DISPLAY[model],
            "train_frames": train_frames,
            "train_episodes": train_episodes,
            "diagnostic_scope": diagnostic_scope,
            "mean_action_l2_first6": fval(action, "mean_action_l2_first6"),
            "mean_step_delta_l2_first6": fval(action, step_delta_key),
            "gripper_close_fraction": fval(action, "gripper_close_fraction"),
            "gripper_switch_rate_per_1000_frames": fval(gripper, "switch_rate_per_1000_frames"),
            "chunk_mean_boundary_jump_l2_first6": fval(chunk, "mean_boundary_jump_l2_first6"),
            "chunk_q90_boundary_jump_l2_first6": fval(chunk, "q90_boundary_jump_l2_first6"),
            "raw_static_brightness_mean": fval(static_raw, "brightness_mean"),
            "raw_gripper_brightness_mean": fval(gripper_raw, "brightness_mean"),
            "train_input_static_brightness_mean": fval(static_train_input, "brightness_mean"),
            "train_input_gripper_brightness_mean": fval(gripper_train_input, "brightness_mean"),
            "raw_static_contrast_mean": fval(static_raw, "contrast_mean"),
            "raw_gripper_contrast_mean": fval(gripper_raw, "contrast_mean"),
            "train_input_static_contrast_mean": fval(static_train_input, "contrast_mean"),
            "train_input_gripper_contrast_mean": fval(gripper_train_input, "contrast_mean"),
        }
    return diagnostics


def visual_row(rows: list[dict[str, str]], camera: str, prefer_augmented: bool) -> dict[str, str]:
    candidates = [row for row in rows if row.get("camera") == camera]
    if not candidates:
        return {}
    if prefer_augmented:
        for row in candidates:
            if row.get("source") == "augmented" or row.get("environment") == "B_augmented":
                return row
    for row in candidates:
        if row.get("source") == "raw" or row.get("environment") in {"B", "B_raw_for_aug"}:
            return row
    return candidates[0]


def pairwise_rows(
    summaries: dict[str, dict[str, Any]],
    convergence: dict[str, dict[str, Any]],
    diagnostics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    metrics = [
        "best_val_action_l1",
        "final_val_action_l1",
        "final_minus_best_val_action_l1",
        "selected_checkpoint_val_action_l1",
        "first_step_within_1pct_best_val",
        "train_frames",
        "mean_action_l2_first6",
        "mean_step_delta_l2_first6",
        "chunk_mean_boundary_jump_l2_first6",
        "chunk_q90_boundary_jump_l2_first6",
        "gripper_switch_rate_per_1000_frames",
        "train_input_static_brightness_mean",
        "train_input_static_contrast_mean",
    ]
    rows = []
    combined = {}
    for model in MODELS:
        combined[model] = {}
        combined[model].update(summaries.get(model, {}))
        combined[model].update(convergence.get(model, {}))
        combined[model].update(diagnostics.get(model, {}))
    for comparison, left, right, split_note in PAIRS:
        for metric in metrics:
            left_value = combined[left].get(metric, math.nan)
            right_value = combined[right].get(metric, math.nan)
            delta = float(left_value) - float(right_value)
            rows.append(
                {
                    "comparison": comparison,
                    "left_experiment": left,
                    "right_experiment": right,
                    "metric": metric,
                    "left_value": left_value,
                    "right_value": right_value,
                    "delta_left_minus_right": delta,
                    "relative_delta_left_minus_right": delta / float(right_value) if float(right_value) != 0 else math.nan,
                    "lower_is_better": metric
                    in {
                        "best_val_action_l1",
                        "final_val_action_l1",
                        "final_minus_best_val_action_l1",
                        "selected_checkpoint_val_action_l1",
                        "first_step_within_1pct_best_val",
                        "mean_step_delta_l2_first6",
                        "chunk_mean_boundary_jump_l2_first6",
                        "chunk_q90_boundary_jump_l2_first6",
                        "gripper_switch_rate_per_1000_frames",
                    },
                    "validation_comparability": split_note,
                }
            )
    return rows


def compact_pairwise_loss_rows(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {
        "best_val_action_l1",
        "final_val_action_l1",
        "final_minus_best_val_action_l1",
        "selected_checkpoint_val_action_l1",
        "first_step_within_1pct_best_val",
    }
    return [row for row in pair_rows if row["metric"] in wanted]


def key_finding_rows(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for row in pair_rows:
        by_pair.setdefault(str(row["comparison"]), {})[str(row["metric"])] = row
    rows = []
    for comparison, left, right, split_note in PAIRS:
        metrics = by_pair[comparison]
        best_delta = float(metrics["best_val_action_l1"]["delta_left_minus_right"])
        selected_delta = float(metrics["selected_checkpoint_val_action_l1"]["delta_left_minus_right"])
        overfit_delta = float(metrics["final_minus_best_val_action_l1"]["delta_left_minus_right"])
        train_ratio = float(metrics["train_frames"]["left_value"]) / float(metrics["train_frames"]["right_value"])
        action_delta = float(metrics["mean_step_delta_l2_first6"]["delta_left_minus_right"])
        chunk_delta = float(metrics["chunk_q90_boundary_jump_l2_first6"]["delta_left_minus_right"])
        visual_delta = float(metrics["train_input_static_brightness_mean"]["delta_left_minus_right"])
        if split_note == "same_ABC_validation_split":
            loss_interpretation = (
                f"{left} has lower best validation Action L1"
                if best_delta < 0
                else f"{right} has lower best validation Action L1"
            )
        else:
            loss_interpretation = "different validation split; use loss deltas as diagnostics only"
        rows.append(
            {
                "comparison": comparison,
                "left_experiment": left,
                "right_experiment": right,
                "validation_comparability": split_note,
                "best_val_action_l1_delta_left_minus_right": best_delta,
                "selected_checkpoint_val_delta_left_minus_right": selected_delta,
                "overfit_gap_delta_left_minus_right": overfit_delta,
                "train_frame_ratio_left_over_right": train_ratio,
                "mean_step_delta_l2_delta_left_minus_right": action_delta,
                "chunk_q90_boundary_jump_delta_left_minus_right": chunk_delta,
                "train_input_static_brightness_delta_left_minus_right": visual_delta,
                "interpretation": loss_interpretation,
            }
        )
    return rows


def plot_pairwise_val_bars(summaries: dict[str, dict[str, Any]], figure_dir: Path) -> None:
    labels = []
    best = []
    final = []
    for model in ["act_B", "act_B_aug", "act_ABC", "act_ABC_aug", "act_ABC_size_matched", "act_ABC_size_matched_aug"]:
        labels.append(DISPLAY[model])
        best.append(summaries[model]["best_val_action_l1"])
        final.append(summaries[model]["final_val_action_l1"])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=180, constrained_layout=True)
    ax.bar(x - 0.18, best, width=0.36, color="#4c78a8", label="Best validation Action L1")
    ax.bar(x + 0.18, final, width=0.36, color="#f58518", label="Final validation Action L1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=24, ha="right")
    ax.set_ylabel("Action L1")
    title(ax, "Best vs Final Validation Action L1")
    ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=False, fontsize=8)
    ax.text(0.01, 0.98, "B and ABC bars use different validation splits.", transform=ax.transAxes, va="top", fontsize=8)
    save_fig(fig, figure_dir / "best_final_val_action_l1.png")


def plot_overfit_convergence(summaries: dict[str, dict[str, Any]], convergence: dict[str, dict[str, Any]], figure_dir: Path) -> None:
    models = ["act_B", "act_B_aug", "act_ABC", "act_ABC_aug", "act_ABC_size_matched", "act_ABC_size_matched_aug"]
    labels = [DISPLAY[m] for m in models]
    overfit = [summaries[m]["final_minus_best_val_action_l1"] for m in models]
    first_steps = [convergence[m]["first_step_within_1pct_best_val"] for m in models]
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), dpi=180, constrained_layout=True)
    axes[0].bar(x, overfit, color="#b23a48", edgecolor="#222222", linewidth=0.35)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=28, ha="right")
    axes[0].set_ylabel("Final - best validation Action L1")
    title(axes[0], "Overfitting Indicator", fontsize=9)
    axes[1].bar(x, first_steps, color="#72b7b2", edgecolor="#222222", linewidth=0.35)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=28, ha="right")
    axes[1].set_ylabel("Training step")
    title(axes[1], "First Step Within 1% of Best Validation", fontsize=9)
    for ax in axes:
        ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    fig.suptitle("Overfitting and Convergence Speed", fontsize=11)
    save_fig(fig, figure_dir / "overfit_convergence_summary.png")


def plot_pairwise_loss_panels(metrics: dict[str, list[dict[str, float]]], figure_dir: Path) -> None:
    fig, axes = plt.subplots(len(PAIRS), 1, figsize=(9.5, 13.0), dpi=180, constrained_layout=True)
    colors = {"left": "#4c78a8", "right": "#f58518"}
    for ax, (comparison, left, right, split_note) in zip(axes, PAIRS, strict=True):
        for side, model in [("left", left), ("right", right)]:
            rows = metrics[model]
            val_steps, val_l1 = finite_points(rows, "val_action_l1")
            train_steps, train_l1 = finite_points(rows, "train_action_l1")
            train_smooth = trailing_mean(train_l1, 1000)
            ax.plot(train_steps, train_smooth, color=colors[side], linewidth=0.9, alpha=0.38, linestyle="--")
            ax.plot(val_steps, val_l1, color=colors[side], linewidth=1.35, label=f"{DISPLAY[model]} val")
        title(ax, f"{comparison.replace('_', ' ')} ({split_note})", fontsize=9)
        ax.set_ylabel("Action L1")
        ax.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.7)
        ax.legend(frameon=False, fontsize=8, loc="best")
    axes[-1].set_xlabel("Training step")
    fig.suptitle("Requested Pairwise Training Dynamics", fontsize=12)
    save_fig(fig, figure_dir / "requested_pairwise_loss_curves.png")


def plot_data_action_visual(diagnostics: dict[str, dict[str, Any]], figure_dir: Path) -> None:
    models = ["act_B", "act_B_aug", "act_ABC", "act_ABC_aug", "act_ABC_size_matched", "act_ABC_size_matched_aug"]
    labels = [DISPLAY[m] for m in models]
    metrics = [
        ("train_frames", "Train Frames"),
        ("mean_action_l2_first6", "Mean Action L2"),
        ("mean_step_delta_l2_first6", "Mean Step Delta L2"),
        ("train_input_static_brightness_mean", "Train-Input Static Brightness"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), dpi=180, constrained_layout=True)
    for ax, (key, label) in zip(axes.ravel(), metrics, strict=True):
        vals = [diagnostics[m][key] for m in models]
        ax.bar(np.arange(len(models)), vals, color="#4c78a8", edgecolor="#222222", linewidth=0.35)
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(labels, rotation=28, ha="right")
        title(ax, label, fontsize=9)
        ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    fig.suptitle("Data Scale, Action, and Visual Diagnostics", fontsize=12)
    save_fig(fig, figure_dir / "data_action_visual_summary.png")


def plot_pairwise_delta_heatmap(pair_rows: list[dict[str, Any]], figure_dir: Path) -> None:
    metrics = [
        "best_val_action_l1",
        "final_minus_best_val_action_l1",
        "train_frames",
        "mean_action_l2_first6",
        "mean_step_delta_l2_first6",
        "chunk_q90_boundary_jump_l2_first6",
        "gripper_switch_rate_per_1000_frames",
        "train_input_static_brightness_mean",
    ]
    comparisons = [pair[0] for pair in PAIRS]
    grid = np.full((len(comparisons), len(metrics)), np.nan, dtype=np.float64)
    by_key = {(row["comparison"], row["metric"]): row for row in pair_rows}
    for i, comparison in enumerate(comparisons):
        for j, metric in enumerate(metrics):
            row = by_key[(comparison, metric)]
            grid[i, j] = float(row["relative_delta_left_minus_right"])
    clipped = np.clip(grid, -0.25, 0.25)
    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=180, constrained_layout=True)
    im = ax.imshow(clipped, aspect="auto", cmap="coolwarm", vmin=-0.25, vmax=0.25)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([metric.replace("_", "\n") for metric in metrics], fontsize=7)
    ax.set_yticks(np.arange(len(comparisons)))
    ax.set_yticklabels([label.replace("_", " ") for label in comparisons], fontsize=8)
    for i in range(len(comparisons)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{grid[i, j]:+.2%}", ha="center", va="center", fontsize=6)
    title(ax, "Pairwise Relative Delta Heatmap (Left - Right)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Relative delta, clipped to +/-25%")
    save_fig(fig, figure_dir / "pairwise_relative_delta_heatmap.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path("project"))
    args = parser.parse_args()

    project_dir = args.project_dir
    table_dir = project_dir / "tables" / "task2" / "comparisons"
    figure_dir = project_dir / "figures" / "task2" / "comparisons"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    main_rows = read_csv(project_dir / "tables" / "main_training_results.csv")
    task2_rows = read_csv(project_dir / "tables" / "task2_full_training_summary.csv")
    selection_rows = read_csv(project_dir / "tables" / "model_selection_checkpoints.csv")

    run_dirs = model_run_dirs(main_rows, task2_rows)
    metrics = load_metrics(run_dirs)
    summaries = training_summary_rows(main_rows, selection_rows)
    convergence = convergence_rows(metrics, summaries)
    diagnostics = scalar_diagnostics(project_dir)
    pair_rows = pairwise_rows(summaries, convergence, diagnostics)
    key_rows = key_finding_rows(pair_rows)

    model_rows = []
    for model in MODELS:
        row = {}
        row.update(summaries[model])
        row.update(convergence[model])
        row.update(diagnostics[model])
        model_rows.append(row)

    write_rows(table_dir / "comparison_model_summary.csv", model_rows)
    write_rows(table_dir / "pairwise_comparison_metrics.csv", pair_rows)
    write_rows(table_dir / "pairwise_loss_overfit_convergence.csv", compact_pairwise_loss_rows(pair_rows))
    write_rows(table_dir / "pairwise_key_findings.csv", key_rows)
    write_rows(project_dir / "tables" / "task2_pairwise_effects.csv", key_rows)
    save_latex(table_dir / "comparison_model_summary.tex", model_rows)
    save_latex(table_dir / "pairwise_loss_overfit_convergence.tex", compact_pairwise_loss_rows(pair_rows))
    save_latex(table_dir / "pairwise_key_findings.tex", key_rows)

    plot_pairwise_val_bars(summaries, figure_dir)
    plot_overfit_convergence(summaries, convergence, figure_dir)
    plot_pairwise_loss_panels(metrics, figure_dir)
    plot_data_action_visual(diagnostics, figure_dir)
    plot_pairwise_delta_heatmap(pair_rows, figure_dir)

    manifest = [
        {"figure": str(path), "kind": path.stem, "note": "Task 2 requested cross-model comparison"}
        for path in sorted(figure_dir.glob("*.png"))
    ]
    write_rows(table_dir / "comparison_figure_manifest.csv", manifest)
    print(f"Wrote {len(model_rows)} model summary rows and {len(pair_rows)} pairwise metric rows")
    print(f"Wrote {len(manifest)} comparison figures to {figure_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
