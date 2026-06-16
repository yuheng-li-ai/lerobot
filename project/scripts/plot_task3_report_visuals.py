#!/usr/bin/env python
"""Generate report-ready Task 3 visualizations.

The script only reads existing Task 3 CSV outputs. It creates categorized
figures with wide layouts, wrapped labels, and tight bounding boxes so labels
and headers are not clipped in the report.
"""

from __future__ import annotations

import argparse
import csv
import os
import textwrap
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np


ACTION_DIMS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
MODEL_ORDER = [
    "act_B",
    "act_B_aug",
    "act_ABC",
    "act_ABC_aug",
    "act_ABC_size_matched",
    "act_ABC_size_matched_aug",
]
DISPLAY_NAMES = {
    "act_B": "ACT-B",
    "act_B_aug": "ACT-B\nAug",
    "act_ABC": "ACT-ABC",
    "act_ABC_aug": "ACT-ABC\nAug",
    "act_ABC_size_matched": "ACT-ABC\nSize-Matched",
    "act_ABC_size_matched_aug": "ACT-ABC\nSize-Matched\nAug",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ordered(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_name = {row["experiment"]: row for row in rows}
    return [by_name[name] for name in MODEL_ORDER if name in by_name]


def f(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "chunking": root / "chunking",
        "action_error": root / "action_error",
        "visual_shift": root / "visual_shift",
        "summary": root / "summary",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def label_for(name: str) -> str:
    return DISPLAY_NAMES.get(name, "\n".join(textwrap.wrap(name, width=12)))


def heatmap(
    values: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    path: Path,
    title: str,
    cbar_label: str,
    *,
    annotate: bool = False,
    figsize: tuple[float, float] = (12, 6),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, pad=16)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(cbar_label)
    if annotate:
        finite = values[np.isfinite(values)]
        threshold = float(finite.mean()) if finite.size else 0.0
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                color = "white" if values[i, j] > threshold else "black"
                ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", color=color, fontsize=8)
    save(fig, path)


def plot_chunk_horizon_heatmap(horizon_rows: list[dict[str, str]], out: Path) -> Path:
    by_model: dict[str, dict[int, float]] = {}
    for row in horizon_rows:
        if row["checkpoint_mode"] != "selected":
            continue
        by_model.setdefault(row["experiment"], {})[int(row["horizon_step"])] = float(row["action_l1"])
    models = [m for m in MODEL_ORDER if m in by_model]
    horizons = list(range(100))
    values = np.array([[by_model[m].get(h, np.nan) for h in horizons] for m in models])
    fig, ax = plt.subplots(figsize=(15.5, 5.8))
    im = ax.imshow(values, aspect="auto", cmap="magma")
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels([label_for(m) for m in models], fontsize=9)
    tick_positions = list(range(0, 100, 10)) + [99]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(t) for t in tick_positions])
    ax.set_xlabel("Position inside predicted ACT action chunk")
    ax.set_ylabel("Model")
    ax.set_title("D Action L1 Across the 100-Step ACT Chunk Horizon", pad=16)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Action L1 on D")
    save(fig, out / "chunk_horizon_error_heatmap.png")
    return out / "chunk_horizon_error_heatmap.png"


def plot_action_dim_heatmap(result_rows: list[dict[str, str]], out: Path) -> Path:
    rows = ordered(result_rows)
    values = np.array([[f(row, f"l1_{dim}") for dim in ACTION_DIMS] for row in rows])
    path = out / "action_dimension_error_heatmap_D.png"
    heatmap(
        values,
        [label_for(row["experiment"]) for row in rows],
        ACTION_DIMS,
        path,
        "Zero-Shot D Action Error by Dimension",
        "L1 error",
        annotate=True,
        figsize=(11.5, 6.2),
    )
    return path


def plot_pose_gripper_breakdown(result_rows: list[dict[str, str]], out: Path) -> Path:
    rows = ordered(result_rows)
    labels = [label_for(row["experiment"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.38
    pose = [f(row, "action_l1_first6") for row in rows]
    grip = [f(row, "gripper_l1") for row in rows]
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    ax.bar(x - width / 2, pose, width, label="First 6D pose action", color="#4C78A8")
    ax.bar(x + width / 2, grip, width, label="Gripper action", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("L1 error on D")
    ax.set_title("Pose vs Gripper Error Under D Visual Shift", pad=16)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    path = out / "pose_vs_gripper_error_D.png"
    save(fig, path)
    return path


def plot_within_vs_boundary(robust_rows: list[dict[str, str]], out: Path) -> Path:
    rows = ordered(robust_rows)
    labels = [label_for(row["experiment"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.38
    step = [f(row, "pred_step_delta_l2") for row in rows]
    boundary = [f(row, "pred_boundary_jump_l2") for row in rows]
    fig, ax = plt.subplots(figsize=(12.8, 5.4))
    ax.bar(x - width / 2, step, width, label="Within-chunk step delta", color="#54A24B")
    ax.bar(x + width / 2, boundary, width, label="Chunk boundary jump", color="#E45756")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("L2 over first 6 action dims")
    ax.set_title("ACT Is Smooth Inside Chunks but Can Jump at Refresh Boundaries", pad=16)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    path = out / "within_chunk_vs_boundary_jump_D.png"
    save(fig, path)
    return path


def plot_boundary_tail_risk(result_rows: list[dict[str, str]], out: Path) -> Path:
    rows = ordered(result_rows)
    labels = [label_for(row["experiment"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.28
    mean = [f(row, "pred_mean_boundary_jump_l2_first6") for row in rows]
    q90 = [f(row, "pred_q90_boundary_jump_l2_first6") for row in rows]
    q99 = [f(row, "pred_q99_boundary_jump_l2_first6") for row in rows]
    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.bar(x - width, mean, width, label="Mean", color="#4C78A8")
    ax.bar(x, q90, width, label="q90", color="#F58518")
    ax.bar(x + width, q99, width, label="q99", color="#E45756")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Boundary jump L2")
    ax.set_title("Chunk Boundary Tail Risk on D", pad=16)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3)
    path = out / "chunk_boundary_tail_risk_D.png"
    save(fig, path)
    return path


def plot_oversmoothing(robust_rows: list[dict[str, str]], out: Path) -> Path:
    rows = ordered(robust_rows)
    values = [1.0 - f(row, "smoothness_ratio_pred_over_gt") for row in rows]
    labels = [label_for(row["experiment"]) for row in rows]
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.bar(np.arange(len(rows)), values, color="#72B7B2")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("1 - predicted step delta / D ground-truth step delta")
    ax.set_title("Oversmoothing Index Induced by ACT Chunking", pad=16)
    ax.grid(axis="y", alpha=0.25)
    path = out / "oversmoothing_index_D.png"
    save(fig, path)
    return path


def plot_horizon_degradation(robust_rows: list[dict[str, str]], out: Path) -> Path:
    rows = ordered(robust_rows)
    labels = [label_for(row["experiment"]) for row in rows]
    values = [f(row, "chunk_horizon_degradation_last10_minus_first10") for row in rows]
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.bar(np.arange(len(rows)), values, color="#B279A2")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Last 10 chunk steps L1 - first 10 chunk steps L1")
    ax.set_title("Late-Horizon Degradation Within ACT Chunks", pad=16)
    ax.grid(axis="y", alpha=0.25)
    path = out / "chunk_horizon_degradation_D.png"
    save(fig, path)
    return path


def plot_visual_shift_vs_error(robust_rows: list[dict[str, str]], out: Path) -> Path:
    rows = ordered(robust_rows)
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    for row in rows:
        name = row["experiment"]
        marker = "o" if row["train_visual_env"] == "B" else "s"
        color = "#F58518" if "aug" in name else "#4C78A8"
        ax.scatter(
            f(row, "mean_rgb_l2_to_D"),
            f(row, "action_l1_D"),
            s=105,
            marker=marker,
            color=color,
            edgecolor="black",
            linewidth=0.6,
        )
        ax.annotate(label_for(name).replace("\n", " "), (f(row, "mean_rgb_l2_to_D"), f(row, "action_l1_D")), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Training visual distribution distance to D (mean RGB L2)")
    ax.set_ylabel("Zero-shot D Action L1")
    ax.set_title("Visual Shift Explains Much of the D Action Error Gap", pad=16)
    ax.grid(alpha=0.25)
    path = out / "visual_shift_vs_action_error_D.png"
    save(fig, path)
    return path


def plot_visual_shift_radar_like(shift_rows: list[dict[str, str]], out: Path) -> Path:
    rows = [row for row in shift_rows if row["source_env"] in {"A", "B", "C", "ABC"}]
    labels = [row["source_env"] for row in rows]
    metrics = ["mean_rgb_l2_to_D", "mean_brightness_abs_to_D", "mean_contrast_abs_to_D"]
    metric_labels = ["RGB mean L2", "Brightness gap", "Contrast gap"]
    values = np.array([[float(row[m]) for m in metrics] for row in rows])
    # Normalize per metric for visual comparability without hiding raw CSV values.
    denom = values.max(axis=0)
    norm = values / np.where(denom == 0, 1, denom)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(metrics))
    width = 0.18
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    for i, label in enumerate(labels):
        ax.bar(x + (i - 1.5) * width, norm[i], width, label=label, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Normalized distance to D")
    ax.set_title("Visual Shift to D by Training Environment", pad=16)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Source")
    path = out / "visual_shift_distance_summary_D.png"
    save(fig, path)
    return path


def write_summary_table(robust_rows: list[dict[str, str]], table_dir: Path) -> Path:
    rows = []
    for row in ordered(robust_rows):
        oversmoothing = 1.0 - f(row, "smoothness_ratio_pred_over_gt")
        rows.append(
            {
                "experiment": row["experiment"],
                "train_visual_env": row["train_visual_env"],
                "zero_shot_D_action_l1": f(row, "action_l1_D"),
                "visual_rgb_l2_to_D": f(row, "mean_rgb_l2_to_D"),
                "oversmoothing_index": oversmoothing,
                "boundary_amplification": f(row, "boundary_amplification_pred_over_gt"),
                "late_horizon_degradation": f(row, "chunk_horizon_degradation_last10_minus_first10"),
            }
        )
    path = table_dir / "task3_report_visual_summary.csv"
    write_rows(path, rows)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task3-table-dir", type=Path, default=Path("project/tables/task3"))
    parser.add_argument("--figure-root", type=Path, default=Path("project/figures/task3_report"))
    parser.add_argument("--table-dir", type=Path, default=Path("project/tables/task3_report"))
    args = parser.parse_args()

    dirs = ensure_dirs(args.figure_root)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    result_rows = read_rows(args.task3_table_dir / "zero_shot_D_results.csv")
    horizon_rows = read_rows(args.task3_table_dir / "zero_shot_D_chunk_horizon.csv")
    robust_rows = read_rows(args.task3_table_dir / "chunk_visual_shift_robustness.csv")
    shift_rows = read_rows(args.task3_table_dir / "visual_shift_to_D_task3.csv")

    outputs = [
        ("chunking", plot_chunk_horizon_heatmap(horizon_rows, dirs["chunking"]), "D action error by horizon step inside ACT chunks."),
        ("chunking", plot_within_vs_boundary(robust_rows, dirs["chunking"]), "Within-chunk smoothness versus chunk refresh boundary jumps."),
        ("chunking", plot_boundary_tail_risk(result_rows, dirs["chunking"]), "Mean/q90/q99 boundary jump risk."),
        ("chunking", plot_oversmoothing(robust_rows, dirs["chunking"]), "Oversmoothing index relative to D demonstrations."),
        ("chunking", plot_horizon_degradation(robust_rows, dirs["chunking"]), "Late-horizon error degradation within each chunk."),
        ("action_error", plot_action_dim_heatmap(result_rows, dirs["action_error"]), "Per-action-dimension D error heatmap."),
        ("action_error", plot_pose_gripper_breakdown(result_rows, dirs["action_error"]), "Pose and gripper error breakdown."),
        ("visual_shift", plot_visual_shift_vs_error(robust_rows, dirs["visual_shift"]), "Visual shift versus zero-shot action error."),
        ("visual_shift", plot_visual_shift_radar_like(shift_rows, dirs["visual_shift"]), "Normalized visual distance to D."),
    ]
    summary_table = write_summary_table(robust_rows, args.table_dir)
    manifest_rows = [
        {"category": category, "artifact": str(path), "description": description}
        for category, path, description in outputs
    ]
    manifest_rows.append(
        {
            "category": "summary",
            "artifact": str(summary_table),
            "description": "Compact report summary table with error, visual shift, oversmoothing, and boundary amplification.",
        }
    )
    write_rows(args.table_dir / "task3_report_visual_manifest.csv", manifest_rows)
    for row in manifest_rows:
        print(f"{row['category']}: {row['artifact']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
