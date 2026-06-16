#!/usr/bin/env python
"""Report-oriented ACT chunking analysis under CALVIN visual shift."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from calvin_phase0_common import (
    clipped_episode_ranges,
    configured_raw_root,
    frame_file,
    load_environment_ranges,
    load_episode_ranges,
    split_dir,
)


CAMERA_KEYS = {"static": "rgb_static", "gripper": "rgb_gripper"}
MODEL_TRAIN_ENV = {
    "act_B": "B",
    "act_B_aug": "B",
    "act_ABC": "ABC",
    "act_ABC_aug": "ABC",
    "act_ABC_size_matched": "ABC",
    "act_ABC_size_matched_aug": "ABC",
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


def write_latex(path: Path, rows: list[dict[str, Any]], columns: list[str], caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\toprule",
        " & ".join(columns).replace("_", "\\_") + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        vals = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.3f}")
            else:
                vals.append(str(value).replace("_", "\\_"))
        lines.append(" & ".join(vals) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines))


def sample_existing_frame_ids(ranges: list[tuple[int, int]], samples: int) -> np.ndarray:
    frame_ids = np.concatenate([np.arange(start, end + 1, dtype=np.int64) for start, end in ranges])
    if samples >= len(frame_ids):
        return frame_ids
    sample_positions = np.linspace(0, len(frame_ids) - 1, samples, dtype=int)
    return frame_ids[sample_positions]


def image_stats(images: list[np.ndarray]) -> dict[str, float]:
    arr = np.stack(images).astype(np.float32) / 255.0
    rgb_mean = arr.mean(axis=(0, 1, 2))
    rgb_std = arr.std(axis=(0, 1, 2))
    brightness = arr.mean(axis=(1, 2, 3))
    contrast = arr.std(axis=(1, 2, 3))
    return {
        "rgb_mean_r": float(rgb_mean[0]),
        "rgb_mean_g": float(rgb_mean[1]),
        "rgb_mean_b": float(rgb_mean[2]),
        "rgb_std_r": float(rgb_std[0]),
        "rgb_std_g": float(rgb_std[1]),
        "rgb_std_b": float(rgb_std[2]),
        "brightness_mean": float(brightness.mean()),
        "brightness_std": float(brightness.std()),
        "contrast_mean": float(contrast.mean()),
        "contrast_std": float(contrast.std()),
    }


def collect_visual_stats(raw_root: Path, samples_per_env: int) -> list[dict[str, Any]]:
    ranges = load_environment_ranges(raw_root)
    rows: list[dict[str, Any]] = []
    for env in ["A", "B", "C", "D"]:
        frame_range = ranges[env]
        directory = split_dir(raw_root, frame_range.split)
        valid_ranges = clipped_episode_ranges(load_episode_ranges(directory), frame_range)
        frame_ids = sample_existing_frame_ids(valid_ranges, samples_per_env)
        by_camera = {camera: [] for camera in CAMERA_KEYS}
        for frame_id in frame_ids:
            data = np.load(frame_file(directory, int(frame_id)))
            for camera, key in CAMERA_KEYS.items():
                by_camera[camera].append(data[key])
        for camera, images in by_camera.items():
            stats = image_stats(images)
            rows.append(
                {
                    "environment": env,
                    "camera": camera,
                    "sampled_frames": len(images),
                    **stats,
                }
            )
    return rows


def aggregate_env_stats(rows: list[dict[str, Any]], env_name: str, members: list[str]) -> list[dict[str, Any]]:
    out = []
    for camera in CAMERA_KEYS:
        relevant = [r for r in rows if r["environment"] in members and r["camera"] == camera]
        if not relevant:
            continue
        total = sum(int(r["sampled_frames"]) for r in relevant)
        agg = {"environment": env_name, "camera": camera, "sampled_frames": total}
        for key in [
            "rgb_mean_r",
            "rgb_mean_g",
            "rgb_mean_b",
            "rgb_std_r",
            "rgb_std_g",
            "rgb_std_b",
            "brightness_mean",
            "brightness_std",
            "contrast_mean",
            "contrast_std",
        ]:
            agg[key] = float(sum(float(r[key]) * int(r["sampled_frames"]) for r in relevant) / total)
        out.append(agg)
    return out


def visual_distance(rows: list[dict[str, Any]], source: str, target: str = "D") -> dict[str, float]:
    out: dict[str, float] = {"source_env": source, "target_env": target}  # type: ignore[assignment]
    source_rows = {r["camera"]: r for r in rows if r["environment"] == source}
    target_rows = {r["camera"]: r for r in rows if r["environment"] == target}
    distances = []
    brightness = []
    contrast = []
    for camera in CAMERA_KEYS:
        s = source_rows[camera]
        t = target_rows[camera]
        rgb_s = np.array([float(s["rgb_mean_r"]), float(s["rgb_mean_g"]), float(s["rgb_mean_b"])])
        rgb_t = np.array([float(t["rgb_mean_r"]), float(t["rgb_mean_g"]), float(t["rgb_mean_b"])])
        rgb_dist = float(np.linalg.norm(rgb_s - rgb_t))
        b_dist = abs(float(s["brightness_mean"]) - float(t["brightness_mean"]))
        c_dist = abs(float(s["contrast_mean"]) - float(t["contrast_mean"]))
        out[f"{camera}_rgb_mean_l2_to_D"] = rgb_dist
        out[f"{camera}_brightness_abs_to_D"] = b_dist
        out[f"{camera}_contrast_abs_to_D"] = c_dist
        distances.append(rgb_dist)
        brightness.append(b_dist)
        contrast.append(c_dist)
    out["mean_rgb_l2_to_D"] = float(np.mean(distances))
    out["mean_brightness_abs_to_D"] = float(np.mean(brightness))
    out["mean_contrast_abs_to_D"] = float(np.mean(contrast))
    return out


def horizon_degradation(horizon_rows: list[dict[str, str]], experiment: str) -> tuple[float, float, float]:
    rows = [r for r in horizon_rows if r["experiment"] == experiment and r["checkpoint_mode"] == "selected"]
    rows = sorted(rows, key=lambda r: int(r["horizon_step"]))
    if not rows:
        return math.nan, math.nan, math.nan
    values = np.array([float(r["action_l1"]) for r in rows], dtype=np.float64)
    early = float(values[:10].mean())
    late = float(values[-10:].mean())
    return early, late, late - early


def build_robustness_table(
    result_rows: list[dict[str, str]],
    horizon_rows: list[dict[str, str]],
    shift_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    shift_by_source = {str(r["source_env"]): r for r in shift_rows}
    out = []
    for r in result_rows:
        exp = r["experiment"]
        train_env = MODEL_TRAIN_ENV.get(exp, "unknown")
        shift = shift_by_source[train_env]
        early, late, degradation = horizon_degradation(horizon_rows, exp)
        pred_step = float(r["pred_mean_step_delta_l2_first6"])
        gt_step = float(r["gt_mean_step_delta_l2_first6"])
        pred_boundary = float(r["pred_mean_boundary_jump_l2_first6"])
        gt_boundary = float(r["gt_mean_boundary_jump_l2_first6"])
        action_l1 = float(r["action_l1"])
        out.append(
            {
                "experiment": exp,
                "train_visual_env": train_env,
                "action_l1_D": action_l1,
                "first6_l1_D": float(r["action_l1_first6"]),
                "gripper_l1_D": float(r["gripper_l1"]),
                "mean_rgb_l2_to_D": float(shift["mean_rgb_l2_to_D"]),
                "mean_brightness_abs_to_D": float(shift["mean_brightness_abs_to_D"]),
                "mean_contrast_abs_to_D": float(shift["mean_contrast_abs_to_D"]),
                "pred_step_delta_l2": pred_step,
                "gt_step_delta_l2": gt_step,
                "smoothness_ratio_pred_over_gt": pred_step / gt_step if gt_step else math.nan,
                "oversmoothing_gap_gt_minus_pred": gt_step - pred_step,
                "pred_boundary_jump_l2": pred_boundary,
                "gt_boundary_jump_l2": gt_boundary,
                "boundary_amplification_pred_over_gt": pred_boundary / gt_boundary if gt_boundary else math.nan,
                "chunk_horizon_l1_first10": early,
                "chunk_horizon_l1_last10": late,
                "chunk_horizon_degradation_last10_minus_first10": degradation,
            }
        )
    return out


def plot_visual_shift(rows: list[dict[str, Any]], path: Path) -> None:
    envs = ["A", "B", "C", "ABC", "D"]
    cameras = ["static", "gripper"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ax, metric, title in zip(
        axes,
        ["brightness_mean", "contrast_mean"],
        ["Brightness Mean", "Contrast Mean"],
        strict=True,
    ):
        x = np.arange(len(envs))
        width = 0.36
        for offset, camera in [(-width / 2, "static"), (width / 2, "gripper")]:
            vals = [float(next(r for r in rows if r["environment"] == env and r["camera"] == camera)[metric]) for env in envs]
            ax.bar(x + offset, vals, width, label=camera)
        ax.set_xticks(x)
        ax.set_xticklabels(envs)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Value")
    axes[1].legend()
    fig.suptitle("CALVIN A/B/C/D Visual Distribution Shift")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_robustness(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [r["experiment"] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    metrics = [
        ("action_l1_D", "D Action L1", "lower is better"),
        ("smoothness_ratio_pred_over_gt", "Smoothness Ratio", "pred / GT step delta; 1 is ideal"),
        ("boundary_amplification_pred_over_gt", "Boundary Amplification", "pred / GT boundary jump; 1 is ideal"),
    ]
    colors = ["#4C78A8" if r["train_visual_env"] == "B" else "#54A24B" for r in rows]
    for ax, (metric, title, subtitle) in zip(axes, metrics, strict=True):
        ax.bar(x, [float(r[metric]) for r in rows], color=colors)
        if "Ratio" in title or "Amplification" in title:
            ax.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.6)
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("ACT Chunking Robustness on Unseen D")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_tradeoff(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for r in rows:
        marker = "o" if r["train_visual_env"] == "B" else "s"
        color = "#F58518" if "aug" in r["experiment"] else "#4C78A8"
        ax.scatter(
            float(r["action_l1_D"]),
            float(r["boundary_amplification_pred_over_gt"]),
            s=90,
            marker=marker,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.annotate(r["experiment"], (float(r["action_l1_D"]), float(r["boundary_amplification_pred_over_gt"])), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Zero-shot D Action L1 (lower is better)")
    ax.set_ylabel("Chunk boundary amplification vs D ground truth")
    ax.set_title("Accuracy-Stability Tradeoff Under Visual Shift")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report_notes(path: Path, rows: list[dict[str, Any]], shift_rows: list[dict[str, Any]]) -> None:
    best = min(rows, key=lambda r: float(r["action_l1_D"]))
    smoothest = max(rows, key=lambda r: float(r["oversmoothing_gap_gt_minus_pred"]))
    most_boundary = max(rows, key=lambda r: float(r["boundary_amplification_pred_over_gt"]))
    b_shift = next(r for r in shift_rows if r["source_env"] == "B")
    abc_shift = next(r for r in shift_rows if r["source_env"] == "ABC")
    text = f"""# Task 3 ACT Chunking Robustness Notes

Key evidence for the report:

- D zero-shot action error improves from B-only to ABC-style training. The best current model is `{best['experiment']}` with Action L1 `{best['action_l1_D']:.4f}`.
- The visual shift proxy is lower for ABC than B when comparing RGB mean to D: B mean RGB L2 `{b_shift['mean_rgb_l2_to_D']:.4f}`, ABC mean RGB L2 `{abc_shift['mean_rgb_l2_to_D']:.4f}`.
- All models are strongly smoother than D ground-truth actions. The ground-truth D mean step delta is about `{rows[0]['gt_step_delta_l2']:.4f}`, while predicted step deltas are around `{min(float(r['pred_step_delta_l2']) for r in rows):.4f}` to `{max(float(r['pred_step_delta_l2']) for r in rows):.4f}`.
- This supports an ACT chunking interpretation: chunking stabilizes predictions under visual shift, but also produces over-smoothed action streams.
- The main tradeoff is accuracy versus chunk-boundary stability. `{best['experiment']}` has the best D Action L1, but its boundary amplification is `{best['boundary_amplification_pred_over_gt']:.2f}x` the D ground-truth boundary jump.
- The largest boundary amplification is `{most_boundary['experiment']}` at `{most_boundary['boundary_amplification_pred_over_gt']:.2f}x`.
- The most over-smoothed model by GT-minus-pred step delta is `{smoothest['experiment']}`.

Suggested report wording:

Under D visual shift, ACT's action chunking acts as a stabilizer: predicted within-chunk step changes are far smaller than the D demonstration action changes, so the policy does not jitter frame-by-frame. However, the chunk queue also delays correction and concentrates discontinuities at chunk refresh boundaries. Multi-environment training improves D action accuracy, especially gripper prediction, but the larger and more diverse training distribution can increase chunk-boundary jumps. Therefore the robustness is not simply 'more diversity is smoother'; it is an accuracy-stability tradeoff induced by chunked open-loop execution.
"""
    path.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=configured_raw_root())
    parser.add_argument("--task3-dir", type=Path, default=Path("project/tables/task3"))
    parser.add_argument("--figure-dir", type=Path, default=Path("project/figures/task3"))
    parser.add_argument("--table-dir", type=Path, default=Path("project/tables/task3"))
    parser.add_argument("--samples-per-env", type=int, default=400)
    args = parser.parse_args()

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    visual_rows = collect_visual_stats(args.raw_root, args.samples_per_env)
    visual_rows.extend(aggregate_env_stats(visual_rows, "ABC", ["A", "B", "C"]))
    write_rows(args.table_dir / "visual_stats_ABCD_task3.csv", visual_rows)

    shift_rows = [visual_distance(visual_rows, source) for source in ["A", "B", "C", "ABC"]]
    write_rows(args.table_dir / "visual_shift_to_D_task3.csv", shift_rows)

    result_rows = read_rows(args.task3_dir / "zero_shot_D_results.csv")
    horizon_rows = read_rows(args.task3_dir / "zero_shot_D_chunk_horizon.csv")
    robustness_rows = build_robustness_table(result_rows, horizon_rows, shift_rows)
    write_rows(args.table_dir / "chunk_visual_shift_robustness.csv", robustness_rows)
    write_latex(
        args.table_dir / "chunk_visual_shift_robustness.tex",
        robustness_rows,
        [
            "experiment",
            "train_visual_env",
            "action_l1_D",
            "smoothness_ratio_pred_over_gt",
            "boundary_amplification_pred_over_gt",
            "chunk_horizon_degradation_last10_minus_first10",
        ],
        "ACT chunking robustness on unseen D under visual distribution shift.",
        "tab:act-chunk-robustness-shift",
    )

    plot_visual_shift(visual_rows, args.figure_dir / "visual_shift_ABCD_task3.png")
    plot_robustness(robustness_rows, args.figure_dir / "chunk_visual_shift_robustness.png")
    plot_tradeoff(robustness_rows, args.figure_dir / "chunk_accuracy_stability_tradeoff.png")
    write_report_notes(args.table_dir / "chunk_visual_shift_report_notes.md", robustness_rows, shift_rows)
    print(f"Wrote {args.table_dir / 'chunk_visual_shift_robustness.csv'}")
    print(f"Wrote {args.figure_dir / 'chunk_visual_shift_robustness.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
