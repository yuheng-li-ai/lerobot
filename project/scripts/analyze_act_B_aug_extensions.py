#!/usr/bin/env python
"""One-to-one ACT-B augmentation visualizations matching the ACT-B baseline."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_act import (  # noqa: E402
    ImageBatchAugmenter,
    _apply_imagenet_stats,
    _build_act_config,
    _load_config,
    _make_dataset,
    _move_uint8_images_to_float,
    _parse_episode_spec,
    _set_seed,
)
from lerobot.datasets import LeRobotDatasetMetadata  # noqa: E402


def load_metrics(metrics_path: Path) -> list[dict[str, float]]:
    rows = []
    with metrics_path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({key: float(value) for key, value in row.items()})
    return rows


def finite_points(rows: list[dict[str, float]], key: str) -> tuple[np.ndarray, np.ndarray]:
    points = [(int(row["step"]), row[key]) for row in rows if math.isfinite(row[key])]
    return np.array([step for step, _ in points]), np.array([value for _, value in points], dtype=np.float64)


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


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_loss_outputs(rows: list[dict[str, float]], run_dir: Path, table_dir: Path, figure_dir: Path) -> None:
    train_steps, train_l1 = finite_points(rows, "train_action_l1")
    val_steps, val_l1 = finite_points(rows, "val_action_l1")
    train_smooth = trailing_mean(train_l1, 1000)
    best_idx = int(np.argmin(val_l1))
    best_step = int(val_steps[best_idx])
    best_val = float(val_l1[best_idx])
    final = rows[-1]

    summary = [
        {
            "first_train_action_l1": rows[0]["train_action_l1"],
            "final_train_action_l1": final["train_action_l1"],
            "final_train_action_l1_1000step_mean": float(train_smooth[-1]),
            "best_val_step": best_step,
            "best_val_action_l1": best_val,
            "final_val_action_l1": final["val_action_l1"],
            "final_minus_best_val_action_l1": final["val_action_l1"] - best_val,
            "num_steps": int(final["step"]),
            "num_val_points": len(val_steps),
        }
    ]
    write_rows(table_dir / "act_B_aug_overfitting_summary.csv", summary)

    rows_by_step = {int(row["step"]): row for row in rows}
    checkpoint_rows = [
        {
            "experiment": "act_B_aug",
            "selection": "best_val_metric_only",
            "step": best_step,
            "val_action_l1": best_val,
            "checkpoint": "",
            "checkpoint_exists": False,
            "notes": "best recorded validation step, but no checkpoint was saved at step 15000",
        },
        {
            "experiment": "act_B_aug",
            "selection": "nearest_saved_before_best",
            "step": 10000,
            "val_action_l1": rows_by_step[10000]["val_action_l1"],
            "checkpoint": str(run_dir / "checkpoints" / "step_00010000"),
            "checkpoint_exists": (run_dir / "checkpoints" / "step_00010000").is_dir(),
            "notes": "saved checkpoint before the best metric-only validation step",
        },
        {
            "experiment": "act_B_aug",
            "selection": "nearest_saved_after_best",
            "step": 20000,
            "val_action_l1": rows_by_step[20000]["val_action_l1"],
            "checkpoint": str(run_dir / "checkpoints" / "step_00020000"),
            "checkpoint_exists": (run_dir / "checkpoints" / "step_00020000").is_dir(),
            "notes": "saved checkpoint after the best metric-only validation step",
        },
        {
            "experiment": "act_B_aug",
            "selection": "final",
            "step": int(final["step"]),
            "val_action_l1": final["val_action_l1"],
            "checkpoint": str(run_dir / "checkpoint"),
            "checkpoint_exists": (run_dir / "checkpoint").is_dir(),
            "notes": "full 100k-step endpoint",
        },
    ]
    write_rows(table_dir / "act_B_aug_checkpoint_selection.csv", checkpoint_rows)

    smooth_rows = []
    for step, raw, smooth in zip(train_steps, train_l1, train_smooth, strict=True):
        if int(step) % 1000 != 0 and int(step) != int(train_steps[-1]):
            continue
        smooth_rows.append(
            {
                "step": int(step),
                "train_action_l1_raw": float(raw),
                "train_action_l1_trailing_mean": float(smooth),
                "train_smoothing_window_steps": min(int(step), 1000),
            }
        )
    write_rows(table_dir / "act_B_aug_train_l1_smoothed.csv", smooth_rows)

    gap_rows = []
    raw_gaps = []
    smooth_gaps = []
    gap_steps = []
    train_by_step = {int(step): value for step, value in zip(train_steps, train_l1, strict=True)}
    for row in rows:
        train_raw = row["train_action_l1"]
        val = row["val_action_l1"]
        if not (math.isfinite(train_raw) and math.isfinite(val)):
            continue
        step = int(row["step"])
        window_start = max(1, step - 1000 + 1)
        window_values = [train_by_step[idx] for idx in range(window_start, step + 1) if idx in train_by_step]
        train_mean = float(np.mean(window_values)) if window_values else train_raw
        raw_gap = val - train_raw
        smooth_gap = val - train_mean
        gap_steps.append(step)
        raw_gaps.append(raw_gap)
        smooth_gaps.append(smooth_gap)
        if step % 5000 == 0 or step == int(final["step"]):
            gap_rows.append(
                {
                    "step": step,
                    "train_action_l1_raw": train_raw,
                    "train_action_l1_trailing_mean": train_mean,
                    "train_smoothing_window_steps": len(window_values),
                    "val_action_l1": val,
                    "val_minus_train_action_l1_raw": raw_gap,
                    "val_minus_train_action_l1_smoothed": smooth_gap,
                }
            )
    write_rows(table_dir / "act_B_aug_train_val_gap.csv", gap_rows)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=180)
    ax.plot(train_steps, train_l1, color="#9ecae1", linewidth=0.35, alpha=0.22, label="Train Action L1, raw")
    ax.plot(train_steps, train_smooth, color="#2f6f9f", linewidth=1.35, label="Train Action L1, 1000-step mean")
    ax.plot(val_steps, val_l1, color="#b23a48", linewidth=1.2, label="Validation Action L1")
    ax.axvline(best_step, color="#2f9f6f", linestyle="--", linewidth=1.0, label="Best val")
    ax.scatter([best_step], [best_val], color="#2f9f6f", s=28, zorder=3)
    ax.scatter([final["step"]], [final["val_action_l1"]], color="#1f1f1f", s=24, zorder=3, label="Final val")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Action L1")
    ax.set_title("ACT-B-Aug Training and Validation Action L1")
    ax.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "act_B_aug_loss_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 3.8), dpi=180)
    smooth_gap_array = np.asarray(smooth_gaps)
    ax.plot(gap_steps, raw_gaps, color="#b7a4d9", linewidth=0.6, alpha=0.45, label="Raw mini-batch gap")
    ax.plot(gap_steps, smooth_gaps, color="#6d4c9f", linewidth=1.3, label="Gap vs 1000-step train mean")
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.fill_between(gap_steps, 0, smooth_gap_array, where=smooth_gap_array >= 0, color="#b7a4d9", alpha=0.28)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation L1 - train L1")
    ax.set_title("ACT-B-Aug Train-Val Generalization Gap")
    ax.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "act_B_aug_train_val_gap.png")
    plt.close(fig)


def _image_from_tensor(tensor: torch.Tensor, size: tuple[int, int] | None = None) -> Image.Image:
    image = tensor.detach().cpu().clamp(0, 1)
    while image.ndim > 3:
        image = image[0]
    arr = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    if size is not None:
        pil = pil.resize(size)
    return pil


def _add_label(img: Image.Image, label: str) -> Image.Image:
    label_h = 24
    canvas = Image.new("RGB", (img.width, img.height + label_h), "white")
    canvas.paste(img, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 5), label, fill=(0, 0, 0))
    return canvas


def _visual_row(environment: str, camera: str, images: list[np.ndarray]) -> dict[str, Any]:
    stacked = np.concatenate([image.reshape(-1, 3) for image in images], axis=0)
    brightness = np.array([image.reshape(-1, 3).mean(axis=1).mean() for image in images])
    contrast = np.array([image.reshape(-1, 3).mean(axis=1).std() for image in images])
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    return {
        "environment": environment,
        "camera": camera,
        "sampled_frames": len(images),
        "rgb_mean_r": float(mean[0]),
        "rgb_mean_g": float(mean[1]),
        "rgb_mean_b": float(mean[2]),
        "rgb_std_r": float(std[0]),
        "rgb_std_g": float(std[1]),
        "rgb_std_b": float(std[2]),
        "brightness_mean": float(brightness.mean()),
        "brightness_std": float(brightness.std()),
        "contrast_mean": float(contrast.mean()),
        "contrast_std": float(contrast.std()),
    }


def write_augmented_visual_outputs(config_path: Path, table_dir: Path, figure_dir: Path, sample_count: int) -> None:
    cfg = _load_config(config_path)
    _set_seed(int(cfg["experiment"]["seed"]))
    policy_cfg = _build_act_config(cfg["policy"])
    data_cfg = cfg["dataset"]
    ds_meta = LeRobotDatasetMetadata(data_cfg["repo_id"], root=data_cfg["root"])
    train_episodes = _parse_episode_spec(data_cfg["train_episodes"], ds_meta.total_episodes)
    dataset = _make_dataset(
        data_cfg["repo_id"],
        data_cfg["root"],
        train_episodes,
        policy_cfg,
        return_uint8=bool(data_cfg.get("return_uint8", True)),
        video_backend=data_cfg.get("video_backend"),
    )
    if data_cfg.get("use_imagenet_stats", True):
        _apply_imagenet_stats(dataset)

    augmenter = ImageBatchAugmenter(cfg.get("augmentation", {}))
    camera_keys = list(dataset.meta.camera_keys)
    ids = np.linspace(0, len(dataset) - 1, min(sample_count, len(dataset)), dtype=int)

    original_by_camera: dict[str, list[np.ndarray]] = {key: [] for key in camera_keys}
    augmented_by_camera: dict[str, list[np.ndarray]] = {key: [] for key in camera_keys}
    sample_cells: list[Image.Image] = []
    sample_ids = set(ids[np.linspace(0, len(ids) - 1, min(6, len(ids)), dtype=int)].tolist())

    for idx in ids:
        item = dataset[int(idx)]
        batch = {key: item[key].unsqueeze(0).clone() for key in camera_keys}
        batch = _move_uint8_images_to_float(batch, camera_keys)
        augmented = augmenter({key: value.clone() for key, value in batch.items()}, camera_keys)
        for cam_key in camera_keys:
            orig_img = batch[cam_key][0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            aug_img = augmented[cam_key][0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            original_by_camera[cam_key].append(orig_img)
            augmented_by_camera[cam_key].append(aug_img)
        if int(idx) in sample_ids:
            static_orig = _add_label(_image_from_tensor(batch["observation.images.static"], (160, 160)), f"raw {int(idx)}")
            static_aug = _add_label(_image_from_tensor(augmented["observation.images.static"], (160, 160)), f"aug {int(idx)}")
            grip_orig = _add_label(_image_from_tensor(batch["observation.images.gripper"], (160, 160)), f"raw grip")
            grip_aug = _add_label(_image_from_tensor(augmented["observation.images.gripper"], (160, 160)), f"aug grip")
            sample_cells.extend([static_orig, static_aug, grip_orig, grip_aug])

    visual_rows = []
    for cam_key in camera_keys:
        short = "static" if "static" in cam_key else "gripper"
        visual_rows.append(_visual_row("B_raw_for_aug", short, original_by_camera[cam_key]))
        visual_rows.append(_visual_row("B_augmented", short, augmented_by_camera[cam_key]))
    write_rows(table_dir / "act_B_aug_visual_stats.csv", visual_rows)

    columns = 4
    cell_w, cell_h = 160, 184
    rows = math.ceil(len(sample_cells) / columns)
    grid = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    for idx, cell in enumerate(sample_cells):
        grid.paste(cell, ((idx % columns) * cell_w, (idx // columns) * cell_h))
    grid.save(figure_dir / "act_B_aug_samples.png")

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), dpi=180, sharey=True, constrained_layout=True)
    channels = ["r", "g", "b"]
    colors = ["#c44e52", "#55a868", "#4c72b0"]
    for ax, camera in zip(axes, ["static", "gripper"], strict=True):
        raw_row = next(row for row in visual_rows if row["environment"] == "B_raw_for_aug" and row["camera"] == camera)
        aug_row = next(row for row in visual_rows if row["environment"] == "B_augmented" and row["camera"] == camera)
        x = np.arange(3)
        raw_means = np.array([raw_row[f"rgb_mean_{ch}"] for ch in channels])
        aug_means = np.array([aug_row[f"rgb_mean_{ch}"] for ch in channels])
        ax.bar(x - 0.18, raw_means, width=0.36, color=colors, alpha=0.42, edgecolor="#222222", linewidth=0.3, label="raw")
        ax.bar(x + 0.18, aug_means, width=0.36, color=colors, alpha=0.92, edgecolor="#222222", linewidth=0.3, label="aug")
        ax.set_xticks(x)
        ax.set_xticklabels(["R", "G", "B"])
        ax.set_title(f"{camera} camera")
        ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel("Pixel value, normalized")
    axes[1].legend(frameon=False)
    fig.suptitle("ACT-B-Aug Visual Color Profile")
    fig.savefig(figure_dir / "act_B_aug_visual_color_profile.png", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), dpi=180, constrained_layout=True)
    for camera, color in [("static", "#4c78a8"), ("gripper", "#f58518")]:
        raw_images = original_by_camera[f"observation.images.{camera}"]
        aug_images = augmented_by_camera[f"observation.images.{camera}"]
        raw_brightness = [image.reshape(-1, 3).mean(axis=1).mean() for image in raw_images]
        aug_brightness = [image.reshape(-1, 3).mean(axis=1).mean() for image in aug_images]
        raw_contrast = [image.reshape(-1, 3).mean(axis=1).std() for image in raw_images]
        aug_contrast = [image.reshape(-1, 3).mean(axis=1).std() for image in aug_images]
        axes[0].hist(raw_brightness, bins=45, color=color, alpha=0.28, label=f"{camera} raw")
        axes[0].hist(aug_brightness, bins=45, color=color, alpha=0.68, histtype="stepfilled", label=f"{camera} aug")
        axes[1].hist(raw_contrast, bins=45, color=color, alpha=0.28, label=f"{camera} raw")
        axes[1].hist(aug_contrast, bins=45, color=color, alpha=0.68, histtype="stepfilled", label=f"{camera} aug")
    axes[0].set_xlabel("Brightness")
    axes[1].set_xlabel("Contrast")
    for ax in axes:
        ax.set_ylabel("Sampled frame count")
        ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("ACT-B-Aug Brightness and Contrast")
    fig.savefig(figure_dir / "act_B_aug_brightness_contrast_hist.png", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def copy_data_invariant_artifacts(table_dir: Path, figure_dir: Path) -> list[dict[str, Any]]:
    mappings = [
        ("figure", "env_B_action_distribution.png", "act_B_aug_action_distribution.png", "same B action labels; augmentation changes images only"),
        ("figure", "env_B_action_smoothness.png", "act_B_aug_action_smoothness.png", "same B action labels; augmentation changes images only"),
        ("figure", "env_B_chunk_baseline.png", "act_B_aug_chunk_baseline.png", "same ground-truth action chunks"),
        ("figure", "env_B_action_violin.png", "act_B_aug_action_violin.png", "same B action labels"),
        ("figure", "env_B_action_delta_heatmap.png", "act_B_aug_action_delta_heatmap.png", "same B action labels"),
        ("figure", "env_B_task_frequency.png", "act_B_aug_task_frequency.png", "same B task distribution"),
        ("figure", "env_B_representative_trajectory_strip.png", "act_B_aug_representative_trajectory_strip.png", "same B trajectory labels"),
        ("figure", "env_B_gripper_diagnostics.png", "act_B_aug_gripper_diagnostics.png", "same binary gripper labels"),
        ("figure", "env_B_gripper_timeline.png", "act_B_aug_gripper_timeline.png", "same binary gripper labels"),
        ("table", "env_B_action_summary.csv", "act_B_aug_action_summary.csv", "same B action labels"),
        ("table", "env_B_action_stats.csv", "act_B_aug_action_stats.csv", "same B action labels"),
        ("table", "env_B_action_smoothness.csv", "act_B_aug_action_smoothness.csv", "same B action labels"),
        ("table", "env_B_chunk_baseline.csv", "act_B_aug_chunk_baseline.csv", "same ground-truth action chunks"),
        ("table", "env_B_chunk_baseline_summary.csv", "act_B_aug_chunk_baseline_summary.csv", "same ground-truth action chunks"),
        ("table", "env_B_action_delta_heatmap.csv", "act_B_aug_action_delta_heatmap.csv", "same B action labels"),
        ("table", "env_B_task_counts.csv", "act_B_aug_task_counts.csv", "same B task distribution"),
        ("table", "env_B_representative_trajectory.csv", "act_B_aug_representative_trajectory.csv", "same B trajectory labels"),
        ("table", "env_B_gripper_summary.csv", "act_B_aug_gripper_summary.csv", "same binary gripper labels"),
        ("table", "env_B_gripper_runs.csv", "act_B_aug_gripper_runs.csv", "same binary gripper labels"),
    ]
    coverage = []
    for kind, baseline_name, aug_name, note in mappings:
        base_dir = figure_dir if kind == "figure" else table_dir
        src = base_dir / baseline_name
        dst = base_dir / aug_name
        shutil.copyfile(src, dst)
        coverage.append(
            {
                "category": kind,
                "baseline_artifact": str(src),
                "augmentation_artifact": str(dst),
                "status": "copied_data_invariant",
                "notes": note,
            }
        )
    return coverage


def write_baseline_aug_comparison(table_dir: Path) -> None:
    baseline_path = table_dir / "act_B_overfitting_summary.csv"
    aug_path = table_dir / "act_B_aug_overfitting_summary.csv"
    with baseline_path.open() as f:
        baseline = next(csv.DictReader(f))
    with aug_path.open() as f:
        aug = next(csv.DictReader(f))
    row = {
        "metric": "validation_action_l1",
        "act_B_best": float(baseline["best_val_action_l1"]),
        "act_B_best_step": int(float(baseline["best_val_step"])),
        "act_B_final": float(baseline["final_val_action_l1"]),
        "act_B_aug_best": float(aug["best_val_action_l1"]),
        "act_B_aug_best_step": int(float(aug["best_val_step"])),
        "act_B_aug_final": float(aug["final_val_action_l1"]),
        "aug_minus_baseline_best": float(aug["best_val_action_l1"]) - float(baseline["best_val_action_l1"]),
        "aug_minus_baseline_final": float(aug["final_val_action_l1"]) - float(baseline["final_val_action_l1"]),
    }
    write_rows(table_dir / "act_B_vs_aug_summary.csv", [row])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("project/configs/act_B_aug_full.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("project"))
    parser.add_argument("--visual-sample-count", type=int, default=2000)
    args = parser.parse_args()

    table_dir = args.output_dir / "tables"
    figure_dir = args.output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = load_metrics(args.run_dir / "metrics.csv")
    write_loss_outputs(metrics_rows, args.run_dir, table_dir, figure_dir)
    write_augmented_visual_outputs(args.config, table_dir, figure_dir, args.visual_sample_count)
    coverage = copy_data_invariant_artifacts(table_dir, figure_dir)
    coverage.extend(
        [
            {
                "category": "figure",
                "baseline_artifact": "project/figures/loss_curve_act_B.png",
                "augmentation_artifact": "project/figures/act_B_aug_loss_curve.png",
                "status": "generated_from_aug_metrics",
                "notes": "uses ACT-B-aug full-training metrics",
            },
            {
                "category": "figure",
                "baseline_artifact": "project/figures/train_val_gap_act_B.png",
                "augmentation_artifact": "project/figures/act_B_aug_train_val_gap.png",
                "status": "generated_from_aug_metrics",
                "notes": "uses ACT-B-aug full-training metrics",
            },
            {
                "category": "figure",
                "baseline_artifact": "project/figures/env_B_samples.png",
                "augmentation_artifact": "project/figures/act_B_aug_samples.png",
                "status": "generated_from_augmented_images",
                "notes": "raw-vs-augmented B samples",
            },
            {
                "category": "figure",
                "baseline_artifact": "project/figures/env_B_visual_color_profile.png",
                "augmentation_artifact": "project/figures/act_B_aug_visual_color_profile.png",
                "status": "generated_from_augmented_images",
                "notes": "raw-vs-augmented RGB means",
            },
            {
                "category": "figure",
                "baseline_artifact": "project/figures/env_B_brightness_contrast_hist.png",
                "augmentation_artifact": "project/figures/act_B_aug_brightness_contrast_hist.png",
                "status": "generated_from_augmented_images",
                "notes": "raw-vs-augmented brightness/contrast",
            },
            {
                "category": "table",
                "baseline_artifact": "project/tables/act_B_overfitting_summary.csv",
                "augmentation_artifact": "project/tables/act_B_aug_overfitting_summary.csv",
                "status": "generated_from_aug_metrics",
                "notes": "uses ACT-B-aug full-training metrics",
            },
            {
                "category": "table",
                "baseline_artifact": "project/tables/act_B_checkpoint_selection.csv",
                "augmentation_artifact": "project/tables/act_B_aug_checkpoint_selection.csv",
                "status": "generated_from_aug_metrics",
                "notes": "best-val and final ACT-B-aug checkpoints",
            },
            {
                "category": "table",
                "baseline_artifact": "project/tables/act_B_train_l1_smoothed.csv",
                "augmentation_artifact": "project/tables/act_B_aug_train_l1_smoothed.csv",
                "status": "generated_from_aug_metrics",
                "notes": "1000-step train L1 mean",
            },
            {
                "category": "table",
                "baseline_artifact": "project/tables/act_B_train_val_gap.csv",
                "augmentation_artifact": "project/tables/act_B_aug_train_val_gap.csv",
                "status": "generated_from_aug_metrics",
                "notes": "validation gap vs smoothed train L1",
            },
            {
                "category": "table",
                "baseline_artifact": "project/tables/env_B_visual_stats.csv",
                "augmentation_artifact": "project/tables/act_B_aug_visual_stats.csv",
                "status": "generated_from_augmented_images",
                "notes": "raw-vs-augmented visual statistics",
            },
        ]
    )
    write_rows(table_dir / "act_B_aug_visualization_coverage.csv", sorted(coverage, key=lambda row: row["augmentation_artifact"]))
    write_baseline_aug_comparison(table_dir)

    print(f"Wrote ACT-B-aug one-to-one figures to {figure_dir}")
    print(f"Wrote ACT-B-aug one-to-one tables to {table_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
