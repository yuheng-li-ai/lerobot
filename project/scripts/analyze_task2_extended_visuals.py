#!/usr/bin/env python3
"""Task 2 per-model dataset/action visualizations.

This complements ``analyze_task2_visuals.py``. It intentionally keeps every
figure inside an experiment-specific folder and does not create cross-model
comparison figures.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_act import (  # noqa: E402
    ImageBatchAugmenter,
    _build_act_config,
    _load_config,
    _make_dataset,
    _move_uint8_images_to_float,
    _set_seed,
)


EXPERIMENTS = [
    "act_ABC",
    "act_ABC_size_matched",
    "act_ABC_aug",
    "act_ABC_size_matched_aug",
]

CONFIGS = {
    "act_ABC": Path("project/configs/act_ABC_full.yaml"),
    "act_ABC_size_matched": Path("project/configs/act_ABC_size_matched_full.yaml"),
    "act_ABC_aug": Path("project/configs/act_ABC_aug_full.yaml"),
    "act_ABC_size_matched_aug": Path("project/configs/act_ABC_size_matched_aug_full.yaml"),
}

DISPLAY = {
    "act_ABC": "ACT-ABC",
    "act_ABC_size_matched": "ACT-ABC Size-Matched",
    "act_ABC_aug": "ACT-ABC Aug",
    "act_ABC_size_matched_aug": "ACT-ABC Size-Matched Aug",
}

ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)


def set_title(ax: plt.Axes, text: str, fontsize: int = 10) -> None:
    ax.set_title(textwrap.fill(text, width=72), pad=16, fontsize=fontsize)


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    cumsum = np.cumsum(np.insert(values.astype(np.float64), 0, 0.0))
    out = np.empty_like(values, dtype=np.float64)
    for idx in range(len(values)):
        start = max(0, idx + 1 - window)
        out[idx] = (cumsum[idx + 1] - cumsum[start]) / (idx + 1 - start)
    return out


def downsample_rows(array: np.ndarray, max_rows: int) -> np.ndarray:
    if len(array) <= max_rows:
        return array
    ids = np.linspace(0, len(array) - 1, max_rows, dtype=int)
    return array[ids]


def image_from_tensor(tensor: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    image = tensor.detach().cpu()
    while image.ndim > 3:
        image = image[0]
    if image.dtype == torch.uint8:
        arr = image.permute(1, 2, 0).numpy()
    else:
        arr = (image.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr).resize(size)


def add_label(img: Image.Image, label: str) -> Image.Image:
    label_h = 26
    canvas = Image.new("RGB", (img.width, img.height + label_h), "white")
    canvas.paste(img, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 6), label, fill=(0, 0, 0))
    return canvas


def selected_episode_info(root: Path, episodes: list[int]) -> pd.DataFrame:
    path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    df = pd.read_parquet(path)
    selected = df[df["episode_index"].isin(set(episodes))].copy()
    selected["task_label"] = selected["tasks"].map(normalize_task_label)
    return selected


def normalize_task_label(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if len(value) == 0:
            return "unknown"
        return str(value[0])
    if isinstance(value, (list, tuple)):
        if not value:
            return "unknown"
        return str(value[0])
    return str(value)


def env_from_task(task: str) -> str:
    for env in ("A", "B", "C", "D"):
        if f"_{env}_" in task:
            return env
    return "unknown"


def task_rows_from_episodes(episodes_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    grouped = episodes_df.groupby("task_label", sort=True).agg(
        episodes=("episode_index", "count"),
        frames=("length", "sum"),
    )
    for task, row in grouped.iterrows():
        rows.append(
            {
                "task": str(task),
                "environment": env_from_task(str(task)),
                "episodes": int(row["episodes"]),
                "frames": int(row["frames"]),
            }
        )
    return rows


def load_actions(root: Path, episodes: list[int]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    episode_set = set(int(ep) for ep in episodes)
    by_episode: dict[int, np.ndarray] = {}
    for parquet_path in sorted((root / "data").glob("chunk-*/file-*.parquet")):
        df = pd.read_parquet(parquet_path, columns=["episode_index", "frame_index", "action"])
        df = df[df["episode_index"].isin(episode_set)]
        if df.empty:
            continue
        for episode, group in df.groupby("episode_index", sort=True):
            group = group.sort_values("frame_index")
            by_episode[int(episode)] = np.stack(group["action"].to_numpy()).astype(np.float64)
    ordered = [ep for ep in sorted(episode_set) if ep in by_episode]
    arrays = [by_episode[ep] for ep in ordered]
    if not arrays:
        raise RuntimeError("No selected actions found")
    lengths = np.asarray([len(array) for array in arrays], dtype=np.int64)
    return np.concatenate(arrays, axis=0), lengths, ordered


def episode_diffs(action_array: np.ndarray, range_lengths: np.ndarray) -> np.ndarray:
    diffs = []
    cursor = 0
    for length in range_lengths:
        segment = action_array[cursor : cursor + int(length)]
        cursor += int(length)
        if len(segment) > 1:
            diffs.append(np.diff(segment, axis=0))
    return np.concatenate(diffs, axis=0) if diffs else np.empty((0, action_array.shape[1]))


def load_dataset(cfg: dict[str, Any], episodes: list[int]):
    policy_cfg = _build_act_config(cfg["policy"])
    data_cfg = cfg["dataset"]
    return _make_dataset(
        data_cfg["repo_id"],
        data_cfg["root"],
        episodes,
        policy_cfg,
        return_uint8=bool(data_cfg.get("return_uint8", True)),
        video_backend=data_cfg.get("video_backend"),
    )


def visual_row(label: str, camera: str, images: list[np.ndarray]) -> dict[str, Any]:
    stacked = np.concatenate([image.reshape(-1, 3) for image in images], axis=0)
    brightness = np.asarray([image.reshape(-1, 3).mean(axis=1).mean() for image in images])
    contrast = np.asarray([image.reshape(-1, 3).mean(axis=1).std() for image in images])
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    return {
        "source": label,
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


def collect_visuals(
    experiment: str,
    cfg: dict[str, Any],
    figure_dir: Path,
    table_dir: Path,
    sample_count: int,
    sample_grid_count: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    data_cfg = cfg["dataset"]
    episodes = [int(ep) for ep in data_cfg["train_episodes"]]
    dataset = load_dataset(cfg, episodes)
    camera_keys = list(dataset.meta.camera_keys)
    ids = np.linspace(0, len(dataset) - 1, min(sample_count, len(dataset)), dtype=int)
    sample_ids = set(ids[np.linspace(0, len(ids) - 1, min(sample_grid_count, len(ids)), dtype=int)].tolist())

    aug_enabled = bool(cfg.get("augmentation", {}).get("enabled", False))
    augmenter = ImageBatchAugmenter(cfg.get("augmentation", {}))
    _set_seed(int(cfg["experiment"]["seed"]))

    raw_by_camera: dict[str, list[np.ndarray]] = {key: [] for key in camera_keys}
    aug_by_camera: dict[str, list[np.ndarray]] = {key: [] for key in camera_keys}
    cells: list[Image.Image] = []

    for idx in ids:
        item = dataset[int(idx)]
        batch = {key: item[key].unsqueeze(0).clone() for key in camera_keys}
        batch = _move_uint8_images_to_float(batch, camera_keys)
        augmented = augmenter({key: value.clone() for key, value in batch.items()}, camera_keys)
        for cam_key in camera_keys:
            raw = batch[cam_key][0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            raw_by_camera[cam_key].append(raw)
            if aug_enabled:
                aug = augmented[cam_key][0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                aug_by_camera[cam_key].append(aug)
        if int(idx) in sample_ids:
            if aug_enabled:
                cells.extend(
                    [
                        add_label(image_from_tensor(batch["observation.images.static"], (150, 150)), f"raw static {int(idx)}"),
                        add_label(image_from_tensor(augmented["observation.images.static"], (150, 150)), "aug static"),
                        add_label(image_from_tensor(batch["observation.images.gripper"], (150, 150)), "raw gripper"),
                        add_label(image_from_tensor(augmented["observation.images.gripper"], (150, 150)), "aug gripper"),
                    ]
                )
            else:
                cells.extend(
                    [
                        add_label(image_from_tensor(batch["observation.images.static"], (170, 170)), f"static {int(idx)}"),
                        add_label(image_from_tensor(batch["observation.images.gripper"], (170, 170)), f"gripper {int(idx)}"),
                    ]
                )

    rows = []
    arrays: dict[str, np.ndarray] = {}
    for cam_key in camera_keys:
        camera = "static" if "static" in cam_key else "gripper"
        rows.append(visual_row("raw", camera, raw_by_camera[cam_key]))
        raw_brightness = np.asarray([image.reshape(-1, 3).mean(axis=1).mean() for image in raw_by_camera[cam_key]])
        raw_contrast = np.asarray([image.reshape(-1, 3).mean(axis=1).std() for image in raw_by_camera[cam_key]])
        arrays[f"raw_{camera}_brightness"] = raw_brightness
        arrays[f"raw_{camera}_contrast"] = raw_contrast
        if aug_enabled:
            rows.append(visual_row("augmented", camera, aug_by_camera[cam_key]))
            aug_brightness = np.asarray([image.reshape(-1, 3).mean(axis=1).mean() for image in aug_by_camera[cam_key]])
            aug_contrast = np.asarray([image.reshape(-1, 3).mean(axis=1).std() for image in aug_by_camera[cam_key]])
            arrays[f"augmented_{camera}_brightness"] = aug_brightness
            arrays[f"augmented_{camera}_contrast"] = aug_contrast

    write_rows(table_dir / "visual_stats.csv", rows)

    columns = 4 if aug_enabled else sample_grid_count
    cell_w = 150 if aug_enabled else 170
    cell_h = 176 if aug_enabled else 196
    grid_rows = math.ceil(len(cells) / columns)
    grid = Image.new("RGB", (columns * cell_w, grid_rows * cell_h), "white")
    for cell_idx, cell in enumerate(cells):
        grid.paste(cell, ((cell_idx % columns) * cell_w, (cell_idx // columns) * cell_h))
    grid.save(figure_dir / "dataset_samples.png")

    plot_visual_color_profile(experiment, rows, figure_dir)
    plot_brightness_contrast(experiment, rows, arrays, figure_dir, aug_enabled)
    if aug_enabled:
        plot_augmentation_verification(experiment, rows, figure_dir)
    return rows, arrays


def plot_visual_color_profile(experiment: str, rows: list[dict[str, Any]], figure_dir: Path) -> None:
    channels = ["r", "g", "b"]
    colors = ["#c44e52", "#55a868", "#4c72b0"]
    sources = sorted({row["source"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), dpi=180, sharey=True, constrained_layout=True)
    for ax, camera in zip(axes, ["static", "gripper"], strict=True):
        camera_rows = [row for row in rows if row["camera"] == camera]
        x = np.arange(3)
        if len(sources) == 1:
            row = camera_rows[0]
            means = np.asarray([row[f"rgb_mean_{ch}"] for ch in channels])
            stds = np.asarray([row[f"rgb_std_{ch}"] for ch in channels])
            ax.bar(x, means, yerr=stds, color=colors, edgecolor="#222222", linewidth=0.4, capsize=3)
        else:
            width = 0.34
            for src_idx, source in enumerate(sources):
                row = next(row for row in camera_rows if row["source"] == source)
                means = np.asarray([row[f"rgb_mean_{ch}"] for ch in channels])
                ax.bar(x + (src_idx - 0.5) * width, means, width=width, color=colors, alpha=0.46 + src_idx * 0.38, edgecolor="#222222", linewidth=0.3, label=source)
        ax.set_xticks(x)
        ax.set_xticklabels(["R", "G", "B"])
        set_title(ax, f"{camera} camera", fontsize=9)
        ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel("Pixel value, normalized")
    if len(sources) > 1:
        axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{DISPLAY[experiment]} Visual Color Profile", fontsize=11)
    save_fig(fig, figure_dir / "visual_color_profile.png")


def plot_brightness_contrast(
    experiment: str,
    rows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    figure_dir: Path,
    aug_enabled: bool,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), dpi=180, constrained_layout=True)
    if aug_enabled:
        specs = [
            ("raw_static", "#4c78a8", "static raw", 0.28),
            ("augmented_static", "#4c78a8", "static aug", 0.70),
            ("raw_gripper", "#f58518", "gripper raw", 0.28),
            ("augmented_gripper", "#f58518", "gripper aug", 0.70),
        ]
    else:
        specs = [("raw_static", "#4c78a8", "static", 0.65), ("raw_gripper", "#f58518", "gripper", 0.65)]
    for prefix, color, label, alpha in specs:
        axes[0].hist(arrays[f"{prefix}_brightness"], bins=45, color=color, alpha=alpha, label=label)
        axes[1].hist(arrays[f"{prefix}_contrast"], bins=45, color=color, alpha=alpha, label=label)
    axes[0].set_xlabel("Brightness")
    axes[1].set_xlabel("Contrast")
    for ax in axes:
        ax.set_ylabel("Sampled frame count")
        ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
        ax.legend(frameon=False, fontsize=8)
    set_title(axes[0], "Brightness", fontsize=9)
    set_title(axes[1], "Contrast", fontsize=9)
    fig.suptitle(f"{DISPLAY[experiment]} Brightness and Contrast", fontsize=11)
    save_fig(fig, figure_dir / "brightness_contrast_hist.png")


def plot_augmentation_verification(experiment: str, rows: list[dict[str, Any]], figure_dir: Path) -> None:
    cameras = ["static", "gripper"]
    deltas = []
    for camera in cameras:
        raw = next(row for row in rows if row["source"] == "raw" and row["camera"] == camera)
        aug = next(row for row in rows if row["source"] == "augmented" and row["camera"] == camera)
        deltas.append(
            [
                abs(float(aug["brightness_mean"]) - float(raw["brightness_mean"])),
                abs(float(aug["contrast_mean"]) - float(raw["contrast_mean"])),
            ]
        )
    fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=180, constrained_layout=True)
    x = np.arange(len(cameras))
    delta = np.asarray(deltas)
    ax.bar(x - 0.18, delta[:, 0], width=0.36, color="#4c78a8", label="Brightness delta")
    ax.bar(x + 0.18, delta[:, 1], width=0.36, color="#f58518", label="Contrast delta")
    ax.set_xticks(x)
    ax.set_xticklabels(cameras)
    ax.set_ylabel("Mean absolute distribution shift")
    set_title(ax, f"{DISPLAY[experiment]} Augmentation Verification")
    ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=False)
    save_fig(fig, figure_dir / "augmentation_verification.png")


def action_stats_rows(action_array: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for dim, name in enumerate(ACTION_NAMES):
        vals = action_array[:, dim]
        rows.append(
            {
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
    return rows


def write_action_summary(action_array: np.ndarray, lengths: np.ndarray, table_dir: Path) -> None:
    diffs = episode_diffs(action_array, lengths)
    gripper = action_array[:, 6]
    row = {
        "num_frames": int(action_array.shape[0]),
        "num_episodes": int(len(lengths)),
        "mean_action_l2_first6": float(np.linalg.norm(action_array[:, :6], axis=1).mean()),
        "std_action_l2_first6": float(np.linalg.norm(action_array[:, :6], axis=1).std()),
        "mean_step_delta_l2_first6": float(np.linalg.norm(diffs[:, :6], axis=1).mean()),
        "std_step_delta_l2_first6": float(np.linalg.norm(diffs[:, :6], axis=1).std()),
        "gripper_close_fraction": float((gripper < 0).mean()),
        "gripper_open_fraction": float((gripper > 0).mean()),
        "gripper_switch_count": int(np.count_nonzero(np.diff(np.sign(gripper)))),
    }
    write_rows(table_dir / "action_summary.csv", [row])


def plot_action_distribution(experiment: str, rows: list[dict[str, Any]], figure_dir: Path) -> None:
    names = [row["action_name"] for row in rows]
    means = np.asarray([row["mean"] for row in rows])
    stds = np.asarray([row["std"] for row in rows])
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=180, constrained_layout=True)
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, color="#4c78a8", edgecolor="#222222", linewidth=0.5, capsize=3)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Relative action value")
    set_title(ax, f"{DISPLAY[experiment]} Action Distribution")
    ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    save_fig(fig, figure_dir / "action_distribution.png")


def plot_action_violin(experiment: str, action_array: np.ndarray, figure_dir: Path) -> None:
    sampled = downsample_rows(action_array, 50000)
    data = [sampled[:, dim] for dim in range(sampled.shape[1])]
    fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=180, constrained_layout=True)
    parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#4c78a8")
        body.set_edgecolor("#1f1f1f")
        body.set_alpha(0.72)
    parts["cmedians"].set_color("#1f1f1f")
    parts["cmedians"].set_linewidth(1.1)
    ax.axhline(0, color="#222222", linewidth=0.7)
    ax.set_xticks(np.arange(1, len(ACTION_NAMES) + 1))
    ax.set_xticklabels(ACTION_NAMES, rotation=30, ha="right")
    ax.set_ylabel("Relative action value")
    set_title(ax, f"{DISPLAY[experiment]} Per-Dimension Action Distribution")
    ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    save_fig(fig, figure_dir / "action_violin.png")


def plot_action_smoothness(experiment: str, action_array: np.ndarray, lengths: np.ndarray, figure_dir: Path, table_dir: Path) -> None:
    diffs = episode_diffs(action_array, lengths)
    abs_diffs = np.abs(diffs)
    delta_l2 = np.linalg.norm(diffs[:, :6], axis=1)
    rows = []
    for dim, name in enumerate(ACTION_NAMES):
        vals = abs_diffs[:, dim]
        rows.append(
            {
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
            "action_dim": -1,
            "action_name": "l2_first6",
            "mean_abs_delta": float(delta_l2.mean()),
            "std_abs_delta": float(delta_l2.std()),
            "q50_abs_delta": float(np.quantile(delta_l2, 0.50)),
            "q90_abs_delta": float(np.quantile(delta_l2, 0.90)),
            "q99_abs_delta": float(np.quantile(delta_l2, 0.99)),
        }
    )
    write_rows(table_dir / "action_smoothness.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), dpi=180, constrained_layout=True)
    axes[0].hist(downsample_rows(delta_l2[:, None], 100000).ravel(), bins=80, color="#4c78a8", alpha=0.88)
    axes[0].set_xlabel("||a_t - a_{t-1}||, first 6 dims")
    axes[0].set_ylabel("Count")
    set_title(axes[0], "Step Delta L2", fontsize=9)
    means = np.asarray([row["mean_abs_delta"] for row in rows[:-1]])
    axes[1].bar(np.arange(len(ACTION_NAMES)), means, color="#72b7b2", edgecolor="#222222", linewidth=0.4)
    axes[1].set_xticks(np.arange(len(ACTION_NAMES)))
    axes[1].set_xticklabels(ACTION_NAMES, rotation=35, ha="right")
    axes[1].set_ylabel("Mean absolute delta")
    set_title(axes[1], "Per-Dimension Smoothness", fontsize=9)
    for ax in axes:
        ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    fig.suptitle(f"{DISPLAY[experiment]} Action Smoothness", fontsize=11)
    save_fig(fig, figure_dir / "action_smoothness.png")


def plot_gripper(experiment: str, action_array: np.ndarray, lengths: np.ndarray, figure_dir: Path, table_dir: Path) -> None:
    gripper = action_array[:, 6]
    close = gripper < 0
    open_ = gripper > 0
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
    write_rows(
        table_dir / "gripper_summary.csv",
        [
            {
                "num_frames": int(len(gripper)),
                "close_count": int(close.sum()),
                "open_count": int(open_.sum()),
                "close_fraction": float(close.mean()),
                "open_fraction": float(open_.mean()),
                "switch_count": int(np.count_nonzero(np.diff(np.sign(gripper)))),
                "switch_rate_per_1000_frames": float(np.count_nonzero(np.diff(np.sign(gripper))) / len(gripper) * 1000),
                "mean_run_length_frames": float(run_lengths_arr.mean()),
                "median_run_length_frames": float(np.median(run_lengths_arr)),
                "q90_run_length_frames": float(np.quantile(run_lengths_arr, 0.90)),
                "q99_run_length_frames": float(np.quantile(run_lengths_arr, 0.99)),
            }
        ],
    )
    run_rows = [
        {
            "run_index": idx,
            "gripper_state": "open" if state > 0 else "close",
            "state_value": float(state),
            "run_length_frames": int(length),
        }
        for idx, (length, state) in enumerate(zip(run_lengths_arr, run_states_arr, strict=True))
    ]
    write_rows(table_dir / "gripper_runs.csv", run_rows)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), dpi=180, constrained_layout=True)
    axes[0].bar(["close (-1)", "open (+1)"], [float(close.mean()), float(open_.mean())], color=["#b23a48", "#2f6f9f"], edgecolor="#222222", linewidth=0.5)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Frame fraction")
    set_title(axes[0], "Gripper State Balance", fontsize=9)
    clipped = np.clip(run_lengths_arr, 0, np.quantile(run_lengths_arr, 0.99))
    axes[1].hist(clipped, bins=60, color="#6d4c9f", alpha=0.84)
    axes[1].set_xlabel("Run length, frames")
    axes[1].set_ylabel("Run count")
    set_title(axes[1], "Open/Close Run Lengths", fontsize=9)
    for ax in axes:
        ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    fig.suptitle(f"{DISPLAY[experiment]} Gripper Diagnostics", fontsize=11)
    save_fig(fig, figure_dir / "gripper_diagnostics.png")

    longest_idx = int(np.argmax(lengths))
    cursor = int(lengths[:longest_idx].sum())
    long_len = int(lengths[longest_idx])
    window_len = min(1200, long_len)
    local_start = max(0, long_len // 2 - window_len // 2)
    segment = gripper[cursor + local_start : cursor + local_start + window_len]
    x = np.arange(len(segment))
    switches = np.flatnonzero(np.diff(np.sign(segment)) != 0) + 1
    fig, ax = plt.subplots(figsize=(8.0, 2.8), dpi=180, constrained_layout=True)
    ax.step(x, segment, where="post", color="#2f6f9f", linewidth=1.2)
    for switch_idx in switches:
        ax.axvline(int(switch_idx), color="#b23a48", linewidth=0.55, alpha=0.45)
    ax.set_yticks([-1, 1])
    ax.set_yticklabels(["close", "open"])
    ax.set_xlabel("Frame in representative 1,200-frame window")
    set_title(ax, f"{DISPLAY[experiment]} Gripper State Timeline")
    ax.grid(True, axis="x", color="#d0d0d0", linewidth=0.4, alpha=0.5)
    save_fig(fig, figure_dir / "gripper_timeline.png")


def plot_chunk_baseline(experiment: str, action_array: np.ndarray, lengths: np.ndarray, figure_dir: Path, table_dir: Path, chunk_size: int) -> None:
    rows = []
    boundary_jumps = []
    within_delta = []
    first_last = []
    cursor = 0
    chunk_index = 0
    for episode_index, length in enumerate(lengths):
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
            rows.append(
                {
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
    write_rows(table_dir / "chunk_baseline.csv", rows)
    write_rows(
        table_dir / "chunk_baseline_summary.csv",
        [
            {
                "chunk_size": chunk_size,
                "num_chunks": len(rows),
                "mean_within_chunk_delta_l2_first6": float(np.mean(within_delta)),
                "mean_first_last_delta_l2_first6": float(np.mean(first_last)),
                "mean_boundary_jump_l2_first6": float(np.mean(boundary_jumps)),
                "q90_boundary_jump_l2_first6": float(np.quantile(boundary_jumps, 0.90)),
                "q99_boundary_jump_l2_first6": float(np.quantile(boundary_jumps, 0.99)),
            }
        ],
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), dpi=180, constrained_layout=True)
    axes[0].hist(downsample_rows(np.asarray(within_delta)[:, None], 50000).ravel(), bins=60, color="#54a24b", alpha=0.86)
    axes[0].set_xlabel("Mean within-chunk delta L2")
    axes[0].set_ylabel("Chunk count")
    set_title(axes[0], f"Within-Chunk Variation, c={chunk_size}", fontsize=9)
    axes[1].hist(downsample_rows(np.asarray(boundary_jumps)[:, None], 50000).ravel(), bins=60, color="#e45756", alpha=0.86)
    axes[1].set_xlabel("Boundary jump L2")
    axes[1].set_ylabel("Boundary count")
    set_title(axes[1], "Boundary-Style Jumps", fontsize=9)
    for ax in axes:
        ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    fig.suptitle(f"{DISPLAY[experiment]} Chunk Baseline", fontsize=11)
    save_fig(fig, figure_dir / "chunk_baseline.png")


def plot_action_delta_heatmap(experiment: str, action_array: np.ndarray, lengths: np.ndarray, figure_dir: Path, table_dir: Path, bins: int) -> None:
    diffs = np.abs(episode_diffs(action_array, lengths))
    ids = np.linspace(0, len(diffs), bins + 1, dtype=int)
    heat = np.zeros((bins, diffs.shape[1]), dtype=np.float64)
    rows = []
    for idx in range(bins):
        block = diffs[ids[idx] : ids[idx + 1]]
        if len(block) == 0:
            continue
        heat[idx] = block.mean(axis=0)
        row = {"bin": idx, "start_delta_index": int(ids[idx]), "end_delta_index": int(ids[idx + 1])}
        for dim, name in enumerate(ACTION_NAMES):
            row[f"{name}_mean_abs_delta"] = float(heat[idx, dim])
        rows.append(row)
    write_rows(table_dir / "action_delta_heatmap.csv", rows)
    fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=180, constrained_layout=True)
    im = ax.imshow(heat.T, aspect="auto", interpolation="nearest", cmap="magma")
    ax.set_yticks(np.arange(len(ACTION_NAMES)))
    ax.set_yticklabels(ACTION_NAMES)
    ax.set_xlabel("Time bin across selected training data")
    set_title(ax, f"{DISPLAY[experiment]} Action Delta Heatmap")
    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
    cbar.set_label("Mean absolute delta")
    save_fig(fig, figure_dir / "action_delta_heatmap.png")


def plot_task_frequency(experiment: str, task_rows: list[dict[str, Any]], figure_dir: Path, table_dir: Path) -> None:
    write_rows(table_dir / "task_frequency.csv", task_rows)
    rows = sorted(task_rows, key=lambda row: int(row["frames"]), reverse=True)
    labels = [str(row["task"]).replace("_", " ") for row in rows]
    counts = np.asarray([int(row["frames"]) for row in rows])
    fig, ax = plt.subplots(figsize=(7.4, 3.8), dpi=180, constrained_layout=True)
    y = np.arange(len(labels))
    ax.barh(y, counts, color="#4c78a8", edgecolor="#222222", linewidth=0.35)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Training frames")
    set_title(ax, f"{DISPLAY[experiment]} Task / Environment Frequency")
    ax.grid(True, axis="x", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    save_fig(fig, figure_dir / "task_frequency.png")


def plot_representative_trajectory(
    experiment: str,
    cfg: dict[str, Any],
    action_array: np.ndarray,
    lengths: np.ndarray,
    figure_dir: Path,
    table_dir: Path,
    sample_count: int,
) -> None:
    longest_idx = int(np.argmax(lengths))
    cursor = int(lengths[:longest_idx].sum())
    length = int(lengths[longest_idx])
    window_len = min(1200, length)
    local_start = max(0, length // 2 - window_len // 2)
    local_end = local_start + window_len
    segment = action_array[cursor + local_start : cursor + local_end]
    norm = np.linalg.norm(segment[:, :6], axis=1)
    smooth = trailing_mean(norm, 31)

    data_cfg = cfg["dataset"]
    episodes = [int(ep) for ep in data_cfg["train_episodes"]]
    dataset = load_dataset(cfg, episodes)
    global_ids = np.linspace(cursor + local_start, cursor + local_end - 1, sample_count, dtype=int)
    images = []
    rows = []
    for gid in global_ids:
        item = dataset[int(gid)]
        img = image_from_tensor(item["observation.images.static"], (150, 150))
        images.append(img)
        local_idx = int(gid - (cursor + local_start))
        rows.append(
            {
                "selected_dataset_index": int(gid),
                "representative_episode_rank": longest_idx,
                "window_start_local": int(local_start),
                "window_end_local": int(local_end - 1),
                "action_l2_first6_raw": float(norm[local_idx]),
                "action_l2_first6_trailing_mean": float(smooth[local_idx]),
            }
        )
    write_rows(table_dir / "representative_trajectory.csv", rows)

    fig = plt.figure(figsize=(10.0, 4.8), dpi=180, constrained_layout=True)
    gs = fig.add_gridspec(2, sample_count, height_ratios=[1.0, 0.78])
    for idx, img in enumerate(images):
        ax_img = fig.add_subplot(gs[0, idx])
        ax_img.imshow(img)
        ax_img.set_title(str(int(global_ids[idx])), fontsize=7, pad=6)
        ax_img.axis("off")
    ax = fig.add_subplot(gs[1, :])
    rel_x = np.linspace(0, 1, len(norm))
    ax.plot(rel_x, norm, color="#9ecae1", linewidth=0.6, alpha=0.45, label="Raw action norm")
    ax.plot(rel_x, smooth, color="#2f6f9f", linewidth=1.5, alpha=0.96, label="31-frame trailing mean")
    sample_x = (global_ids - (cursor + local_start)) / max(window_len - 1, 1)
    for sx in sample_x:
        ax.axvline(float(sx), color="#e45756", linewidth=0.7, alpha=0.45)
    ax.set_xlabel("Relative time in trajectory")
    ax.set_ylabel("Action L2, first 6 dims")
    set_title(ax, f"{DISPLAY[experiment]} Representative Trajectory Window")
    ax.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    save_fig(fig, figure_dir / "representative_trajectory_strip.png")


def process_experiment(args: argparse.Namespace, experiment: str) -> list[dict[str, Any]]:
    cfg = _load_config(CONFIGS[experiment])
    root = Path(cfg["dataset"]["root"])
    episodes = [int(ep) for ep in cfg["dataset"]["train_episodes"]]
    figure_dir = args.figure_root / experiment
    table_dir = args.table_root / experiment
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    episodes_df = selected_episode_info(root, episodes)
    write_rows(
        table_dir / "selected_episode_summary.csv",
        [
            {
                "num_episodes": int(len(episodes_df)),
                "num_frames": int(episodes_df["length"].sum()),
                "min_episode_index": int(episodes_df["episode_index"].min()),
                "max_episode_index": int(episodes_df["episode_index"].max()),
            }
        ],
    )
    task_rows = task_rows_from_episodes(episodes_df)
    plot_task_frequency(experiment, task_rows, figure_dir, table_dir)

    action_array, lengths, _ = load_actions(root, episodes)
    stats_rows = action_stats_rows(action_array)
    write_rows(table_dir / "action_stats.csv", stats_rows)
    write_action_summary(action_array, lengths, table_dir)
    plot_action_distribution(experiment, stats_rows, figure_dir)
    plot_action_violin(experiment, action_array, figure_dir)
    plot_action_smoothness(experiment, action_array, lengths, figure_dir, table_dir)
    plot_gripper(experiment, action_array, lengths, figure_dir, table_dir)
    plot_chunk_baseline(experiment, action_array, lengths, figure_dir, table_dir, args.chunk_size)
    plot_action_delta_heatmap(experiment, action_array, lengths, figure_dir, table_dir, args.heatmap_bins)
    collect_visuals(experiment, cfg, figure_dir, table_dir, args.visual_sample_count, args.image_sample_count)
    plot_representative_trajectory(experiment, cfg, action_array, lengths, figure_dir, table_dir, args.trajectory_sample_count)

    manifest = []
    for path in sorted(figure_dir.glob("*.png")):
        manifest.append(
            {
                "experiment": experiment,
                "figure": str(path),
                "kind": path.stem,
                "note": "single-model Task 2 diagnostic; no cross-model comparison",
            }
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-root", type=Path, default=Path("project/figures/task2"))
    parser.add_argument("--table-root", type=Path, default=Path("project/tables/task2"))
    parser.add_argument("--visual-sample-count", type=int, default=1200)
    parser.add_argument("--image-sample-count", type=int, default=6)
    parser.add_argument("--trajectory-sample-count", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--heatmap-bins", type=int, default=100)
    args = parser.parse_args()

    manifest = []
    for experiment in EXPERIMENTS:
        print(f"[Task2 visuals] processing {experiment}", flush=True)
        manifest.extend(process_experiment(args, experiment))
    write_rows(Path("project/tables/task2_extended_visualization_manifest.csv"), manifest)
    write_rows(Path("project/tables/task2_visualization_manifest.csv"), manifest)
    print(f"Wrote {len(manifest)} Task 2 figures under {args.figure_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
