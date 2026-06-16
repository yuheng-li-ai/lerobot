#!/usr/bin/env python3
"""ACT-Lang visualization suite.

The outputs are intentionally split into two folders:
- act_lang_core: training/data/action diagnostics that must be regenerated
  because the dataset and model family changed.
- act_lang_language: diagnostics unique to language-conditioned policies.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset, default_collate

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.policies.act.processor_act import make_act_pre_post_processors  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402

from train_act import (  # noqa: E402
    _apply_imagenet_stats,
    _load_config,
    _move_uint8_images_to_float,
    _parse_episode_spec,
)
from train_act_lang import ACTLangPolicy, _attach_dataset_features, _build_act_lang_config  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display: str
    config: Path
    run_dir: Path
    dataset_group: str


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


def write_latex(path: Path, rows: list[dict[str, Any]], float_format: str = "%.4f") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_latex(path, index=False, escape=True, float_format=float_format)


def fval(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def finite_points(rows: list[dict[str, str]], key: str) -> tuple[np.ndarray, np.ndarray]:
    points = []
    for row in rows:
        value = fval(row.get(key))
        if math.isfinite(value):
            points.append((int(float(row["step"])), value))
    return np.asarray([p[0] for p in points]), np.asarray([p[1] for p in points], dtype=np.float64)


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
    fig.savefig(path, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)


def short_label(text: str, width: int = 26) -> str:
    text = str(text)
    if len(text) <= width:
        return text
    return text[: width - 1] + "..."


def progress(label: str, done: int, total: int, detail: str = "", width: int = 32) -> None:
    total = max(1, int(total))
    done = min(max(0, int(done)), total)
    pct = done / total
    filled = min(width, max(0, int(round(width * pct))))
    bar = "#" * filled + "-" * (width - filled)
    suffix = f" {detail}" if detail else ""
    print(f"PROGRESS act_lang_visuals [{bar}] {done}/{total} ({pct * 100:5.1f}%) {label}{suffix}", flush=True)


def task_index_to_text(root: str | Path) -> dict[int, str]:
    tasks = pd.read_parquet(Path(root) / "meta/tasks.parquet")
    return {int(row["task_index"]): str(task) for task, row in tasks.iterrows()}


def sample_parquet_rows(
    root: str | Path,
    episodes: list[int],
    columns: list[str],
    max_samples: int,
) -> pd.DataFrame:
    root = Path(root)
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")
    episode_set = set(int(ep) for ep in episodes)
    read_columns = ["episode_index", *[col for col in columns if col != "episode_index"]]
    quota = max(1, int(math.ceil(max_samples / len(files))))
    chunks = []
    for file in files:
        df = pd.read_parquet(file, columns=read_columns)
        df = df[df["episode_index"].isin(episode_set)]
        if df.empty:
            continue
        take = min(quota, len(df))
        indices = np.linspace(0, len(df) - 1, take, dtype=np.int64)
        chunks.append(df.iloc[indices])
    if not chunks:
        return pd.DataFrame(columns=read_columns)
    out = pd.concat(chunks, ignore_index=True)
    if len(out) > max_samples:
        indices = np.linspace(0, len(out) - 1, max_samples, dtype=np.int64)
        out = out.iloc[indices].reset_index(drop=True)
    return out


def stack_series(series: pd.Series) -> np.ndarray:
    return np.stack([np.asarray(value) for value in series.to_list()], axis=0)


def load_model_specs(project_dir: Path) -> list[ModelSpec]:
    candidates = [
        ModelSpec(
            "act_lang_B",
            "ACT-Lang-B",
            project_dir / "configs/act_lang_B_full.yaml",
            Path("/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_B/20260606_112848_act_lang_B_full_gpu0"),
            "B",
        ),
        ModelSpec(
            "act_lang_ABC_200k",
            "ACT-Lang-ABC 200k",
            project_dir / "configs/act_lang_ABC_continue_200k.yaml",
            latest_run_dir(Path("/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_ABC_continue_200k")),
            "ABC",
        ),
        ModelSpec(
            "act_lang_ABC_size_matched",
            "ACT-Lang-ABC Size-Matched",
            project_dir / "configs/act_lang_ABC_size_matched_full.yaml",
            Path("/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_ABC_size_matched/20260606_164015_act_lang_ABC_size_matched_full_gpu2"),
            "ABC",
        ),
    ]
    return [
        spec
        for spec in candidates
        if spec.run_dir is not None
        and (spec.run_dir / "metrics.csv").is_file()
        and (spec.run_dir / "checkpoint/model.safetensors").is_file()
    ]


def latest_run_dir(root: Path) -> Path | None:
    if not root.is_dir():
        return None
    run_dirs = sorted(path for path in root.iterdir() if path.is_dir() and path.name != "logs")
    return run_dirs[-1] if run_dirs else None


def load_metrics_for_spec(spec: ModelSpec) -> list[dict[str, str]]:
    if spec.key != "act_lang_ABC_200k":
        return read_csv(spec.run_dir / "metrics.csv")

    base_metrics = Path(
        "/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_ABC/"
        "20260606_163515_act_lang_ABC_full_gpu1/metrics.csv"
    )
    rows = read_csv(base_metrics) + read_csv(spec.run_dir / "metrics.csv")
    by_step: dict[int, dict[str, str]] = {}
    for row in rows:
        by_step[int(float(row["step"]))] = row
    return [by_step[step] for step in sorted(by_step)]


def load_train_context(spec: ModelSpec):
    cfg = _load_config(spec.config)
    if os.environ.get("ACT_DEVICE_OVERRIDE"):
        cfg.setdefault("policy", {})["device"] = os.environ["ACT_DEVICE_OVERRIDE"]
    data_cfg = cfg["dataset"]
    policy_cfg = _build_act_lang_config(cfg["policy"])
    ds_meta = LeRobotDatasetMetadata(data_cfg["repo_id"], root=data_cfg["root"])
    _attach_dataset_features(policy_cfg, ds_meta)
    return cfg, data_cfg, policy_cfg, ds_meta


def make_dataset(data_cfg: dict[str, Any], policy_cfg, episodes: list[int], return_uint8: bool = True):
    ds_meta = LeRobotDatasetMetadata(data_cfg["repo_id"], root=data_cfg["root"])
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    return LeRobotDataset(
        data_cfg["repo_id"],
        root=data_cfg["root"],
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        return_uint8=return_uint8,
        video_backend=data_cfg.get("video_backend"),
    )


def sample_subset(dataset: LeRobotDataset, max_samples: int) -> Subset:
    count = min(max_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, count, dtype=np.int64).tolist()
    return Subset(dataset, indices)


def action_l1_per_item(batch: dict[str, Any], actions_hat: torch.Tensor) -> torch.Tensor:
    abs_err = (batch[ACTION] - actions_hat).abs()
    mask = ~batch["action_is_pad"].unsqueeze(-1)
    numerator = (abs_err * mask).sum(dim=(1, 2))
    denominator = (mask.sum(dim=(1, 2)) * abs_err.shape[-1]).clamp_min(1)
    return numerator / denominator


def action_l1_per_dim(batch: dict[str, Any], actions_hat: torch.Tensor) -> torch.Tensor:
    abs_err = (batch[ACTION] - actions_hat).abs()
    mask = ~batch["action_is_pad"].unsqueeze(-1)
    numerator = (abs_err * mask).sum(dim=1)
    denominator = mask.sum(dim=1).clamp_min(1)
    return numerator / denominator


def action_l1_per_horizon(batch: dict[str, Any], actions_hat: torch.Tensor) -> torch.Tensor:
    abs_err = (batch[ACTION] - actions_hat).abs().mean(dim=-1)
    valid = ~batch["action_is_pad"]
    numerator = (abs_err * valid).sum(dim=0)
    denominator = valid.sum(dim=0).clamp_min(1)
    return numerator / denominator


def chunk_smoothness(actions: torch.Tensor) -> torch.Tensor:
    if actions.shape[1] < 2:
        return torch.zeros(actions.shape[0], device=actions.device)
    delta = actions[:, 1:, :6] - actions[:, :-1, :6]
    return torch.linalg.norm(delta, dim=-1).mean(dim=1)


def prepare_batch(batch: dict[str, Any], preprocessor, camera_keys: list[str]) -> dict[str, Any]:
    batch = _move_uint8_images_to_float(batch, camera_keys)
    return preprocessor(batch)


def load_policy_and_data(spec: ModelSpec, checkpoint_name: str = "checkpoint"):
    cfg, data_cfg, policy_cfg, ds_meta = load_train_context(spec)
    checkpoint = spec.run_dir / checkpoint_name
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing model checkpoint for {spec.key}: {checkpoint}")
    policy = ACTLangPolicy.from_pretrained(checkpoint, config=policy_cfg, local_files_only=True, strict=True)

    train_eps = _parse_episode_spec(data_cfg["train_episodes"], ds_meta.total_episodes)
    val_eps = _parse_episode_spec(data_cfg["val_episodes"], ds_meta.total_episodes)
    train_ds = make_dataset(data_cfg, policy_cfg, train_eps)
    val_ds = make_dataset(data_cfg, policy_cfg, val_eps)
    if data_cfg.get("use_imagenet_stats", True):
        _apply_imagenet_stats(train_ds)
        _apply_imagenet_stats(val_ds)
    preprocessor, _ = make_act_pre_post_processors(config=policy.config, dataset_stats=train_ds.meta.stats)
    return cfg, policy, preprocessor, train_ds, val_ds


def plot_training_core(specs: list[ModelSpec], core_fig: Path, core_tab: Path, smoothing_window: int) -> None:
    summary_rows = []
    gap_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), dpi=180, constrained_layout=True)
    colors = ["#2f6f9f", "#b23a48", "#4e8f5a", "#8c5fbf"]

    for idx, spec in enumerate(specs):
        metrics = load_metrics_for_spec(spec)
        train_steps, train_l1 = finite_points(metrics, "train_action_l1")
        val_steps, val_l1 = finite_points(metrics, "val_action_l1")
        if len(train_steps) == 0 or len(val_steps) == 0:
            continue
        smooth = trailing_mean(train_l1, smoothing_window)
        color = colors[idx % len(colors)]
        axes[0].plot(train_steps, smooth, color=color, linewidth=1.1, alpha=0.92, label=f"{spec.display} train")
        axes[0].plot(val_steps, val_l1, color=color, linewidth=1.0, linestyle="--", alpha=0.92, label=f"{spec.display} val")

        smooth_by_step = {int(step): float(value) for step, value in zip(train_steps, smooth, strict=True)}
        gap_steps = []
        gaps = []
        for row in metrics:
            step = int(float(row["step"]))
            val = fval(row.get("val_action_l1"))
            if not math.isfinite(val) or step not in smooth_by_step:
                continue
            gap = val - smooth_by_step[step]
            gap_steps.append(step)
            gaps.append(gap)
            gap_rows.append(
                {
                    "experiment": spec.key,
                    "display_name": spec.display,
                    "step": step,
                    "val_action_l1": val,
                    "train_action_l1_trailing_mean": smooth_by_step[step],
                    "val_minus_train_action_l1_smoothed": gap,
                }
            )
        axes[1].plot(gap_steps, gaps, color=color, linewidth=1.2, label=spec.display)

        best_idx = int(np.argmin(val_l1))
        final = metrics[-1]
        summary_rows.append(
            {
                "experiment": spec.key,
                "display_name": spec.display,
                "run_dir": str(spec.run_dir),
                "dataset_group": spec.dataset_group,
                "final_step": int(float(final["step"])),
                "final_train_action_l1": fval(final.get("train_action_l1")),
                "final_val_action_l1": fval(final.get("val_action_l1")),
                "best_val_step": int(val_steps[best_idx]),
                "best_val_action_l1": float(val_l1[best_idx]),
                "final_minus_best_val_action_l1": fval(final.get("val_action_l1")) - float(val_l1[best_idx]),
            }
        )

    axes[0].set_title("ACT-Lang Training and Validation Action L1")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Action L1")
    axes[0].grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.7)
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    axes[1].axhline(0, color="#222222", linewidth=0.8)
    axes[1].set_title("ACT-Lang Train-Val Generalization Gap")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Validation L1 - smoothed train L1")
    axes[1].grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.7)
    axes[1].legend(frameon=False, fontsize=8)
    save_fig(fig, core_fig / "act_lang_loss_and_gap.png")

    write_rows(core_tab / "training_summary.csv", summary_rows)
    write_latex(core_tab / "training_summary.tex", summary_rows)
    write_rows(core_tab / "train_val_gap.csv", gap_rows)
    plot_training_supplementary(specs, summary_rows, core_fig, core_tab)


def plot_training_supplementary(
    specs: list[ModelSpec],
    summary_rows: list[dict[str, Any]],
    core_fig: Path,
    core_tab: Path,
) -> None:
    if not summary_rows:
        return

    labels = [row["display_name"] for row in summary_rows]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=180, constrained_layout=True)
    best = [float(row["best_val_action_l1"]) for row in summary_rows]
    final = [float(row["final_val_action_l1"]) for row in summary_rows]
    axes[0].bar(x - 0.18, best, width=0.36, color="#2f6f9f", label="Best val")
    axes[0].bar(x + 0.18, final, width=0.36, color="#b23a48", label="Final val")
    axes[0].set_xticks(x, labels=labels, rotation=25, ha="right", fontsize=7)
    axes[0].set_ylabel("Validation Action L1")
    axes[0].set_title("ACT-Lang Best vs Final Validation")
    axes[0].grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    axes[0].legend(frameon=False, fontsize=8)

    gaps = [float(row["final_minus_best_val_action_l1"]) for row in summary_rows]
    axes[1].bar(x, gaps, color="#7a6f3d")
    axes[1].axhline(0, color="#222222", linewidth=0.8)
    axes[1].set_xticks(x, labels=labels, rotation=25, ha="right", fontsize=7)
    axes[1].set_ylabel("Final val - best val")
    axes[1].set_title("End-of-Run Overfit / Underfit Indicator")
    axes[1].grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    save_fig(fig, core_fig / "act_lang_validation_summary_bars.png")

    step_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), dpi=180, constrained_layout=True)
    colors = ["#2f6f9f", "#b23a48", "#4e8f5a", "#8c5fbf"]
    for idx, spec in enumerate(specs):
        metrics = load_metrics_for_spec(spec)
        steps, step_s = finite_points(metrics, "step_s")
        if len(steps) == 0:
            continue
        finite_step_s = step_s[np.isfinite(step_s)]
        q95 = float(np.quantile(finite_step_s, 0.95))
        clipped = np.clip(step_s, 0, q95)
        smooth = trailing_mean(clipped, 100)
        color = colors[idx % len(colors)]
        axes[0].plot(steps, smooth, color=color, linewidth=1.1, label=spec.display)
        step_rows.append(
            {
                "experiment": spec.key,
                "display_name": spec.display,
                "median_step_s": float(np.median(finite_step_s)),
                "mean_step_s": float(np.mean(finite_step_s)),
                "q95_step_s": q95,
                "max_step_s": float(np.max(finite_step_s)),
            }
        )

        val_steps, val_l1 = finite_points(metrics, "val_action_l1")
        if len(val_steps) > 1:
            axes[1].plot(val_steps, np.gradient(val_l1), color=color, linewidth=1.1, label=spec.display)

    axes[0].set_title("Step-Time Profile, 100-Step Mean")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Step time, seconds")
    axes[0].grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.7)
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].axhline(0, color="#222222", linewidth=0.8)
    axes[1].set_title("Validation L1 Local Slope")
    axes[1].set_xlabel("Validation step")
    axes[1].set_ylabel("Delta val L1 per validation point")
    axes[1].grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.7)
    axes[1].legend(frameon=False, fontsize=7)
    save_fig(fig, core_fig / "act_lang_runtime_and_convergence.png")
    write_rows(core_tab / "step_time_summary.csv", step_rows)
    write_latex(core_tab / "step_time_summary.tex", step_rows)


def dataset_action_and_task_core(specs: list[ModelSpec], core_fig: Path, core_tab: Path, max_samples: int) -> None:
    dataset_specs = []
    seen = set()
    for spec in specs:
        if spec.dataset_group in seen:
            continue
        seen.add(spec.dataset_group)
        dataset_specs.append(spec)

    smooth_rows = []
    task_rows = []
    dataset_arrays: dict[str, np.ndarray] = {}
    dataset_task_counters: dict[str, Counter[str]] = {}
    dataset_language_embeddings: dict[str, np.ndarray] = {}
    dataset_language_tasks: dict[str, list[str]] = {}
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), dpi=180, constrained_layout=True)
    colors = {"B": "#2f6f9f", "ABC": "#b23a48"}
    for dataset_idx, spec in enumerate(dataset_specs, start=1):
        progress(
            "core_dataset_profile",
            dataset_idx - 1,
            len(dataset_specs),
            f"reading {spec.dataset_group} parquet",
        )
        _, data_cfg, policy_cfg, ds_meta = load_train_context(spec)
        train_eps = _parse_episode_spec(data_cfg["train_episodes"], ds_meta.total_episodes)
        task_map = task_index_to_text(data_cfg["root"])
        sample = sample_parquet_rows(
            data_cfg["root"],
            train_eps,
            ["action", "task_index", "observation.language_embedding"],
            max_samples,
        )
        action_arr = stack_series(sample["action"]).astype(np.float64)
        task_counter: Counter[str] = Counter()
        task_names = []
        for task_index in sample["task_index"].to_numpy():
            task_name = task_map.get(int(task_index), f"task_{int(task_index)}")
            task_names.append(task_name)
            task_counter[task_name] += 1
        dataset_arrays[spec.dataset_group] = action_arr
        dataset_task_counters[spec.dataset_group] = task_counter
        dataset_language_embeddings[spec.dataset_group] = stack_series(sample["observation.language_embedding"]).astype(np.float64)
        dataset_language_tasks[spec.dataset_group] = task_names
        first6 = action_arr[:, :6]
        delta = np.linalg.norm(np.diff(first6, axis=0), axis=1)
        axes[0].hist(delta, bins=40, histtype="step", linewidth=1.35, density=True, color=colors[spec.dataset_group], label=spec.dataset_group)
        smooth_rows.append(
            {
                "dataset_group": spec.dataset_group,
                "sampled_frames": int(len(action_arr)),
                "mean_action_l2_first6": float(np.linalg.norm(first6, axis=1).mean()),
                "mean_step_delta_l2_first6": float(delta.mean()),
                "q90_step_delta_l2_first6": float(np.quantile(delta, 0.9)),
                "gripper_close_fraction": float((action_arr[:, 6] < 0).mean()),
            }
        )
        for task, count in task_counter.most_common(20):
            task_rows.append({"dataset_group": spec.dataset_group, "task": task, "sampled_count": count})
        progress("core_dataset_profile", dataset_idx, len(dataset_specs), f"done {spec.dataset_group}")

    top_tasks = task_rows[:16]
    labels = [short_label(row["task"], 30) for row in top_tasks]
    values = [row["sampled_count"] for row in top_tasks]
    groups = [row["dataset_group"] for row in top_tasks]
    bar_colors = [colors.get(group, "#777777") for group in groups]
    axes[1].barh(np.arange(len(labels)), values, color=bar_colors)
    axes[1].set_yticks(np.arange(len(labels)), labels=labels, fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Sampled frames")
    axes[1].set_title("Top Language Tasks in Sampled Training Frames")
    axes[0].set_xlabel("L2 step delta, first 6 action dimensions")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Action Smoothness on Language Datasets")
    axes[0].grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.7)
    axes[0].legend(frameon=False)
    save_fig(fig, core_fig / "act_lang_dataset_action_and_task_profile.png")
    write_rows(core_tab / "dataset_action_smoothness.csv", smooth_rows)
    write_latex(core_tab / "dataset_action_smoothness.tex", smooth_rows)
    write_rows(core_tab / "task_frequency_sampled.csv", task_rows)
    plot_dataset_supplementary(dataset_arrays, dataset_task_counters, dataset_language_embeddings, dataset_language_tasks, core_fig, core_tab)


def plot_dataset_supplementary(
    dataset_arrays: dict[str, np.ndarray],
    task_counters: dict[str, Counter[str]],
    language_embeddings: dict[str, np.ndarray],
    language_tasks: dict[str, list[str]],
    core_fig: Path,
    core_tab: Path,
) -> None:
    if not dataset_arrays:
        return

    action_dim_rows = []
    dims = [f"a{i}" for i in range(7)]
    fig, axes = plt.subplots(1, len(dataset_arrays), figsize=(6.2 * len(dataset_arrays), 4.8), dpi=180, constrained_layout=True)
    if len(dataset_arrays) == 1:
        axes = [axes]
    for ax, (group, arr) in zip(axes, dataset_arrays.items(), strict=False):
        data = [arr[:, dim] for dim in range(7)]
        parts = ax.violinplot(data, showmeans=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor("#2f6f9f" if group == "B" else "#b23a48")
            body.set_alpha(0.55)
        if "cmeans" in parts:
            parts["cmeans"].set_color("#222222")
            parts["cmeans"].set_linewidth(0.8)
        ax.set_xticks(np.arange(1, 8), labels=dims)
        ax.set_title(f"{group}: Action-Dimension Distribution")
        ax.set_ylabel("Action value")
        ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
        for dim in range(7):
            values = arr[:, dim]
            action_dim_rows.append(
                {
                    "dataset_group": group,
                    "action_dim": dim,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "q10": float(np.quantile(values, 0.1)),
                    "q50": float(np.quantile(values, 0.5)),
                    "q90": float(np.quantile(values, 0.9)),
                }
            )
    save_fig(fig, core_fig / "act_lang_action_dimension_violin.png")
    write_rows(core_tab / "action_dimension_distribution.csv", action_dim_rows)

    fig, axes = plt.subplots(1, len(dataset_arrays), figsize=(5.8 * len(dataset_arrays), 4.8), dpi=180, constrained_layout=True)
    if len(dataset_arrays) == 1:
        axes = [axes]
    for ax, (group, arr) in zip(axes, dataset_arrays.items(), strict=False):
        corr = np.corrcoef(arr[:, :7], rowvar=False)
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(np.arange(7), labels=dims)
        ax.set_yticks(np.arange(7), labels=dims)
        ax.set_title(f"{group}: Action Correlation")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_fig(fig, core_fig / "act_lang_action_correlation_matrices.png")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5), dpi=180, constrained_layout=True)
    for group, arr in dataset_arrays.items():
        color = "#2f6f9f" if group == "B" else "#b23a48"
        axes[0].hist(arr[:, 6], bins=40, histtype="step", linewidth=1.35, density=True, color=color, label=group)
        axes[1].hist(np.linalg.norm(arr[:, :6], axis=1), bins=40, histtype="step", linewidth=1.35, density=True, color=color, label=group)
    axes[0].set_title("Gripper Action Distribution")
    axes[0].set_xlabel("Action dimension 6")
    axes[0].set_ylabel("Density")
    axes[1].set_title("End-Effector Action Norm Distribution")
    axes[1].set_xlabel("L2 norm, first 6 dimensions")
    for ax in axes:
        ax.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.7)
        ax.legend(frameon=False)
    save_fig(fig, core_fig / "act_lang_gripper_and_action_norm_distribution.png")

    for group, counter in task_counters.items():
        rows = [{"dataset_group": group, "task": task, "sampled_count": count} for task, count in counter.most_common(30)]
        labels = [short_label(row["task"], 42) for row in rows]
        values = [row["sampled_count"] for row in rows]
        fig, ax = plt.subplots(figsize=(8.0, 7.2), dpi=180, constrained_layout=True)
        ax.barh(np.arange(len(labels)), values, color="#2f6f9f" if group == "B" else "#b23a48")
        ax.set_yticks(np.arange(len(labels)), labels=labels, fontsize=6.5)
        ax.invert_yaxis()
        ax.set_xlabel("Sampled frames")
        ax.set_title(f"{group}: Top Language Task Frequency")
        ax.grid(True, axis="x", color="#d0d0d0", linewidth=0.5, alpha=0.7)
        save_fig(fig, core_fig / f"act_lang_task_frequency_{group}.png")

    keyword_rows = []
    keywords = [
        "open",
        "close",
        "slide",
        "lift",
        "place",
        "put",
        "move",
        "turn",
        "push",
        "pull",
        "rotate",
        "grasp",
        "block",
        "drawer",
        "switch",
        "lamp",
        "door",
    ]
    for group, tasks in language_tasks.items():
        for keyword in keywords:
            keyword_rows.append(
                {
                    "dataset_group": group,
                    "keyword": keyword,
                    "sampled_count": sum(1 for task in tasks if keyword in task.lower()),
                }
            )
    write_rows(core_tab / "task_keyword_profile.csv", keyword_rows)
    if keyword_rows:
        fig, ax = plt.subplots(figsize=(9.6, 4.8), dpi=180, constrained_layout=True)
        groups = list(dataset_arrays)
        x = np.arange(len(keywords))
        width = 0.8 / max(1, len(groups))
        for idx, group in enumerate(groups):
            vals = [row["sampled_count"] for row in keyword_rows if row["dataset_group"] == group]
            ax.bar(x + (idx - (len(groups) - 1) / 2) * width, vals, width=width, label=group)
        ax.set_xticks(x, labels=keywords, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel("Sampled frame count")
        ax.set_title("Language Task Keyword Profile")
        ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
        ax.legend(frameon=False)
        save_fig(fig, core_fig / "act_lang_task_keyword_profile.png")

    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=180, constrained_layout=True)
    embedding_rows = []
    for group, emb in language_embeddings.items():
        norms = np.linalg.norm(emb, axis=1)
        ax.hist(norms, bins=35, histtype="step", linewidth=1.35, density=True, label=group)
        embedding_rows.append(
            {
                "dataset_group": group,
                "mean_embedding_norm": float(np.mean(norms)),
                "std_embedding_norm": float(np.std(norms)),
                "q10_embedding_norm": float(np.quantile(norms, 0.1)),
                "q90_embedding_norm": float(np.quantile(norms, 0.9)),
            }
        )
    ax.set_title("Language Embedding Norm Distribution")
    ax.set_xlabel("Embedding L2 norm")
    ax.set_ylabel("Density")
    ax.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=False)
    save_fig(fig, core_fig / "act_lang_language_embedding_norms.png")
    write_rows(core_tab / "language_embedding_norms.csv", embedding_rows)
    write_latex(core_tab / "language_embedding_norms.tex", embedding_rows)


@torch.no_grad()
def evaluate_language_effects(
    specs: list[ModelSpec],
    lang_fig: Path,
    lang_tab: Path,
    max_eval_samples: int,
    batch_size: int,
) -> None:
    ablation_rows = []
    per_task_rows = []
    dim_error_rows = []
    horizon_error_rows = []
    sensitivity_dim_rows = []
    sensitivity_horizon_rows = []
    smoothness_rows = []
    norm_scatter_rows = []
    representative_done = False

    for spec in specs:
        if not (spec.run_dir / "checkpoint/model.safetensors").is_file():
            continue
        progress("language_model_eval", 0, max(1, max_eval_samples), spec.key)
        cfg, policy, preprocessor, _, val_ds = load_policy_and_data(spec)
        policy.eval()
        camera_keys = list(val_ds.meta.camera_keys)
        loader = DataLoader(sample_subset(val_ds, max_eval_samples), batch_size=batch_size, shuffle=False, num_workers=0)

        per_task_values: dict[str, list[float]] = defaultdict(list)
        ablation_values = {"correct": [], "zero_language": [], "wrong_language": []}
        dim_error_values: dict[str, list[np.ndarray]] = {"correct": [], "zero_language": [], "wrong_language": []}
        horizon_error_values: dict[str, list[np.ndarray]] = {"correct": [], "zero_language": [], "wrong_language": []}
        sensitivity_dim_values: dict[str, list[np.ndarray]] = {"zero_language": [], "wrong_language": []}
        sensitivity_horizon_values: dict[str, list[np.ndarray]] = {"zero_language": [], "wrong_language": []}
        smoothness_values: dict[str, list[np.ndarray]] = {"gt": [], "correct": [], "zero_language": [], "wrong_language": []}
        selected_batch = None
        processed = 0
        for batch_idx, batch in enumerate(loader, start=1):
            if selected_batch is None:
                selected_batch = batch
            prepared = prepare_batch(batch, preprocessor, camera_keys)
            correct = policy.predict_action_chunk(prepared)
            l1_correct = action_l1_per_item(prepared, correct)
            zero_batch = dict(prepared)
            zero_batch[policy.config.language_embedding_key] = torch.zeros_like(prepared[policy.config.language_embedding_key])
            zero = policy.predict_action_chunk(zero_batch)
            l1_zero = action_l1_per_item(prepared, zero)
            wrong_batch = dict(prepared)
            wrong_batch[policy.config.language_embedding_key] = torch.roll(
                prepared[policy.config.language_embedding_key], shifts=1, dims=0
            )
            wrong = policy.predict_action_chunk(wrong_batch)
            l1_wrong = action_l1_per_item(prepared, wrong)

            preds = {"correct": correct, "zero_language": zero, "wrong_language": wrong}
            for mode, pred in preds.items():
                dim_error_values[mode].append(action_l1_per_dim(prepared, pred).detach().cpu().numpy())
                horizon_error_values[mode].append(action_l1_per_horizon(prepared, pred).detach().cpu().numpy())
                smoothness_values[mode].append(chunk_smoothness(pred).detach().cpu().numpy())
            smoothness_values["gt"].append(chunk_smoothness(prepared[ACTION]).detach().cpu().numpy())

            for mode, pred in [("zero_language", zero), ("wrong_language", wrong)]:
                diff = (correct - pred).abs()
                sensitivity_dim_values[mode].append(diff.mean(dim=1).detach().cpu().numpy())
                sensitivity_horizon_values[mode].append(diff.mean(dim=-1).mean(dim=0).detach().cpu().numpy())

            gt_norm = torch.linalg.norm(prepared[ACTION][:, 0, :6], dim=-1).detach().cpu().numpy()
            pred_norm = torch.linalg.norm(correct[:, 0, :6], dim=-1).detach().cpu().numpy()
            for task, gt_value, pred_value in zip(batch["task"], gt_norm, pred_norm, strict=True):
                norm_scatter_rows.append(
                    {
                        "experiment": spec.key,
                        "display_name": spec.display,
                        "task": str(task),
                        "gt_action_norm_first_step_first6": float(gt_value),
                        "pred_action_norm_first_step_first6": float(pred_value),
                    }
                )

            for task, value in zip(batch["task"], l1_correct.detach().cpu().numpy(), strict=True):
                per_task_values[str(task)].append(float(value))
            for mode, values in [
                ("correct", l1_correct),
                ("zero_language", l1_zero),
                ("wrong_language", l1_wrong),
            ]:
                for value in values.detach().cpu().numpy():
                    ablation_values[mode].append(float(value))
                    ablation_rows.append(
                        {
                            "experiment": spec.key,
                            "display_name": spec.display,
                            "mode": mode,
                            "action_l1": float(value),
                        }
                    )
            processed += len(batch["task"])
            progress("language_model_eval", processed, max_eval_samples, f"{spec.key} batch={batch_idx}")

        for mode, values in dim_error_values.items():
            if not values:
                continue
            arr = np.concatenate(values, axis=0)
            for dim in range(arr.shape[1]):
                dim_error_rows.append(
                    {
                        "experiment": spec.key,
                        "display_name": spec.display,
                        "mode": mode,
                        "action_dim": dim,
                        "mean_action_l1": float(arr[:, dim].mean()),
                        "std_action_l1": float(arr[:, dim].std()),
                        "q90_action_l1": float(np.quantile(arr[:, dim], 0.9)),
                    }
                )
        for mode, values in horizon_error_values.items():
            if not values:
                continue
            arr = np.stack(values, axis=0).mean(axis=0)
            for chunk_t, value in enumerate(arr):
                horizon_error_rows.append(
                    {
                        "experiment": spec.key,
                        "display_name": spec.display,
                        "mode": mode,
                        "chunk_t": chunk_t,
                        "mean_action_l1": float(value),
                    }
                )
        for mode, values in sensitivity_dim_values.items():
            if not values:
                continue
            arr = np.concatenate(values, axis=0)
            for dim in range(arr.shape[1]):
                sensitivity_dim_rows.append(
                    {
                        "experiment": spec.key,
                        "display_name": spec.display,
                        "ablation_mode": mode,
                        "action_dim": dim,
                        "mean_pred_l1_delta": float(arr[:, dim].mean()),
                        "q90_pred_l1_delta": float(np.quantile(arr[:, dim], 0.9)),
                    }
                )
        for mode, values in sensitivity_horizon_values.items():
            if not values:
                continue
            arr = np.stack(values, axis=0).mean(axis=0)
            for chunk_t, value in enumerate(arr):
                sensitivity_horizon_rows.append(
                    {
                        "experiment": spec.key,
                        "display_name": spec.display,
                        "ablation_mode": mode,
                        "chunk_t": chunk_t,
                        "mean_pred_l1_delta": float(value),
                    }
                )
        for mode, values in smoothness_values.items():
            if not values:
                continue
            arr = np.concatenate(values, axis=0)
            smoothness_rows.append(
                {
                    "experiment": spec.key,
                    "display_name": spec.display,
                    "mode": mode,
                    "mean_chunk_delta_l2_first6": float(arr.mean()),
                    "std_chunk_delta_l2_first6": float(arr.std()),
                    "q90_chunk_delta_l2_first6": float(np.quantile(arr, 0.9)),
                }
            )

        for task, values in per_task_values.items():
            per_task_rows.append(
                {
                    "experiment": spec.key,
                    "display_name": spec.display,
                    "task": task,
                    "num_samples": len(values),
                    "mean_action_l1": float(np.mean(values)),
                    "std_action_l1": float(np.std(values)),
                }
            )

        plot_ablation_for_model(spec, ablation_values, lang_fig)
        plot_per_task_for_model(spec, per_task_values, lang_fig)
        plot_model_prediction_supplementary(
            spec,
            dim_error_values,
            horizon_error_values,
            sensitivity_dim_values,
            sensitivity_horizon_values,
            smoothness_values,
            lang_fig,
        )

        if selected_batch is not None:
            progress("language_model_eval", max_eval_samples, max_eval_samples, f"{spec.key} plotting language diagnostics")
            plot_prompt_conditioning(spec, policy, preprocessor, selected_batch, camera_keys, lang_fig, lang_tab)
            plot_language_distance_matrix(spec, policy, preprocessor, selected_batch, camera_keys, lang_fig, lang_tab)
            if not representative_done:
                plot_representative_strip(spec, policy, preprocessor, val_ds, camera_keys, lang_fig, lang_tab)
                representative_done = True

    write_rows(lang_tab / "language_ablation_action_l1.csv", ablation_rows)
    write_latex(lang_tab / "language_ablation_action_l1.tex", summarize_ablation(ablation_rows))
    write_rows(lang_tab / "per_task_validation_action_l1.csv", per_task_rows)
    top_per_task = sorted(per_task_rows, key=lambda r: r["mean_action_l1"], reverse=True)[:30]
    write_rows(lang_tab / "per_task_validation_action_l1_top.csv", top_per_task)
    write_latex(lang_tab / "per_task_validation_action_l1_top.tex", top_per_task)
    write_rows(lang_tab / "per_action_dim_error.csv", dim_error_rows)
    write_latex(lang_tab / "per_action_dim_error.tex", dim_error_rows)
    write_rows(lang_tab / "chunk_horizon_error.csv", horizon_error_rows)
    write_rows(lang_tab / "language_sensitivity_by_action_dim.csv", sensitivity_dim_rows)
    write_latex(lang_tab / "language_sensitivity_by_action_dim.tex", sensitivity_dim_rows)
    write_rows(lang_tab / "language_sensitivity_by_chunk_horizon.csv", sensitivity_horizon_rows)
    write_rows(lang_tab / "predicted_chunk_smoothness.csv", smoothness_rows)
    write_latex(lang_tab / "predicted_chunk_smoothness.tex", smoothness_rows)
    write_rows(lang_tab / "pred_vs_gt_action_norm.csv", norm_scatter_rows)
    plot_language_ablation_summary(ablation_rows, lang_fig)
    plot_per_task_summary(per_task_rows, lang_fig)
    plot_language_prediction_summaries(
        dim_error_rows,
        horizon_error_rows,
        sensitivity_dim_rows,
        sensitivity_horizon_rows,
        smoothness_rows,
        norm_scatter_rows,
        lang_fig,
    )


def summarize_ablation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    display_by_key = {}
    for row in rows:
        grouped[(row["experiment"], row["display_name"], row["mode"])].append(float(row["action_l1"]))
        display_by_key[row["experiment"]] = row["display_name"]
    out = []
    for (exp, display, mode), values in grouped.items():
        out.append(
            {
                "experiment": exp,
                "display_name": display,
                "mode": mode,
                "num_samples": len(values),
                "mean_action_l1": float(np.mean(values)),
                "median_action_l1": float(np.median(values)),
            }
        )
    return sorted(out, key=lambda row: (row["experiment"], row["mode"]))


def plot_ablation_for_model(spec: ModelSpec, values_by_mode: dict[str, list[float]], lang_fig: Path) -> None:
    modes = ["correct", "zero_language", "wrong_language"]
    values = [values_by_mode[mode] for mode in modes]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=180, constrained_layout=True)
    box = ax.boxplot(values, patch_artist=True, tick_labels=["Correct", "Zero", "Wrong"], showfliers=False)
    for patch, color in zip(box["boxes"], ["#2f6f9f", "#b7a4d9", "#b23a48"], strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    ax.set_title(f"{spec.display}: Language Ablation")
    ax.set_ylabel("Validation Action L1")
    ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    save_fig(fig, lang_fig / f"{spec.key}_language_ablation_boxplot.png")


def plot_language_ablation_summary(rows: list[dict[str, Any]], lang_fig: Path) -> None:
    summary = summarize_ablation(rows)
    if not summary:
        return
    labels = []
    values = []
    colors = []
    color_by_mode = {"correct": "#2f6f9f", "zero_language": "#b7a4d9", "wrong_language": "#b23a48"}
    for row in summary:
        labels.append(f"{row['display_name']}\n{row['mode'].replace('_', ' ')}")
        values.append(row["mean_action_l1"])
        colors.append(color_by_mode[row["mode"]])
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(labels)), 4.8), dpi=180, constrained_layout=True)
    ax.bar(np.arange(len(labels)), values, color=colors)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Mean validation Action L1")
    ax.set_title("Correct vs Zero vs Wrong Language Conditioning")
    ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    save_fig(fig, lang_fig / "language_ablation_summary.png")


def plot_per_task_for_model(spec: ModelSpec, values_by_task: dict[str, list[float]], lang_fig: Path) -> None:
    rows = sorted(
        [
            (task, len(values), float(np.mean(values)))
            for task, values in values_by_task.items()
            if len(values) >= 2
        ],
        key=lambda item: item[2],
        reverse=True,
    )[:18]
    if not rows:
        return
    labels = [short_label(task, 36) for task, _, _ in rows]
    values = [value for _, _, value in rows]
    fig, ax = plt.subplots(figsize=(7.6, 6.2), dpi=180, constrained_layout=True)
    ax.barh(np.arange(len(labels)), values, color="#b23a48")
    ax.set_yticks(np.arange(len(labels)), labels=labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Mean validation Action L1")
    ax.set_title(f"{spec.display}: Hardest Language Tasks")
    ax.grid(True, axis="x", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    save_fig(fig, lang_fig / f"{spec.key}_per_task_validation_l1.png")


def plot_per_task_summary(rows: list[dict[str, Any]], lang_fig: Path) -> None:
    if not rows:
        return
    eligible = [row for row in rows if int(row["num_samples"]) >= 2]
    top = sorted(eligible, key=lambda r: r["mean_action_l1"], reverse=True)[:24]
    if not top:
        return
    labels = [short_label(f"{row['display_name']}: {row['task']}", 48) for row in top]
    values = [row["mean_action_l1"] for row in top]
    fig, ax = plt.subplots(figsize=(8.8, 7.4), dpi=180, constrained_layout=True)
    ax.barh(np.arange(len(labels)), values, color="#7a6f3d")
    ax.set_yticks(np.arange(len(labels)), labels=labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("Mean validation Action L1")
    ax.set_title("Hardest Sampled Language Tasks Across ACT-Lang Models")
    ax.grid(True, axis="x", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    save_fig(fig, lang_fig / "per_task_validation_l1_summary.png")


def mean_stack(values: list[np.ndarray], *, concat: bool) -> np.ndarray | None:
    if not values:
        return None
    if concat:
        return np.concatenate(values, axis=0)
    return np.stack(values, axis=0).mean(axis=0)


def plot_model_prediction_supplementary(
    spec: ModelSpec,
    dim_error_values: dict[str, list[np.ndarray]],
    horizon_error_values: dict[str, list[np.ndarray]],
    sensitivity_dim_values: dict[str, list[np.ndarray]],
    sensitivity_horizon_values: dict[str, list[np.ndarray]],
    smoothness_values: dict[str, list[np.ndarray]],
    lang_fig: Path,
) -> None:
    modes = ["correct", "zero_language", "wrong_language"]
    mode_labels = {"correct": "Correct", "zero_language": "Zero lang", "wrong_language": "Wrong lang"}
    colors = {"correct": "#2f6f9f", "zero_language": "#b7a4d9", "wrong_language": "#b23a48", "gt": "#222222"}

    fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=180, constrained_layout=True)
    x = np.arange(7)
    width = 0.24
    for idx, mode in enumerate(modes):
        arr = mean_stack(dim_error_values[mode], concat=True)
        if arr is None:
            continue
        ax.bar(x + (idx - 1) * width, arr.mean(axis=0), width=width, color=colors[mode], label=mode_labels[mode])
    ax.set_xticks(x, labels=[f"a{i}" for i in range(7)])
    ax.set_ylabel("Mean Action L1")
    ax.set_title(f"{spec.display}: Per-Action-Dimension Error")
    ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=False, fontsize=8)
    save_fig(fig, lang_fig / f"{spec.key}_per_action_dim_error.png")

    fig, ax = plt.subplots(figsize=(7.8, 4.4), dpi=180, constrained_layout=True)
    for mode in modes:
        arr = mean_stack(horizon_error_values[mode], concat=False)
        if arr is None:
            continue
        ax.plot(np.arange(len(arr)), arr, linewidth=1.25, color=colors[mode], label=mode_labels[mode])
    ax.set_xlabel("Chunk timestep")
    ax.set_ylabel("Action L1")
    ax.set_title(f"{spec.display}: Chunk-Horizon Error")
    ax.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=False, fontsize=8)
    save_fig(fig, lang_fig / f"{spec.key}_chunk_horizon_error.png")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), dpi=180, constrained_layout=True)
    for mode, color in [("zero_language", "#7b62a3"), ("wrong_language", "#b23a48")]:
        arr = mean_stack(sensitivity_dim_values[mode], concat=True)
        if arr is not None:
            axes[0].plot(np.arange(arr.shape[1]), arr.mean(axis=0), marker="o", linewidth=1.2, color=color, label=mode_labels[mode])
        harr = mean_stack(sensitivity_horizon_values[mode], concat=False)
        if harr is not None:
            axes[1].plot(np.arange(len(harr)), harr, linewidth=1.2, color=color, label=mode_labels[mode])
    axes[0].set_xticks(np.arange(7), labels=[f"a{i}" for i in range(7)])
    axes[0].set_ylabel("Mean |pred(correct)-pred(ablation)|")
    axes[0].set_title("Language Sensitivity by Action Dim")
    axes[1].set_xlabel("Chunk timestep")
    axes[1].set_ylabel("Mean |pred(correct)-pred(ablation)|")
    axes[1].set_title("Language Sensitivity by Chunk Horizon")
    for ax in axes:
        ax.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.7)
        ax.legend(frameon=False, fontsize=8)
    save_fig(fig, lang_fig / f"{spec.key}_language_sensitivity_profiles.png")

    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=180, constrained_layout=True)
    labels = []
    values = []
    bar_colors = []
    for mode in ["gt", *modes]:
        arr = mean_stack(smoothness_values[mode], concat=True)
        if arr is None:
            continue
        labels.append(mode_labels.get(mode, "GT"))
        values.append(float(arr.mean()))
        bar_colors.append(colors[mode])
    ax.bar(np.arange(len(labels)), values, color=bar_colors)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=20, ha="right")
    ax.set_ylabel("Mean chunk delta L2, first 6 dims")
    ax.set_title(f"{spec.display}: Predicted Chunk Smoothness")
    ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    save_fig(fig, lang_fig / f"{spec.key}_predicted_chunk_smoothness.png")


def rows_to_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def plot_language_prediction_summaries(
    dim_error_rows: list[dict[str, Any]],
    horizon_error_rows: list[dict[str, Any]],
    sensitivity_dim_rows: list[dict[str, Any]],
    sensitivity_horizon_rows: list[dict[str, Any]],
    smoothness_rows: list[dict[str, Any]],
    norm_scatter_rows: list[dict[str, Any]],
    lang_fig: Path,
) -> None:
    dim_df = rows_to_df(dim_error_rows)
    if not dim_df.empty:
        correct = dim_df[dim_df["mode"] == "correct"]
        pivot = correct.pivot(index="display_name", columns="action_dim", values="mean_action_l1")
        fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=180, constrained_layout=True)
        im = ax.imshow(pivot.to_numpy(dtype=float), cmap="viridis")
        ax.set_xticks(np.arange(pivot.shape[1]), labels=[f"a{i}" for i in pivot.columns])
        ax.set_yticks(np.arange(pivot.shape[0]), labels=list(pivot.index), fontsize=7)
        ax.set_title("Correct-Language Error by Action Dimension")
        fig.colorbar(im, ax=ax, label="Mean Action L1")
        save_fig(fig, lang_fig / "summary_correct_language_per_action_dim_error_heatmap.png")

        fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=180, constrained_layout=True)
        for mode, style in [("correct", "-"), ("zero_language", "--"), ("wrong_language", ":")]:
            mode_df = dim_df[dim_df["mode"] == mode]
            grouped = mode_df.groupby("action_dim")["mean_action_l1"].mean()
            ax.plot(grouped.index, grouped.values, linestyle=style, marker="o", linewidth=1.2, label=mode.replace("_", " "))
        ax.set_xticks(np.arange(7), labels=[f"a{i}" for i in range(7)])
        ax.set_ylabel("Mean Action L1 across models")
        ax.set_title("Language Ablation Error Profile by Action Dimension")
        ax.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.7)
        ax.legend(frameon=False)
        save_fig(fig, lang_fig / "summary_ablation_error_by_action_dim.png")

    horizon_df = rows_to_df(horizon_error_rows)
    if not horizon_df.empty:
        fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=180, constrained_layout=True)
        for (display, mode), group in horizon_df.groupby(["display_name", "mode"]):
            if mode != "correct":
                continue
            ax.plot(group["chunk_t"], group["mean_action_l1"], linewidth=1.1, label=display)
        ax.set_xlabel("Chunk timestep")
        ax.set_ylabel("Action L1")
        ax.set_title("Correct-Language Chunk-Horizon Error Across Models")
        ax.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.7)
        ax.legend(frameon=False, fontsize=7)
        save_fig(fig, lang_fig / "summary_correct_language_chunk_horizon_error.png")

    sens_df = rows_to_df(sensitivity_dim_rows)
    if not sens_df.empty:
        fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=180, constrained_layout=True)
        for mode, color in [("zero_language", "#7b62a3"), ("wrong_language", "#b23a48")]:
            mode_df = sens_df[sens_df["ablation_mode"] == mode]
            grouped = mode_df.groupby("action_dim")["mean_pred_l1_delta"].mean()
            ax.plot(grouped.index, grouped.values, marker="o", linewidth=1.2, color=color, label=mode.replace("_", " "))
        ax.set_xticks(np.arange(7), labels=[f"a{i}" for i in range(7)])
        ax.set_ylabel("Mean prediction delta")
        ax.set_title("Average Language Sensitivity by Action Dimension")
        ax.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.7)
        ax.legend(frameon=False)
        save_fig(fig, lang_fig / "summary_language_sensitivity_by_action_dim.png")

    sens_h_df = rows_to_df(sensitivity_horizon_rows)
    if not sens_h_df.empty:
        fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=180, constrained_layout=True)
        for mode, color in [("zero_language", "#7b62a3"), ("wrong_language", "#b23a48")]:
            mode_df = sens_h_df[sens_h_df["ablation_mode"] == mode]
            grouped = mode_df.groupby("chunk_t")["mean_pred_l1_delta"].mean()
            ax.plot(grouped.index, grouped.values, linewidth=1.2, color=color, label=mode.replace("_", " "))
        ax.set_xlabel("Chunk timestep")
        ax.set_ylabel("Mean prediction delta")
        ax.set_title("Average Language Sensitivity by Chunk Horizon")
        ax.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.7)
        ax.legend(frameon=False)
        save_fig(fig, lang_fig / "summary_language_sensitivity_by_chunk_horizon.png")

    smooth_df = rows_to_df(smoothness_rows)
    if not smooth_df.empty:
        pivot = smooth_df.pivot(index="display_name", columns="mode", values="mean_chunk_delta_l2_first6")
        order = [col for col in ["gt", "correct", "zero_language", "wrong_language"] if col in pivot.columns]
        fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=180, constrained_layout=True)
        x = np.arange(len(pivot.index))
        width = 0.8 / max(1, len(order))
        colors = {"gt": "#222222", "correct": "#2f6f9f", "zero_language": "#b7a4d9", "wrong_language": "#b23a48"}
        for idx, mode in enumerate(order):
            ax.bar(x + (idx - (len(order) - 1) / 2) * width, pivot[mode], width=width, color=colors[mode], label=mode.replace("_", " "))
        ax.set_xticks(x, labels=list(pivot.index), rotation=25, ha="right", fontsize=7)
        ax.set_ylabel("Mean chunk delta L2, first 6 dims")
        ax.set_title("GT vs Predicted Chunk Smoothness")
        ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
        ax.legend(frameon=False, fontsize=7)
        save_fig(fig, lang_fig / "summary_predicted_chunk_smoothness.png")

    norm_df = rows_to_df(norm_scatter_rows)
    if not norm_df.empty:
        fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=180, constrained_layout=True)
        for display, group in norm_df.groupby("display_name"):
            ax.scatter(
                group["gt_action_norm_first_step_first6"],
                group["pred_action_norm_first_step_first6"],
                s=9,
                alpha=0.45,
                label=display,
            )
        max_value = float(
            max(
                norm_df["gt_action_norm_first_step_first6"].max(),
                norm_df["pred_action_norm_first_step_first6"].max(),
            )
        )
        ax.plot([0, max_value], [0, max_value], color="#222222", linewidth=0.8, linestyle="--")
        ax.set_xlabel("GT first-step action norm")
        ax.set_ylabel("Predicted first-step action norm")
        ax.set_title("Predicted vs GT Action Norm")
        ax.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.7)
        ax.legend(frameon=False, fontsize=7)
        save_fig(fig, lang_fig / "summary_pred_vs_gt_action_norm.png")


@torch.no_grad()
def plot_prompt_conditioning(spec, policy, preprocessor, raw_batch, camera_keys, lang_fig, lang_tab) -> None:
    batch = prepare_batch(raw_batch, preprocessor, camera_keys)
    language_key = policy.config.language_embedding_key
    tasks = list(raw_batch["task"])
    embeddings = batch[language_key]
    unique = []
    for idx, task in enumerate(tasks):
        if task not in [item[0] for item in unique]:
            unique.append((str(task), embeddings[idx : idx + 1]))
        if len(unique) >= 5:
            break
    if len(unique) < 2:
        return

    one_obs = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            one_obs[key] = value[:1].clone()
        else:
            one_obs[key] = value

    rows = []
    dim_summary = []
    pred_by_task = []
    fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=180, constrained_layout=True)
    for task, embedding in unique:
        conditioned = dict(one_obs)
        conditioned[language_key] = embedding.to(one_obs[ACTION].device)
        pred = policy.predict_action_chunk(conditioned)[0].detach().cpu().numpy()
        pred_by_task.append((task, pred))
        norm = np.linalg.norm(pred[:, :6], axis=1)
        ax.plot(np.arange(len(norm)), norm, linewidth=1.25, label=short_label(task, 28))
        for t, value in enumerate(norm):
            if t % 10 == 0:
                rows.append({"experiment": spec.key, "task": task, "chunk_t": t, "pred_action_norm_first6": float(value)})
        for dim in range(pred.shape[1]):
            dim_summary.append(
                {
                    "experiment": spec.key,
                    "task": task,
                    "action_dim": dim,
                    "mean_pred_action": float(pred[:, dim].mean()),
                    "mean_abs_pred_action": float(np.abs(pred[:, dim]).mean()),
                }
            )
    ax.set_title(f"{spec.display}: Prompt-Conditioned Action Chunk")
    ax.set_xlabel("Chunk timestep")
    ax.set_ylabel("Predicted action norm, first 6 dims")
    ax.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=False, fontsize=7)
    save_fig(fig, lang_fig / f"{spec.key}_prompt_conditioned_action_chunks.png")
    write_rows(lang_tab / f"{spec.key}_prompt_conditioned_action_chunks.csv", rows)
    write_rows(lang_tab / f"{spec.key}_prompt_conditioned_action_dim_summary.csv", dim_summary)

    if pred_by_task:
        heat = np.asarray([[np.abs(pred[:, dim]).mean() for dim in range(pred.shape[1])] for _, pred in pred_by_task])
        fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=180, constrained_layout=True)
        im = ax.imshow(heat, cmap="viridis")
        ax.set_xticks(np.arange(7), labels=[f"a{i}" for i in range(7)])
        ax.set_yticks(np.arange(len(pred_by_task)), labels=[short_label(task, 34) for task, _ in pred_by_task], fontsize=6.5)
        ax.set_title(f"{spec.display}: Prompt-Conditioned Action-Dim Magnitude")
        fig.colorbar(im, ax=ax, label="Mean |pred action|")
        save_fig(fig, lang_fig / f"{spec.key}_prompt_conditioned_action_dim_heatmap.png")


@torch.no_grad()
def plot_language_distance_matrix(spec, policy, preprocessor, raw_batch, camera_keys, lang_fig, lang_tab) -> None:
    batch = prepare_batch(raw_batch, preprocessor, camera_keys)
    language_key = policy.config.language_embedding_key
    tasks = list(raw_batch["task"])
    embeddings = batch[language_key]
    unique = []
    for idx, task in enumerate(tasks):
        if task not in [item[0] for item in unique]:
            unique.append((str(task), embeddings[idx : idx + 1]))
        if len(unique) >= 8:
            break
    if len(unique) < 2:
        return

    one_obs = {key: value[:1].clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    preds = []
    for _, embedding in unique:
        conditioned = dict(one_obs)
        conditioned[language_key] = embedding.to(one_obs[ACTION].device)
        preds.append(policy.predict_action_chunk(conditioned)[0].detach().cpu().numpy())
    n = len(preds)
    dist = np.zeros((n, n), dtype=np.float64)
    rows = []
    for i in range(n):
        for j in range(n):
            value = float(np.abs(preds[i] - preds[j]).mean())
            dist[i, j] = value
            rows.append(
                {
                    "experiment": spec.key,
                    "task_i": unique[i][0],
                    "task_j": unique[j][0],
                    "mean_action_chunk_l1": value,
                }
            )
    fig, ax = plt.subplots(figsize=(6.8, 5.8), dpi=180, constrained_layout=True)
    im = ax.imshow(dist, cmap="magma")
    labels = [short_label(task, 18) for task, _ in unique]
    ax.set_xticks(np.arange(n), labels=labels, rotation=45, ha="right", fontsize=6.5)
    ax.set_yticks(np.arange(n), labels=labels, fontsize=6.5)
    ax.set_title(f"{spec.display}: Cross-Language Action Distance")
    fig.colorbar(im, ax=ax, label="Mean chunk L1")
    save_fig(fig, lang_fig / f"{spec.key}_cross_language_action_distance.png")
    write_rows(lang_tab / f"{spec.key}_cross_language_action_distance.csv", rows)


@torch.no_grad()
def plot_representative_strip(spec, policy, preprocessor, val_ds, camera_keys, lang_fig, lang_tab) -> None:
    count = min(10, len(val_ds))
    indices = np.linspace(0, min(len(val_ds) - 1, 120), count, dtype=np.int64).tolist()
    items = [val_ds[int(idx)] for idx in indices]
    raw_batch = default_collate(items)
    batch = prepare_batch(raw_batch, preprocessor, camera_keys)
    pred = policy.predict_action_chunk(batch).detach().cpu().numpy()
    gt = batch[ACTION].detach().cpu().numpy()
    gt_norm = np.linalg.norm(gt[:, 0, :6], axis=1)
    pred_norm = np.linalg.norm(pred[:, 0, :6], axis=1)
    task = str(raw_batch["task"][0])

    frame_imgs = []
    for item in items:
        img = item["observation.images.static"]
        arr = img.detach().cpu().numpy()
        if arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        arr = np.clip(arr * 255.0 if arr.max() <= 1.5 else arr, 0, 255).astype(np.uint8)
        frame_imgs.append(Image.fromarray(arr).resize((160, 160)))
    canvas = Image.new("RGB", (160 * count, 204), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, img in enumerate(frame_imgs):
        canvas.paste(img, (160 * idx, 0))
        draw.text((160 * idx + 4, 164), f"f{indices[idx]}", fill=(20, 20, 20))
    draw.text((6, 184), textwrap.shorten(f"task: {task}", width=140), fill=(20, 20, 20))
    strip_path = lang_fig / f"{spec.key}_representative_language_trajectory_frames.png"
    strip_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(strip_path)

    fig, ax = plt.subplots(figsize=(7.6, 3.5), dpi=180, constrained_layout=True)
    ax.plot(indices, gt_norm, marker="o", linewidth=1.2, label="GT action norm")
    ax.plot(indices, pred_norm, marker="s", linewidth=1.2, label="Predicted action norm")
    ax.set_title(f"{spec.display}: Representative Language Trajectory")
    ax.set_xlabel("Sample index in validation episode window")
    ax.set_ylabel("Action norm, first 6 dims")
    ax.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=False)
    save_fig(fig, lang_fig / f"{spec.key}_representative_language_trajectory_action_norm.png")
    write_rows(
        lang_tab / f"{spec.key}_representative_language_trajectory.csv",
        [
            {
                "experiment": spec.key,
                "task": task,
                "sample_index": int(idx),
                "gt_action_norm_first6": float(gt_v),
                "pred_action_norm_first6": float(pred_v),
            }
            for idx, gt_v, pred_v in zip(indices, gt_norm, pred_norm, strict=True)
        ],
    )


def plot_language_embedding_pca(specs: list[ModelSpec], lang_fig: Path, lang_tab: Path, max_samples: int) -> None:
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), dpi=180, constrained_layout=True)
    axes = list(axes)
    plotted = 0
    for spec in specs:
        if spec.dataset_group not in {"B", "ABC"}:
            continue
        if any(row["dataset_group"] == spec.dataset_group for row in rows):
            continue
        progress("language_embedding_pca", plotted, 2, f"reading {spec.dataset_group} parquet")
        _, data_cfg, policy_cfg, ds_meta = load_train_context(spec)
        train_eps = _parse_episode_spec(data_cfg["train_episodes"], ds_meta.total_episodes)
        task_map = task_index_to_text(data_cfg["root"])
        sample = sample_parquet_rows(
            data_cfg["root"],
            train_eps,
            ["observation.language_embedding", "task_index"],
            max_samples,
        )
        tasks = [task_map.get(int(idx), f"task_{int(idx)}") for idx in sample["task_index"].to_numpy()]
        raw_x = stack_series(sample["observation.language_embedding"]).astype(np.float64)
        x = raw_x - raw_x.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        coords = x @ vt[:2].T
        counts = Counter(tasks)
        top = {task for task, _ in counts.most_common(8)}
        ax = axes[plotted]
        for task in top:
            mask = np.array([t == task for t in tasks])
            ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.72, label=short_label(task, 18))
        other_mask = np.array([t not in top for t in tasks])
        if other_mask.any():
            ax.scatter(coords[other_mask, 0], coords[other_mask, 1], s=4, alpha=0.10, color="#777777", label="other")
        ax.set_title(f"{spec.dataset_group} Language Embedding PCA")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(frameon=False, fontsize=5.5, markerscale=1.6)
        ax.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.5)
        for task, (pc1, pc2) in zip(tasks, coords, strict=True):
            rows.append({"dataset_group": spec.dataset_group, "task": task, "pc1": float(pc1), "pc2": float(pc2)})
        top_tasks = [task for task, _ in counts.most_common(10)]
        centroids = []
        for task in top_tasks:
            mask = np.array([t == task for t in tasks])
            centroids.append(raw_x[mask].mean(axis=0))
        if centroids:
            c = np.stack(centroids, axis=0)
            c = c / np.linalg.norm(c, axis=1, keepdims=True).clip(min=1e-8)
            sim = c @ c.T
            fig_sim, ax_sim = plt.subplots(figsize=(7.0, 6.2), dpi=180, constrained_layout=True)
            im = ax_sim.imshow(sim, cmap="coolwarm", vmin=-1, vmax=1)
            labels = [short_label(task, 24) for task in top_tasks]
            ax_sim.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right", fontsize=6.5)
            ax_sim.set_yticks(np.arange(len(labels)), labels=labels, fontsize=6.5)
            ax_sim.set_title(f"{spec.dataset_group}: Task Embedding Centroid Similarity")
            fig_sim.colorbar(im, ax=ax_sim, label="Cosine similarity")
            save_fig(fig_sim, lang_fig / f"language_embedding_task_centroid_similarity_{spec.dataset_group}.png")
        plotted += 1
        progress("language_embedding_pca", plotted, 2, f"done {spec.dataset_group}")
        if plotted >= 2:
            break
    if plotted:
        save_fig(fig, lang_fig / "language_embedding_pca_B_vs_ABC.png")
    else:
        plt.close(fig)
    write_rows(lang_tab / "language_embedding_pca.csv", rows)


def write_manifest(core_fig: Path, lang_fig: Path, core_tab: Path, lang_tab: Path) -> None:
    rows = []
    for group, folder in [("core", core_fig), ("language", lang_fig)]:
        for path in sorted(folder.glob("*.png")):
            rows.append({"group": group, "artifact_type": "figure", "path": str(path)})
    for group, folder in [("core", core_tab), ("language", lang_tab)]:
        for path in sorted(folder.glob("*.csv")):
            rows.append({"group": group, "artifact_type": "table_csv", "path": str(path)})
        for path in sorted(folder.glob("*.tex")):
            rows.append({"group": group, "artifact_type": "table_tex", "path": str(path)})
    write_rows(lang_tab.parent / "act_lang_visualization_manifest.csv", rows)


def clean_output_dirs(*folders: Path) -> None:
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        for path in folder.iterdir():
            if path.is_file() and path.suffix in {".png", ".csv", ".tex"}:
                path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path("project"))
    parser.add_argument("--max-eval-samples", type=int, default=384)
    parser.add_argument("--max-data-samples", type=int, default=3000)
    parser.add_argument("--max-pca-samples", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--smoothing-window", type=int, default=1000)
    parser.add_argument("--no-clean", action="store_true", help="Do not remove previous ACT-Lang figures/tables first.")
    args = parser.parse_args()

    project_dir = args.project_dir
    core_fig = project_dir / "figures/act_lang_core"
    lang_fig = project_dir / "figures/act_lang_language"
    core_tab = project_dir / "tables/act_lang_core"
    lang_tab = project_dir / "tables/act_lang_language"
    for folder in [core_fig, lang_fig, core_tab, lang_tab]:
        folder.mkdir(parents=True, exist_ok=True)
    if not args.no_clean:
        clean_output_dirs(core_fig, lang_fig, core_tab, lang_tab)

    specs = load_model_specs(project_dir)
    if not specs:
        raise SystemExit("No ACT-Lang runs found.")

    progress("start", 0, 5, f"models={','.join(spec.key for spec in specs)}")
    plot_training_core(specs, core_fig, core_tab, args.smoothing_window)
    progress("training_core", 1, 5, "done")
    dataset_action_and_task_core(specs, core_fig, core_tab, args.max_data_samples)
    progress("dataset_core", 2, 5, "done")
    evaluate_language_effects(specs, lang_fig, lang_tab, args.max_eval_samples, args.batch_size)
    progress("language_effects", 3, 5, "done")
    plot_language_embedding_pca(specs, lang_fig, lang_tab, args.max_pca_samples)
    progress("language_embedding_pca", 4, 5, "done")
    write_manifest(core_fig, lang_fig, core_tab, lang_tab)
    progress("manifest", 5, 5, "done")
    print(
        json.dumps(
            {
                "event": "act_lang_visuals_complete",
                "num_models": len(specs),
                "models": [spec.key for spec in specs],
                "core_figures": str(core_fig),
                "language_figures": str(lang_fig),
                "core_tables": str(core_tab),
                "language_tables": str(lang_tab),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
