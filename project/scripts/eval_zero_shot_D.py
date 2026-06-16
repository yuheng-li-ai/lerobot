#!/usr/bin/env python
"""Zero-shot ACT evaluation on CALVIN D.

This is an offline D evaluation harness for the project ACT checkpoints. It
measures imitation error on the unseen D split and diagnostics tied to ACT's
chunked action execution. It intentionally does not run full CALVIN language
rollouts because the trained ACT datasets do not include a language-goal input.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies import get_policy_class
from lerobot.policies.act import ACTConfig  # noqa: F401 - registers the "act" config choice.
from lerobot.processor import DataProcessorPipeline
from lerobot.utils.constants import ACTION


ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
DEFAULT_MODEL_TABLE = Path("project/tables/model_selection_checkpoints.csv")
DEFAULT_RUN_TABLE = Path("project/tables/full_training_summary_with_B_aug.csv")
DEFAULT_D_ROOT = Path("/EXT_DISK/users/zengzixuan/processed-calvin/calvin_D")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    condition: str
    checkpoint_mode: str
    checkpoint: Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(path: Path, rows: list[dict[str, Any]], columns: list[str], caption: str, label: str) -> None:
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
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value).replace("_", "\\_"))
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines))


def load_model_specs(
    checkpoint_mode: str,
    model_names: list[str] | None,
    selection_table: Path,
    run_table: Path,
) -> list[ModelSpec]:
    selected_rows = {r["experiment"]: r for r in read_rows(selection_table)}
    run_rows = {r["experiment"]: r for r in read_rows(run_table)}
    names = model_names or list(selected_rows)
    specs: list[ModelSpec] = []
    for name in names:
        if name not in selected_rows:
            raise KeyError(f"Unknown model {name!r}; available: {sorted(selected_rows)}")
        condition = run_rows.get(name, {}).get("condition", selected_rows[name].get("phase", "unknown"))
        modes = ["selected", "final"] if checkpoint_mode == "both" else [checkpoint_mode]
        for mode in modes:
            if mode == "selected":
                checkpoint = Path(selected_rows[name]["selected_available_checkpoint"])
            elif mode == "final":
                if name not in run_rows:
                    raise KeyError(f"No final checkpoint row for {name!r} in {run_table}")
                checkpoint = Path(run_rows[name]["run_dir"]) / "checkpoint"
            else:
                raise ValueError(f"Unsupported checkpoint mode {mode!r}")
            if not checkpoint.is_dir():
                raise FileNotFoundError(checkpoint)
            specs.append(ModelSpec(name=name, condition=condition, checkpoint_mode=mode, checkpoint=checkpoint))
    return specs


def episode_ranges(dataset: LeRobotDataset) -> list[tuple[int, int, int]]:
    episodes = dataset.meta.episodes
    ranges: list[tuple[int, int, int]] = []
    total = int(dataset.meta.total_episodes)
    for ep_idx in range(total):
        start = int(episodes["dataset_from_index"][ep_idx])
        end = int(episodes["dataset_to_index"][ep_idx])
        ranges.append((ep_idx, start, end))
    return ranges


def make_query_indices(dataset: LeRobotDataset, stride: int, max_queries: int | None) -> tuple[list[int], dict[int, int]]:
    indices: list[int] = []
    index_to_episode: dict[int, int] = {}
    for ep_idx, start, end in episode_ranges(dataset):
        for idx in range(start, end, stride):
            indices.append(idx)
            index_to_episode[idx] = ep_idx
            if max_queries is not None and len(indices) >= max_queries:
                return indices, index_to_episode
    return indices, index_to_episode


def move_uint8_images_to_float(batch: dict[str, Any], camera_keys: list[str]) -> dict[str, Any]:
    for key in camera_keys:
        value = batch.get(key)
        if isinstance(value, torch.Tensor) and value.dtype == torch.uint8:
            batch[key] = value.to(dtype=torch.float32) / 255.0
    return batch


def as_action_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict) and ACTION in value:
        return value[ACTION]
    raise TypeError(f"Expected action tensor or transition dict, got {type(value)!r}")


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def safe_quantile(values: list[float], q: float) -> float:
    return float(np.quantile(values, q)) if values else math.nan


def format_progress(done: int, total: int, elapsed_s: float, width: int = 30) -> str:
    pct = done / max(total, 1)
    filled = min(width, max(0, int(round(width * pct))))
    bar = "#" * filled + "-" * (width - filled)
    eta_s = elapsed_s * (total - done) / done if done > 0 else math.nan
    return f"[{bar}] {done}/{total} ({pct * 100:5.1f}%) elapsed={elapsed_s / 60:.1f}m eta={eta_s / 60:.1f}m"


def summarize_dimension_errors(errors: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for dim, name in enumerate(ACTION_NAMES):
        dim_values = errors[..., dim][valid]
        out[f"l1_{name}"] = float(dim_values.mean()) if dim_values.size else math.nan
    return out


def load_policy_and_processors(checkpoint: Path, device: str):
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.device = device
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(checkpoint, config=cfg)
    preprocessor = DataProcessorPipeline.from_pretrained(checkpoint, config_filename="policy_preprocessor.json")
    postprocessor = DataProcessorPipeline.from_pretrained(checkpoint, config_filename="policy_postprocessor.json")
    return policy, preprocessor, postprocessor


@torch.no_grad()
def evaluate_model(
    spec: ModelSpec,
    dataset: LeRobotDataset,
    query_indices: list[int],
    *,
    batch_size: int,
    num_workers: int,
    device: str,
    log_freq: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    policy, preprocessor, postprocessor = load_policy_and_processors(spec.checkpoint, device)
    policy.reset()
    camera_keys = list(dataset.meta.camera_keys)
    subset = Subset(dataset, query_indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        prefetch_factor=None if num_workers == 0 else 2,
        persistent_workers=num_workers > 0,
    )

    total_abs = 0.0
    total_abs_first6 = 0.0
    total_gripper_abs = 0.0
    total_count = 0
    total_count_first6 = 0
    total_count_gripper = 0
    dim_abs_chunks: list[np.ndarray] = []
    dim_valid_chunks: list[np.ndarray] = []
    horizon_abs_sum = np.zeros(policy.config.n_action_steps, dtype=np.float64)
    horizon_count = np.zeros(policy.config.n_action_steps, dtype=np.int64)
    pred_step_deltas: list[float] = []
    gt_step_deltas: list[float] = []
    boundary_jumps: list[float] = []
    gt_boundary_jumps: list[float] = []
    pred_action_norms: list[float] = []
    gt_action_norms: list[float] = []
    previous_pred_last: np.ndarray | None = None
    previous_gt_last: np.ndarray | None = None
    start_s = time.perf_counter()
    total_batches = len(loader)

    for batch_idx, batch in enumerate(loader, start=1):
        batch = move_uint8_images_to_float(batch, camera_keys)
        gt = batch[ACTION].detach().cpu().float()
        valid = ~batch["action_is_pad"].detach().cpu().bool()
        processed = preprocessor(batch)
        pred_norm = policy.predict_action_chunk(processed)
        pred = as_action_tensor(postprocessor({ACTION: pred_norm})).detach().cpu().float()
        if pred.ndim == 2:
            pred = pred.unsqueeze(0)

        pred_np = pred.numpy()
        gt_np = gt.numpy()
        valid_np = valid.numpy()
        abs_err = np.abs(pred_np - gt_np)
        dim_abs_chunks.append(abs_err)
        dim_valid_chunks.append(valid_np)

        total_abs += float(abs_err[valid_np].sum())
        total_count += int(valid_np.sum() * abs_err.shape[-1])
        first6_err = abs_err[..., :6]
        total_abs_first6 += float(first6_err[valid_np].sum())
        total_count_first6 += int(valid_np.sum() * 6)
        gripper_err = abs_err[..., 6]
        total_gripper_abs += float(gripper_err[valid_np].sum())
        total_count_gripper += int(valid_np.sum())

        per_horizon = abs_err.mean(axis=-1)
        for h in range(per_horizon.shape[1]):
            h_valid = valid_np[:, h]
            if h_valid.any():
                horizon_abs_sum[h] += float(per_horizon[:, h][h_valid].sum())
                horizon_count[h] += int(h_valid.sum())

        for pred_seq, gt_seq, mask in zip(pred_np, gt_np, valid_np, strict=True):
            valid_len = int(mask.sum())
            if valid_len <= 0:
                continue
            pred_valid = pred_seq[:valid_len]
            gt_valid = gt_seq[:valid_len]
            pred_action_norms.extend(np.linalg.norm(pred_valid[:, :6], axis=1).tolist())
            gt_action_norms.extend(np.linalg.norm(gt_valid[:, :6], axis=1).tolist())
            if valid_len > 1:
                pred_step_deltas.extend(np.linalg.norm(np.diff(pred_valid[:, :6], axis=0), axis=1).tolist())
                gt_step_deltas.extend(np.linalg.norm(np.diff(gt_valid[:, :6], axis=0), axis=1).tolist())
            if previous_pred_last is not None:
                boundary_jumps.append(float(np.linalg.norm(pred_valid[0, :6] - previous_pred_last[:6])))
            if previous_gt_last is not None:
                gt_boundary_jumps.append(float(np.linalg.norm(gt_valid[0, :6] - previous_gt_last[:6])))
            previous_pred_last = pred_valid[-1]
            previous_gt_last = gt_valid[-1]

        if log_freq > 0 and (batch_idx % log_freq == 0 or batch_idx == total_batches):
            elapsed_s = time.perf_counter() - start_s
            processed_queries = min(batch_idx * batch_size, len(query_indices))
            print(
                f"PROGRESS {spec.name}:{spec.checkpoint_mode} "
                f"{format_progress(batch_idx, total_batches, elapsed_s)} "
                f"queries={processed_queries}/{len(query_indices)}",
                flush=True,
            )

    all_abs = np.concatenate(dim_abs_chunks, axis=0)
    all_valid = np.concatenate(dim_valid_chunks, axis=0)
    dim_metrics = summarize_dimension_errors(all_abs, all_valid)
    result: dict[str, Any] = {
        "experiment": spec.name,
        "condition": spec.condition,
        "checkpoint_mode": spec.checkpoint_mode,
        "checkpoint": str(spec.checkpoint),
        "queries": len(query_indices),
        "valid_predicted_actions": int(all_valid.sum()),
        "action_l1": total_abs / max(total_count, 1),
        "action_l1_first6": total_abs_first6 / max(total_count_first6, 1),
        "gripper_l1": total_gripper_abs / max(total_count_gripper, 1),
        "pred_mean_step_delta_l2_first6": safe_mean(pred_step_deltas),
        "pred_q90_step_delta_l2_first6": safe_quantile(pred_step_deltas, 0.90),
        "pred_q99_step_delta_l2_first6": safe_quantile(pred_step_deltas, 0.99),
        "gt_mean_step_delta_l2_first6": safe_mean(gt_step_deltas),
        "pred_mean_boundary_jump_l2_first6": safe_mean(boundary_jumps),
        "pred_q90_boundary_jump_l2_first6": safe_quantile(boundary_jumps, 0.90),
        "pred_q99_boundary_jump_l2_first6": safe_quantile(boundary_jumps, 0.99),
        "gt_mean_boundary_jump_l2_first6": safe_mean(gt_boundary_jumps),
        "pred_mean_action_norm_first6": safe_mean(pred_action_norms),
        "gt_mean_action_norm_first6": safe_mean(gt_action_norms),
    }
    result.update(dim_metrics)

    horizon_rows: list[dict[str, Any]] = []
    for h, (err_sum, count) in enumerate(zip(horizon_abs_sum, horizon_count, strict=True)):
        horizon_rows.append(
            {
                "experiment": spec.name,
                "checkpoint_mode": spec.checkpoint_mode,
                "horizon_step": h,
                "action_l1": float(err_sum / count) if count else math.nan,
                "count": int(count),
            }
        )

    chunk_rows = [
        {
            "experiment": spec.name,
            "checkpoint_mode": spec.checkpoint_mode,
            "metric": "pred_step_delta_l2_first6",
            "mean": safe_mean(pred_step_deltas),
            "q90": safe_quantile(pred_step_deltas, 0.90),
            "q99": safe_quantile(pred_step_deltas, 0.99),
        },
        {
            "experiment": spec.name,
            "checkpoint_mode": spec.checkpoint_mode,
            "metric": "pred_boundary_jump_l2_first6",
            "mean": safe_mean(boundary_jumps),
            "q90": safe_quantile(boundary_jumps, 0.90),
            "q99": safe_quantile(boundary_jumps, 0.99),
        },
        {
            "experiment": spec.name,
            "checkpoint_mode": spec.checkpoint_mode,
            "metric": "gt_boundary_jump_l2_first6",
            "mean": safe_mean(gt_boundary_jumps),
            "q90": safe_quantile(gt_boundary_jumps, 0.90),
            "q99": safe_quantile(gt_boundary_jumps, 0.99),
        },
    ]
    return result, horizon_rows, chunk_rows


def plot_bar(rows: list[dict[str, Any]], key: str, path: Path, ylabel: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{r['experiment']}\n{r['checkpoint_mode']}" for r in rows]
    values = [float(r[key]) for r in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.1), 4.8))
    colors = ["#4C78A8" if "aug" not in r["experiment"] else "#F58518" for r in rows]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_horizon(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["experiment"], row["checkpoint_mode"])].append(row)
    fig, ax = plt.subplots(figsize=(9, 5))
    for (experiment, mode), group in grouped.items():
        group = sorted(group, key=lambda r: int(r["horizon_step"]))
        ax.plot(
            [int(r["horizon_step"]) for r in group],
            [float(r["action_l1"]) for r in group],
            label=f"{experiment} {mode}",
            linewidth=1.8,
        )
    ax.set_xlabel("Predicted action horizon within chunk")
    ax.set_ylabel("Action L1 on D")
    ax.set_title("Zero-Shot D Error Across ACT Chunk Horizon")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(os.environ.get("CALVIN_LEROBOT_ROOT", DEFAULT_D_ROOT.parent)) / "calvin_D")
    parser.add_argument("--repo-id", default="local/calvin_D")
    parser.add_argument("--selection-table", type=Path, default=DEFAULT_MODEL_TABLE)
    parser.add_argument("--run-table", type=Path, default=DEFAULT_RUN_TABLE)
    parser.add_argument("--output-dir", type=Path, default=Path("project/outputs/task3/zero_shot_D"))
    parser.add_argument("--figure-dir", type=Path, default=Path("project/figures/task3"))
    parser.add_argument("--table-dir", type=Path, default=Path("project/tables/task3"))
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--checkpoint-mode", choices=["selected", "final", "both"], default="selected")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-freq", type=int, default=10, help="Print tail-visible progress every N batches.")
    parser.add_argument("--query-stride", type=int, default=None, help="Defaults to policy n_action_steps.")
    parser.add_argument("--max-queries", type=int, default=None, help="Use a small value for smoke tests.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    specs = load_model_specs(args.checkpoint_mode, args.models, args.selection_table, args.run_table)
    first_cfg = PreTrainedConfig.from_pretrained(specs[0].checkpoint)
    stride = int(args.query_stride or first_cfg.n_action_steps)
    meta = LeRobotDatasetMetadata(args.repo_id, root=str(args.dataset_root))
    delta_timestamps = resolve_delta_timestamps(first_cfg, meta)
    dataset = LeRobotDataset(
        args.repo_id,
        root=str(args.dataset_root),
        delta_timestamps=delta_timestamps,
        return_uint8=True,
    )
    query_indices, _ = make_query_indices(dataset, stride=stride, max_queries=args.max_queries)
    if not query_indices:
        raise RuntimeError("No D query indices were selected.")

    manifest = {
        "dataset_root": str(args.dataset_root),
        "repo_id": args.repo_id,
        "total_d_frames": int(dataset.num_frames),
        "total_d_episodes": int(dataset.num_episodes),
        "query_stride": stride,
        "queries": len(query_indices),
        "checkpoint_mode": args.checkpoint_mode,
        "device": args.device,
        "models": [spec.__dict__ | {"checkpoint": str(spec.checkpoint)} for spec in specs],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"event": "start", **manifest}, indent=2), flush=True)

    result_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    for spec in specs:
        print(json.dumps({"event": "model_start", "experiment": spec.name, "checkpoint": str(spec.checkpoint)}), flush=True)
        result, model_horizon, model_chunk = evaluate_model(
            spec,
            dataset,
            query_indices,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            log_freq=args.log_freq,
        )
        result_rows.append(result)
        horizon_rows.extend(model_horizon)
        chunk_rows.extend(model_chunk)
        print(json.dumps({"event": "model_done", **result}), flush=True)

    results_csv = args.table_dir / "zero_shot_D_results.csv"
    horizon_csv = args.table_dir / "zero_shot_D_chunk_horizon.csv"
    chunk_csv = args.table_dir / "zero_shot_D_action_chunks.csv"
    write_rows(results_csv, result_rows)
    write_rows(horizon_csv, horizon_rows)
    write_rows(chunk_csv, chunk_rows)
    write_latex_table(
        args.table_dir / "zero_shot_D_results.tex",
        result_rows,
        ["experiment", "checkpoint_mode", "action_l1", "action_l1_first6", "gripper_l1", "pred_mean_boundary_jump_l2_first6"],
        "Zero-shot CALVIN D offline action error and ACT chunk diagnostics.",
        "tab:zero-shot-d-results",
    )

    plot_bar(result_rows, "action_l1", args.figure_dir / "zero_shot_action_l1_D.png", "Action L1", "Zero-Shot D Action Error")
    plot_bar(
        result_rows,
        "pred_mean_step_delta_l2_first6",
        args.figure_dir / "action_smoothness_D.png",
        "Mean step delta L2",
        "Predicted Action Smoothness on D",
    )
    plot_bar(
        result_rows,
        "pred_mean_boundary_jump_l2_first6",
        args.figure_dir / "chunk_boundary_jump_D.png",
        "Boundary jump L2",
        "ACT Chunk Boundary Jump on D",
    )
    plot_horizon(horizon_rows, args.figure_dir / "chunk_horizon_error_D.png")

    print(
        json.dumps(
            {
                "event": "done",
                "results_csv": str(results_csv),
                "horizon_csv": str(horizon_csv),
                "chunk_csv": str(chunk_csv),
                "figure_dir": str(args.figure_dir),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
