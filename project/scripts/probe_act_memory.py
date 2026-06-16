#!/usr/bin/env python
"""Probe ACT CUDA memory for candidate batch sizes without saving checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.act import ACTConfig
from lerobot.utils.constants import IMAGENET_STATS


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _parse_episode_spec(spec: Any, total_episodes: int) -> list[int]:
    if spec is None or spec == "all":
        return list(range(total_episodes))
    if isinstance(spec, list):
        return [int(ep) for ep in spec]
    if isinstance(spec, str) and ":" in spec:
        start_s, end_s = spec.split(":", 1)
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else total_episodes
        return list(range(start, end))
    if isinstance(spec, str):
        return [int(part) for part in spec.split(",") if part.strip()]
    return [int(spec)]


def _apply_imagenet_stats(dataset: LeRobotDataset) -> None:
    for key in dataset.meta.camera_keys:
        for stats_type, stats in IMAGENET_STATS.items():
            dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)


def _uint8_to_float(batch: dict[str, Any], camera_keys: list[str]) -> dict[str, Any]:
    for key in camera_keys:
        if batch[key].dtype == torch.uint8:
            batch[key] = batch[key].to(dtype=torch.float32) / 255.0
    return batch


def _build_policy_cfg(raw_policy_cfg: dict[str, Any]) -> ACTConfig:
    kwargs = dict(raw_policy_cfg)
    kwargs.pop("type", None)
    return ACTConfig(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--episodes", default=None, help="Episode spec for probing, e.g. 0:4.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    data_cfg = cfg["dataset"]
    policy_cfg = _build_policy_cfg(cfg["policy"])
    repo_id = data_cfg["repo_id"]
    root = data_cfg["root"]
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    episodes = _parse_episode_spec(args.episodes or data_cfg["train_episodes"], meta.total_episodes)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, meta)

    dataset = LeRobotDataset(
        repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        return_uint8=bool(data_cfg.get("return_uint8", True)),
        video_backend=data_cfg.get("video_backend"),
    )
    if data_cfg.get("use_imagenet_stats", True):
        _apply_imagenet_stats(dataset)

    results = []
    for batch_size in args.batch_sizes:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        policy = make_policy(policy_cfg, ds_meta=dataset.meta)
        preprocessor, _ = make_pre_post_processors(policy_cfg=policy.config, dataset_stats=dataset.meta.stats)
        optimizer = policy.config.get_optimizer_preset().build(policy.parameters())
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        batch = _uint8_to_float(batch, list(dataset.meta.camera_keys))
        start = time.perf_counter()
        batch = preprocessor(batch)
        policy.train()
        loss, loss_dict = policy.forward(batch)
        loss.backward()
        optimizer.step()
        elapsed_s = time.perf_counter() - start
        if torch.cuda.is_available():
            peak_allocated_gb = torch.cuda.max_memory_allocated() / 1024**3
            peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
        else:
            peak_allocated_gb = math.nan
            peak_reserved_gb = math.nan
        result = {
            "batch_size": batch_size,
            "step_s": elapsed_s,
            "loss": float(loss.item()),
            "action_l1": float(loss_dict["l1_loss"]),
            "cuda_peak_allocated_gb": peak_allocated_gb,
            "cuda_peak_reserved_gb": peak_reserved_gb,
        }
        print(json.dumps(result), flush=True)
        results.append(result)
        del optimizer, preprocessor, policy, loader, batch, loss
    print(json.dumps({"event": "summary", "results": results}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
