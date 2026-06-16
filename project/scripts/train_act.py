#!/usr/bin/env python
"""Small ACT training harness for CALVIN LeRobot datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.act import ACTConfig
from lerobot.utils.constants import ACTION, IMAGENET_STATS


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a mapping config in {path}")
    return cfg


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_act_config(policy_cfg: dict[str, Any]) -> ACTConfig:
    kwargs = dict(policy_cfg)
    policy_type = kwargs.pop("type", "act")
    if policy_type != "act":
        raise ValueError(f"Only ACT is supported by this harness, got {policy_type!r}")
    return ACTConfig(**kwargs)


def _parse_episode_spec(spec: Any, total_episodes: int) -> list[int]:
    if spec is None or spec == "all":
        return list(range(total_episodes))
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, list):
        return [int(ep) for ep in spec]
    if isinstance(spec, str):
        if ":" in spec:
            start_s, end_s = spec.split(":", 1)
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else total_episodes
            if start < 0 or end > total_episodes or start >= end:
                raise ValueError(f"Invalid episode range {spec!r} for {total_episodes} episodes")
            return list(range(start, end))
        return [int(part) for part in spec.split(",") if part.strip()]
    raise TypeError(f"Unsupported episode spec: {spec!r}")


def _make_dataset(
    repo_id: str,
    root: str,
    episodes: list[int],
    policy_cfg: ACTConfig,
    *,
    return_uint8: bool,
    video_backend: str | None,
) -> LeRobotDataset:
    ds_meta = LeRobotDatasetMetadata(repo_id, root=root)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    return LeRobotDataset(
        repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        return_uint8=return_uint8,
        video_backend=video_backend,
    )


def _apply_imagenet_stats(dataset: LeRobotDataset) -> None:
    for key in dataset.meta.camera_keys:
        for stats_type, stats in IMAGENET_STATS.items():
            dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)


def _move_uint8_images_to_float(batch: dict[str, Any], camera_keys: Iterable[str]) -> dict[str, Any]:
    for cam_key in camera_keys:
        value = batch.get(cam_key)
        if isinstance(value, torch.Tensor) and value.dtype == torch.uint8:
            batch[cam_key] = value.to(dtype=torch.float32) / 255.0
    return batch


class ImageBatchAugmenter:
    """Conservative photometric augmentation for batched camera tensors."""

    def __init__(self, cfg: dict[str, Any] | None) -> None:
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.brightness = float(cfg.get("brightness", 0.0))
        self.contrast = float(cfg.get("contrast", 0.0))
        self.saturation = float(cfg.get("saturation", 0.0))
        self.gaussian_noise_std = float(cfg.get("gaussian_noise_std", 0.0))
        self.probability = float(cfg.get("probability", 1.0))
        self.clip_min = float(cfg.get("clip_min", 0.0))
        self.clip_max = float(cfg.get("clip_max", 1.0))

    def state_dict(self) -> dict[str, float | bool]:
        return {
            "enabled": self.enabled,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "gaussian_noise_std": self.gaussian_noise_std,
            "probability": self.probability,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
        }

    @staticmethod
    def _factor(num_images: int, magnitude: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if magnitude <= 0:
            return torch.ones((num_images, 1, 1, 1), device=device, dtype=dtype)
        low = max(0.0, 1.0 - magnitude)
        high = 1.0 + magnitude
        return torch.empty((num_images, 1, 1, 1), device=device, dtype=dtype).uniform_(low, high)

    def _augment_tensor(self, value: torch.Tensor) -> torch.Tensor:
        if not self.enabled or value.ndim < 4:
            return value
        original_shape = value.shape
        flat = value.reshape(-1, *original_shape[-3:])
        if flat.shape[1] != 3:
            return value

        apply_mask = torch.rand((flat.shape[0], 1, 1, 1), device=flat.device, dtype=flat.dtype) < self.probability
        augmented = flat

        if self.brightness > 0:
            bright = self._factor(flat.shape[0], self.brightness, flat.device, flat.dtype)
            augmented = torch.where(apply_mask, augmented * bright, augmented)

        if self.contrast > 0:
            contrast = self._factor(flat.shape[0], self.contrast, flat.device, flat.dtype)
            mean = augmented.mean(dim=(-2, -1), keepdim=True)
            contrasted = (augmented - mean) * contrast + mean
            augmented = torch.where(apply_mask, contrasted, augmented)

        if self.saturation > 0:
            saturation = self._factor(flat.shape[0], self.saturation, flat.device, flat.dtype)
            weights = torch.tensor([0.299, 0.587, 0.114], device=flat.device, dtype=flat.dtype).view(1, 3, 1, 1)
            gray = (augmented * weights).sum(dim=1, keepdim=True)
            saturated = (augmented - gray) * saturation + gray
            augmented = torch.where(apply_mask, saturated, augmented)

        if self.gaussian_noise_std > 0:
            noisy = augmented + torch.randn_like(augmented) * self.gaussian_noise_std
            augmented = torch.where(apply_mask, noisy, augmented)

        return augmented.clamp(self.clip_min, self.clip_max).reshape(original_shape)

    def __call__(self, batch: dict[str, Any], camera_keys: Iterable[str]) -> dict[str, Any]:
        if not self.enabled:
            return batch
        for cam_key in camera_keys:
            value = batch.get(cam_key)
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                batch[cam_key] = self._augment_tensor(value)
        return batch


def _make_loader(dataset: LeRobotDataset, batch_size: int, num_workers: int, shuffle: bool) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        prefetch_factor=None if num_workers == 0 else 2,
        persistent_workers=num_workers > 0,
    )


def _next_batch(iterator: Iterable[dict[str, Any]], loader: torch.utils.data.DataLoader):
    try:
        batch = next(iterator)  # type: ignore[arg-type]
        return batch, iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


@torch.no_grad()
def _validate(
    policy: torch.nn.Module,
    preprocessor,
    val_loader: torch.utils.data.DataLoader,
    camera_keys: list[str],
    max_batches: int,
) -> dict[str, float]:
    # ACT's VAE branch only returns latent parameters in train mode. Keep train
    # mode under no_grad so validation measures supervised training loss without
    # optimizer updates.
    was_training = policy.training
    policy.train()
    total_loss = 0.0
    total_l1 = 0.0
    n = 0
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        batch = _move_uint8_images_to_float(batch, camera_keys)
        batch = preprocessor(batch)
        loss, loss_dict = policy.forward(batch)
        total_loss += float(loss.item())
        total_l1 += float(loss_dict["l1_loss"])
        n += 1
    policy.train(was_training)
    if n == 0:
        return {"val_loss": math.nan, "val_action_l1": math.nan}
    return {"val_loss": total_loss / n, "val_action_l1": total_l1 / n}


def _write_metrics_header(path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "train_loss",
                "train_action_l1",
                "train_kld",
                "val_loss",
                "val_action_l1",
                "lr",
                "step_s",
            ],
        )
        writer.writeheader()


def _append_metrics(path: Path, row: dict[str, float | int]) -> None:
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


def _save_checkpoint(
    output_dir: Path,
    policy,
    preprocessor,
    postprocessor,
    optimizer: torch.optim.Optimizer,
    step: int,
    *,
    checkpoint_name: str = "checkpoint",
) -> Path:
    checkpoint_dir = output_dir / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(checkpoint_dir)
    preprocessor.save_pretrained(checkpoint_dir)
    postprocessor.save_pretrained(checkpoint_dir)
    torch.save({"step": step, "optimizer": optimizer.state_dict()}, checkpoint_dir / "training_state.pt")
    return checkpoint_dir


def _format_progress(step: int, total_steps: int, elapsed_s: float, width: int = 32) -> str:
    pct = step / max(total_steps, 1)
    filled = min(width, max(0, int(round(width * pct))))
    bar = "#" * filled + "-" * (width - filled)
    if step > 0:
        eta_s = elapsed_s * (total_steps - step) / step
    else:
        eta_s = math.nan
    return (
        f"[{bar}] {step}/{total_steps} ({pct * 100:5.1f}%) "
        f"elapsed={elapsed_s / 3600:.2f}h eta={eta_s / 3600:.2f}h"
    )


def _self_check(output_dir: Path, metrics_path: Path, checkpoint_dir: Path) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    checks["metrics_csv_exists"] = metrics_path.is_file()
    rows: list[dict[str, str]] = []
    if metrics_path.is_file():
        with metrics_path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
    checks["metrics_rows"] = len(rows)
    finite_train = False
    finite_val = False
    if rows:
        train_values = [float(r["train_action_l1"]) for r in rows if r.get("train_action_l1")]
        val_values = [float(r["val_action_l1"]) for r in rows if r.get("val_action_l1")]
        finite_train = all(math.isfinite(v) for v in train_values) and len(train_values) > 0
        finite_val_values = [v for v in val_values if math.isfinite(v)]
        finite_val = len(finite_val_values) > 0 and math.isfinite(val_values[-1])
    checks["finite_train_action_l1"] = finite_train
    checks["finite_val_action_l1"] = finite_val
    checks["checkpoint_dir_exists"] = checkpoint_dir.is_dir()
    checks["policy_config_exists"] = (checkpoint_dir / "config.json").is_file()
    checks["model_weights_exists"] = any(checkpoint_dir.glob("*.safetensors")) or any(checkpoint_dir.glob("*.bin"))
    checks["preprocessor_exists"] = any(checkpoint_dir.glob("*preprocessor*.json"))
    checks["postprocessor_exists"] = any(checkpoint_dir.glob("*postprocessor*.json"))
    checks["training_state_exists"] = (checkpoint_dir / "training_state.pt").is_file()
    checks["passed"] = all(
        bool(checks[k])
        for k in [
            "metrics_csv_exists",
            "finite_train_action_l1",
            "finite_val_action_l1",
            "checkpoint_dir_exists",
            "policy_config_exists",
            "model_weights_exists",
            "preprocessor_exists",
            "postprocessor_exists",
            "training_state_exists",
        ]
    )
    with (output_dir / "self_check.json").open("w") as f:
        json.dump(checks, f, indent=2)
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None, help="Override experiment.output_dir.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    device_override = os.environ.get("ACT_DEVICE_OVERRIDE")
    if device_override:
        cfg.setdefault("policy", {})["device"] = device_override
    exp_cfg = cfg["experiment"]
    data_cfg = cfg["dataset"]
    train_cfg = cfg["training"]
    aug_cfg = cfg.get("augmentation", {})
    policy_cfg = _build_act_config(cfg["policy"])

    output_dir = args.output_dir if args.output_dir is not None else Path(exp_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config_snapshot.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    _set_seed(int(exp_cfg["seed"]))

    repo_id = data_cfg["repo_id"]
    root = data_cfg["root"]
    video_backend = data_cfg.get("video_backend")
    return_uint8 = bool(data_cfg.get("return_uint8", True))
    ds_meta = LeRobotDatasetMetadata(repo_id, root=root)
    train_episodes = _parse_episode_spec(data_cfg["train_episodes"], ds_meta.total_episodes)
    val_episodes = _parse_episode_spec(data_cfg["val_episodes"], ds_meta.total_episodes)

    train_dataset = _make_dataset(
        repo_id, root, train_episodes, policy_cfg, return_uint8=return_uint8, video_backend=video_backend
    )
    val_dataset = _make_dataset(
        repo_id, root, val_episodes, policy_cfg, return_uint8=return_uint8, video_backend=video_backend
    )
    if data_cfg.get("use_imagenet_stats", True):
        _apply_imagenet_stats(train_dataset)
        _apply_imagenet_stats(val_dataset)

    policy = make_policy(policy_cfg, ds_meta=train_dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        dataset_stats=train_dataset.meta.stats,
    )

    optimizer = policy.config.get_optimizer_preset().build(policy.parameters())
    train_loader = _make_loader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(train_cfg["num_workers"]),
        shuffle=True,
    )
    val_loader = _make_loader(
        val_dataset,
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(train_cfg["num_workers"]),
        shuffle=False,
    )
    train_iter = iter(train_loader)
    camera_keys = list(train_dataset.meta.camera_keys)
    augmenter = ImageBatchAugmenter(aug_cfg)
    metrics_path = output_dir / "metrics.csv"
    _write_metrics_header(metrics_path)

    print(
        json.dumps(
            {
                "event": "start",
                "output_dir": str(output_dir),
                "device": policy.config.device,
                "train_episodes": train_episodes,
                "val_episodes": val_episodes,
                "train_frames": train_dataset.num_frames,
                "val_frames": val_dataset.num_frames,
                "camera_keys": camera_keys,
                "chunk_size": policy.config.chunk_size,
                "n_action_steps": policy.config.n_action_steps,
                "augmentation": augmenter.state_dict(),
            },
            indent=2,
        ),
        flush=True,
    )

    policy.train()
    steps = int(train_cfg["steps"])
    val_freq = int(train_cfg["val_freq"])
    log_freq = int(train_cfg["log_freq"])
    save_freq = int(train_cfg.get("save_freq", 0))
    progress_bar = bool(train_cfg.get("progress_bar", True))
    grad_clip_norm = float(train_cfg["grad_clip_norm"])
    max_val_batches = int(train_cfg["max_val_batches"])
    last_val = {"val_loss": math.nan, "val_action_l1": math.nan}
    run_start = time.perf_counter()

    for step in range(1, steps + 1):
        start = time.perf_counter()
        batch, train_iter = _next_batch(train_iter, train_loader)
        batch = _move_uint8_images_to_float(batch, camera_keys)
        batch = augmenter(batch, camera_keys)
        batch = preprocessor(batch)
        loss, loss_dict = policy.forward(batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % val_freq == 0 or step == steps:
            last_val = _validate(policy, preprocessor, val_loader, camera_keys, max_val_batches)

        row = {
            "step": step,
            "train_loss": float(loss.item()),
            "train_action_l1": float(loss_dict["l1_loss"]),
            "train_kld": float(loss_dict.get("kld_loss", math.nan)),
            "val_loss": float(last_val["val_loss"]),
            "val_action_l1": float(last_val["val_action_l1"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "step_s": float(time.perf_counter() - start),
        }
        _append_metrics(metrics_path, row)
        if step % log_freq == 0:
            log_row = dict(row)
            log_row["grad_norm"] = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)
            print(json.dumps({"event": "step", **log_row}), flush=True)
            if progress_bar:
                print(f"PROGRESS {exp_cfg['name']} {_format_progress(step, steps, time.perf_counter() - run_start)}", flush=True)

        if bool(train_cfg.get("save_checkpoint", True)) and save_freq > 0 and step % save_freq == 0 and step != steps:
            checkpoint_name = f"checkpoints/step_{step:08d}"
            _save_checkpoint(output_dir, policy, preprocessor, postprocessor, optimizer, step, checkpoint_name=checkpoint_name)

    checkpoint_dir = _save_checkpoint(output_dir, policy, preprocessor, postprocessor, optimizer, steps)
    checks = _self_check(output_dir, metrics_path, checkpoint_dir)
    print(json.dumps({"event": "self_check", **checks}, indent=2), flush=True)
    return 0 if checks["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
