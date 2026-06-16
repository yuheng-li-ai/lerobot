#!/usr/bin/env python
"""Verify ACT-B image augmentation without training."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from lerobot.datasets import LeRobotDatasetMetadata

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_act import (  # noqa: E402
    ImageBatchAugmenter,
    _apply_imagenet_stats,
    _build_act_config,
    _load_config,
    _make_dataset,
    _make_loader,
    _move_uint8_images_to_float,
    _parse_episode_spec,
    _set_seed,
)


def _first_image(tensor: torch.Tensor) -> torch.Tensor:
    image = tensor.detach().cpu()
    while image.ndim > 3:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dims after indexing, got shape {tuple(image.shape)}")
    if image.shape[0] != 3:
        raise ValueError(f"Expected CHW image tensor, got shape {tuple(image.shape)}")
    return image.clamp(0, 1)


def _show_tensor_image(ax, image: torch.Tensor, title: str) -> None:
    ax.imshow(image.permute(1, 2, 0).numpy())
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("project"))
    args = parser.parse_args()

    cfg = _load_config(args.config)
    _set_seed(int(cfg["experiment"]["seed"]))

    output_dir = args.output_dir
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    policy_cfg = _build_act_config(cfg["policy"])
    data_cfg = cfg["dataset"]
    train_cfg = cfg["training"]
    aug_cfg = cfg.get("augmentation", {})
    augmenter = ImageBatchAugmenter(aug_cfg)
    if not augmenter.enabled:
        raise ValueError("Augmentation config is disabled; expected augmentation.enabled=true")

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

    loader = _make_loader(dataset, batch_size=int(train_cfg["batch_size"]), num_workers=0, shuffle=False)
    batch = next(iter(loader))
    camera_keys = list(dataset.meta.camera_keys)
    batch = _move_uint8_images_to_float(batch, camera_keys)
    original = {key: batch[key].clone() for key in camera_keys}
    augmented = augmenter({key: value.clone() for key, value in original.items()}, camera_keys)

    rows = []
    fig, axes = plt.subplots(
        len(camera_keys),
        2,
        figsize=(6.8, 3.15 * len(camera_keys)),
        dpi=180,
        constrained_layout=True,
    )
    if len(camera_keys) == 1:
        axes = axes.reshape(1, 2)

    for row_idx, cam_key in enumerate(camera_keys):
        orig = original[cam_key]
        aug = augmented[cam_key]
        delta = (aug - orig).abs()
        rows.append(
            {
                "experiment": cfg["experiment"]["name"],
                "camera": cam_key,
                "augmentation_enabled": augmenter.enabled,
                "mean_abs_pixel_delta": float(delta.mean()),
                "max_abs_pixel_delta": float(delta.max()),
                "original_min": float(orig.min()),
                "original_max": float(orig.max()),
                "augmented_min": float(aug.min()),
                "augmented_max": float(aug.max()),
                "batch_shape": "x".join(str(dim) for dim in orig.shape),
            }
        )
        _show_tensor_image(axes[row_idx, 0], _first_image(orig), f"{cam_key}\noriginal")
        _show_tensor_image(axes[row_idx, 1], _first_image(aug), f"{cam_key}\naugmented")

    fig.suptitle("ACT-B Augmentation Verification")
    fig.savefig(figure_dir / "act_B_aug_verification.png", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    _write_rows(table_dir / "act_B_aug_verification.csv", rows)

    print(f"Wrote augmentation verification figure to {figure_dir / 'act_B_aug_verification.png'}")
    print(f"Wrote augmentation verification table to {table_dir / 'act_B_aug_verification.csv'}")
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
