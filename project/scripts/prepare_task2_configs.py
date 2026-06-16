#!/usr/bin/env python
"""Generate Task 2 ACT-ABC configs and deterministic episode splits."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
TABLE_DIR = PROJECT_ROOT / "tables"
ABC_ROOT = Path("/EXT_DISK/users/zengzixuan/processed-calvin/calvin_ABC")
B_TRAIN_FRAMES = 535_403


POLICY_CFG = {
    "type": "act",
    # Direct config runs avoid GPU0. Task 2 launchers set CUDA_VISIBLE_DEVICES to
    # a nonzero physical GPU and override this to cuda:0 inside that isolated view.
    "device": "cuda:1",
    "use_amp": False,
    "push_to_hub": False,
    "pretrained_backbone_weights": "ResNet18_Weights.IMAGENET1K_V1",
    "n_obs_steps": 1,
    "chunk_size": 100,
    "n_action_steps": 100,
    "vision_backbone": "resnet18",
    "dim_model": 512,
    "n_heads": 8,
    "dim_feedforward": 3200,
    "n_encoder_layers": 4,
    "n_decoder_layers": 1,
    "use_vae": True,
    "latent_dim": 32,
    "n_vae_encoder_layers": 4,
    "dropout": 0.1,
    "kl_weight": 10.0,
    "optimizer_lr": 1.0e-5,
    "optimizer_weight_decay": 1.0e-4,
    "optimizer_lr_backbone": 1.0e-5,
}


AUG_CFG = {
    "enabled": True,
    "probability": 1.0,
    "brightness": 0.12,
    "contrast": 0.12,
    "saturation": 0.08,
    "gaussian_noise_std": 0.01,
    "clip_min": 0.0,
    "clip_max": 1.0,
}


MINI_TRAINING = {
    "batch_size": 2,
    "steps": 20,
    "num_workers": 0,
    "log_freq": 1,
    "val_freq": 10,
    "max_val_batches": 4,
    "grad_clip_norm": 10.0,
    "save_checkpoint": True,
    "progress_bar": True,
}


FULL_TRAINING = {
    "batch_size": 32,
    "steps": 100_000,
    "num_workers": 4,
    "log_freq": 100,
    "val_freq": 5_000,
    "max_val_batches": 64,
    "grad_clip_norm": 10.0,
    "save_checkpoint": True,
    "save_freq": 10_000,
    "progress_bar": True,
}


def _task_name(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return str(value[0])
    if hasattr(value, "tolist"):
        items = value.tolist()
        if isinstance(items, list):
            return str(items[0])
        return str(items)
    return str(value)


def _load_episodes() -> pd.DataFrame:
    episodes_path = ABC_ROOT / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not episodes_path.is_file():
        raise FileNotFoundError(episodes_path)
    df = pd.read_parquet(episodes_path)
    df["task"] = df["tasks"].map(_task_name)
    return df[["episode_index", "task", "length", "dataset_from_index", "dataset_to_index"]].copy()


def _split_full_by_episode_count(df: pd.DataFrame) -> tuple[list[int], list[int]]:
    train: list[int] = []
    val: list[int] = []
    for _, group in df.groupby("task", sort=True):
        ordered = group.sort_values("episode_index")
        n_train = round(len(ordered) * 0.9)
        train.extend(int(v) for v in ordered.iloc[:n_train]["episode_index"])
        val.extend(int(v) for v in ordered.iloc[n_train:]["episode_index"])
    return sorted(train), sorted(val)


def _select_even_frame_budget(group: pd.DataFrame, target_frames: float) -> list[int]:
    ordered = group.sort_values("episode_index").reset_index(drop=True)
    mean_len = max(float(ordered["length"].mean()), 1.0)
    n_select = max(1, min(len(ordered), round(target_frames / mean_len)))

    def spaced_indices(n: int) -> list[int]:
        if n >= len(ordered):
            return list(range(len(ordered)))
        if n == 1:
            return [len(ordered) // 2]
        positions = [round(i * (len(ordered) - 1) / (n - 1)) for i in range(n)]
        return sorted(set(int(pos) for pos in positions))

    selected_positions = spaced_indices(n_select)
    selected = set(selected_positions)

    def selected_frames() -> int:
        return int(ordered.iloc[sorted(selected)]["length"].sum())

    # Adjust by adding/removing evenly spaced candidates until the frame budget
    # cannot be improved. The result is deterministic and approximately balanced.
    improved = True
    while improved:
        improved = False
        current = selected_frames()
        current_err = abs(current - target_frames)
        best_selected = set(selected)
        best_err = current_err

        if current < target_frames:
            for pos in range(len(ordered)):
                if pos in selected:
                    continue
                candidate = set(selected)
                candidate.add(pos)
                frames = int(ordered.iloc[sorted(candidate)]["length"].sum())
                err = abs(frames - target_frames)
                if err < best_err:
                    best_selected = candidate
                    best_err = err
        else:
            for pos in list(selected):
                if len(selected) <= 1:
                    continue
                candidate = set(selected)
                candidate.remove(pos)
                frames = int(ordered.iloc[sorted(candidate)]["length"].sum())
                err = abs(frames - target_frames)
                if err < best_err:
                    best_selected = candidate
                    best_err = err

        if best_err < current_err:
            selected = best_selected
            improved = True

    return sorted(int(v) for v in ordered.iloc[sorted(selected)]["episode_index"])


def _split_size_matched(df: pd.DataFrame, full_train: list[int], full_val: list[int]) -> tuple[list[int], list[int]]:
    train_df = df[df["episode_index"].isin(full_train)]
    target_per_env = B_TRAIN_FRAMES / 3.0
    selected: list[int] = []
    for _, group in train_df.groupby("task", sort=True):
        selected.extend(_select_even_frame_budget(group, target_per_env))
    return sorted(selected), sorted(full_val)


def _mini_split(df: pd.DataFrame) -> tuple[list[int], list[int]]:
    train: list[int] = []
    val: list[int] = []
    for _, group in df.groupby("task", sort=True):
        ordered = group.sort_values("episode_index")
        train.extend(int(v) for v in ordered.iloc[:2]["episode_index"])
        val.extend(int(v) for v in ordered.iloc[2:4]["episode_index"])
    return sorted(train), sorted(val)


def _frames(df: pd.DataFrame, episodes: list[int]) -> int:
    return int(df[df["episode_index"].isin(episodes)]["length"].sum())


def _config(
    *,
    name: str,
    output_dir: str,
    train_episodes: list[int],
    val_episodes: list[int],
    training: dict[str, Any],
    augmentation: bool,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "experiment": {"name": name, "seed": 1000, "output_dir": output_dir},
        "dataset": {
            "repo_id": "local/calvin_ABC",
            "root": str(ABC_ROOT),
            "train_episodes": train_episodes,
            "val_episodes": val_episodes,
            "use_imagenet_stats": True,
            "return_uint8": True,
            "video_backend": None,
        },
        "policy": dict(POLICY_CFG),
        "training": dict(training),
    }
    if augmentation:
        cfg["augmentation"] = dict(AUG_CFG)
    return cfg


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, width=120)


def main() -> int:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_episodes()
    full_train, full_val = _split_full_by_episode_count(df)
    size_train, size_val = _split_size_matched(df, full_train, full_val)
    mini_train, mini_val = _mini_split(df)

    split_payload = {
        "dataset": {"repo_id": "local/calvin_ABC", "root": str(ABC_ROOT)},
        "reference": {
            "act_B_full_train_frames": B_TRAIN_FRAMES,
            "size_matched_target_total_train_frames": B_TRAIN_FRAMES,
            "size_matched_target_train_frames_per_environment": B_TRAIN_FRAMES / 3.0,
        },
        "splits": {
            "mini": {"train_episodes": mini_train, "val_episodes": mini_val},
            "full": {"train_episodes": full_train, "val_episodes": full_val},
            "size_matched": {"train_episodes": size_train, "val_episodes": size_val},
        },
    }
    _write_yaml(CONFIG_DIR / "task2_episode_splits.yaml", split_payload)

    config_specs = [
        ("act_ABC", "project/outputs/task2/act_ABC/mini_trial", mini_train, mini_val, MINI_TRAINING, False),
        ("act_ABC_aug", "project/outputs/task2/act_ABC_aug/mini_trial", mini_train, mini_val, MINI_TRAINING, True),
        (
            "act_ABC_size_matched",
            "project/outputs/task2/act_ABC_size_matched/mini_trial",
            mini_train,
            mini_val,
            MINI_TRAINING,
            False,
        ),
        (
            "act_ABC_size_matched_aug",
            "project/outputs/task2/act_ABC_size_matched_aug/mini_trial",
            mini_train,
            mini_val,
            MINI_TRAINING,
            True,
        ),
        (
            "act_ABC_full",
            "/EXT_DISK/users/zengzixuan/calvin_runs/task2/act_ABC/full_manual_override",
            full_train,
            full_val,
            FULL_TRAINING,
            False,
        ),
        (
            "act_ABC_aug_full",
            "/EXT_DISK/users/zengzixuan/calvin_runs/task2/act_ABC_aug/full_manual_override",
            full_train,
            full_val,
            FULL_TRAINING,
            True,
        ),
        (
            "act_ABC_size_matched_full",
            "/EXT_DISK/users/zengzixuan/calvin_runs/task2/act_ABC_size_matched/full_manual_override",
            size_train,
            size_val,
            FULL_TRAINING,
            False,
        ),
        (
            "act_ABC_size_matched_aug_full",
            "/EXT_DISK/users/zengzixuan/calvin_runs/task2/act_ABC_size_matched_aug/full_manual_override",
            size_train,
            size_val,
            FULL_TRAINING,
            True,
        ),
    ]
    for name, output_dir, train_eps, val_eps, training, augmentation in config_specs:
        _write_yaml(
            CONFIG_DIR / f"{name}.yaml",
            _config(
                name=name,
                output_dir=output_dir,
                train_episodes=train_eps,
                val_episodes=val_eps,
                training=training,
                augmentation=augmentation,
            ),
        )

    summary_rows = []
    split_defs = {
        "mini": (mini_train, mini_val),
        "full": (full_train, full_val),
        "size_matched": (size_train, size_val),
    }
    for split_name, (train_eps, val_eps) in split_defs.items():
        for subset_name, episodes in [("train", train_eps), ("val", val_eps)]:
            subset = df[df["episode_index"].isin(episodes)]
            for task, group in subset.groupby("task", sort=True):
                summary_rows.append(
                    {
                        "split": split_name,
                        "subset": subset_name,
                        "environment": task.replace("calvin_env_", "").replace("_play", ""),
                        "num_episodes": len(group),
                        "num_frames": int(group["length"].sum()),
                        "episode_min": int(group["episode_index"].min()) if len(group) else "",
                        "episode_max": int(group["episode_index"].max()) if len(group) else "",
                    }
                )
            summary_rows.append(
                {
                    "split": split_name,
                    "subset": subset_name,
                    "environment": "ABC_total",
                    "num_episodes": len(subset),
                    "num_frames": int(subset["length"].sum()),
                    "episode_min": int(subset["episode_index"].min()) if len(subset) else "",
                    "episode_max": int(subset["episode_index"].max()) if len(subset) else "",
                }
            )

    with (TABLE_DIR / "task2_episode_splits.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "subset", "environment", "num_episodes", "num_frames", "episode_min", "episode_max"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    matrix_rows = [
        {
            "experiment": "act_ABC_full",
            "config": "project/configs/act_ABC_full.yaml",
            "augmentation": "no",
            "size_control": "full ABC data",
            "default_manual_gpu": 1,
        },
        {
            "experiment": "act_ABC_aug_full",
            "config": "project/configs/act_ABC_aug_full.yaml",
            "augmentation": "yes, matched to ACT-B-aug",
            "size_control": "full ABC data",
            "default_manual_gpu": 3,
        },
        {
            "experiment": "act_ABC_size_matched_full",
            "config": "project/configs/act_ABC_size_matched_full.yaml",
            "augmentation": "no",
            "size_control": "balanced to ACT-B train-frame budget",
            "default_manual_gpu": 2,
        },
        {
            "experiment": "act_ABC_size_matched_aug_full",
            "config": "project/configs/act_ABC_size_matched_aug_full.yaml",
            "augmentation": "yes, matched to ACT-B-aug",
            "size_control": "balanced to ACT-B train-frame budget",
            "default_manual_gpu": 4,
        },
    ]
    with (TABLE_DIR / "task2_experiment_matrix.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["experiment", "config", "augmentation", "size_control", "default_manual_gpu"],
        )
        writer.writeheader()
        writer.writerows(matrix_rows)

    full_train_frames = _frames(df, full_train)
    size_train_frames = _frames(df, size_train)
    exposure_rows = [
        {
            "experiment": "act_B_aug_reference",
            "train_frames": B_TRAIN_FRAMES,
            "per_sample_aug_strength": "task1_aug",
            "aug_probability": AUG_CFG["probability"],
            "augmented_frame_exposure_ratio_vs_act_B_aug": 1.0,
            "notes": "Task 1 reference augmentation magnitude.",
        },
        {
            "experiment": "act_ABC_size_matched_aug_full",
            "train_frames": size_train_frames,
            "per_sample_aug_strength": "task1_aug",
            "aug_probability": AUG_CFG["probability"],
            "augmented_frame_exposure_ratio_vs_act_B_aug": size_train_frames / B_TRAIN_FRAMES,
            "notes": "Same per-sample augmentation magnitude; frame budget is matched to ACT-B.",
        },
        {
            "experiment": "act_ABC_aug_full",
            "train_frames": full_train_frames,
            "per_sample_aug_strength": "task1_aug",
            "aug_probability": AUG_CFG["probability"],
            "augmented_frame_exposure_ratio_vs_act_B_aug": full_train_frames / B_TRAIN_FRAMES,
            "notes": "Same per-sample augmentation magnitude; total exposure rises with the larger ABC dataset.",
        },
    ]
    with (TABLE_DIR / "task2_augmentation_exposure.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "train_frames",
                "per_sample_aug_strength",
                "aug_probability",
                "augmented_frame_exposure_ratio_vs_act_B_aug",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(exposure_rows)

    print(yaml.safe_dump({"generated_configs": [name for name, *_ in config_specs], "summary": summary_rows}, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
