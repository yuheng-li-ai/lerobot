#!/usr/bin/env python
"""CALVIN D success-rate evaluation for ACT-Lang checkpoints.

This script uses CALVIN's official long-horizon sequence/task-oracle logic, but
wraps the local LeRobot ACT-Lang checkpoints as a `reset()` / `step(obs, goal)`
policy. It evaluates only language-conditioned checkpoints; the older
non-language ACT checkpoints remain covered by the offline D action-error
analysis.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from numpy import pi
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALVIN_ROOT = Path(os.environ.get("CALVIN_REPO", "/home/zengzixuan/cvprojects/calvin"))
CALVIN_MODELS = CALVIN_ROOT / "calvin_models"
CALVIN_ENV_REPO = CALVIN_ROOT / "calvin_env"
CALVIN_RAW = Path(os.environ.get("CALVIN_RAW", "/SSD_DISK/users/zengzixuan/calvin/task_ABC_D"))

for path in [PROJECT_ROOT, Path(__file__).resolve().parent, CALVIN_ROOT, CALVIN_MODELS, CALVIN_ENV_REPO]:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.pop("LEROBOT_HOME", None)

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.processor import DataProcessorPipeline  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402

try:
    from train_act_lang import ACTLangConfig, ACTLangPolicy  # noqa: E402
except ModuleNotFoundError:
    from project.scripts.train_act_lang import ACTLangConfig, ACTLangPolicy  # noqa: E402


DEFAULT_EP_LEN = 360
ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
DEFAULT_MODELS = {
    "act_lang_B_100k": Path(
        "/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_B/"
        "20260606_112848_act_lang_B_full_gpu0/checkpoint"
    ),
    "act_lang_ABC_size_matched_100k": Path(
        "/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_ABC_size_matched/"
        "20260606_164015_act_lang_ABC_size_matched_full_gpu2/checkpoint"
    ),
    "act_lang_ABC_200k": Path(
        "/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_ABC_continue_200k/"
        "20260607_083058_act_lang_ABC_full_continue_200k_gpu1/checkpoint"
    ),
}

try:
    import pyhash

    _FNV_HASHER = pyhash.fnv1_32()
except ModuleNotFoundError:
    _FNV_HASHER = None


@contextlib.contextmanager
def temp_seed(seed: int):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def stable_initial_condition_seed(initial_condition: dict[str, Any]) -> int:
    if _FNV_HASHER is not None:
        return int(_FNV_HASHER(str(initial_condition.values())))
    # Fallback only used if pyhash is unavailable; pyhash is present in the current env.
    import zlib

    return int(zlib.crc32(str(initial_condition.values()).encode("utf-8")) & 0xFFFFFFFF)


def get_env_state_for_initial_condition(initial_condition: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Official CALVIN initial-state conversion, kept local to avoid heavy training imports."""
    robot_obs = np.array(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ]
    )
    block_rot_z_range = (pi / 2 - pi / 8, pi / 2 + pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]
    seed = stable_initial_condition_seed(initial_condition)
    with temp_seed(seed):
        np.random.shuffle(block_table)
        scene_obs = np.zeros(24)
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
        scene_obs[4] = initial_condition["lightbulb"]
        scene_obs[5] = initial_condition["led"]
        if initial_condition["red_block"] == "slider_right":
            scene_obs[6:9] = block_slider_right
        elif initial_condition["red_block"] == "slider_left":
            scene_obs[6:9] = block_slider_left
        else:
            scene_obs[6:9] = block_table[0]
        scene_obs[11] = np.random.uniform(*block_rot_z_range)
        if initial_condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif initial_condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        elif initial_condition["red_block"] == "table":
            scene_obs[12:15] = block_table[1]
        else:
            scene_obs[12:15] = block_table[0]
        scene_obs[17] = np.random.uniform(*block_rot_z_range)
        if initial_condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif initial_condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        else:
            scene_obs[18:21] = block_table[1]
        scene_obs[23] = np.random.uniform(*block_rot_z_range)
    return robot_obs, scene_obs


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    checkpoint: Path


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


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def safe_quantile(values: list[float], q: float) -> float:
    return float(np.quantile(values, q)) if values else math.nan


def normalize_text(value: Any) -> str:
    if isinstance(value, (list, tuple)) and value:
        return normalize_text(value[0])
    return str(value).strip()


def normalize_embedding(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape == (1, 384):
        arr = arr[0]
    if arr.shape != (384,):
        raise ValueError(f"Expected language embedding shape (384,) or (1, 384), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Language embedding contains NaN or Inf")
    return arr.astype(np.float32, copy=False)


def load_validation_goal_tables(
    raw_validation_dir: Path,
    val_annotations_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, str]]:
    """Map CALVIN task names and official annotation strings to 384D embeddings."""
    ann_path = raw_validation_dir / "lang_annotations" / "auto_lang_ann.npy"
    if not ann_path.is_file():
        raise FileNotFoundError(ann_path)
    annotations = np.load(ann_path, allow_pickle=True).item()
    tasks = annotations["language"]["task"]
    embeddings = annotations["language"]["emb"]

    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for task, embedding in zip(tasks, embeddings, strict=True):
        grouped[str(task)].append(normalize_embedding(embedding))
    task_to_embedding = {task: np.stack(vals, axis=0).mean(axis=0).astype(np.float32) for task, vals in grouped.items()}

    val_annotations = yaml.safe_load(val_annotations_path.read_text())
    task_to_text: dict[str, str] = {}
    text_to_task: dict[str, str] = {}
    missing = []
    for task, texts in val_annotations.items():
        text = normalize_text(texts)
        task_to_text[task] = text
        text_to_task[text] = task
        if task not in task_to_embedding:
            missing.append(task)
    if missing:
        raise KeyError(f"Official validation tasks missing from raw language embeddings: {missing}")
    return task_to_embedding, text_to_task, task_to_text


def format_progress(done: int, total: int, elapsed_s: float, width: int = 30) -> str:
    pct = done / max(total, 1)
    filled = min(width, max(0, int(round(width * pct))))
    eta_s = elapsed_s * (total - done) / done if done > 0 else math.nan
    return (
        f"[{'#' * filled}{'-' * (width - filled)}] "
        f"{done}/{total} ({pct * 100:5.1f}%) elapsed={elapsed_s / 60:.1f}m eta={eta_s / 60:.1f}m"
    )


def to_chw_float_tensor(image: Any) -> torch.Tensor:
    tensor = image.detach().cpu() if isinstance(image, torch.Tensor) else torch.as_tensor(image)
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dims, got {tuple(tensor.shape)}")
    if tensor.shape[-1] in {1, 3}:
        tensor = tensor.permute(2, 0, 1)
    tensor = tensor.to(dtype=torch.float32)
    if tensor.max().item() > 2.0:
        tensor = tensor / 255.0
    return tensor


def to_state_tensor(obs: dict[str, Any]) -> torch.Tensor:
    if "robot_obs" in obs:
        value = obs["robot_obs"]
    elif "robot_obs_raw" in obs:
        value = obs["robot_obs_raw"]
    else:
        raise KeyError("CALVIN observation is missing robot_obs")
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 2 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.shape != (15,):
        raise ValueError(f"Expected robot_obs shape (15,), got {tuple(tensor.shape)}")
    return tensor.to(dtype=torch.float32)


class ACTLangCalvinPolicy:
    """Adapter from raw CALVIN observations to the local ACT-Lang policy."""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        device: str,
        task_to_embedding: dict[str, np.ndarray],
        text_to_task: dict[str, str],
        rollout_action_steps: int | None,
    ) -> None:
        self.spec = spec
        self.device = device
        self.task_to_embedding = task_to_embedding
        self.text_to_task = text_to_task
        cfg = PreTrainedConfig.from_pretrained(spec.checkpoint)
        if not isinstance(cfg, ACTLangConfig):
            raise TypeError(f"Expected ACTLangConfig in {spec.checkpoint}, got {type(cfg).__name__}")
        cfg.device = device
        if rollout_action_steps is not None:
            if rollout_action_steps < 1 or rollout_action_steps > cfg.chunk_size:
                raise ValueError(
                    f"rollout_action_steps must be in [1, {cfg.chunk_size}], got {rollout_action_steps}"
                )
            cfg.n_action_steps = int(rollout_action_steps)

        self.policy = ACTLangPolicy.from_pretrained(
            spec.checkpoint,
            config=cfg,
            local_files_only=True,
            strict=True,
        )
        self.policy.to(device)
        self.policy.eval()
        self.preprocessor = DataProcessorPipeline.from_pretrained(
            spec.checkpoint,
            config_filename="policy_preprocessor.json",
        )
        self.postprocessor = DataProcessorPipeline.from_pretrained(
            spec.checkpoint,
            config_filename="policy_postprocessor.json",
        )
        self.reset()
        self.current_subtask = ""
        self.current_language = ""
        self.current_actions: list[np.ndarray] = []
        self.current_chunk_starts: list[int] = []

    def reset(self) -> None:
        self.policy.reset()

    def begin_subtask(self, subtask: str, language: str) -> None:
        self.reset()
        self.current_subtask = subtask
        self.current_language = language
        self.current_actions = []
        self.current_chunk_starts = []

    def _task_for_goal(self, goal: str) -> str:
        goal_text = normalize_text(goal)
        if goal_text in self.text_to_task:
            return self.text_to_task[goal_text]
        if goal_text in self.task_to_embedding:
            return goal_text
        raise KeyError(
            f"Cannot map goal text to CALVIN task: {goal_text!r}. "
            "The wrapper expects official new_playtable_validation annotations."
        )

    def _batch_from_obs(self, obs: dict[str, Any], goal: str) -> dict[str, torch.Tensor]:
        if "rgb_obs" not in obs:
            raise KeyError("CALVIN observation is missing rgb_obs")
        rgb_obs = obs["rgb_obs"]
        task = self._task_for_goal(goal)
        embedding = torch.from_numpy(self.task_to_embedding[task]).to(dtype=torch.float32)
        return {
            "observation.images.static": to_chw_float_tensor(rgb_obs["rgb_static"]).unsqueeze(0),
            "observation.images.gripper": to_chw_float_tensor(rgb_obs["rgb_gripper"]).unsqueeze(0),
            "observation.state": to_state_tensor(obs).unsqueeze(0),
            self.policy.config.language_embedding_key: embedding.unsqueeze(0),
        }

    @torch.no_grad()
    def step(self, obs: dict[str, Any], goal: str) -> np.ndarray:
        new_chunk = bool(
            hasattr(self.policy, "_action_queue")
            and len(getattr(self.policy, "_action_queue")) == 0
        )
        if new_chunk:
            self.current_chunk_starts.append(len(self.current_actions))

        batch = self._batch_from_obs(obs, goal)
        processed = self.preprocessor(batch)
        action_norm = self.policy.select_action(processed)
        action = self.postprocessor({ACTION: action_norm})[ACTION]
        action_np = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
        action_np[-1] = 1.0 if action_np[-1] > 0 else -1.0
        self.current_actions.append(action_np)
        return action_np

    def finish_subtask(self, success: bool, steps: int) -> dict[str, Any]:
        actions = np.asarray(self.current_actions, dtype=np.float32)
        deltas: list[float] = []
        boundary_jumps: list[float] = []
        if len(actions) > 1:
            deltas = np.linalg.norm(np.diff(actions[:, :6], axis=0), axis=1).astype(float).tolist()
        for start in self.current_chunk_starts:
            if start > 0 and start < len(actions):
                boundary_jumps.append(float(np.linalg.norm(actions[start, :6] - actions[start - 1, :6])))
        return {
            "num_actions": int(len(actions)),
            "num_chunks": int(len(self.current_chunk_starts)),
            "success": bool(success),
            "steps": int(steps),
            "mean_action_delta_l2_first6": safe_mean(deltas),
            "q90_action_delta_l2_first6": safe_quantile(deltas, 0.90),
            "mean_chunk_boundary_jump_l2_first6": safe_mean(boundary_jumps),
            "q90_chunk_boundary_jump_l2_first6": safe_quantile(boundary_jumps, 0.90),
            "mean_action_norm_l2_first6": safe_mean(np.linalg.norm(actions[:, :6], axis=1).astype(float).tolist())
            if len(actions)
            else math.nan,
            "mean_abs_gripper": safe_mean(np.abs(actions[:, 6]).astype(float).tolist()) if len(actions) else math.nan,
        }


def count_chain_success(results: list[int]) -> dict[str, float]:
    count = Counter(results)
    return {f"sr_chain_{i}": sum(count[j] for j in reversed(range(i, 6))) / max(len(results), 1) for i in range(1, 6)}


def instantiate_calvin_components(dataset_path: Path):
    try:
        import types

        utils_stub = types.ModuleType("calvin_agent.evaluation.utils")
        utils_stub.temp_seed = temp_seed
        sys.modules.setdefault("calvin_agent.evaluation.utils", utils_stub)
        from calvin_agent.evaluation.multistep_sequences import get_sequences
        from calvin_env.envs.play_table_env import get_env
        import hydra
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "CALVIN success-rate evaluation is missing a required module: "
            f"{exc.name!r}. The official rollout requires the CALVIN environment package "
            f"under {CALVIN_ENV_REPO} plus its runtime dependencies."
        ) from exc

    conf_dir = CALVIN_MODELS / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    obs_space = {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": [],
        "state_obs": ["robot_obs", "scene_obs"],
        "actions": ["rel_actions"],
    }
    env = get_env(dataset_path / "validation", obs_space=obs_space, show_gui=False)
    return env, task_oracle, val_annotations, get_sequences, get_env_state_for_initial_condition


def evaluate_model(
    spec: ModelSpec,
    *,
    env,
    task_oracle,
    val_annotations,
    eval_sequences,
    get_env_state_for_initial_condition,
    device: str,
    task_to_embedding: dict[str, np.ndarray],
    text_to_task: dict[str, str],
    rollout_action_steps: int | None,
    ep_len: int,
    log_freq: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    model = ACTLangCalvinPolicy(
        spec,
        device=device,
        task_to_embedding=task_to_embedding,
        text_to_task=text_to_task,
        rollout_action_steps=rollout_action_steps,
    )
    sequence_rows: list[dict[str, Any]] = []
    subtask_rows: list[dict[str, Any]] = []
    results: list[int] = []
    task_success = Counter()
    task_total = Counter()
    start_time = time.perf_counter()

    for seq_idx, (initial_state, eval_sequence) in enumerate(eval_sequences, start=1):
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        successful_prefix = 0

        for subtask_idx, subtask in enumerate(eval_sequence, start=1):
            obs = env.get_obs()
            lang_annotation = normalize_text(val_annotations[subtask][0])
            model.begin_subtask(subtask, lang_annotation)
            start_info = env.get_info()
            success = False
            steps_taken = ep_len

            for step in range(ep_len):
                action = model.step(obs, lang_annotation)
                obs, _, _, current_info = env.step(action)
                current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
                if len(current_task_info) > 0:
                    success = True
                    steps_taken = step + 1
                    break

            chunk_metrics = model.finish_subtask(success, steps_taken)
            task_total[subtask] += 1
            if success:
                task_success[subtask] += 1
                successful_prefix += 1
            subtask_rows.append(
                {
                    "model": spec.key,
                    "display_name": spec.display_name,
                    "sequence_idx": seq_idx,
                    "subtask_idx": subtask_idx,
                    "subtask": subtask,
                    "language": lang_annotation,
                    **chunk_metrics,
                }
            )
            if not success:
                break

        results.append(successful_prefix)
        sequence_rows.append(
            {
                "model": spec.key,
                "display_name": spec.display_name,
                "sequence_idx": seq_idx,
                "successful_subtasks": successful_prefix,
                "sequence": " -> ".join(eval_sequence),
            }
        )
        if log_freq > 0 and (seq_idx % log_freq == 0 or seq_idx == len(eval_sequences)):
            print(
                f"PROGRESS {spec.key} {format_progress(seq_idx, len(eval_sequences), time.perf_counter() - start_time)} "
                + " ".join(f"{k}={v * 100:.1f}%" for k, v in count_chain_success(results).items()),
                flush=True,
            )

    chain = count_chain_success(results)
    summary = {
        "model": spec.key,
        "display_name": spec.display_name,
        "checkpoint": str(spec.checkpoint),
        "num_sequences": len(results),
        "avg_successful_sequence_length": float(np.mean(results)) if results else math.nan,
        **chain,
        "attempted_subtasks": int(sum(task_total.values())),
        "single_subtask_success_rate": float(sum(task_success.values()) / max(sum(task_total.values()), 1)),
        "rollout_action_steps": int(model.policy.config.n_action_steps),
        "chunk_size": int(model.policy.config.chunk_size),
    }
    task_rows = [
        {
            "model": spec.key,
            "display_name": spec.display_name,
            "task": task,
            "success": int(task_success[task]),
            "total": int(task_total[task]),
            "success_rate": float(task_success[task] / max(task_total[task], 1)),
        }
        for task in sorted(task_total)
    ]
    return summary, sequence_rows, subtask_rows, {"results": results, "task_rows": task_rows}


def plot_success_chain(summary_rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(5)
    width = 0.8 / max(len(summary_rows), 1)
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for idx, row in enumerate(summary_rows):
        values = [float(row[f"sr_chain_{i}"]) * 100 for i in range(1, 6)]
        ax.bar(x + idx * width, values, width=width, label=row["display_name"])
    ax.set_xticks(x + width * (len(summary_rows) - 1) / 2)
    ax.set_xticklabels([f"{i}/5" for i in range(1, 6)])
    ax.set_ylabel("Success rate (%)")
    ax.set_xlabel("Consecutive subtasks solved")
    ax.set_title("CALVIN D Long-Horizon Success Rate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_avg_sequence_length(summary_rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["display_name"].replace(" ", "\n") for row in summary_rows]
    values = [float(row["avg_successful_sequence_length"]) for row in summary_rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(labels, values, color=["#2f6f9f", "#8f6ab8", "#b23a48"][: len(values)])
    ax.set_ylabel("Average successful sequence length")
    ax.set_ylim(0, 5)
    ax.set_title("CALVIN D Average Successful Chain Length")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_chunk_diagnostics(subtask_rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in subtask_rows:
        grouped[row["display_name"]].append(row)
    labels = []
    smooth = []
    jumps = []
    for label, rows in grouped.items():
        labels.append(label.replace(" ", "\n"))
        smooth_values = [float(r["mean_action_delta_l2_first6"]) for r in rows]
        jump_values = [float(r["mean_chunk_boundary_jump_l2_first6"]) for r in rows]
        smooth.append(safe_mean([v for v in smooth_values if math.isfinite(v)]))
        jumps.append(safe_mean([v for v in jump_values if math.isfinite(v)]))

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    axes[0].bar(x, smooth, color="#2f6f9f")
    axes[0].set_title("Within-Chunk Action Delta")
    axes[0].set_ylabel("Mean L2, first 6 dims")
    axes[1].bar(x, jumps, color="#b23a48")
    axes[1].set_title("Chunk Boundary Jump")
    axes[1].set_ylabel("Mean L2, first 6 dims")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("ACT Chunking Diagnostics During D Rollouts")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_model_specs(model_args: list[str] | None) -> list[ModelSpec]:
    selected = model_args or list(DEFAULT_MODELS)
    specs = []
    display_names = {
        "act_lang_B_100k": "ACT-Lang-B 100k",
        "act_lang_ABC_size_matched_100k": "ACT-Lang-ABC Size-Matched 100k",
        "act_lang_ABC_200k": "ACT-Lang-ABC 200k",
    }
    for key in selected:
        if "=" in key:
            name, checkpoint = key.split("=", 1)
            specs.append(ModelSpec(name, display_names.get(name, name), Path(checkpoint)))
        else:
            if key not in DEFAULT_MODELS:
                raise KeyError(f"Unknown model {key!r}; use one of {sorted(DEFAULT_MODELS)} or name=/path/to/checkpoint")
            specs.append(ModelSpec(key, display_names[key], DEFAULT_MODELS[key]))
    for spec in specs:
        if not (spec.checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(f"Missing ACT-Lang checkpoint weights: {spec.checkpoint}")
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=CALVIN_RAW)
    parser.add_argument("--raw-validation-dir", type=Path, default=CALVIN_RAW / "validation")
    parser.add_argument(
        "--val-annotations",
        type=Path,
        default=CALVIN_MODELS / "conf/annotations/new_playtable_validation.yaml",
    )
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--num-sequences", type=int, default=1000)
    parser.add_argument("--ep-len", type=int, default=DEFAULT_EP_LEN)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--rollout-action-steps",
        type=int,
        default=25,
        help="Number of ACT chunk actions to execute before replanning. Use 100 for trained config default.",
    )
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("project/outputs/task3_success_rate_D"))
    parser.add_argument("--table-dir", type=Path, default=Path("project/tables/task3_success_rate_D"))
    parser.add_argument("--figure-dir", type=Path, default=Path("project/figures/task3_success_rate_D"))
    args = parser.parse_args()

    specs = parse_model_specs(args.models)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    task_to_embedding, text_to_task, task_to_text = load_validation_goal_tables(
        args.raw_validation_dir,
        args.val_annotations,
    )
    env, task_oracle, val_annotations, get_sequences, get_env_state_for_initial_condition = instantiate_calvin_components(
        args.dataset_path
    )
    eval_sequences = get_sequences(args.num_sequences)

    manifest = {
        "dataset_path": str(args.dataset_path),
        "raw_validation_dir": str(args.raw_validation_dir),
        "num_sequences": int(args.num_sequences),
        "ep_len": int(args.ep_len),
        "device": args.device,
        "rollout_action_steps": int(args.rollout_action_steps),
        "models": [spec.__dict__ | {"checkpoint": str(spec.checkpoint)} for spec in specs],
        "goal_embedding_rule": "mean raw validation lang_annotations embedding per official CALVIN task name",
        "task_to_text": task_to_text,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"event": "start", **manifest}, indent=2), flush=True)

    summary_rows: list[dict[str, Any]] = []
    all_sequence_rows: list[dict[str, Any]] = []
    all_subtask_rows: list[dict[str, Any]] = []
    all_task_rows: list[dict[str, Any]] = []
    raw_results: dict[str, Any] = {}

    for spec in specs:
        print(json.dumps({"event": "model_start", "model": spec.key, "checkpoint": str(spec.checkpoint)}), flush=True)
        summary, sequence_rows, subtask_rows, extra = evaluate_model(
            spec,
            env=env,
            task_oracle=task_oracle,
            val_annotations=val_annotations,
            eval_sequences=eval_sequences,
            get_env_state_for_initial_condition=get_env_state_for_initial_condition,
            device=args.device,
            task_to_embedding=task_to_embedding,
            text_to_task=text_to_task,
            rollout_action_steps=args.rollout_action_steps,
            ep_len=int(args.ep_len),
            log_freq=args.log_freq,
        )
        summary_rows.append(summary)
        all_sequence_rows.extend(sequence_rows)
        all_subtask_rows.extend(subtask_rows)
        all_task_rows.extend(extra["task_rows"])
        raw_results[spec.key] = extra["results"]
        print(json.dumps({"event": "model_done", **summary}), flush=True)

    write_rows(args.table_dir / "success_rate_D_summary.csv", summary_rows)
    write_rows(args.table_dir / "success_rate_D_sequences.csv", all_sequence_rows)
    write_rows(args.table_dir / "success_rate_D_subtasks.csv", all_subtask_rows)
    write_rows(args.table_dir / "success_rate_D_task_breakdown.csv", all_task_rows)
    write_json(args.output_dir / "results.json", {"summary": summary_rows, "raw_results": raw_results})

    plot_success_chain(summary_rows, args.figure_dir / "success_rate_chain_D.png")
    plot_avg_sequence_length(summary_rows, args.figure_dir / "avg_successful_sequence_length_D.png")
    plot_chunk_diagnostics(all_subtask_rows, args.figure_dir / "rollout_chunk_diagnostics_D.png")

    print(
        json.dumps(
            {
                "event": "done",
                "summary_csv": str(args.table_dir / "success_rate_D_summary.csv"),
                "sequence_csv": str(args.table_dir / "success_rate_D_sequences.csv"),
                "subtask_csv": str(args.table_dir / "success_rate_D_subtasks.csv"),
                "task_csv": str(args.table_dir / "success_rate_D_task_breakdown.csv"),
                "figure_dir": str(args.figure_dir),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
