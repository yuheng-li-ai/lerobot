#!/usr/bin/env python3
"""Build language-aligned CALVIN LeRobot datasets for success-rate evaluation.

This converter is intentionally stricter than the Phase 0 play-data converter:
one CALVIN language annotation segment becomes one LeRobot episode. This makes
ACT action chunks respect language-goal boundaries, so the generated dataset can
train a policy that is compatible with CALVIN's official `step(obs, goal)` style
success-rate evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

# LeRobot errors if the deprecated variable is still set in some environments.
os.environ.pop("LEROBOT_HOME", None)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from calvin_phase0_common import (  # noqa: E402
    FrameRange,
    configured_lerobot_root,
    configured_raw_root,
    frame_file,
    load_environment_ranges,
    load_episode_ranges,
    parse_env_selection,
    split_dir,
)

FPS = 30
DEFAULT_CHUNK_SIZE = 100
DEFAULT_VAL_FRACTION = 0.10
LANGUAGE_EMBEDDING_DIM = 384

CALVIN_LANG_FEATURES = {
    "observation.images.static": {
        "dtype": "video",
        "shape": (200, 200, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.gripper": {
        "dtype": "video",
        "shape": (84, 84, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (15,),
        "names": [
            "tcp_x",
            "tcp_y",
            "tcp_z",
            "tcp_roll",
            "tcp_pitch",
            "tcp_yaw",
            "gripper_width",
            "joint_0",
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "gripper_action",
        ],
    },
    "observation.language_embedding": {
        "dtype": "float32",
        "shape": (LANGUAGE_EMBEDDING_DIM,),
        "names": [f"lang_emb_{idx:03d}" for idx in range(LANGUAGE_EMBEDDING_DIM)],
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
    },
}

REQUIRED_RAW_KEYS = {
    "rgb_static": ((200, 200, 3), np.uint8),
    "rgb_gripper": ((84, 84, 3), np.uint8),
    "robot_obs": ((15,), np.floating),
    "rel_actions": ((7,), np.floating),
}


def storage_features(use_videos: bool) -> dict[str, dict[str, Any]]:
    """Return LeRobot feature specs compatible with the chosen visual storage."""
    features: dict[str, dict[str, Any]] = {}
    for key, spec in CALVIN_LANG_FEATURES.items():
        copied = dict(spec)
        if not use_videos and copied["dtype"] == "video":
            copied["dtype"] = "image"
        features[key] = copied
    return features


@dataclass(frozen=True)
class LanguageSegment:
    raw_segment_id: int
    env: str
    split_name: str
    raw_episode_id: int
    raw_episode_start: int
    raw_episode_end: int
    segment_start: int
    segment_end: int
    length: int
    task_name: str
    language_text: str
    embedding: np.ndarray
    split: str = "unassigned"


def normalize_language_text(value: Any) -> str:
    """Return one deterministic string from CALVIN language annotations."""
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return str(value.item()).strip()
        if len(value) == 0:
            return ""
        return normalize_language_text(value[0])
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        return normalize_language_text(value[0])
    return str(value).strip()


def normalize_language_embedding(value: Any) -> np.ndarray:
    emb = np.asarray(value, dtype=np.float32)
    if emb.shape == (1, LANGUAGE_EMBEDDING_DIM):
        emb = emb[0]
    if emb.shape != (LANGUAGE_EMBEDDING_DIM,):
        raise ValueError(f"Expected language embedding shape (384,) or (1, 384), got {emb.shape}")
    if not np.isfinite(emb).all():
        raise ValueError("Language embedding contains NaN or Inf")
    return emb.astype(np.float32, copy=False)


def load_language_annotations(split_path: Path) -> dict[str, Any]:
    ann_path = split_path / "lang_annotations" / "auto_lang_ann.npy"
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing CALVIN language annotations: {ann_path}")
    data = np.load(ann_path, allow_pickle=True).item()
    for top_key in ["info", "language"]:
        if top_key not in data:
            raise KeyError(f"{ann_path} missing top-level key {top_key!r}")
    for info_key in ["indx"]:
        if info_key not in data["info"]:
            raise KeyError(f"{ann_path} missing info key {info_key!r}")
    for lang_key in ["ann", "task", "emb"]:
        if lang_key not in data["language"]:
            raise KeyError(f"{ann_path} missing language key {lang_key!r}")

    n = len(data["info"]["indx"])
    for key in ["ann", "task", "emb"]:
        if len(data["language"][key]) != n:
            raise ValueError(
                f"Language annotation length mismatch for {key}: "
                f"{len(data['language'][key])} vs {n} intervals"
            )
    return data


def episode_lookup(episode_ranges: list[tuple[int, int]]) -> dict[int, tuple[int, int, int]]:
    lookup = {}
    for ep_idx, (start, end) in enumerate(episode_ranges):
        for frame_id in range(start, end + 1):
            lookup[frame_id] = (ep_idx, int(start), int(end))
    return lookup


def find_raw_episode(
    segment_start: int, segment_end: int, episode_ranges: list[tuple[int, int]]
) -> tuple[int, int, int]:
    for ep_idx, (ep_start, ep_end) in enumerate(episode_ranges):
        if ep_start <= segment_start and segment_end <= ep_end:
            return ep_idx, int(ep_start), int(ep_end)
    raise ValueError(
        f"Language segment {segment_start}-{segment_end} does not fall inside one raw CALVIN episode"
    )


def select_language_segments(
    raw_root: Path,
    env_selection: str,
    *,
    allow_partial_scene_overlap: bool,
) -> tuple[list[LanguageSegment], dict[str, int]]:
    env_ranges = load_environment_ranges(raw_root)
    selected_envs = parse_env_selection(env_selection)
    segments: list[LanguageSegment] = []
    counters = {
        "annotation_intervals_seen": 0,
        "segments_selected": 0,
        "segments_skipped_no_overlap": 0,
        "segments_skipped_partial_scene_overlap": 0,
    }

    for env in selected_envs:
        frame_range: FrameRange = env_ranges[env]
        split_path = split_dir(raw_root, frame_range.split)
        annotations = load_language_annotations(split_path)
        raw_episode_ranges = load_episode_ranges(split_path)

        intervals = annotations["info"]["indx"]
        tasks = annotations["language"]["task"]
        texts = annotations["language"]["ann"]
        embeddings = annotations["language"]["emb"]

        for raw_segment_id, ((start, end), task_name, language_text, embedding) in enumerate(
            zip(intervals, tasks, texts, embeddings, strict=True)
        ):
            counters["annotation_intervals_seen"] += 1
            seg_start = int(start)
            seg_end = int(end)
            if seg_end < frame_range.start or seg_start > frame_range.end:
                counters["segments_skipped_no_overlap"] += 1
                continue

            fully_inside_scene = frame_range.start <= seg_start and seg_end <= frame_range.end
            if not fully_inside_scene:
                counters["segments_skipped_partial_scene_overlap"] += 1
                if not allow_partial_scene_overlap:
                    raise ValueError(
                        f"Language segment {seg_start}-{seg_end} partially overlaps {env} "
                        f"scene range {frame_range.start}-{frame_range.end}. This would make "
                        "the success-rate training split ambiguous."
                    )
                seg_start = max(seg_start, frame_range.start)
                seg_end = min(seg_end, frame_range.end)

            ep_idx, ep_start, ep_end = find_raw_episode(seg_start, seg_end, raw_episode_ranges)
            emb = normalize_language_embedding(embedding)
            text = normalize_language_text(language_text)
            if not text:
                raise ValueError(f"Language segment {raw_segment_id} has an empty text annotation")

            segments.append(
                LanguageSegment(
                    raw_segment_id=raw_segment_id,
                    env=env,
                    split_name=frame_range.split,
                    raw_episode_id=ep_idx,
                    raw_episode_start=ep_start,
                    raw_episode_end=ep_end,
                    segment_start=seg_start,
                    segment_end=seg_end,
                    length=seg_end - seg_start + 1,
                    task_name=str(task_name),
                    language_text=text,
                    embedding=emb,
                )
            )

    counters["segments_selected"] = len(segments)
    if not segments:
        raise RuntimeError(f"No language segments selected for {env_selection}")
    return segments, counters


def split_segments(
    segments: list[LanguageSegment],
    *,
    val_fraction: float,
    seed: int,
    max_segments: int | None,
) -> list[LanguageSegment]:
    selected = list(segments)
    rng = random.Random(seed)
    rng.shuffle(selected)
    if max_segments is not None:
        selected = selected[:max_segments]
    if len(selected) < 2:
        raise ValueError("Need at least two language segments to create train/val splits")

    val_count = max(1, round(len(selected) * val_fraction))
    val_count = min(val_count, len(selected) - 1)
    val_ids = set(range(val_count))

    split_assigned = []
    for idx, segment in enumerate(selected):
        split_name = "val" if idx in val_ids else "train"
        split_assigned.append(
            LanguageSegment(
                **{
                    **asdict(segment),
                    "embedding": segment.embedding,
                    "split": split_name,
                }
            )
        )

    # Save train episodes first, then val episodes. This keeps downstream configs
    # simple: train_episodes = 0:N_train, val_episodes = N_train:N_total.
    return sorted(split_assigned, key=lambda row: (row.split != "train", row.env, row.raw_segment_id))


def validate_raw_frame(data: np.lib.npyio.NpzFile, frame_id: int) -> None:
    for key, (expected_shape, dtype_kind) in REQUIRED_RAW_KEYS.items():
        if key not in data:
            raise KeyError(f"Frame {frame_id} missing raw key {key!r}")
        value = data[key]
        if value.shape != expected_shape:
            raise ValueError(f"Frame {frame_id} key {key!r} shape {value.shape}, expected {expected_shape}")
        if dtype_kind is np.uint8:
            if value.dtype != np.uint8:
                raise ValueError(f"Frame {frame_id} key {key!r} dtype {value.dtype}, expected uint8")
        elif not np.issubdtype(value.dtype, dtype_kind):
            raise ValueError(f"Frame {frame_id} key {key!r} dtype {value.dtype}, expected floating")
        if key in {"robot_obs", "rel_actions"} and not np.isfinite(value).all():
            raise ValueError(f"Frame {frame_id} key {key!r} contains NaN or Inf")


def build_frame(data: np.lib.npyio.NpzFile, segment: LanguageSegment) -> dict[str, Any]:
    return {
        "observation.images.static": data["rgb_static"],
        "observation.images.gripper": data["rgb_gripper"],
        "observation.state": data["robot_obs"].astype(np.float32, copy=False),
        "observation.language_embedding": segment.embedding,
        "action": data["rel_actions"].astype(np.float32, copy=False),
        "task": segment.language_text,
    }


def embedding_sha1(embedding: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(embedding).tobytes()).hexdigest()


def manifest_row(episode_index: int, segment: LanguageSegment, chunk_size: int) -> dict[str, Any]:
    return {
        "episode_index": episode_index,
        "split": segment.split,
        "environment": segment.env,
        "source_split": segment.split_name,
        "raw_segment_id": segment.raw_segment_id,
        "raw_episode_id": segment.raw_episode_id,
        "raw_episode_start": segment.raw_episode_start,
        "raw_episode_end": segment.raw_episode_end,
        "segment_start": segment.segment_start,
        "segment_end": segment.segment_end,
        "segment_length": segment.length,
        "task_name": segment.task_name,
        "language_text": segment.language_text,
        "language_embedding_dim": LANGUAGE_EMBEDDING_DIM,
        "language_embedding_sha1": embedding_sha1(segment.embedding),
        "max_valid_actions_from_first_frame": min(chunk_size, segment.length),
        "requires_action_padding_for_chunk": int(segment.length < chunk_size),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_segments(
    segments: list[LanguageSegment],
    counters: dict[str, int],
    *,
    dataset_root: Path,
    repo_id: str,
    chunk_size: int,
) -> dict[str, Any]:
    lengths = np.array([segment.length for segment in segments], dtype=np.int64)
    split_counts = {split: sum(segment.split == split for segment in segments) for split in ["train", "val"]}
    env_counts = {env: sum(segment.env == env for segment in segments) for env in sorted({s.env for s in segments})}
    return {
        "repo_id": repo_id,
        "dataset_root": str(dataset_root),
        "fps": FPS,
        "chunk_size_for_checks": chunk_size,
        "num_language_segment_episodes": len(segments),
        "num_frames": int(lengths.sum()),
        "split_counts": split_counts,
        "environment_counts": env_counts,
        "unique_task_names": len({segment.task_name for segment in segments}),
        "unique_language_texts": len({segment.language_text for segment in segments}),
        "language_embedding_dim": LANGUAGE_EMBEDDING_DIM,
        "segment_length_min": int(lengths.min()),
        "segment_length_median": float(np.median(lengths)),
        "segment_length_mean": float(lengths.mean()),
        "segment_length_max": int(lengths.max()),
        "segments_shorter_than_chunk": int((lengths < chunk_size).sum()),
        "segments_at_least_chunk": int((lengths >= chunk_size).sum()),
        "strict_counters": counters,
        "success_rate_requirement_checks": {
            "one_language_segment_per_lerobot_episode": True,
            "language_embedding_feature_present": True,
            "raw_language_text_saved_as_task": True,
            "chunks_do_not_cross_language_segments": True,
            "action_padding_required_when_chunk_exceeds_segment": bool((lengths < chunk_size).any()),
            "train_episodes_before_val_episodes": True,
        },
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")


def progress_line(name: str, current: int, total: int, started_at: float, width: int = 32) -> str:
    elapsed = max(0.0, time.time() - started_at)
    fraction = current / max(1, total)
    filled = min(width, int(round(width * fraction)))
    bar = "#" * filled + "-" * (width - filled)
    if current > 0 and fraction > 0:
        eta_s = elapsed * (1.0 - fraction) / fraction
        eta = f"{eta_s / 60.0:.1f}m"
    else:
        eta = "unknown"
    return (
        f"PROGRESS {name} [{bar}] {current}/{total} "
        f"({fraction * 100:5.1f}%) elapsed={elapsed / 60.0:.1f}m eta={eta}"
    )


def create_dataset(
    *,
    raw_root: Path,
    output_root: Path,
    repo_prefix: str,
    env_selection: str,
    segments: list[LanguageSegment],
    overwrite: bool,
    use_videos: bool,
    image_writer_threads: int,
    batch_encoding_size: int,
    parallel_encoding: bool,
    chunk_size: int,
    tables_dir: Path,
    progress_log_every: int,
) -> Path:
    repo_name = f"{repo_prefix}_{env_selection}"
    repo_id = f"local/{repo_name}"
    dataset_root = output_root / repo_name

    if dataset_root.exists():
        if not overwrite:
            raise FileExistsError(f"{dataset_root} exists. Use --overwrite to replace it.")
        shutil.rmtree(dataset_root)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=dataset_root,
        fps=FPS,
        robot_type="franka_panda_calvin",
        features=storage_features(use_videos),
        use_videos=use_videos,
        image_writer_threads=image_writer_threads,
        batch_encoding_size=batch_encoding_size,
    )

    manifest_rows: list[dict[str, Any]] = []
    iterator = tqdm(
        enumerate(segments),
        total=len(segments),
        desc=f"{env_selection} language segments",
        unit="segment",
        dynamic_ncols=True,
        mininterval=2.0,
        file=sys.stdout,
    )
    started_at = time.time()
    print(progress_line(f"calvin_lang_{env_selection}", 0, len(segments), started_at), flush=True)
    for episode_index, segment in iterator:
        split_path = split_dir(raw_root, segment.split_name)
        iterator.set_postfix(env=segment.env, split=segment.split, start=segment.segment_start)
        for frame_id in range(segment.segment_start, segment.segment_end + 1):
            path = frame_file(split_path, frame_id)
            if not path.exists():
                raise FileNotFoundError(f"Missing raw CALVIN frame: {path}")
            with np.load(path) as data:
                validate_raw_frame(data, frame_id)
                dataset.add_frame(build_frame(data, segment))
        dataset.save_episode(parallel_encoding=parallel_encoding)
        manifest_rows.append(manifest_row(episode_index, segment, chunk_size))
        done = episode_index + 1
        if done == 1 or done == len(segments) or done % progress_log_every == 0:
            print(
                progress_line(f"calvin_lang_{env_selection}", done, len(segments), started_at),
                flush=True,
            )

    dataset.finalize()

    table_prefix = f"calvin_lang_{env_selection}"
    write_csv(tables_dir / f"{table_prefix}_manifest.csv", manifest_rows)

    train_count = sum(segment.split == "train" for segment in segments)
    split_rows = [
        {
            "split": "train",
            "episode_range": f"0:{train_count}",
            "num_episodes": train_count,
        },
        {
            "split": "val",
            "episode_range": f"{train_count}:{len(segments)}",
            "num_episodes": len(segments) - train_count,
        },
    ]
    write_csv(tables_dir / f"{table_prefix}_episode_splits.csv", split_rows)
    return dataset_root


def validate_dataset_readback(
    *,
    dataset_root: Path,
    repo_id: str,
    segments: list[LanguageSegment],
    chunk_size: int,
    readback_episodes: int,
    use_videos: bool,
) -> list[dict[str, Any]]:
    delta_timestamps = {"action": [idx / FPS for idx in range(chunk_size)]}
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        return_uint8=True,
    )

    required_features = set(storage_features(use_videos))
    missing_features = required_features - set(dataset.features)
    if missing_features:
        raise AssertionError(f"Generated dataset missing features: {sorted(missing_features)}")
    if dataset.num_episodes != len(segments):
        raise AssertionError(f"Readback episodes {dataset.num_episodes}, expected {len(segments)}")
    if dataset.num_frames != sum(segment.length for segment in segments):
        raise AssertionError(
            f"Readback frames {dataset.num_frames}, expected {sum(segment.length for segment in segments)}"
        )

    sample_episode_indices = sorted(
        set(
            np.linspace(
                0,
                len(segments) - 1,
                min(readback_episodes, len(segments)),
                dtype=int,
            ).tolist()
        )
    )

    rows = []
    for ep_idx in sample_episode_indices:
        segment = segments[ep_idx]
        ep = dataset.meta.episodes[ep_idx]
        ep_start = int(ep["dataset_from_index"])
        ep_end = int(ep["dataset_to_index"])
        if ep_end - ep_start != segment.length:
            raise AssertionError(
                f"Episode {ep_idx} readback length {ep_end - ep_start}, expected {segment.length}"
            )

        for position_name, item_idx, expected_valid in [
            ("first", ep_start, min(chunk_size, segment.length)),
            ("last", ep_end - 1, 1),
        ]:
            item = dataset[item_idx]
            action = item["action"]
            action_is_pad = item["action_is_pad"]
            lang = item["observation.language_embedding"]
            if tuple(action.shape) != (chunk_size, 7):
                raise AssertionError(f"Episode {ep_idx} {position_name}: action shape {tuple(action.shape)}")
            if tuple(action_is_pad.shape) != (chunk_size,):
                raise AssertionError(
                    f"Episode {ep_idx} {position_name}: action_is_pad shape {tuple(action_is_pad.shape)}"
                )
            if tuple(lang.shape) != (LANGUAGE_EMBEDDING_DIM,):
                raise AssertionError(
                    f"Episode {ep_idx} {position_name}: language embedding shape {tuple(lang.shape)}"
                )
            actual_valid = int((~action_is_pad).sum().item())
            if actual_valid != expected_valid:
                raise AssertionError(
                    f"Episode {ep_idx} {position_name}: valid action count {actual_valid}, "
                    f"expected {expected_valid}"
                )
            if actual_valid < chunk_size and not bool(action_is_pad[actual_valid:].all().item()):
                raise AssertionError(f"Episode {ep_idx} {position_name}: padding tail is not all True")
            rows.append(
                {
                    "episode_index": ep_idx,
                    "position": position_name,
                    "dataset_index": item_idx,
                    "segment_length": segment.length,
                    "expected_valid_actions": expected_valid,
                    "actual_valid_actions": actual_valid,
                    "pad_count": int(action_is_pad.sum().item()),
                }
            )
    return rows


def dry_run_report(
    segments: list[LanguageSegment],
    counters: dict[str, int],
    *,
    output_root: Path,
    repo_prefix: str,
    env_selection: str,
    chunk_size: int,
    tables_dir: Path,
) -> None:
    repo_id = f"local/{repo_prefix}_{env_selection}"
    dataset_root = output_root / f"{repo_prefix}_{env_selection}"
    summary = summarize_segments(
        segments,
        counters,
        dataset_root=dataset_root,
        repo_id=repo_id,
        chunk_size=chunk_size,
    )
    table_prefix = f"calvin_lang_{env_selection}"
    write_summary(tables_dir / f"{table_prefix}_dry_run_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create language-aligned CALVIN LeRobot datasets for official success-rate training."
    )
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--envs", default="B", choices=["A", "B", "C", "D", "ABC"])
    parser.add_argument("--repo-prefix", default="calvin_lang")
    parser.add_argument("--tables-dir", type=Path, default=Path("project/tables"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument("--batch-encoding-size", type=int, default=1)
    parser.add_argument("--parallel-encoding", action="store_true")
    parser.add_argument("--allow-partial-scene-overlap", action="store_true")
    parser.add_argument("--readback-episodes", type=int, default=32)
    parser.add_argument("--progress-log-every", type=int, default=25)
    args = parser.parse_args()

    if not 0.0 < args.val_fraction < 0.5:
        raise ValueError("--val-fraction must be in (0, 0.5)")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.max_segments is not None and args.max_segments < 2:
        raise ValueError("--max-segments must be at least 2")
    if args.progress_log_every <= 0:
        raise ValueError("--progress-log-every must be positive")

    raw_root = configured_raw_root(args.raw_root)
    output_root = configured_lerobot_root(args.output_root)
    os.environ.setdefault("HF_LEROBOT_HOME", str(output_root))
    output_root.mkdir(parents=True, exist_ok=True)

    segments, counters = select_language_segments(
        raw_root,
        args.envs,
        allow_partial_scene_overlap=args.allow_partial_scene_overlap,
    )
    segments = split_segments(
        segments,
        val_fraction=args.val_fraction,
        seed=args.seed,
        max_segments=args.max_segments,
    )

    if args.dry_run:
        dry_run_report(
            segments,
            counters,
            output_root=output_root,
            repo_prefix=args.repo_prefix,
            env_selection=args.envs,
            chunk_size=args.chunk_size,
            tables_dir=args.tables_dir,
        )
        return

    dataset_root = create_dataset(
        raw_root=raw_root,
        output_root=output_root,
        repo_prefix=args.repo_prefix,
        env_selection=args.envs,
        segments=segments,
        overwrite=args.overwrite,
        use_videos=not args.no_videos,
        image_writer_threads=args.image_writer_threads,
        batch_encoding_size=args.batch_encoding_size,
        parallel_encoding=args.parallel_encoding,
        chunk_size=args.chunk_size,
        tables_dir=args.tables_dir,
        progress_log_every=args.progress_log_every,
    )

    repo_id = f"local/{args.repo_prefix}_{args.envs}"
    readback_rows = validate_dataset_readback(
        dataset_root=dataset_root,
        repo_id=repo_id,
        segments=segments,
        chunk_size=args.chunk_size,
        readback_episodes=args.readback_episodes,
        use_videos=not args.no_videos,
    )
    table_prefix = f"calvin_lang_{args.envs}"
    write_csv(args.tables_dir / f"{table_prefix}_readback_checks.csv", readback_rows)

    summary = summarize_segments(
        segments,
        counters,
        dataset_root=dataset_root,
        repo_id=repo_id,
        chunk_size=args.chunk_size,
    )
    summary["readback_checked_episode_positions"] = len(readback_rows)
    summary["readback_status"] = "passed"
    write_summary(args.tables_dir / f"{table_prefix}_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote language-aligned CALVIN dataset to {dataset_root}", flush=True)


if __name__ == "__main__":
    main()
