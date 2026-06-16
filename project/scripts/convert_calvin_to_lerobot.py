#!/usr/bin/env python3
"""Convert raw CALVIN task_ABC_D data into LeRobot datasets.

The converter is intentionally CLI-harnessed for manual execution. It can make
small smoke-test datasets with --max-episodes/--max-frames-per-episode and can
perform full conversion when those limits are omitted.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# LeRobot 0.5.2 errors if deprecated LEROBOT_HOME is set by calvin_env.sh.
os.environ.pop("LEROBOT_HOME", None)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from calvin_phase0_common import (  # noqa: E402
    FrameRange,
    LanguageTaskLookup,
    clipped_episode_ranges,
    configured_lerobot_root,
    configured_raw_root,
    frame_file,
    iter_segmented_ranges,
    load_environment_ranges,
    load_episode_ranges,
    load_language_intervals,
    parse_env_selection,
    split_dir,
)

FPS = 30

CALVIN_FEATURES = {
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
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
    },
}


def selected_ranges(raw_root: Path, selection: str) -> list[FrameRange]:
    env_ranges = load_environment_ranges(raw_root)
    return [env_ranges[env] for env in parse_env_selection(selection)]


def collect_source_ranges(
    raw_root: Path,
    ranges: list[FrameRange],
    segment_frames: int | None,
    max_episodes: int | None,
    max_frames_per_episode: int | None,
) -> list[tuple[Path, FrameRange, int, int]]:
    collected: list[tuple[Path, FrameRange, int, int]] = []
    for frame_range in ranges:
        directory = split_dir(raw_root, frame_range.split)
        episode_ranges = clipped_episode_ranges(load_episode_ranges(directory), frame_range)
        for start, end in iter_segmented_ranges(episode_ranges, segment_frames):
            if max_frames_per_episode is not None:
                end = min(end, start + max_frames_per_episode - 1)
            if start <= end:
                collected.append((directory, frame_range, start, end))
            if max_episodes is not None and len(collected) >= max_episodes:
                return collected
    return collected


def make_task_lookup(split_path: Path, frame_range: FrameRange, task_source: str) -> LanguageTaskLookup:
    fallback = f"calvin_env_{frame_range.env}_play"
    if task_source == "environment":
        return LanguageTaskLookup([], fallback)
    intervals = load_language_intervals(split_path)
    return LanguageTaskLookup(intervals, fallback)


def convert_one_dataset(
    *,
    raw_root: Path,
    output_root: Path,
    selection: str,
    repo_prefix: str,
    overwrite: bool,
    use_videos: bool,
    image_writer_threads: int,
    batch_encoding_size: int,
    segment_frames: int | None,
    max_episodes: int | None,
    max_frames_per_episode: int | None,
    task_source: str,
) -> Path:
    repo_name = f"{repo_prefix}_{selection}"
    repo_id = f"local/{repo_name}"
    dataset_root = output_root / repo_name

    if dataset_root.exists():
        if not overwrite:
            raise FileExistsError(f"{dataset_root} exists. Use --overwrite to replace it.")
        shutil.rmtree(dataset_root)

    ranges = selected_ranges(raw_root, selection)
    source_ranges = collect_source_ranges(
        raw_root=raw_root,
        ranges=ranges,
        segment_frames=segment_frames,
        max_episodes=max_episodes,
        max_frames_per_episode=max_frames_per_episode,
    )
    if not source_ranges:
        raise RuntimeError(f"No source ranges selected for {selection}")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=dataset_root,
        fps=FPS,
        robot_type="franka_panda_calvin",
        features=CALVIN_FEATURES,
        use_videos=use_videos,
        image_writer_threads=image_writer_threads,
        batch_encoding_size=batch_encoding_size,
    )

    task_lookups: dict[tuple[str, str], LanguageTaskLookup] = {}
    episode_iter = tqdm(
        enumerate(source_ranges, start=1),
        total=len(source_ranges),
        desc=f"{selection} episodes",
        unit="episode",
        dynamic_ncols=True,
        mininterval=2.0,
        file=sys.stdout,
    )
    for episode_idx, (directory, frame_range, start, end) in episode_iter:
        key = (str(directory), frame_range.env)
        if key not in task_lookups:
            task_lookups[key] = make_task_lookup(directory, frame_range, task_source)
        lookup = task_lookups[key]
        episode_iter.set_postfix(env=frame_range.env, start=start, end=end)
        print(
            f"[{selection}] episode {episode_idx}/{len(source_ranges)} "
            f"{frame_range.env} frames {start}-{end}",
            flush=True,
        )
        frame_iter = tqdm(
            range(start, end + 1),
            total=end - start + 1,
            desc=f"{selection} {frame_range.env} ep {episode_idx} frames",
            unit="frame",
            dynamic_ncols=True,
            mininterval=2.0,
            leave=False,
            file=sys.stdout,
        )
        for frame_id in frame_iter:
            data = np.load(frame_file(directory, frame_id))
            dataset.add_frame(
                {
                    "observation.images.static": data["rgb_static"],
                    "observation.images.gripper": data["rgb_gripper"],
                    "observation.state": data["robot_obs"].astype(np.float32),
                    "action": data["rel_actions"].astype(np.float32),
                    "task": lookup.task_for_frame(frame_id),
                }
            )
        dataset.save_episode()
        print(f"[{selection}] saved episode {episode_idx}/{len(source_ranges)}", flush=True)

    dataset.finalize()
    print(f"Wrote {selection} LeRobot dataset to {dataset_root}", flush=True)
    return dataset_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", default=["B", "ABC", "D"], choices=["A", "B", "C", "D", "ABC"])
    parser.add_argument("--repo-prefix", default="calvin")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument("--batch-encoding-size", type=int, default=1)
    parser.add_argument("--segment-frames", type=int, default=3000)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-frames-per-episode", type=int, default=None)
    parser.add_argument("--task-source", choices=["environment", "language"], default="environment")
    args = parser.parse_args()

    raw_root = configured_raw_root(args.raw_root)
    output_root = configured_lerobot_root(args.output_root)
    os.environ.setdefault("HF_LEROBOT_HOME", str(output_root))
    output_root.mkdir(parents=True, exist_ok=True)

    for selection in args.datasets:
        convert_one_dataset(
            raw_root=raw_root,
            output_root=output_root,
            selection=selection,
            repo_prefix=args.repo_prefix,
            overwrite=args.overwrite,
            use_videos=not args.no_videos,
            image_writer_threads=args.image_writer_threads,
            batch_encoding_size=args.batch_encoding_size,
            segment_frames=args.segment_frames,
            max_episodes=args.max_episodes,
            max_frames_per_episode=args.max_frames_per_episode,
            task_source=args.task_source,
        )


if __name__ == "__main__":
    main()
