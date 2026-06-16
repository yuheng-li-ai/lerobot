#!/usr/bin/env python3
"""Shared utilities for Phase 0 CALVIN raw-data inspection and conversion."""

from __future__ import annotations

import bisect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

DEFAULT_CALVIN_RAW = Path("/SSD_DISK/users/zengzixuan/calvin/task_ABC_D")
DEFAULT_LEROBOT_ROOT = Path("/EXT_DISK/users/zengzixuan/processed-calvin")
DEFAULT_RUNS_ROOT = Path("/EXT_DISK/users/zengzixuan/calvin_runs")

SCENE_TO_ENV = {
    "calvin_scene_A": "A",
    "calvin_scene_B": "B",
    "calvin_scene_C": "C",
    "calvin_scene_D": "D",
}


@dataclass(frozen=True)
class FrameRange:
    env: str
    split: str
    start: int
    end: int
    scene: str

    @property
    def num_frames(self) -> int:
        return self.end - self.start + 1


def configured_raw_root(raw_root: str | Path | None = None) -> Path:
    return Path(raw_root or os.environ.get("CALVIN_RAW", DEFAULT_CALVIN_RAW)).expanduser()


def configured_lerobot_root(root: str | Path | None = None) -> Path:
    return Path(root or os.environ.get("CALVIN_LEROBOT_ROOT", DEFAULT_LEROBOT_ROOT)).expanduser()


def configured_runs_root(root: str | Path | None = None) -> Path:
    return Path(root or os.environ.get("CALVIN_RUNS", DEFAULT_RUNS_ROOT)).expanduser()


def split_dir(raw_root: Path, split: str) -> Path:
    path = raw_root / split
    if not path.is_dir():
        raise FileNotFoundError(f"Missing CALVIN split directory: {path}")
    return path


def frame_file(directory: Path, frame_id: int) -> Path:
    return directory / f"episode_{frame_id:07d}.npz"


def load_episode_ranges(directory: Path) -> list[tuple[int, int]]:
    ranges = np.load(directory / "ep_start_end_ids.npy")
    return [(int(start), int(end)) for start, end in ranges]


def load_scene_ranges(training_dir: Path) -> dict[str, FrameRange]:
    scene_path = training_dir / "scene_info.npy"
    if not scene_path.exists():
        raise FileNotFoundError(
            f"{scene_path} is required to split task_ABC_D training into A/B/C environments."
        )
    scene_info = np.load(scene_path, allow_pickle=True).item()
    ranges = {}
    for scene, bounds in scene_info.items():
        env = SCENE_TO_ENV.get(scene)
        if env is None:
            continue
        start, end = int(bounds[0]), int(bounds[1])
        ranges[env] = FrameRange(env=env, split="training", start=start, end=end, scene=scene)
    missing = {"A", "B", "C"} - set(ranges)
    if missing:
        raise ValueError(f"scene_info.npy did not define expected environments: {sorted(missing)}")
    return ranges


def load_environment_ranges(raw_root: Path) -> dict[str, FrameRange]:
    training = split_dir(raw_root, "training")
    validation = split_dir(raw_root, "validation")
    ranges = load_scene_ranges(training)

    val_ranges = load_episode_ranges(validation)
    if not val_ranges:
        raise ValueError(f"No validation ranges found in {validation}")
    ranges["D"] = FrameRange(
        env="D",
        split="validation",
        start=min(start for start, _ in val_ranges),
        end=max(end for _, end in val_ranges),
        scene="calvin_scene_D",
    )
    return ranges


def interval_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    return max(0, end - start + 1)


def clipped_episode_ranges(
    episode_ranges: Iterable[tuple[int, int]], frame_range: FrameRange
) -> list[tuple[int, int]]:
    clipped = []
    for ep_start, ep_end in episode_ranges:
        start = max(ep_start, frame_range.start)
        end = min(ep_end, frame_range.end)
        if start <= end:
            clipped.append((start, end))
    return clipped


def load_language_intervals(split_path: Path) -> list[tuple[int, int, str]]:
    ann_path = split_path / "lang_annotations" / "auto_lang_ann.npy"
    if not ann_path.exists():
        return []
    data = np.load(ann_path, allow_pickle=True).item()
    intervals = data["info"]["indx"]
    tasks = data["language"]["task"]
    return [(int(start), int(end), str(task)) for (start, end), task in zip(intervals, tasks, strict=True)]


class LanguageTaskLookup:
    def __init__(self, intervals: list[tuple[int, int, str]], fallback_task: str):
        self._starts = [start for start, _, _ in intervals]
        self._intervals = intervals
        self._fallback_task = fallback_task

    def task_for_frame(self, frame_id: int) -> str:
        idx = bisect.bisect_right(self._starts, frame_id) - 1
        if idx >= 0:
            start, end, task = self._intervals[idx]
            if start <= frame_id <= end:
                return task
        return self._fallback_task


def count_language_tasks(
    intervals: Iterable[tuple[int, int, str]], frame_range: FrameRange
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for start, end, task in intervals:
        if interval_overlap(start, end, frame_range.start, frame_range.end) > 0:
            counts[task] = counts.get(task, 0) + 1
    return counts


def iter_segmented_ranges(
    ranges: Iterable[tuple[int, int]], segment_frames: int | None
) -> Iterable[tuple[int, int]]:
    if segment_frames is None or segment_frames <= 0:
        yield from ranges
        return
    for start, end in ranges:
        cursor = start
        while cursor <= end:
            seg_end = min(end, cursor + segment_frames - 1)
            yield cursor, seg_end
            cursor = seg_end + 1


def parse_env_selection(selection: str) -> list[str]:
    if selection == "ABC":
        return ["A", "B", "C"]
    return [selection]
