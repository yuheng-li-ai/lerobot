#!/usr/bin/env python
"""Download logged W&B training curves and export local figures."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np
import wandb


DEFAULT_MANIFEST = Path("project/tables/wandb_training_curve_runs.csv")
DEFAULT_FIGURE_DIR = Path("project/figures/wandb_export")
DEFAULT_TABLE_DIR = Path("project/tables/wandb_export")
METRICS = [
    ("train/loss", "Train Loss"),
    ("train/action_l1", "Train Action L1"),
    ("val/loss", "Validation Loss"),
    ("val/action_l1", "Validation Action L1"),
]


DISPLAY_NAMES = {
    "act_B": "ACT-B",
    "act_B_aug": "ACT-B Aug",
    "act_ABC": "ACT-ABC",
    "act_ABC_aug": "ACT-ABC Aug",
    "act_ABC_size_matched": "ACT-ABC Size-Matched",
    "act_ABC_size_matched_aug": "ACT-ABC Size-Matched Aug",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def run_path_from_url(url: str) -> str:
    marker = "wandb.ai/"
    if marker not in url:
        raise ValueError(f"Unexpected W&B URL: {url}")
    suffix = url.split(marker, 1)[1]
    parts = suffix.split("/")
    if len(parts) < 4 or parts[2] != "runs":
        raise ValueError(f"Unexpected W&B run URL: {url}")
    entity, project, _, run_id = parts[:4]
    return f"{entity}/{project}/{run_id}"


def download_history(api: wandb.Api, row: dict[str, str]) -> list[dict[str, Any]]:
    run_path = run_path_from_url(row["wandb_url"])
    run = api.run(run_path)
    merged: dict[int, dict[str, Any]] = {}
    # W&B scan_history returns the intersection when multiple sparse metrics
    # are requested together. Fetch each metric independently so dense train
    # curves and sparse validation curves are both preserved.
    for metric, _ in METRICS:
        for item in run.scan_history(keys=["step", metric], page_size=10000):
            step_value = item.get("step", item.get("_step"))
            value = item.get(metric)
            if step_value is None or value is None:
                continue
            step = int(step_value)
            merged.setdefault(step, {"step": step})
            merged[step][metric] = float(value)
    rows = []
    for step in sorted(merged):
        out = {"step": step}
        for metric, _ in METRICS:
            out[metric] = merged[step].get(metric, "")
        rows.append(out)
    if not rows:
        # Fallback for W&B versions where explicit `step` is not available.
        for metric, _ in METRICS:
            for item in run.scan_history(keys=[metric], page_size=10000):
                step_value = item.get("_step")
                value = item.get(metric)
                if step_value is None or value is None:
                    continue
                step = int(step_value)
                merged.setdefault(step, {"step": step})
                merged[step][metric] = float(value)
        for step in sorted(merged):
            out = {"step": step}
            for metric, _ in METRICS:
                out[metric] = merged[step].get(metric, "")
            rows.append(out)
    if not rows:
        raise RuntimeError(f"No W&B history rows downloaded for {row['experiment']}")
    return rows


def metric_series(rows: list[dict[str, Any]], metric: str) -> tuple[np.ndarray, np.ndarray]:
    steps = []
    values = []
    for row in rows:
        value = row.get(metric)
        if value == "" or value is None:
            continue
        steps.append(int(row["step"]))
        values.append(float(value))
    return np.asarray(steps), np.asarray(values)


def plot_single_run(experiment: str, history: list[dict[str, Any]], output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2))
    axes = axes.ravel()
    for ax, (metric, title) in zip(axes, METRICS, strict=True):
        steps, values = metric_series(history, metric)
        ax.plot(steps, values, linewidth=1.6)
        ax.set_title(title, pad=10)
        ax.set_xlabel("Step")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
    fig.suptitle(f"W&B Exported Training Curves: {DISPLAY_NAMES.get(experiment, experiment)}", y=1.02)
    path = output_dir / "per_model" / f"{experiment}_wandb_training_curves.png"
    save(fig, path)
    return path


def plot_metric_comparison(all_history: dict[str, list[dict[str, Any]]], metric: str, title: str, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12.8, 6.2))
    for experiment, history in all_history.items():
        steps, values = metric_series(history, metric)
        ax.plot(steps, values, linewidth=1.5, label=DISPLAY_NAMES.get(experiment, experiment))
    ax.set_title(f"W&B Exported {title} Comparison", pad=14)
    ax.set_xlabel("Step")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    safe_name = metric.replace("/", "_")
    path = output_dir / "comparisons" / f"{safe_name}_comparison_wandb.png"
    save(fig, path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    args = parser.parse_args()

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    manifest = read_rows(args.manifest)
    all_history: dict[str, list[dict[str, Any]]] = {}
    artifact_rows = []
    for row in manifest:
        experiment = row["experiment"]
        print(f"Downloading W&B history for {experiment}: {row['wandb_url']}", flush=True)
        history = download_history(api, row)
        all_history[experiment] = history
        history_path = args.table_dir / f"{experiment}_wandb_history.csv"
        write_rows(history_path, history)
        figure_path = plot_single_run(experiment, history, args.figure_dir)
        artifact_rows.append(
            {
                "category": "per_model",
                "experiment": experiment,
                "metric": "all_core_curves",
                "figure": str(figure_path),
                "source_wandb_url": row["wandb_url"],
                "downloaded_history_csv": str(history_path),
                "history_rows": len(history),
            }
        )

    for metric, title in METRICS:
        figure_path = plot_metric_comparison(all_history, metric, title, args.figure_dir)
        artifact_rows.append(
            {
                "category": "comparison",
                "experiment": "all",
                "metric": metric,
                "figure": str(figure_path),
                "source_wandb_url": "see project/tables/wandb_training_curve_runs.csv",
                "downloaded_history_csv": str(args.table_dir),
                "history_rows": sum(len(history) for history in all_history.values()),
            }
        )

    write_rows(args.table_dir / "wandb_export_manifest.csv", artifact_rows)
    print(f"Wrote figures to {args.figure_dir}")
    print(f"Wrote tables to {args.table_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
