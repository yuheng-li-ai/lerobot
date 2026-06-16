#!/usr/bin/env python
"""Retrospectively log completed ACT training curves to Weights & Biases.

This script logs only the core requirement curves as native W&B time-series:

- train/loss
- train/action_l1
- val/loss
- val/action_l1

It reads already saved `metrics.csv` files, so no training is launched.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Any

import wandb


DEFAULT_SUMMARY = Path("project/tables/full_training_summary_with_B_aug.csv")
DEFAULT_OUTPUT = Path("project/tables/wandb_training_curve_runs.csv")
DEFAULT_PROJECT = "calvin-act-generalization"
DEFAULT_GROUP = "task1_task2_training"


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


def finite_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def should_log_val(row: dict[str, str], previous: tuple[float | None, float | None]) -> bool:
    val_loss = finite_float(row.get("val_loss"))
    val_l1 = finite_float(row.get("val_action_l1"))
    if val_loss is None and val_l1 is None:
        return False
    return (val_loss, val_l1) != previous


def log_one_run(
    run_row: dict[str, str],
    *,
    project: str,
    entity: str | None,
    group: str,
    job_type: str,
    tags: list[str],
    sample_every: int,
    dry_run: bool,
) -> dict[str, Any]:
    experiment = run_row["experiment"]
    run_dir = Path(run_row["run_dir"])
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)

    metrics_rows = read_rows(metrics_path)
    expected_rows = int(run_row.get("metrics_rows") or len(metrics_rows))
    if len(metrics_rows) != expected_rows:
        print(f"WARNING {experiment}: expected {expected_rows} rows, found {len(metrics_rows)}")

    wandb_run = None
    logged_train = 0
    logged_val = 0
    last_val: tuple[float | None, float | None] = (None, None)
    if dry_run:
        run_url = ""
        run_id = ""
    else:
        wandb_run = wandb.init(
            project=project,
            entity=entity,
            name=f"{experiment}_training_curves",
            group=group,
            job_type=job_type,
            tags=tags + [experiment, run_row.get("condition", ""), run_row.get("phase", "")],
            config={
                "experiment": experiment,
                "condition": run_row.get("condition"),
                "phase": run_row.get("phase"),
                "source_run_dir": str(run_dir),
                "source_metrics_csv": str(metrics_path),
                "retrospective_logging": True,
                "sample_every": sample_every,
                "final_step": int(run_row["final_step"]),
                "best_val_step": int(run_row["best_val_step"]),
                "best_val_action_l1": float(run_row["best_val_action_l1"]),
            },
            reinit=True,
        )
        wandb.define_metric("step")
        wandb.define_metric("train/*", step_metric="step")
        wandb.define_metric("val/*", step_metric="step")
        run_url = wandb_run.url
        run_id = wandb_run.id

    for row in metrics_rows:
        step = int(row["step"])
        train_payload: dict[str, float | int] = {"step": step}
        train_loss = finite_float(row.get("train_loss"))
        train_l1 = finite_float(row.get("train_action_l1"))
        if train_loss is not None:
            train_payload["train/loss"] = train_loss
        if train_l1 is not None:
            train_payload["train/action_l1"] = train_l1
        if len(train_payload) > 1 and step % sample_every == 0:
            logged_train += 1
            if not dry_run:
                wandb.log(train_payload, step=step)

        if should_log_val(row, last_val):
            val_loss = finite_float(row.get("val_loss"))
            val_l1 = finite_float(row.get("val_action_l1"))
            val_payload: dict[str, float | int] = {"step": step}
            if val_loss is not None:
                val_payload["val/loss"] = val_loss
            if val_l1 is not None:
                val_payload["val/action_l1"] = val_l1
            last_val = (val_loss, val_l1)
            logged_val += 1
            if not dry_run:
                wandb.log(val_payload, step=step)

    if not dry_run and wandb_run is not None:
        # Small source artifact for provenance.
        artifact = wandb.Artifact(f"{experiment}_metrics_csv", type="training-metrics")
        artifact.add_file(str(metrics_path))
        wandb_run.log_artifact(artifact)
        wandb_run.finish()

    return {
        "experiment": experiment,
        "condition": run_row.get("condition", ""),
        "phase": run_row.get("phase", ""),
        "wandb_project": project,
        "wandb_group": group,
        "wandb_run_name": f"{experiment}_training_curves",
        "wandb_run_id": run_id,
        "wandb_url": run_url,
        "source_metrics_csv": str(metrics_path),
        "source_run_dir": str(run_dir),
        "source_rows": len(metrics_rows),
        "logged_train_points": logged_train,
        "logged_val_points": logged_val,
        "sample_every": sample_every,
        "dry_run": dry_run,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--group", default=DEFAULT_GROUP)
    parser.add_argument("--job-type", default="retrospective_training_curves")
    parser.add_argument("--sample-every", type=int, default=1, help="Log every Nth train step. Validation updates are always logged.")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.sample_every <= 0:
        raise ValueError("--sample-every must be positive")

    rows = read_rows(args.summary)
    if args.models:
        wanted = set(args.models)
        rows = [row for row in rows if row["experiment"] in wanted]
        missing = wanted - {row["experiment"] for row in rows}
        if missing:
            raise KeyError(f"Unknown models in --models: {sorted(missing)}")

    tags = ["calvin", "act", "training-curves", "retrospective"]
    manifest = []
    print(f"Logging {len(rows)} runs to W&B project={args.project!r} group={args.group!r}")
    for row in rows:
        print(f"RUN {row['experiment']} metrics={Path(row['run_dir']) / 'metrics.csv'}")
        manifest.append(
            log_one_run(
                row,
                project=args.project,
                entity=args.entity,
                group=args.group,
                job_type=args.job_type,
                tags=tags,
                sample_every=args.sample_every,
                dry_run=args.dry_run,
            )
        )

    write_rows(args.output, manifest)
    print(f"Wrote manifest: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
