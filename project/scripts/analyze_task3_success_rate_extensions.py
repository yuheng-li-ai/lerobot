#!/usr/bin/env python
"""Extended visual analysis for ACT-Lang CALVIN D success-rate results."""

from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = ["act_lang_B_100k", "act_lang_ABC_size_matched_100k", "act_lang_ABC_200k"]
MODEL_LABELS = {
    "act_lang_B_100k": "B 100k",
    "act_lang_ABC_size_matched_100k": "ABC-SM 100k",
    "act_lang_ABC_200k": "ABC 200k",
}
MODEL_COLORS = {
    "act_lang_B_100k": "#4c78a8",
    "act_lang_ABC_size_matched_100k": "#8f6ab8",
    "act_lang_ABC_200k": "#b23a48",
}
TASK_GROUP_ORDER = ["drawer", "slider", "light", "lift", "place", "push", "rotate", "stack"]


def write_latex_table(path: Path, rows: list[dict[str, Any]], columns: list[str], caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\toprule",
        " & ".join(columns).replace("_", "\\_") + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value).replace("_", "\\_"))
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines))


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def task_group(task: str) -> str:
    if "drawer" in task and not task.startswith("lift"):
        return "drawer"
    if "slider" in task and not task.startswith("lift") and not task.startswith("place"):
        return "slider"
    if "lightbulb" in task or "led" in task:
        return "light"
    if task.startswith("lift"):
        return "lift"
    if task.startswith("place"):
        return "place"
    if task.startswith("push"):
        return "push"
    if task.startswith("rotate"):
        return "rotate"
    if "stack" in task:
        return "stack"
    return "other"


def format_pct(value: float) -> str:
    return f"{100 * value:.1f}"


def load_inputs(table_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(table_dir / "success_rate_D_summary.csv")
    sequences = pd.read_csv(table_dir / "success_rate_D_sequences.csv")
    subtasks = pd.read_csv(table_dir / "success_rate_D_subtasks.csv")
    tasks = pd.read_csv(table_dir / "success_rate_D_task_breakdown.csv")
    for df in [summary, sequences, subtasks, tasks]:
        df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    return summary.sort_values("model"), sequences, subtasks, tasks


def plot_prefix_distribution(sequences: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    counts = (
        sequences.groupby(["model", "successful_subtasks"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    rows = []
    for model in MODEL_ORDER:
        group = counts[counts["model"] == model]
        total = group["count"].sum()
        for k in range(6):
            count = int(group[group["successful_subtasks"] == k]["count"].sum())
            rows.append(
                {
                    "model": model,
                    "display_name": MODEL_LABELS[model],
                    "successful_subtasks": k,
                    "count": count,
                    "fraction": count / total if total else math.nan,
                }
            )
    dist = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    bottom = np.zeros(len(MODEL_ORDER))
    x = np.arange(len(MODEL_ORDER))
    colors = ["#d7e3ef", "#a8c6df", "#75a5c8", "#447da8", "#24577d", "#10364f"]
    for k in range(6):
        vals = [
            float(dist[(dist["model"] == model) & (dist["successful_subtasks"] == k)]["fraction"].iloc[0]) * 100
            for model in MODEL_ORDER
        ]
        ax.bar(x, vals, bottom=bottom, color=colors[k], label=f"{k} solved")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_ylabel("Sequences (%)")
    ax.set_title("Distribution of Successful Subtasks per 5-Step Sequence")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save_fig(fig, fig_dir / "successful_prefix_distribution.png")
    return dist


def plot_attrition(summary: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    x = np.arange(1, 6)
    for _, row in summary.iterrows():
        model = str(row["model"])
        y = [float(row[f"sr_chain_{i}"]) * 100 for i in x]
        ax.plot(x, y, marker="o", linewidth=2.4, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        for xi, yi in zip(x, y, strict=True):
            ax.text(xi, yi + 1.3, f"{yi:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}/5" for i in x])
    ax.set_ylabel("Success rate (%)")
    ax.set_xlabel("Required consecutive subtasks")
    ax.set_title("Long-Horizon Attrition on CALVIN D")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_fig(fig, fig_dir / "long_horizon_attrition_curve.png")


def plot_task_heatmap(tasks: pd.DataFrame, fig_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks = tasks.copy()
    tasks["task_group"] = tasks["task"].map(task_group)
    pivot = tasks.pivot_table(index="task", columns="model", values="success_rate", observed=True)
    pivot = pivot.reindex(columns=MODEL_ORDER)
    pivot["improvement_ABC200k_vs_B"] = pivot["act_lang_ABC_200k"] - pivot["act_lang_B_100k"]
    pivot["task_group"] = [task_group(t) for t in pivot.index]
    pivot = pivot.sort_values(["task_group", "improvement_ABC200k_vs_B"], ascending=[True, False])

    fig_height = max(8.0, 0.28 * len(pivot))
    fig, ax = plt.subplots(figsize=(8.8, fig_height))
    heat = pivot[MODEL_ORDER].to_numpy(dtype=float) * 100
    im = ax.imshow(heat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([textwrap.fill(t, 24) for t in pivot.index], fontsize=7)
    ax.set_title("Per-Task Success Rate on CALVIN D")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat[i, j]:.0f}", ha="center", va="center", fontsize=6, color="#111111")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Success rate (%)")
    save_fig(fig, fig_dir / "per_task_success_rate_heatmap.png")

    group_rows = []
    for (model, group), df in tasks.groupby(["model", "task_group"], observed=True):
        group_rows.append(
            {
                "model": str(model),
                "display_name": MODEL_LABELS[str(model)],
                "task_group": group,
                "success": int(df["success"].sum()),
                "total": int(df["total"].sum()),
                "success_rate": float(df["success"].sum() / max(df["total"].sum(), 1)),
            }
        )
    group_df = pd.DataFrame(group_rows)
    group_pivot = group_df.pivot_table(index="task_group", columns="model", values="success_rate", observed=True)
    group_pivot = group_pivot.reindex(TASK_GROUP_ORDER).dropna(how="all")

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    x = np.arange(len(group_pivot.index))
    width = 0.24
    for i, model in enumerate(MODEL_ORDER):
        vals = group_pivot[model].to_numpy(dtype=float) * 100
        ax.bar(x + (i - 1) * width, vals, width=width, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    ax.set_xticks(x)
    ax.set_xticklabels(group_pivot.index, rotation=0)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Success Rate by Task Group")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    save_fig(fig, fig_dir / "task_group_success_rate.png")
    return pivot.reset_index(), group_df


def plot_first_failure(sequences: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    rows = []
    for _, row in sequences.iterrows():
        solved = int(row["successful_subtasks"])
        first_failed_position = solved + 1 if solved < 5 else 0
        sequence = str(row["sequence"]).split(" -> ")
        failed_task = "none_all_solved" if solved == 5 else sequence[solved]
        rows.append(
            {
                "model": str(row["model"]),
                "display_name": MODEL_LABELS[str(row["model"])],
                "first_failed_position": first_failed_position,
                "failed_task": failed_task,
                "failed_task_group": "none" if solved == 5 else task_group(failed_task),
            }
        )
    fail_df = pd.DataFrame(rows)

    pos_counts = (
        fail_df.groupby(["model", "first_failed_position"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    pos_rows = []
    for model in MODEL_ORDER:
        total = int((fail_df["model"] == model).sum())
        for pos in range(0, 6):
            count = int(pos_counts[(pos_counts["model"] == model) & (pos_counts["first_failed_position"] == pos)]["count"].sum())
            pos_rows.append(
                {
                    "model": model,
                    "display_name": MODEL_LABELS[model],
                    "first_failed_position": pos,
                    "count": count,
                    "fraction": count / total if total else math.nan,
                }
            )
    pos_df = pd.DataFrame(pos_rows)

    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    x = np.arange(6)
    width = 0.24
    labels = ["all solved", "fail @1", "fail @2", "fail @3", "fail @4", "fail @5"]
    for i, model in enumerate(MODEL_ORDER):
        vals = pos_df[pos_df["model"] == model].sort_values("first_failed_position")["fraction"].to_numpy() * 100
        ax.bar(x + (i - 1) * width, vals, width=width, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Sequences (%)")
    ax.set_title("Where Long-Horizon Rollouts First Fail")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    save_fig(fig, fig_dir / "first_failure_position_distribution.png")

    group = (
        fail_df[fail_df["failed_task_group"] != "none"]
        .groupby(["model", "failed_task_group"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    groups = [g for g in TASK_GROUP_ORDER if g in set(group["failed_task_group"])]
    x = np.arange(len(groups))
    width = 0.24
    for i, model in enumerate(MODEL_ORDER):
        vals = []
        total_fail = int((fail_df["model"].eq(model) & fail_df["failed_task_group"].ne("none")).sum())
        for g in groups:
            count = int(group[(group["model"] == model) & (group["failed_task_group"] == g)]["count"].sum())
            vals.append(100 * count / max(total_fail, 1))
        ax.bar(x + (i - 1) * width, vals, width=width, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Failed sequences (%)")
    ax.set_title("First-Failure Task Group")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    save_fig(fig, fig_dir / "first_failure_task_group.png")
    return fail_df


def plot_step_efficiency(subtasks: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    success = subtasks[subtasks["success"]].copy()
    rows = []
    for model, df in success.groupby("model", observed=True):
        rows.append(
            {
                "model": str(model),
                "display_name": MODEL_LABELS[str(model)],
                "successful_subtasks": int(len(df)),
                "median_steps_to_success": float(df["steps"].median()),
                "mean_steps_to_success": float(df["steps"].mean()),
                "q90_steps_to_success": float(df["steps"].quantile(0.90)),
            }
        )
    step_summary = pd.DataFrame(rows).sort_values("model")

    bins = np.arange(0, 361, 30)
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for model in MODEL_ORDER:
        vals = success[success["model"].astype(str) == model]["steps"].to_numpy()
        ax.hist(vals, bins=bins, histtype="step", linewidth=2.3, density=True, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    ax.set_xlabel("Steps to success")
    ax.set_ylabel("Density among successful subtasks")
    ax.set_title("How Quickly Successful Subtasks Are Solved")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    save_fig(fig, fig_dir / "steps_to_success_distribution.png")
    return step_summary


def plot_chunk_success_relation(subtasks: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    metrics = [
        ("mean_action_delta_l2_first6", "Mean action delta"),
        ("mean_chunk_boundary_jump_l2_first6", "Chunk boundary jump"),
        ("mean_action_norm_l2_first6", "Action norm"),
    ]
    rows = []
    for model, df in subtasks.groupby("model", observed=True):
        for success, group in df.groupby("success", observed=True):
            row = {
                "model": str(model),
                "display_name": MODEL_LABELS[str(model)],
                "outcome": "success" if bool(success) else "failure",
                "count": int(len(group)),
            }
            for metric, _ in metrics:
                row[metric] = float(group[metric].mean())
            rows.append(row)
    relation = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6))
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        x = np.arange(len(MODEL_ORDER))
        width = 0.34
        for idx, outcome in enumerate(["success", "failure"]):
            vals = []
            for model in MODEL_ORDER:
                match = relation[(relation["model"] == model) & (relation["outcome"] == outcome)]
                vals.append(float(match[metric].iloc[0]) if len(match) else math.nan)
            ax.bar(
                x + (idx - 0.5) * width,
                vals,
                width=width,
                color="#2f6f9f" if outcome == "success" else "#b23a48",
                label=outcome.title(),
            )
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean value")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Rollout Chunk Diagnostics by Outcome")
    save_fig(fig, fig_dir / "chunk_metrics_success_vs_failure.png")
    return relation


def build_improvement_tables(summary: pd.DataFrame, tasks: pd.DataFrame, tab_dir: Path) -> pd.DataFrame:
    base = summary.set_index("model")
    rows = []
    for target in ["act_lang_ABC_size_matched_100k", "act_lang_ABC_200k"]:
        rows.append(
            {
                "comparison": f"{MODEL_LABELS[target]} vs {MODEL_LABELS['act_lang_B_100k']}",
                "delta_avg_len": float(base.loc[target, "avg_successful_sequence_length"] - base.loc["act_lang_B_100k", "avg_successful_sequence_length"]),
                "delta_1_of_5_sr": float(base.loc[target, "sr_chain_1"] - base.loc["act_lang_B_100k", "sr_chain_1"]),
                "delta_5_of_5_sr": float(base.loc[target, "sr_chain_5"] - base.loc["act_lang_B_100k", "sr_chain_5"]),
                "delta_subtask_sr": float(base.loc[target, "single_subtask_success_rate"] - base.loc["act_lang_B_100k", "single_subtask_success_rate"]),
            }
        )
    rows.append(
        {
            "comparison": "ABC 200k vs ABC-SM 100k",
            "delta_avg_len": float(base.loc["act_lang_ABC_200k", "avg_successful_sequence_length"] - base.loc["act_lang_ABC_size_matched_100k", "avg_successful_sequence_length"]),
            "delta_1_of_5_sr": float(base.loc["act_lang_ABC_200k", "sr_chain_1"] - base.loc["act_lang_ABC_size_matched_100k", "sr_chain_1"]),
            "delta_5_of_5_sr": float(base.loc["act_lang_ABC_200k", "sr_chain_5"] - base.loc["act_lang_ABC_size_matched_100k", "sr_chain_5"]),
            "delta_subtask_sr": float(base.loc["act_lang_ABC_200k", "single_subtask_success_rate"] - base.loc["act_lang_ABC_size_matched_100k", "single_subtask_success_rate"]),
        }
    )
    improvement = pd.DataFrame(rows)
    improvement.to_csv(tab_dir / "success_rate_improvements.csv", index=False)
    write_latex_table(
        tab_dir / "success_rate_improvements.tex",
        improvement.to_dict("records"),
        ["comparison", "delta_avg_len", "delta_1_of_5_sr", "delta_5_of_5_sr", "delta_subtask_sr"],
        "Absolute success-rate improvements on CALVIN D.",
        "tab:task3-success-rate-improvements",
    )

    pivot = tasks.pivot_table(index="task", columns="model", values="success_rate", observed=True)
    pivot = pivot.reindex(columns=MODEL_ORDER)
    pivot["delta_ABC200k_vs_B"] = pivot["act_lang_ABC_200k"] - pivot["act_lang_B_100k"]
    pivot["delta_ABC_SM_vs_B"] = pivot["act_lang_ABC_size_matched_100k"] - pivot["act_lang_B_100k"]
    pivot["task_group"] = [task_group(t) for t in pivot.index]
    top = pivot.sort_values("delta_ABC200k_vs_B", ascending=False).head(12).reset_index()
    top.to_csv(tab_dir / "top_task_improvements_ABC200k_vs_B.csv", index=False)
    write_latex_table(
        tab_dir / "top_task_improvements_ABC200k_vs_B.tex",
        [
            {
                "task": r["task"],
                "task_group": r["task_group"],
                "B_SR": float(r["act_lang_B_100k"]),
                "ABC200k_SR": float(r["act_lang_ABC_200k"]),
                "delta": float(r["delta_ABC200k_vs_B"]),
            }
            for r in top.to_dict("records")
        ],
        ["task", "task_group", "B_SR", "ABC200k_SR", "delta"],
        "Largest per-task success-rate gains from ACT-Lang-B to ACT-Lang-ABC 200k.",
        "tab:task3-top-task-improvements",
    )
    return improvement


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", type=Path, default=Path("project/tables/task3_success_rate_D"))
    parser.add_argument("--figure-dir", type=Path, default=Path("project/figures/task3_success_rate_D_extended"))
    parser.add_argument("--extended-table-dir", type=Path, default=Path("project/tables/task3_success_rate_D_extended"))
    args = parser.parse_args()

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.extended_table_dir.mkdir(parents=True, exist_ok=True)

    summary, sequences, subtasks, tasks = load_inputs(args.table_dir)
    prefix = plot_prefix_distribution(sequences, args.figure_dir)
    plot_attrition(summary, args.figure_dir)
    task_heatmap, group_success = plot_task_heatmap(tasks, args.figure_dir)
    first_fail = plot_first_failure(sequences, args.figure_dir)
    step_summary = plot_step_efficiency(subtasks, args.figure_dir)
    chunk_relation = plot_chunk_success_relation(subtasks, args.figure_dir)
    improvement = build_improvement_tables(summary, tasks, args.extended_table_dir)

    prefix.to_csv(args.extended_table_dir / "successful_prefix_distribution.csv", index=False)
    task_heatmap.to_csv(args.extended_table_dir / "per_task_success_rate_matrix.csv", index=False)
    group_success.to_csv(args.extended_table_dir / "task_group_success_rate.csv", index=False)
    first_fail.to_csv(args.extended_table_dir / "first_failure_taxonomy.csv", index=False)
    step_summary.to_csv(args.extended_table_dir / "steps_to_success_summary.csv", index=False)
    chunk_relation.to_csv(args.extended_table_dir / "chunk_metrics_success_vs_failure.csv", index=False)

    write_latex_table(
        args.extended_table_dir / "steps_to_success_summary.tex",
        step_summary.to_dict("records"),
        ["display_name", "successful_subtasks", "median_steps_to_success", "mean_steps_to_success", "q90_steps_to_success"],
        "Step efficiency for successful CALVIN D subtasks.",
        "tab:task3-steps-to-success",
    )
    write_latex_table(
        args.extended_table_dir / "task_group_success_rate.tex",
        group_success.to_dict("records"),
        ["display_name", "task_group", "success", "total", "success_rate"],
        "CALVIN D success rate aggregated by task group.",
        "tab:task3-task-group-success",
    )

    manifest = pd.DataFrame(
        [
            *[{"artifact_type": "figure", "path": str(p)} for p in sorted(args.figure_dir.glob("*.png"))],
            *[{"artifact_type": "table", "path": str(p)} for p in sorted(args.extended_table_dir.glob("*"))],
        ]
    )
    manifest.to_csv(args.extended_table_dir / "manifest.csv", index=False)
    print(
        {
            "figures": str(args.figure_dir),
            "tables": str(args.extended_table_dir),
            "num_figures": int(len(list(args.figure_dir.glob("*.png")))),
            "num_tables": int(len(list(args.extended_table_dir.glob("*")))),
            "improvement_rows": int(len(improvement)),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
