# LeRobot ACT CALVIN Experiments

This repository contains the task-specific experiment workspace for the embodied AI part of the computer vision final project.
It is not a mirror of the upstream LeRobot library.

## Contents

- `project/configs/`: ACT and ACT-Lang experiment configurations for CALVIN environment B, A/B/C joint training, augmentation, and size-matched variants.
- `project/scripts/`: dataset preparation, training, zero-shot evaluation, visualization, and reporting scripts.
- `project/tables/`: exported metrics, dataset statistics, training curves, zero-shot D diagnostics, and report tables.
- `project/figures/`: generated plots and visual evidence used by the report.
- `project/*.md`: project state, task board, draft notes, and model requirement notes.

Large runtime outputs, checkpoints, local caches, and upstream LeRobot source files are intentionally excluded from this repository.
To run the scripts, use a compatible LeRobot checkout or installed LeRobot environment, then point the scripts at the local CALVIN data and run directories described in the project configs.
