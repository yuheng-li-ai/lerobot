# CALVIN ACT Generalization Study Draft

## Project Aim

Study action policy learning and zero-shot environment generalization using ACT in LeRobot on the CALVIN benchmark. The study compares single-environment ACT-B against multi-environment ACT-ABC, then evaluates zero-shot transfer to environment D with visual-shift and action-chunking analyses.

## Global Execution Rules

- Work phase by phase.
- Update this file after each meaningful step with successes, failures, debugging attempts, and decisions.
- Keep ACT-B and ACT-ABC architecture and hyperparameters identical unless an explicit ablation is being run.
- Save configs, logs, checkpoints, metrics, figures, and tables under `project/`.
- Do not run large-scale training automatically. Agents may run only mini-trials or smoke tests.
- Full training must be prepared as `nohup` or batch scripts for manual launch.
- GPU selection must inspect memory and utilization first, then record the chosen GPU here.

## Phase 0 Log

### 2026-06-03 11:10 UTC - Initial repository and CALVIN inventory

Successes:
- Confirmed LeRobot source tree is present at `/home/zengzixuan/cvprojects/lerobot`.
- Confirmed project root contains LeRobot 0.5.2 metadata in `pyproject.toml`.
- Confirmed ACT implementation exists at `src/lerobot/policies/act/`.
- Confirmed `lerobot-train` CLI entry point exists in `pyproject.toml` and maps to `lerobot.scripts.lerobot_train:main`.
- Confirmed CALVIN framework repository exists at `/home/zengzixuan/cvprojects/calvin`.
- Confirmed CALVIN dataset README and downloader exist at `/home/zengzixuan/cvprojects/calvin/dataset/README.md` and `/home/zengzixuan/cvprojects/calvin/dataset/download_data.sh`.

Failures / blockers:
- `git status --short` failed because `.git/` is an empty directory in this workspace snapshot, so normal git provenance is unavailable.
- No CALVIN dataset split directories were found under `/home/zengzixuan/cvprojects/calvin/dataset`.
- No `task_D_D`, `task_ABC_D`, `task_ABCD_D`, debug dataset, or `episode_*.npz` files were found in the checked local paths.

Decisions:
- Treat `/home/zengzixuan/cvprojects/lerobot` as a source snapshot rather than a normal git checkout.
- Create a contained research workspace at `project/` so experiment artifacts do not mix with LeRobot source files.
- Do not attempt dataset download automatically because the CALVIN ABC split is large and network access/manual storage planning is required.
- Record dataset readiness as missing rather than inventing episode/frame counts.

Observations:
- The CALVIN README documents dataset splits `task_D_D` (166 GB), `task_ABC_D` (517 GB), `task_ABCD_D` (656 GB), and `calvin_debug_dataset` (1.3 GB).
- CALVIN observation keys include `rgb_static` with shape `(200, 200, 3)`, `rgb_gripper` with shape `(84, 84, 3)`, `robot_obs` with shape `(15,)`, and `rel_actions` with shape `(7,)`.
- LeRobot `TrainPipelineConfig` currently raises `NotImplementedError` for `dataset.repo_id` lists, so ACT-ABC should use a prepared merged/balanced dataset or a custom mini-trial harness rather than passing A/B/C directly to `lerobot-train`.

Next actions:
- Verify Python/LeRobot import environment with `uv run`.
- Inspect available GPU state only when mini-trial preparation begins.
- Once CALVIN split paths are available, run dataset statistics and image sampling for A/B/C/D.

### 2026-06-03 11:15 UTC - User note on environment configuration

Decision / note:
- User stated that `~/cvprojects` contains an environment configuration that can be used to understand data allocation.
- Previously observed likely candidate: `/home/zengzixuan/cvprojects/calvin_env.sh`.
- Per user instruction, no further work should be continued in this turn.

## Phase 3 Log

### 2026-06-06 UTC - Zero-shot D offline evaluation harness and success-rate handoff

Successes:
- Added tail-visible progress logging to `project/scripts/eval_zero_shot_D.py`.
- Added manual nohup launcher: `project/scripts/nohup_eval_zero_shot_D.sh`.
- The launcher sources `/home/zengzixuan/cvprojects/calvin_env.sh`, uses `$LEROBOT_PYTHON`, and does not use `uv`.
- Wrote success-rate model/data handoff requirements:
  - `project/TASK3_SUCCESS_RATE_MODEL_REQUIREMENTS.md`

Completed evidence before this note:
- Selected-checkpoint offline D action-error evaluation was run over all 99,022 D frames / 992 ACT chunks for six existing checkpoints.
- Main outputs:
  - `project/tables/task3/zero_shot_D_results.csv`
  - `project/tables/task3/zero_shot_D_action_chunks.csv`
  - `project/tables/task3/zero_shot_D_chunk_horizon.csv`
  - `project/figures/task3/zero_shot_action_l1_D.png`
  - `project/figures/task3/action_smoothness_D.png`
  - `project/figures/task3/chunk_boundary_jump_D.png`
  - `project/figures/task3/chunk_horizon_error_D.png`

Manual command for rerunning selected-checkpoint D offline inference:
```bash
GPU_ID=5 CHECKPOINT_MODE=selected bash project/scripts/nohup_eval_zero_shot_D.sh
```

Monitor after launch:
```bash
tail -f /EXT_DISK/users/zengzixuan/calvin_runs/task3/logs/<RUN_ID>.log
```

Decision:
- The current six ACT checkpoints are not language-conditioned, so CALVIN official success rate is not a valid primary metric for them.
- Current valid Phase 3 metric is offline zero-shot D action error plus ACT chunking diagnostics.
- For official CALVIN success rate, new models must be language-conditioned and implement `reset()` / `step(obs, goal)`.
- If the held-out target remains D, the recommended success-rate pair is `ACT-Lang-B` vs `ACT-Lang-ABC`, not `ACT-Lang-ACD`. Training on A+C+D makes D no longer unseen; in that case the held-out zero-shot environment should change to B.

### 2026-06-06 UTC - Strengthened ACT chunking analysis under visual shift

Successes:
- Added report-oriented analysis script:
  - `project/scripts/analyze_task3_chunk_visual_shift.py`
- Generated D visual-shift and chunk-robustness artifacts:
  - `project/tables/task3/visual_stats_ABCD_task3.csv`
  - `project/tables/task3/visual_shift_to_D_task3.csv`
  - `project/tables/task3/chunk_visual_shift_robustness.csv`
  - `project/tables/task3/chunk_visual_shift_robustness.tex`
  - `project/tables/task3/chunk_visual_shift_report_notes.md`
  - `project/figures/task3/visual_shift_ABCD_task3.png`
  - `project/figures/task3/chunk_visual_shift_robustness.png`
  - `project/figures/task3/chunk_accuracy_stability_tradeoff.png`

Key findings:
- Visual-shift proxy from training distribution to D:
  - B mean RGB L2 to D: `0.2948`
  - ABC mean RGB L2 to D: `0.0353`
  - B mean brightness gap to D: `0.1531`
  - ABC mean brightness gap to D: `0.0103`
- D action error improves from B-only to ABC-style training:
  - ACT-B Action L1: `0.2346`
  - ACT-ABC Action L1: `0.2154`
  - ACT-ABC-aug Action L1: `0.2134`
- ACT chunking stabilizes frame-to-frame predictions but over-smooths relative to D demonstrations:
  - D ground-truth mean step delta L2: `0.1738`
  - predicted mean step delta L2 range: `0.0162` to `0.0215`
  - smoothness ratio pred/GT is only about `0.09` to `0.12`.
- Chunk boundary stability is the main tradeoff:
  - ACT-B boundary amplification: `1.16x`
  - ACT-ABC boundary amplification: `1.82x`
  - ACT-ABC-aug boundary amplification: `1.90x`

Interpretation:
- Multi-environment training reduces visual mismatch to D and lowers D action error, especially gripper error.
- However, ACT's open-loop chunk queue concentrates corrections at chunk refresh points. The more accurate ABC-style models are less over-conservative, but this increases boundary jumps.
- The report should therefore frame ACT chunking robustness as an accuracy-stability tradeoff under visual distribution shift: chunking suppresses jitter, but can over-smooth wrong actions and produce larger discontinuities at chunk boundaries.

### 2026-06-06 UTC - Categorized report-ready Task 3 visualizations

Successes:
- Added categorized plotting script:
  - `project/scripts/plot_task3_report_visuals.py`
- Generated report-ready figures under separated folders:
  - chunking: `project/figures/task3_report/chunking/`
  - action error: `project/figures/task3_report/action_error/`
  - visual shift: `project/figures/task3_report/visual_shift/`
- Generated manifest and compact summary table:
  - `project/tables/task3_report/task3_report_visual_manifest.csv`
  - `project/tables/task3_report/task3_report_visual_summary.csv`

New chunking figures:
- `chunk_horizon_error_heatmap.png`: heatmap of D Action L1 across all 100 positions inside the ACT chunk.
- `within_chunk_vs_boundary_jump_D.png`: compares smooth within-chunk motion against jump size at chunk refresh boundaries.
- `chunk_boundary_tail_risk_D.png`: mean/q90/q99 boundary jump risk.
- `oversmoothing_index_D.png`: `1 - pred_step_delta / D_gt_step_delta`.
- `chunk_horizon_degradation_D.png`: last-10-step minus first-10-step chunk error.

New action-error figures:
- `action_dimension_error_heatmap_D.png`: per-model, per-action-dimension D L1 heatmap.
- `pose_vs_gripper_error_D.png`: separates first-6D pose action error from gripper error.

New visual-shift figures:
- `visual_shift_vs_action_error_D.png`: connects training visual distance to D with zero-shot D Action L1.
- `visual_shift_distance_summary_D.png`: normalized visual distance to D for A/B/C/ABC.

Quality check:
- Image dimensions were verified and the crowded heatmap labels were visually inspected.
- The output uses wrapped model names, wide figures, `bbox_inches="tight"`, and padding to avoid clipped labels/headers.

Report use:
- Use `chunk_horizon_error_heatmap.png` as the main ACT chunking mechanism figure.
- Use `action_dimension_error_heatmap_D.png` to show that gripper error dominates and ABC/ABC-aug reduce this error.
- Use `within_chunk_vs_boundary_jump_D.png` plus `oversmoothing_index_D.png` to support the claim that ACT chunking is smooth but over-conservative under D visual shift.
- Use `visual_shift_vs_action_error_D.png` to connect visual distribution shift to zero-shot action error.

### 2026-06-06 UTC - Retrospective W&B training-curve logging

Successes:
- Added W&B retrospective logging script:
  - `project/scripts/log_completed_runs_to_wandb.py`
- Verified `wandb==0.27.0` is available in the existing conda environment.
- Synced six completed runs to W&B project `calvin-act-generalization`, group `task1_task2_training`.
- Each W&B run logs native time-series curves for:
  - `train/loss`
  - `train/action_l1`
  - `val/loss`
  - `val/action_l1`
- To keep W&B panels responsive, train metrics were logged every 10 steps, and all 20 validation updates were logged.
- The full original `metrics.csv` for each run was uploaded as a W&B artifact for exact provenance.
- Wrote W&B manifest:
  - `project/tables/wandb_training_curve_runs.csv`

W&B run links:
- ACT-B: `https://wandb.ai/yuhengli72-fudan-university-school-of-management/calvin-act-generalization/runs/u1vk6n58`
- ACT-B-Aug: `https://wandb.ai/yuhengli72-fudan-university-school-of-management/calvin-act-generalization/runs/57ajrumx`
- ACT-ABC: `https://wandb.ai/yuhengli72-fudan-university-school-of-management/calvin-act-generalization/runs/4m48536g`
- ACT-ABC-Size-Matched: `https://wandb.ai/yuhengli72-fudan-university-school-of-management/calvin-act-generalization/runs/oaixlsdy`
- ACT-ABC-Aug: `https://wandb.ai/yuhengli72-fudan-university-school-of-management/calvin-act-generalization/runs/paf7zivo`
- ACT-ABC-Size-Matched-Aug: `https://wandb.ai/yuhengli72-fudan-university-school-of-management/calvin-act-generalization/runs/ffw67dhg`

Report wording:
- Use: "Training metrics were recorded locally during training and retrospectively synchronized to W&B for native visualization and export. Full step-level `metrics.csv` files are preserved as W&B artifacts."
- Avoid saying the runs were live-logged to W&B during training.

### 2026-06-06 UTC - Local W&B curve export

Successes:
- Added W&B API export script:
  - `project/scripts/export_wandb_training_curves.py`
- Downloaded the synced W&B run histories back to local CSV files and generated local PNG exports.
- Export folder:
  - `project/figures/wandb_export/`
- Supporting downloaded W&B histories:
  - `project/tables/wandb_export/`

Exported figures:
- Per-model 2x2 training-curve panels:
  - `project/figures/wandb_export/per_model/*_wandb_training_curves.png`
- Cross-model comparison panels:
  - `project/figures/wandb_export/comparisons/train_loss_comparison_wandb.png`
  - `project/figures/wandb_export/comparisons/train_action_l1_comparison_wandb.png`
  - `project/figures/wandb_export/comparisons/val_loss_comparison_wandb.png`
  - `project/figures/wandb_export/comparisons/val_action_l1_comparison_wandb.png`

Verification:
- Each downloaded W&B history CSV contains 10,000 train points and 20 validation points.
- Image dimensions were verified:
  - per-model panels: `2522x1812`
  - comparison panels: approximately `2390x1292`

### 2026-06-03 11:25 UTC - Phase 0 sandbox-external environment review

Successes:
- Removed the accidental `uv run` artifacts from the earlier check: `.venv` and `/tmp/uv-cache`.
- Used the existing conda environment at `/home/zengzixuan/miniforge3/envs/lerobot/bin/python`; no new PyTorch/CUDA install was attempted.
- Read `/home/zengzixuan/cvprojects/calvin_env.sh` and confirmed data allocation:
  - `CALVIN_RAW=/SSD_DISK/users/zengzixuan/calvin/task_ABC_D`
  - `CALVIN_LEROBOT_ROOT=/EXT_DISK/users/zengzixuan/processed-calvin`
  - `CALVIN_RUNS=/EXT_DISK/users/zengzixuan/calvin_runs`
- Confirmed raw CALVIN `task_ABC_D` exists and is readable.
- Confirmed raw training split statistics:
  - episodes: 147
  - frames/timesteps from `ep_lens.npy`: 1,795,045
  - language sequences: 17,870
  - unique language tasks: 34
- Confirmed raw validation split statistics:
  - episodes: 4
  - frames/timesteps from `ep_lens.npy`: 99,022
  - language sequences: 1,087
  - unique language tasks: 34
- Confirmed sample raw npz keys and shapes:
  - `rgb_static`: `uint8 (200, 200, 3)`
  - `rgb_gripper`: `uint8 (84, 84, 3)`
  - `robot_obs`: `float64 (15,)`
  - `rel_actions`: `float64 (7,)`
- Confirmed sandbox-external GPU state with `nvidia-smi`:
  - 8 x NVIDIA RTX A6000
  - each GPU has 49,140 MiB total memory
  - each GPU reported 1 MiB used and 0% utilization at check time
- Confirmed sandbox-external LeRobot/ACT import:
  - LeRobot 0.5.2
  - PyTorch 2.11.0+cu128
  - CUDA available: true
  - CUDA device count: 8
  - ACT defaults: `chunk_size=100`, `n_action_steps=100`, `vision_backbone=resnet18`

Failures / blockers:
- `CALVIN_LEROBOT_ROOT=/EXT_DISK/users/zengzixuan/processed-calvin` exists but is empty, so the raw CALVIN data has not yet been converted into LeRobot dataset format.
- `CALVIN_RUNS=/EXT_DISK/users/zengzixuan/calvin_runs` exists but is empty.
- `calvin_env.sh` exports deprecated `LEROBOT_HOME=/home/zengzixuan/cvprojects/lerobot`; LeRobot 0.5.2 raises a `ValueError` when this variable is present. Commands using this environment file must run `unset LEROBOT_HOME` and should use `HF_LEROBOT_HOME=$CALVIN_LEROBOT_ROOT` instead.

Decisions:
- For future checks and mini-trials, use `/home/zengzixuan/miniforge3/envs/lerobot/bin/python` or the existing conda environment directly, not `uv run`, unless the user explicitly requests dependency synchronization.
- Treat `/SSD_DISK/users/zengzixuan/calvin/task_ABC_D` as the authoritative raw CALVIN ABC->D source.
- Treat `/EXT_DISK/users/zengzixuan/processed-calvin` as the intended LeRobot dataset root, but currently not ready for `LeRobotDataset`.
- GPU 0 is a reasonable default candidate for mini-trials because all 8 GPUs were idle and equivalent at the review time; recheck before any mini-trial.

### 2026-06-03 11:35 UTC - Phase 0 completion harness

Successes:
- Found the authoritative A/B/C split inside raw CALVIN: `/SSD_DISK/users/zengzixuan/calvin/task_ABC_D/training/scene_info.npy`.
  - `calvin_scene_B`: frames 0-598909
  - `calvin_scene_C`: frames 598910-1191338
  - `calvin_scene_A`: frames 1191339-1795044
  - validation is treated as D for ABC->D zero-shot evaluation.
- Wrote Phase 0 scripts:
  - `project/scripts/calvin_phase0_common.py`
  - `project/scripts/prepare_dataset_stats.py`
  - `project/scripts/convert_calvin_to_lerobot.py`
  - `project/scripts/run_phase0_conversion.sh`
- Wrote dataset allocation config: `project/configs/calvin_datasets.yaml`.
- Generated updated Phase 0 tables:
  - `project/tables/dataset_stats_ABC.csv`
  - `project/tables/task_counts_ABCD.csv`
- Generated visual sample grid: `project/figures/env_samples_ABCD.png`.
- Ran a small LeRobot conversion smoke test only:
  - output root: `project/outputs/phase0_smoke`
  - datasets: `smoke_calvin_B`, `smoke_calvin_D`
  - 1 segment per dataset, 4 frames per segment
- Confirmed smoke LeRobot datasets can be read back with `LeRobotDataset`:
  - `observation.images.static`: `(3, 200, 200)`
  - `observation.images.gripper`: `(3, 84, 84)`
  - `observation.state`: `(15,)`
  - `action`: `(7,)`

Failures / fixes:
- Local conda environment originally had `datasets==2.21.0`, whose `Dataset.from_parquet` does not accept a `filters` keyword. This was an environment mismatch with LeRobot's `pyproject.toml` requirement (`datasets>=4.7.0,<5.0.0`), not a LeRobot source bug.
- Reverted the temporary LeRobot source compatibility patch and upgraded the existing conda environment to `datasets==4.8.5`.
- Confirmed the unpatched LeRobot dataset reader can load the smoke-converted datasets after the version fix.
- Sandbox readback needed writable Hugging Face cache variables. The conversion launcher now sets:
  - `HF_HOME=$CALVIN_RUNS/hf_cache`
  - `HF_DATASETS_CACHE=$HF_HOME/datasets`

Decisions:
- The conversion uses the official LeRobot dataset writer API pattern: `LeRobotDataset.create(...)`, `add_frame(...)`, `save_episode(...)`, and `finalize()`. There is no CALVIN-specific converter in this LeRobot worktree, so the CALVIN-to-LeRobot field mapping is a project adapter built on top of the official API.
- Converted LeRobot datasets use the minimal ACT schema only:
  - `observation.images.static` from raw `rgb_static`
  - `observation.images.gripper` from raw `rgb_gripper`
  - `observation.state` from raw `robot_obs`
  - `action` from raw `rel_actions`
- `scene_obs`, depth, and tactile streams are intentionally omitted from the default ACT datasets to avoid privileged state leakage and keep ACT-B/ACT-ABC comparable.
- Full conversion is prepared but not launched by the agent. User should run `project/scripts/run_phase0_conversion.sh`.

Manual command for full conversion:
```bash
bash project/scripts/run_phase0_conversion.sh
```

Expected converted datasets after the manual run:
- `/EXT_DISK/users/zengzixuan/processed-calvin/calvin_B`
- `/EXT_DISK/users/zengzixuan/processed-calvin/calvin_ABC`
- `/EXT_DISK/users/zengzixuan/processed-calvin/calvin_D`

### 2026-06-03 11:45 UTC - Environment correction after review

Correction:
- The temporary idea of patching LeRobot for old `datasets` compatibility was rejected. The source file `src/lerobot/datasets/io_utils.py` was restored to the official `filters=` implementation.
- The correct fix was applied to the existing conda environment:
  - upgraded `datasets` from `2.21.0` to `4.8.5`
  - this satisfies LeRobot's `pyproject.toml` requirement `datasets>=4.7.0,<5.0.0`
- `pip check` reports no broken requirements.
- Smoke LeRobot datasets read successfully with the official reader after the environment fix.
- Sandbox-external CUDA verification after the fix:
  - `cuda_available=True`
  - `cuda_device_count=8`
  - all devices are NVIDIA RTX A6000

Clarification:
- The converter is not copied from an official CALVIN-specific LeRobot converter; no such converter exists in this worktree.
- It does use LeRobot's official dataset writer API and examples: `LeRobotDataset.create(...)`, `add_frame(...)`, `save_episode(...)`, `finalize()`.
- The CALVIN-specific part is the adapter mapping raw fields to LeRobot features, using CALVIN's documented raw keys and the local `scene_info.npy` environment partition.

### 2026-06-03 11:55 UTC - Nohup conversion progress logging

Successes:
- Updated `project/scripts/convert_calvin_to_lerobot.py` to emit explicit `tqdm` progress bars for dataset episodes and per-episode frame conversion.
- Updated `project/scripts/run_phase0_conversion.sh` to use `PYTHONUNBUFFERED=1` and `python -u` so `tail -f` sees progress while the process is running.
- Ran a 3-frame smoke conversion with stdout/stderr redirected to `project/logs/nohup_progress_smoke.log`.
- Confirmed the log contains tail-visible progress lines such as `B episodes: 100%|...|` and `B B ep 1 frames`.

Manual monitoring command:
```bash
tail -f /EXT_DISK/users/zengzixuan/calvin_runs/phase0_conversion_logs/convert_B_ABC_D.log
```

### 2026-06-04 UTC - Full Phase 0 conversion verification

Successes:
- User launched `project/scripts/run_phase0_conversion.sh`; background PID `2375367` has exited.
- Conversion log exists at `/EXT_DISK/users/zengzixuan/calvin_runs/phase0_conversion_logs/convert_B_ABC_D.log`.
- Log tail ends with `Wrote D LeRobot dataset to /EXT_DISK/users/zengzixuan/processed-calvin/calvin_D`.
- No `Traceback`, `Error`, `ERROR`, `Exception`, `failed`, or `Failed` strings were found in the conversion log.
- Full converted datasets exist:
  - `/EXT_DISK/users/zengzixuan/processed-calvin/calvin_B`
  - `/EXT_DISK/users/zengzixuan/processed-calvin/calvin_ABC`
  - `/EXT_DISK/users/zengzixuan/processed-calvin/calvin_D`
- No residual raw image files were found under converted `images/` directories after video encoding.
- LeRobot metadata and sample readback succeeded for all converted datasets using `LeRobotDatasetMetadata` and `LeRobotDataset`.

Verified converted dataset sizes:
- `local/calvin_B`: 235 converted episodes/segments, 598,910 frames, 1 task, fps 30.
- `local/calvin_ABC`: 679 converted episodes/segments, 1,795,045 frames, 3 tasks, fps 30.
- `local/calvin_D`: 35 converted episodes/segments, 99,022 frames, 1 task, fps 30.

Verified sample schema on first and last frames of each dataset:
- `observation.images.static`: `(3, 200, 200)`, `torch.float32`
- `observation.images.gripper`: `(3, 84, 84)`, `torch.float32`
- `observation.state`: `(15,)`, `torch.float32`
- `action`: `(7,)`, `torch.float32`

Notes:
- The converted episode counts are higher than the raw episode counts because conversion used `--segment-frames 3000` to split long raw play episodes into smaller LeRobot episodes for manageable video files and downstream loading.
- Phase 0 dataset conversion is now sufficient for Phase 1/2 mini-trial preparation.

## Phase 1 Log

### 2026-06-04 09:50 UTC - ACT-B mini-trial harness and run

Successes:
- Wrote ACT-B config: `project/configs/act_B.yaml`.
- Wrote project ACT training harness: `project/scripts/train_act.py`.
- Ran syntax and CLI checks with `/home/zengzixuan/miniforge3/envs/lerobot/bin/python`; no `uv run` was used.
- Checked GPU state with sandbox-external `nvidia-smi`.
  - All 8 GPUs were idle: 1 MiB used, 0% utilization.
  - Selected GPU 0 for the mini-trial.
- Ran a bounded ACT-B mini-trial only:
  - dataset: `local/calvin_B`
  - root: `/EXT_DISK/users/zengzixuan/processed-calvin/calvin_B`
  - train episodes: `[0, 1, 2, 3]`
  - val episodes: `[4, 5]`
  - train frames: 7,912
  - val frames: 4,878
  - steps: 20
  - batch size: 2
  - device: `cuda:0`
  - action chunk: `chunk_size=100`, `n_action_steps=100`
- Saved mini-trial outputs:
  - `project/outputs/act_B/mini_trial/config_snapshot.yaml`
  - `project/outputs/act_B/mini_trial/metrics.csv`
  - `project/outputs/act_B/mini_trial/self_check.json`
  - `project/outputs/act_B/mini_trial/checkpoint/`
- Checkpoint contents were verified:
  - `config.json`
  - `model.safetensors`
  - `policy_preprocessor.json`
  - `policy_postprocessor.json`
  - normalizer/unnormalizer safetensors
  - `training_state.pt`
- Final script self-check passed.
- Post-run `nvidia-smi` showed GPU 0 back to 1 MiB used and 0% utilization.

Mini-trial metrics:
- Metrics rows: 20.
- Step 1 train total loss: 85.8137; train Action L1: 0.7908.
- Step 10 validation total loss: 17.8115; validation Action L1: 0.6525.
- Step 20 train total loss: 8.7388; train Action L1: 0.7995.
- Step 20 validation total loss: 10.1121; validation Action L1: 0.6652.

Failures / fixes:
- First run failed at validation step 10 because ACT's VAE branch only returns latent parameters in training mode. The harness originally used `policy.eval()` for validation, causing `mu/log_sigma` to be `None` during KL computation. Fixed by running validation under `torch.no_grad()` while keeping `policy.train()` mode, so supervised training loss is measured without optimizer updates.
- Second run completed training and checkpointing, but self-check failed because early metrics rows had expected `NaN` validation values before the first validation step. Fixed the self-check to require at least one finite validation Action L1 and a finite final validation Action L1.

Decisions:
- This first successful mini-trial used `pretrained_backbone_weights: null` only to avoid an implicit network download before the cache policy was decided.
- Keep `use_amp=false` for the mini-trial to reduce moving parts. If later changed for full training, ACT-B and ACT-ABC must change together.
- The mini-trial verifies dataset loading, ACT forward/backward, validation loss computation, checkpoint save, processor save, and metrics logging. It is not evidence of convergence quality.

### 2026-06-04 10:00 UTC - ResNet18 ImageNet cache and ACT-B mini-trial rerun

Successes:
- Confirmed ResNet18 ImageNet weights can be downloaded once through torchvision and then reused from cache.
- Cached weights at:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/torch_cache/hub/checkpoints/resnet18-f37072fd.pth`
  - size: 46,830,571 bytes (~45 MB)
- Updated `project/configs/act_B.yaml` to use:
  - `pretrained_backbone_weights: ResNet18_Weights.IMAGENET1K_V1`
- Updated `project/configs/calvin_datasets.yaml` to document:
  - `TORCH_HOME=$CALVIN_RUNS/torch_cache`
  - cached ResNet18 checkpoint path
- Reran the bounded ACT-B 20-step mini-trial with:
  - `TORCH_HOME=/EXT_DISK/users/zengzixuan/calvin_runs/torch_cache`
  - `CUDA_VISIBLE_DEVICES=0`
  - existing conda Python only; no `uv run`
- The rerun did not print a download message during model initialization, confirming the cache was reused.
- Checkpoint config now records:
  - `pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1`
  - `vision_backbone=resnet18`
- Final script self-check passed again.
- Post-run `nvidia-smi` showed all GPUs idle: 1 MiB used and 0% utilization.

Updated mini-trial metrics with ImageNet-pretrained ResNet18:
- Metrics rows: 20.
- Step 1 train total loss: 85.7644; train Action L1: 0.7415.
- Step 10 validation total loss: 17.8585; validation Action L1: 0.6983.
- Step 20 train total loss: 8.7509; train Action L1: 0.8123.
- Step 20 validation total loss: 10.0965; validation Action L1: 0.6469.

Decisions:
- The canonical ACT-B config now uses ImageNet-pretrained ResNet18.
- ACT-ABC must use the same `pretrained_backbone_weights: ResNet18_Weights.IMAGENET1K_V1` and the same `TORCH_HOME` cache path unless an explicit backbone initialization ablation is declared.
- Future training scripts must export `TORCH_HOME="$CALVIN_RUNS/torch_cache"` before constructing ACT.

### 2026-06-04 10:04 UTC - Shared environment script cleanup

Successes:
- Updated `/home/zengzixuan/cvprojects/calvin_env.sh` to be the canonical environment setup for CALVIN/LeRobot experiments.
- The script now exports:
  - `CALVIN_RAW=/SSD_DISK/users/zengzixuan/calvin/task_ABC_D`
  - `CALVIN_LEROBOT_ROOT=/EXT_DISK/users/zengzixuan/processed-calvin`
  - `CALVIN_RUNS=/EXT_DISK/users/zengzixuan/calvin_runs`
  - `LEROBOT_SOURCE=/home/zengzixuan/cvprojects/lerobot`
  - `LEROBOT_PYTHON=/home/zengzixuan/miniforge3/envs/lerobot/bin/python`
  - `HF_LEROBOT_HOME=$CALVIN_LEROBOT_ROOT`
  - `HF_HOME=$CALVIN_RUNS/hf_cache`
  - `HF_DATASETS_CACHE=$HF_HOME/datasets`
  - `TORCH_HOME=$CALVIN_RUNS/torch_cache`
- The script now unsets deprecated `LEROBOT_HOME`, which LeRobot 0.5.2 rejects.
- The script prepends `$LEROBOT_SOURCE/src` to `PYTHONPATH` if it is not already present.
- Verification after sourcing the script:
  - `LEROBOT_HOME` is unset.
  - `PYTHONPATH` includes LeRobot source.
  - `LEROBOT_PYTHON` imports LeRobot 0.5.2.
  - `LEROBOT_PYTHON` imports PyTorch 2.11.0+cu128.
  - `TORCH_HOME` points to `/EXT_DISK/users/zengzixuan/calvin_runs/torch_cache`.

Decision:
- Future agents should source `/home/zengzixuan/cvprojects/calvin_env.sh` and use `$LEROBOT_PYTHON` rather than `uv run`.

### 2026-06-04 10:20 UTC - ACT-B full-training launcher and resource estimate

Successes:
- Updated `project/scripts/train_act.py`:
  - supports episode range strings such as `"0:212"`;
  - supports `--output-dir` overrides for timestamped full-training runs;
  - prints tail-visible ASCII progress lines, for example `PROGRESS act_B_full [########------------------------] ...`;
  - supports periodic checkpoints through `training.save_freq`.
- Added memory probe script: `project/scripts/probe_act_memory.py`.
- Ran a one-batch GPU memory probe on GPU 0 with ImageNet-pretrained ResNet18:
  - batch size 8: peak reserved 1.066 GB
  - batch size 16: peak reserved 1.232 GB
  - batch size 32: peak reserved 2.107 GB
- Added full-scale ACT-B config: `project/configs/act_B_full.yaml`.
- Added manual nohup launcher: `project/scripts/nohup_train_act_B.sh`.
- Validated full config parsing, launcher shell syntax, and episode range parsing.
- Did not launch full-scale training.

Full ACT-B training design:
- Dataset: `local/calvin_B`.
- Deterministic split:
  - train episodes: `0:212`
  - validation episodes: `212:235`
- Actual split sizes:
  - train frames: 535,403
  - validation frames: 63,507
- Batch size: 32.
- Steps: 100,000.
- Approximate train-frame passes: 100,000 / (535,403 / 32) = 5.98.
- Validation every 5,000 steps over up to 64 validation batches.
- Checkpoints every 10,000 steps plus final checkpoint.

Resource estimate:
- GPU: NVIDIA RTX A6000, 49,140 MiB.
- Measured one-batch peak reserved memory at batch size 32: 2.107 GB.
- Expected full-run reserved memory: roughly 3-6 GB, leaving large headroom on A6000.
- Runtime estimate: roughly 6-12 hours for 100k steps, depending mostly on video decode/DataLoader throughput and validation/checkpoint overhead.
- Checkpoint disk estimate: roughly 5.9 GB for nine periodic checkpoints plus the final checkpoint, based on the mini-trial checkpoint directory size (~592 MB).

Manual launch command:
```bash
GPU_ID=0 bash project/scripts/nohup_train_act_B.sh
```

Tail monitoring:
```bash
tail -f /EXT_DISK/users/zengzixuan/calvin_runs/act_B/logs/<RUN_ID>.log
```

Decision:
- Use batch size 32 for ACT-B full training: it is conservative relative to A6000 memory and keeps ACT-B/ACT-ABC comparison easy to mirror.

### 2026-06-05 05:29 UTC - ACT-B full training completion check

Successes:
- User manually launched ACT-B full training with:
  - `GPU_ID=0 bash project/scripts/nohup_train_act_B.sh`
- Verified full run directory:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_B/20260604_101320_act_B_full_gpu0`
- Verified log:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_B/logs/20260604_101320_act_B_full_gpu0.log`
- Full training reached `100000/100000` steps.
- Final self-check passed:
  - metrics rows: 100,000
  - finite train Action L1: true
  - finite validation Action L1: true
  - checkpoint exists: true
  - model weights, processors, and training state exist: true
- No `Traceback`, `Error`, `ERROR`, `Exception`, `failed`, or `Failed` strings were found in the log.
- Training process exited and all GPUs were idle after completion.
- Final checkpoint exists:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_B/20260604_101320_act_B_full_gpu0/checkpoint`
- Periodic checkpoints exist at steps:
  - 10k, 20k, 30k, 40k, 50k, 60k, 70k, 80k, 90k.
- Disk usage:
  - final checkpoint: 592 MB
  - periodic checkpoints: 5.2 GB
  - full run directory: 5.8 GB

Full-training metrics:
- Step 1 train total loss: 82.8938; train Action L1: 0.8239.
- Step 100,000 train total loss: 0.4588; train Action L1: 0.4586.
- Step 100,000 validation total loss: 0.6230; validation Action L1: 0.6229.
- Best recorded validation Action L1: 0.5404 at step 10,000.
- Last pre-final validation plateau before step 100,000 was approximately 0.6391 Action L1, then final validation improved to 0.6229.

Interpretation:
- The model clearly fit the single-environment B training data; train Action L1 decreased from ~0.824 to ~0.459.
- Validation Action L1 was best early at step 10k and later worsened, which is an early overfitting indicator for ACT-B.
- For downstream comparison, keep both the final checkpoint and the step-10k checkpoint available:
  - final checkpoint for the planned "full training" endpoint;
  - step-10k checkpoint as a candidate early-stopped ACT-B model.

### 2026-06-05 06:10 UTC - ACT-B non-training extensions

Successes:
- Added Task 1 extension analysis script:
  - `project/scripts/analyze_act_B_extensions.py`
- Ran only offline analysis; no training job and no GPU workload was launched.
- Generated ACT-B loss/overfitting and checkpoint-selection artifacts:
  - `project/figures/loss_curve_act_B.png`
  - `project/tables/act_B_overfitting_summary.csv`
  - `project/tables/act_B_checkpoint_selection.csv`
- Generated environment B visual-distribution artifacts:
  - `project/figures/env_B_samples.png`
  - `project/tables/env_B_visual_stats.csv`
  - `project/tables/env_B_task_counts.csv`
- Generated environment B action-distribution artifacts:
  - `project/figures/env_B_action_distribution.png`
  - `project/tables/env_B_action_summary.csv`
  - `project/tables/env_B_action_stats.csv`
- Verified generated figures can be opened and have nonzero dimensions:
  - `loss_curve_act_B.png`: 1260x756
  - `env_B_samples.png`: 1200x448
  - `env_B_action_distribution.png`: 1260x720

Key results:
- Full-training overfitting summary:
  - final train Action L1: 0.4586
  - best validation Action L1: 0.5404 at step 10,000
  - final validation Action L1: 0.6229
  - final minus best validation Action L1: +0.0825
- Checkpoint recommendation for downstream ACT-B evaluation:
  - early-stopped candidate: `/EXT_DISK/users/zengzixuan/calvin_runs/act_B/20260604_101320_act_B_full_gpu0/checkpoints/step_00010000`
  - final endpoint: `/EXT_DISK/users/zengzixuan/calvin_runs/act_B/20260604_101320_act_B_full_gpu0/checkpoint`
- Environment B static-camera visual statistics from 2,000 sampled frames:
  - RGB mean: R=0.7977, G=0.6982, B=0.5858
  - RGB std: R=0.1858, G=0.1848, B=0.2180
  - brightness mean/std: 0.6939 / 0.0126
  - contrast mean/std: 0.1731 / 0.0040
- Environment B gripper-camera visual statistics from 2,000 sampled frames:
  - RGB mean: R=0.8107, G=0.6828, B=0.5749
  - RGB std: R=0.1985, G=0.2035, B=0.1948
  - brightness mean/std: 0.6895 / 0.0361
  - contrast mean/std: 0.1584 / 0.0374
- Environment B action statistics over 598,910 frames:
  - mean L2 norm of first 6 action dimensions: 0.5027
  - mean consecutive action delta L2: 0.2208
  - gripper close fraction: 0.4705
  - gripper open fraction: 0.5295
  - gripper switch count: 14,142

Interpretation:
- Task 1 now has the required non-training extension material for a small research-style analysis: overfitting evidence, checkpoint-selection rationale, visual statistics, sample visualization, and action-distribution statistics.
- The early validation optimum at step 10k suggests ACT-B may memorize environment-B-specific behavior or over-optimize the train split after early convergence. Both final and early-stopped checkpoints should be carried into zero-shot D evaluation.
- The environment B visual stats are now a baseline for later A/B/C/D visual-shift comparison.
- The action summary is a baseline for later ACT-ABC and zero-shot D action smoothness/chunk-boundary analysis.

### 2026-06-05 06:25 UTC - Task 1 extension scope before augmentation experiments

Purpose:
- Freeze the ACT-B baseline analysis scope before any later data-augmentation model is trained, so baseline outputs and augmentation outputs do not get mixed.
- Failure taxonomy is intentionally excluded from Task 1 for now because D zero-shot evaluation has not been run yet.
- All items below are offline analysis/visualization only. They must not launch training.

Already completed for baseline ACT-B:
- ACT-B mini-trial and self-check.
- ACT-B full training script and manual full run verification.
- ACT-B full-run loss curve.
- ACT-B overfitting summary.
- ACT-B checkpoint selection table.
- Environment B image samples.
- Environment B RGB/brightness/contrast summary statistics.
- Environment B action summary statistics.
- Environment B per-action summary table.
- Environment B task-count table.

Baseline ACT-B extensions to add next:
- `Train-Val Generalization Gap`: plot validation Action L1 minus nearest train Action L1 over step, to make overfitting clearer than the raw loss curve alone.
- `Action smoothness`: compute and visualize consecutive action changes, using the dataset action trajectory as the environment-B baseline.
- `Chunk analysis`: compute chunk-window action variation and boundary-style jumps from the ground-truth action stream as a baseline; later model-predicted chunk-boundary jumps should use the same naming convention.
- `Per-Action-Dimension Distribution`: add violin or box plots for all 7 action dimensions.
- `Action Delta Heatmap`: plot action-delta magnitude by time bin and action dimension.
- `Environment B Visual Color Profile`: bar chart comparing static and gripper camera RGB mean/std.
- `Brightness / Contrast Histogram`: histograms for sampled static and gripper camera brightness/contrast.
- `Task Frequency Bar Chart`: visualize environment-B language/task distribution.
- `Representative Trajectory Strip`: show static camera frames from one B trajectory together with the action-norm curve.

File organization rule:
- Baseline ACT-B outputs stay under:
  - figures: `project/figures/`
  - tables: `project/tables/`
  - run/checkpoint artifacts: `/EXT_DISK/users/zengzixuan/calvin_runs/act_B/20260604_101320_act_B_full_gpu0`
- Future data-augmentation model outputs must use distinct names, for example:
  - `act_B_aug_*` for tables/figures
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_B_aug/...` for run artifacts
- Do not overwrite baseline ACT-B files with augmentation-model analysis.

### 2026-06-05 06:55 UTC - Task 1 baseline extension figures completed

Successes:
- Extended `project/scripts/analyze_act_B_extensions.py` to generate the planned Task 1 baseline figures/tables in one place.
- Reran the script as offline analysis only:
  - no training launched
  - no GPU workload launched
  - raw environment B data was read from `/SSD_DISK/users/zengzixuan/calvin/task_ABC_D`
- Generated and verified the following new figures:
  - `project/figures/train_val_gap_act_B.png`
  - `project/figures/env_B_action_smoothness.png`
  - `project/figures/env_B_chunk_baseline.png`
  - `project/figures/env_B_action_violin.png`
  - `project/figures/env_B_action_delta_heatmap.png`
  - `project/figures/env_B_visual_color_profile.png`
  - `project/figures/env_B_brightness_contrast_hist.png`
  - `project/figures/env_B_task_frequency.png`
  - `project/figures/env_B_representative_trajectory_strip.png`
- Generated the following supporting tables:
  - `project/tables/act_B_train_val_gap.csv`
  - `project/tables/env_B_action_smoothness.csv`
  - `project/tables/env_B_chunk_baseline.csv`
  - `project/tables/env_B_chunk_baseline_summary.csv`
  - `project/tables/env_B_action_delta_heatmap.csv`
  - `project/tables/env_B_representative_trajectory.csv`

Figure verification:
- `train_val_gap_act_B.png`: 1260x684
- `env_B_action_smoothness.png`: 1475x648
- `env_B_chunk_baseline.png`: 1475x648
- `env_B_action_violin.png`: 1332x720
- `env_B_action_delta_heatmap.png`: 1368x720
- `env_B_visual_color_profile.png`: 1440x648
- `env_B_brightness_contrast_hist.png`: 1440x648
- `env_B_task_frequency.png`: 1440x1080
- `env_B_representative_trajectory_strip.png`: 1800x827

Key results:
- Train-validation gap:
  - uses validation Action L1 minus the trailing 1,000-step mean train Action L1.
  - step 10,000 smoothed gap: `-0.1288`
  - step 40,000 smoothed gap: `+0.0193`
  - step 90,000 smoothed gap: `+0.1573`
  - step 100,000 smoothed gap: `+0.1762`
  - interpretation: validation is initially easier/lower than the smoothed train metric, then the gap becomes positive after roughly 40k, consistent with overfitting.
- Action smoothness baseline:
  - mean absolute per-step delta for first 6 action dimensions:
    - dx `0.0425`, dy `0.0361`, dz `0.0451`
    - droll `0.0879`, dpitch `0.0621`, dyaw `0.0688`
  - mean L2 per-step delta over first 6 dims: `0.1781`
  - q99 L2 per-step delta over first 6 dims: `0.6160`
- Chunk baseline with `chunk_size=100`:
  - chunks analyzed: `6,022`
  - mean within-chunk delta L2 over first 6 dims: `0.1784`
  - mean first-last chunk delta L2 over first 6 dims: `0.7254`
  - mean boundary-style jump L2 over first 6 dims: `0.1792`
  - q90/q99 boundary-style jump L2: `0.3245 / 0.6073`
- Representative trajectory strip:
  - uses a long environment-B trajectory segment and 10 static-camera frames.
  - stores frame ids and action norms in `project/tables/env_B_representative_trajectory.csv`.

Decisions:
- Failure taxonomy remains excluded until D zero-shot evaluation exists.
- Current smoothness/chunk numbers are ground-truth dataset baselines. Later model-predicted action smoothness and model-predicted chunk-boundary jumps should reuse these names as reference baselines.
- Future augmentation-model analysis should generate analogous files with `act_B_aug_*` names or a clearly separated augmentation output directory.

### 2026-06-05 07:05 UTC - Train-val gap smoothing correction

Issue:
- The first version of `project/tables/act_B_train_val_gap.csv` used `val_action_l1 - train_action_l1` at the same step.
- This made the table and figure visibly noisy because `train_action_l1` is a single shuffled mini-batch metric, while validation Action L1 is averaged over up to 64 validation batches.
- Therefore the old gap mixed two different noise levels and was not the best overfitting diagnostic.

Fix:
- Updated `project/scripts/analyze_act_B_extensions.py`.
- `project/tables/act_B_train_val_gap.csv` now records:
  - raw single-batch train Action L1
  - trailing 1,000-step mean train Action L1
  - validation Action L1
  - raw gap
  - smoothed gap
- `project/figures/train_val_gap_act_B.png` now plots:
  - raw mini-batch gap as a faint reference curve
  - smoothed gap as the main curve
- Reran the offline analysis script only. No training and no GPU workload were launched.

Corrected gap values:
- step 10,000 smoothed gap: `-0.1288`
- step 40,000 smoothed gap: `+0.0193`
- step 60,000 smoothed gap: `+0.0831`
- step 80,000 smoothed gap: `+0.1470`
- step 100,000 smoothed gap: `+0.1762`

Interpretation:
- The smoothed gap still supports the same conclusion: ACT-B begins to show a positive generalization gap after roughly 40k steps.
- The corrected table is better for reporting because it compares validation against a local train trend rather than a single noisy train batch.

### 2026-06-05 07:15 UTC - Train Action L1 smoothing correction

Issue:
- `project/figures/loss_curve_act_B.png` still plotted raw `train_action_l1` as the main train curve.
- Raw `train_action_l1` is logged from one shuffled mini-batch per step, so the curve is naturally jagged and can visually exaggerate instability.
- This was a visualization/logging issue, not evidence that the ACT optimization was run differently from the intended LeRobot-style training loop.

Fix:
- Updated `project/scripts/analyze_act_B_extensions.py`.
- Added a trailing 1,000-step mean for train Action L1.
- `project/figures/loss_curve_act_B.png` now shows:
  - raw train mini-batch Action L1 as a faint reference curve
  - 1,000-step mean train Action L1 as the main train curve
  - validation Action L1 as before
- Added `project/tables/act_B_train_l1_smoothed.csv`.
- Reran the offline analysis script only. No training and no GPU workload were launched.

Corrected train trend:
- step 1,000 smoothed train Action L1: `0.7553`
- step 5,000 smoothed train Action L1: `0.6926`
- step 50,000 smoothed train Action L1: `0.5389`
- step 75,000 smoothed train Action L1: `0.4840`
- step 100,000 smoothed train Action L1: `0.4467`

Interpretation:
- The smoothed train curve clearly decreases over training, while the raw mini-batch curve remains useful only as an audit/reference signal.
- Future reports should use the smoothed train curve for visual interpretation and keep raw values only for transparency.

### 2026-06-05 07:25 UTC - Representative trajectory strip correction

Issue:
- `project/figures/env_B_representative_trajectory_strip.png` appeared very jagged.
- The old version selected the longest environment-B segment (`456757-500843`, about 44k frames) and drew a connected raw action-norm curve from 800 sparse samples across that whole play segment.
- This compressed a long multi-task play sequence into one curve and connected points separated by roughly 55 frames, so the figure visually exaggerated action-norm oscillation.
- This was a visualization choice issue, not a model-output instability issue.

Fix:
- Updated `project/scripts/analyze_act_B_extensions.py`.
- The representative trajectory strip now uses a local contiguous 1,200-frame window from the same long B segment:
  - source long segment: `456757-500843`
  - plotted window: `478200-479399`
- `project/figures/env_B_representative_trajectory_strip.png` now plots:
  - raw action norm as a faint curve
  - 31-frame trailing mean action norm as the main curve
- `project/tables/env_B_representative_trajectory.csv` now records:
  - source segment start/end
  - plotted window start/end
  - sampled frame ids
  - raw action L2 over the first 6 action dimensions
  - 31-frame trailing mean action L2
- Refreshed only the representative trajectory figure/table. No training and no GPU workload were launched.

Interpretation:
- Some local action variation remains expected because CALVIN relative actions include stop/start, reaching, rotation, and manipulation transitions.
- The corrected figure is now suitable as a representative local trajectory visualization, while raw high-frequency variation remains visible as a faint audit signal.

### 2026-06-05 07:35 UTC - Separate gripper diagnostics

Decision:
- Keep the existing action-distribution and action-smoothness findings that include the gripper dimension, because they document that the 7th action dimension behaves differently from the first 6 dimensions.
- Add separate gripper-specific figures/tables so the binary open/close command is not interpreted as a continuous motion dimension.
- Future data-augmentation analysis must also keep the current baseline gripper files and generate separate augmentation gripper files, using names such as `act_B_aug_gripper_*` or a clearly separated augmentation output directory.

Successes:
- Added gripper-specific diagnostics to `project/scripts/analyze_act_B_extensions.py`.
- Generated baseline gripper figures:
  - `project/figures/env_B_gripper_diagnostics.png`
  - `project/figures/env_B_gripper_timeline.png`
- Generated baseline gripper tables:
  - `project/tables/env_B_gripper_summary.csv`
  - `project/tables/env_B_gripper_runs.csv`
- Reran offline analysis only. No training and no GPU workload were launched.

Key results:
- Gripper action values are strictly binary in environment B:
  - close command `-1.0`: `281,768` frames, fraction `0.4705`
  - open command `+1.0`: `317,142` frames, fraction `0.5295`
- Number of open/close switches: `14,142`
- Switch rate: `23.61` switches per 1,000 frames
- Mean gripper run length: `42.35` frames
- Median gripper run length: `38` frames
- q90/q99 gripper run length: `65 / 133.58` frames

Interpretation:
- The gripper dimension should be reported as a binary state/transition signal, not as a continuous action dimension like `dx` or `dyaw`.
- For motion smoothness, the main metric should remain the L2 delta over the first 6 action dimensions.
- For gripper behavior, the appropriate diagnostics are open/close balance, switch count/rate, run-length distribution, and timeline plots.

### 2026-06-05 07:55 UTC - ACT-B data augmentation harness, verification, and mini-trial

Goal:
- Prepare a data-augmentation version of ACT-B before any full augmentation training.
- Verify that augmentation actually changes images.
- Verify that augmented ACT-B can run a short training pre-run.
- Generate a manual `nohup` launcher with tail-visible progress bars.

Successes:
- Updated `project/scripts/train_act.py` with a config-driven `ImageBatchAugmenter`.
- Augmentation is applied only to training camera tensors after uint8-to-float conversion and before the LeRobot preprocessor.
- Validation batches are not augmented, so validation Action L1 remains comparable to baseline ACT-B.
- The ACT architecture, optimizer, batch size, split, and full-training step count remain matched to ACT-B; the explicit ablation is visual augmentation only.
- Added augmentation verification script:
  - `project/scripts/verify_act_B_augmentation.py`
- Added augmentation configs:
  - `project/configs/act_B_aug.yaml`
  - `project/configs/act_B_aug_full.yaml`
- Added manual full-training launcher:
  - `project/scripts/nohup_train_act_B_aug.sh`
- Static checks passed:
  - Python compile for `train_act.py` and `verify_act_B_augmentation.py`
  - YAML parse for `act_B_aug.yaml` and `act_B_aug_full.yaml`
  - bash syntax check for `nohup_train_act_B_aug.sh`

Augmentation settings:
- `enabled: true`
- `probability: 1.0`
- `brightness: 0.12`
- `contrast: 0.12`
- `saturation: 0.08`
- `gaussian_noise_std: 0.01`
- `clip_min: 0.0`
- `clip_max: 1.0`

Augmentation verification:
- Command used:
  - `project/scripts/verify_act_B_augmentation.py --config project/configs/act_B_aug.yaml --output-dir project`
- Outputs:
  - `project/figures/act_B_aug_verification.png`
  - `project/tables/act_B_aug_verification.csv`
- Results:
  - static camera mean/max absolute pixel delta: `0.02936 / 0.10341`
  - gripper camera mean/max absolute pixel delta: `0.01769 / 0.08147`
  - original and augmented image ranges stayed within `[0.0, 1.0]`

Mini-trial:
- GPU check before launch:
  - all 8 NVIDIA RTX A6000 GPUs were idle: 1 MiB used, 0% utilization.
  - selected GPU 0.
- Ran only a 20-step mini-trial:
  - config: `project/configs/act_B_aug.yaml`
  - output: `project/outputs/act_B_aug/mini_trial`
  - train episodes: `[0, 1, 2, 3]`
  - validation episodes: `[4, 5]`
  - train frames: `7,912`
  - validation frames: `4,878`
  - batch size: `2`
  - steps: `20`
- Progress-bar logging worked, for example:
  - `PROGRESS act_B_aug [################----------------] 10/20 ( 50.0%) elapsed=... eta=...`
- Self-check passed:
  - metrics rows: `20`
  - finite train Action L1: true
  - finite validation Action L1: true
  - checkpoint exists: true
  - model weights, processors, and training state exist: true
- Final mini-trial metrics:
  - step 20 train loss: `8.7504`
  - step 20 train Action L1: `0.8117`
  - step 20 validation loss: `10.1135`
  - step 20 validation Action L1: `0.6636`
- Post-run GPU check:
  - all GPUs returned to idle: 1 MiB used, 0% utilization.

Manual full-training launcher:
```bash
GPU_ID=0 bash project/scripts/nohup_train_act_B_aug.sh
```

The script prints the exact log path. Monitor with:
```bash
tail -f /EXT_DISK/users/zengzixuan/calvin_runs/act_B_aug/logs/<RUN_ID>.log
```

Expected tail-visible progress line:
```text
PROGRESS act_B_aug_full [########------------------------] 25000/100000 ( 25.0%) elapsed=... eta=...
```

Decision:
- Full-scale augmentation training was not launched by the agent.
- Future augmentation model figures/tables should use distinct names such as `act_B_aug_*` and should include the same baseline diagnostics, including separate gripper analysis.

### 2026-06-06 UTC - ACT-B data augmentation full-training completion check

Context:
- User manually launched ACT-B data-augmentation full training on GPU 0.
- The agent only checked outputs after the user reported completion.

Run:
- Run directory:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_B_aug/20260605_063735_act_B_aug_full_gpu0`
- Log:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_B_aug/logs/20260605_063735_act_B_aug_full_gpu0.log`
- PID file:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_B_aug/20260605_063735_act_B_aug_full_gpu0/nohup.pid`

Completion checks:
- Training reached `100000/100000` steps.
- Tail-visible progress reached:
  - `PROGRESS act_B_aug_full [################################] 100000/100000 (100.0%) elapsed=18.67h eta=0.00h`
- `self_check.json` passed:
  - metrics rows: `100,000`
  - finite train Action L1: true
  - finite validation Action L1: true
  - checkpoint exists: true
  - model weights, processors, and training state exist: true
- No `Traceback`, `ERROR`, `Exception`, `failed`, or CUDA OOM errors were found in the inspected log.
- The nohup process has exited.
- Post-run GPU check:
  - all 8 RTX A6000 GPUs were idle, including GPU 0: 1 MiB used, 0% utilization.

Artifacts:
- Final checkpoint:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_B_aug/20260605_063735_act_B_aug_full_gpu0/checkpoint`
- Periodic checkpoints:
  - steps 10k through 90k under `/EXT_DISK/users/zengzixuan/calvin_runs/act_B_aug/20260605_063735_act_B_aug_full_gpu0/checkpoints/`
- Metrics:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_B_aug/20260605_063735_act_B_aug_full_gpu0/metrics.csv`
- Full run disk usage:
  - about `5.8G`
- Added a full-training row to:
  - `project/tables/main_training_results.csv`

Full-training metrics:
- Step 1 train Action L1: `0.8053`
- Final train loss: `0.4853`
- Final raw train Action L1: `0.4852`
- Final trailing 1,000-step mean train Action L1: `0.4515`
- Final validation loss: `0.6487`
- Final validation Action L1: `0.6486`
- Best validation Action L1: `0.5409` at step `15,000`
- Final minus best validation Action L1: `+0.1076`

Initial comparison with ACT-B baseline:
- ACT-B best validation Action L1:
  - `0.5404` at step `10,000`
- ACT-B-aug best validation Action L1:
  - `0.5409` at step `15,000`
- ACT-B final validation Action L1:
  - `0.6229`
- ACT-B-aug final validation Action L1:
  - `0.6486`

Interpretation:
- Augmentation did not improve environment-B validation Action L1 in this full run.
- The best validation score is essentially tied with baseline, but the final checkpoint is worse than baseline final validation.
- ACT-B-aug also shows a stronger late overfitting signal than baseline:
  - baseline final-minus-best validation gap: `+0.0825`
  - augmentation final-minus-best validation gap: `+0.1076`
- For downstream comparison, keep both:
  - ACT-B-aug best validation metric: step `15,000` (metric only; no checkpoint was saved at this step)
  - ACT-B-aug best saved checkpoint: step `10,000`
  - ACT-B-aug nearest saved checkpoints around best metric: step `10,000` and step `20,000`
  - ACT-B-aug final checkpoint: step `100,000`
- The important test for augmentation is likely zero-shot D or cross-environment evaluation, not B validation alone.

### 2026-06-06 UTC - ACT-B augmentation one-to-one visualization completion

Goal:
- Make ACT-B-aug visualization follow the same baseline ACT-B visualization structure.
- Preserve all baseline files.
- Generate distinct `act_B_aug_*` figures/tables so later reports can compare baseline vs augmentation without missing rows or overwriting artifacts.

Successes:
- Added one-to-one augmentation analysis script:
  - `project/scripts/analyze_act_B_aug_extensions.py`
- Generated a coverage manifest:
  - `project/tables/act_B_aug_visualization_coverage.csv`
- Coverage check:
  - 29 one-to-one baseline-to-augmentation entries.
  - missing artifact count: `0`.
- Generated comparison table:
  - `project/tables/act_B_vs_aug_summary.csv`

Generated ACT-B-aug model-metric counterparts:
- `project/figures/act_B_aug/act_B_aug_loss_curve.png`
- `project/figures/act_B_aug/act_B_aug_train_val_gap.png`
- `project/tables/act_B_aug_overfitting_summary.csv`
- `project/tables/act_B_aug_checkpoint_selection.csv`
- `project/tables/act_B_aug_train_l1_smoothed.csv`
- `project/tables/act_B_aug_train_val_gap.csv`

Generated ACT-B-aug visual augmentation counterparts:
- `project/figures/act_B_aug/act_B_aug_samples.png`
- `project/figures/act_B_aug/act_B_aug_visual_color_profile.png`
- `project/figures/act_B_aug/act_B_aug_brightness_contrast_hist.png`
- `project/tables/act_B_aug_visual_stats.csv`

Generated ACT-B-aug data-invariant counterparts:
- These are copied from the environment-B baseline because image augmentation changes only camera observations during training; B action labels, task counts, gripper labels, chunk labels, and representative action trajectories are unchanged.
- Figures:
  - `project/figures/act_B_aug/act_B_aug_action_distribution.png`
  - `project/figures/act_B_aug/act_B_aug_action_smoothness.png`
  - `project/figures/act_B_aug/act_B_aug_chunk_baseline.png`
  - `project/figures/act_B_aug/act_B_aug_action_violin.png`
  - `project/figures/act_B_aug/act_B_aug_action_delta_heatmap.png`
  - `project/figures/act_B_aug/act_B_aug_task_frequency.png`
  - `project/figures/act_B_aug/act_B_aug_representative_trajectory_strip.png`
  - `project/figures/act_B_aug/act_B_aug_gripper_diagnostics.png`
  - `project/figures/act_B_aug/act_B_aug_gripper_timeline.png`
- Tables:
  - `project/tables/act_B_aug_action_summary.csv`
  - `project/tables/act_B_aug_action_stats.csv`
  - `project/tables/act_B_aug_action_smoothness.csv`
  - `project/tables/act_B_aug_chunk_baseline.csv`
  - `project/tables/act_B_aug_chunk_baseline_summary.csv`
  - `project/tables/act_B_aug_action_delta_heatmap.csv`
  - `project/tables/act_B_aug_task_counts.csv`
  - `project/tables/act_B_aug_representative_trajectory.csv`
  - `project/tables/act_B_aug_gripper_summary.csv`
  - `project/tables/act_B_aug_gripper_runs.csv`

Checkpoint-selection correction:
- ACT-B-aug best validation Action L1 occurred at step `15,000`, but no checkpoint was saved there because save frequency was every `10,000` steps.
- `project/tables/act_B_aug_checkpoint_selection.csv` now marks step `15,000` as `best_val_metric_only`.
- Actual saved candidates around the best metric are:
  - step `10,000`: validation Action L1 `0.5478`
  - step `20,000`: validation Action L1 `0.5516`
  - final step `100,000`: validation Action L1 `0.6486`
- Among saved checkpoints, step `10,000` is selected as `best_saved_checkpoint` because its validation Action L1 is lower than step `20,000`.
- Best saved ACT-B-aug checkpoint path:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_B_aug/20260605_063735_act_B_aug_full_gpu0/checkpoints/step_00010000`

Key comparison table:
- `project/tables/act_B_vs_aug_summary.csv` records:
  - ACT-B best validation Action L1: `0.5404` at step `10,000`
  - ACT-B-aug best validation Action L1: `0.5409` at step `15,000`
  - augmentation minus baseline best validation: `+0.00052`
  - augmentation minus baseline final validation: `+0.02568`

Decision:
- Use `project/tables/act_B_aug_visualization_coverage.csv` as the checklist for report assembly.
- Use step `10,000` as the ACT-B-aug best saved checkpoint for downstream evaluation unless a future rerun saves a step `15,000` checkpoint.
- Do not treat step `15,000` as an evaluable checkpoint unless a future rerun saves checkpoints every `5,000` steps.

### 2026-06-06 UTC - ACT-B figure folder organization

Successes:
- Created separate figure folders for easier report assembly:
  - `project/figures/act_B_baseline/`
  - `project/figures/act_B_aug/`
- Copied baseline ACT-B figures into `project/figures/act_B_baseline/`.
- Copied ACT-B augmentation figures into `project/figures/act_B_aug/`.
- Generated figure-folder index:
  - `project/tables/act_B_figure_folder_index.csv`
- After user request, deleted the root-level duplicate ACT-B/ACT-B-aug images from `project/figures/`.
- `project/figures/env_samples_ABCD.png` remains at root because it is the Phase 0 overview figure, not a duplicate of the two Task 1 folders.

Counts:
- `project/figures/act_B_baseline/`: `14` PNG figures.
- `project/figures/act_B_aug/`: `15` PNG figures.
- The augmentation folder has one extra figure because it includes `act_B_aug_verification.png`.

Decision:
- Use the subfolders for report writing and slide assembly.
- Treat the subfolder files as the current report-ready image locations.

### 2026-06-06 UTC - ACT-B figure title clipping QA

Context:
- User noticed that several figure titles / headers looked clipped, especially:
  - baseline / augmentation brightness and contrast histograms
  - baseline / augmentation gripper diagnostics
  - baseline / augmentation visual color profile
  - ACT-B augmentation verification

Fix:
- Updated the plotting scripts to use constrained figure layout and tight export padding for the affected multi-panel figures:
  - `project/scripts/analyze_act_B_extensions.py`
  - `project/scripts/analyze_act_B_aug_extensions.py`
  - `project/scripts/verify_act_B_augmentation.py`
- Regenerated the affected offline figures only. No training was launched.
- Copied the regenerated figures back into the report-ready folders:
  - `project/figures/act_B_baseline/`
  - `project/figures/act_B_aug/`

Verification:
- Manually inspected the affected PNGs; the titles are now visible and not cut off:
  - `project/figures/act_B_baseline/env_B_brightness_contrast_hist.png`
  - `project/figures/act_B_baseline/env_B_gripper_diagnostics.png`
  - `project/figures/act_B_baseline/env_B_visual_color_profile.png`
  - `project/figures/act_B_aug/act_B_aug_brightness_contrast_hist.png`
  - `project/figures/act_B_aug/act_B_aug_gripper_diagnostics.png`
  - `project/figures/act_B_aug/act_B_aug_visual_color_profile.png`
  - `project/figures/act_B_aug/act_B_aug_verification.png`
- Repaired `project/tables/act_B_aug_visualization_coverage.csv` so it points to the current subfolder paths instead of the deleted root-level duplicate image paths.
- Coverage check now reports `0` missing artifacts.
- Current folder counts:
  - baseline: `14` PNG figures
  - augmentation: `15` PNG figures
  - root `project/figures/`: only `env_samples_ABCD.png`

Decision:
- Use the subfolder paths in `project/tables/act_B_aug_visualization_coverage.csv` and `project/tables/act_B_figure_folder_index.csv` as the source of truth for Task 1 report assembly.

## Phase 2 Log

### 2026-06-05 06:56 UTC - Task 2 initial configuration, no visualization

Goal:
- Prepare Task 2 experiment configs and manual launchers for ACT-ABC, ACT-ABC-aug, ACT-ABC-size-matched, and ACT-ABC-size-matched-aug.
- Do not generate visualizations in this step.
- Do not start full-scale training.
- Avoid GPU0 because GPU0 is currently occupied by the ACT-B augmentation experiment.

GPU status:
- GPU state was checked outside the sandbox with `nvidia-smi`.
- GPU0 had an active Python process using about `2951 MiB`; GPU0 is reserved and must not be used for Task 2.
- GPUs 1-7 were idle at check time, each reporting `1 MiB` used and `0%` utilization.

Successes:
- Updated `project/scripts/train_act.py` to support `ACT_DEVICE_OVERRIDE`.
  - This lets Task 2 launchers set `CUDA_VISIBLE_DEVICES=<nonzero physical GPU>` and run the policy internally on `cuda:0` within that isolated visible-device view.
  - Direct config runs default to `policy.device: cuda:1`, so accidental direct runs avoid GPU0.
- Added deterministic Task 2 config/split generator:
  - `project/scripts/prepare_task2_configs.py`
- Generated Task 2 episode split artifact:
  - `project/configs/task2_episode_splits.yaml`
- Generated Task 2 split and experiment tables:
  - `project/tables/task2_episode_splits.csv`
  - `project/tables/task2_experiment_matrix.csv`
- Generated mini-trial configs:
  - `project/configs/act_ABC.yaml`
  - `project/configs/act_ABC_aug.yaml`
  - `project/configs/act_ABC_size_matched.yaml`
  - `project/configs/act_ABC_size_matched_aug.yaml`
- Generated full-training configs:
  - `project/configs/act_ABC_full.yaml`
  - `project/configs/act_ABC_aug_full.yaml`
  - `project/configs/act_ABC_size_matched_full.yaml`
  - `project/configs/act_ABC_size_matched_aug_full.yaml`
- Added Task 2 manual launchers:
  - `project/scripts/nohup_train_task2.sh`
  - `project/scripts/nohup_train_act_ABC.sh`
  - `project/scripts/nohup_train_act_ABC_aug.sh`
  - `project/scripts/nohup_train_act_ABC_size_matched.sh`
  - `project/scripts/nohup_train_act_ABC_size_matched_aug.sh`

Split design:
- ACT-ABC full split is stratified by environment using about 90% converted episodes for train and 10% for validation per environment.
- ACT-ABC full train:
  - A: 204 episodes, 542,459 frames
  - B: 212 episodes, 535,403 frames
  - C: 195 episodes, 531,235 frames
  - total: 611 episodes, 1,609,097 frames
- ACT-ABC validation:
  - A: 23 episodes, 61,247 frames
  - B: 23 episodes, 63,507 frames
  - C: 22 episodes, 61,194 frames
  - total: 68 episodes, 185,948 frames
- ACT-ABC-size-matched is balanced to the ACT-B full-training frame budget (`535,403` train frames):
  - A: 65 episodes, 178,481 frames
  - B: 74 episodes, 178,458 frames
  - C: 68 episodes, 178,517 frames
  - total: 207 episodes, 535,456 frames
  - mismatch from ACT-B train-frame budget: +53 frames
- Size-matched validation reuses the same stratified ABC validation split as ACT-ABC full, so validation comparisons are made on the same held-out A/B/C set.

Augmentation design:
- `act_ABC_aug*` and `act_ABC_size_matched_aug*` use the same augmentation parameters as ACT-B-aug:
  - brightness `0.12`
  - contrast `0.12`
  - saturation `0.08`
  - gaussian noise std `0.01`
  - probability `1.0`
- Architecture, optimizer, batch size, step count, chunk size, and VAE settings are unchanged from ACT-B unless the explicit augmentation or size-matched data-control condition applies.

Manual launcher defaults:
- `act_ABC_full`: GPU1
- `act_ABC_size_matched_full`: GPU2
- `act_ABC_aug_full`: GPU3
- `act_ABC_size_matched_aug_full`: GPU4
- `project/scripts/nohup_train_task2.sh` refuses to launch if `GPU_ID=0`.

Verification:
- Python compile checks passed for `project/scripts/train_act.py` and `project/scripts/prepare_task2_configs.py`.
- Bash syntax checks passed for all Task 2 launchers.
- YAML parse checks passed for all generated Task 2 configs and `task2_episode_splits.yaml`.
- GPU0 refusal was tested with `GPU_ID=0 bash project/scripts/nohup_train_task2.sh act_ABC_full`; it exited before `nohup` and printed the expected refusal message.

Not done:
- No visualizations were generated.
- No mini-trials were launched.
- No full-scale training was launched.

### 2026-06-05 07:15 UTC - Task 2 nohup harness and four mini-trials

Goal:
- Add executable Task 2 harness scripts that keep artifacts separated from Task 1.
- Run one bounded mini-trial harness to verify all four Task 2 experiment configs.
- Do not generate visualizations.
- Do not launch full-scale training.
- Do not use GPU0.

New scripts:
- `project/scripts/nohup_task2_mini_trials.sh`
  - launches four bounded 20-step mini-trials in parallel on physical GPUs 1-4.
  - writes logs under `project/logs/task2/mini_trials/<RUN_ID>/`.
  - writes outputs under `project/outputs/task2/<experiment>/mini_trials/<RUN_ID>/`.
  - logs tail-visible progress bars such as `PROGRESS act_ABC [########------------------------] ...`.
- `project/scripts/nohup_task2_full_parallel.sh`
  - optional manual full-training launcher for selected Task 2 full experiments.
  - defaults to all four Task 2 full experiments.
  - staggers launches with `STAGGER_SECONDS=300` by default.
  - delegates to `project/scripts/nohup_train_task2.sh`, which refuses GPU0.
- `project/scripts/tail_task2_latest_logs.sh`
  - tails the latest mini-trial logs or latest full-training log for a given Task 2 group.

Augmentation exposure policy:
- Size-matched augmentation keeps the Task 1 per-sample augmentation strength and matches Task 1 data quantity:
  - `act_ABC_size_matched_aug_full`: 535,456 train frames.
  - `act_B_aug_reference`: 535,403 train frames.
  - exposure ratio: `1.0001x`.
- Non-size-matched ABC augmentation keeps the Task 1 per-sample augmentation strength and lets total exposure rise with the larger ABC dataset:
  - `act_ABC_aug_full`: 1,609,097 train frames.
  - exposure ratio versus ACT-B-aug: `3.0054x`.
- This is recorded in `project/tables/task2_augmentation_exposure.csv`.

GPU check before mini-trials:
- Checked outside sandbox with `nvidia-smi`.
- GPU0 was active with the ACT-B augmentation process:
  - memory used: about `2951 MiB`
  - utilization at one check: `74-78%`
- GPUs 1-7 were idle.
- Mini-trial harness assigned:
  - `act_ABC`: physical GPU1
  - `act_ABC_size_matched`: physical GPU2
  - `act_ABC_aug`: physical GPU3
  - `act_ABC_size_matched_aug`: physical GPU4

Mini-trial run:
- Command:
  - `bash project/scripts/nohup_task2_mini_trials.sh`
- Run id:
  - `20260605_071512_task2_mini`
- All four mini-trials exited with status 0.
- All four self-checks passed.
- All four logs contain tail-visible progress bars.
- No `Traceback`, `Error`, `ERROR`, `Exception`, `failed`, or `Failed` strings were found in the mini-trial logs.
- GPU state after harness showed GPU1-3 idle and GPU4 with only transient utilization reporting; memory on GPUs 1-4 returned to `1 MiB`. GPU0 remained occupied by the separate ACT-B augmentation process.

Mini-trial artifacts:
- `project/tables/task2_mini_trial_results.csv`
- `project/logs/task2/mini_trials/20260605_071512_task2_mini/`
- `project/outputs/task2/act_ABC/mini_trials/20260605_071512_task2_mini/`
- `project/outputs/task2/act_ABC_size_matched/mini_trials/20260605_071512_task2_mini/`
- `project/outputs/task2/act_ABC_aug/mini_trials/20260605_071512_task2_mini/`
- `project/outputs/task2/act_ABC_size_matched_aug/mini_trials/20260605_071512_task2_mini/`

Final mini-trial metrics:
- `act_ABC`: final train Action L1 `0.7776`, final validation Action L1 `0.9289`.
- `act_ABC_size_matched`: final train Action L1 `0.7780`, final validation Action L1 `0.9292`.
- `act_ABC_aug`: final train Action L1 `0.7811`, final validation Action L1 `0.9163`.
- `act_ABC_size_matched_aug`: final train Action L1 `0.7807`, final validation Action L1 `0.9149`.

Manual full-training commands prepared:
```bash
GPU_ID=1 bash project/scripts/nohup_train_task2.sh act_ABC_full
GPU_ID=2 bash project/scripts/nohup_train_task2.sh act_ABC_size_matched_full
GPU_ID=3 bash project/scripts/nohup_train_task2.sh act_ABC_aug_full
GPU_ID=4 bash project/scripts/nohup_train_task2.sh act_ABC_size_matched_aug_full
```

Optional staggered all-job launcher:
```bash
STAGGER_SECONDS=300 bash project/scripts/nohup_task2_full_parallel.sh
```

Tail helpers:
```bash
project/scripts/tail_task2_latest_logs.sh mini
project/scripts/tail_task2_latest_logs.sh full act_ABC
```

### 2026-06-05 07:46 UTC - Five-job parallel IO and process check

Context:
- User launched five concurrent full-scale jobs:
  - `act_B_aug_full` on GPU0
  - `act_ABC_full` on GPU1
  - `act_ABC_size_matched_full` on GPU2
  - `act_ABC_aug_full` on GPU3
  - `act_ABC_size_matched_aug_full` on GPU4

GPU/process state:
- Confirmed with sandbox-external `nvidia-smi` and `ps`.
- Each active GPU process used about `2942 MiB`.
- Current PIDs:
  - GPU0 / ACT-B-aug: `3450887`
  - GPU1 / ACT-ABC: `70946`
  - GPU2 / ACT-ABC-size-matched: `1674751`
  - GPU3 / ACT-ABC-aug: `2163153`
  - GPU4 / ACT-ABC-size-matched-aug: `3093463`

IO / parallelism checks:
- `vmstat 1 5` showed CPU is busy but IO wait stayed at `0%`.
- Per-process open file descriptors were low relative to the soft limit:
  - ACT-B-aug: 97 open FDs
  - each Task 2 job: 79 open FDs
  - soft limit: 1024 open files
- A 10-second `/proc/<pid>/io` delta sample showed near-zero physical `read_bytes` and only tiny writes, suggesting data is being served mostly from page cache at that moment rather than blocked on disk.
- External disks are high but not full:
  - `/EXT_DISK`: about 90% used, about 2.0T available
  - `/SSD_DISK`: about 90% used, about 1.1T available

Log checks:
- Located active logs:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_B_aug/logs/20260605_063735_act_B_aug_full_gpu0.log`
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_ABC/logs/20260605_072243_act_ABC_full_gpu1.log`
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_ABC_size_matched/logs/20260605_072743_act_ABC_size_matched_full_gpu2.log`
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_ABC_aug/logs/20260605_073244_act_ABC_aug_full_gpu3.log`
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_ABC_size_matched_aug/logs/20260605_073744_act_ABC_size_matched_aug_full_gpu4.log`
- No `Traceback`, `Error`, `ERROR`, `Exception`, `failed`, `Failed`, or `Too many open files` strings appeared in the inspected log tails.
- Task 2 full jobs have not yet reached step 5000 validation, so `val_loss=NaN` in Task 2 logs is expected.

Current progress snapshot:
- ACT-B-aug: around step `12,800 / 100,000`.
- ACT-ABC: around step `3,900 / 100,000`.
- ACT-ABC-size-matched: around step `2,400 / 100,000`.
- ACT-ABC-aug: around step `1,400 / 100,000`.
- ACT-ABC-size-matched-aug: around step `600 / 100,000`.

Interpretation:
- There is no evidence of an IO correctness issue or read collision.
- There is no current evidence of disk IO wait or file descriptor exhaustion.
- Augmented jobs show slower and more variable step times than non-augmented jobs, which is consistent with extra CPU/tensor augmentation work rather than dataset corruption or read collision.
- Continue monitoring after Task 2 reaches step 5000, because validation and later checkpoints are the next IO-heavy phases.

Fix:
- Updated `project/scripts/tail_task2_latest_logs.sh` so it can find both the current active run layout (`$CALVIN_RUNS/act_ABC*/logs`) and the future Task 2 namespaced layout (`$CALVIN_RUNS/task2/act_ABC*/logs`).
- Added `full-all` mode:
```bash
project/scripts/tail_task2_latest_logs.sh full-all
```

### 2026-06-05 07:55 UTC - Step-5000 validation IO check

Context:
- User reported Task 2 reached the first validation point.
- Rechecked five concurrent full jobs around `act_ABC_full` step 5000.

Findings:
- `act_ABC_full` completed the step-5000 validation:
  - step 5000 train Action L1: `0.7228`
  - step 5000 validation Action L1: `0.5884`
  - step 5000 `step_s`: `27.89`
- Immediately after validation, `act_ABC_full` step time returned to normal:
  - step 5100 `step_s`: `0.183`
  - step 5200 `step_s`: `0.176`
  - step 5300 `step_s`: `0.178`
  - step 5400 `step_s`: `0.167`
- `vmstat 1 5` again showed IO wait at `0%`.
- Open file descriptors remained low:
  - ACT-B-aug: 97
  - ACT-ABC: 97
  - ACT-ABC-size-matched: 79
  - ACT-ABC-aug: 79
  - ACT-ABC-size-matched-aug: 79
- Disk free space unchanged at a safe level for current runs:
  - `/EXT_DISK`: about 2.0T free
  - `/SSD_DISK`: about 1.1T free
- No errors were observed in the inspected log tails.

Interpretation:
- The 27.9-second spike at step 5000 is expected validation overhead (`max_val_batches=64`), not a persistent IO contention problem.
- There is still no evidence of data read collision, file descriptor exhaustion, or disk IO wait.
- Continue running. Recheck near the first checkpoint boundary (`step 10000`) because checkpoint writing is the next expected IO-heavy event.

### 2026-06-05 08:20 UTC - Step-10000 checkpoint IO check

Context:
- User reported a step-10000 checkpoint boundary.
- Rechecked the five concurrent full jobs.

Current progress:
- `act_B_aug_full`: around step `15,800`.
- `act_ABC_full`: around step `10,200`.
- `act_ABC_size_matched_full`: around step `6,100`.
- `act_ABC_aug_full`: around step `4,600`.
- `act_ABC_size_matched_aug_full`: around step `3,700`.

Checkpoint findings:
- `act_ABC_full` reached step 10000 and wrote a complete checkpoint:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_ABC/20260605_072243_act_ABC_full_gpu1/checkpoints/step_00010000/`
  - expected files are present: config, model safetensors, preprocessor, postprocessor, normalizer/unnormalizer safetensors, and `training_state.pt`.
  - checkpoint directory size: `592M`.
- `act_B_aug_full` also has a complete step-10000 checkpoint:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_B_aug/20260605_063735_act_B_aug_full_gpu0/checkpoints/step_00010000/`
  - checkpoint directory size: `592M`.
- The other three Task 2 jobs had not yet reached step 10000 at this check, so no checkpoint is expected for them yet.

Step-10000 metric / timing snapshot:
- `act_ABC_full` step 10000:
  - train Action L1: `0.6765`
  - validation Action L1: `0.5698`
  - `step_s`: `26.36`
- Training resumed normally after the checkpoint/validation step:
  - step 10100 `step_s`: `0.161`
  - step 10200 `step_s`: `0.144`

IO / parallelism checks:
- `vmstat 1 5` again showed IO wait at `0%`.
- Open file descriptors remained low:
  - ACT-B-aug: 97
  - ACT-ABC: 97
  - ACT-ABC-size-matched: 97
  - ACT-ABC-aug: 79
  - ACT-ABC-size-matched-aug: 79
- No `Traceback`, `Error`, `ERROR`, `Exception`, `failed`, `Failed`, or `Too many open files` strings appeared in the inspected log tails.

Interpretation:
- The first Task 2 checkpoint write completed cleanly.
- The `26.36s` step time at step 10000 is expected because validation and checkpoint saving coincide.
- Since normal step time resumed immediately afterward and IO wait stayed at `0%`, there is still no evidence of problematic IO collision.
- Recheck as the remaining Task 2 jobs reach step 10000, especially the augmented runs because their step times are slower and more variable.

### 2026-06-06 UTC - Task 2 full training completion and model selection

Successes:
- User reported the four Task 2 full runs completed.
- Confirmed all GPUs returned to idle:
  - GPUs 0-7 each showed `1 MiB` used and `0%` utilization after completion.
- Parsed full-run metrics for:
  - `act_ABC`
  - `act_ABC_size_matched`
  - `act_ABC_aug`
  - `act_ABC_size_matched_aug`
  - plus Task 1 references `act_B` and `act_B_aug`.
- All completed runs have:
  - `100,000` metrics rows.
  - final checkpoint present.
  - self-check passed.
  - no inspected log errors.
- Wrote summary tables:
  - `project/tables/task2_full_training_summary.csv`
  - `project/tables/full_training_summary_with_B_aug.csv`
  - `project/tables/task2_checkpoint_candidates.csv`
  - `project/tables/model_selection_checkpoints.csv`
  - `project/tables/task2_internal_effects.csv`
- Updated `project/tables/main_training_results.csv` with full Task 2 rows and B-aug full-training row.

Task 2 validation results on the shared ABC validation split:

| Experiment | Final Val Action L1 | Best Val Step | Best Val Action L1 | Best Step Has Checkpoint | Selected Available Checkpoint |
|---|---:|---:|---:|---|---|
| `act_ABC` | `0.5810` | 25,000 | `0.5552` | no | step 30,000 (`0.5565`) |
| `act_ABC_size_matched` | `0.6165` | 15,000 | `0.5653` | no | step 10,000 (`0.5736`) |
| `act_ABC_aug` | `0.5872` | 25,000 | `0.5559` | no | step 30,000 (`0.5588`) |
| `act_ABC_size_matched_aug` | `0.6304` | 15,000 | `0.5669` | no | step 10,000 (`0.5714`) |

Selected Task 2 checkpoints:
- `act_ABC`:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_ABC/20260605_072243_act_ABC_full_gpu1/checkpoints/step_00030000`
- `act_ABC_size_matched`:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_ABC_size_matched/20260605_072743_act_ABC_size_matched_full_gpu2/checkpoints/step_00010000`
- `act_ABC_aug`:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_ABC_aug/20260605_073244_act_ABC_aug_full_gpu3/checkpoints/step_00030000`
- `act_ABC_size_matched_aug`:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_ABC_size_matched_aug/20260605_073744_act_ABC_size_matched_aug_full_gpu4/checkpoints/step_00010000`

Checkpoint-selection rule:
- Validation runs every 5k steps, but checkpoints are saved every 10k steps.
- If the best validation step does not have a checkpoint, compare the previous and next saved checkpoints and select the one with lower validation Action L1.
- The adjacent checkpoint comparison is recorded in `project/tables/model_selection_checkpoints.csv`.

Task 2 internal comparisons:
- Full ABC vs size-matched ABC:
  - `act_ABC` best val Action L1 `0.5552`
  - `act_ABC_size_matched` best val Action L1 `0.5653`
  - full ABC is better by `0.0101` on the shared ABC validation split.
- Full ABC augmentation:
  - `act_ABC_aug` best val Action L1 `0.5559`
  - `act_ABC` best val Action L1 `0.5552`
  - augmentation is slightly worse by `0.0007`; no meaningful benefit here.
- Size-matched augmentation:
  - `act_ABC_size_matched_aug` best val Action L1 `0.5669`
  - `act_ABC_size_matched` best val Action L1 `0.5653`
  - augmentation is slightly worse by `0.0016`; no meaningful benefit here.
- Under augmentation, full ABC remains better than size-matched:
  - `act_ABC_aug` is better than `act_ABC_size_matched_aug` by `0.0109`.

Task 1 augmentation note:
- `act_B_aug` did not improve the Task 1 B validation metric:
  - `act_B` best val Action L1: `0.5404` at step 10,000.
  - `act_B_aug` best val Action L1: `0.5409` at step 15,000.
  - best-step delta `B_aug - B = +0.0005`, so augmentation is effectively neutral/slightly worse.
  - final validation is also worse for B-aug: `0.6486` vs `0.6229`.
- Since Task 1 B validation and Task 2 ABC validation are different distributions, absolute B-vs-ABC validation values should not be used as the final generalization claim. The proper cross-environment claim should come from zero-shot D evaluation.

Interpretation:
- For Task 2 supervised validation on A/B/C, full ABC training is better than size-matched ABC, suggesting that the extra data volume helps beyond the balanced data-control condition.
- The current augmentation settings do not show a clear benefit in Task 1 or Task 2.
- All Task 2 models overfit after early validation minima; selected available checkpoints should be used for downstream D evaluation alongside final checkpoints.

### 2026-06-06 UTC - Task 2 per-model visualization pass

Decision:
- User requested Task 2 visualization following the draft style, but asked not to draw cross-model comparison figures yet.
- To keep the structure consistent with Task 1, every Task 2 model now has its own figure folder and table folder.

Generated script:
- `project/scripts/analyze_task2_visuals.py`

Generated per-model figure folders:
- `project/figures/task2/act_ABC/`
- `project/figures/task2/act_ABC_size_matched/`
- `project/figures/task2/act_ABC_aug/`
- `project/figures/task2/act_ABC_size_matched_aug/`

Generated per-model table folders:
- `project/tables/task2/act_ABC/`
- `project/tables/task2/act_ABC_size_matched/`
- `project/tables/task2/act_ABC_aug/`
- `project/tables/task2/act_ABC_size_matched_aug/`

Each model folder contains:
- `loss_curve.png`
- `train_val_gap.png`
- `checkpoint_selection.png`
- `step_time_profile.png`
- `dataset_split.png`

Supporting tables per model:
- `checkpoint_selection.csv`
- `dataset_split.csv`
- `full_training_summary.csv`
- `model_selection.csv`
- `step_time_profile.csv`
- `train_l1_smoothed.csv`
- `train_val_gap.csv`

Manifest:
- `project/tables/task2_visualization_manifest.csv`
- 20 figure entries total.
- Every entry is marked as a single-model diagnostic, not a cross-model comparison.

Figure layout / title QA:
- Used `constrained_layout=True`, `bbox_inches="tight"`, `pad_inches=0.28`, and extra title padding to avoid the title/header clipping seen earlier in Task 1.
- Checked all PNG dimensions; every Task 2 figure is at least `1561 px` wide.
- Visually inspected representative figures:
  - `project/figures/task2/act_ABC/loss_curve.png`
  - `project/figures/task2/act_ABC_size_matched_aug/checkpoint_selection.png`
  - `project/figures/task2/act_ABC_size_matched_aug/dataset_split.png`
- The inspected titles, legends, and headers were not clipped.

Current interpretation:
- These figures are diagnostic artifacts for individual runs only.
- Cross-model comparison plots were deferred in this first pass and later generated in the Task 2 requested comparison pass.

Follow-up correction:
- User correctly pointed out that this first pass was too small compared with Task 1.
- Root cause: the first Task 2 pass only covered training diagnostics and dataset split, but Task 1 also included dataset visual samples, visual statistics, action distribution, action smoothness, chunk diagnostics, gripper diagnostics, task frequency, and representative trajectory strips.
- Decision: keep the "no cross-model comparison figure yet" constraint, but expand every Task 2 model folder with Task1-style single-model dataset/action diagnostics.

### 2026-06-06 UTC - Task 2 extended per-model visualization pass

Generated script:
- `project/scripts/analyze_task2_extended_visuals.py`

Execution:
- Ran with `--visual-sample-count 1200 --image-sample-count 6`.
- No GPU was used.
- No training was launched.
- The script reads the converted LeRobot ABC parquet data for action diagnostics and samples video frames for visual diagnostics.
- For augmented models, image diagnostics include raw-vs-augmented samples/statistics; action diagnostics are computed from the unchanged action labels.

Generated figure counts:
- `project/figures/task2/act_ABC/`: 17 PNGs
- `project/figures/task2/act_ABC_size_matched/`: 17 PNGs
- `project/figures/task2/act_ABC_aug/`: 18 PNGs
- `project/figures/task2/act_ABC_size_matched_aug/`: 18 PNGs
- Total Task 2 single-model figures: 70 PNGs.

Each non-augmented model folder now contains:
- `loss_curve.png`
- `train_val_gap.png`
- `checkpoint_selection.png`
- `step_time_profile.png`
- `dataset_split.png`
- `dataset_samples.png`
- `visual_color_profile.png`
- `brightness_contrast_hist.png`
- `action_distribution.png`
- `action_violin.png`
- `action_smoothness.png`
- `gripper_diagnostics.png`
- `gripper_timeline.png`
- `chunk_baseline.png`
- `action_delta_heatmap.png`
- `task_frequency.png`
- `representative_trajectory_strip.png`

Augmented model folders contain all of the above plus:
- `augmentation_verification.png`

Generated / updated tables:
- Per-model table folders under `project/tables/task2/<experiment>/`.
- Each folder now contains 19 CSV tables covering:
  - training diagnostics
  - checkpoint selection
  - selected episode summary
  - task/environment frequency
  - visual stats
  - action stats and summary
  - action smoothness
  - gripper diagnostics
  - chunk baseline
  - representative trajectory
- Updated manifests:
  - `project/tables/task2_visualization_manifest.csv`
  - `project/tables/task2_extended_visualization_manifest.csv`

Figure QA:
- Count check passed: 70 total Task 2 figures.
- Dimension check passed for all figures.
- Visually inspected:
  - `project/figures/task2/act_ABC/dataset_samples.png`
  - `project/figures/task2/act_ABC_size_matched_aug/dataset_samples.png`
  - `project/figures/task2/act_ABC_size_matched_aug/visual_color_profile.png`
  - `project/figures/task2/act_ABC_size_matched_aug/representative_trajectory_strip.png`
- The inspected figure labels, titles, legends, and headers were not clipped.

Current interpretation:
- Task 2 now has Task1-style per-model visualization coverage.
- Cross-model comparison plots were generated afterward in the Task 2 requested comparison pass.

### 2026-06-06 UTC - Task 2 requested comparison pass

Decision:
- User requested the comparison section, including:
  - `ABC` vs `ABC_aug`
  - `ABC` vs `ABC_size_matched`
  - `ABC` vs `B`
  - `ABC_size_matched` vs `B`
  - `ABC_size_matched_aug` vs `B_aug`
- User also requested metrics beyond losses.
- The comparison section therefore includes loss/overfitting/convergence plus data scale, action smoothness, chunk boundary, gripper, and visual diagnostics.

Generated script:
- `project/scripts/compare_task2_results.py`

Generated figures:
- `project/figures/task2/comparisons/best_final_val_action_l1.png`
- `project/figures/task2/comparisons/overfit_convergence_summary.png`
- `project/figures/task2/comparisons/requested_pairwise_loss_curves.png`
- `project/figures/task2/comparisons/data_action_visual_summary.png`
- `project/figures/task2/comparisons/pairwise_relative_delta_heatmap.png`

Generated tables:
- `project/tables/task2/comparisons/comparison_model_summary.csv`
- `project/tables/task2/comparisons/pairwise_comparison_metrics.csv`
- `project/tables/task2/comparisons/pairwise_key_findings.csv`
- `project/tables/task2/comparisons/pairwise_loss_overfit_convergence.csv`
- LaTeX versions:
  - `project/tables/task2/comparisons/comparison_model_summary.tex`
  - `project/tables/task2/comparisons/pairwise_key_findings.tex`
  - `project/tables/task2/comparisons/pairwise_loss_overfit_convergence.tex`
- Root pairwise effects table refreshed:
  - `project/tables/task2_pairwise_effects.csv`

Comparison design:
- For each requested pair, the table reports:
  - best validation Action L1 delta
  - selected-checkpoint validation delta
  - final-minus-best overfitting gap delta
  - train frame ratio
  - mean step delta L2 delta
  - chunk q90 boundary jump delta
  - train-input static brightness delta
- `validation_comparability` is explicitly recorded.
- Same-ABC-validation comparisons can be interpreted directly for supervised validation.
- B-vs-ABC loss comparisons are marked as `different_validation_split`; these are diagnostics only and should not be used as final generalization claims.

Key findings:
- `ABC` vs `ABC_aug` on the same ABC validation split:
  - best validation Action L1 delta `ABC - ABC_aug = -0.0007`.
  - selected-checkpoint validation delta `-0.0023`.
  - overfitting gap delta `-0.0054`.
  - interpretation: full `ABC` remains slightly better than `ABC_aug`; augmentation still does not show a clear supervised-validation gain.
- `ABC` vs `ABC_size_matched` on the same ABC validation split:
  - best validation Action L1 delta `-0.0101`.
  - selected-checkpoint validation delta `-0.0171`.
  - train-frame ratio `3.005x`.
  - interpretation: the full ABC run is better than size-matched ABC on ABC validation, consistent with a data-volume benefit.
- `ABC` vs `B`:
  - best validation Action L1 delta is `+0.0148`, but validation splits differ.
  - overfitting gap is much smaller for `ABC` than `B` (`-0.0566` delta).
  - ABC train frames are `3.005x` B.
  - action step delta and chunk q90 boundary jump are lower for ABC diagnostics than B diagnostics.
  - visual brightness differs substantially; ABC train-input static brightness is about `0.142` lower than B.
- `ABC_size_matched` vs `B`:
  - frame budgets are matched (`1.0001x`).
  - best validation Action L1 delta is `+0.0249`, but validation splits differ.
  - overfitting gap is lower for size-matched ABC (`-0.0313` delta).
  - visual brightness still differs strongly from B, showing that the size-matched comparison controls volume but not visual distribution.
- `ABC_size_matched_aug` vs `B_aug`:
  - frame budgets are matched (`1.0001x`).
  - best validation Action L1 delta is `+0.0259`, but validation splits differ.
  - overfitting gap is lower for size-matched ABC augmentation (`-0.0441` delta).
  - train-input static brightness remains much lower than B augmentation (`-0.1365` delta).

Figure QA:
- Checked comparison figure dimensions.
- Visually inspected:
  - `requested_pairwise_loss_curves.png`
  - `pairwise_relative_delta_heatmap.png`
  - `data_action_visual_summary.png`
  - `overfit_convergence_summary.png`
- Titles, legends, and axes were not clipped.

Current interpretation:
- Task 2 comparison is now complete for the requested pairs.
- Direct supervised-validation conclusion: `act_ABC` is best among the compared ABC-validation models; current augmentation is neutral/slightly harmful on supervised validation.
- Diagnostic B-vs-ABC conclusion: ABC/size-matched ABC show lower overfitting gaps and smoother action/chunk diagnostics, but D zero-shot evaluation is required for any generalization claim.

## Task 3 Success-Rate Data Preparation Log

### 2026-06-06 UTC - Language-aligned CALVIN B/ABC dataset builder

Context:
- User confirmed the existing non-language ACT checkpoints should not be patched.
- New models must be retrained for official CALVIN success-rate evaluation:
  - `ACT-Lang-B`
  - `ACT-Lang-ABC`
- The existing processed datasets (`local/calvin_B`, `local/calvin_ABC`) are not sufficient because they were converted with environment-level tasks and do not contain language-goal inputs.

Decision:
- Build new language-aligned LeRobot datasets from raw CALVIN annotations.
- One CALVIN language annotation segment becomes one LeRobot episode.
- This ensures ACT action chunks never cross language-goal boundaries.
- Add `observation.language_embedding` from CALVIN `auto_lang_ann.npy` as a 384D float32 feature.
- Save the raw language instruction as the LeRobot `task` string for traceability.
- Keep train episodes before val episodes so downstream configs can use simple ranges.

Generated script:
- `project/scripts/prepare_calvin_language_dataset.py`
- Launcher script:
  - `project/scripts/run_prepare_calvin_language_datasets.sh`
  - `project/scripts/nohup_prepare_calvin_language_datasets.sh`

Success-rate compliance checks built into the script:
- verifies raw CALVIN language annotation keys: `ann`, `task`, `emb`, `indx`;
- verifies selected language segments fall fully inside the requested scene split;
- verifies every segment is contained inside one raw CALVIN episode;
- verifies raw frame keys/shapes/dtypes:
  - `rgb_static`: `(200, 200, 3)` uint8
  - `rgb_gripper`: `(84, 84, 3)` uint8
  - `robot_obs`: `(15,)` finite float
  - `rel_actions`: `(7,)` finite float
- verifies `observation.language_embedding` is finite float32 shape `(384,)`;
- writes one LeRobot episode per language segment;
- readback-checks LeRobot `action` chunks with `chunk_size=100`;
- verifies `action_is_pad` is correct at sampled episode starts/ends;
- writes manifest, split table, readback checks, and summary JSON.

Dry-run results:
- B:
  - output target: `/EXT_DISK/users/zengzixuan/processed-calvin/calvin_lang_B`
  - repo id: `local/calvin_lang_B`
  - language segment episodes: `6115`
  - frames: `367096`
  - split: train `5503`, val `612`
  - unique task names: `34`
  - unique language texts: `389`
  - segment length min/median/max: `34 / 65 / 65`
  - all segments are shorter than `chunk_size=100`, so padding is required and expected.
- ABC:
  - output target: `/EXT_DISK/users/zengzixuan/processed-calvin/calvin_lang_ABC`
  - repo id: `local/calvin_lang_ABC`
  - language segment episodes: `17870`
  - frames: `1071743`
  - split: train `16083`, val `1787`
  - environment counts: A `6089`, B `6115`, C `5666`
  - unique task names: `34`
  - unique language texts: `389`
  - segment length min/median/max: `34 / 65 / 65`
  - all segments are shorter than `chunk_size=100`, so padding is required and expected.

Validation performed:
- `py_compile` passed for `project/scripts/prepare_calvin_language_dataset.py`.
- Dry-run summaries were generated:
  - `project/tables/calvin_lang_B_dry_run_summary.json`
  - `project/tables/calvin_lang_ABC_dry_run_summary.json`
- A small real smoke dataset was generated under `/tmp` with `--max-segments 4 --no-videos`.
- Smoke readback passed:
  - generated one episode per language segment;
  - checked 8 episode-edge positions;
  - verified language embedding shape `(384,)`;
  - verified action chunk shape `(100, 7)`;
  - verified `action_is_pad` masks padded tail actions rather than crossing into another language segment.

Formal generation commands, manual:

```bash
bash project/scripts/run_prepare_calvin_language_datasets.sh
```

Launcher options:
- `DRY_RUN_ONLY=1 bash project/scripts/run_prepare_calvin_language_datasets.sh`
- `MAX_SEGMENTS=20 bash project/scripts/run_prepare_calvin_language_datasets.sh`
- `OVERWRITE=0 bash project/scripts/run_prepare_calvin_language_datasets.sh`

Launcher verification:
- `DRY_RUN_ONLY=1 MAX_SEGMENTS=4 READBACK_EPISODES=4 bash project/scripts/run_prepare_calvin_language_datasets.sh` passed.
- This checked both B and ABC dry-run branches without generating full datasets.
- `project/scripts/nohup_prepare_calvin_language_datasets.sh` was added for manual background execution.
- The nohup launcher defaults:
  - `CLEAN=1`: removes previous `calvin_lang_B`, `calvin_lang_ABC`, and generated final language tables before launch.
  - `FAST_STORAGE=1`: uses `--no-videos` image storage to avoid encoding tens of thousands of tiny videos.
  - prints stable `PROGRESS calvin_lang_* [####----]` lines for `tail -f`.
- Direct dry-run of the generated `run.sh` passed with `CLEAN=0 DRY_RUN_ONLY=1 MAX_SEGMENTS=4 READBACK_EPISODES=4`.
- The Codex tool environment did not keep the nohup dry-run child alive after launch, but the generated `run.sh` itself executed successfully; user terminal execution should behave like prior full-training nohup scripts.

Expected outputs:
- `local/calvin_lang_B` at `/EXT_DISK/users/zengzixuan/processed-calvin/calvin_lang_B`
- `local/calvin_lang_ABC` at `/EXT_DISK/users/zengzixuan/processed-calvin/calvin_lang_ABC`
- `project/tables/calvin_lang_B_manifest.csv`
- `project/tables/calvin_lang_B_episode_splits.csv`
- `project/tables/calvin_lang_B_readback_checks.csv`
- `project/tables/calvin_lang_B_summary.json`
- `project/tables/calvin_lang_ABC_manifest.csv`
- `project/tables/calvin_lang_ABC_episode_splits.csv`
- `project/tables/calvin_lang_ABC_readback_checks.csv`
- `project/tables/calvin_lang_ABC_summary.json`

Important implication:
- Because all CALVIN language segments are at most 65 frames, keeping `chunk_size=100` is acceptable only with padding-aware loss.
- For rollout, `n_action_steps=100` should be reconsidered for `ACT-Lang-*`; otherwise the model may execute untrained padded tail positions.

### 2026-06-06 UTC - ACT-Lang-B full-training launcher prepared

Context:
- User reported that the B language-aligned dataset generation was nearly complete and requested a nohup training script aligned with previous ACT-B training.

Generated files:
- `project/configs/act_lang_B_full.yaml`
- `project/scripts/nohup_train_act_lang_B.sh`

Strict alignment with ACT-B:
- The following fields match `project/configs/act_B_full.yaml` exactly:
  - ResNet18 ImageNet backbone
  - `n_obs_steps=1`
  - `chunk_size=100`
  - `n_action_steps=100`
  - `dim_model=512`
  - `n_heads=8`
  - `dim_feedforward=3200`
  - encoder/decoder/VAE layer counts
  - `dropout=0.1`
  - `kl_weight=10.0`
  - optimizer learning rates and weight decay
  - `batch_size=32`
  - `steps=100000`
  - `num_workers=4`
  - log/validation/save frequencies
  - `grad_clip_norm=10.0`
- Intended differences:
  - dataset: `local/calvin_lang_B`
  - train/val episodes: `0:5503` and `5503:6115`
  - policy type: `act_lang`
  - added `observation.language_embedding` branch with dimension `384`.

Safety guard:
- The launcher refuses to run if `project/scripts/train_act_lang.py` does not exist.
- This is intentional: `project/scripts/train_act.py` builds plain ACT and would ignore `observation.language_embedding`, producing a non-language model that is not official-success-rate compatible.
- The launcher also checks that the full `calvin_lang_B` dataset exists and contains `observation.language_embedding` shape `(384,)`, at least `6115` episodes, and at least `367096` frames before launching.

Verification:
- `bash -n project/scripts/nohup_train_act_lang_B.sh` passed.
- `py_compile project/scripts/prepare_calvin_language_dataset.py` passed.
- Config comparison against `act_B_full.yaml` showed all intended shared architecture/training hyperparameters are identical.

Manual launch command after ACT-Lang implementation and full B dataset are ready:

```bash
GPU_ID=0 bash project/scripts/nohup_train_act_lang_B.sh
```

### 2026-06-09 UTC - Extended success-rate visualizations and discussion

Context:
- User asked whether the success-rate results can support more visualizations and deeper discussion.
- Added a reproducible analysis script:
  - `project/scripts/analyze_task3_success_rate_extensions.py`

Generated extended figures:
- `project/figures/task3_success_rate_D_extended/long_horizon_attrition_curve.png`
- `project/figures/task3_success_rate_D_extended/successful_prefix_distribution.png`
- `project/figures/task3_success_rate_D_extended/per_task_success_rate_heatmap.png`
- `project/figures/task3_success_rate_D_extended/task_group_success_rate.png`
- `project/figures/task3_success_rate_D_extended/first_failure_position_distribution.png`
- `project/figures/task3_success_rate_D_extended/first_failure_task_group.png`
- `project/figures/task3_success_rate_D_extended/steps_to_success_distribution.png`
- `project/figures/task3_success_rate_D_extended/chunk_metrics_success_vs_failure.png`

Generated extended tables:
- `project/tables/task3_success_rate_D_extended/success_rate_improvements.csv`
- `project/tables/task3_success_rate_D_extended/success_rate_improvements.tex`
- `project/tables/task3_success_rate_D_extended/top_task_improvements_ABC200k_vs_B.csv`
- `project/tables/task3_success_rate_D_extended/top_task_improvements_ABC200k_vs_B.tex`
- `project/tables/task3_success_rate_D_extended/task_group_success_rate.csv`
- `project/tables/task3_success_rate_D_extended/task_group_success_rate.tex`
- `project/tables/task3_success_rate_D_extended/steps_to_success_summary.csv`
- `project/tables/task3_success_rate_D_extended/steps_to_success_summary.tex`
- `project/tables/task3_success_rate_D_extended/first_failure_taxonomy.csv`
- `project/tables/task3_success_rate_D_extended/chunk_metrics_success_vs_failure.csv`
- `project/tables/task3_success_rate_D_extended/manifest.csv`

Key additional findings:
- Absolute improvement over ACT-Lang-B:
  - ACT-Lang-ABC-size-matched: average chain length `+0.887`, 1/5 SR `+33.8` percentage points, 5/5 SR `+4.2` points, subtask SR `+30.0` points.
  - ACT-Lang-ABC 200k: average chain length `+1.266`, 1/5 SR `+43.7` points, 5/5 SR `+8.6` points, subtask SR `+37.5` points.
- Continuing ABC from size-matched/shorter training to the 200k full ABC model still adds:
  - average chain length `+0.379`
  - 1/5 SR `+9.9` points
  - 5/5 SR `+4.4` points
  - subtask SR `+7.5` points
- Successful-prefix distribution shows the main qualitative shift:
  - ACT-Lang-B fails at the first subtask in `73.2%` of sequences and solves all five in `0.0%`.
  - ACT-Lang-ABC-size-matched fails at the first subtask in `39.4%` and solves all five in `4.2%`.
  - ACT-Lang-ABC 200k fails at the first subtask in `29.5%` and solves all five in `8.6%`.
- Task-group success shows where environment diversity helps most:
  - Light tasks: B `0.0%`, ABC-size-matched `50.6%`, ABC 200k `77.0%`.
  - Slider tasks: B `11.3%`, ABC-size-matched `86.2%`, ABC 200k `83.3%`.
  - Drawer tasks: B `63.5%`, ABC-size-matched `92.6%`, ABC 200k `93.0%`.
  - Lift, rotate, and stack remain comparatively difficult even for ABC 200k.
- Largest per-task gains from B to ABC 200k include:
  - `turn_on_lightbulb`: `0.0%` to `86.1%`
  - `turn_off_led`: `0.0%` to `85.4%`
  - `turn_on_led`: `0.0%` to `75.7%`
  - `move_slider_right`: `17.2%` to `91.8%`
  - `move_slider_left`: `3.4%` to `71.6%`
- Step efficiency among successful subtasks:
  - B: median `66`, mean `107.3`, q90 `235`
  - ABC-size-matched: median `73`, mean `99.1`, q90 `207`
  - ABC 200k: median `71`, mean `96.9`, q90 `199.4`
  - Interpretation: ABC 200k solves many more subtasks and does not pay a step-efficiency penalty on successes.
- Chunk diagnostics by outcome:
  - Successful subtasks have lower mean chunk boundary jumps than failed subtasks for all three models.
  - For ABC 200k, success boundary jump mean is `0.3052` vs failure `0.4306`.
  - This supports a report claim that robust ACT chunking under visual shift is not merely "smoother"; stable chunk transitions correlate with success, while failed rollouts often show larger chunk discontinuities.

Report discussion angle:
- The headline should not only be "ABC has higher success rate." The stronger story is:
  1. B often fails before any long-horizon compounding can begin.
  2. ABC-size-matched moves many sequences past the first and second subtasks, showing environment diversity helps zero-shot initiation and short-horizon generalization.
  3. ABC 200k further improves chain length and 5/5 success, showing the full ABC data volume/training budget is needed for long-horizon robustness.
  4. The remaining gap is concentrated in rotation, lifting, and stacking tasks, suggesting manipulation precision and contact-rich primitives remain failure modes under D visual shift.

### 2026-06-09 UTC - Latest Task 3 status pointer

- CALVIN D ACT-Lang success-rate evaluation is complete.
- Full result record is above under `2026-06-09 UTC - Task 3 ACT-Lang official success-rate evaluation completed`.
- Main result table:
  - `project/tables/task3_success_rate_D/success_rate_D_summary.csv`
  - `project/tables/task3_success_rate_D/success_rate_D_summary.tex`
- Main figures:
  - `project/figures/task3_success_rate_D/success_rate_chain_D.png`
  - `project/figures/task3_success_rate_D/avg_successful_sequence_length_D.png`
  - `project/figures/task3_success_rate_D/rollout_chunk_diagnostics_D.png`

### 2026-06-09 UTC - Task 3 ACT-Lang official success-rate evaluation completed

Context:
- User reported that the CALVIN D success-rate run should be complete and asked to verify results.
- This success-rate evaluation is separate from the earlier non-language ACT offline D action-error evaluation.
- The old six non-language ACT checkpoints remain part of Task 3 for offline action error, visual shift, and chunking analysis; official success rate is computed only for language-conditioned models.

Run status:
- No active `eval_act_lang_success_rate_D.py` process was found.
- Latest run:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/task3_success_rate_D/20260608_063854_act_lang_success_rate_D_gpu0_n1000_a25`
- Log:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/task3_success_rate_D/logs/20260608_063854_act_lang_success_rate_D_gpu0_n1000_a25.log`
- The log ended with `event: done`.
- Evaluation used GPU0, 1000 CALVIN D long-horizon sequences, `ep_len=360`, and `rollout_action_steps=25`.

Success-rate model set:
- `ACT-Lang-B 100k`
- `ACT-Lang-ABC Size-Matched 100k`
- `ACT-Lang-ABC 200k`
- The under-converged `ACT-Lang-ABC 100k` is retained as a training diagnostic but excluded from official success-rate reporting.

Artifacts:
- Summary table:
  - `project/tables/task3_success_rate_D/success_rate_D_summary.csv`
  - `project/tables/task3_success_rate_D/success_rate_D_summary.tex`
- Sequence-level table:
  - `project/tables/task3_success_rate_D/success_rate_D_sequences.csv`
- Subtask-level table with rollout chunk diagnostics:
  - `project/tables/task3_success_rate_D/success_rate_D_subtasks.csv`
- Per-task success breakdown:
  - `project/tables/task3_success_rate_D/success_rate_D_task_breakdown.csv`
- Chunk diagnostics LaTeX table:
  - `project/tables/task3_success_rate_D/rollout_chunk_diagnostics_D.tex`
- Figures:
  - `project/figures/task3_success_rate_D/success_rate_chain_D.png`
  - `project/figures/task3_success_rate_D/avg_successful_sequence_length_D.png`
  - `project/figures/task3_success_rate_D/rollout_chunk_diagnostics_D.png`

Main results:

| Model | Avg successful length | 1/5 SR | 2/5 SR | 3/5 SR | 4/5 SR | 5/5 SR | Single-subtask SR |
|---|---:|---:|---:|---:|---:|---:|---:|
| ACT-Lang-B 100k | 0.361 | 26.8% | 7.1% | 2.0% | 0.2% | 0.0% | 26.5% |
| ACT-Lang-ABC Size-Matched 100k | 1.248 | 60.6% | 34.1% | 17.3% | 8.6% | 4.2% | 56.6% |
| ACT-Lang-ABC 200k | 1.627 | 70.5% | 44.4% | 25.1% | 14.1% | 8.6% | 64.0% |

Interpretation:
- Multi-environment language-conditioned training strongly improves zero-shot D success rate over single-environment B training.
- Size-matched ABC already improves substantially over B, so environment diversity matters beyond raw data volume.
- Continuing ACT-Lang-ABC from 100k to 200k was justified: the 200k model is the strongest official success-rate model.
- The 5/5 long-horizon success rate remains low, showing that cross-environment compounding errors are still severe even when 1/5 and 2/5 success improve.

Action chunking notes:
- Main rollout used `rollout_action_steps=25` rather than executing all 100 predicted actions before replanning.
- This is a deliberate padding-aware choice: ACT-Lang was trained with `chunk_size=100`, but CALVIN language segments are much shorter and the padded tail is not supervised.
- Rollout chunk diagnostics from attempted subtasks:
  - ACT-Lang-B 100k: mean step delta `0.0522`, mean boundary jump `0.3548`, mean action norm `0.3324`.
  - ACT-Lang-ABC Size-Matched 100k: mean step delta `0.0609`, mean boundary jump `0.3856`, mean action norm `0.4223`.
  - ACT-Lang-ABC 200k: mean step delta `0.0564`, mean boundary jump `0.3514`, mean action norm `0.3997`.
- ACT-Lang-ABC 200k has the best success rate while keeping chunk boundary jump comparable to or slightly below ACT-Lang-B, supporting the claim that multi-environment training improves cross-environment robustness without relying on more abrupt chunk transitions.

Failure/task observations:
- ACT-Lang-B fails many light-switch tasks completely in this D rollout sample.
- ACT-Lang-ABC 200k remains weak on rotation and stacking-style tasks, e.g. `stack_block` and some block rotations.
- Drawer and placement subtasks are much stronger for the ABC models, especially `open_drawer`, `close_drawer`, and `place_in_drawer`.

### 2026-06-07 UTC - ACT-Lang visualization script and nohup launcher prepared

Context:
- User requested ACT-Lang visualizations in two separate folders.
- Requirement: avoid exact duplicates where possible, but regenerate baseline-style diagnostics because both dataset and model family changed.
- The previous direct run was slow and did not provide enough tail-visible progress, so a nohup launcher with progress output was added.

Implementation:
- Added `project/scripts/analyze_act_lang_visuals.py`.
- Added `project/scripts/nohup_analyze_act_lang_visuals.sh`.
- The analysis script now prints tail-visible lines:
  - `PROGRESS act_lang_visuals [########------------------------] ...`
- Slow non-model sections were changed to read parquet columns directly instead of decoding images:
  - action/task diagnostics read `action`, `task_index`
  - language embedding PCA reads `observation.language_embedding`, `task_index`
- Image decoding and GPU forward are used only where actually needed:
  - correct / zero / wrong language ablation
  - per-task validation L1
  - prompt-conditioned action chunks
  - cross-language action distance
  - representative language trajectory strip

Output folders:
- Core / regenerated baseline-style ACT-Lang diagnostics:
  - `project/figures/act_lang_core/`
  - `project/tables/act_lang_core/`
- Language-specific diagnostics:
  - `project/figures/act_lang_language/`
  - `project/tables/act_lang_language/`

Current completed models included:
- `act_lang_B`
- `act_lang_ABC_100k`
- `act_lang_ABC_size_matched`

Note:
- `act_lang_ABC_continue_200k` is skipped until it has a final `checkpoint/model.safetensors`.
- After 200k finishes, rerun the same visualization launcher and it will be included automatically.

Smoke test:
- Ran a small smoke with:
  - `max_eval_samples=8`
  - `max_data_samples=60`
  - `max_pca_samples=60`
  - `batch_size=4`
- Smoke completed successfully.
- Verified generated figure/table folders and progress output.
- The smoke output is only a low-sample functional check; formal figures should be regenerated with the nohup launcher below.

Manual formal visualization command:

```bash
cd /home/zengzixuan/cvprojects/lerobot
source /home/zengzixuan/cvprojects/calvin_env.sh
GPU_ID=0 MAX_EVAL_SAMPLES=128 MAX_DATA_SAMPLES=3000 MAX_PCA_SAMPLES=1500 BATCH_SIZE=8 bash project/scripts/nohup_analyze_act_lang_visuals.sh
```

Monitor command:

```bash
tail -f /EXT_DISK/users/zengzixuan/calvin_runs/act_lang_visuals/logs/<RUN_ID>.log
```

### 2026-06-07 UTC - ACT-Lang formal visualizations completed

Formal visualization run:
- Run directory:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_visuals/20260607_090527_act_lang_visuals_gpu0`
- Log:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_visuals/logs/20260607_090527_act_lang_visuals_gpu0.log`
- The log ended with:
  - `event: act_lang_visuals_complete`
  - `num_models: 3`
  - models included:
    - `act_lang_B`
    - `act_lang_ABC_100k`
    - `act_lang_ABC_size_matched`
- Confirmed the visualization process exited.

Generated core ACT-Lang outputs:
- Figures:
  - `project/figures/act_lang_core/act_lang_loss_and_gap.png`
  - `project/figures/act_lang_core/act_lang_dataset_action_and_task_profile.png`
- Tables:
  - `project/tables/act_lang_core/training_summary.csv`
  - `project/tables/act_lang_core/training_summary.tex`
  - `project/tables/act_lang_core/train_val_gap.csv`
  - `project/tables/act_lang_core/dataset_action_smoothness.csv`
  - `project/tables/act_lang_core/dataset_action_smoothness.tex`
  - `project/tables/act_lang_core/task_frequency_sampled.csv`

Generated language-specific outputs:
- Figures include:
  - `language_ablation_summary.png`
  - `language_embedding_pca_B_vs_ABC.png`
  - `per_task_validation_l1_summary.png`
  - per-model language ablation boxplots
  - per-model per-task validation L1 plots
  - per-model prompt-conditioned action chunk plots
  - per-model cross-language action distance matrices
  - representative language trajectory strip and action-norm plot
- Tables include:
  - `project/tables/act_lang_language/language_ablation_action_l1.csv`
  - `project/tables/act_lang_language/language_ablation_action_l1.tex`
  - `project/tables/act_lang_language/per_task_validation_action_l1.csv`
  - `project/tables/act_lang_language/per_task_validation_action_l1_top.csv`
  - `project/tables/act_lang_language/per_task_validation_action_l1_top.tex`
  - `project/tables/act_lang_language/language_embedding_pca.csv`
  - per-model prompt-conditioned chunk CSVs
  - per-model cross-language action distance CSVs

Quick table checks:
- `training_summary.csv`: `3` rows.
- `language_ablation_action_l1.csv`: `1152` rows.
- `per_task_validation_action_l1.csv`: `316` rows.

200k continuation status at this check:
- `act_lang_ABC_continue_200k` had not yet produced a final checkpoint.
- Latest observed log step: `106800 / 200000`.
- Continuation progress: about `6.8%` of the additional 100k steps.
- Estimated remaining time in log: about `8.8h`.
- Therefore `act_lang_ABC_continue_200k` was not included in the current formal visualization run.
- After it finishes, rerun `project/scripts/nohup_analyze_act_lang_visuals.sh`; the script will include the 200k model automatically once `checkpoint/model.safetensors` exists.

### 2026-06-07 UTC - ACT-Lang supplementary visualizations expanded

Context:
- User felt the ACT-Lang visualization set was too small and requested adding any meaningful non-duplicative figures.
- The analysis script was expanded to include both additional regenerated core diagnostics and language-specific supplementary diagnostics.

Implementation update:
- Updated `project/scripts/analyze_act_lang_visuals.py`.
- Updated `project/scripts/nohup_analyze_act_lang_visuals.sh` with explicit audit lines:
  - `Python starts at ...`
  - `Python finished at ... with exit_code=...`
- Note: detached nohup processes launched from this agent tooling can be cleaned up by the tool environment; user-launched terminal nohup should work normally.
- To generate the new formal outputs immediately, the formal analysis was run in the foreground with:
  - `MAX_EVAL_SAMPLES=128`
  - `MAX_DATA_SAMPLES=3000`
  - `MAX_PCA_SAMPLES=1500`
  - `BATCH_SIZE=8`

New / expanded core diagnostics:
- `act_lang_validation_summary_bars.png`
- `act_lang_runtime_and_convergence.png`
- `act_lang_action_dimension_violin.png`
- `act_lang_action_correlation_matrices.png`
- `act_lang_gripper_and_action_norm_distribution.png`
- `act_lang_task_frequency_B.png`
- `act_lang_task_frequency_ABC.png`
- `act_lang_task_keyword_profile.png`
- `act_lang_language_embedding_norms.png`
- Existing core figures retained:
  - `act_lang_loss_and_gap.png`
  - `act_lang_dataset_action_and_task_profile.png`

New / expanded language-specific diagnostics:
- Per model:
  - `*_per_action_dim_error.png`
  - `*_chunk_horizon_error.png`
  - `*_language_sensitivity_profiles.png`
  - `*_predicted_chunk_smoothness.png`
  - `*_prompt_conditioned_action_dim_heatmap.png`
- Summary figures:
  - `summary_correct_language_per_action_dim_error_heatmap.png`
  - `summary_ablation_error_by_action_dim.png`
  - `summary_correct_language_chunk_horizon_error.png`
  - `summary_language_sensitivity_by_action_dim.png`
  - `summary_language_sensitivity_by_chunk_horizon.png`
  - `summary_predicted_chunk_smoothness.png`
  - `summary_pred_vs_gt_action_norm.png`
  - `language_embedding_task_centroid_similarity_B.png`
  - `language_embedding_task_centroid_similarity_ABC.png`
- Existing language figures retained:
  - language ablation boxplots and summary
  - prompt-conditioned action chunk plots
  - cross-language action distance matrices
  - per-task validation L1 plots and summary
  - language embedding PCA
  - representative language trajectory strip

Output inventory after expansion:
- `project/figures/act_lang_core`: `11` PNG figures.
- `project/figures/act_lang_language`: `41` PNG figures.
- `project/tables/act_lang_core`: `12` table files.
- `project/tables/act_lang_language`: `25` table files.

Superseded note:
- This visualization set originally included `act_lang_ABC_100k` because the 200k continuation had not yet finished.
- The 2026-06-08 redraw below replaces that setup with `act_lang_ABC_200k` and removes the standalone `act_lang_ABC_100k` visualization outputs.

### 2026-06-08 UTC - ACT-Lang visualizations redrawn with ABC 0-200k merged

Context:
- User requested fully redrawing ACT-Lang visualizations with ABC shown as one 0-200k run.
- User explicitly requested not keeping the 100k-only ABC model as a comparison.

Code changes:
- Updated `project/scripts/analyze_act_lang_visuals.py`.
- `load_model_specs()` now uses exactly three models:
  - `act_lang_B`
  - `act_lang_ABC_200k`
  - `act_lang_ABC_size_matched`
- Removed standalone `act_lang_ABC_100k` from the visualization model list.
- Added `load_metrics_for_spec()`:
  - for `act_lang_ABC_200k`, it merges:
    - original ABC metrics steps `1-100000`
    - continuation metrics steps `100001-200000`
  - the combined ABC metrics now span `1-200000` with `200000` rows.
- Language-specific inference for `act_lang_ABC_200k` uses the 200k final checkpoint:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_ABC_continue_200k/20260607_083058_act_lang_ABC_full_continue_200k_gpu1/checkpoint`
- Added default output cleanup before rendering, so old `ABC_100k` figures/tables do not remain in the folders.

Validation:
- Static check passed:
  - `python -m py_compile project/scripts/analyze_act_lang_visuals.py`
- Metrics span check:
  - `act_lang_B`: steps `1-100000`, `100000` rows
  - `act_lang_ABC_200k`: steps `1-200000`, `200000` rows
  - `act_lang_ABC_size_matched`: steps `1-100000`, `100000` rows
- Smoke visualization run passed.
- Formal visualization run completed with:
  - models:
    - `act_lang_B`
    - `act_lang_ABC_200k`
    - `act_lang_ABC_size_matched`

Output inventory after redraw:
- `project/figures/act_lang_core`: `11` PNG figures.
- `project/figures/act_lang_language`: `41` PNG figures.
- `project/tables/act_lang_core`: `12` table files.
- `project/tables/act_lang_language`: `25` table files.
- Confirmed no output filenames contain `ABC_100k` or `100k`.

Key table check:
- `project/tables/act_lang_core/training_summary.csv` now has:
  - `act_lang_B`, final step `100000`
  - `act_lang_ABC_200k`, final step `200000`, best validation step `195000`
  - `act_lang_ABC_size_matched`, final step `100000`

### 2026-06-08 UTC - ACT-Lang visualization data provenance recorded

Context:
- User asked whether the concrete data inside the ACT-Lang figures had been recorded in `draft.md`.
- Answer: before this note, `draft.md` recorded figure inventory and key training conclusions, but not every figure's backing numeric data.
- The full numeric data was already saved in CSV/TEX files under:
  - `project/tables/act_lang_core/`
  - `project/tables/act_lang_language/`
- This note records the figure-to-table mapping and key numeric summaries so future agents can trace each figure without guessing.

Figure data mapping:
- Core training figures:
  - `act_lang_loss_and_gap.png`
    - backed by `training_summary.csv` and `train_val_gap.csv`
  - `act_lang_validation_summary_bars.png`
    - backed by `training_summary.csv`
  - `act_lang_runtime_and_convergence.png`
    - backed by `step_time_summary.csv` and training metrics CSVs in each run directory
- Core dataset/action figures:
  - `act_lang_dataset_action_and_task_profile.png`
    - backed by `dataset_action_smoothness.csv` and `task_frequency_sampled.csv`
  - `act_lang_action_dimension_violin.png`
    - backed by `action_dimension_distribution.csv`
  - `act_lang_action_correlation_matrices.png`
    - backed by sampled action arrays from the language datasets; summary stats are in `action_dimension_distribution.csv`
  - `act_lang_gripper_and_action_norm_distribution.png`
    - backed by sampled action arrays; scalar summary in `dataset_action_smoothness.csv`
  - `act_lang_task_frequency_B.png`, `act_lang_task_frequency_ABC.png`
    - backed by `task_frequency_sampled.csv`
  - `act_lang_task_keyword_profile.png`
    - backed by `task_keyword_profile.csv`
  - `act_lang_language_embedding_norms.png`
    - backed by `language_embedding_norms.csv`
- Language-specific figures:
  - `language_ablation_summary.png` and per-model `*_language_ablation_boxplot.png`
    - backed by `language_ablation_action_l1.csv` and `language_ablation_action_l1.tex`
  - per-model `*_per_action_dim_error.png` and `summary_correct_language_per_action_dim_error_heatmap.png`
    - backed by `per_action_dim_error.csv` and `per_action_dim_error.tex`
  - per-model `*_chunk_horizon_error.png` and `summary_correct_language_chunk_horizon_error.png`
    - backed by `chunk_horizon_error.csv`
  - per-model `*_language_sensitivity_profiles.png`, `summary_language_sensitivity_by_action_dim.png`, and `summary_language_sensitivity_by_chunk_horizon.png`
    - backed by `language_sensitivity_by_action_dim.csv`, `language_sensitivity_by_action_dim.tex`, and `language_sensitivity_by_chunk_horizon.csv`
  - per-model `*_predicted_chunk_smoothness.png` and `summary_predicted_chunk_smoothness.png`
    - backed by `predicted_chunk_smoothness.csv` and `predicted_chunk_smoothness.tex`
  - per-model `*_prompt_conditioned_action_chunks.png`
    - backed by `*_prompt_conditioned_action_chunks.csv`
  - per-model `*_prompt_conditioned_action_dim_heatmap.png`
    - backed by `*_prompt_conditioned_action_dim_summary.csv`
  - per-model `*_cross_language_action_distance.png`
    - backed by `*_cross_language_action_distance.csv`
  - `language_embedding_pca_B_vs_ABC.png`
    - backed by `language_embedding_pca.csv`
  - `language_embedding_task_centroid_similarity_B.png` and `language_embedding_task_centroid_similarity_ABC.png`
    - backed by task-centroid cosine similarities computed from the same sampled language embeddings used for PCA
  - `summary_pred_vs_gt_action_norm.png`
    - backed by `pred_vs_gt_action_norm.csv`
  - `per_task_validation_l1_summary.png` and per-model `*_per_task_validation_l1.png`
    - backed by `per_task_validation_action_l1.csv`, `per_task_validation_action_l1_top.csv`, and `per_task_validation_action_l1_top.tex`

Key numeric summaries:
- Training summary:
  - ACT-Lang-B:
    - final step `100000`
    - final train Action L1 `0.245766`
    - final val Action L1 `0.394312`
    - best val Action L1 `0.392651` at step `95000`
  - ACT-Lang-ABC 200k:
    - final step `200000`
    - final train Action L1 `0.270620`
    - final val Action L1 `0.394804`
    - best val Action L1 `0.390484` at step `195000`
  - ACT-Lang-ABC Size-Matched:
    - final step `100000`
    - final train Action L1 `0.273167`
    - final val Action L1 `0.475930`
    - best val Action L1 `0.471781` at step `75000`
- Dataset/action profile:
  - B sampled frames `2704`
    - mean action L2 first 6 dims `0.522019`
    - mean step-delta L2 first 6 dims `0.744850`
    - q90 step-delta L2 first 6 dims `1.212080`
    - gripper close fraction `0.538462`
  - ABC sampled frames `2868`
    - mean action L2 first 6 dims `0.522032`
    - mean step-delta L2 first 6 dims `0.752431`
    - q90 step-delta L2 first 6 dims `1.201287`
    - gripper close fraction `0.502441`
- Language embedding norms:
  - B mean embedding norm `7.250050`, std `0.400903`
  - ABC mean embedding norm `7.245489`, std `0.410259`
- Language ablation mean Action L1, `128` validation samples per model/mode:
  - ACT-Lang-B:
    - correct `0.335245`
    - zero language `0.570149`
    - wrong language `0.695287`
  - ACT-Lang-ABC 200k:
    - correct `0.375421`
    - zero language `0.670071`
    - wrong language `0.746481`
  - ACT-Lang-ABC Size-Matched:
    - correct `0.456537`
    - zero language `0.725740`
    - wrong language `0.800650`
- Predicted chunk smoothness, mean chunk delta L2 over first 6 action dims:
  - ACT-Lang-B:
    - GT `0.231467`
    - correct language `0.563942`
    - zero language `0.465774`
    - wrong language `0.578668`
  - ACT-Lang-ABC 200k:
    - GT `0.257957`
    - correct language `0.580672`
    - zero language `0.472505`
    - wrong language `0.573664`
  - ACT-Lang-ABC Size-Matched:
    - GT `0.257957`
    - correct language `0.606373`
    - zero language `0.514228`
    - wrong language `0.634046`

Interpretation from recorded figure data:
- The 0-200k ACT-Lang-ABC model is the best ABC-family model by validation Action L1 and strongly improves over size-matched.
- Zeroing or shuffling language embeddings substantially worsens action L1 for all three models, supporting that the language token is behaviorally active rather than unused.
- The figure-level data is LaTeX/report-ready through the CSV/TEX tables above; `draft.md` now records the provenance and headline values, while exhaustive per-dimension/per-task/per-horizon values remain in the corresponding tables.

### 2026-06-08 UTC - ACT-Lang-ABC 100k-to-200k continuation completed

Run checked:
- `/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_ABC_continue_200k/20260607_083058_act_lang_ABC_full_continue_200k_gpu1`
- Log:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_ABC_continue_200k/logs/20260607_083058_act_lang_ABC_full_continue_200k_gpu1.log`

Completion checks:
- Final checkpoint exists:
  - `checkpoint/model.safetensors`
  - `checkpoint/training_state.pt`
  - policy config and pre/postprocessor files
- Periodic checkpoints exist from:
  - `step_00110000` through `step_00190000`
- `training_state.pt` reports `step: 200000`.
- `metrics.csv` has `100000` rows, covering continuation steps `100001` to `200000`.
- `self_check.json` reports `passed: true`.
- No training process remains active.
- GPU memory returned to idle state.

Resume metadata:
- Resumed from:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_ABC/20260606_163515_act_lang_ABC_full_gpu1/checkpoint`
- Resume step:
  - `100000`
- Target total step:
  - `200000`
- Additional steps:
  - `100000`

Continuation metrics:
- Final step:
  - `200000`
- Final train Action L1:
  - `0.270620`
- Final validation Action L1:
  - `0.394804`
- Best validation step:
  - `195000`
- Best validation Action L1:
  - `0.390484`
- Final minus best validation Action L1:
  - `+0.004319`
- Validation snapshots:
  - step `105000`: `0.414416`
  - step `150000`: `0.401710`
  - step `190000`: `0.394913`
  - step `195000`: `0.390484`
  - step `200000`: `0.394804`
- Language minibatch sensitivity at self-check:
  - `0.475532`

Comparison:
- ACT-Lang-ABC 100k:
  - best validation Action L1: `0.411574` at step `100000`
- ACT-Lang-ABC 200k continuation:
  - best validation Action L1: `0.390484` at step `195000`
- Absolute improvement:
  - `0.021090` Action L1
- Relative improvement vs 100k best:
  - about `5.1%`
- The user's earlier concern that ABC full had not converged at 100k was correct.

Updated table:
- Wrote `project/tables/act_lang_training_results_with_200k.csv`.

Interpretation:
- Continuing ABC full was beneficial and brought ACT-Lang-ABC below ACT-Lang-B's best validation Action L1 on its own ABC validation split.
- The final checkpoint at 200k is slightly worse than the metric-best validation point at 195k, but 195k was not saved as a checkpoint.
- Because save frequency is every 10k, the nearest saved checkpoint before the metric-best `195000` point is `step_00190000`, but it is not the best available saved checkpoint.
- After comparing saved checkpoint validation points, the best available saved model is the final checkpoint:
  - `checkpoint/`
  - step `200000`
  - validation Action L1 `0.394804`
- Next needed action: rerun ACT-Lang visualizations so `act_lang_ABC_continue_200k` is included.

### 2026-06-07 UTC - ACT-Lang-ABC full continuation to 200k prepared

Context:
- User asked whether ACT-Lang-ABC full can continue from the completed `100k` checkpoint or must restart.
- Decision: continue from the existing `100k` checkpoint because the run saved both model weights and optimizer/training state.

Source checkpoint:
- `/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_ABC/20260606_163515_act_lang_ABC_full_gpu1/checkpoint`
- Required files present:
  - `model.safetensors`
  - `training_state.pt`
  - `config.json`
  - policy pre/postprocessor files

Code changes:
- Updated `project/scripts/train_act_lang.py` with resume support:
  - new CLI argument: `--resume-from`
  - accepts either a checkpoint directory or a run directory containing `checkpoint/`
  - loads ACT-Lang weights from `model.safetensors`
  - restores optimizer state and `resume_step` from `training_state.pt`
  - writes absolute metric step numbers from `resume_step + 1`
  - saves continuation checkpoints with absolute step names, e.g. `step_00110000`
  - writes `resume_info.json`
  - prints tail-visible continuation progress with both continuation progress and absolute total step
- Changed the start log to summarize episode ranges instead of printing every episode id.

New files:
- Config:
  - `project/configs/act_lang_ABC_continue_200k.yaml`
- Launcher:
  - `project/scripts/nohup_train_act_lang_ABC_continue_200k.sh`
- Smoke config:
  - `project/configs/act_lang_ABC_resume_smoke.yaml`

Config alignment check:
- `act_lang_ABC_continue_200k.yaml` was compared against `act_lang_ABC_full.yaml`.
- It matches the original ABC full config except for intended continuation fields:
  - `experiment.name`
  - `experiment.output_dir`
  - `dataset._notes`
  - `training.steps: 200000`
  - `training.resume_from`
- Dataset split, architecture, language conditioning, optimizer, batch size, workers, validation frequency, checkpoint frequency, and loss settings remain aligned with ABC full.

Resume smoke test:
- Ran a 2-step resume smoke from the real `100k` checkpoint to `100002`.
- Output:
  - `project/outputs/act_lang_ABC_resume_smoke`
- Results:
  - loaded local weights successfully
  - restored `resume_step: 100000`
  - metrics rows: `2`
  - first metric step: `100001`
  - last metric step: `100002`
  - final checkpoint `training_state.pt` step: `100002`
  - self-check passed
  - language minibatch check passed
  - `language_sensitivity_l1: 0.31866684556007385`

Static checks:
- `python -m py_compile project/scripts/train_act_lang.py` passed.
- `bash -n project/scripts/nohup_train_act_lang_ABC_continue_200k.sh` passed.
- YAML parsing passed.

Manual launch command:

```bash
cd /home/zengzixuan/cvprojects/lerobot
source /home/zengzixuan/cvprojects/calvin_env.sh
GPU_ID=1 bash project/scripts/nohup_train_act_lang_ABC_continue_200k.sh
```

Monitor command:

```bash
tail -f /EXT_DISK/users/zengzixuan/calvin_runs/act_lang_ABC_continue_200k/logs/<RUN_ID>.log
```

Tail command will be printed by the launcher. Expected progress line:

```text
PROGRESS act_lang_B_full [########------------------------] 25000/100000 ( 25.0%) elapsed=... eta=...
```

### 2026-06-06 UTC - ACT-Lang-ABC full and size-matched launchers prepared

Context:
- User requested ABC language-based training scripts ahead of the ABC dataset finishing.
- Scope is limited to:
  - `ACT-Lang-ABC`
  - `ACT-Lang-ABC-size-matched`
- No augmentation variants are prepared for the language-based ABC stage.

Generated files:
- `project/configs/act_lang_ABC_full.yaml`
- `project/configs/act_lang_ABC_size_matched_full.yaml`
- `project/scripts/nohup_train_act_lang_ABC_task.sh`
- `project/scripts/nohup_train_act_lang_ABC.sh`
- `project/scripts/nohup_train_act_lang_ABC_size_matched.sh`
- `project/scripts/nohup_train_act_lang_ABC_pair.sh`
- `project/tables/act_lang_ABC_size_matched_split_summary.json`

GPU defaults:
- `ACT-Lang-B`: GPU0 via `project/scripts/nohup_train_act_lang_B.sh`
- `ACT-Lang-ABC`: GPU1 via `project/scripts/nohup_train_act_lang_ABC.sh`
- `ACT-Lang-ABC-size-matched`: GPU2 via `project/scripts/nohup_train_act_lang_ABC_size_matched.sh`
- The ABC launcher refuses GPU0 by default to avoid colliding with `ACT-Lang-B`.

Delay behavior:
- Preferred ABC entry point is `project/scripts/nohup_train_act_lang_ABC_pair.sh`.
- It launches `ACT-Lang-ABC` first, waits `STAGGER_SECONDS=300`, then launches `ACT-Lang-ABC-size-matched`.
- This staggers dataset metadata/cache startup IO.
- Set `STAGGER_SECONDS=0` only if IO staggering is not needed.

Strict alignment:
- `ACT-Lang-ABC` and `ACT-Lang-ABC-size-matched` match `ACT-Lang-B` on:
  - ResNet18 ImageNet backbone
  - `n_obs_steps=1`
  - `chunk_size=100`
  - `n_action_steps=100`
  - transformer dimensions/layers/heads
  - VAE/KL settings
  - optimizer
  - `batch_size=32`
  - `steps=100000`
  - logging, validation, checkpoint, and grad-clipping settings.
- Intended differences are dataset split and default GPU only.

Language ABC split:
- Full ABC:
  - train episodes: `0:16083`
  - val episodes: `16083:17870`
- Size-matched ABC:
  - target: `ACT-Lang-B` train frames `330455`
  - selected train episodes: `5517`
  - selected train frames: `330546`
  - delta vs B train frames: `+91`
  - selected by environment:
    - A: `1830` episodes, `110177` frames
    - B: `1836` episodes, `110176` frames
    - C: `1851` episodes, `110193` frames
  - val episodes reuse full ABC validation: `16083:17870`

Safety guard:
- ABC launchers refuse to run until:
  - `project/scripts/train_act_lang.py` exists;
  - `/EXT_DISK/users/zengzixuan/processed-calvin/calvin_lang_ABC` exists;
  - the generated dataset has `observation.language_embedding` shape `(384,)`, at least `17870` episodes, and at least `1071743` frames.

Verification:
- `bash -n` passed for all three ABC launcher scripts.
- Config comparison showed all intended shared policy/training hyperparameters are identical across `ACT-Lang-B`, `ACT-Lang-ABC`, and `ACT-Lang-ABC-size-matched`.

Manual launch commands after ACT-Lang implementation and full ABC dataset are ready:

```bash
bash project/scripts/nohup_train_act_lang_ABC_pair.sh
```

### 2026-06-06 UTC - ACT-Lang training harness implemented and minibatch self-check passed

Implemented files:
- `project/scripts/train_act_lang.py`
- `project/configs/act_lang_B.yaml`

Design decision:
- Standard LeRobot ACT does not consume `observation.language_embedding`; adding that field to the dataset is not enough for language-conditioned CALVIN success-rate evaluation.
- Implemented a local `act_lang` policy/config in the training harness.
- Architecture stays aligned with ACT-B/ACT-ABC except for one additional transformer encoder conditioning token:
  - language input: `observation.language_embedding`
  - language dimension: `384`
  - projection: `Linear(384, 512)`
  - fusion: append as one encoder token after latent/robot-state tokens and before image tokens.
- Language embedding is marked as `FeatureType.LANGUAGE` with `IDENTITY` normalization.
- Image/action/state normalization and ACT hyperparameters remain aligned with the previous baseline configs.

Debugging:
- First sandboxed mini run failed because CUDA and `/EXT_DISK` HuggingFace cache were blocked by the sandbox.
- Escalated run then found a real implementation bug: direct local policy construction did not call `.to(config.device)`, unlike LeRobot's factory. Fixed by moving `ACTLangPolicy` to `policy.config.device` before optimizer/preprocessor use.

Minibatch self-check:
- Command:

```bash
source /home/zengzixuan/cvprojects/calvin_env.sh >/dev/null
ACT_DEVICE_OVERRIDE=cuda:0 "$LEROBOT_PYTHON" project/scripts/train_act_lang.py \
  --config project/configs/act_lang_B.yaml \
  --output-dir project/outputs/act_lang_B/mini_trial_selfcheck
```

- Device: `cuda:0`
- Train episodes: `0:8`, `468` frames
- Val episodes: `5503:5507`, `241` frames
- Batch size: `2`
- Steps: `2`
- Cameras: static and gripper
- Raw language shape: `[2, 384]`
- Processed language shape/device: `[2, 384]` on `cuda:0`
- Raw action chunk shape: `[2, 100, 7]`
- Predicted action chunk shape: `[2, 100, 7]`
- Language sensitivity check: replacing language embeddings with zeros changed predictions by mean L1 `0.006950919`, so language is connected to the forward path.

Mini metrics:
- Step 1: train Action L1 `0.878052`, val Action L1 `1.138959`
- Step 2: train Action L1 `0.886268`, val Action L1 `1.057600`

Saved artifacts:
- `project/outputs/act_lang_B/mini_trial_selfcheck/metrics.csv`
- `project/outputs/act_lang_B/mini_trial_selfcheck/minibatch_check.json`
- `project/outputs/act_lang_B/mini_trial_selfcheck/self_check.json`
- `project/outputs/act_lang_B/mini_trial_selfcheck/checkpoint/`

Self-check result:
- `passed: true`
- Metrics CSV exists and has 2 rows.
- Train and validation Action L1 are finite.
- Checkpoint, `config.json`, model weights, preprocessor, postprocessor, and optimizer training state all exist.

Next manual full-training command for B:

```bash
GPU_ID=0 bash project/scripts/nohup_train_act_lang_B.sh
```

ABC note:
- `ACT-Lang-ABC` launchers are now unblocked from the code side, but still depend on a complete `/EXT_DISK/users/zengzixuan/processed-calvin/calvin_lang_ABC` dataset passing the launcher guard.

### 2026-06-06 UTC - ACT-Lang config and ABC launcher self-check

Official LeRobot ACT reference points checked:
- Official LeRobot ACT docs describe ACT as using:
  - ResNet-18 vision backbone
  - transformer encoder/decoder
  - multi-camera RGB input
  - current robot joint state
  - learned latent variable `z`
  - future action chunks
- Official training docs use `policy.type=act`, `policy.device=cuda`, standard `lerobot-train`, and note that 100k steps on a single GPU is a common training scale.
- Official docs say batch size can start from 8 and be adjusted by GPU memory. Our batch size remains 32 to stay aligned with the previous ACT-B/ACT-ABC experimental setup and because the A6000 memory budget supports it.

Config self-check:
- Compared `project/configs/act_lang_B_full.yaml`, `project/configs/act_lang_ABC_full.yaml`, and `project/configs/act_lang_ABC_size_matched_full.yaml`.
- Shared policy fields match exactly across B, ABC full, and ABC size-matched.
- Shared training fields match exactly across B, ABC full, and ABC size-matched.
- Core ACT fields match local LeRobot `ACTConfig()` defaults:
  - `vision_backbone=resnet18`
  - `pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1`
  - `n_obs_steps=1`
  - `chunk_size=100`
  - `n_action_steps=100`
  - `dim_model=512`
  - `n_heads=8`
  - `dim_feedforward=3200`
  - `n_encoder_layers=4`
  - `n_decoder_layers=1`
  - `use_vae=true`
  - `latent_dim=32`
  - `n_vae_encoder_layers=4`
  - `dropout=0.1`
  - `kl_weight=10.0`
  - `optimizer_lr=1e-5`
  - `optimizer_weight_decay=1e-4`
  - `optimizer_lr_backbone=1e-5`

Important interpretation:
- The only deliberate deviation from official plain ACT is `policy.type=act_lang` plus the language-conditioning token.
- This deviation is necessary because official plain ACT does not consume `observation.language_embedding`; training with `policy.type=act` would produce another non-language model and would not satisfy CALVIN language-conditioned success-rate evaluation.

Code fix from self-check:
- `project/scripts/train_act_lang.py` previously worked when executed as a script, but failed when imported as `project.scripts.train_act_lang` from the repo root because it used `from train_act import ...`.
- Added a fallback import from `project.scripts.train_act`.
- `py_compile` and module import now pass.

ABC launcher code-path check:
- `bash -n` passed for:
  - `project/scripts/nohup_train_act_lang_B.sh`
  - `project/scripts/nohup_train_act_lang_ABC_task.sh`
  - `project/scripts/nohup_train_act_lang_ABC.sh`
  - `project/scripts/nohup_train_act_lang_ABC_size_matched.sh`
  - `project/scripts/nohup_train_act_lang_ABC_pair.sh`
- ABC task launcher resolves:
  - full config: `project/configs/act_lang_ABC_full.yaml`
  - size-matched config: `project/configs/act_lang_ABC_size_matched_full.yaml`
  - train script: `project/scripts/train_act_lang.py`
  - dataset root: `$CALVIN_LEROBOT_ROOT/calvin_lang_ABC`
- GPU mapping is correct:
  - external `GPU_ID=1/2`
  - `CUDA_VISIBLE_DEVICES=$GPU_ID`
  - model sees `ACT_DEVICE_OVERRIDE=cuda:0`, which is correct inside the masked process.
- Pair launcher behavior is correct:
  - launches ABC full first with `START_DELAY_SECONDS=0`
  - waits `STAGGER_SECONDS=300`
  - launches ABC size-matched with `START_DELAY_SECONDS=0`

Current ABC dataset guard status:
- Current partial ABC language dataset was observed at roughly `948` episodes and `56888` frames during this check.
- Launcher requires at least `17870` episodes and `1071743` frames.
- Therefore current ABC launcher would fail fast before `nohup` training starts.

ABC split parser check:
- Full ABC config parses to:
  - train length `16083`, episodes `0..16082`
  - val length `1787`, episodes `16083..17869`
- Size-matched ABC config parses to:
  - train length `5517`
  - val length `1787`, episodes `16083..17869`
- Size-matched summary:
  - selected train frames `330546`
  - target B train frames `330455`
  - delta `+91`
  - selected by env: A `1830`, B `1836`, C `1851`

Conclusion:
- ACT architecture/training hyperparameters are strictly aligned with official ACT defaults and with each other.
- The language-token addition is an explicit and necessary custom extension, not an accidental config drift.
- ABC training instructions are code-valid and will become effective once the full ABC language dataset is complete; with the current partial dataset they correctly refuse to launch training.

### 2026-06-06 UTC - ACT-Lang-B accidental launch cleanup

Context:
- User reported forgetting the conda/environment setup and requested that the current B training process be checked, stopped if still running, and its training artifacts deleted before a clean restart.

Process check:
- Found active ACT-Lang-B full training process:
  - main PID: `1458908`
  - command: `/home/zengzixuan/miniforge3/envs/lerobot/bin/python project/scripts/train_act_lang.py --config project/configs/act_lang_B_full.yaml --output-dir /EXT_DISK/users/zengzixuan/calvin_runs/act_lang_B/20260606_110700_act_lang_B_full_gpu0`
  - child dataloader worker PIDs were also present.
  - GPU memory observed before stopping: about `2874 MiB`.
- Note: the process was using the configured `lerobot` Python path, but cleanup was still performed as requested for a clean restart.

Cleanup performed:
- Sent `TERM` to the main process and child processes.
- Confirmed no remaining ACT-Lang-B training process.
- Deleted run directory:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_B/20260606_110700_act_lang_B_full_gpu0`
- Deleted log file:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_B/logs/20260606_110700_act_lang_B_full_gpu0.log`
- Post-cleanup check:
  - only empty `act_lang_B/logs/` directory remains under the ACT-Lang-B run group.
  - no GPU training process reported by `nvidia-smi`.

Clean restart command:

```bash
cd /home/zengzixuan/cvprojects/lerobot
source /home/zengzixuan/cvprojects/calvin_env.sh
GPU_ID=0 bash project/scripts/nohup_train_act_lang_B.sh
```

### 2026-06-06 UTC - Confirmed 12-worker DataLoader works on this machine

User request:
- Check local records/system context and confirm `num_workers=12` can work before restarting ACT-Lang-B.

Machine resources:
- CPU: `112` logical CPUs
- CPU model: Intel Xeon Gold 6348, 2 sockets, 28 cores/socket, 2 threads/core
- Memory: about `503GiB` total, about `495GiB` available at check time
- File descriptor limit: `1048576`
- User process limit: `2061949`

Local history:
- Previous successful full runs used `num_workers=4`:
  - ACT-B
  - ACT-ABC
  - ACT-ABC-size-matched
  - ACT-B-aug
- No previous full run record with `num_workers=12` was found.
- No local logs contained DataLoader worker failure, shared-memory bus error, too-many-open-files, or worker-killed errors.

12-worker smoke test:
- Ran a no-training DataLoader smoke on `local/calvin_lang_B`.
- Used:
  - `batch_size=32`
  - `num_workers=12`
  - `prefetch_factor=2`
  - `persistent_workers=true`
  - first `64` ACT-Lang-B language episodes
- Successfully read 3 batches.
- Dataset subset:
  - `64` episodes
  - `3836` frames
- Output shapes per batch:
  - `observation.images.static`: `[32, 3, 200, 200]`
  - `observation.images.gripper`: `[32, 3, 84, 84]`
  - `observation.state`: `[32, 15]`
  - `observation.language_embedding`: `[32, 384]`
  - `action`: `[32, 100, 7]`
  - `action_is_pad`: `[32, 100]`
- Elapsed time for 3 batches after worker startup: `6.219s`

Conclusion:
- `num_workers=12` is supported by the local CPU/memory/ulimit environment and by the ACT-Lang-B dataset/DataLoader path.
- This confirms the setting can work mechanically.
- It does not guarantee maximum throughput; after restart, compare ETA/GPU utilization after about 1000 steps.

### 2026-06-06 UTC - DataLoader worker-count recommendation for B/ABC/ABC-size parallel training

Question:
- User asked for a reasonable CPU/DataLoader worker count that can speed training while also considering later parallel `ACT-Lang-ABC` and `ACT-Lang-ABC-size-matched` runs.

External reference checked:
- PyTorch DataLoader documentation/tutorial says:
  - increase `num_workers` when transforms/decoding are expensive, storage is slow, or GPU is idle waiting for data;
  - each worker maintains a prefetch queue controlled by `prefetch_factor`;
  - too many workers can waste CPU/memory and can cause shared-memory pressure;
  - `persistent_workers=True` avoids worker restart overhead.

Local constraints:
- Machine has `112` logical CPUs and `56` physical cores.
- `/dev/shm` has `252G` available.
- Memory available was about `495GiB`.
- File descriptor/process limits are high.
- Existing ACT-Lang DataLoader uses:
  - `prefetch_factor=2` for `num_workers>0`
  - `persistent_workers=True`
  - `pin_memory=True` when CUDA is visible.

Reasoning:
- At `num_workers=4`, four workers were each near `100%` CPU and GPU utilization was bursty, meaning the input pipeline was limiting throughput.
- `num_workers=12` passed a real ACT-Lang-B DataLoader smoke test.
- If B, ABC full, and ABC size-matched run concurrently:
  - `12 workers/job * 3 jobs = 36 workers`
  - plus three main training processes
  - still below `56` physical cores and far below `112` logical CPUs.
- `num_workers=16` for three concurrent jobs would be `48` workers, close to physical-core saturation once main processes, OS, logging, and storage handling are included.
- Higher worker counts can increase random IO pressure on `/EXT_DISK`, where all processed CALVIN datasets live.

Recommendation:
- Use `num_workers=12` per ACT-Lang training job as the current balanced setting.
- Do not increase beyond 12 before measuring the 12-worker full run.
- If three concurrent jobs show high IO wait or worse ETA, reduce all ACT-Lang jobs to `num_workers=8`.
- Keep the same value across B/ABC/ABC-size unless explicitly running an input-pipeline ablation.

Current config status:
- `project/configs/act_lang_B_full.yaml`: `num_workers=12`
- `project/configs/act_lang_ABC_full.yaml`: `num_workers=12`
- `project/configs/act_lang_ABC_size_matched_full.yaml`: `num_workers=12`

### 2026-06-06 UTC - ABC ACT-Lang worker count staged to 16 after conversion

Context:
- User asked whether, after ABC data conversion finishes, ABC full and ABC size-matched can use `num_workers=16`.
- Decision: yes, set ABC full and ABC size-matched to 16 while keeping B at 12.

Config changes:
- `project/configs/act_lang_B_full.yaml`: remains `num_workers=12`
- `project/configs/act_lang_ABC_full.yaml`: changed to `num_workers=16`
- `project/configs/act_lang_ABC_size_matched_full.yaml`: changed to `num_workers=16`

Rationale:
- Once ABC data conversion is complete, the conversion process will no longer compete for CPU and `/EXT_DISK` IO.
- Planned concurrent training load:
  - B: `12` workers
  - ABC full: `16` workers
  - ABC size-matched: `16` workers
  - total DataLoader workers: `44`
- This remains below `56` physical cores and is much safer than `24 + 24 + 12 = 60` workers.
- This is an input-pipeline throughput setting, not a model/training ablation.

Self-check:
- Config parse passed.
- Confirmed worker counts:
  - B `12`
  - ABC full `16`
  - ABC size-matched `16`
- ABC full and ABC size-matched have no policy mismatches.
- ABC full and ABC size-matched have no training mismatches except the intended dataset split.
- `bash -n` passed for all ACT-Lang ABC launcher scripts.

ABC launch command after conversion completes:

```bash
cd /home/zengzixuan/cvprojects/lerobot
source /home/zengzixuan/cvprojects/calvin_env.sh
ABC_GPU_ID=1 SIZE_GPU_ID=2 STAGGER_SECONDS=300 bash project/scripts/nohup_train_act_lang_ABC_pair.sh
```

### 2026-06-06 UTC - ACT-Lang-B validation NaN check

Question:
- User noticed `val_loss` and `val_action_l1` are NaN in the ACT-Lang-B training log and asked whether the code was wrong.

Checked:
- Current ACT-Lang-B run:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_B/20260606_112848_act_lang_B_full_gpu0`
- Full config:
  - `training.val_freq=5000`
  - `training.max_val_batches=64`
- Training code:
  - `last_val = {"val_loss": math.nan, "val_action_l1": math.nan}`
  - validation is called only when `step % val_freq == 0` or `step == steps`
  - therefore steps `1..4999` intentionally log NaN validation values.

Current status:
- Latest inspected step was around `2600`, so the first validation had not happened yet.
- This explains the NaN values.
- No validation traceback or error was found in the log.
- Mini-trial previously produced finite validation loss because its config used `val_freq=1`.

Conclusion:
- No code bug was found.
- Full-run validation should first appear at step `5000`.
- Around step `5000`, training may pause longer while running up to `64` validation batches.

### 2026-06-06 UTC - ABC launch debug and GPU1 cleanup

Context:
- User reported that after running the ABC nohup, only GPU0 and GPU1 were active, while GPU2 was idle.
- User requested pausing GPU1, deleting GPU1 training artifacts, and providing a clean retraining command.

Observed GPU state:
- GPU0 was running ACT-Lang-B:
  - run: `/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_B/20260606_112848_act_lang_B_full_gpu0`
- GPU1 was running ACT-Lang-ABC full:
  - run: `/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_ABC/20260606_160942_act_lang_ABC_full_gpu1`
- GPU2 had no training process.

Debug finding:
- Two ABC run directories/logs existed, and both were `act_lang_ABC_full` on GPU1:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_ABC/20260606_160942_act_lang_ABC_full_gpu1`
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_ABC/20260606_161444_act_lang_ABC_full_gpu1`
- Both logs showed:
  - `EXPERIMENT=act_lang_ABC_full`
  - `GPU_ID=1`
  - `CONFIG=project/configs/act_lang_ABC_full.yaml`
- No `act_lang_ABC_size_matched` run directory/log existed.
- The second full run failed with exit code `137`:
  - `Killed`
  - likely because a second full ABC job was accidentally launched on the same GPU/CPU/IO path.

Conclusion:
- GPU2 did not fail during training; it was never assigned a size-matched run.
- The evidence indicates duplicate ABC full launch rather than the intended ABC full + ABC size-matched pair.

Cleanup performed:
- Stopped GPU1 ABC full training.
- Deleted both GPU1 ABC full run directories and logs:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_ABC/20260606_160942_act_lang_ABC_full_gpu1`
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_ABC/20260606_161444_act_lang_ABC_full_gpu1`
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_ABC/logs/20260606_160942_act_lang_ABC_full_gpu1.log`
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_ABC/logs/20260606_161444_act_lang_ABC_full_gpu1.log`
- Verified GPU apps after cleanup:
  - only ACT-Lang-B remained on GPU0.

Clean ABC retraining command:

```bash
cd /home/zengzixuan/cvprojects/lerobot
source /home/zengzixuan/cvprojects/calvin_env.sh
ABC_GPU_ID=1 SIZE_GPU_ID=2 STAGGER_SECONDS=300 bash project/scripts/nohup_train_act_lang_ABC_pair.sh
```

### 2026-06-06 UTC - Confirmed ABC training dataset paths and split

User request:
- Confirm that ABC training is using the correct dataset.

Dataset check:
- ABC language dataset root:
  - `/EXT_DISK/users/zengzixuan/processed-calvin/calvin_lang_ABC`
- Dataset metadata:
  - `total_episodes=17870`
  - `total_frames=1071743`
  - `total_tasks=389`
- Required language-conditioned features exist:
  - `observation.images.static`: image `[200, 200, 3]`
  - `observation.images.gripper`: image `[84, 84, 3]`
  - `observation.state`: float32 `[15]`
  - `observation.language_embedding`: float32 `[384]`
  - `action`: float32 `[7]`

Config check:
- `project/configs/act_lang_ABC_full.yaml`
  - `repo_id=local/calvin_lang_ABC`
  - `root=/EXT_DISK/users/zengzixuan/processed-calvin/calvin_lang_ABC`
  - train episodes: `0:16083`
  - val episodes: `16083:17870`
  - `policy.type=act_lang`
  - `language_embedding_key=observation.language_embedding`
  - `num_workers=16`
- `project/configs/act_lang_ABC_size_matched_full.yaml`
  - `repo_id=local/calvin_lang_ABC`
  - `root=/EXT_DISK/users/zengzixuan/processed-calvin/calvin_lang_ABC`
  - train episodes: size-matched list of `5517` episodes
  - val episodes: `16083:17870`
  - `policy.type=act_lang`
  - `language_embedding_key=observation.language_embedding`
  - `num_workers=16`

Current run check:
- Current ABC full run:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_ABC/20260606_163515_act_lang_ABC_full_gpu1`
- Its log reports:
  - `train_frames=964219`
  - `val_frames=107524`
  - `language_embedding_key=observation.language_embedding`
  - minibatch language shape `[32, 384]`
  - language processed on `cuda:0` within the GPU1-masked process
- This confirms the active ABC full run is using the full `calvin_lang_ABC` language dataset, not B and not the old non-language ABC dataset.

Size-matched status:
- At the time of checking, `ACT-Lang-ABC-size-matched` had not started yet.
- The pair process was still within the `STAGGER_SECONDS=300` wait window.
- Therefore the absence of GPU2/run files at that moment was expected.

Conclusion:
- ABC full is using the correct dataset.
- ABC size-matched config is also pointed at the correct dataset and split; verify after the 300s stagger once it starts.

### 2026-06-07 UTC - ACT-Lang full training results checked

User request:
- User reported that the language-conditioned runs should be finished and asked to check results.

Process status:
- No active ACT-Lang training process was found.
- No GPU training process was reported by `nvidia-smi`.

Completed runs:
- ACT-Lang-B:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_B/20260606_112848_act_lang_B_full_gpu0`
- ACT-Lang-ABC:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_ABC/20260606_163515_act_lang_ABC_full_gpu1`
- ACT-Lang-ABC-size-matched:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_ABC_size_matched/20260606_164015_act_lang_ABC_size_matched_full_gpu2`

Saved summary table:
- `project/tables/act_lang_training_results.csv`

Training/result summary:

| Model | Train frames | Val frames | Steps | Workers | Elapsed h | Final train L1 | Final val L1 | Best val step | Best val L1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ACT-Lang-B | 330455 | 36641 | 100000 | 12 | 13.1927 | 0.245766 | 0.394312 | 95000 | 0.392651 |
| ACT-Lang-ABC | 964219 | 107524 | 100000 | 16 | 10.1110 | 0.316102 | 0.411574 | 100000 | 0.411574 |
| ACT-Lang-ABC-size-matched | 330546 | 107524 | 100000 | 16 | 10.0998 | 0.273167 | 0.475930 | 75000 | 0.471781 |

Artifact checks:
- Each run has:
  - `metrics.csv` with `100000` rows
  - final `checkpoint/`
  - periodic checkpoints at steps `10000` through `90000`
  - `config.json`
  - `model.safetensors`
  - preprocessor and postprocessor configs/weights
  - `training_state.pt`
  - `self_check.json`
- All three `self_check.json` files report `passed=true`.
- Language-conditioning minibatch checks passed for all three:
  - ACT-Lang-B language sensitivity L1: `0.008130`
  - ACT-Lang-ABC language sensitivity L1: `0.007477`
  - ACT-Lang-ABC-size-matched language sensitivity L1: `0.007691`

Interpretation:
- ACT-Lang-B has the best validation Action L1 among the three on its own validation split.
- ACT-Lang-ABC full has higher validation Action L1 than B, but it is evaluated on the larger ABC language validation split, not B's validation split.
- ACT-Lang-ABC-size-matched has the worst validation Action L1 among these offline validation numbers, suggesting size-matched environment diversity alone did not improve offline action L1 under this split.
- These are supervised offline losses. They do not replace official CALVIN success-rate evaluation.

Speed note:
- ACT-Lang-ABC full used about `2.92x` the B train frames, but training time was shorter because training is fixed at `100000` optimizer steps, not one or more epochs over all frames.
- ABC and size-matched also used `num_workers=16`, while B used `num_workers=12`.
- Therefore runtime should not be interpreted as proportional to dataset frame count.

### 2026-06-07 UTC - ACT-Lang-ABC full convergence check

Question:
- User suspected ACT-Lang-ABC full may not have converged because its best validation loss occurs at the final 100k step.

Validation trend summary:
- ACT-Lang-B:
  - best val Action L1: `0.392651` at step `95000`
  - final val Action L1: `0.394312`
  - final minus best: `+0.001661`
  - interpretation: roughly plateaued/slightly worse at the end.
- ACT-Lang-ABC full:
  - best val Action L1: `0.411574` at step `100000`
  - final val Action L1: `0.411574`
  - final minus best: `0.0`
  - last 25k delta: `-0.018408`
  - last 10k delta: `-0.012015`
  - last 5k delta: `-0.006415`
  - interpretation: still improving at the final checkpoint; likely undertrained at 100k.
- ACT-Lang-ABC-size-matched:
  - best val Action L1: `0.471781` at step `75000`
  - final val Action L1: `0.475930`
  - final minus best: `+0.004148`
  - interpretation: plateaued/overfit after 75k.

Additional evidence:
- ACT-Lang-ABC full has `964219` train frames vs B `330455`, about `2.92x` more frames.
- Training used fixed `100000` optimizer steps for both, not fixed epochs.
- Therefore ABC full received much less coverage per frame/task than B.
- ABC full final train Action L1 `0.316102` is higher than B final train Action L1 `0.245766`, consistent with a harder/larger or less-converged training problem.

Conclusion:
- User's suspicion is well supported.
- ACT-Lang-ABC full's current best being at 100k is a strong sign that it had not fully saturated by the end of the run.
- Reasonable next experiment: continue ACT-Lang-ABC full to `150k` or `200k` total steps, evaluating every `5k`, without changing model/data/batch/loss.
- ACT-Lang-ABC-size-matched does not show the same need for continuation; its best validation checkpoint is currently step `75000`.

### 2026-06-06 UTC - ACT-Lang-B low GPU utilization diagnosis

Context:
- User observed low GPU utilization after restarting ACT-Lang-B full training.

Observed run:
- Run directory:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_B/20260606_111107_act_lang_B_full_gpu0`
- Log:
  - `/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_B/logs/20260606_111107_act_lang_B_full_gpu0.log`
- Main training PID:
  - `1459564`
- DataLoader workers:
  - `1459835`, `1459836`, `1459837`, `1459838`

Diagnosis:
- A single `nvidia-smi` snapshot showed GPU0 at `0%` utilization with about `2883 MiB` allocated.
- Process inspection showed the four DataLoader workers were each near `98-100%` CPU.
- A follow-up snapshot 30 seconds later showed GPU0 at `75%` utilization.
- Interpretation: training is not dead; it is bursty. The model computes on GPU briefly, then waits for CPU/dataloader/image decoding/parquet reads.
- Current input pipeline is likely the bottleneck, not the ACT-Lang model itself.

Progress at diagnosis:
- Step 100: ETA about `36.48h`
- Step 200: ETA about `36.45h`
- Step 300: ETA about `36.39h`

Recommendation:
- Do not change batch size first because batch size is part of the experiment's training setup.
- If throughput remains poor after more warmup, the least invasive restart is increasing only `training.num_workers` for all ACT-Lang full configs, e.g. from `4` to `8` or `12`.
- This changes input pipeline parallelism but not model architecture, optimizer, loss, batch size, or dataset split.

### 2026-06-06 UTC - ACT-Lang full configs switched to 12 DataLoader workers

Context:
- User requested stopping the current B run, deleting its training artifacts, and changing CPU/DataLoader workers to 12.

Stopped and deleted:
- Stopped active ACT-Lang-B run:
  - PID: `1459564`
  - run directory: `/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_B/20260606_111107_act_lang_B_full_gpu0`
  - log file: `/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_B/logs/20260606_111107_act_lang_B_full_gpu0.log`
- Confirmed no remaining GPU training process.
- Confirmed only empty `act_lang_B/logs/` directory remains in the ACT-Lang-B run group.

Config changes:
- Set `training.num_workers: 12` in:
  - `project/configs/act_lang_B_full.yaml`
  - `project/configs/act_lang_ABC_full.yaml`
  - `project/configs/act_lang_ABC_size_matched_full.yaml`

Rationale:
- This is an input-pipeline-only change.
- It does not change ACT architecture, language fusion, dataset split, batch size, optimizer, loss, number of steps, validation frequency, or checkpoint frequency.
- Applied to B/ABC/ABC-size-matched together to preserve full-training config alignment.

Self-check:
- YAML parsing passed.
- All three ACT-Lang full configs now report `num_workers=12`.
- Shared policy fields still match across B, ABC full, and ABC size-matched.
- Shared training fields still match across B, ABC full, and ABC size-matched.
- `bash -n` passed for ACT-Lang B and ABC launchers.

Clean restart command:

```bash
cd /home/zengzixuan/cvprojects/lerobot
source /home/zengzixuan/cvprojects/calvin_env.sh
GPU_ID=0 bash project/scripts/nohup_train_act_lang_B.sh
```
