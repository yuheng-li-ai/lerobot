# Project State

## Current Phase

Phase 2 - Task 2 full training complete, summarized, visualized, and compared; ready for Phase 3 zero-shot D evaluation.

## Status Summary

- LeRobot source is present.
- ACT policy implementation is present.
- CALVIN framework repo is present.
- Raw CALVIN `task_ABC_D` is present at the configured SSD path.
- Raw CALVIN A/B/C/D partitioning is now identified.
- LeRobot conversion scripts are prepared, smoke-tested, user-run, and verified.
- Converted LeRobot-format CALVIN datasets B, ABC, and D are present and readable.
- Sandbox-external CUDA/GPU checks pass with 8 idle RTX A6000 GPUs.
- ACT-B mini-trial has been run and self-checked.
- ACT-B full-scale training was launched manually by the user and completed.
- ACT-B non-training extension analyses are complete: overfitting, checkpoint selection, B visual statistics, B samples, and B action distribution.
- ACT-B data-augmentation harness is prepared, augmentation verification passed, 20-step mini-trial passed, and manual full-training launcher is ready.
- Task 2 ACT-ABC, ACT-ABC-aug, ACT-ABC-size-matched, and ACT-ABC-size-matched-aug configs and manual launchers are prepared.
- Task 2 full-training launchers default to GPUs 1-4 and refuse GPU0.
- Task 2 four-way mini-trial harness passed on GPUs 1-4 while GPU0 remained reserved for ACT-B augmentation.
- Task 2 four full runs completed and were summarized.
- Task 2 per-model diagnostic visualizations were generated in separate folders.
- Task 2 requested cross-model comparisons are now generated, including loss, overfitting, convergence, data scale, action/chunk/gripper, and visual diagnostics.
- Task 1 B-augmentation full run completed and was summarized.
- Phase 3 selected-checkpoint offline D action-error evaluation is implemented and has been run for all six existing non-language ACT checkpoints.
- CALVIN official success-rate evaluation is deferred to new language-conditioned ACT models because the current checkpoints do not accept language goals.
- Core training/validation curves for all six completed runs have been retrospectively synchronized to W&B as native W&B curves.

## Verified Paths

| Item | Path | Status |
|---|---|---|
| LeRobot source | `/home/zengzixuan/cvprojects/lerobot` | present |
| CALVIN framework | `/home/zengzixuan/cvprojects/calvin` | present |
| Raw CALVIN ABC->D | `/SSD_DISK/users/zengzixuan/calvin/task_ABC_D` | present |
| Converted LeRobot CALVIN root | `/EXT_DISK/users/zengzixuan/processed-calvin` | present, contains converted B/ABC/D datasets |
| Training output root | `/EXT_DISK/users/zengzixuan/calvin_runs` | present, contains ACT-B full run and caches |
| Legacy CALVIN repo dataset root | `/home/zengzixuan/cvprojects/calvin/dataset` | present, no split data found |
| CALVIN downloader | `/home/zengzixuan/cvprojects/calvin/dataset/download_data.sh` | present |
| Environment configuration | `/home/zengzixuan/cvprojects/calvin_env.sh` | authoritative setup; unsets deprecated `LEROBOT_HOME` |
| Research workspace | `/home/zengzixuan/cvprojects/lerobot/project` | present |

## Environment Review

| Item | Result |
|---|---|
| Existing Python env | `/home/zengzixuan/miniforge3/envs/lerobot/bin/python` |
| LeRobot | 0.5.2 |
| PyTorch | 2.11.0+cu128 |
| datasets | 4.8.5 |
| Torchvision ResNet18 cache | `/EXT_DISK/users/zengzixuan/calvin_runs/torch_cache/hub/checkpoints/resnet18-f37072fd.pth` |
| CUDA outside sandbox | available |
| GPUs outside sandbox | 8 x NVIDIA RTX A6000, 49,140 MiB each |
| ACT default chunk | `chunk_size=100`, `n_action_steps=100` |
| Required env fix | `unset LEROBOT_HOME; export HF_LEROBOT_HOME=$CALVIN_LEROBOT_ROOT; export TORCH_HOME=$CALVIN_RUNS/torch_cache` |

Environment correction:
- `datasets` was upgraded in the existing conda env to match LeRobot's declared dependency range.
- No LeRobot source compatibility patch is required for dataset reading.
- `pip check` passed after the upgrade.
- `/home/zengzixuan/cvprojects/calvin_env.sh` now unsets deprecated `LEROBOT_HOME`, exports `LEROBOT_PYTHON`, configures HF caches, configures `TORCH_HOME`, and prepends LeRobot `src` to `PYTHONPATH`.

## Raw Dataset Review

| Split | Episodes | Frames / Timesteps | Language Sequences | Unique Tasks |
|---|---:|---:|---:|---:|
| `task_ABC_D/training` | 147 | 1,795,045 | 17,870 | 34 |
| `task_ABC_D/validation` | 4 | 99,022 | 1,087 | 34 |

## Environment Partitions

| Environment | Source | Frame Range | Frames | Raw Episodes / Segments |
|---|---|---:|---:|---:|
| A | `calvin_scene_A` training | 1,191,339-1,795,044 | 603,706 | 46 |
| B | `calvin_scene_B` training | 0-598,909 | 598,910 | 67 |
| C | `calvin_scene_C` training | 598,910-1,191,338 | 592,429 | 34 |
| D | validation | non-contiguous validation episodes | 99,022 | 4 |

## Phase 0 Artifacts

| Artifact | Path |
|---|---|
| Dataset allocation config | `project/configs/calvin_datasets.yaml` |
| Phase 0 stats script | `project/scripts/prepare_dataset_stats.py` |
| LeRobot converter | `project/scripts/convert_calvin_to_lerobot.py` |
| Manual conversion launcher | `project/scripts/run_phase0_conversion.sh` |
| Dataset stats table | `project/tables/dataset_stats_ABC.csv` |
| Task counts table | `project/tables/task_counts_ABCD.csv` |
| A/B/C/D sample figure | `project/figures/env_samples_ABCD.png` |
| Smoke converted datasets | `project/outputs/phase0_smoke` |

## Phase 3 Artifacts

| Artifact | Path |
|---|---|
| Zero-shot D offline evaluator | `project/scripts/eval_zero_shot_D.py` |
| Manual nohup launcher | `project/scripts/nohup_eval_zero_shot_D.sh` |
| Selected-checkpoint D results | `project/tables/task3/zero_shot_D_results.csv` |
| Selected-checkpoint chunk diagnostics | `project/tables/task3/zero_shot_D_action_chunks.csv` |
| Chunk horizon table | `project/tables/task3/zero_shot_D_chunk_horizon.csv` |
| D action L1 figure | `project/figures/task3/zero_shot_action_l1_D.png` |
| D action smoothness figure | `project/figures/task3/action_smoothness_D.png` |
| D chunk boundary figure | `project/figures/task3/chunk_boundary_jump_D.png` |
| D chunk horizon figure | `project/figures/task3/chunk_horizon_error_D.png` |
| Success-rate model/data requirements | `project/TASK3_SUCCESS_RATE_MODEL_REQUIREMENTS.md` |

## W&B Training Curve Artifacts

| Artifact | Path |
|---|---|
| Retrospective W&B logging script | `project/scripts/log_completed_runs_to_wandb.py` |
| W&B run manifest | `project/tables/wandb_training_curve_runs.csv` |
| W&B local export script | `project/scripts/export_wandb_training_curves.py` |
| W&B exported figures | `project/figures/wandb_export` |
| W&B downloaded histories | `project/tables/wandb_export` |

W&B project:
- `calvin-act-generalization`

W&B group:
- `task1_task2_training`

## Phase 1 ACT-B Mini-Trial

| Item | Value |
|---|---|
| Config | `project/configs/act_B.yaml` |
| Harness | `project/scripts/train_act.py` |
| Output dir | `project/outputs/act_B/mini_trial` |
| Dataset | `local/calvin_B` |
| Dataset root | `/EXT_DISK/users/zengzixuan/processed-calvin/calvin_B` |
| Train episodes | `[0, 1, 2, 3]` |
| Validation episodes | `[4, 5]` |
| Train frames | 7,912 |
| Validation frames | 4,878 |
| Device | `cuda:0` |
| Steps | 20 |
| Batch size | 2 |
| Chunk size | 100 |
| Backbone initialization | `ResNet18_Weights.IMAGENET1K_V1` |
| Final train Action L1 | 0.8123 |
| Final validation Action L1 | 0.6469 |
| Self-check | passed |

Mini-trial artifacts:
- `project/outputs/act_B/mini_trial/metrics.csv`
- `project/outputs/act_B/mini_trial/self_check.json`
- `project/outputs/act_B/mini_trial/checkpoint/model.safetensors`
- `project/outputs/act_B/mini_trial/checkpoint/training_state.pt`
- `project/outputs/act_B/mini_trial/checkpoint/policy_preprocessor.json`
- `project/outputs/act_B/mini_trial/checkpoint/policy_postprocessor.json`

## Phase 1 ACT-B Full-Training Preparation

| Item | Value |
|---|---|
| Full config | `project/configs/act_B_full.yaml` |
| Manual launcher | `project/scripts/nohup_train_act_B.sh` |
| Train episodes | `0:212` |
| Validation episodes | `212:235` |
| Train frames | 535,403 |
| Validation frames | 63,507 |
| Batch size | 32 |
| Steps | 100,000 |
| Approximate train-frame passes | 5.98 |
| Validation frequency | every 5,000 steps |
| Checkpoint frequency | every 10,000 steps plus final |
| Progress in log | `PROGRESS act_B_full [####----] ...` every 100 steps |
| Measured peak reserved memory | 2.107 GB for batch size 32 one-batch probe |
| Expected full-run reserved memory | roughly 3-6 GB |
| Estimated runtime | roughly 6-12 hours |
| Estimated checkpoint disk | roughly 5.9 GB |
| Full-scale training launched by agent | no |

Manual launch:
```bash
GPU_ID=0 bash project/scripts/nohup_train_act_B.sh
```

After launch, the script prints the exact log path. Monitor with:
```bash
tail -f /EXT_DISK/users/zengzixuan/calvin_runs/act_B/logs/<RUN_ID>.log
```

## Phase 1 ACT-B Full-Training Result

| Item | Value |
|---|---|
| Run dir | `/EXT_DISK/users/zengzixuan/calvin_runs/act_B/20260604_101320_act_B_full_gpu0` |
| Log | `/EXT_DISK/users/zengzixuan/calvin_runs/act_B/logs/20260604_101320_act_B_full_gpu0.log` |
| Steps completed | 100,000 |
| Metrics rows | 100,000 |
| Self-check | passed |
| Final train loss | 0.4588 |
| Final train Action L1 | 0.4586 |
| Final validation loss | 0.6230 |
| Final validation Action L1 | 0.6229 |
| Best validation Action L1 | 0.5404 at step 10,000 |
| Final checkpoint | `/EXT_DISK/users/zengzixuan/calvin_runs/act_B/20260604_101320_act_B_full_gpu0/checkpoint` |
| Periodic checkpoints | steps 10k through 90k |
| Full run disk usage | 5.8 GB |
| Log errors | none found |
| Training process | exited |
| GPU after completion | idle |

Interpretation:
- ACT-B fit environment B substantially.
- Best validation Action L1 occurred at step 10k, while final validation Action L1 was higher. This is an overfitting warning and should be included in Task 1 analysis.
- Keep both final and step-10k checkpoints for downstream evaluation/analysis.

## Phase 1 ACT-B Non-Training Extensions

| Item | Value |
|---|---|
| Analysis script | `project/scripts/analyze_act_B_extensions.py` |
| Loss curve figure | `project/figures/act_B_baseline/loss_curve_act_B.png` |
| Smoothed train L1 table | `project/tables/act_B_train_l1_smoothed.csv` |
| Train-val gap figure | `project/figures/act_B_baseline/train_val_gap_act_B.png` |
| Environment B samples | `project/figures/act_B_baseline/env_B_samples.png` |
| Action distribution figure | `project/figures/act_B_baseline/env_B_action_distribution.png` |
| Action smoothness figure | `project/figures/act_B_baseline/env_B_action_smoothness.png` |
| Chunk baseline figure | `project/figures/act_B_baseline/env_B_chunk_baseline.png` |
| Action violin figure | `project/figures/act_B_baseline/env_B_action_violin.png` |
| Action delta heatmap | `project/figures/act_B_baseline/env_B_action_delta_heatmap.png` |
| Gripper diagnostics | `project/figures/act_B_baseline/env_B_gripper_diagnostics.png` |
| Gripper timeline | `project/figures/act_B_baseline/env_B_gripper_timeline.png` |
| Visual color profile | `project/figures/act_B_baseline/env_B_visual_color_profile.png` |
| Brightness/contrast histogram | `project/figures/act_B_baseline/env_B_brightness_contrast_hist.png` |
| Task frequency figure | `project/figures/act_B_baseline/env_B_task_frequency.png` |
| Representative trajectory strip | `project/figures/act_B_baseline/env_B_representative_trajectory_strip.png` |
| Overfitting summary | `project/tables/act_B_overfitting_summary.csv` |
| Train-val gap table | `project/tables/act_B_train_val_gap.csv` |
| Checkpoint selection | `project/tables/act_B_checkpoint_selection.csv` |
| B visual stats | `project/tables/env_B_visual_stats.csv` |
| B action summary | `project/tables/env_B_action_summary.csv` |
| B action stats | `project/tables/env_B_action_stats.csv` |
| B action smoothness | `project/tables/env_B_action_smoothness.csv` |
| B gripper summary | `project/tables/env_B_gripper_summary.csv` |
| B gripper runs | `project/tables/env_B_gripper_runs.csv` |
| B chunk baseline | `project/tables/env_B_chunk_baseline.csv` |
| B chunk summary | `project/tables/env_B_chunk_baseline_summary.csv` |
| B action delta heatmap table | `project/tables/env_B_action_delta_heatmap.csv` |
| B task counts | `project/tables/env_B_task_counts.csv` |
| B representative trajectory table | `project/tables/env_B_representative_trajectory.csv` |

Key extension results:
- Best validation Action L1 was `0.5404` at step `10,000`; final validation Action L1 was `0.6229`.
- The final-minus-best validation gap is `+0.0825`, so the full ACT-B run has a clear overfitting/early-stopping signal.
- Loss-curve display now uses a 1,000-step trailing mean for train Action L1, with raw mini-batch train Action L1 kept as a faint reference.
- Smoothed train Action L1 decreases from `0.7553` at step 1k to `0.4467` at step 100k.
- Recommended ACT-B checkpoints to keep for later comparison:
  - early-stopped candidate: `/EXT_DISK/users/zengzixuan/calvin_runs/act_B/20260604_101320_act_B_full_gpu0/checkpoints/step_00010000`
  - final endpoint: `/EXT_DISK/users/zengzixuan/calvin_runs/act_B/20260604_101320_act_B_full_gpu0/checkpoint`
- Environment B static-camera baseline from 2,000 sampled frames:
  - RGB mean `(0.7977, 0.6982, 0.5858)`
  - RGB std `(0.1858, 0.1848, 0.2180)`
  - brightness mean/std `0.6939 / 0.0126`
  - contrast mean/std `0.1731 / 0.0040`
- Environment B action baseline over 598,910 frames:
  - mean L2 norm of first 6 action dimensions `0.5027`
  - mean consecutive action delta L2 over first 6 dimensions `0.1781`
  - gripper close/open fractions `0.4705 / 0.5295`
  - gripper switch count `14,142`
- Train-validation generalization gap:
  - computed as validation Action L1 minus trailing 1,000-step mean train Action L1
  - step 40k smoothed gap `+0.0193`
  - step 100k smoothed gap `+0.1762`
- Chunk baseline with `chunk_size=100`:
  - chunks analyzed `6,022`
  - mean within-chunk delta L2 over first 6 dimensions `0.1784`
  - mean boundary-style jump L2 over first 6 dimensions `0.1792`
  - q90/q99 boundary-style jump L2 `0.3245 / 0.6073`
- Gripper diagnostics:
  - gripper values are strictly binary: close `-1.0`, open `+1.0`
  - close/open fractions `0.4705 / 0.5295`
  - switch count `14,142`
  - switch rate `23.61` per 1,000 frames
  - median run length `38` frames

## Phase 2 Initial Configuration

| Item | Value |
|---|---|
| Split/config generator | `project/scripts/prepare_task2_configs.py` |
| Episode split config | `project/configs/task2_episode_splits.yaml` |
| Episode split table | `project/tables/task2_episode_splits.csv` |
| Experiment matrix table | `project/tables/task2_experiment_matrix.csv` |
| ACT-ABC mini config | `project/configs/act_ABC.yaml` |
| ACT-ABC full config | `project/configs/act_ABC_full.yaml` |
| ACT-ABC-aug mini config | `project/configs/act_ABC_aug.yaml` |
| ACT-ABC-aug full config | `project/configs/act_ABC_aug_full.yaml` |
| ACT-ABC-size-matched mini config | `project/configs/act_ABC_size_matched.yaml` |
| ACT-ABC-size-matched full config | `project/configs/act_ABC_size_matched_full.yaml` |
| ACT-ABC-size-matched-aug mini config | `project/configs/act_ABC_size_matched_aug.yaml` |
| ACT-ABC-size-matched-aug full config | `project/configs/act_ABC_size_matched_aug_full.yaml` |
| Generic manual launcher | `project/scripts/nohup_train_task2.sh` |
| ACT-ABC launcher | `project/scripts/nohup_train_act_ABC.sh` |
| ACT-ABC-aug launcher | `project/scripts/nohup_train_act_ABC_aug.sh` |
| ACT-ABC-size-matched launcher | `project/scripts/nohup_train_act_ABC_size_matched.sh` |
| ACT-ABC-size-matched-aug launcher | `project/scripts/nohup_train_act_ABC_size_matched_aug.sh` |

Task 2 GPU policy:
- GPU0 is reserved for ACT-B augmentation and must not be used by Task 2.
- `nohup_train_task2.sh` refuses `GPU_ID=0`.
- Defaults are GPU1 for ACT-ABC, GPU2 for ACT-ABC-size-matched, GPU3 for ACT-ABC-aug, and GPU4 for ACT-ABC-size-matched-aug.

Task 2 split summary:

| Split | Train Episodes | Train Frames | Validation Episodes | Validation Frames |
|---|---:|---:|---:|---:|
| ACT-ABC full | 611 | 1,609,097 | 68 | 185,948 |
| ACT-ABC-size-matched | 207 | 535,456 | 68 | 185,948 |

Task 2 initial verification:
- Python compile passed for updated/generated Task 2 scripts.
- YAML parse passed for all Task 2 configs.
- Bash syntax passed for all Task 2 launchers.
- GPU0 refusal path was tested and exited before launching `nohup`.

## Phase 2 Mini-Trial Harness

| Item | Value |
|---|---|
| Harness | `project/scripts/nohup_task2_mini_trials.sh` |
| Run id | `20260605_071512_task2_mini` |
| Logs | `project/logs/task2/mini_trials/20260605_071512_task2_mini/` |
| Results table | `project/tables/task2_mini_trial_results.csv` |
| Tail helper | `project/scripts/tail_task2_latest_logs.sh mini` |
| Full staggered launcher | `project/scripts/nohup_task2_full_parallel.sh` |
| Visualizations generated | none |
| Full training launched by agent | no |

| Experiment | Physical GPU | Output Dir | Self-check | Final Val Action L1 |
|---|---:|---|---|---:|
| `act_ABC` | 1 | `project/outputs/task2/act_ABC/mini_trials/20260605_071512_task2_mini` | passed | 0.9289 |
| `act_ABC_size_matched` | 2 | `project/outputs/task2/act_ABC_size_matched/mini_trials/20260605_071512_task2_mini` | passed | 0.9292 |
| `act_ABC_aug` | 3 | `project/outputs/task2/act_ABC_aug/mini_trials/20260605_071512_task2_mini` | passed | 0.9163 |
| `act_ABC_size_matched_aug` | 4 | `project/outputs/task2/act_ABC_size_matched_aug/mini_trials/20260605_071512_task2_mini` | passed | 0.9149 |

Task 2 augmentation exposure:
- `act_ABC_size_matched_aug_full` uses the same per-sample augmentation strength as Task 1 and has `1.0001x` Task 1 augmented frame exposure.
- `act_ABC_aug_full` uses the same per-sample augmentation strength as Task 1 and has `3.0054x` Task 1 augmented frame exposure because the full ABC train split is larger.
- Source table: `project/tables/task2_augmentation_exposure.csv`.

## Current Parallel Full Runs

As of 2026-06-05 07:46 UTC, five full-scale runs are active:

| Experiment | GPU | PID | Log |
|---|---:|---:|---|
| `act_B_aug_full` | 0 | 3450887 | `/EXT_DISK/users/zengzixuan/calvin_runs/act_B_aug/logs/20260605_063735_act_B_aug_full_gpu0.log` |
| `act_ABC_full` | 1 | 70946 | `/EXT_DISK/users/zengzixuan/calvin_runs/act_ABC/logs/20260605_072243_act_ABC_full_gpu1.log` |
| `act_ABC_size_matched_full` | 2 | 1674751 | `/EXT_DISK/users/zengzixuan/calvin_runs/act_ABC_size_matched/logs/20260605_072743_act_ABC_size_matched_full_gpu2.log` |
| `act_ABC_aug_full` | 3 | 2163153 | `/EXT_DISK/users/zengzixuan/calvin_runs/act_ABC_aug/logs/20260605_073244_act_ABC_aug_full_gpu3.log` |
| `act_ABC_size_matched_aug_full` | 4 | 3093463 | `/EXT_DISK/users/zengzixuan/calvin_runs/act_ABC_size_matched_aug/logs/20260605_073744_act_ABC_size_matched_aug_full_gpu4.log` |

Parallelism check:
- `vmstat` showed IO wait at `0%`.
- Open file descriptors were low: 79-97 per process against a soft limit of 1024.
- No IO or Python errors found in inspected log tails.
- `project/scripts/tail_task2_latest_logs.sh full-all` can now tail all active Task 2 full logs.

Step-5000 validation check:
- `act_ABC_full` reached and completed validation at step 5000.
- Step-5000 validation Action L1: `0.5884`.
- Validation step took `27.89s`, then normal training step time resumed around `0.17-0.18s`.
- IO wait remained `0%`; no open-file pressure or log errors observed.
- Next recommended check: near step 10000 checkpoint writing.

Step-10000 checkpoint check:
- `act_ABC_full` reached step 10000 and wrote a complete `592M` checkpoint at `/EXT_DISK/users/zengzixuan/calvin_runs/act_ABC/20260605_072243_act_ABC_full_gpu1/checkpoints/step_00010000/`.
- Step-10000 validation Action L1: `0.5698`.
- Step 10000 took `26.36s`, then normal training step time resumed around `0.14-0.16s`.
- `act_B_aug_full` also has a complete `592M` step-10000 checkpoint.
- The remaining Task 2 jobs had not reached step 10000 at the check.
- IO wait remained `0%`; no open-file pressure or log errors observed.

## Phase 2 Full Training Results

| Experiment | Final Val Action L1 | Best Val Step | Best Val Action L1 | Selected Available Checkpoint |
|---|---:|---:|---:|---|
| `act_ABC` | 0.5810 | 25,000 | 0.5552 | `/EXT_DISK/users/zengzixuan/calvin_runs/act_ABC/20260605_072243_act_ABC_full_gpu1/checkpoints/step_00030000` |
| `act_ABC_size_matched` | 0.6165 | 15,000 | 0.5653 | `/EXT_DISK/users/zengzixuan/calvin_runs/act_ABC_size_matched/20260605_072743_act_ABC_size_matched_full_gpu2/checkpoints/step_00010000` |
| `act_ABC_aug` | 0.5872 | 25,000 | 0.5559 | `/EXT_DISK/users/zengzixuan/calvin_runs/act_ABC_aug/20260605_073244_act_ABC_aug_full_gpu3/checkpoints/step_00030000` |
| `act_ABC_size_matched_aug` | 0.6304 | 15,000 | 0.5669 | `/EXT_DISK/users/zengzixuan/calvin_runs/act_ABC_size_matched_aug/20260605_073744_act_ABC_size_matched_aug_full_gpu4/checkpoints/step_00010000` |

Full-run summary artifacts:
- `project/tables/task2_full_training_summary.csv`
- `project/tables/full_training_summary_with_B_aug.csv`
- `project/tables/task2_checkpoint_candidates.csv`
- `project/tables/model_selection_checkpoints.csv`
- `project/tables/task2_internal_effects.csv`

Per-model visualization artifacts:
- Training diagnostics script: `project/scripts/analyze_task2_visuals.py`
- Extended dataset/action diagnostics script: `project/scripts/analyze_task2_extended_visuals.py`
- Manifest: `project/tables/task2_visualization_manifest.csv`
- Extended manifest: `project/tables/task2_extended_visualization_manifest.csv`
- Figure folders:
  - `project/figures/task2/act_ABC/`
  - `project/figures/task2/act_ABC_size_matched/`
  - `project/figures/task2/act_ABC_aug/`
  - `project/figures/task2/act_ABC_size_matched_aug/`
- Table folders:
  - `project/tables/task2/act_ABC/`
  - `project/tables/task2/act_ABC_size_matched/`
  - `project/tables/task2/act_ABC_aug/`
  - `project/tables/task2/act_ABC_size_matched_aug/`
- Figure counts:
  - `act_ABC`: 17 PNGs
  - `act_ABC_size_matched`: 17 PNGs
  - `act_ABC_aug`: 18 PNGs
  - `act_ABC_size_matched_aug`: 18 PNGs
  - total: 70 PNGs
- Each model folder contains training diagnostics:
  - `loss_curve.png`
  - `train_val_gap.png`
  - `checkpoint_selection.png`
  - `step_time_profile.png`
  - `dataset_split.png`
- Each model folder also contains dataset/action diagnostics:
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
- Augmented model folders additionally contain `augmentation_verification.png`.
- Figure title clipping QA completed on 2026-06-06:
  - all generated PNGs were checked for dimensions.
  - representative long-title, sample-grid, augmentation, and trajectory figures were visually inspected.
  - titles, legends, and headers were not clipped.
- Cross-model comparison figures have now been generated in the comparison folder below.

Comparison artifacts:
- Script: `project/scripts/compare_task2_results.py`
- Figure folder: `project/figures/task2/comparisons/`
- Table folder: `project/tables/task2/comparisons/`
- Root pairwise effects table: `project/tables/task2_pairwise_effects.csv`
- Comparison figures:
  - `best_final_val_action_l1.png`
  - `overfit_convergence_summary.png`
  - `requested_pairwise_loss_curves.png`
  - `data_action_visual_summary.png`
  - `pairwise_relative_delta_heatmap.png`
- Comparison tables:
  - `comparison_model_summary.csv`
  - `pairwise_comparison_metrics.csv`
  - `pairwise_loss_overfit_convergence.csv`
  - `pairwise_key_findings.csv`
  - matching LaTeX files for model summary, key findings, and loss/overfit/convergence.
- Requested pairwise comparisons:
  - `act_ABC` vs `act_ABC_aug`
  - `act_ABC` vs `act_ABC_size_matched`
  - `act_ABC` vs `act_B`
  - `act_ABC_size_matched` vs `act_B`
  - `act_ABC_size_matched_aug` vs `act_B_aug`
- Comparability rule:
  - `ABC` vs `ABC_aug` and `ABC` vs `ABC_size_matched` use the same ABC validation split and can be interpreted directly for supervised validation.
  - B-vs-ABC loss comparisons use different validation splits and are diagnostic only; final generalization claims should come from D zero-shot evaluation.

Interpretation snapshot:
- On the shared ABC validation split, `act_ABC` is better than `act_ABC_size_matched` by `0.0101` best validation Action L1.
- `act_ABC_aug` is slightly worse than `act_ABC` by `0.0007`; no meaningful augmentation gain.
- `act_ABC_size_matched_aug` is slightly worse than `act_ABC_size_matched` by `0.0016`; no meaningful augmentation gain.
- Task 1 `act_B_aug` also did not improve B validation: best `0.5409` vs B baseline best `0.5404`.
- Use selected available checkpoints for D evaluation, because best validation steps often do not have saved checkpoints.
- Representative trajectory strip:
  - uses a contiguous 1,200-frame window `478200-479399` from source B segment `456757-500843`
  - raw action norm is kept as a faint reference
  - 31-frame trailing mean action norm is the main displayed curve
- Failure taxonomy is intentionally deferred until D zero-shot evaluation exists.

## Phase 1 ACT-B Data Augmentation

| Item | Value |
|---|---|
| Augmented mini config | `project/configs/act_B_aug.yaml` |
| Augmented full config | `project/configs/act_B_aug_full.yaml` |
| Training harness | `project/scripts/train_act.py` |
| Augmentation verifier | `project/scripts/verify_act_B_augmentation.py` |
| Manual launcher | `project/scripts/nohup_train_act_B_aug.sh` |
| Verification figure | `project/figures/act_B_aug/act_B_aug_verification.png` |
| Verification table | `project/tables/act_B_aug_verification.csv` |
| Mini-trial output | `project/outputs/act_B_aug/mini_trial` |

Augmentation settings:
- brightness `0.12`
- contrast `0.12`
- saturation `0.08`
- Gaussian noise std `0.01`
- probability `1.0`
- image values clipped to `[0.0, 1.0]`

Verification results:
- static camera mean/max absolute pixel delta `0.02936 / 0.10341`
- gripper camera mean/max absolute pixel delta `0.01769 / 0.08147`
- original and augmented ranges stayed within `[0.0, 1.0]`

Mini-trial result:
- selected GPU 0 after confirming all 8 RTX A6000 GPUs were idle.
- 20-step run completed.
- progress-bar lines printed as `PROGRESS act_B_aug [...]`.
- self-check passed.
- final train Action L1 `0.8117`
- final validation Action L1 `0.6636`
- post-run GPUs idle.

Manual launch:
```bash
GPU_ID=0 bash project/scripts/nohup_train_act_B_aug.sh
```

Monitor:
```bash
tail -f /EXT_DISK/users/zengzixuan/calvin_runs/act_B_aug/logs/<RUN_ID>.log
```

The full-training log prints lines like:
```text
PROGRESS act_B_aug_full [########------------------------] 25000/100000 ( 25.0%) elapsed=... eta=...
```

Full-scale augmentation training launched by agent: no.

Full-training completion:
- User manually launched ACT-B-aug full training on GPU 0.
- Run dir: `/EXT_DISK/users/zengzixuan/calvin_runs/act_B_aug/20260605_063735_act_B_aug_full_gpu0`
- Log: `/EXT_DISK/users/zengzixuan/calvin_runs/act_B_aug/logs/20260605_063735_act_B_aug_full_gpu0.log`
- Steps completed: `100,000`
- Runtime from progress log: `18.67h`
- Self-check: passed
- Metrics rows: `100,000`
- Final train loss `0.4853`
- Final train Action L1 `0.4852`
- Final validation loss `0.6487`
- Final validation Action L1 `0.6486`
- Best validation Action L1 `0.5409` at step `15,000`
- Step `15,000` is metric-only; no checkpoint was saved at that step because save frequency was every `10,000` steps.
- Saved candidates around the best metric:
  - step 10k validation Action L1 `0.5478`
  - step 20k validation Action L1 `0.5516`
- ACT-B-aug best saved checkpoint: step 10k, because it beats the nearest saved step 20k checkpoint.
- Best saved ACT-B-aug checkpoint path: `/EXT_DISK/users/zengzixuan/calvin_runs/act_B_aug/20260605_063735_act_B_aug_full_gpu0/checkpoints/step_00010000`
- Final-minus-best validation Action L1 `+0.1076`
- Final checkpoint: `/EXT_DISK/users/zengzixuan/calvin_runs/act_B_aug/20260605_063735_act_B_aug_full_gpu0/checkpoint`
- Periodic checkpoints: steps 10k through 90k
- Full run disk usage: about `5.8G`
- Log errors: none found in inspected error scan
- Training process: exited
- GPU after completion: idle

Comparison to ACT-B baseline:
- ACT-B best validation Action L1 `0.5404` at step 10k.
- ACT-B-aug best validation Action L1 `0.5409` at step 15k.
- ACT-B final validation Action L1 `0.6229`.
- ACT-B-aug final validation Action L1 `0.6486`.
- Augmentation did not improve B validation in this full run; zero-shot D remains the more important test for augmentation.

One-to-one visualization completion:
- Analysis script: `project/scripts/analyze_act_B_aug_extensions.py`
- Coverage manifest: `project/tables/act_B_aug_visualization_coverage.csv`
- Coverage entries: `29`
- Missing artifacts in coverage check: `0`
- Comparison table: `project/tables/act_B_vs_aug_summary.csv`
- Generated ACT-B-aug metric figures/tables:
  - `project/figures/act_B_aug/act_B_aug_loss_curve.png`
  - `project/figures/act_B_aug/act_B_aug_train_val_gap.png`
  - `project/tables/act_B_aug_overfitting_summary.csv`
  - `project/tables/act_B_aug_checkpoint_selection.csv`
  - `project/tables/act_B_aug_train_l1_smoothed.csv`
  - `project/tables/act_B_aug_train_val_gap.csv`
- Generated ACT-B-aug visual augmentation outputs:
  - `project/figures/act_B_aug/act_B_aug_samples.png`
  - `project/figures/act_B_aug/act_B_aug_visual_color_profile.png`
  - `project/figures/act_B_aug/act_B_aug_brightness_contrast_hist.png`
  - `project/tables/act_B_aug_visual_stats.csv`
- Generated ACT-B-aug data-invariant counterparts for action distribution, smoothness, chunk baseline, action violin, action delta heatmap, task frequency, representative trajectory, and separate gripper diagnostics.
- Figure folders for report assembly:
  - baseline: `project/figures/act_B_baseline/` (`14` PNG files)
  - augmentation: `project/figures/act_B_aug/` (`15` PNG files, including augmentation verification)
  - index: `project/tables/act_B_figure_folder_index.csv`
- Root-level duplicate ACT-B/ACT-B-aug figures were deleted after the folder split; `project/figures/env_samples_ABCD.png` remains at root because it is the Phase 0 overview figure.
- Figure title clipping QA completed on 2026-06-06:
  - brightness/contrast, gripper diagnostics, visual color profile, and augmentation verification were regenerated with safer layout/padding.
  - affected PNGs were visually checked; titles are no longer clipped.
  - `project/tables/act_B_aug_visualization_coverage.csv` now points to the current subfolder paths and has `0` missing artifacts.

Converter basis:
- Uses official LeRobot writer API: `LeRobotDataset.create`, `add_frame`, `save_episode`, `finalize`.
- CALVIN field mapping is a project adapter because no CALVIN-specific LeRobot converter exists in this worktree.

## Manual Conversion Command

```bash
bash project/scripts/run_phase0_conversion.sh
```

Expected outputs:

| Dataset | Repo ID | Root |
|---|---|---|
| ACT-B train data | `local/calvin_B` | `/EXT_DISK/users/zengzixuan/processed-calvin/calvin_B` |
| ACT-ABC train data | `local/calvin_ABC` | `/EXT_DISK/users/zengzixuan/processed-calvin/calvin_ABC` |
| D eval data | `local/calvin_D` | `/EXT_DISK/users/zengzixuan/processed-calvin/calvin_D` |

## Converted Dataset Verification

| Dataset | Episodes / Segments | Frames | Tasks | Readback |
|---|---:|---:|---:|---|
| `local/calvin_B` | 235 | 598,910 | 1 | passed |
| `local/calvin_ABC` | 679 | 1,795,045 | 3 | passed |
| `local/calvin_D` | 35 | 99,022 | 1 | passed |

Readback schema:
- `observation.images.static`: `(3, 200, 200)`, `torch.float32`
- `observation.images.gripper`: `(3, 84, 84)`, `torch.float32`
- `observation.state`: `(15,)`, `torch.float32`
- `action`: `(7,)`, `torch.float32`

## Important Constraints

- Agents must not launch full-scale training.
- Agents may run only mini-trials, smoke tests, data inspection, figure/table generation, and analysis scripts.
- Full training scripts must be generated for manual execution.
- ACT-B and ACT-ABC must keep identical architecture and hyperparameters unless an explicit ablation is active.
- All experiment artifacts belong under `project/`.

## Open Requirements

- Finish the official-success-rate ABC training dataset:
  - `local/calvin_lang_ABC`
- Train new language-conditioned models rather than patching old checkpoints:
  - `ACT-Lang-B`
  - `ACT-Lang-ABC`
- Decide rollout `n_action_steps` for `ACT-Lang-*`; all CALVIN language segments are at most 65 frames, so `chunk_size=100` requires padding-aware loss and may not justify executing 100 actions per query.

## Language-Conditioned Dataset Builder

Script:
- `project/scripts/prepare_calvin_language_dataset.py`

Launcher:
- `project/scripts/run_prepare_calvin_language_datasets.sh`
- `project/scripts/nohup_prepare_calvin_language_datasets.sh`

Purpose:
- Rebuild B and ABC datasets from raw CALVIN language annotations for official CALVIN success-rate evaluation.
- One language annotation segment becomes one LeRobot episode.
- Adds `observation.language_embedding` with CALVIN's 384D embedding.
- Saves the raw language instruction as the LeRobot `task`.
- Keeps train episodes before val episodes for simple downstream ranges.

Dry-run summaries:
- `project/tables/calvin_lang_B_dry_run_summary.json`
- `project/tables/calvin_lang_ABC_dry_run_summary.json`

Dry-run counts:
- B: `6115` language-segment episodes, `367096` frames, train `5503`, val `612`.
- ABC: `17870` language-segment episodes, `1071743` frames, train `16083`, val `1787`.
- ABC environment counts: A `6089`, B `6115`, C `5666`.
- All language segments are shorter than `chunk_size=100`; padding is expected and checked.

Smoke verification:
- A `/tmp` smoke dataset with 4 B language segments passed LeRobot readback.
- Verified action chunks shape `(100, 7)`, language embedding shape `(384,)`, and correct `action_is_pad` at episode boundaries.

Manual generation commands:

```bash
bash project/scripts/run_prepare_calvin_language_datasets.sh
```

Launcher smoke check:
- `DRY_RUN_ONLY=1 MAX_SEGMENTS=4 READBACK_EPISODES=4 bash project/scripts/run_prepare_calvin_language_datasets.sh` passed.
- `project/scripts/nohup_prepare_calvin_language_datasets.sh` is the preferred manual full-generation launcher.
- Nohup launcher defaults to `CLEAN=1` and `FAST_STORAGE=1`.
- `CLEAN=1` removes previous `calvin_lang_B`, `calvin_lang_ABC`, and final generated language tables before launching.
- `FAST_STORAGE=1` uses `--no-videos` image storage for speed; `FAST_STORAGE=0` uses video storage and is expected to be slower for many short language episodes.
- Tail logs show stable `PROGRESS calvin_lang_* [####----]` lines.

## ACT-Lang-B Training Launcher

Files:
- `project/configs/act_lang_B_full.yaml`
- `project/scripts/nohup_train_act_lang_B.sh`
- `project/configs/act_lang_ABC_full.yaml`
- `project/configs/act_lang_ABC_size_matched_full.yaml`
- `project/scripts/nohup_train_act_lang_ABC_task.sh`
- `project/scripts/nohup_train_act_lang_ABC.sh`
- `project/scripts/nohup_train_act_lang_ABC_size_matched.sh`
- `project/scripts/nohup_train_act_lang_ABC_pair.sh`
- `project/tables/act_lang_ABC_size_matched_split_summary.json`

Status:
- Prepared and code-unblocked for B.
- `project/scripts/train_act_lang.py` now exists and passed a 2-step ACT-Lang-B minibatch self-check.
- Launchers still refuse incomplete/missing language datasets and therefore prevent accidentally training plain ACT without language conditioning.
- ABC language stage includes only full and size-matched variants; no augmentation variants are prepared.

ACT-Lang implementation:
- File: `project/scripts/train_act_lang.py`
- Adds one language conditioning token to ACT's transformer encoder.
- Consumes `observation.language_embedding` with shape `(384,)`.
- Projects language via `Linear(384, 512)`.
- Keeps image/action/state preprocessing and ACT hyperparameters aligned with prior ACT-B/ACT-ABC configs.
- Marks language as `FeatureType.LANGUAGE` with identity normalization.
- Import self-check:
  - Works both as a CLI script and as `project.scripts.train_act_lang` from the repository root.
  - `py_compile` passes.

Official ACT alignment check:
- `act_lang_B_full`, `act_lang_ABC_full`, and `act_lang_ABC_size_matched_full` share identical policy/training hyperparameters.
- Core ACT fields match local LeRobot `ACTConfig()` defaults:
  - ResNet18 ImageNet backbone
  - `n_obs_steps=1`
  - `chunk_size=100`
  - `n_action_steps=100`
  - `dim_model=512`
  - `n_heads=8`
  - `dim_feedforward=3200`
  - `n_encoder_layers=4`
  - `n_decoder_layers=1`
  - VAE enabled, `latent_dim=32`, `n_vae_encoder_layers=4`
  - `dropout=0.1`
  - `kl_weight=10.0`
  - optimizer lr/backbone lr `1e-5`, weight decay `1e-4`
- `batch_size=32` is retained for strict experimental alignment with previous ACT-B/ACT-ABC runs; official docs treat batch size as adjustable by GPU memory.
- `num_workers=12` is used for ACT-Lang-B full training after diagnosing CPU/DataLoader bottleneck at `num_workers=4`.
- After ABC conversion finishes, `num_workers=16` is staged for both ACT-Lang-ABC full and ACT-Lang-ABC size-matched.
- Worker-count changes only affect input pipeline parallelism; model architecture, loss, optimizer, batch size, and dataset splits are unchanged.
- Plain official `policy.type=act` does not consume `observation.language_embedding`, so `policy.type=act_lang` is the intentional extension required for CALVIN language-conditioned success-rate evaluation.

ACT-Lang-B mini self-check:
- Config: `project/configs/act_lang_B.yaml`
- Output: `project/outputs/act_lang_B/mini_trial_selfcheck/`
- Device: `cuda:0`
- Train episodes: `0:8`, `468` frames
- Val episodes: `5503:5507`, `241` frames
- Batch size: `2`
- Steps: `2`
- Raw language shape: `[2, 384]`
- Predicted action chunk shape: `[2, 100, 7]`
- Language sensitivity L1 when zeroing language: `0.006950919`
- Final self-check: `passed: true`
- Metrics:
  - step 1 train Action L1 `0.878052`, val Action L1 `1.138959`
  - step 2 train Action L1 `0.886268`, val Action L1 `1.057600`

Alignment:
- Matches `act_B_full.yaml` on ACT backbone, transformer size, VAE/KL, optimizer, batch size, steps, validation frequency, checkpoint frequency, and progress logging.
- Intended differences are only:
  - dataset `local/calvin_lang_B`
  - train/val ranges `0:5503` and `5503:6115`
  - policy type `act_lang`
  - added `observation.language_embedding` dimension `384`.
- `act_lang_ABC_full.yaml` and `act_lang_ABC_size_matched_full.yaml` match `act_lang_B_full.yaml` on all shared policy/training hyperparameters.
- ACT-Lang default GPU allocation:
  - B: GPU0
  - ABC full: GPU1
  - ABC size-matched: GPU2
- Preferred ABC entry point is `project/scripts/nohup_train_act_lang_ABC_pair.sh`.
- It launches ABC full first, waits `STAGGER_SECONDS=300`, then launches ABC size-matched to avoid simultaneous dataset metadata/cache IO.
- ABC launcher code-path check:
  - `bash -n` passes for all ACT-Lang B/ABC launcher scripts.
  - The ABC task launcher points to `project/scripts/train_act_lang.py`, not plain `train_act.py`.
  - It sets `CUDA_VISIBLE_DEVICES=$GPU_ID` and `ACT_DEVICE_OVERRIDE=cuda:0`, so the masked process sees the selected physical GPU as `cuda:0`.
  - It checks dataset schema/counts before writing and launching the nohup run script.
- Current ABC language dataset is incomplete and should not be trained yet:
  - observed during latest check: roughly `948` episodes and `56888` frames
  - required by launcher: at least `17870` episodes and `1071743` frames
  - current launcher would fail fast before starting training.
- ABC size-matched split:
  - target B train frames: `330455`
  - selected train frames: `330546`
  - delta: `+91`
  - selected episodes: A `1830`, B `1836`, C `1851`
  - validation reuses full ABC val range `16083:17870`.

Manual command after dataset and ACT-Lang training code are ready:

```bash
GPU_ID=0 bash project/scripts/nohup_train_act_lang_B.sh
bash project/scripts/nohup_train_act_lang_ABC_pair.sh
```

## Last Updated

2026-06-06 UTC
