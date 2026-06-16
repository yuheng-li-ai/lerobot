# Task Board

## Phase 0 - Environment and Dataset Preparation

- [x] Create contained research workspace under `project/`.
- [x] Verify LeRobot source tree presence.
- [x] Verify ACT implementation presence.
- [x] Locate CALVIN framework repository.
- [x] Check configured raw CALVIN dataset path.
- [x] Verify LeRobot Python import environment with existing conda Python.
- [x] Confirm configured raw CALVIN ABC->D path.
- [x] Collect raw split episode counts, frame counts, and task counts.
- [x] Generate `tables/dataset_stats_ABC.csv` with currently verified raw split statistics.
- [x] Confirm per-environment A/B/C/D partition mapping inside raw data.
- [x] Write LeRobot conversion script for B/ABC/D.
- [x] Smoke-test small LeRobot conversion without full conversion.
- [x] Generate `figures/env_samples_ABCD.png`.
- [x] User runs full LeRobot conversion via `project/scripts/run_phase0_conversion.sh`.
- [x] Verify full converted B/ABC/D datasets load after manual conversion.

## Phase 1 - ACT-B

- [x] Prepare ACT-B config.
- [x] Select GPU for mini-trial and record choice.
- [x] Run ACT-B mini-trial only.
- [x] Save mini-trial logs, metrics, and checkpoint.
- [x] Generate full-scale ACT-B `nohup` script for manual launch.
- [x] User manually ran ACT-B full training.
- [x] Verify ACT-B full training outputs.
- [x] Generate ACT-B loss curve and overfitting summary.
- [x] Smooth ACT-B train Action L1 visualization while preserving raw mini-batch values.
- [x] Record ACT-B checkpoint selection rationale.
- [x] Analyze environment B visual statistics.
- [x] Generate environment B image samples.
- [x] Analyze environment B action distribution.
- [x] Plot ACT-B train-validation generalization gap.
- [x] Analyze environment B action smoothness.
- [x] Analyze environment B chunk baseline.
- [x] Plot per-action-dimension violin distribution.
- [x] Add separate environment B gripper diagnostics while preserving existing gripper findings.
- [x] Plot environment B action delta heatmap.
- [x] Plot environment B visual color profile.
- [x] Plot environment B brightness/contrast histograms.
- [x] Plot environment B task frequency.
- [x] Build representative environment B trajectory strip.
- [x] Defer failure taxonomy until D zero-shot evaluation exists.
- [x] Add config-driven ACT-B visual augmentation harness.
- [x] Verify ACT-B augmentation changes images.
- [x] Run ACT-B augmentation 20-step mini-trial only.
- [x] Save ACT-B augmentation mini-trial metrics and checkpoint.
- [x] Generate ACT-B augmentation full-training nohup launcher.
- [x] Confirm ACT-B augmentation launcher has tail-visible progress bars.
- [x] User manually ran ACT-B augmentation full training.
- [x] Verify ACT-B augmentation full-training outputs.
- [x] Generate ACT-B augmentation baseline-matched visualizations and tables.
- [x] Verify ACT-B augmentation visualization coverage is one-to-one with baseline.
- [x] Correct ACT-B augmentation checkpoint selection: step 15k is metric-only, not a saved checkpoint.

## Phase 2 - ACT-ABC

- [x] Prepare ACT-ABC config with architecture/hyperparameters matched to ACT-B.
- [x] Prepare ACT-ABC-size-matched or ACT-ABC-balanced config.
- [x] Prepare ACT-ABC-aug config aligned with ACT-B-aug augmentation settings.
- [x] Prepare ACT-ABC-size-matched-aug config aligned with ACT-B-aug augmentation settings.
- [x] Generate Task 2 episode split table.
- [x] Generate Task 2 full-scale manual launchers that refuse GPU0.
- [x] Add Task 2 mini-trial harness with classified outputs/logs.
- [x] Select GPU for mini-trial and record choice.
- [x] Run ACT-ABC mini-trial only.
- [x] Run ACT-ABC-aug mini-trial only.
- [x] Run ACT-ABC-size-matched mini-trial only.
- [x] Run ACT-ABC-size-matched-aug mini-trial only.
- [x] Save mini-trial logs, metrics, and checkpoint.
- [x] Confirm Task 2 mini-trial logs contain tail-visible progress bars.
- [x] Generate full-scale ACT-ABC `nohup` script for manual launch.
- [x] Generate staggered Task 2 full-training launcher for manual launch.
- [x] User manually ran Task 2 full training experiments.
- [x] Verify Task 2 full training outputs.
- [x] Record Task 2 best validation and selected available checkpoint candidates.
- [x] Summarize Task 2 augmentation and size-matched effects in CSV tables.
- [x] Generate Task 2 extended per-model diagnostic visualizations in separate figure/table folders.
- [x] Compare ACT-B vs ACT-ABC training/validation losses and auxiliary diagnostics.

## Phase 3 - Zero-Shot D

- [x] Evaluate ACT-B on D.
- [x] Evaluate ACT-ABC on D.
- [x] Compare success rate, action L1 error, and average steps.
- [x] Analyze RGB mean/std, brightness, and contrast shift.
- [x] Analyze action smoothness and chunk boundary jumps.
- [ ] Build failure taxonomy table.

## Language-Conditioned Success-Rate Models

- [x] Generate complete `local/calvin_lang_B` dataset with 384D language embeddings.
- [x] Finish complete `local/calvin_lang_ABC` dataset with 384D language embeddings.
- [x] Prepare ACT-Lang-B full training config and launcher.
- [x] Prepare ACT-Lang-ABC full and size-matched configs/launchers.
- [x] Implement `project/scripts/train_act_lang.py`.
- [x] Run ACT-Lang-B 2-step minibatch self-check.
- [x] Verify language embedding is present, finite, shape `[batch, 384]`, and on GPU.
- [x] Verify ACT-Lang predicted action chunk shape `[batch, 100, 7]`.
- [x] Verify zeroing language changes predicted actions.
- [x] Verify mini metrics, checkpoint, policy config, weights, preprocessor, postprocessor, and optimizer state are saved.
- [x] User manually launches ACT-Lang-B full training.
- [x] User manually launches ACT-Lang-ABC full and size-matched training after ABC language dataset is complete.
- [x] Evaluate ACT-Lang-B 100k, ACT-Lang-ABC-size-matched 100k, and ACT-Lang-ABC 200k on CALVIN D success rate.

## Safety

- [x] Confirm no full-scale training was launched by agents.
- [x] Confirm Task 1 extension work used offline analysis only, with no new training launched.
- [x] Confirm Task 2 avoided GPU0 during mini-trials.
- [x] Confirm ACT-Lang agent run was limited to a 2-step mini self-check and did not launch full-scale training.
