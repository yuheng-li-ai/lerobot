# Task 3 Success-Rate Model and Dataset Requirements

## Core Constraint

CALVIN official success rate requires a goal-conditioned policy. The current ACT checkpoints are not goal-conditioned: they map `rgb_static + rgb_gripper + robot_obs` to `rel_actions`. They are valid for offline zero-shot action error on D, but not for official CALVIN language-conditioned success rate.

To report official success rate, the new models must accept a language goal or equivalent task embedding at inference time.

## Important Split Rule

If the final zero-shot environment is still **D**, the multi-environment training model must not train on D.

Recommended pair for D zero-shot success rate:
- `ACT-Lang-B`: train on environment B only.
- `ACT-Lang-ABC`: train on environments A+B+C.
- Evaluate both on environment D with CALVIN official long-horizon sequences.

If you train `ACT-Lang-ACD`, then D is no longer unseen. In that case the held-out zero-shot environment should be B, not D.

## Model Interface Requirement

The model must implement the CALVIN evaluation interface:

```python
class CalvinCompatiblePolicy:
    def reset(self) -> None:
        """Clear recurrent state/action queues at the start of every subtask."""

    def step(self, obs: dict, goal: str | np.ndarray) -> np.ndarray:
        """Return one 7D CALVIN relative action."""
```

At evaluation time:
- `obs` comes from `calvin_env`.
- `goal` is the current language instruction from CALVIN validation annotations.
- output action must be CALVIN `rel_actions` scale: 7D `(dx, dy, dz, droll, dpitch, dyaw, gripper)`.
- `reset()` must clear ACT's chunk queue before every new subtask.

## Required Model Inputs

Minimum inputs:
- `rgb_static`: static camera RGB, shape `(200, 200, 3)`.
- `rgb_gripper`: gripper camera RGB, shape `(84, 84, 3)`.
- `robot_obs`: robot proprioception, shape `(15,)`.
- language goal:
  - raw text instruction, or
  - precomputed language embedding aligned with the instruction.

Recommended model design:
- Keep ACT visual/action architecture as close as possible to the existing ACT runs.
- Add only a language-conditioning branch:
  - text encoder or frozen sentence embedding projection;
  - fuse language into ACT transformer tokens or conditioning vector.
- Keep `chunk_size=100` and `n_action_steps=100` unless explicitly running a chunk ablation.

## Dataset Requirement

The success-rate training dataset must be language-aligned, not just play-data action cloning.

Use raw CALVIN:
- `/SSD_DISK/users/zengzixuan/calvin/task_ABC_D/training`
- language metadata from `training/lang_annotations/auto_lang_ann.npy`
- environment split from `training/scene_info.npy`

Each training sample should contain:
- observation at timestep `t`;
- language instruction for the annotated segment containing `t`;
- target action chunk `rel_actions[t : t + chunk_size]`;
- `action_is_pad` mask when the chunk crosses segment or episode boundary.

Recommended fields:
- `observation.images.static`
- `observation.images.gripper`
- `observation.state`
- `observation.language` or `observation.language_embedding`
- `action`
- `action_is_pad`
- metadata: environment id, raw episode id, segment id, task name, start/end frame.

Boundary rule:
- Do not let action chunks cross language segment boundaries unless the label remains valid for the full chunk.
- Safer default: truncate/pad chunks at the language segment end.

## Required Training Variants

For D zero-shot success rate:

### `ACT-Lang-B`

Dataset:
- environment B frames only.
- language segments whose frame ranges fall inside `calvin_scene_B`.
- hold out validation segments from B for checkpoint selection.

Purpose:
- single-environment goal-conditioned baseline.

### `ACT-Lang-ABC`

Dataset:
- environments A+B+C.
- balanced or at least environment-labeled sampling is strongly recommended.
- hold out validation segments from A/B/C for checkpoint selection.

Purpose:
- multi-environment goal-conditioned model for D zero-shot transfer.

Optional fair-data extension:
- `ACT-Lang-ABC-size-matched`: same number of frames or language segments as `ACT-Lang-B`, sampled across A/B/C.
- This distinguishes data volume from environment diversity.

## Evaluation Requirement

Use CALVIN official evaluator:
- instantiate environment from raw validation folder;
- run official multistep sequences;
- feed the current language instruction to `model.step(obs, goal)`;
- compute:
  - success rate for 1/5 through 5/5 subtasks;
  - average successful sequence length;
  - optional per-task success table.

Primary report metrics:
- `SR@1`, `SR@2`, `SR@3`, `SR@4`, `SR@5`.
- average successful chain length.
- action smoothness and chunk boundary jump during rollouts.
- failure taxonomy from failed rollout videos.

## ACT Chunking Analysis Requirement

During rollout, log:
- every predicted action;
- chunk id;
- position within chunk;
- whether a fresh chunk was queried;
- language subtask;
- success/failure.

Compute:
- within-chunk step delta L2 over first 6 action dims;
- boundary jump L2 between the last action of one chunk and first action of the next;
- correction delay after a failed/incorrect chunk, if observable;
- gripper switch timing relative to task success/failure.

Interpretation target:
- Does multi-environment language-conditioned training reduce visual-shift failures on D?
- Does ACT chunking stabilize motion, or does it over-smooth wrong actions under D visual shift?
- Are failures dominated by wrong task grounding, object localization, grasping, or chunk-boundary instability?

## Minimal Deliverables For The Other Agent

- Language-aligned dataset builder for B and ABC.
- Configs for `ACT-Lang-B` and `ACT-Lang-ABC`.
- Mini-trial for each model.
- Full-training nohup scripts only; do not launch full training automatically.
- CALVIN-compatible wrapper implementing `reset()` and `step(obs, goal)`.
- Official CALVIN evaluation script producing success-rate tables.
- Rollout logging for action chunk diagnostics.
