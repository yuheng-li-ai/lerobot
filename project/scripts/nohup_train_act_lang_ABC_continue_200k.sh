#!/usr/bin/env bash
set -euo pipefail

# Continue ACT-Lang-ABC full from the completed 100k checkpoint to 200k total
# optimizer steps. This must be launched manually by the user.

source /home/zengzixuan/cvprojects/calvin_env.sh

EXPERIMENT="act_lang_ABC_full_continue_200k"
GPU_ID="${GPU_ID:-1}"
CONFIG="${CONFIG:-$LEROBOT_SOURCE/project/configs/act_lang_ABC_continue_200k.yaml}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$LEROBOT_SOURCE/project/scripts/train_act_lang.py}"
RESUME_FROM="${RESUME_FROM:-/EXT_DISK/users/zengzixuan/calvin_runs/act_lang_ABC/20260606_163515_act_lang_ABC_full_gpu1/checkpoint}"
DATASET_ROOT="$CALVIN_LEROBOT_ROOT/calvin_lang_ABC"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export ACT_DEVICE_OVERRIDE="cuda:0"
export PYTHONUNBUFFERED=1

RUN_ID="$(date -u +%Y%m%d_%H%M%S)_${EXPERIMENT}_gpu${GPU_ID}"
RUN_GROUP="act_lang_ABC_continue_200k"
RUN_DIR="$CALVIN_RUNS/$RUN_GROUP/$RUN_ID"
LOG_DIR="$CALVIN_RUNS/$RUN_GROUP/logs"
LOG_FILE="$LOG_DIR/${RUN_ID}.log"
RUN_SCRIPT="$RUN_DIR/run.sh"

mkdir -p "$RUN_DIR" "$LOG_DIR"

echo "Starting ACT-Lang-ABC continuation manually."
echo "EXPERIMENT=$EXPERIMENT"
echo "GPU_ID=$GPU_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "ACT_DEVICE_OVERRIDE=$ACT_DEVICE_OVERRIDE"
echo "CONFIG=$CONFIG"
echo "TRAIN_SCRIPT=$TRAIN_SCRIPT"
echo "RESUME_FROM=$RESUME_FROM"
echo "DATASET_ROOT=$DATASET_ROOT"
echo "RUN_DIR=$RUN_DIR"
echo "LOG_FILE=$LOG_FILE"
echo "TORCH_HOME=$TORCH_HOME"
echo "HF_LEROBOT_HOME=$HF_LEROBOT_HOME"
echo

if [[ "$GPU_ID" == "0" ]]; then
  echo "Refusing to launch $EXPERIMENT on GPU0 by default; GPU0 is reserved for ACT-Lang-B unless explicitly needed." >&2
  echo "Use a nonzero GPU_ID, e.g. GPU_ID=1." >&2
  exit 3
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: Missing config: $CONFIG" >&2
  exit 4
fi

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "ERROR: Missing ACT-Lang training entry point: $TRAIN_SCRIPT" >&2
  exit 5
fi

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "ERROR: Missing ACT-Lang-ABC dataset: $DATASET_ROOT" >&2
  exit 6
fi

if [[ ! -f "$RESUME_FROM/model.safetensors" || ! -f "$RESUME_FROM/training_state.pt" ]]; then
  cat >&2 <<EOF
ERROR: Resume checkpoint is incomplete:
  $RESUME_FROM

Expected model.safetensors and training_state.pt.
EOF
  exit 7
fi

echo "Checking ACT-Lang-ABC dataset, config, and resume checkpoint..."
"$LEROBOT_PYTHON" - "$CONFIG" "$DATASET_ROOT" "$RESUME_FROM" <<'PY'
import json
import sys
from pathlib import Path

import torch
import yaml

config_path = Path(sys.argv[1])
root = Path(sys.argv[2])
resume_from = Path(sys.argv[3])
cfg = yaml.safe_load(config_path.read_text())

info_path = root / "meta" / "info.json"
if not info_path.is_file():
    raise SystemExit(f"Missing dataset info: {info_path}")
info = json.loads(info_path.read_text())
features = info["features"]
required = {
    "observation.images.static": (200, 200, 3),
    "observation.images.gripper": (84, 84, 3),
    "observation.state": (15,),
    "observation.language_embedding": (384,),
    "action": (7,),
}
for key, expected_shape in required.items():
    if key not in features:
        raise SystemExit(f"Missing required feature: {key}")
    actual_shape = tuple(features[key]["shape"])
    if actual_shape != expected_shape:
        raise SystemExit(f"{key} shape {actual_shape}, expected {expected_shape}")

state = torch.load(resume_from / "training_state.pt", map_location="cpu")
resume_step = int(state["step"])
target_steps = int(cfg["training"]["steps"])
if resume_step != 100000:
    raise SystemExit(f"Expected resume step 100000, got {resume_step}")
if target_steps != 200000:
    raise SystemExit(f"Expected target training.steps 200000, got {target_steps}")

train_eps = cfg["dataset"]["train_episodes"]
val_eps = cfg["dataset"]["val_episodes"]
print(json.dumps({
    "continuation_check": "passed",
    "root": str(root),
    "episodes": int(info["total_episodes"]),
    "frames": int(info["total_frames"]),
    "total_tasks": info.get("total_tasks"),
    "language_embedding_shape": features["observation.language_embedding"]["shape"],
    "train_episodes": train_eps,
    "val_episodes": val_eps,
    "resume_from": str(resume_from),
    "resume_step": resume_step,
    "target_steps": target_steps,
    "additional_steps": target_steps - resume_step,
}, indent=2), flush=True)
PY
echo

echo "GPU state before launch:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
echo

cat > "$RUN_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
trap 'echo "FAILED at line \${LINENO} with exit code \$?"' ERR

source /home/zengzixuan/cvprojects/calvin_env.sh
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export ACT_DEVICE_OVERRIDE="cuda:0"
export PYTHONUNBUFFERED=1

echo "RUN_ID=$RUN_ID"
echo "EXPERIMENT=$EXPERIMENT"
echo "Started wrapper at \$(date -Is)"
echo "GPU_ID=$GPU_ID"
echo "CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES"
echo "ACT_DEVICE_OVERRIDE=\$ACT_DEVICE_OVERRIDE"
echo "CONFIG=$CONFIG"
echo "TRAIN_SCRIPT=$TRAIN_SCRIPT"
echo "RESUME_FROM=$RESUME_FROM"
echo "RUN_DIR=$RUN_DIR"
echo

"$LEROBOT_PYTHON" "$TRAIN_SCRIPT" \\
  --config "$CONFIG" \\
  --resume-from "$RESUME_FROM" \\
  --output-dir "$RUN_DIR"
EOF

chmod +x "$RUN_SCRIPT"
nohup /bin/bash "$RUN_SCRIPT" > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$RUN_DIR/nohup.pid"

echo "Launched PID=$PID"
echo "PID file: $RUN_DIR/nohup.pid"
echo "Run script: $RUN_SCRIPT"
echo
echo "Monitor progress with:"
echo "  tail -f $LOG_FILE"
echo
echo "The log should print tail-visible continuation progress lines like:"
echo "  PROGRESS $EXPERIMENT [########------------------------] 25000/100000 ( 25.0%) elapsed=... eta=... total_step=125000/200000 resumed_from=100000"
echo
echo "Final checkpoint will be:"
echo "  $RUN_DIR/checkpoint"
echo "Periodic checkpoints will be under:"
echo "  $RUN_DIR/checkpoints/"
