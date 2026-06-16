#!/usr/bin/env bash
set -euo pipefail

# Manual launcher for ACT-Lang-B full-scale training.
# This is intentionally aligned with nohup_train_act_B.sh, but it refuses to
# run unless an ACT-Lang training entry point exists. Standard train_act.py must
# not be used because it builds plain ACT and would ignore language embeddings.

source /home/zengzixuan/cvprojects/calvin_env.sh

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

CONFIG="$LEROBOT_SOURCE/project/configs/act_lang_B_full.yaml"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$LEROBOT_SOURCE/project/scripts/train_act_lang.py}"
DATASET_ROOT="$CALVIN_LEROBOT_ROOT/calvin_lang_B"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_act_lang_B_full_gpu${GPU_ID}"
RUN_DIR="$CALVIN_RUNS/act_lang_B/$RUN_ID"
LOG_DIR="$CALVIN_RUNS/act_lang_B/logs"
LOG_FILE="$LOG_DIR/${RUN_ID}.log"

mkdir -p "$RUN_DIR" "$LOG_DIR"

echo "Starting ACT-Lang-B full training manually."
echo "GPU_ID=$GPU_ID"
echo "CONFIG=$CONFIG"
echo "TRAIN_SCRIPT=$TRAIN_SCRIPT"
echo "DATASET_ROOT=$DATASET_ROOT"
echo "RUN_DIR=$RUN_DIR"
echo "LOG_FILE=$LOG_FILE"
echo "TORCH_HOME=$TORCH_HOME"
echo "HF_LEROBOT_HOME=$HF_LEROBOT_HOME"
echo

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: Missing config: $CONFIG" >&2
  exit 2
fi

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  cat >&2 <<EOF
ERROR: Missing ACT-Lang training entry point:
  $TRAIN_SCRIPT

Do not replace this with project/scripts/train_act.py.
Plain train_act.py builds standard ACT and ignores observation.language_embedding,
so it would not produce a CALVIN success-rate-compatible model.
EOF
  exit 3
fi

if [[ ! -d "$DATASET_ROOT" ]]; then
  cat >&2 <<EOF
ERROR: Missing ACT-Lang-B dataset:
  $DATASET_ROOT

Generate it first:
  bash project/scripts/nohup_prepare_calvin_language_datasets.sh
EOF
  exit 4
fi

echo "Checking ACT-Lang-B dataset schema..."
"$LEROBOT_PYTHON" - <<'PY'
import json
from pathlib import Path

root = Path("/EXT_DISK/users/zengzixuan/processed-calvin/calvin_lang_B")
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
episodes = int(info["total_episodes"])
frames = int(info["total_frames"])
if episodes < 6115:
    raise SystemExit(f"Expected full B dataset to have at least 6115 episodes, got {episodes}")
if frames < 367096:
    raise SystemExit(f"Expected full B dataset to have at least 367096 frames, got {frames}")
print(json.dumps({
    "dataset_schema_check": "passed",
    "root": str(root),
    "episodes": episodes,
    "frames": frames,
    "total_tasks": info.get("total_tasks"),
    "language_embedding_shape": features["observation.language_embedding"]["shape"],
}, indent=2), flush=True)
PY
echo

echo "GPU state before launch:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
echo

nohup "$LEROBOT_PYTHON" "$TRAIN_SCRIPT" \
  --config "$CONFIG" \
  --output-dir "$RUN_DIR" \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$RUN_DIR/nohup.pid"

echo "Launched PID=$PID"
echo "PID file: $RUN_DIR/nohup.pid"
echo
echo "Monitor progress with:"
echo "  tail -f $LOG_FILE"
echo
echo "The log should print tail-visible progress lines like:"
echo "  PROGRESS act_lang_B_full [########------------------------] 25000/100000 ( 25.0%) elapsed=... eta=..."
echo
echo "Final checkpoint will be:"
echo "  $RUN_DIR/checkpoint"
echo "Periodic checkpoints will be under:"
echo "  $RUN_DIR/checkpoints/"
