#!/usr/bin/env bash
set -euo pipefail

# Manual launcher for ACT-Lang ABC full-scale training variants.
# Supported experiments:
#   - act_lang_ABC_full
#   - act_lang_ABC_size_matched_full
#
# This launcher intentionally refuses to use project/scripts/train_act.py because
# plain ACT ignores observation.language_embedding.

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {act_lang_ABC_full|act_lang_ABC_size_matched_full}" >&2
  exit 2
fi

EXPERIMENT="$1"
source /home/zengzixuan/cvprojects/calvin_env.sh

case "$EXPERIMENT" in
  act_lang_ABC_full)
    DEFAULT_GPU_ID=1
    CONFIG="$LEROBOT_SOURCE/project/configs/act_lang_ABC_full.yaml"
    RUN_GROUP="act_lang_ABC"
    ;;
  act_lang_ABC_size_matched_full)
    DEFAULT_GPU_ID=2
    CONFIG="$LEROBOT_SOURCE/project/configs/act_lang_ABC_size_matched_full.yaml"
    RUN_GROUP="act_lang_ABC_size_matched"
    ;;
  *)
    echo "Unknown ACT-Lang ABC experiment: $EXPERIMENT" >&2
    exit 2
    ;;
esac

GPU_ID="${GPU_ID:-$DEFAULT_GPU_ID}"
START_DELAY_SECONDS="${START_DELAY_SECONDS:-300}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$LEROBOT_SOURCE/project/scripts/train_act_lang.py}"
DATASET_ROOT="$CALVIN_LEROBOT_ROOT/calvin_lang_ABC"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export ACT_DEVICE_OVERRIDE="cuda:0"
export PYTHONUNBUFFERED=1

RUN_ID="$(date -u +%Y%m%d_%H%M%S)_${EXPERIMENT}_gpu${GPU_ID}"
RUN_DIR="$CALVIN_RUNS/$RUN_GROUP/$RUN_ID"
LOG_DIR="$CALVIN_RUNS/$RUN_GROUP/logs"
LOG_FILE="$LOG_DIR/${RUN_ID}.log"
RUN_SCRIPT="$RUN_DIR/run.sh"

mkdir -p "$RUN_DIR" "$LOG_DIR"

echo "Starting ACT-Lang ABC full training manually."
echo "EXPERIMENT=$EXPERIMENT"
echo "GPU_ID=$GPU_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "ACT_DEVICE_OVERRIDE=$ACT_DEVICE_OVERRIDE"
echo "START_DELAY_SECONDS=$START_DELAY_SECONDS"
echo "CONFIG=$CONFIG"
echo "TRAIN_SCRIPT=$TRAIN_SCRIPT"
echo "DATASET_ROOT=$DATASET_ROOT"
echo "RUN_DIR=$RUN_DIR"
echo "LOG_FILE=$LOG_FILE"
echo "TORCH_HOME=$TORCH_HOME"
echo "HF_LEROBOT_HOME=$HF_LEROBOT_HOME"
echo

if [[ "$GPU_ID" == "0" ]]; then
  echo "Refusing to launch $EXPERIMENT on GPU0 by default; GPU0 is reserved for ACT-Lang-B unless explicitly needed." >&2
  echo "Use a nonzero GPU_ID, e.g. GPU_ID=1 or GPU_ID=2." >&2
  exit 3
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: Missing config: $CONFIG" >&2
  exit 4
fi

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  cat >&2 <<EOF
ERROR: Missing ACT-Lang training entry point:
  $TRAIN_SCRIPT

Do not replace this with project/scripts/train_act.py.
Plain train_act.py builds standard ACT and ignores observation.language_embedding.
EOF
  exit 5
fi

if [[ ! -d "$DATASET_ROOT" ]]; then
  cat >&2 <<EOF
ERROR: Missing ACT-Lang-ABC dataset:
  $DATASET_ROOT

Generate it first:
  bash project/scripts/nohup_prepare_calvin_language_datasets.sh
EOF
  exit 6
fi

echo "Checking ACT-Lang-ABC dataset schema and config split..."
"$LEROBOT_PYTHON" - "$CONFIG" "$DATASET_ROOT" <<'PY'
import json
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
root = Path(sys.argv[2])
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
episodes = int(info["total_episodes"])
frames = int(info["total_frames"])
if episodes < 17870:
    raise SystemExit(f"Expected full ABC dataset to have at least 17870 episodes, got {episodes}")
if frames < 1071743:
    raise SystemExit(f"Expected full ABC dataset to have at least 1071743 frames, got {frames}")
train_eps = cfg["dataset"]["train_episodes"]
val_eps = cfg["dataset"]["val_episodes"]
train_count = len(train_eps) if isinstance(train_eps, list) else train_eps
print(json.dumps({
    "dataset_schema_check": "passed",
    "root": str(root),
    "episodes": episodes,
    "frames": frames,
    "total_tasks": info.get("total_tasks"),
    "language_embedding_shape": features["observation.language_embedding"]["shape"],
    "train_episodes": train_count,
    "val_episodes": val_eps,
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
echo "START_DELAY_SECONDS=$START_DELAY_SECONDS"
echo "CONFIG=$CONFIG"
echo "TRAIN_SCRIPT=$TRAIN_SCRIPT"
echo "RUN_DIR=$RUN_DIR"
echo

if [[ "$START_DELAY_SECONDS" != "0" ]]; then
  remaining="$START_DELAY_SECONDS"
  while [[ "\$remaining" -gt 0 ]]; do
    echo "START_DELAY $EXPERIMENT remaining=\${remaining}s"
    sleep 30
    remaining=\$(( remaining - 30 ))
  done
fi

echo "Training starts at \$(date -Is)"
"$LEROBOT_PYTHON" "$TRAIN_SCRIPT" \\
  --config "$CONFIG" \\
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
echo "The log will first show the 300s countdown:"
echo "  START_DELAY $EXPERIMENT remaining=300s"
echo
echo "The log should print tail-visible progress lines like:"
echo "  PROGRESS $EXPERIMENT [########------------------------] 25000/100000 ( 25.0%) elapsed=... eta=..."
echo
echo "Final checkpoint will be:"
echo "  $RUN_DIR/checkpoint"
echo "Periodic checkpoints will be under:"
echo "  $RUN_DIR/checkpoints/"
