#!/usr/bin/env bash
set -euo pipefail

# Manual launcher for Task 2 full-scale training.
# This script starts nohup only when explicitly run by the user.
# It refuses GPU0 because GPU0 is reserved for the ACT-B augmentation run.

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {act_ABC_full|act_ABC_aug_full|act_ABC_size_matched_full|act_ABC_size_matched_aug_full}" >&2
  exit 2
fi

EXPERIMENT="$1"
source /home/zengzixuan/cvprojects/calvin_env.sh

case "$EXPERIMENT" in
  act_ABC_full)
    DEFAULT_GPU_ID=1
    CONFIG="$LEROBOT_SOURCE/project/configs/act_ABC_full.yaml"
    RUN_GROUP="task2/act_ABC"
    ;;
  act_ABC_size_matched_full)
    DEFAULT_GPU_ID=2
    CONFIG="$LEROBOT_SOURCE/project/configs/act_ABC_size_matched_full.yaml"
    RUN_GROUP="task2/act_ABC_size_matched"
    ;;
  act_ABC_aug_full)
    DEFAULT_GPU_ID=3
    CONFIG="$LEROBOT_SOURCE/project/configs/act_ABC_aug_full.yaml"
    RUN_GROUP="task2/act_ABC_aug"
    ;;
  act_ABC_size_matched_aug_full)
    DEFAULT_GPU_ID=4
    CONFIG="$LEROBOT_SOURCE/project/configs/act_ABC_size_matched_aug_full.yaml"
    RUN_GROUP="task2/act_ABC_size_matched_aug"
    ;;
  *)
    echo "Unknown Task 2 experiment: $EXPERIMENT" >&2
    exit 2
    ;;
esac

GPU_ID="${GPU_ID:-$DEFAULT_GPU_ID}"
if [[ "$GPU_ID" == "0" ]]; then
  echo "Refusing to launch $EXPERIMENT on GPU0. GPU0 is reserved for ACT-B augmentation." >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export ACT_DEVICE_OVERRIDE="cuda:0"
export PYTHONUNBUFFERED=1

RUN_ID="$(date -u +%Y%m%d_%H%M%S)_${EXPERIMENT}_gpu${GPU_ID}"
RUN_DIR="$CALVIN_RUNS/$RUN_GROUP/$RUN_ID"
LOG_DIR="$CALVIN_RUNS/$RUN_GROUP/logs"
LOG_FILE="$LOG_DIR/${RUN_ID}.log"

mkdir -p "$RUN_DIR" "$LOG_DIR"

echo "Starting Task 2 full training manually."
echo "EXPERIMENT=$EXPERIMENT"
echo "GPU_ID=$GPU_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "ACT_DEVICE_OVERRIDE=$ACT_DEVICE_OVERRIDE"
echo "CONFIG=$CONFIG"
echo "RUN_DIR=$RUN_DIR"
echo "LOG_FILE=$LOG_FILE"
echo "TORCH_HOME=$TORCH_HOME"
echo "HF_LEROBOT_HOME=$HF_LEROBOT_HOME"
echo
echo "GPU state before launch:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
echo

nohup "$LEROBOT_PYTHON" "$LEROBOT_SOURCE/project/scripts/train_act.py" \
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
echo "The log prints tail-visible progress lines like:"
echo "  PROGRESS $EXPERIMENT [########------------------------] 25000/100000 ( 25.0%) elapsed=... eta=..."
echo
echo "Final checkpoint will be:"
echo "  $RUN_DIR/checkpoint"
echo "Periodic checkpoints will be under:"
echo "  $RUN_DIR/checkpoints/"
