#!/usr/bin/env bash
set -euo pipefail

# Manual launcher for ACT-B full-scale training.
# This script starts nohup only when explicitly run by the user.

source /home/zengzixuan/cvprojects/calvin_env.sh

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

CONFIG="$LEROBOT_SOURCE/project/configs/act_B_full.yaml"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_act_B_full_gpu${GPU_ID}"
RUN_DIR="$CALVIN_RUNS/act_B/$RUN_ID"
LOG_DIR="$CALVIN_RUNS/act_B/logs"
LOG_FILE="$LOG_DIR/${RUN_ID}.log"

mkdir -p "$RUN_DIR" "$LOG_DIR"

echo "Starting ACT-B full training manually."
echo "GPU_ID=$GPU_ID"
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
echo "  PROGRESS act_B_full [########------------------------] 25000/100000 ( 25.0%) elapsed=... eta=..."
echo
echo "Final checkpoint will be:"
echo "  $RUN_DIR/checkpoint"
echo "Periodic checkpoints will be under:"
echo "  $RUN_DIR/checkpoints/"
