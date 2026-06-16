#!/usr/bin/env bash
set -euo pipefail

# Launch Task 3 offline zero-shot D evaluation with tail-visible progress.
# This uses the existing conda environment from calvin_env.sh; it does not use uv.

source /home/zengzixuan/cvprojects/calvin_env.sh

GPU_ID="${GPU_ID:-5}"
CHECKPOINT_MODE="${CHECKPOINT_MODE:-selected}"  # selected | final | both
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_FREQ="${LOG_FREQ:-5}"
MAX_QUERIES="${MAX_QUERIES:-}"  # optional smoke-test cap, e.g. MAX_QUERIES=20
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_zero_shot_D_${CHECKPOINT_MODE}_gpu${GPU_ID}}"

LOG_DIR="$CALVIN_RUNS/task3/logs"
OUT_DIR="$LEROBOT_SOURCE/project/outputs/task3/${RUN_ID}"
FIG_DIR="$LEROBOT_SOURCE/project/figures/task3"
TABLE_DIR="$LEROBOT_SOURCE/project/tables/task3"
LOG_PATH="$LOG_DIR/${RUN_ID}.log"

mkdir -p "$LOG_DIR" "$OUT_DIR" "$FIG_DIR" "$TABLE_DIR"

echo "GPU status before launch:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits
echo

cmd=(
  "$LEROBOT_PYTHON" "$LEROBOT_SOURCE/project/scripts/eval_zero_shot_D.py"
  --checkpoint-mode "$CHECKPOINT_MODE"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --log-freq "$LOG_FREQ"
  --device cuda:0
  --output-dir "$OUT_DIR"
  --figure-dir "$FIG_DIR"
  --table-dir "$TABLE_DIR"
)

if [[ -n "$MAX_QUERIES" ]]; then
  cmd+=(--max-queries "$MAX_QUERIES")
fi

echo "Launching Task 3 zero-shot D evaluation"
echo "RUN_ID=$RUN_ID"
echo "GPU_ID=$GPU_ID"
echo "CHECKPOINT_MODE=$CHECKPOINT_MODE"
echo "LOG_PATH=$LOG_PATH"
echo "Command: CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONUNBUFFERED=1 ${cmd[*]}"

CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 nohup "${cmd[@]}" >"$LOG_PATH" 2>&1 &
PID=$!

echo "PID=$PID"
echo "Monitor:"
echo "tail -f $LOG_PATH"
