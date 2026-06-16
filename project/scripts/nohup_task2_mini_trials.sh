#!/usr/bin/env bash
set -euo pipefail

# Launch the four Task 2 mini-trials under nohup and wait for completion.
# This is a bounded harness: 20 steps per experiment, batch size 2, no GPU0.

source /home/zengzixuan/cvprojects/calvin_env.sh

RUN_ID="$(date -u +%Y%m%d_%H%M%S)_task2_mini"
BASE_DIR="$LEROBOT_SOURCE/project/outputs/task2"
LOG_DIR="$LEROBOT_SOURCE/project/logs/task2/mini_trials/$RUN_ID"
mkdir -p "$BASE_DIR" "$LOG_DIR"

EXPERIMENTS=(
  "act_ABC:1:$LEROBOT_SOURCE/project/configs/act_ABC.yaml"
  "act_ABC_size_matched:2:$LEROBOT_SOURCE/project/configs/act_ABC_size_matched.yaml"
  "act_ABC_aug:3:$LEROBOT_SOURCE/project/configs/act_ABC_aug.yaml"
  "act_ABC_size_matched_aug:4:$LEROBOT_SOURCE/project/configs/act_ABC_size_matched_aug.yaml"
)

echo "Starting Task 2 mini-trial harness."
echo "RUN_ID=$RUN_ID"
echo "LOG_DIR=$LOG_DIR"
echo "GPU0 is reserved and will not be used."
echo
echo "GPU state before launch:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
echo

pids=()
status_files=()
for item in "${EXPERIMENTS[@]}"; do
  IFS=":" read -r EXPERIMENT GPU_ID CONFIG <<< "$item"
  if [[ "$GPU_ID" == "0" ]]; then
    echo "Refusing to launch $EXPERIMENT on GPU0." >&2
    exit 3
  fi

  OUT_DIR="$BASE_DIR/$EXPERIMENT/mini_trials/$RUN_ID"
  LOG_FILE="$LOG_DIR/${EXPERIMENT}_gpu${GPU_ID}.log"
  STATUS_FILE="$LOG_DIR/${EXPERIMENT}_gpu${GPU_ID}.status"
  mkdir -p "$OUT_DIR"
  status_files+=("$STATUS_FILE")

  echo "Launching $EXPERIMENT mini-trial on physical GPU $GPU_ID"
  echo "  CONFIG=$CONFIG"
  echo "  OUT_DIR=$OUT_DIR"
  echo "  LOG_FILE=$LOG_FILE"

  (
    set +e
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    ACT_DEVICE_OVERRIDE="cuda:0" \
    PYTHONUNBUFFERED=1 \
      "$LEROBOT_PYTHON" "$LEROBOT_SOURCE/project/scripts/train_act.py" \
        --config "$CONFIG" \
        --output-dir "$OUT_DIR" \
        > "$LOG_FILE" 2>&1
    code=$?
    echo "$code" > "$STATUS_FILE"
    exit "$code"
  ) &
  pids+=("$!")
done

echo
echo "Mini-trials launched. PIDs: ${pids[*]}"
echo "Tail one log with:"
echo "  tail -f $LOG_DIR/act_ABC_gpu1.log"
echo "Or tail all logs with:"
echo "  tail -f $LOG_DIR/*.log"
echo

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

echo
echo "Task 2 mini-trial harness completed."
for status_file in "${status_files[@]}"; do
  if [[ -f "$status_file" ]]; then
    echo "$(basename "$status_file" .status): exit $(cat "$status_file")"
  else
    echo "$(basename "$status_file" .status): missing status"
    failed=1
  fi
done
echo
echo "GPU state after harness:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv

exit "$failed"
