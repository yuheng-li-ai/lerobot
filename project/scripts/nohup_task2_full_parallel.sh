#!/usr/bin/env bash
set -euo pipefail

# IO-aware manual launcher for Task 2 full-scale jobs.
# Starts selected full experiments with nohup, refuses GPU0, and staggers launches
# to avoid all jobs hitting metadata/video/checkpoint IO at the same instant.

source /home/zengzixuan/cvprojects/calvin_env.sh

STAGGER_SECONDS="${STAGGER_SECONDS:-300}"
EXPERIMENTS=("$@")
if [[ ${#EXPERIMENTS[@]} -eq 0 ]]; then
  EXPERIMENTS=(
    act_ABC_full
    act_ABC_size_matched_full
    act_ABC_aug_full
    act_ABC_size_matched_aug_full
  )
fi

echo "Preparing Task 2 full parallel launch."
echo "Experiments: ${EXPERIMENTS[*]}"
echo "STAGGER_SECONDS=$STAGGER_SECONDS"
echo "GPU0 is reserved and launcher will refuse GPU_ID=0."
echo
echo "GPU state before launch:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
echo

for idx in "${!EXPERIMENTS[@]}"; do
  EXPERIMENT="${EXPERIMENTS[$idx]}"
  if [[ "$idx" -gt 0 && "$STAGGER_SECONDS" -gt 0 ]]; then
    echo "Waiting ${STAGGER_SECONDS}s before launching $EXPERIMENT..."
    sleep "$STAGGER_SECONDS"
  fi
  echo
  echo "Launching $EXPERIMENT"
  bash "$LEROBOT_SOURCE/project/scripts/nohup_train_task2.sh" "$EXPERIMENT"
done

echo
echo "All requested Task 2 full jobs have been submitted."
echo "Monitor logs under:"
echo "  $CALVIN_RUNS/task2/*/logs/"
