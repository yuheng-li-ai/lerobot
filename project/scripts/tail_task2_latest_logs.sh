#!/usr/bin/env bash
set -euo pipefail

# Tail latest Task 2 mini or full logs. Usage:
#   project/scripts/tail_task2_latest_logs.sh mini
#   project/scripts/tail_task2_latest_logs.sh full act_ABC

MODE="${1:-mini}"
GROUP="${2:-}"
source /home/zengzixuan/cvprojects/calvin_env.sh >/dev/null

latest_log() {
  local group="$1"
  local roots=(
    "$CALVIN_RUNS/task2/$group/logs"
    "$CALVIN_RUNS/$group/logs"
  )
  local found=""
  for root in "${roots[@]}"; do
    if [[ -d "$root" ]]; then
      local candidate
      candidate="$(find "$root" -maxdepth 1 -type f -name '*.log' | sort | tail -n 1)"
      if [[ -n "$candidate" ]]; then
        found="$candidate"
      fi
    fi
  done
  echo "$found"
}

case "$MODE" in
  mini)
    ROOT="$LEROBOT_SOURCE/project/logs/task2/mini_trials"
    latest="$(find "$ROOT" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
    if [[ -z "$latest" ]]; then
      echo "No mini-trial log directory found under $ROOT" >&2
      exit 1
    fi
    echo "Tailing latest Task 2 mini logs: $latest"
    tail -f "$latest"/*.log
    ;;
  full)
    if [[ -z "$GROUP" ]]; then
      echo "Usage: $0 full {act_ABC|act_ABC_aug|act_ABC_size_matched|act_ABC_size_matched_aug}" >&2
      exit 2
    fi
    latest="$(latest_log "$GROUP")"
    if [[ -z "$latest" ]]; then
      echo "No full log found for group $GROUP under $CALVIN_RUNS/task2/$GROUP/logs or $CALVIN_RUNS/$GROUP/logs" >&2
      exit 1
    fi
    echo "Tailing latest Task 2 full log: $latest"
    tail -f "$latest"
    ;;
  full-all)
    groups=(
      act_ABC
      act_ABC_size_matched
      act_ABC_aug
      act_ABC_size_matched_aug
    )
    logs=()
    for group in "${groups[@]}"; do
      latest="$(latest_log "$group")"
      if [[ -n "$latest" ]]; then
        logs+=("$latest")
      else
        echo "Warning: no log found yet for $group" >&2
      fi
    done
    if [[ ${#logs[@]} -eq 0 ]]; then
      echo "No Task 2 full logs found." >&2
      exit 1
    fi
    printf 'Tailing Task 2 full logs:\n'
    printf '  %s\n' "${logs[@]}"
    tail -f "${logs[@]}"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac
