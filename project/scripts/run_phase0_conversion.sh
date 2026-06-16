#!/usr/bin/env bash
set -euo pipefail

source /home/zengzixuan/cvprojects/calvin_env.sh
unset LEROBOT_HOME
export HF_LEROBOT_HOME="${CALVIN_LEROBOT_ROOT}"
export HF_HOME="${CALVIN_RUNS}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PYTHONUNBUFFERED=1

cd /home/zengzixuan/cvprojects/lerobot

PY=/home/zengzixuan/miniforge3/envs/lerobot/bin/python
LOG_DIR="${CALVIN_RUNS}/phase0_conversion_logs"
mkdir -p "${LOG_DIR}" "${HF_DATASETS_CACHE}"

"${PY}" -u project/scripts/prepare_dataset_stats.py \
  --raw-root "${CALVIN_RAW}" \
  --output-dir project

nohup "${PY}" -u project/scripts/convert_calvin_to_lerobot.py \
  --raw-root "${CALVIN_RAW}" \
  --output-root "${CALVIN_LEROBOT_ROOT}" \
  --datasets B ABC D \
  --repo-prefix calvin \
  --overwrite \
  --segment-frames 3000 \
  --image-writer-threads 8 \
  > "${LOG_DIR}/convert_B_ABC_D.log" 2>&1 &

echo "Started Phase 0 LeRobot conversion as PID $!"
echo "Log: ${LOG_DIR}/convert_B_ABC_D.log"
