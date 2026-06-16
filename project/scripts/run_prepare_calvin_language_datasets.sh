#!/usr/bin/env bash
# Build CALVIN language-aligned LeRobot datasets for official success-rate training.
#
# Outputs:
#   /EXT_DISK/users/zengzixuan/processed-calvin/calvin_lang_B
#   /EXT_DISK/users/zengzixuan/processed-calvin/calvin_lang_ABC
#
# Usage:
#   bash project/scripts/run_prepare_calvin_language_datasets.sh
#
# Optional:
#   OVERWRITE=0 bash project/scripts/run_prepare_calvin_language_datasets.sh
#   DRY_RUN_ONLY=1 bash project/scripts/run_prepare_calvin_language_datasets.sh
#   MAX_SEGMENTS=20 bash project/scripts/run_prepare_calvin_language_datasets.sh

set -euo pipefail

source /home/zengzixuan/cvprojects/calvin_env.sh

export HF_LEROBOT_HOME="${CALVIN_LEROBOT_ROOT}"
export HF_HOME="${CALVIN_RUNS}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${CALVIN_RUNS}/torch_cache"

SCRIPT="${LEROBOT_SOURCE}/project/scripts/prepare_calvin_language_dataset.py"
TABLES_DIR="${LEROBOT_SOURCE}/project/tables"

OVERWRITE="${OVERWRITE:-1}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-0}"
MAX_SEGMENTS="${MAX_SEGMENTS:-}"
READBACK_EPISODES="${READBACK_EPISODES:-64}"

COMMON_ARGS=(
  --tables-dir "${TABLES_DIR}"
  --readback-episodes "${READBACK_EPISODES}"
)

if [[ -n "${MAX_SEGMENTS}" ]]; then
  COMMON_ARGS+=(--max-segments "${MAX_SEGMENTS}")
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  WRITE_ARGS=(--overwrite)
else
  WRITE_ARGS=()
fi

echo "CALVIN_RAW=${CALVIN_RAW}"
echo "CALVIN_LEROBOT_ROOT=${CALVIN_LEROBOT_ROOT}"
echo "CALVIN_RUNS=${CALVIN_RUNS}"
echo "LEROBOT_SOURCE=${LEROBOT_SOURCE}"
echo "LEROBOT_PYTHON=${LEROBOT_PYTHON}"
echo "HF_HOME=${HF_HOME}"
echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
echo "TORCH_HOME=${TORCH_HOME}"
echo

echo "Step 1/4: dry-run B"
"${LEROBOT_PYTHON}" "${SCRIPT}" --envs B --dry-run "${COMMON_ARGS[@]}"

echo
echo "Step 2/4: dry-run ABC"
"${LEROBOT_PYTHON}" "${SCRIPT}" --envs ABC --dry-run "${COMMON_ARGS[@]}"

if [[ "${DRY_RUN_ONLY}" == "1" ]]; then
  echo
  echo "DRY_RUN_ONLY=1, stopping before dataset generation."
  exit 0
fi

echo
echo "Step 3/4: generate language-aligned B dataset"
"${LEROBOT_PYTHON}" "${SCRIPT}" --envs B "${WRITE_ARGS[@]}" "${COMMON_ARGS[@]}"

echo
echo "Step 4/4: generate language-aligned ABC dataset"
"${LEROBOT_PYTHON}" "${SCRIPT}" --envs ABC "${WRITE_ARGS[@]}" "${COMMON_ARGS[@]}"

echo
echo "Done."
echo "B dataset:   ${CALVIN_LEROBOT_ROOT}/calvin_lang_B"
echo "ABC dataset: ${CALVIN_LEROBOT_ROOT}/calvin_lang_ABC"
echo
echo "Key tables:"
echo "  ${TABLES_DIR}/calvin_lang_B_summary.json"
echo "  ${TABLES_DIR}/calvin_lang_B_manifest.csv"
echo "  ${TABLES_DIR}/calvin_lang_B_episode_splits.csv"
echo "  ${TABLES_DIR}/calvin_lang_ABC_summary.json"
echo "  ${TABLES_DIR}/calvin_lang_ABC_manifest.csv"
echo "  ${TABLES_DIR}/calvin_lang_ABC_episode_splits.csv"
