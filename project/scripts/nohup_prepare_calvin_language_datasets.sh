#!/usr/bin/env bash
# Launch language-aligned CALVIN B + ABC dataset generation in the background.
#
# Default behavior:
#   - CLEAN=1: remove previous calvin_lang_B / calvin_lang_ABC outputs first.
#   - FAST_STORAGE=1: use image storage (--no-videos), faster for many short
#     language-segment episodes than encoding thousands of tiny videos.
#   - Runs B first, then ABC.
#
# Usage:
#   bash project/scripts/nohup_prepare_calvin_language_datasets.sh
#
# Monitor:
#   tail -f <printed LOG_FILE>
#
# Options:
#   CLEAN=0 bash project/scripts/nohup_prepare_calvin_language_datasets.sh
#   FAST_STORAGE=0 bash project/scripts/nohup_prepare_calvin_language_datasets.sh
#   DRY_RUN_ONLY=1 bash project/scripts/nohup_prepare_calvin_language_datasets.sh
#   MAX_SEGMENTS=20 bash project/scripts/nohup_prepare_calvin_language_datasets.sh
#   READBACK_EPISODES=128 bash project/scripts/nohup_prepare_calvin_language_datasets.sh

set -euo pipefail

source /home/zengzixuan/cvprojects/calvin_env.sh

export HF_LEROBOT_HOME="${CALVIN_LEROBOT_ROOT}"
export HF_HOME="${CALVIN_RUNS}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${CALVIN_RUNS}/torch_cache"
export MPLCONFIGDIR="${CALVIN_RUNS}/matplotlib_cache"

SCRIPT="${LEROBOT_SOURCE}/project/scripts/prepare_calvin_language_dataset.py"
TABLES_DIR="${LEROBOT_SOURCE}/project/tables"
RUN_ID="$(date +%Y%m%d_%H%M%S)_calvin_lang_datasets"
LOG_DIR="${CALVIN_RUNS}/language_datasets/logs"
RUN_DIR="${CALVIN_RUNS}/language_datasets/${RUN_ID}"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
PID_FILE="${RUN_DIR}/nohup.pid"
RUN_SCRIPT="${RUN_DIR}/run.sh"

CLEAN="${CLEAN:-1}"
FAST_STORAGE="${FAST_STORAGE:-1}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-0}"
MAX_SEGMENTS="${MAX_SEGMENTS:-}"
READBACK_EPISODES="${READBACK_EPISODES:-64}"
PROGRESS_LOG_EVERY="${PROGRESS_LOG_EVERY:-25}"

mkdir -p "${LOG_DIR}" "${RUN_DIR}" "${HF_DATASETS_CACHE}" "${MPLCONFIGDIR}"

COMMON_ARGS=(
  --tables-dir "${TABLES_DIR}"
  --readback-episodes "${READBACK_EPISODES}"
  --progress-log-every "${PROGRESS_LOG_EVERY}"
)

if [[ -n "${MAX_SEGMENTS}" ]]; then
  COMMON_ARGS+=(--max-segments "${MAX_SEGMENTS}")
fi

if [[ "${FAST_STORAGE}" == "1" ]]; then
  COMMON_ARGS+=(--no-videos)
fi

COMMON_ARGS_Q=""
for arg in "${COMMON_ARGS[@]}"; do
  COMMON_ARGS_Q+=" $(printf "%q" "${arg}")"
done

cat <<EOF
Launching language-aligned CALVIN dataset generation.
RUN_ID=${RUN_ID}
CLEAN=${CLEAN}
FAST_STORAGE=${FAST_STORAGE}
DRY_RUN_ONLY=${DRY_RUN_ONLY}
MAX_SEGMENTS=${MAX_SEGMENTS:-<full>}
READBACK_EPISODES=${READBACK_EPISODES}
PROGRESS_LOG_EVERY=${PROGRESS_LOG_EVERY}

Outputs:
  ${CALVIN_LEROBOT_ROOT}/calvin_lang_B
  ${CALVIN_LEROBOT_ROOT}/calvin_lang_ABC

Log:
  ${LOG_FILE}
EOF

cat > "${RUN_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
trap 'echo "FAILED at line \${LINENO} with exit code \$?"' ERR

source /home/zengzixuan/cvprojects/calvin_env.sh
export HF_LEROBOT_HOME="${CALVIN_LEROBOT_ROOT}"
export HF_HOME="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"
export TORCH_HOME="${TORCH_HOME}"
export MPLCONFIGDIR="${MPLCONFIGDIR}"

SCRIPT="${SCRIPT}"
TABLES_DIR="${TABLES_DIR}"
COMMON_ARGS=(${COMMON_ARGS_Q})

echo "RUN_ID=${RUN_ID}"
echo "Started at \$(date -Is)"
echo "CALVIN_RAW=\${CALVIN_RAW}"
echo "CALVIN_LEROBOT_ROOT=\${CALVIN_LEROBOT_ROOT}"
echo "CALVIN_RUNS=\${CALVIN_RUNS}"
echo "LEROBOT_SOURCE=\${LEROBOT_SOURCE}"
echo "LEROBOT_PYTHON=\${LEROBOT_PYTHON}"
echo "HF_HOME=${HF_HOME}"
echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
echo "TORCH_HOME=${TORCH_HOME}"
echo "FAST_STORAGE=${FAST_STORAGE}"
echo "DRY_RUN_ONLY=${DRY_RUN_ONLY}"
echo

if [[ "${CLEAN}" == "1" ]]; then
  echo "Cleaning previous language datasets and generated language tables..."
  rm -rf \
    "${CALVIN_LEROBOT_ROOT}/calvin_lang_B" \
    "${CALVIN_LEROBOT_ROOT}/calvin_lang_ABC"
  rm -f \
    "${TABLES_DIR}/calvin_lang_B_manifest.csv" \
    "${TABLES_DIR}/calvin_lang_B_episode_splits.csv" \
    "${TABLES_DIR}/calvin_lang_B_readback_checks.csv" \
    "${TABLES_DIR}/calvin_lang_B_summary.json" \
    "${TABLES_DIR}/calvin_lang_ABC_manifest.csv" \
    "${TABLES_DIR}/calvin_lang_ABC_episode_splits.csv" \
    "${TABLES_DIR}/calvin_lang_ABC_readback_checks.csv" \
    "${TABLES_DIR}/calvin_lang_ABC_summary.json"
fi

echo
echo "Step 1/4: dry-run B"
"\${LEROBOT_PYTHON}" "\${SCRIPT}" --envs B --dry-run "\${COMMON_ARGS[@]}"

echo
echo "Step 2/4: dry-run ABC"
"\${LEROBOT_PYTHON}" "\${SCRIPT}" --envs ABC --dry-run "\${COMMON_ARGS[@]}"

if [[ "${DRY_RUN_ONLY}" == "1" ]]; then
  echo
  echo "DRY_RUN_ONLY=1, stopping before dataset generation."
  exit 0
fi

echo
echo "Step 3/4: generate B"
"\${LEROBOT_PYTHON}" "\${SCRIPT}" --envs B --overwrite "\${COMMON_ARGS[@]}"

echo
echo "Step 4/4: generate ABC"
"\${LEROBOT_PYTHON}" "\${SCRIPT}" --envs ABC --overwrite "\${COMMON_ARGS[@]}"

echo
echo "Finished at \$(date -Is)"
echo "Final sizes:"
du -sh \
  "${CALVIN_LEROBOT_ROOT}/calvin_lang_B" \
  "${CALVIN_LEROBOT_ROOT}/calvin_lang_ABC"
EOF

chmod +x "${RUN_SCRIPT}"
nohup /bin/bash "${RUN_SCRIPT}" > "${LOG_FILE}" 2>&1 &

PID="$!"
echo "${PID}" > "${PID_FILE}"

cat <<EOF

Launched PID=${PID}
PID file:
  ${PID_FILE}

Tail progress with:
  tail -f ${LOG_FILE}

Look for lines like:
  PROGRESS calvin_lang_B [########------------------------] 1500/6115 ( 24.5%) elapsed=... eta=...
  PROGRESS calvin_lang_ABC [########------------------------] 4500/17870 ( 25.2%) elapsed=... eta=...
EOF
