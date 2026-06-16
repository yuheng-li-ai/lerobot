#!/usr/bin/env bash
set -euo pipefail

# Launch CALVIN D official-style success-rate evaluation for the three
# language-conditioned ACT checkpoints:
#   - ACT-Lang-B 100k
#   - ACT-Lang-ABC-size-matched 100k
#   - ACT-Lang-ABC 200k
#
# Example:
#   GPU_ID=3 NUM_SEQUENCES=1000 ROLLOUT_ACTION_STEPS=25 bash project/scripts/nohup_eval_act_lang_success_rate_D.sh
#   tail -f /EXT_DISK/users/zengzixuan/calvin_runs/task3_success_rate_D/logs/<RUN_ID>.log

source /home/zengzixuan/cvprojects/calvin_env.sh >/dev/null

export CALVIN_REPO="${CALVIN_REPO:-/home/zengzixuan/cvprojects/calvin}"
export PYTHONPATH="${CALVIN_REPO}:${CALVIN_REPO}/calvin_models:${CALVIN_REPO}/calvin_env:${LEROBOT_SOURCE}:${LEROBOT_SOURCE}/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${CALVIN_RUNS}/matplotlib_cache"
export HF_HOME="${CALVIN_RUNS}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${CALVIN_RUNS}/torch_cache"
unset LEROBOT_HOME

GPU_ID="${GPU_ID:-0}"
NUM_SEQUENCES="${NUM_SEQUENCES:-1000}"
EP_LEN="${EP_LEN:-360}"
ROLLOUT_ACTION_STEPS="${ROLLOUT_ACTION_STEPS:-25}"
LOG_FREQ="${LOG_FREQ:-10}"
RUN_GROUP="${RUN_GROUP:-task3_success_rate_D}"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_act_lang_success_rate_D_gpu${GPU_ID}_n${NUM_SEQUENCES}_a${ROLLOUT_ACTION_STEPS}"
RUN_DIR="${CALVIN_RUNS}/${RUN_GROUP}/${RUN_ID}"
LOG_DIR="${CALVIN_RUNS}/${RUN_GROUP}/logs"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

mkdir -p "${RUN_DIR}" "${LOG_DIR}" project/tables/task3_success_rate_D project/figures/task3_success_rate_D

if ! "${LEROBOT_PYTHON}" - <<'PY' >/dev/null 2>&1
import calvin_env.envs.play_table_env
PY
then
  cat >&2 <<EOF
ERROR: CALVIN environment package is missing.

Expected import to work:
  import calvin_env.envs.play_table_env

Expected the calvin_env submodule/package under:
  ${CALVIN_REPO}/calvin_env

Initialize/install that package before running success-rate rollouts.
EOF
  exit 2
fi

cat > "${RUN_DIR}/run.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source /home/zengzixuan/cvprojects/calvin_env.sh >/dev/null
export CALVIN_REPO="${CALVIN_REPO}"
export PYTHONPATH="${PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR}"
export HF_HOME="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"
export TORCH_HOME="${TORCH_HOME}"
unset LEROBOT_HOME
cd "${LEROBOT_SOURCE}"
echo "RUN_ID=${RUN_ID}"
echo "GPU_ID=${GPU_ID}"
echo "NUM_SEQUENCES=${NUM_SEQUENCES}"
echo "EP_LEN=${EP_LEN}"
echo "ROLLOUT_ACTION_STEPS=${ROLLOUT_ACTION_STEPS}"
echo "RUN_DIR=${RUN_DIR}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${LEROBOT_PYTHON}" project/scripts/eval_act_lang_success_rate_D.py \\
  --dataset-path "${CALVIN_RAW}" \\
  --num-sequences "${NUM_SEQUENCES}" \\
  --ep-len "${EP_LEN}" \\
  --device cuda:0 \\
  --rollout-action-steps "${ROLLOUT_ACTION_STEPS}" \\
  --log-freq "${LOG_FREQ}" \\
  --output-dir "${RUN_DIR}" \\
  --table-dir project/tables/task3_success_rate_D \\
  --figure-dir project/figures/task3_success_rate_D
EOF
chmod +x "${RUN_DIR}/run.sh"

nohup bash "${RUN_DIR}/run.sh" > "${LOG_FILE}" 2>&1 &
PID="$!"

cat <<EOF
Launched ACT-Lang CALVIN D success-rate evaluation.
PID: ${PID}
RUN_DIR: ${RUN_DIR}
LOG_FILE: ${LOG_FILE}

Monitor:
  tail -f ${LOG_FILE}

Outputs:
  project/tables/task3_success_rate_D
  project/figures/task3_success_rate_D
  ${RUN_DIR}
EOF
