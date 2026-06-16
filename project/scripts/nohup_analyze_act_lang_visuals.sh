#!/usr/bin/env bash
set -euo pipefail

# Manual launcher for ACT-Lang visualizations.
# This does not train. It loads completed ACT-Lang checkpoints and writes:
#   project/figures/act_lang_core/
#   project/figures/act_lang_language/
#   project/tables/act_lang_core/
#   project/tables/act_lang_language/

source /home/zengzixuan/cvprojects/calvin_env.sh

GPU_ID="${GPU_ID:-0}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-128}"
MAX_DATA_SAMPLES="${MAX_DATA_SAMPLES:-3000}"
MAX_PCA_SAMPLES="${MAX_PCA_SAMPLES:-1500}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PROJECT_DIR="${PROJECT_DIR:-$LEROBOT_SOURCE/project}"
ANALYZE_SCRIPT="${ANALYZE_SCRIPT:-$LEROBOT_SOURCE/project/scripts/analyze_act_lang_visuals.py}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export ACT_DEVICE_OVERRIDE="cuda:0"
export PYTHONUNBUFFERED=1

RUN_ID="$(date -u +%Y%m%d_%H%M%S)_act_lang_visuals_gpu${GPU_ID}"
RUN_GROUP="act_lang_visuals"
RUN_DIR="$CALVIN_RUNS/$RUN_GROUP/$RUN_ID"
LOG_DIR="$CALVIN_RUNS/$RUN_GROUP/logs"
LOG_FILE="$LOG_DIR/${RUN_ID}.log"
RUN_SCRIPT="$RUN_DIR/run.sh"

mkdir -p "$RUN_DIR" "$LOG_DIR"

echo "Starting ACT-Lang visualization analysis manually."
echo "GPU_ID=$GPU_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "ACT_DEVICE_OVERRIDE=$ACT_DEVICE_OVERRIDE"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "ANALYZE_SCRIPT=$ANALYZE_SCRIPT"
echo "MAX_EVAL_SAMPLES=$MAX_EVAL_SAMPLES"
echo "MAX_DATA_SAMPLES=$MAX_DATA_SAMPLES"
echo "MAX_PCA_SAMPLES=$MAX_PCA_SAMPLES"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "RUN_DIR=$RUN_DIR"
echo "LOG_FILE=$LOG_FILE"
echo

if [[ ! -f "$ANALYZE_SCRIPT" ]]; then
  echo "ERROR: Missing analysis script: $ANALYZE_SCRIPT" >&2
  exit 4
fi

echo "GPU state before launch:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
echo

cat > "$RUN_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
trap 'echo "FAILED at line \${LINENO} with exit code \$?"' ERR

source /home/zengzixuan/cvprojects/calvin_env.sh
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export ACT_DEVICE_OVERRIDE="cuda:0"
export PYTHONUNBUFFERED=1

echo "RUN_ID=$RUN_ID"
echo "Started at \$(date -Is)"
echo "GPU_ID=$GPU_ID"
echo "CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES"
echo "ACT_DEVICE_OVERRIDE=\$ACT_DEVICE_OVERRIDE"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "ANALYZE_SCRIPT=$ANALYZE_SCRIPT"
echo

echo "Python starts at \$(date -Is)"
set +e
"\$LEROBOT_PYTHON" "$ANALYZE_SCRIPT" \\
  --project-dir "$PROJECT_DIR" \\
  --max-eval-samples "$MAX_EVAL_SAMPLES" \\
  --max-data-samples "$MAX_DATA_SAMPLES" \\
  --max-pca-samples "$MAX_PCA_SAMPLES" \\
  --batch-size "$BATCH_SIZE"
status=\$?
set -e
echo "Python finished at \$(date -Is) with exit_code=\$status"
exit "\$status"
EOF

chmod +x "$RUN_SCRIPT"
nohup /bin/bash "$RUN_SCRIPT" > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$RUN_DIR/nohup.pid"

echo "Launched PID=$PID"
echo "PID file: $RUN_DIR/nohup.pid"
echo "Run script: $RUN_SCRIPT"
echo
echo "Monitor progress with:"
echo "  tail -f $LOG_FILE"
echo
echo "The log prints progress lines like:"
echo "  PROGRESS act_lang_visuals [########------------------------] 1/5 ( 20.0%) training_core done"
echo
echo "Outputs:"
echo "  $PROJECT_DIR/figures/act_lang_core/"
echo "  $PROJECT_DIR/figures/act_lang_language/"
echo "  $PROJECT_DIR/tables/act_lang_core/"
echo "  $PROJECT_DIR/tables/act_lang_language/"
