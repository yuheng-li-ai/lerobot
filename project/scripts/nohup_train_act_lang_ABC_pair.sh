#!/usr/bin/env bash
set -euo pipefail

# Launch ACT-Lang-ABC full first, then ACT-Lang-ABC-size-matched after an IO
# staggering delay. This avoids both jobs hitting dataset metadata/cache startup
# at the same instant.
#
# Usage:
#   bash project/scripts/nohup_train_act_lang_ABC_pair.sh
#
# Options:
#   ABC_GPU_ID=1 SIZE_GPU_ID=2 STAGGER_SECONDS=300 bash project/scripts/nohup_train_act_lang_ABC_pair.sh

source /home/zengzixuan/cvprojects/calvin_env.sh

ABC_GPU_ID="${ABC_GPU_ID:-1}"
SIZE_GPU_ID="${SIZE_GPU_ID:-2}"
STAGGER_SECONDS="${STAGGER_SECONDS:-300}"

if [[ "$ABC_GPU_ID" == "$SIZE_GPU_ID" ]]; then
  echo "ERROR: ABC_GPU_ID and SIZE_GPU_ID must be different." >&2
  exit 2
fi

echo "Launching ACT-Lang-ABC pair."
echo "ABC full GPU: $ABC_GPU_ID"
echo "ABC size-matched GPU: $SIZE_GPU_ID"
echo "Stagger delay: ${STAGGER_SECONDS}s"
echo

echo "Step 1/2: launch ACT-Lang-ABC full"
START_DELAY_SECONDS=0 GPU_ID="$ABC_GPU_ID" \
  bash "$LEROBOT_SOURCE/project/scripts/nohup_train_act_lang_ABC.sh"

echo
echo "Waiting ${STAGGER_SECONDS}s before launching ACT-Lang-ABC-size-matched..."
for remaining in $(seq "$STAGGER_SECONDS" -30 1); do
  echo "PAIR_STAGGER remaining=${remaining}s"
  sleep 30
done

echo
echo "Step 2/2: launch ACT-Lang-ABC-size-matched"
START_DELAY_SECONDS=0 GPU_ID="$SIZE_GPU_ID" \
  bash "$LEROBOT_SOURCE/project/scripts/nohup_train_act_lang_ABC_size_matched.sh"

echo
echo "Both ABC jobs have been launched."
echo "Check the log paths printed above with tail -f."
