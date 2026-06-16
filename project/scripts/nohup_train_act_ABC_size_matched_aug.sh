#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/nohup_train_task2.sh" act_ABC_size_matched_aug_full
