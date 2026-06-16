#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/nohup_train_act_lang_ABC_task.sh" act_lang_ABC_full
