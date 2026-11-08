#!/usr/bin/env bash
# Convert the official checkpoint to the lerobot 0.7.0 processor format.
# The Hub checkpoint predates the processor pipeline, so lerobot-eval cannot load it directly.
# Nothing is pushed anywhere; the migrated copy is written locally.
#
# Usage:
#   bash scripts/migrate_baseline.sh
set -eu
python -m lerobot.processor.migrate_policy_normalization \
  --pretrained-path lerobot/act_aloha_sim_transfer_cube_human \
  --output-dir outputs/baseline/act_official_migrated
