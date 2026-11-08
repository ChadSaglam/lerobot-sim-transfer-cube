#!/usr/bin/env bash
# Resume a run from its last checkpoint and train up to a new step count.
# Optimizer state and data order continue from the checkpoint; only --steps changes.
#
# Usage:
#   bash scripts/resume.sh <seed> <target_steps>
# Example (seed 1000, 50k -> 80k):
#   bash scripts/resume.sh 1000 80000
set -eu
seed=${1:?usage: resume.sh <seed> <target_steps>}
target=${2:?usage: resume.sh <seed> <target_steps>}
run="outputs/train/act_transfer_cube_seed${seed}"

lerobot-train \
  --resume=true \
  --config_path="${run}/checkpoints/last/pretrained_model/train_config.json" \
  --steps="${target}"
