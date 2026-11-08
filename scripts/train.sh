#!/usr/bin/env bash
# Train ACT from scratch on lerobot/aloha_sim_transfer_cube_human.
#
# Usage:
#   bash scripts/train.sh <seed> [steps] [device]
# Example (the runs in the README):
#   bash scripts/train.sh 1000 50000 mps
#
# Output: outputs/train/act_transfer_cube_seed<seed>/checkpoints/<step>/pretrained_model
# Notes:
#   --env_eval_freq=0 disables in-training rollouts; evaluate checkpoints with scripts/eval.sh.
#   On macOS this is required: MuJoCo cannot render inside the async worker processes.
set -eu
seed=${1:?usage: train.sh <seed> [steps] [device]}
steps=${2:-50000}
device=${3:-mps}

lerobot-train \
  --policy.type=act \
  --policy.device="${device}" \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --env_eval_freq=0 \
  --seed="${seed}" \
  --steps="${steps}" \
  --batch_size=8 \
  --save_freq=10000 \
  --log_freq=200 \
  --output_dir="outputs/train/act_transfer_cube_seed${seed}" \
  --job_name="act_transfer_cube_seed${seed}"
