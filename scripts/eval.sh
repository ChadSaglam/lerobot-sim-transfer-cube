#!/usr/bin/env bash
# Evaluate a policy in the AlohaTransferCube simulation with a fixed seed.
#
# Usage:
#   bash scripts/eval.sh <policy_path> <n_episodes> <job_name> [device]
# Examples:
#   bash scripts/eval.sh outputs/train/act_transfer_cube_seed1000/checkpoints/080000/pretrained_model 200 act_seed1000_080000_n200
#   bash scripts/eval.sh outputs/baseline/act_official_migrated 200 baseline_official_act_n200
#
# Output: outputs/eval/<job_name>/eval_info.json (success rate + 95% Wilson interval) and
#         outputs/eval/<job_name>/videos/ (the first 10 rollouts).
# Notes:
#   --eval.use_async_envs=false runs the simulators in the main process. Required on macOS,
#   where MuJoCo rendering crashes in worker processes (exit code -6).
set -eu
policy_path=${1:?usage: eval.sh <policy_path> <n_episodes> <job_name> [device]}
n_episodes=${2:?usage: eval.sh <policy_path> <n_episodes> <job_name> [device]}
job_name=${3:?usage: eval.sh <policy_path> <n_episodes> <job_name> [device]}
device=${4:-mps}

lerobot-eval \
  --policy.path="${policy_path}" \
  --policy.device="${device}" \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes="${n_episodes}" \
  --eval.batch_size=10 \
  --eval.use_async_envs=false \
  --seed=1000 \
  --output_dir="outputs/eval/${job_name}" \
  --job_name="${job_name}"
