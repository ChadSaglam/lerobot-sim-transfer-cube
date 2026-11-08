#!/usr/bin/env bash
# Overnight queue for the ACT / AlohaTransferCube experiments.
#
# Everything runs sequentially on purpose: training and evaluation share the
# same Apple GPU (MPS), and two jobs at once halve the speed of both.
#
# 1. Wait for the seed-1000 training that is already running to finish.
# 2. Evaluate its checkpoints (50 episodes each; 200 for the final one).
# 3. Evaluate the official baseline with 200 episodes for a tighter interval.
# 4. Train a second seed (1001) with the exact same config.
# 5. Evaluate its checkpoints the same way.
#
# Usage (from the project root, inside a caffeinate so the Mac stays awake):
#   nohup caffeinate -is bash scripts/overnight_queue.sh > outputs/train/overnight_queue.log 2>&1 &

set -u
# Detach stdin: macOS nohup leaves it attached to the terminal, and a Python process started
# after that terminal is closed dies with "init_sys_streams: Bad file descriptor".
exec < /dev/null
cd "$(dirname "$0")/.." || exit 1
# shellcheck disable=SC1091
source .venv/bin/activate

DATASET=lerobot/aloha_sim_transfer_cube_human
TASK=AlohaTransferCube-v0
STEPS=50000
BATCH=8
SAVE_FREQ=10000
EVAL_SEED=1000
BASELINE=outputs/baseline/act_official_migrated

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

evaluate() {
  # evaluate <policy_path> <n_episodes> <job_name>
  local policy_path=$1 n_episodes=$2 job_name=$3
  local out="outputs/eval/${job_name}"
  if [ -f "${out}/eval_info.json" ]; then
    log "skip ${job_name} (already evaluated)"
    return 0
  fi
  log "eval ${job_name}: ${n_episodes} episodes"
  lerobot-eval \
    --policy.path="${policy_path}" \
    --policy.device=mps \
    --env.type=aloha \
    --env.task="${TASK}" \
    --eval.n_episodes="${n_episodes}" \
    --eval.batch_size=10 \
    --eval.use_async_envs=false \
    --seed="${EVAL_SEED}" \
    --output_dir="${out}" \
    --job_name="${job_name}" \
    > "outputs/eval/${job_name}.log" 2>&1
  log "eval ${job_name} exit code $?"
}

evaluate_run() {
  # evaluate_run <seed>: all checkpoints with 50 episodes, the final one also with 200.
  local seed=$1
  local run="outputs/train/act_transfer_cube_seed${seed}"
  local ck step
  for ck in "${run}"/checkpoints/0*/pretrained_model; do
    [ -d "${ck}" ] || continue
    step=$(basename "$(dirname "${ck}")")
    evaluate "${ck}" 50 "act_seed${seed}_${step}_n50"
  done
  if [ -d "${run}/checkpoints/050000/pretrained_model" ]; then
    evaluate "${run}/checkpoints/050000/pretrained_model" 200 "act_seed${seed}_050000_n200"
  fi
}

train_seed() {
  # train_seed <seed>
  local seed=$1
  local out="outputs/train/act_transfer_cube_seed${seed}"
  if [ -d "${out}/checkpoints/050000" ]; then
    log "skip training seed ${seed} (final checkpoint exists)"
    return 0
  fi
  log "train seed ${seed}: ${STEPS} steps, batch ${BATCH}"
  lerobot-train \
    --policy.type=act \
    --policy.device=mps \
    --policy.push_to_hub=false \
    --dataset.repo_id="${DATASET}" \
    --env.type=aloha \
    --env.task="${TASK}" \
    --env_eval_freq=0 \
    --seed="${seed}" \
    --steps="${STEPS}" \
    --batch_size="${BATCH}" \
    --save_freq="${SAVE_FREQ}" \
    --log_freq=200 \
    --output_dir="${out}" \
    --job_name="act_transfer_cube_seed${seed}" \
    > "${out}.log" 2>&1
  log "train seed ${seed} exit code $?"
}

mkdir -p outputs/eval outputs/train

# 1. Wait for the training that is already running (started by hand).
log "waiting for the running lerobot-train to finish"
while pgrep -f "lerobot-train --policy.type=act" > /dev/null; do
  sleep 120
done
log "no lerobot-train running, queue starts"

# 2. + 3. Seed 1000 checkpoints and the baseline with a tighter interval.
evaluate_run 1000
evaluate "${BASELINE}" 200 "baseline_official_act_n200"

# 4. + 5. Second seed.
train_seed 1001
evaluate_run 1001

log "queue finished"
