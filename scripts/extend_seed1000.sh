#!/usr/bin/env bash
# Extend the seed-1000 run from 50k to 80k steps so it matches the step count of the
# official checkpoint (lerobot/act_aloha_sim_transfer_cube_human was trained for 80k).
#
# 1. Wait until scripts/overnight_queue.sh has finished (nothing else may use the GPU).
# 2. Resume seed 1000 from checkpoints/last (50k) and train to 80k. Optimizer state and
#    data order continue from the checkpoint; only --steps is overridden.
# 3. Evaluate the new checkpoints: 60k/70k/80k with 50 episodes, 80k also with 200.
#
# Usage (from the project root):
#   nohup caffeinate -is bash scripts/extend_seed1000.sh > outputs/train/extend_seed1000.log 2>&1 &

set -u
# Detach stdin: macOS nohup leaves it attached to the terminal, and a Python process started
# after that terminal is closed dies with "init_sys_streams: Bad file descriptor".
exec < /dev/null
cd "$(dirname "$0")/.." || exit 1
# shellcheck disable=SC1091
source .venv/bin/activate

RUN=outputs/train/act_transfer_cube_seed1000
TASK=AlohaTransferCube-v0
EVAL_SEED=1000
TARGET_STEPS=80000

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

evaluate() {
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

# 1. Wait for the overnight queue (and anything else on the GPU) to finish.
log "waiting for overnight_queue.sh to finish"
while ! grep -q "queue finished" outputs/train/overnight_queue.log 2>/dev/null \
      || pgrep -f "lerobot-(train|eval)" > /dev/null; do
  sleep 120
done
log "GPU is free, extending seed 1000 to ${TARGET_STEPS} steps"

# 2. Resume to 80k.
if [ -d "${RUN}/checkpoints/080000" ]; then
  log "skip training (080000 checkpoint exists)"
else
  lerobot-train \
    --resume=true \
    --config_path="${RUN}/checkpoints/last/pretrained_model/train_config.json" \
    --steps="${TARGET_STEPS}" \
    > "${RUN}_resume80k.log" 2>&1
  log "resume exit code $?"
fi

# 3. Evaluate the new checkpoints.
for step in 060000 070000 080000; do
  ck="${RUN}/checkpoints/${step}/pretrained_model"
  [ -d "${ck}" ] && evaluate "${ck}" 50 "act_seed1000_${step}_n50"
done
if [ -d "${RUN}/checkpoints/080000/pretrained_model" ]; then
  evaluate "${RUN}/checkpoints/080000/pretrained_model" 200 "act_seed1000_080000_n200"
fi

log "extend finished"
