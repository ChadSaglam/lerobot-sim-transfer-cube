#!/usr/bin/env bash
# Live status screen for the overnight runs. Refreshes every 10 s. Quit with Ctrl+C.
cd "$(dirname "$0")/.." || exit 1

latest_progress() {
  # last tqdm progress line of a training log (tqdm uses \r, not \n)
  tail -c 2000 "$1" 2>/dev/null | tr '\r' '\n' | grep -E "/[0-9]+ \[" | tail -1 | sed 's/^Training: *//'
}
latest_loss() {
  tr '\r' '\n' < "$1" 2>/dev/null | grep -a "step:" | tail -1 | grep -o "step:[0-9K]* .*grdn:[0-9.]*"
}

while true; do
  clear
  echo "ACT / AlohaTransferCube   $(date '+%H:%M:%S')"
  echo "--------------------------------------------------------------"
  for seed in 1000 1001; do
    log="outputs/train/act_transfer_cube_seed${seed}.log"
    [ -f "outputs/train/act_transfer_cube_seed${seed}_resume80k.log" ] && log="outputs/train/act_transfer_cube_seed${seed}_resume80k.log"
    if [ -f "$log" ]; then
      echo "seed ${seed}:  $(latest_progress "$log")"
      echo "            $(latest_loss "$log")"
      echo "            checkpoints: $(ls outputs/train/act_transfer_cube_seed${seed}/checkpoints 2>/dev/null | grep -v last | tr '\n' ' ')"
    fi
  done
  echo "--------------------------------------------------------------"
  echo "queue:"
  newest=$(ls -t outputs/train/detached_*.log outputs/train/overnight_queue.log outputs/train/extend_seed1000.log 2>/dev/null | head -1)
  [ -n "$newest" ] && echo "  ($newest)" && tail -4 "$newest" | cut -c1-110 | sed 's/^/  /'
  echo "--------------------------------------------------------------"
  echo "eval results so far:"
  for f in outputs/eval/*.log; do
    [ -f "$f" ] || continue
    r=$(grep "Success rate" "$f" | tail -1 | sed 's/.*bot_eval.py:[0-9]* //')
    [ -n "$r" ] && printf '  %-32s %s\n' "$(basename "$f" .log)" "$r"
  done
  echo
  echo "processes: $(pgrep -f 'lerobot-(train|eval)' | wc -l | tr -d ' ')   (Ctrl+C to quit this screen; the runs keep going)"
  sleep 10
done
