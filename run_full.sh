#!/bin/bash
# Phase 0 full runs: B1 -> B0 -> B2, Tier 0, 1.5B tokens each, seed 42.
# Usage: bash run_full.sh <B1_LR> <B2_LR>
#   B1_LR = winner of the B1 sweep; B0 (untied dense) uses B1's LR.
#   B2_LR = winner of the B2 sweep (B2 is swept separately - audit fix).
# Data: local FineWeb-Edu text only (--data local). Resume-aware: if a
# run's latest.pt exists, the run is continued with --resume.
set -u
B1_LR="${1:?usage: run_full.sh <B1_LR> <B2_LR>}"
B2_LR="${2:?usage: run_full.sh <B1_LR> <B2_LR>}"
cd /mnt/workspace
mkdir -p runs
LOG=runs/full.log

run_one() {
  local model="$1" lr="$2" dir="$3"
  echo "=== $(date '+%F %T') START $model lr=$lr dir=$dir ===" >> "$LOG"
  local resume=""
  if [ -f "$dir/latest.pt" ]; then resume="--resume"; fi
  python train.py --model "$model" --lr "$lr" --max-tokens 1.5B \
    --data local --text-dir /mnt/workspace/fw_text \
    --tokenizer bpe --tokenizer-file /mnt/workspace/tokenizer.json \
    --ckpt-dir "$dir" $resume >> "$LOG" 2>&1
  echo "=== $(date '+%F %T') END $model rc=$? ===" >> "$LOG"
}

run_one B1 "$B1_LR" runs/full_b1
run_one B0 "$B1_LR" runs/full_b0
run_one B2 "$B2_LR" runs/full_b2
echo "=== $(date '+%F %T') ALL FULL RUNS DONE ===" >> "$LOG"
