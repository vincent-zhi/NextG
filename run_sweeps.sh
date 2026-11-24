#!/bin/bash
# Phase 0 LR sweeps: B1 and B2 get SEPARATE 3-point sweeps (round-2 audit
# fix - B2's Loop-muP residual scale makes B1's LR non-transferable).
# Points: 0.0015 / 0.003 / 0.006 (0.5x / 1x / 2x of the 3e-3 centre),
# 0.15B tokens each, Tier 0, seed 42, --data local.
# Usage: bash run_sweeps.sh
set -u
cd /mnt/workspace
mkdir -p runs
LOG=runs/sweep.log

sweep_one() {
  local model="$1" lr="$2" dir="$3"
  echo "=== $(date '+%F %T') START sweep $model lr=$lr dir=$dir ===" >> "$LOG"
  python train.py --model "$model" --lr "$lr" --max-tokens 0.15B \
    --data local --text-dir /mnt/workspace/fw_text \
    --tokenizer bpe --tokenizer-file /mnt/workspace/tokenizer.json \
    --ckpt-dir "$dir" >> "$LOG" 2>&1
  echo "=== $(date '+%F %T') END sweep $model lr=$lr rc=$? ===" >> "$LOG"
}

sweep_one B1 0.0015 runs/b1_lr1
sweep_one B1 0.003  runs/b1_lr2
sweep_one B1 0.006  runs/b1_lr3
sweep_one B2 0.0015 runs/b2_lr1
sweep_one B2 0.003  runs/b2_lr2
sweep_one B2 0.006  runs/b2_lr3
echo "=== $(date '+%F %T') ALL SWEEPS DONE ===" >> "$LOG"
