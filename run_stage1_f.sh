#!/usr/bin/env bash
# Stage 1 (Direction F): thin-boundary crucible, FLOP-matched controls,
# and the frozen OOD crucible. Sequential, gated, single V100.
set -euo pipefail
cd "$(dirname "$0")"

COMMON="--max-tokens 60M --seq-len 1024 --batch-tokens 65536 --micro-batch 8 \
  --lr 3e-3 --seed 42 --tokenizer byte --data fallback --synthetic-fraction 1.0 \
  --families graph,ast --eval-every 150 --probe-every 1000 --ckpt-every 450 \
  --no-compile"

mkdir -p logs
echo "=== Stage 1 CI ==="
python test_efg.py 2>&1 | tee logs/stage1_ci.log

for pair in "B2_THIN:stage1_b2_thin" "B2_NARROW:stage1_b2_narrow" "F_VCYCLE:stage1_f_vcycle"; do
  name="${pair%%:*}"; dir="${pair##*:}"
  echo "=== Stage 1 training $name -> runs/$dir ==="
  python train.py --model "$name" $COMMON --ckpt-dir "runs/$dir" \
    2>&1 | tee "logs/${dir}.log"
done

echo "=== Stage 1 evaluation (160 fixed OOD) ==="
python eval_stage1_f.py --ckpt-root runs --out runs/stage1_report.json \
  2>&1 | tee logs/stage1_eval.log
echo "=== Stage 1 done ==="
