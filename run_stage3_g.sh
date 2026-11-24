#!/usr/bin/env bash
# Stage 3 (Direction G): holographic coefficients vs exact param/VRAM-matched
# dense control on real FineWeb-Edu language modelling. Sequential, gated.
set -euo pipefail
cd "$(dirname "$0")"

COMMON="--max-tokens 150M --seq-len 1024 --batch-tokens 65536 --micro-batch 8 \
  --lr 3e-3 --seed 42 --tokenizer byte --data local \
  --text-dir /mnt/systemdisk/fw_mini --synthetic-fraction 0.0 \
  --eval-every 250 --eval-batches 32 --probe-every 1000 --ckpt-every 500 \
  --no-compile"

mkdir -p logs
echo "=== Stage 3 CI ==="
python test_efg.py 2>&1 | tee logs/stage3_ci.log

echo "=== Stage 3: DENSE_NARROW (17.052M non-emb) ==="
python train.py --model DENSE_NARROW $COMMON --ckpt-dir runs/stage3_dense_narrow \
  2>&1 | tee logs/stage3_dense_narrow.log

echo "=== Stage 3: G_WHT (17.063M non-emb) ==="
python train.py --model G_WHT $COMMON --ckpt-dir runs/stage3_g_wht \
  2>&1 | tee logs/stage3_g_wht.log

echo "=== Stage 3 evaluation ==="
python eval_stage3_g.py --ckpt-root runs --out runs/stage3_report.json \
  2>&1 | tee logs/stage3_eval.log

RUN_B0=$(python -c "import json;print(json.load(open('runs/stage3_report.json'))['verdict']['run_b0'])")
if [ "$RUN_B0" = "True" ]; then
  echo "=== Stage 3: B0_FULL (G beat DENSE_NARROW on fw_bpb) ==="
  python train.py --model B0 $COMMON --ckpt-dir runs/stage3_b0_full \
    2>&1 | tee logs/stage3_b0_full.log
  python eval_stage3_g.py --ckpt-root runs --out runs/stage3_report.json \
    2>&1 | tee logs/stage3_eval.log
else
  echo "G_WHT did not beat DENSE_NARROW on fw_bpb -> skipping B0_FULL (saves ~55 min)"
fi
echo "=== Stage 3 done ==="
