#!/usr/bin/env bash
# Run the full toy-1 / toy-2 comparison matrix.  Idempotent: existing JSONs are
# skipped, so it can be re-invoked after a partial run.  CPU by default.
#
# Stopped by directive (2026-11-25): toy-2 no_zero_lock / soft / SAT full+primal
# were killed mid-matrix.  They are intentionally NOT in this script; only the
# controls already needed for the post-mortem are kept, plus the untrained /
# neural-V diagnostics.
set -euo pipefail
cd "$(dirname "$0")"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
STEPS1="${STEPS1:-3000}"
STEPS2="${STEPS2:-1500}"
mkdir -p runs/toy1 runs/toy2 logs

python toys/toy1_semigroup.py --smoke
python toys/toy2_primal_dual.py --smoke

# ---- toy 1: ST engine, one-hop controls, ablations, index-order control ----
for m in st edge1hop softmin mlp vector; do
  if [ -f "runs/toy1/$m.json" ]; then echo "skip toy1/$m"; continue; fi
  echo "== toy1 $m"
  python toys/toy1_semigroup.py --model "$m" --steps "$STEPS1" --batch 32 \
    --eval-n 200 --lr 3e-3 --out "runs/toy1/$m.json" --device cpu \
    --log-every 500 > "logs/toy1_$m.log" 2>&1
done

# hard core: no explicit D supervision.  Earlier 8-17% numbers were an lr=1e-3
# artifact; lr=3e-3 learns H2-H8 (this is the run that overturns the
# "needs explicit supervision" claim).
if [ ! -f runs/toy1/st_hardcore.json ]; then
  echo "== toy1 st (hard core, no D supervision)"
  python toys/toy1_semigroup.py --model st --no-d-supervision --steps "$STEPS1" \
    --batch 32 --eval-n 200 --lr 3e-3 --out runs/toy1/st_hardcore.json \
    --device cpu --log-every 500 > logs/toy1_st_hardcore.log 2>&1
fi

# warm-start + tau anneal variant (fails; see handoff 3.7)
if [ ! -f runs/toy1/st_hardcore_warm.json ]; then
  echo "== toy1 st warm"
  python toys/toy1_semigroup.py --model st --no-d-supervision --steps "$STEPS1" \
    --batch 32 --eval-n 200 --lr 3e-3 --warm-frac 0.3 \
    --tau-start 2.0 --tau-end 0.05 --tau-anneal-frac 0.5 \
    --out runs/toy1/st_hardcore_warm.json --device cpu \
    --log-every 500 > logs/toy1_st_hardcore_warm.log 2>&1
fi

# node-permutation controls: if accuracy collapses, the st routing signal is an
# artifact of spine nodes occupying the low indices (argmin tie-break).
if [ ! -f runs/toy1/st_hardcore_perm.json ]; then
  echo "== toy1 st_hardcore with node permutation"
  python toys/toy1_semigroup.py --model st --no-d-supervision --node-perm \
    --steps "$STEPS1" --batch 32 --eval-n 200 --lr 3e-3 \
    --out runs/toy1/st_hardcore_perm.json --device cpu \
    --log-every 500 > logs/toy1_st_hardcore_perm.log 2>&1
fi
if [ ! -f runs/toy1/st_perm.json ]; then
  echo "== toy1 st with node permutation"
  python toys/toy1_semigroup.py --model st --node-perm --steps "$STEPS1" \
    --batch 32 --eval-n 200 --lr 3e-3 --out runs/toy1/st_perm.json \
    --device cpu --log-every 500 > logs/toy1_st_perm.log 2>&1
fi

# ---- toy 2: sudoku.  full + the 2 controls that survived the stop order. ----
for m in full open primal; do
  if [ -f "runs/toy2/sudoku_$m.json" ]; then echo "skip toy2/sudoku_$m"; continue; fi
  echo "== toy2 sudoku $m"
  python toys/toy2_primal_dual.py --dataset sudoku --mode "$m" --steps "$STEPS2" \
    --out "runs/toy2/sudoku_$m.json" > "logs/toy2_sudoku_$m.log" 2>&1
done

# untrained-encoder + same analytic solver: the control that exposed the false
# positive (the trained encoder is not better than random init).
if [ ! -f runs/toy2/sudoku_untrained.json ]; then
  echo "== toy2 sudoku untrained control"
  python toys/diag_toy2_auroc.py --dataset sudoku --mode full \
    --out runs/toy2/sudoku_untrained.json > logs/toy2_sudoku_untrained.log 2>&1
fi

# neural verifier V_theta: train on 2-cell hard negatives, then re-run the same
# solver with the learned V.  Pre-check (untrained V): T64 < 25% of cells.
if [ ! -f runs/toy2/sudoku_neuralv_vsuper.json ]; then
  echo "== toy2 sudoku learned-V (verifier-supervised)"
  python toys/toy2_primal_dual.py --dataset sudoku --mode full --neural-v \
    --v-train-steps 800 --steps "$STEPS2" \
    --out runs/toy2/sudoku_neuralv_vsuper.json > logs/toy2_sudoku_neuralv_vsuper.log 2>&1
fi

python make_toy_report.py
