#!/usr/bin/env bash
# Run the full toy-1 / toy-2 comparison matrix.  Idempotent: existing JSONs are
# skipped, so it can be re-invoked after a partial run.  CPU by default; the
# GPU stays free for the Stage-3 G_WHT job.
set -euo pipefail
cd "$(dirname "$0")"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
STEPS1="${STEPS1:-3000}"
STEPS2="${STEPS2:-1500}"
mkdir -p runs/toy1 runs/toy2 logs

python toys/toy1_semigroup.py --smoke
python toys/toy2_primal_dual.py --smoke

for m in st edge1hop softmin mlp vector; do
  if [ -f "runs/toy1/$m.json" ]; then echo "skip toy1/$m"; continue; fi
  echo "== toy1 $m"
  python toys/toy1_semigroup.py --model "$m" --steps "$STEPS1" --batch 32 \
    --eval-n 200 --lr 3e-3 --out "runs/toy1/$m.json" --device cpu \
    --log-every 500 > "logs/toy1_$m.log" 2>&1
done

if [ ! -f runs/toy1/st_hardcore.json ]; then
  echo "== toy1 st (hard core, no D supervision)"
  python toys/toy1_semigroup.py --model st --no-d-supervision --steps "$STEPS1" \
    --batch 32 --eval-n 200 --lr 3e-3 --out runs/toy1/st_hardcore.json \
    --device cpu --log-every 500 > logs/toy1_st_hardcore.log 2>&1
fi

for m in full open primal no_zero_lock soft; do
  if [ -f "runs/toy2/sudoku_$m.json" ]; then echo "skip toy2/sudoku_$m"; continue; fi
  echo "== toy2 sudoku $m"
  python toys/toy2_primal_dual.py --dataset sudoku --mode "$m" --steps "$STEPS2" \
    --out "runs/toy2/sudoku_$m.json" > "logs/toy2_sudoku_$m.log" 2>&1
done

for m in full open primal; do
  if [ -f "runs/toy2/sat_$m.json" ]; then echo "skip toy2/sat_$m"; continue; fi
  echo "== toy2 sat $m"
  python toys/toy2_primal_dual.py --dataset sat --mode "$m" --steps "$STEPS2" \
    --out "runs/toy2/sat_$m.json" > "logs/toy2_sat_$m.log" 2>&1
done

python make_toy_report.py
