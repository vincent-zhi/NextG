#!/usr/bin/env bash
set -uo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHON=python3
export MAX_TOKENS=100M

echo "=== EXTEND: resume B2/TB/TC_TAPE to 100M ==="
bash sprint.sh train
echo "EXTEND_TRAIN_EXIT:$?"

echo "=== B2_RAND: train confound control B2 --rand-loops to 100M ==="
bash sprint.sh b2rand
echo "B2RAND_EXIT:$?"

echo "=== EVAL: all checkpoints (incl b2_rand) ==="
bash sprint.sh eval
echo "EVAL_EXIT:$?"
echo "ALL_DONE"
