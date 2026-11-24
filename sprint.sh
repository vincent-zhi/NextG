#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-python}"
RUNS_DIR="${RUNS_DIR:-runs}"
MAX_TOKENS="${MAX_TOKENS:-50M}"
TOKENIZER="${TOKENIZER:-byte}"
TOKENIZER_FILE="${TOKENIZER_FILE:-}"
MICRO_BATCH="${MICRO_BATCH:-8}"
EVAL_MAX_NEW="${EVAL_MAX_NEW:-24}"

COMMON=(
  --max-tokens "$MAX_TOKENS"
  --seq-len 1024
  --batch-tokens 65536
  --micro-batch "$MICRO_BATCH"
  --synthetic-fraction 1.0
  --data fallback
  --lr 3e-3
  --eval-every 100
  --eval-batches 16
  --ckpt-every 250
  --tokenizer "$TOKENIZER"
)
if [[ "$TOKENIZER" == "bpe" ]]; then
  if [[ -z "$TOKENIZER_FILE" ]]; then
    printf '%s\n' 'TOKENIZER=bpe requires TOKENIZER_FILE=/path/to/tokenizer.json' >&2
    exit 2
  fi
  COMMON+=(--tokenizer-file "$TOKENIZER_FILE")
fi

usage() {
  cat <<'EOF'
Usage: bash sprint.sh <command>

Commands:
  check     Check GPU/PyTorch, then run tests.py and test_sprint.py
  smoke     Run 20 steps and resume to 30 for B2, TB, and TC_TAPE
  train     Run or resume the three 50M-token sprint models
  eval      Evaluate the latest B2, TB, and TC_TAPE checkpoints
  b2rand    Run or resume the B2 --rand-loops confound control
  extend    Resume all sprint models to 100M tokens
  all       check -> smoke -> train -> eval

Environment:
  PYTHON=python  RUNS_DIR=runs  MAX_TOKENS=50M  MICRO_BATCH=8
  TOKENIZER=byte  TOKENIZER_FILE=/path/to/tokenizer.json  EVAL_MAX_NEW=24
EOF
}

run_logged() {
  local log="$1"
  shift
  mkdir -p -- "$(dirname -- "$log")"
  "$@" 2>&1 | tee "$log"
}

check() {
  nvidia-smi
  "$PYTHON" -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
  "$PYTHON" tests.py
  "$PYTHON" test_sprint.py
}

smoke() {
  local model dir extra
  for model in B2 TB TC_TAPE; do
    dir="$RUNS_DIR/smoke_${model,,}"
    if [[ -f "$dir/latest.pt" ]]; then
      printf '%s\n' "Refusing to overwrite existing smoke checkpoint: $dir/latest.pt" >&2
      return 1
    fi
    extra=()
    [[ "$model" == "TB" ]] && extra+=(--rand-loops)
    [[ "$model" == "TC_TAPE" ]] && extra+=(--tape-chunk-size 128)
    run_logged "$dir/smoke20.log" "$PYTHON" train.py \
      --model "$model" "${COMMON[@]}" "${extra[@]}" \
      --max-steps 20 --log-every 1 --eval-every 10 --eval-batches 1 \
      --probe-every 1 --ckpt-every 10 --ckpt-dir "$dir"
    run_logged "$dir/smoke30.log" "$PYTHON" train.py \
      --model "$model" "${COMMON[@]}" "${extra[@]}" \
      --max-steps 30 --resume --log-every 1 --eval-every 10 --eval-batches 1 \
      --probe-every 1 --ckpt-every 10 --ckpt-dir "$dir"
    "$PYTHON" - "$dir/train_log.jsonl" "$dir/latest.pt" <<'PY'
import json
import math
import sys
import torch

log_path, ckpt_path = sys.argv[1:]
with open(log_path, encoding="utf-8") as handle:
    records = [json.loads(line) for line in handle if line.strip()]
train = [record for record in records if record.get("type") == "train"]
assert train, "no training telemetry"
assert all(math.isfinite(record["loss"]) for record in train), "non-finite loss"
assert all(math.isfinite(record["grad_scale"]) and record["grad_scale"] > 0 for record in train), "invalid grad scale"
peak = max(record.get("peak_vram_gb", 0.0) for record in train)
assert peak < 12.0, f"peak VRAM {peak:.2f} GB exceeds 12 GB"
state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
assert state["step"] == 30, f"resume stopped at step {state['step']}"
print(f"SMOKE PASS: step=30 peak_vram={peak:.2f}GB min_grad_scale={min(r['grad_scale'] for r in train):.1f}")
PY
  done
}

run_model() {
  local model="$1"
  local dir="$2"
  shift 2
  local resume=()
  if [[ -f "$dir/latest.pt" ]]; then
    resume=(--resume)
  fi
  run_logged "$dir/run.log" "$PYTHON" train.py \
    --model "$model" "${COMMON[@]}" --ckpt-dir "$dir" "${resume[@]}" "$@"
}

train() {
  run_model B2 "$RUNS_DIR/sprint_b2"
  run_model TB "$RUNS_DIR/sprint_tb" --rand-loops
  run_model TC_TAPE "$RUNS_DIR/sprint_tc" --tape-chunk-size 128
}

eval_one() {
  local model="$1"
  local dir="$2"
  local tokenizer_args=()
  if [[ ! -f "$dir/latest.pt" ]]; then
    printf '%s\n' "Missing checkpoint: $dir/latest.pt" >&2
    return 1
  fi
  if [[ -n "$TOKENIZER_FILE" ]]; then
    tokenizer_args=(--tokenizer-file "$TOKENIZER_FILE")
  fi
  run_logged "$dir/eval.log" "$PYTHON" eval_crucible.py \
    --ckpt "$dir/latest.pt" --model "$model" \
    --tokenizer "$TOKENIZER" "${tokenizer_args[@]}" \
    --max-new "$EVAL_MAX_NEW"
}

eval_all() {
  eval_one B2 "$RUNS_DIR/sprint_b2"
  eval_one TB "$RUNS_DIR/sprint_tb"
  eval_one TC_TAPE "$RUNS_DIR/sprint_tc"
  if [[ -f "$RUNS_DIR/sprint_b2_rand/latest.pt" ]]; then
    eval_one B2 "$RUNS_DIR/sprint_b2_rand"
  fi
}

case "${1:-}" in
  check) check ;;
  smoke) smoke ;;
  train) train ;;
  eval) eval_all ;;
  b2rand) run_model B2 "$RUNS_DIR/sprint_b2_rand" --rand-loops ;;
  extend) MAX_TOKENS=100M exec bash "$0" train ;;
  all) check; smoke; train; eval_all ;;
  *) usage; exit 2 ;;
esac
