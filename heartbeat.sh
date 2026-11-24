#!/bin/bash
# Phase 0 heartbeat: every 300s, POST the latest training status to a private
# ntfy.sh topic so progress can be read from outside the instance without
# opening the WebIDE. Payload = timestamps, last train_log line per run,
# GPU util/mem. No credentials, no code, only step/loss numbers.
# Usage: nohup bash heartbeat.sh >/dev/null 2>&1 &
TOPIC="lv-phase0-b334362f34d5ceb7bf"
cd /mnt/workspace || exit 1
while true; do
  msg="$(date '+%F %T')"
  for d in runs/b1_lr1 runs/b1_lr2 runs/b1_lr3 runs/b2_lr1 runs/b2_lr2 runs/b2_lr3 runs/full_b1 runs/full_b0 runs/full_b2; do
    if [ -s "$d/train_log.jsonl" ]; then
      msg="$msg | $d: $(tail -n 1 "$d/train_log.jsonl" | cut -c1-160)"
    fi
  done
  smi="$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)"
  msg="$msg | GPU: $smi"
  curl -s -m 20 -H "Title: phase0" -d "$msg" "https://ntfy.sh/$TOPIC" >/dev/null 2>&1 || true
  sleep 300
done
