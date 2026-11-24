"""Prepare a local mini FineWeb-Edu shard for Stage 3 (Direction G).

Streams the first N documents of HuggingFaceFW/fineweb-edu (sample-10BT) via
the ModelScope mirror and writes {"text": ...} one per line.  LocalFilesTextCorpus
then enforces the document-level 98/2 train/eval holdout on this shard.
"""
from __future__ import annotations

import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/mnt/systemdisk/fw_mini")
    ap.add_argument("--num-docs", type=int, default=120_000)
    ap.add_argument("--subset", default="sample-10BT")
    ap.add_argument("--min-chars", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "shard0.jsonl")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        with open(out_path) as f:
            have = sum(1 for _ in f)
        if have >= args.num_docs:
            print(f"already have {have} docs at {out_path}; nothing to do")
            return
        print(f"found partial shard with {have} docs; restarting cleanly")
        os.remove(out_path)

    from modelscope.msdatasets import MsDataset

    ds = MsDataset.load("HuggingFaceFW/fineweb-edu", subset_name=args.subset,
                        split="train", use_streaming=True)
    t0 = time.time()
    n = 0
    with open(out_path + ".tmp", "w", encoding="utf-8") as f:
        for row in ds:
            text = row.get("text", "") if isinstance(row, dict) else str(row)
            if len(text) <= args.min_chars:
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n += 1
            if n % 5000 == 0:
                f.flush()
                print(f"{n} docs, {os.path.getsize(out_path + '.tmp')/1e6:.1f} MB, "
                      f"{time.time()-t0:.0f}s", flush=True)
            if n >= args.num_docs:
                break
    os.replace(out_path + ".tmp", out_path)
    print(f"DONE {n} docs -> {out_path} "
          f"({os.path.getsize(out_path)/1e6:.1f} MB, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
