"""Stage 1 (Direction F) evaluation: fixed 160-item OOD crucible + N sweeps.

Loads the three Stage-1 checkpoints, measures exact-match accuracy on
80 graph_ood + 80 ast_ood items (fixed seeds) and 80 in-distribution items,
sweeps the loop count, and reports params / throughput / peak VRAM / syn_nll.

Usage:
  python eval_stage1_f.py --ckpt-root runs
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics

import torch
import torch.nn.functional as F

from data_pipeline import ASTGenerator, ByteTokenizer, GraphSatGenerator, Phase0Data
from model import LoopConfig
from tracks_efg import EFG_MODELS, build_efg, efg_config

SWEEPS = {
    "B2_THIN": [1, 2, 4, 8, 16, 32],
    "B2_NARROW": [4, 7, 14, 28],
    "F_VCYCLE": [2, 5, 10, 20, 40],
}
N_TRAIN = {"B2_THIN": 4, "B2_NARROW": 14, "F_VCYCLE": 10}
CKPT_DIRS = {
    "B2_THIN": "stage1_b2_thin",
    "B2_NARROW": "stage1_b2_narrow",
    "F_VCYCLE": "stage1_f_vcycle",
}
GENS = {"graph": GraphSatGenerator, "ast": ASTGenerator}


def build_probes(n_ood: int = 80, n_in: int = 40):
    """Fixed probes: n_ood OOD + n_in in-distribution per family."""
    out = []
    for i, (fam, cls) in enumerate(GENS.items()):
        g_in = cls(random.Random(777 + i))
        g_ood = cls(random.Random(888 + i), ood=True)
        for _ in range(n_in):
            ex = g_in.sample()
            out.append(dict(prompt=ex.prompt, answer=ex.answer, fam=fam, ood=False))
        for _ in range(n_ood):
            ex = g_ood.sample()
            out.append(dict(prompt=ex.prompt, answer=ex.answer, fam=fam, ood=True))
    return out


@torch.no_grad()
def greedy_decode(model, tok, prompt: str, device, amp_dtype, N=None,
                  max_new: int = 24):
    x = torch.tensor([tok.encode(prompt)], device=device)
    gen: list[int] = []
    ctx = (torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None
           else torch.autocast("cpu", enabled=False))
    for _ in range(max_new):
        with ctx:
            logits = model(x, N=N)
        nxt = int(logits[0, -1].argmax())
        if nxt == tok.eos_id:
            break
        gen.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    return tok.decode(gen).strip()


@torch.no_grad()
def probe_accuracy(model, tok, probes, device, amp_dtype, N, max_new=24):
    stats: dict[str, list[int]] = {}
    for p in probes:
        pred = greedy_decode(model, tok, p["prompt"], device, amp_dtype,
                             N=N, max_new=max_new)
        key = f"{p['fam']}{'_ood' if p['ood'] else ''}"
        c, n = stats.get(key, (0, 0))
        stats[key] = (c + int(pred == p["answer"].strip()), n + 1)
    return {k: c / max(1, n) for k, (c, n) in stats.items()}


@torch.no_grad()
def syn_nll(model, eval_batches, device, amp_dtype):
    tot, cnt = 0.0, 0
    for ids, targets, mask, syn in eval_batches:
        ids = ids.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        syn = syn.to(device)
        with torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else \
                torch.autocast("cpu", enabled=False):
            logits = model(ids)
        losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                 targets.reshape(-1), ignore_index=-100,
                                 reduction="none").reshape(targets.shape)
        m = (mask == 1) & (syn == 1)
        tot += (losses * m).sum().item()
        cnt += int(m.sum())
    return tot / max(1, cnt)


def load_model(name, ckpt_dir, device):
    cfg = efg_config(name, vocabulary_size())
    model = build_efg(name, cfg).to(device)
    state = torch.load(os.path.join(ckpt_dir, "latest.pt"), map_location="cpu",
                       weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def vocabulary_size():
    return ByteTokenizer().vocab_size


def train_metrics(ckpt_dir):
    path = os.path.join(ckpt_dir, "train_log.jsonl")
    toks, vram = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") == "train":
                toks.append(r.get("tok_per_s", 0))
                vram.append(r.get("peak_vram_gb", 0))
            elif r.get("type") == "eval" and r.get("syn_nll") is not None:
                pass
    return (statistics.median(toks[-20:]) if toks else 0.0,
            max(vram) if vram else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default="runs")
    ap.add_argument("--out", default="runs/stage1_report.json")
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--models", default=",".join(EFG_MODELS))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.float16 if device == "cuda" else None
    tok = ByteTokenizer()
    probes = build_probes()
    n_ood = sum(1 for p in probes if p["ood"])
    print(f"probes: {n_ood} OOD + {len(probes)-n_ood} in-dist", flush=True)

    data = Phase0Data(seed=42, tokenizer=tok, text_corpus=None,
                      synthetic_fraction=1.0, seq_len=1024,
                      families=("graph", "ast"))
    batches = [data.next_batch(8) for _ in range(8)]

    report = {}
    for name in args.models.split(","):
        ckpt_dir = os.path.join(args.ckpt_root, CKPT_DIRS[name])
        model = load_model(name, ckpt_dir, device)
        params = sum(p.numel() for p in model.parameters())
        nonemb = sum(p.numel() for n, p in model.named_parameters()
                     if "embed" not in n)
        nll = syn_nll(model, batches, device, amp_dtype)
        tok_s, peak = train_metrics(ckpt_dir)
        n_train = N_TRAIN[name]
        in_dist = probe_accuracy(model, tok,
                                 [p for p in probes if not p["ood"]],
                                 device, amp_dtype, N=n_train,
                                 max_new=args.max_new)
        curve = {}
        for N in SWEEPS[name]:
            acc = probe_accuracy(model, tok,
                                 [p for p in probes if p["ood"]],
                                 device, amp_dtype, N=N,
                                 max_new=args.max_new)
            ood_mean = 0.5 * (acc.get("graph_ood", 0) + acc.get("ast_ood", 0))
            curve[N] = {"graph_ood": acc.get("graph_ood", 0.0),
                        "ast_ood": acc.get("ast_ood", 0.0),
                        "ood_mean": ood_mean}
            print(f"[{name}] N={N:3d} graph_ood={acc.get('graph_ood',0):.3f} "
                  f"ast_ood={acc.get('ast_ood',0):.3f} mean={ood_mean:.3f}",
                  flush=True)
        report[name] = {
            "params": params, "nonemb_params": nonemb,
            "syn_nll": nll, "tok_per_s": tok_s, "peak_vram_gb": peak,
            "in_dist_acc": in_dist,
            "in_dist_mean": 0.5 * (in_dist.get("graph", 0) + in_dist.get("ast", 0)),
            "curve": {str(k): v for k, v in curve.items()},
            "n_train": n_train,
            "ood_at_train": curve[n_train]["ood_mean"],
        }
        del model
        torch.cuda.empty_cache()

    # Verdict against pre-registered gate
    b2t = report["B2_THIN"]["ood_at_train"]
    b2n = report["B2_NARROW"]["ood_at_train"]
    fv = report["F_VCYCLE"]
    f10 = fv["curve"]["10"]["ood_mean"]
    f20 = fv["curve"]["20"]["ood_mean"]
    speed_ok = fv["tok_per_s"] >= 0.90 * report["B2_THIN"]["tok_per_s"]
    beats_both = (f10 - b2t >= 0.03) and (f10 - b2n >= 0.03)
    extend_ok = f20 + 1e-9 >= f10
    report["verdict"] = {
        "F_ood_minus_B2_THIN": f10 - b2t,
        "F_ood_minus_B2_NARROW": f10 - b2n,
        "speed_ratio_vs_B2_THIN": fv["tok_per_s"] / max(1e-9, report["B2_THIN"]["tok_per_s"]),
        "Nc10_to_Nc20_delta": f20 - f10,
        "beats_both_by_3pct": beats_both,
        "extends_ok": extend_ok,
        "speed_ok": speed_ok,
        "F": "GREEN" if (beats_both and extend_ok and speed_ok) else "RED",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["verdict"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
