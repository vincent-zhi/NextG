"""Stage 3 (Direction G) measurement: exact non-emb params, static
weight+optimizer VRAM, peak VRAM, tok/s, and the held-out fw_bpb curve.

Static VRAM is measured by building each model, running exactly ONE optimizer
step (so AdamW moments exist), and summing all parameter, gradient and
optimizer-state bytes -- no training tokens are consumed.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

import torch

from data_pipeline import ByteTokenizer
from tracks_efg import build_stage3, stage3_config


def static_vram_mb(name: str, vocab: int = 258) -> tuple[float, float, float]:
    """Return (non-emb params, weight+optimizer MB, +grad MB)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = stage3_config(name, vocab)
    model = build_stage3(name, cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
    x = torch.randint(0, vocab, (1, 32), device=dev)
    loss = model(x).float().sum()
    loss.backward()
    opt.step()
    wopt = 0
    for p in model.parameters():
        wopt += p.numel() * p.element_size()
    grads = sum(p.grad.numel() * p.grad.element_size()
                for p in model.parameters() if p.grad is not None)
    nbytes = 0
    for st in opt.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                nbytes += v.numel() * v.element_size()
    nonemb = sum(p.numel() for n, p in model.named_parameters() if "embed" not in n)
    del model, opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return nonemb, (wopt + nbytes) / 2**20, (wopt + nbytes + grads) / 2**20


def log_metrics(ckpt_dir: str) -> dict:
    path = os.path.join(ckpt_dir, "train_log.jsonl")
    toks, vram, bpb = [], [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") == "train":
                toks.append(r.get("tok_per_s", 0))
                vram.append(r.get("peak_vram_gb", 0))
            elif r.get("type") == "eval" and r.get("fw_tokens", 0) > 0:
                bpb.append({"step": r["step"], "fw_bpb": r["fw_bpb"],
                            "fw_nll": r["fw_nll"]})
    return {"tok_per_s": statistics.median(toks[-20:]) if toks else 0.0,
            "peak_vram_gb": max(vram) if vram else 0.0,
            "bpb_curve": bpb,
            "final_fw_bpb": bpb[-1]["fw_bpb"] if bpb else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default="runs")
    ap.add_argument("--out", default="runs/stage3_report.json")
    args = ap.parse_args()
    names = ("DENSE_NARROW", "G_WHT")
    dirs = {"DENSE_NARROW": "stage3_dense_narrow", "G_WHT": "stage3_g_wht"}
    report = {}
    for name in names:
        nonemb, static, static_g = static_vram_mb(name)
        m = log_metrics(os.path.join(args.ckpt_root, dirs[name]))
        report[name] = {"nonemb_params": nonemb, "static_vram_mb": static,
                        "static_vram_incl_grad_mb": static_g, **m}
        print(f"{name}: nonemb={nonemb/1e6:.3f}M static={static:.1f}MB "
              f"peak={m['peak_vram_gb']:.2f}GB tok/s={m['tok_per_s']:.0f} "
              f"final_fw_bpb={m['final_fw_bpb']}", flush=True)
    # B0_FULL static VRAM is analytic + measured without training
    b0_nonemb, b0_static, b0_static_g = static_vram_mb("B0_FULL")
    report["B0_FULL"] = {"nonemb_params": b0_nonemb,
                         "static_vram_mb": b0_static,
                         "static_vram_incl_grad_mb": b0_static_g}
    b0_log = os.path.join(args.ckpt_root, "stage3_b0_full", "train_log.jsonl")
    if os.path.exists(b0_log):
        report["B0_FULL"].update(log_metrics(os.path.dirname(b0_log)))
    d, g = report["DENSE_NARROW"], report["G_WHT"]
    ratio = b0_static / g["static_vram_mb"]
    d_bpb = d["final_fw_bpb"]
    g_bpb = g["final_fw_bpb"]
    verdict = {
        "static_vram_ratio_vs_B0": ratio,
        "g_minus_dense_bpb": (g_bpb - d_bpb) if (g_bpb is not None and d_bpb is not None) else None,
        "tok_ratio_vs_dense": g["tok_per_s"] / max(1e-9, d["tok_per_s"]),
        "vram_ok": ratio >= 3.5,
        "bpb_ok": (g_bpb is not None and d_bpb is not None and d_bpb - g_bpb >= 0.02),
        "speed_ok": None,  # filled below if B0_FULL was trained
        "run_b0": bool(g_bpb is not None and d_bpb is not None and g_bpb < d_bpb),
    }
    b0 = report["B0_FULL"]
    if b0.get("tok_per_s"):
        verdict["speed_ok"] = (g["tok_per_s"] >= 0.85 * b0["tok_per_s"] or
                               g["tok_per_s"] >= 0.70 * d["tok_per_s"])
        verdict["tok_ratio_vs_b0"] = g["tok_per_s"] / max(1e-9, b0["tok_per_s"])
        verdict["g_minus_b0_bpb"] = (g_bpb - b0["final_fw_bpb"]
                                     if b0.get("final_fw_bpb") is not None else None)
    else:
        verdict["speed_ok"] = g["tok_per_s"] >= 0.70 * d["tok_per_s"]
    verdict["G"] = ("GREEN" if (verdict["vram_ok"] and verdict["bpb_ok"]
                                and verdict["speed_ok"]) else "RED")
    report["verdict"] = verdict
    json.dump(report, open(args.out, "w"), indent=2)
    print(json.dumps(verdict, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
