"""Stage 2 (Direction E): closed-loop amortized latent verifier, best-of-K.

Gate 2A: 3-minute-style Oracle Pass@4 pre-check on identical trajectories.
Gate 2B: collect real success/failure trajectories, train the LatentVerifier,
and compare six FLOP-accounted rows on 160 fixed OOD items.

Usage:
  python eval_stage2_e.py --ckpt runs/stage1_b2_thin --model B2_THIN
  python eval_stage2_e.py --ckpt runs/stage1_f_vcycle --model F_VCYCLE
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_pipeline import ASTGenerator, ByteTokenizer, GraphSatGenerator
from model import RMSNorm, run_stack
from tracks_efg import FVCycle, build_efg, efg_config

GENS = {"graph": GraphSatGenerator, "ast": ASTGenerator}
D = 512
MAX_NEW = 8  # byte-level answers are <= 4 tokens


# --------------------------------------------------------------------------- #
# Deterministic position-consistent perturbation + feature capture
# --------------------------------------------------------------------------- #
def _noise(seed, max_len, dim, device, dtype):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randn((1, max_len, dim), generator=g, device=device,
                       dtype=torch.float32).to(dtype)


def _perturb(h, sigma_rel, seed, max_len):
    if sigma_rel <= 0.0:
        return h
    T = h.shape[1]
    n = _noise(seed, max_len, h.shape[-1], h.device, h.dtype)[:, :T]
    rms = h.detach().float().pow(2).mean(dim=-1, keepdim=True).sqrt()
    return h + (sigma_rel * rms * n.float()).to(h.dtype)


@torch.no_grad()
def forward_features(model, ids, N, sigma_rel=0.0, traj_seed=0,
                     max_len=None, amp_dtype=torch.float16):
    """B2 / F_VCYCLE forward with the perturbation injected at the end of the
    first loop iteration, returning (logits, h_last, x_pre_last).

    Trajectory k=0 uses sigma_rel=0 and is bitwise the clean model.  The noise
    block is drawn once per (seed, max_len) and sliced to the current T, so the
    positions 0..T-1 receive identical perturbations at every decoding step."""
    max_len = max_len or ids.shape[1]
    ctx = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None \
        else torch.autocast("cpu", enabled=False)
    with ctx:
        h = model.embed(ids)
        h = run_stack(model.pre_layers, h)
        x_pre = h
        if isinstance(model, FVCycle):
            d0 = model.d0
            x_c = h[..., :d0]
            h_c = x_c
            for t in range(N):
                h_c = h_c + model.scale_c * model.coarse_core_delta(h_c + x_c)
                if t == 0:
                    h_c = _perturb(h_c, sigma_rel, traj_seed, max_len)
            h = torch.cat([h_c, h[..., d0:]], dim=-1)
            for _ in range(model.N_f):
                inp = h + x_pre
                h = h + model.scale_f * (run_stack(model.core_layers, inp) - inp)
        else:
            scale = model.loop_scale_value()
            for t in range(N):
                inp = h + x_pre
                h = h + scale * model.core_delta(inp)
                if t == 0:
                    h = _perturb(h, sigma_rel, traj_seed, max_len)
        h = run_stack(model.post_layers, h)
        return model.head(h), h[:, -1], x_pre[:, -1]


# --------------------------------------------------------------------------- #
# Batched greedy decoding (exact-length groups: no padding, so causality and
# RoPE positions are untouched)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def decode_groups(model, tok, prompts, N, sigma_rel, traj_seed, device,
                  amp_dtype, max_new=MAX_NEW):
    """Return lists aligned with `prompts`: pred str, correct bool, mean logp,
    features (B, 2D).  Prompts are grouped by exact token length."""
    from collections import defaultdict
    enc = [tok.encode(p["prompt"]) for p in prompts]
    groups = defaultdict(list)
    for i, ids in enumerate(enc):
        groups[len(ids)].append(i)

    preds = [None] * len(prompts)
    logps = [0.0] * len(prompts)
    feats = [None] * len(prompts)
    for _, idxs in groups.items():
        ids = torch.tensor([enc[i] for i in idxs], device=device)
        B = ids.shape[0]
        x = ids
        gen = [[] for _ in range(B)]
        lp = torch.zeros(B, device=device)
        active = torch.ones(B, dtype=torch.bool, device=device)
        max_len = ids.shape[1] + max_new
        for step in range(max_new):
            logits, hl, xp = forward_features(
                model, x, N, sigma_rel, traj_seed, max_len=max_len,
                amp_dtype=amp_dtype)
            if step == 0:
                f = torch.cat([hl.float(), xp.float()], dim=-1)
                for b, i in enumerate(idxs):
                    feats[i] = f[b]
            logp = torch.log_softmax(logits[:, -1].float(), dim=-1)
            nxt = logp.argmax(dim=-1)
            tok_lp = logp.gather(1, nxt.unsqueeze(1)).squeeze(1)
            lp = lp + tok_lp * active
            for b, i in enumerate(idxs):
                if active[b]:
                    t = int(nxt[b])
                    if t == tok.eos_id:
                        active[b] = False
                    else:
                        gen[b].append(t)
            if not bool(active.any()) or step == max_new - 1:
                break
            # finished rows keep decoding harmlessly; their outputs are masked
            x = torch.cat([x, nxt.unsqueeze(1)], dim=1)
        for b, i in enumerate(idxs):
            pred = tok.decode(gen[b]).strip()
            preds[i] = pred
            logps[i] = float(lp[b] / max(1, len(gen[b])))
    correct = [preds[i] == p["answer"].strip() for i, p in enumerate(prompts)]
    return preds, correct, logps, torch.stack(feats)


# --------------------------------------------------------------------------- #
# Verifier
# --------------------------------------------------------------------------- #
class LatentVerifier(nn.Module):
    def __init__(self, d: int = D):
        super().__init__()
        self.norm = RMSNorm(2 * d)
        self.fc1 = nn.Linear(2 * d, d // 2)
        self.fc2 = nn.Linear(d // 2, 1)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        return self.fc2(self.drop(F.silu(self.fc1(self.norm(x))))).squeeze(-1)


def auroc(scores, labels):
    """scores: higher => more likely correct."""
    pairs = sorted(zip(scores, labels), key=lambda t: t[0])
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        rank_sum += avg_rank * sum(1 for k in range(i, j) if pairs[k][1] == 1)
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def make_probes(n_per=80, seed=4242):
    out = []
    for i, (fam, cls) in enumerate(GENS.items()):
        g = cls(random.Random(seed + i), ood=True)
        for _ in range(n_per):
            ex = g.sample()
            out.append(dict(prompt=ex.prompt, answer=ex.answer, fam=fam))
    return out


def make_collect(n_per=750, seed_in=5150, seed_ood=6250):
    out = []
    for i, (fam, cls) in enumerate(GENS.items()):
        g_in = cls(random.Random(seed_in + i))
        g_ood = cls(random.Random(seed_ood + i), ood=True)
        for k in range(n_per):
            ex = (g_in if k % 2 == 0 else g_ood).sample()
            out.append(dict(prompt=ex.prompt, answer=ex.answer, fam=fam))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/stage1_b2_thin")
    ap.add_argument("--model", default="B2_THIN")
    ap.add_argument("--stage1-report", default="runs/stage1_report.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sigmas", default="0.02,0.05,0.10,0.20")
    ap.add_argument("--n-collect", type=int, default=750)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force-gate2a", action="store_true",
                    help="continue past Gate 2A (debug / report-only)")
    args = ap.parse_args()
    out_path = args.out or f"runs/stage2_report_{args.model}.json"
    device, amp = args.device, (torch.float16 if args.device == "cuda" else None)
    tok = ByteTokenizer()

    cfg = efg_config(args.model, tok.vocab_size)
    model = build_efg(args.model, cfg).to(device)
    state = torch.load(os.path.join(args.ckpt, "latest.pt"), map_location="cpu",
                       weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    n_train = cfg.N
    print(f"loaded {args.model} from {args.ckpt} (N_train={n_train})", flush=True)

    report = {"model": args.model, "ckpt": args.ckpt, "n_train": n_train}

    # ---------- base woken-up gate (from Stage 1 fixed OOD set) ---------- #
    base_ok = True
    if os.path.exists(args.stage1_report):
        try:
            s1 = json.load(open(args.stage1_report))
            key = args.model if args.model in s1 else "B2_THIN"
            curve = s1[key]["curve"]
            n_half = str(max(1, n_train // 2))
            n_str = str(n_train)
            acc_half = curve[n_half]["ood_mean"] if n_half in curve else None
            acc_full = curve[n_str]["ood_mean"] if n_str in curve else None
            base_ok = (acc_full is not None and acc_full >= 0.15
                       and (acc_half is None or acc_full > acc_half))
            report["base_gate"] = {"acc_n_half": acc_half, "acc_n": acc_full,
                                   "woken_up": base_ok}
        except Exception as e:
            report["base_gate_error"] = repr(e)
    if not base_ok and not args.force_gate2a:
        report["E"] = "RED (BASE_SLEEPING)"
        json.dump(report, open(out_path, "w"), indent=2)
        print(json.dumps(report, indent=2))
        return

    probes = make_probes()
    print(f"{len(probes)} OOD eval prompts", flush=True)

    # ---------------------------- Gate 2A ------------------------------- #
    t0 = time.time()
    gate2a = {}
    for sigma in [float(s) for s in args.sigmas.split(",")]:
        clean = decode_groups(model, tok, probes, n_train, 0.0, 0, device, amp)
        trues = []
        for k in range(1, 4):
            _, corr, _, _ = decode_groups(model, tok, probes, n_train, sigma,
                                          1000 + k, device, amp)
            trues.append(corr)
        pass1 = sum(clean[1]) / len(probes)
        oracle = sum(1 for i in range(len(probes))
                     if clean[1][i] or any(t[i] for t in trues)) / len(probes)
        gate2a[sigma] = {"pass1": pass1, "oracle_pass4": oracle,
                         "headroom": oracle - pass1}
        print(f"sigma={sigma:.2f} pass1={pass1:.3f} oracle4={oracle:.3f} "
              f"headroom={oracle-pass1:+.3f} ({time.time()-t0:.0f}s)", flush=True)
    report["gate2a"] = gate2a
    best_sigma = max(gate2a, key=lambda s: gate2a[s]["headroom"])
    max_headroom = gate2a[best_sigma]["headroom"]
    report["gate2a_best_sigma"] = best_sigma
    if max_headroom < 0.05 and not args.force_gate2a:
        report["E"] = "RED (NO_DIVERSITY_HEADROOM)"
        json.dump(report, open(out_path, "w"), indent=2)
        print(json.dumps(report, indent=2))
        return
    sigma = best_sigma
    print(f"Gate 2A PASS; sigma*={sigma}", flush=True)

    # ---------------------- Gate 2B: collect ---------------------------- #
    prompts = make_collect(args.n_collect)
    feats, labels, prompt_of, setting_of = [], [], [], []
    t0 = time.time()
    for ti, k in enumerate(range(6)):  # 6 trajectories at N=4
        _, corr, _, f = decode_groups(model, tok, prompts, n_train, sigma,
                                      (0 if k == 0 else 2000 + k), device, amp)
        feats.append(f); labels.append(corr)
        prompt_of += list(range(len(prompts)))
        setting_of += [f"traj{k}"] * len(prompts)
    for N in (2, 4, 8, 16):  # clean depth sweeps
        _, corr, _, f = decode_groups(model, tok, prompts, N, 0.0, 0, device, amp)
        feats.append(f); labels.append(corr)
        prompt_of += list(range(len(prompts)))
        setting_of += [f"depth{N}"] * len(prompts)
        print(f"collected depth N={N} ({time.time()-t0:.0f}s)", flush=True)
    X = torch.cat(feats, dim=0).to(device)
    y = torch.tensor([1.0 if c else 0.0 for lab in labels for c in lab],
                     device=device)
    rep = torch.tensor(prompt_of)
    report["collect"] = {"n_prompts": len(prompts), "n_examples": int(X.shape[0]),
                         "pos_rate": float(y.mean()), "seconds": round(time.time()-t0, 1)}
    print(f"collected {X.shape[0]} examples, pos_rate={float(y.mean()):.3f}",
          flush=True)

    # ---------------------- Verifier training --------------------------- #
    ver = LatentVerifier().to(device)
    opt = torch.optim.AdamW(ver.parameters(), lr=1e-3, weight_decay=1e-2)
    # prompt-level split (no leakage)
    perm = torch.randperm(len(prompts), generator=torch.Generator().manual_seed(7))
    val_prompts = set(perm[: len(prompts) // 10].tolist())
    val_mask = torch.tensor([int(p) in val_prompts for p in rep.tolist()])
    vm = val_mask.to(device)
    Xtr, ytr = X[~vm], y[~vm]
    rep_tr = rep[~val_mask]
    Xva, yva = X[vm], y[vm]

    # ranking pairs from the same prompt (correct vs wrong)
    pairs = []
    by_prompt: dict[int, list[int]] = {}
    for i, p in enumerate(rep_tr.tolist()):
        by_prompt.setdefault(p, []).append(i)
    for p, idxs in by_prompt.items():
        pos_idx = [i for i in idxs if ytr[i] == 1]
        neg_idx = [i for i in idxs if ytr[i] == 0]
        for a in pos_idx:
            for b in neg_idx[:2]:
                pairs.append((a, b))
    pairs = torch.tensor(pairs, dtype=torch.long, device=device) if pairs else \
        torch.zeros((0, 2), dtype=torch.long, device=device)
    print(f"train {int((~val_mask).sum())} / val {int(val_mask.sum())}, "
          f"{len(pairs)} ranking pairs", flush=True)

    for ep in range(args.epochs):
        ver.train()
        idx = torch.randperm(len(Xtr), generator=torch.Generator().manual_seed(100 + ep))
        for s in range(0, len(idx), 512):
            b = idx[s:s + 512]
            e = ver(Xtr[b])
            bce = F.binary_cross_entropy_with_logits(e, 1.0 - ytr[b])
            loss = bce
            pb = pairs
            if len(pb):
                sel = torch.randperm(len(pb))[: min(512, len(pb))]
                a, c = pb[sel, 0], pb[sel, 1]
                ei = ver(Xtr[a])
                ej = ver(Xtr[c])
                loss = loss + F.relu(1.0 - (ej - ei)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    ver.eval()
    with torch.no_grad():
        ev = ver(Xva)
        va_bce = F.binary_cross_entropy_with_logits(ev, 1.0 - yva).item()
        va_auc = auroc((-ev).tolist(), yva.tolist())
    report["verifier"] = {"val_bce": va_bce, "val_auroc": va_auc,
                          "params": sum(p.numel() for p in ver.parameters())}
    print(f"verifier val_bce={va_bce:.4f} val_auroc={va_auc:.3f}", flush=True)

    # ---------------------- Gate 2B: eval ------------------------------- #
    def evaluate():
        # 4 trajectories at N=n_train, sigma*
        traj = []
        for k in range(4):
            _, corr, lps, f = decode_groups(model, tok, probes, n_train,
                                            0.0 if k == 0 else sigma,
                                            0 if k == 0 else 2000 + k,
                                            device, amp)
            with torch.no_grad():
                energy = ver(f).tolist()
            traj.append({"correct": corr, "logp": lps, "energy": energy})
        n = len(probes)
        res = {}
        res["B2_N4_K1"] = sum(traj[0]["correct"]) / n
        e2 = [0] * n
        lc = [0] * n
        oracle = [0] * n
        for i in range(n):
            energies = [traj[k]["energy"][i] for k in range(4)]
            logps = [traj[k]["logp"][i] for k in range(4)]
            e2[i] = traj[min(range(4), key=lambda k: energies[k])]["correct"][i]
            lc[i] = traj[max(range(4), key=lambda k: logps[k])]["correct"][i]
            oracle[i] = any(traj[k]["correct"][i] for k in range(4))
        res["E2_verifier_bo4"] = sum(e2) / n
        res["logit_conf_bo4"] = sum(lc) / n
        res["oracle_pass4"] = sum(oracle) / n
        # N=16 exact inference-FLOP match to N=4,K=4 (clean)
        _, c16, _, _ = decode_groups(model, tok, probes, 16, 0.0, 0, device, amp)
        res["B2_N16_K1"] = sum(c16) / n
        # E1 auto-depth: argmin E over {2,4,8,16,32}
        depth_acc, depth_E = {}, {}
        for N in (2, 4, 8, 16, 32):
            _, cN, _, fN = decode_groups(model, tok, probes, N, 0.0, 0, device, amp)
            with torch.no_grad():
                eN = ver(fN).tolist()
            depth_acc[N] = cN
            depth_E[N] = eN
        chosen = [min(depth_E, key=lambda N: depth_E[N][i]) for i in range(n)]
        res["E1_auto_depth"] = sum(depth_acc[N][i] for i, N in enumerate(chosen)) / n
        res["B2_N32_K1"] = sum(depth_acc[32]) / n
        best_sweep = max(sum(depth_acc[N]) / n for N in depth_acc)
        res["depth_sweep_acc"] = {str(N): sum(depth_acc[N]) / n for N in depth_acc}
        res["overthinking_drop"] = best_sweep - res["B2_N32_K1"]
        res["e1_drop_reduction"] = 1.0 - (best_sweep - res["E1_auto_depth"]) / \
            max(1e-9, res["overthinking_drop"])
        # AUROC on the 4-trajectory pool
        scores_e, scores_l, labs = [], [], []
        for k in range(4):
            scores_e += [-x for x in traj[k]["energy"]]
            scores_l += traj[k]["logp"]
            labs += [1 if c else 0 for c in traj[k]["correct"]]
        res["auroc_verifier"] = auroc(scores_e, labs)
        res["auroc_logit_conf"] = auroc(scores_l, labs)
        res["auto_depth_choices"] = {str(N): chosen.count(N) for N in depth_E}
        return res

    res = evaluate()
    report["gate2b"] = res
    beats_controls = (res["E2_verifier_bo4"] - res["B2_N16_K1"] >= 0.03 and
                      res["E2_verifier_bo4"] - res["logit_conf_bo4"] >= 0.03)
    e1_ok = (res["E1_auto_depth"] > res["B2_N4_K1"] and
             res["e1_drop_reduction"] >= 0.5)
    auc_ok = res["auroc_verifier"] >= 0.70
    report["E"] = "GREEN" if ((beats_controls and auc_ok) or e1_ok) else "RED"
    report["verdict_detail"] = {"beats_both": beats_controls, "auroc_ok": auc_ok,
                                "e1_ok": e1_ok}
    json.dump(report, open(out_path, "w"), indent=2)
    print(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
