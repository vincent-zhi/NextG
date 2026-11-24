#!/usr/bin/env python3
"""Toy 2 — primal-dual constraint inversion (v2.1 frozen spec).

One solver, two constraint domains:
  * 6x6 Sudoku (2x3 boxes), N_var=36, M=18 (6 rows + 6 cols + 6 boxes),
    solution-grid-disjoint train/OOD split, OOD = 18-22 blanks and not
    solvable by naked singles alone.
  * unique-solution, non-unit-propagation 3-SAT (n=8..10 train, n=12..14 OOD).

Solver mechanics (v2.1):
  * Hard-STE simplex: p_hard = onehot(argmax p_soft),
    e = sg(p_hard - p_soft) + p_soft
  * Charbonnier rho = sqrt(||v||^2 + eps^2) - eps
  * dual lambda <- clip(beta*lambda + gamma*rho, 0, lambda_max)
  * a SINGLE (1+lambda)*rho gate multiplies the violation value v
    (zero-lock: all rho=0 => gate=0 => dz=0 exactly)
  * E = 0.5 * sum_m ||v_m||^2, monotone term sum_t relu(E_{t+1} - E_t)

Loss: CE + l1*mean_m||v_m||^2 + l2*max(0, delta - max_m||v_m(e^-)||)
      + l3*sum_t relu(E_{t+1}-E_t)

Controls: --mode open (T=0), primal (lambda=0), no_zero_lock (unit direction),
soft (no STE).

Usage:
  python toy2_primal_dual.py --smoke
  python toy2_primal_dual.py --dataset sudoku --mode full --steps 1500
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

# ==========================================================================
# 6x6 Sudoku
# ==========================================================================
ROWS = [[r * 6 + c for c in range(6)] for r in range(6)]
COLS = [[r * 6 + c for r in range(6)] for c in range(6)]
BOX = [(r // 2) * 3 + (c // 3) for r in range(6) for c in range(6)]
BOXES = [[i for i in range(36) if BOX[i] == b] for b in range(6)]
CONSTRAINTS = ROWS + COLS + BOXES  # 18, each with 6 cells


def _cand(g, i):
    used = set()
    for j in ROWS[i // 6]:
        used.add(g[j])
    for j in COLS[i % 6]:
        used.add(g[j])
    for j in BOXES[BOX[i]]:
        used.add(g[j])
    return [v for v in range(1, 7) if v not in used]


def count_solutions(g, limit=2):
    g = list(g)

    def rec():
        best_i, best_c = -1, None
        for i in range(36):
            if g[i] == 0:
                c = _cand(g, i)
                if not c:
                    return 0
                if best_c is None or len(c) < len(best_c):
                    best_i, best_c = i, c
        if best_c is None:
            return 1
        tot = 0
        for v in best_c:
            g[best_i] = v
            tot += rec()
            g[best_i] = 0
            if tot >= limit:
                break
        return tot

    return rec()


def random_solution(rng):
    g = [0] * 36

    def rec():
        best_i, best_c = -1, None
        for i in range(36):
            if g[i] == 0:
                c = _cand(g, i)
                if not c:
                    return False
                if best_c is None or len(c) < len(best_c):
                    best_i, best_c = i, c
        if best_c is None:
            return True
        rng.shuffle(best_c)
        for v in best_c:
            g[best_i] = v
            if rec():
                return True
            g[best_i] = 0
        return False

    assert rec()
    return g


def naked_singles(g):
    g = list(g)
    changed = True
    while changed:
        changed = False
        for i in range(36):
            if g[i] == 0:
                c = _cand(g, i)
                if len(c) == 1:
                    g[i] = c[0]
                    changed = True
    return all(v != 0 for v in g)


def make_puzzle(sol, target_blanks, rng):
    g = list(sol)
    order = list(range(36))
    rng.shuffle(order)
    blanks = 0
    for i in order:
        if blanks >= target_blanks:
            break
        keep = g[i]
        g[i] = 0
        if count_solutions(g, 2) != 1:
            g[i] = keep
        else:
            blanks += 1
    return g, blanks


def build_sudoku_pool(n_sol, per_sol, blanks_range, rng, nontrivial):
    sols, seen = [], set()
    while len(sols) < n_sol:
        s = random_solution(rng)
        key = tuple(s)
        if key in seen:
            continue
        seen.add(key)
        sols.append(s)
    out = []
    for idx, s in enumerate(sols):
        for _ in range(per_sol):
            got = None
            for _try in range(30):
                target = rng.randint(*blanks_range)
                g, bl = make_puzzle(s, target, rng)
                if bl < blanks_range[0]:
                    continue
                if nontrivial and naked_singles(g):
                    continue
                got = (g, s, bl)
                break
            if got is None:  # fall back to any unique puzzle
                g, bl = make_puzzle(s, blanks_range[0], rng)
                got = (g, s, bl)
            out.append((got, idx))
    return out


# ==========================================================================
# unique, non-UP 3-SAT
# ==========================================================================
def _lit_masks(n):
    masks = {}
    for v in range(n):
        for val in (0, 1):
            m = 0
            for a in range(1 << n):
                if ((a >> v) & 1) == val:
                    m |= 1 << a
            masks[(v, val)] = m
    return masks


def _unit_prop(clauses, n):
    assign = [None] * n
    changed = True
    while changed:
        changed = False
        for cl in clauses:
            sat = False
            un = []
            for v, s in cl:
                if assign[v] is not None and assign[v] == (1 if s > 0 else 0):
                    sat = True
                    break
                if assign[v] is None:
                    un.append((v, s))
            if sat:
                continue
            if not un:
                return False, assign
            if len(un) == 1:
                v, s = un[0]
                val = 1 if s > 0 else 0
                if assign[v] is not None and assign[v] != val:
                    return False, assign
                if assign[v] is None:
                    assign[v] = val
                    changed = True
    return all(a is not None for a in assign), assign


def gen_sat(n, rng, ratio=4.2, non_up=True, max_tries=5000):
    m = max(2, int(round(ratio * n)))
    lit = _lit_masks(n)
    full = (1 << (1 << n)) - 1
    for _ in range(max_tries):
        clauses, mask = [], full
        for _ in range(m):
            vs = rng.sample(range(n), 3)
            ss = [rng.choice((1, -1)) for _ in range(3)]
            clauses.append(list(zip(vs, ss)))
            cm = 0
            for v, s in zip(vs, ss):
                cm |= lit[(v, 1 if s > 0 else 0)]
            mask &= cm
            if mask == 0:
                break
        if mask.bit_count() != 1:
            continue
        if non_up:
            done, _ = _unit_prop(clauses, n)
            if done:
                continue
        sol = mask.bit_length() - 1
        return clauses, [(sol >> v) & 1 for v in range(n)]
    return None


def build_sat_pool(n, count, rng, non_up=True):
    out = []
    while len(out) < count:
        r = gen_sat(n, rng, non_up=non_up)
        if r is not None:
            out.append(r)
    return out


# ==========================================================================
# encoder: per-cell head with a shared global context (n-independent)
# ==========================================================================
class Encoder(nn.Module):
    def __init__(self, n_cells_max, n_vals, d_emb=24, d_glob=96, hidden=128):
        super().__init__()
        self.n_max, self.n_vals = n_cells_max, n_vals
        self.cell_emb = nn.Embedding(n_cells_max, d_emb)
        self.glob = nn.Sequential(nn.Linear(n_cells_max * n_vals, d_glob), nn.GELU(),
                                  nn.Linear(d_glob, d_glob))
        self.head = nn.Sequential(nn.Linear(d_emb + d_glob + n_vals, hidden), nn.GELU(),
                                  nn.Linear(hidden, n_vals))

    def forward(self, x):
        B, n, _ = x.shape
        assert n <= self.n_max
        xp = F.pad(x, (0, 0, 0, self.n_max - n))
        g = self.glob(xp.reshape(B, -1))
        ci = self.cell_emb.weight[:n].unsqueeze(0).expand(B, n, -1)
        return self.head(torch.cat([ci, g.unsqueeze(1).expand(B, n, -1), x], dim=-1))


# ==========================================================================
# task containers (all tensors already on device)
# ==========================================================================
class SudokuTask:
    n_cells, n_cells_max, n_vals, M = 36, 36, 6, 18

    def __init__(self, puzzles, solutions, device):
        p = torch.tensor(puzzles, dtype=torch.long, device=device)
        self.sol = torch.tensor(solutions, dtype=torch.long, device=device)
        self.clue_mask = p > 0
        self.clue_vals = (p - 1).clamp(min=0)
        oh = F.one_hot(self.clue_vals, 6).float() * self.clue_mask[..., None].float()
        self.x = oh
        self.sup_mask = ~self.clue_mask

    def clamp_e(self, e):
        oh = F.one_hot(self.clue_vals, 6).float()
        return torch.where(self.clue_mask[..., None], oh, e)

    def clamp_z(self, z):
        oh = F.one_hot(self.clue_vals, 6).float()
        return torch.where(self.clue_mask[..., None], (2 * oh - 1) * 30.0, z)

    def violation(self, e):
        return torch.stack([e[:, idx].sum(1) - 1.0 for idx in CONSTRAINTS], dim=1)  # (B,18,6)


class SATTask:
    n_vals = 2

    def __init__(self, instances, device):
        cl, sol = zip(*instances)
        self.n = len(sol[0])
        self.M = len(cl[0])
        self.n_cells, self.n_cells_max = self.n, 14
        cv = torch.tensor([[[v for v, s in c] for c in inst] for inst in cl], device=device)
        cs = torch.tensor([[[s for v, s in c] for c in inst] for inst in cl], device=device)
        self.cv, self.cs = cv, cs
        self.sol = torch.tensor(sol, device=device)
        B = cv.shape[0]
        x = torch.zeros(B, self.n, 2, device=device)
        pos = (cs + 1) // 2
        for b in range(B):
            for l in range(3):
                x[b].scatter_add_(0, cv[b, :, l].unsqueeze(-1),
                                  F.one_hot(pos[b, :, l], 2).float())
        self.x = x
        self.sup_mask = torch.ones(B, self.n, dtype=torch.bool, device=device)

    def clamp_e(self, e):
        return e

    def clamp_z(self, z):
        return z

    def violation(self, e):
        B, n, _ = e.shape
        ev = e.gather(1, self.cv.reshape(B, -1, 1).expand(B, -1, 1)).reshape(B, self.M, 3, 2)
        pos = ((self.cs + 1) // 2).unsqueeze(-1)
        sat = ev.gather(-1, pos).squeeze(-1)
        return (1.0 - sat).prod(-1, keepdim=True)  # (B,M,1)


# ==========================================================================
# solver
# ==========================================================================
def charbonnier(nv, eps):
    return torch.sqrt(nv * nv + eps * eps) - eps


def ste_hard(p):
    idx = p.argmax(-1, keepdim=True)
    hard = torch.zeros_like(p).scatter_(-1, idx, 1.0)
    return hard + (p - p.detach())


def solve(z, task, T, eta, beta, gamma, lam_max, mode, eps):
    B, n, _ = z.shape
    lam = torch.zeros(B, task.M, device=z.device)
    Es = []
    for _ in range(T):
        p = torch.softmax(z, dim=-1)
        e = p if mode == "soft" else ste_hard(p)
        e = task.clamp_e(e)
        v = task.violation(e)
        nv = v.norm(dim=-1)
        rho = charbonnier(nv, eps)
        lam = torch.clamp(beta * lam + gamma * rho, 0.0, lam_max)
        with torch.enable_grad():
            ed = e.detach().requires_grad_(True)
            vd = task.violation(ed)
            nvd = vd.norm(dim=-1)
            if mode == "no_zero_lock":
                obj = ((1.0 + lam.detach()) * nvd).sum()
            else:
                gate = ((1.0 + lam.detach()) * rho.detach()).unsqueeze(-1)
                obj = 0.5 * (gate * (vd * vd).sum(-1)).sum()
            g = torch.autograd.grad(obj, ed)[0]
        z = task.clamp_z(z - eta * g.detach())
        Es.append(0.5 * (nv * nv).sum(-1))
    return z, Es


# ==========================================================================
# model + loss
# ==========================================================================
class Model(nn.Module):
    def __init__(self, task):
        super().__init__()
        self.enc = Encoder(task.n_cells_max, task.n_vals)

    def forward(self, task, T, args):
        z0 = self.enc(task.x)
        if T <= 0:
            return z0, [], z0
        z, Es = solve(z0, task, T, args.eta, args.beta, args.gamma,
                      args.lam_max, args.mode, args.eps)
        return z, Es, z0


def corruption(task, e):
    """Flip one random supervised cell; e^- is a plausible local alternative."""
    B, n, _ = e.shape
    idx = []
    for b in range(B):
        sup = task.sup_mask[b].nonzero().flatten()
        idx.append(int(sup[torch.randint(len(sup), (1,)).item()]))
    idx = torch.tensor(idx, device=e.device)
    alt = e.clone()
    alt[torch.arange(B, device=e.device), idx] = 0.0
    alt[torch.arange(B, device=e.device), idx, torch.randint(0, e.shape[-1], (B,), device=e.device)] = 1.0
    return alt


def loss_fn(model, task, T, args):
    z, Es, _ = model(task, T, args)
    p = torch.softmax(z, dim=-1)
    sup = task.sup_mask
    ce = F.cross_entropy(z[sup], task.sol[sup] - (1 if isinstance(task, SudokuTask) else 0))
    e = task.clamp_e(ste_hard(p))
    v = task.violation(e)
    l1 = 0.5 * (v * v).sum(-1).mean()
    vb = task.violation(corruption(task, e))
    l2 = torch.relu(args.delta - vb.norm(dim=-1).max(-1).values).mean() if vb.numel() else z.new_zeros(())
    if len(Es) > 1:
        E = torch.stack(Es, dim=-1)
        l3 = torch.relu(E[..., 1:] - E[..., :-1]).sum(-1).mean()
    else:
        l3 = z.new_zeros(())
    return ce + args.l1 * l1 + args.l2 * l2 + args.l3 * l3, ce.item(), l1.item(), l3.item()


@torch.no_grad()
def evaluate(model, task, T, args):
    z, Es, _ = model(task, T, args)
    p = torch.softmax(z, dim=-1)
    pred = z.argmax(-1) + (1 if isinstance(task, SudokuTask) else 0)
    sup = task.sup_mask
    acc_cells = ((pred == task.sol) & sup).sum().item() / sup.sum().item()
    acc_full = (pred == task.sol).all(-1).float().mean().item()
    e = task.clamp_e(ste_hard(p))
    v_hard = task.violation(e)
    nv_hard = v_hard.norm(dim=-1)
    label = (nv_hard < 1e-6).flatten().cpu()
    score = (-charbonnier(nv_hard, args.eps)).flatten().cpu()
    e_soft = task.clamp_e(p)
    nv_soft = task.violation(e_soft).norm(dim=-1)
    score_soft = (-nv_soft).flatten().cpu()
    return dict(acc_cells=acc_cells, acc_full=acc_full,
                auroc_hard_score=auroc(score_soft, label),
                viol_rate=1.0 - label.float().mean().item())


def auroc(scores, labels):
    labels = labels.bool()
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = torch.cat([pos, neg])
    order = torch.argsort(allv, stable=True)
    ranks = torch.empty(len(allv))
    ranks[order] = torch.arange(1, len(allv) + 1, dtype=torch.float)
    # average ties
    uniq = torch.unique(allv)
    if len(uniq) < len(allv):
        for u in uniq:
            m = allv == u
            if m.sum() > 1:
                ranks[m] = ranks[m].mean()
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ==========================================================================
# datasets + training
# ==========================================================================
def make_tasks(args, rng, device):
    if args.dataset == "sudoku":
        train = build_sudoku_pool(args.n_sol, args.per_sol, (10, 14), rng, False)
        test = build_sudoku_pool(max(20, args.n_sol // 8), 2, (18, 22), rng, True)
        tr = [x[0] for x, idx in train if idx % 5 != 0]
        te = [x[0] for x, idx in test] + [x[0] for x, idx in train if idx % 5 == 0]
        tr_task = SudokuTask([p for p, s, b in tr], [s for p, s, b in tr], device)
        te_task = SudokuTask([p for p, s, b in te], [s for p, s, b in te], device)
        return tr_task, te_task
    n_tr = args.sat_n_train
    train = build_sat_pool(n_tr, args.n_sat, rng, non_up=True)
    test = build_sat_pool(args.sat_n_ood, max(20, args.n_sat // 8), rng, non_up=True)
    return SATTask(train, device), SATTask(test, device)


def train(args):
    device = torch.device(args.device)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()
    tr, te = make_tasks(args, rng, device)
    print(f"data: train={tr.x.shape[0]} test={te.x.shape[0]} "
          f"cells={tr.n_cells} vals={tr.n_vals} M={tr.M} ({time.time()-t0:.1f}s)", flush=True)
    model = Model(tr).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    for step in range(1, args.steps + 1):
        loss, ce, l1, l3 = loss_fn(model, tr, args.T_train, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0 or step == 1:
            print(f"[{args.dataset}/{args.mode}] step {step:5d} loss={loss.item():.4f} "
                  f"ce={ce:.4f} l1={l1:.3f} l3={l3:.4f} |g|={gn.item():.2f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
    res = {"dataset": args.dataset, "mode": args.mode, "steps": args.steps,
           "T_train": args.T_train, "eta": args.eta,
           "n_train": int(tr.x.shape[0]), "n_test": int(te.x.shape[0]), "eval": {}}
    for T in (8, 16, 32, 64):
        r = evaluate(model, te, T, args)
        res["eval"][f"T{T}"] = r
        print(f"  T={T:2d} acc_cells={r['acc_cells']*100:6.2f}% "
              f"acc_full={r['acc_full']*100:5.2f}% auroc={r['auroc_hard_score']} "
              f"viol={r['viol_rate']*100:.2f}%", flush=True)
    a8 = res["eval"]["T8"]["acc_cells"]
    a64 = res["eval"]["T64"]["acc_cells"]
    res["retention_8_to_64"] = (a64 / a8) if a8 > 0 else None
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"retention acc(T64)/acc(T8) = {res['retention_8_to_64']}  -> {args.out}")


# ==========================================================================
# smoke
# ==========================================================================
def smoke():
    rng = random.Random(0)
    # Sudoku: uniqueness-preserving removal, solution recovered
    sol = random_solution(rng)
    assert count_solutions(sol, 2) == 1
    g, bl = make_puzzle(sol, 20, rng)
    assert count_solutions(g, 2) == 1, "puzzle not unique"
    assert all(g[i] == 0 or g[i] == sol[i] for i in range(36))
    print(f"sudoku: unique puzzle with {bl} blanks")

    # 3-SAT: unique + non-UP
    r = gen_sat(8, rng)
    assert r is not None
    cl, solv = r
    done, _ = _unit_prop(cl, 8)
    assert not done, "UP should not solve the instance"
    print(f"sat: n=8 m={len(cl)} unique non-UP instance")

    # zero-lock: at a satisfied hard assignment the step is exactly zero
    task = SudokuTask([sol], [sol], torch.device("cpu"))
    z = torch.full((1, 36, 6), -30.0)
    z.scatter_(-1, torch.tensor(sol).view(1, 36, 1), 30.0)
    z = task.clamp_z(z)
    z2, Es = solve(z, task, 3, 0.3, 0.9, 0.5, 5.0, "full", 1e-3)
    assert (z2 - z).abs().max() < 1e-6, (z2 - z).abs().max()
    assert max(E.item() for E in Es) < 1e-6
    print("zero-lock: satisfied assignment => dz = 0 exactly")

    # a wrong assignment produces a nonzero step
    zw = torch.zeros(1, 36, 6)
    zw2, Es2 = solve(zw, task, 1, 0.3, 0.9, 0.5, 5.0, "full", 1e-3)
    assert (zw2 - zw).abs().max() > 1e-3
    print("nonzero step on an unsatisfied assignment")

    # forward + loss on both datasets
    args = argparse.Namespace(mode="full", eta=0.3, beta=0.9, gamma=0.5,
                              lam_max=5.0, eps=1e-3, delta=0.5,
                              l1=0.1, l2=0.01, l3=0.01, T_train=4)
    for ds in ("sudoku", "sat"):
        if ds == "sudoku":
            pool = build_sudoku_pool(20, 2, (18, 22), rng, True)[:24]
            t = SudokuTask([p for (p, s, b), _ in pool], [s for (p, s, b), _ in pool],
                           torch.device("cpu"))
        else:
            pool = build_sat_pool(8, 16, rng)
            t = SATTask(pool, torch.device("cpu"))
        m = Model(t)
        loss, ce, l1v, l3v = loss_fn(m, t, 4, args)
        assert torch.isfinite(loss)
        r = evaluate(m, t, 8, args)
        print(f"forward ok: {ds} loss={loss.item():.3f} ce={ce:.3f} "
              f"acc={r['acc_cells']*100:.1f}% auroc={r['auroc_hard_score']}")
    print("smoke: all checks passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sudoku", choices=["sudoku", "sat"])
    ap.add_argument("--mode", default="full",
                    choices=["full", "open", "primal", "no_zero_lock", "soft"])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--T-train", type=int, default=8)
    ap.add_argument("--eta", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--lam-max", type=float, default=5.0)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--l1", type=float, default=0.1)
    ap.add_argument("--l2", type=float, default=0.01)
    ap.add_argument("--l3", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-sol", type=int, default=200)
    ap.add_argument("--per-sol", type=int, default=2)
    ap.add_argument("--n-sat", type=int, default=200)
    ap.add_argument("--sat-n-train", type=int, default=8)
    ap.add_argument("--sat-n-ood", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="toys/runs_toy2/sudoku_full.json")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    else:
        train(args)


if __name__ == "__main__":
    main()
