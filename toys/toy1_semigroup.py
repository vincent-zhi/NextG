#!/usr/bin/env python3
"""Toy 1 — ST-Tropical semigroup doubling with an SO(3)^B payload.

v2.1 frozen spec:
  * scalar non-negative D, D_ii = 0, non-edges = D_MAX
  * V in SO(3)^B (B=2) via a unit-quaternion encoder (no exp singularity)
  * full-m soft gradient with a hard forward (straight-through) for D and V
  * train: H in [1,4], K in [6,12], T=2 (4 hops), full 2-step BPTT
  * test : H in {8,16,32}, K in {20,48}, T in {3,4,5}
  * controls: Vector-Loop, Edge-1Hop, MLP-Payload, SoftMin-LSE

The graph is a layered DAG with a unique backbone s->t.  Backbone edges all
carry cost token 0 (0.5); every off-backbone edge carries token >= 4 (> 2.0).
Any non-backbone s->t path has H edges, at least one of which is off-backbone,
so its cost is > H * 0.5 = backbone cost.  The backbone is therefore the unique
shortest path for every H.  The op tokens on the backbone are random elements
of S_4 (the 24 cube rotations); the target is their ordered product.

Usage:
  python toy1_semigroup.py --smoke
  python toy1_semigroup.py --model st --steps 3000 --out runs_toy1/st.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# S_4 as the 24 rotations of the cube (all det = +1, faithful)
# --------------------------------------------------------------------------
def cube_rotations() -> torch.Tensor:
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            M = torch.zeros(3, 3)
            for i, (p, s) in enumerate(zip(perm, signs)):
                M[i, p] = s
            if torch.det(M).item() > 0.5:
                mats.append(M)
    R = torch.stack(mats)
    assert R.shape == (24, 3, 3), R.shape
    return R


def mult_table(R: torch.Tensor) -> torch.Tensor:
    n = R.shape[0]
    table = torch.zeros(n, n, dtype=torch.long)
    for a in range(n):
        for b in range(n):
            P = R[a] @ R[b]
            d = (P.unsqueeze(0) - R).abs().sum((-1, -2))
            table[a, b] = int(d.argmin())
    for a in range(n):
        for b in range(n):
            assert (R[table[a, b]] - R[a] @ R[b]).abs().max().item() < 1e-5
    return table


def identity_index(R: torch.Tensor) -> int:
    return int((R - torch.eye(3)).abs().sum((-1, -2)).argmin())


def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """q: (..., 4) -> (..., 3, 3). Differentiable, smooth everywhere on S^3."""
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q.unbind(-1)
    row0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1)
    row1 = torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1)
    row2 = torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)
    return torch.stack([row0, row1, row2], dim=-2)


# cost token -> true edge cost.  Token 0 is the backbone; tokens 4..7 are the
# off-backbone distractors and are strictly more expensive than any backbone.
COST_VALUES = (0.5, 0.8, 1.1, 1.5, 2.0, 2.6, 3.3, 4.0)
N_COST = len(COST_VALUES)
N_OP = 24
D_MAX = 1.0e4


# --------------------------------------------------------------------------
# graph generator
# --------------------------------------------------------------------------
def make_batch(Bk: int, K: int, H: int, rng: random.Random,
               R: torch.Tensor, table: torch.Tensor, ident: int,
               device: torch.device):
    """A batch of layered DAGs with a unique backbone 0 -> 1 -> ... -> H."""
    assert K >= H + 1
    cost = torch.zeros(Bk, K, K, dtype=torch.long, device=device)
    op = torch.zeros(Bk, K, K, 2, dtype=torch.long, device=device)
    edge = torch.zeros(Bk, K, K, dtype=torch.bool, device=device)
    true_cost = torch.zeros(Bk, K, K, device=device)
    true_dist = torch.zeros(Bk, device=device)
    target = torch.zeros(Bk, 2, dtype=torch.long, device=device)

    for b in range(Bk):
        layers = [[i] for i in range(H + 1)]
        for e in range(H + 1, K):
            layers[rng.randrange(H)].append(e)

        # backbone: token 0, random op element per block
        for l in range(H):
            i, j = l, l + 1
            edge[b, i, j] = True
            cost[b, i, j] = 0
            true_cost[b, i, j] = COST_VALUES[0]
            op[b, i, j, 0] = rng.randrange(N_OP)
            op[b, i, j, 1] = rng.randrange(N_OP)

        # distractors: token 4..7, never cheaper than the backbone
        for l in range(H):
            for i in layers[l]:
                for _ in range(rng.randint(1, 3)):
                    j = rng.choice(layers[l + 1])
                    if i == j or bool(edge[b, i, j]):
                        continue
                    edge[b, i, j] = True
                    t = rng.randrange(4, N_COST)
                    cost[b, i, j] = t
                    true_cost[b, i, j] = COST_VALUES[t]
                    op[b, i, j, 0] = rng.randrange(N_OP)
                    op[b, i, j, 1] = rng.randrange(N_OP)

        d = 0.0
        for l in range(H):
            d += COST_VALUES[0]
        true_dist[b] = d

        for blk in range(2):
            idx = ident
            for l in range(H):
                g = int(op[b, l, l + 1, blk])
                idx = int(table[idx, g])
            target[b, blk] = idx

    diag = torch.eye(K, dtype=torch.bool, device=device).unsqueeze(0).expand(Bk, K, K).clone()
    s = torch.zeros(Bk, dtype=torch.long, device=device)
    t = torch.full((Bk,), H, dtype=torch.long, device=device)
    return dict(cost=cost, op=op, edge=edge, diag=diag, true_cost=true_cost,
                true_dist=true_dist, target=target, s=s, t=t, H=H, K=K)


# --------------------------------------------------------------------------
# ST-Tropical model (variants: st / edge1hop / softmin)
# --------------------------------------------------------------------------
class STModel(nn.Module):
    def __init__(self, B=2, d_emb=32, tau=0.05, eps_self=1e-5,
                 d_max=D_MAX, variant="st"):
        super().__init__()
        self.B = B
        self.tau = tau
        self.eps_self = eps_self
        self.d_max = d_max
        self.variant = variant
        self.cost_emb = nn.Embedding(N_COST, d_emb)
        self.cost_head = nn.Linear(d_emb, 1)
        self.op_emb = nn.Embedding(N_OP, d_emb)
        self.op_head = nn.Linear(d_emb, 4)
        self.cls_head = nn.Sequential(nn.Linear(9, 64), nn.GELU(), nn.Linear(64, N_OP))

    # -- encoder: tokens -> D0, V0 -----------------------------------------
    def encode(self, cost_tok, op_tok, edge, diag):
        Bk, K, _ = cost_tok.shape
        d_edge = F.softplus(self.cost_head(self.cost_emb(cost_tok)).squeeze(-1))
        D0 = torch.where(diag, torch.zeros_like(d_edge),
                         torch.where(edge, d_edge, torch.full_like(d_edge, self.d_max)))
        q = self.op_head(self.op_emb(op_tok)).view(Bk, K, K, self.B, 4)
        V0e = quat_to_rot(q)
        I = torch.eye(3, device=cost_tok.device).view(1, 1, 1, 1, 3, 3)
        V0 = torch.where(diag[..., None, None, None], I.expand_as(V0e),
                         torch.where(edge[..., None, None, None], V0e,
                                     torch.zeros_like(V0e)))
        return D0, V0

    # -- one tropical doubling ---------------------------------------------
    def _square(self, D, V, D0=None, V0=None):
        Bk, K, _ = D.shape
        Dr = D0 if self.variant == "edge1hop" else D
        Vr = V0 if self.variant == "edge1hop" else V

        # S[b,i,k,j] = D[b,i,j] + Dr[b,j,k]
        S = D.unsqueeze(2) + Dr.transpose(1, 2).unsqueeze(1)

        eye = torch.eye(K, device=D.device)
        S_adj = S - self.eps_self * eye.view(1, 1, K, K)
        jstar = S_adj.argmin(dim=-1)                    # (Bk,K,K)
        D_hard = S.min(dim=-1).values                   # (Bk,K,K)

        b_idx = torch.arange(Bk, device=D.device).view(Bk, 1, 1, 1).expand(Bk, K, K, 1)
        i_idx = torch.arange(K, device=D.device).view(1, K, 1, 1).expand(Bk, K, K, 1)
        j_idx = jstar.unsqueeze(-1)
        k_idx = torch.arange(K, device=D.device).view(1, 1, K, 1).expand(Bk, K, K, 1)
        V_left = V[b_idx, i_idx, j_idx].squeeze(3)      # (Bk,K,K,B,3,3)
        V_right = Vr[b_idx, j_idx, k_idx].squeeze(3)    # (Bk,K,K,B,3,3)
        V_hard = V_left @ V_right

        alpha = torch.softmax(-S / self.tau, dim=-1)    # (Bk,K,K,K)
        V_soft = torch.einsum("bikj,bijmpq,bjkmqr->bikmpr", alpha, V, Vr)
        logsum = torch.logsumexp(-(S - D_hard.unsqueeze(-1)) / self.tau, dim=-1)
        D_soft = D_hard - self.tau * logsum

        if self.variant == "softmin":
            D_new = D_soft
        else:
            D_new = D_soft + (D_hard - D_soft).detach()
        V_new = V_soft + (V_hard - V_soft).detach()
        return D_new, V_new

    # -- full forward -------------------------------------------------------
    def forward(self, cost_tok, op_tok, edge, diag, s, t, T):
        D0, V0 = self.encode(cost_tok, op_tok, edge, diag)
        D, V = D0, V0
        I = torch.eye(3, device=D.device).view(1, 1, 1, 1, 3, 3)
        for _ in range(T):
            D, V = self._square(D, V, D0, V0)
            D = D.masked_fill(diag, 0.0)
            V = torch.where(diag[..., None, None, None], I.expand_as(V), V)
        Bk = D.shape[0]
        b = torch.arange(Bk, device=D.device)
        Vs = V[b, s, t]                                  # (Bk,B,3,3)
        logits = self.cls_head(Vs.reshape(Bk, self.B, 9))
        return logits, D, D0


# --------------------------------------------------------------------------
# MLP payload control (same D, non-associative V)
# --------------------------------------------------------------------------
class MLPPayloadModel(nn.Module):
    def __init__(self, B=2, d_emb=32, dv=32, tau=0.05, eps_self=1e-5,
                 d_max=D_MAX, hidden=96):
        super().__init__()
        self.B, self.dv, self.tau, self.eps_self, self.d_max = B, dv, tau, eps_self, d_max
        self.cost_emb = nn.Embedding(N_COST, d_emb)
        self.cost_head = nn.Linear(d_emb, 1)
        self.op_emb = nn.Embedding(N_OP, d_emb)
        self.op_head = nn.Linear(d_emb, dv)
        self.phi = nn.Sequential(nn.Linear(2 * B * dv, hidden), nn.GELU(),
                                 nn.Linear(hidden, B * dv))
        self.cls_head = nn.Sequential(nn.Linear(B * dv, 64), nn.GELU(), nn.Linear(64, B * N_OP))

    def encode(self, cost_tok, op_tok, edge, diag):
        Bk, K, _ = cost_tok.shape
        d_edge = F.softplus(self.cost_head(self.cost_emb(cost_tok)).squeeze(-1))
        D0 = torch.where(diag, torch.zeros_like(d_edge),
                         torch.where(edge, d_edge, torch.full_like(d_edge, self.d_max)))
        V0 = self.op_head(self.op_emb(op_tok)).view(Bk, K, K, self.B * self.dv)
        V0 = torch.where(edge[..., None], V0, torch.zeros_like(V0))
        V0 = torch.where(diag[..., None], torch.zeros_like(V0), V0)
        return D0, V0

    def _square(self, D, V, D0=None, V0=None):
        Bk, K, _ = D.shape
        S = D.unsqueeze(2) + D.transpose(1, 2).unsqueeze(1)
        eye = torch.eye(K, device=D.device)
        jstar = (S - self.eps_self * eye.view(1, 1, K, K)).argmin(dim=-1)
        D_hard = S.min(dim=-1).values
        b_idx = torch.arange(Bk, device=D.device).view(Bk, 1, 1, 1).expand(Bk, K, K, 1)
        i_idx = torch.arange(K, device=D.device).view(1, K, 1, 1).expand(Bk, K, K, 1)
        j_idx = jstar.unsqueeze(-1)
        k_idx = torch.arange(K, device=D.device).view(1, 1, K, 1).expand(Bk, K, K, 1)
        Vl = V[b_idx, i_idx, j_idx].squeeze(3)          # (Bk,K,K,B*dv)
        Vr = V[b_idx, j_idx, k_idx].squeeze(3)
        phi = torch.cat([Vl, Vr], dim=-1)
        V_hard = self.phi(phi)
        alpha = torch.softmax(-S / self.tau, dim=-1)
        phi_all = torch.cat([V.unsqueeze(2).expand(Bk, K, K, K, V.shape[-1]),
                             V.transpose(1, 2).unsqueeze(1).expand(Bk, K, K, K, V.shape[-1])],
                            dim=-1)
        V_all = self.phi(phi_all)                       # (Bk,K,K,K,B*dv)
        V_soft = (alpha.unsqueeze(-1) * V_all).sum(dim=3)
        logsum = torch.logsumexp(-(S - D_hard.unsqueeze(-1)) / self.tau, dim=-1)
        D_soft = D_hard - self.tau * logsum
        D_new = D_soft + (D_hard - D_soft).detach()
        V_new = V_soft + (V_hard - V_soft).detach()
        return D_new, V_new

    def forward(self, cost_tok, op_tok, edge, diag, s, t, T):
        D0, V0 = self.encode(cost_tok, op_tok, edge, diag)
        D, V = D0, V0
        for _ in range(T):
            D, V = self._square(D, V)
            D = D.masked_fill(diag, 0.0)
            V = torch.where(diag[..., None], torch.zeros_like(V), V)
        Bk = D.shape[0]
        b = torch.arange(Bk, device=D.device)
        Vs = V[b, s, t]                                  # (Bk,B*dv)
        logits = self.cls_head(Vs).view(Bk, self.B, N_OP)
        return logits, D, D0


# --------------------------------------------------------------------------
# Vector-Loop control: one-hop GNN, N looped iterations
# --------------------------------------------------------------------------
class VectorLoopModel(nn.Module):
    def __init__(self, B=2, d_emb=32, dh=64):
        super().__init__()
        self.B, self.dh = B, dh
        self.cost_emb = nn.Embedding(N_COST, d_emb)
        self.op_emb = nn.Embedding(N_OP, d_emb)
        self.edge_mlp = nn.Sequential(nn.Linear(2 * dh + 2 * d_emb, dh), nn.GELU(),
                                      nn.Linear(dh, dh))
        self.node_mlp = nn.Sequential(nn.Linear(2 * dh, dh), nn.GELU(),
                                      nn.Linear(dh, dh))
        self.mask_emb = nn.Parameter(torch.zeros(1, 1, dh))
        self.cls_head = nn.Sequential(nn.Linear(2 * dh, 64), nn.GELU(),
                                      nn.Linear(64, B * N_OP))

    def forward(self, cost_tok, op_tok, edge, diag, s, t, N):
        Bk, K, _ = cost_tok.shape
        e = torch.cat([self.cost_emb(cost_tok), self.op_emb(op_tok).mean(-2)], dim=-1)
        h = self.mask_emb.expand(Bk, K, self.dh)
        for _ in range(N):
            hi = h.unsqueeze(2).expand(Bk, K, K, self.dh)
            hj = h.unsqueeze(1).expand(Bk, K, K, self.dh)
            msg = self.edge_mlp(torch.cat([hi, hj, e], dim=-1))
            msg = msg * edge[..., None].float()
            agg = msg.sum(dim=2)
            h = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        b = torch.arange(Bk, device=h.device)
        hs, ht = h[b, s], h[b, t]
        logits = self.cls_head(torch.cat([hs, ht], dim=-1)).view(Bk, self.B, N_OP)
        return logits, None, None


# --------------------------------------------------------------------------
# training / evaluation
# --------------------------------------------------------------------------
def batch_loss(model, batch, T, d_weight, model_kind):
    logits, D, D0 = model(batch["cost"], batch["op"], batch["edge"], batch["diag"],
                          batch["s"], batch["t"], T)
    Bk = logits.shape[0]
    ce = F.cross_entropy(logits.reshape(-1, N_OP), batch["target"].reshape(-1))
    extra = torch.zeros((), device=logits.device)
    if d_weight > 0 and D is not None:
        m = batch["edge"] & ~batch["diag"]
        extra = extra + F.mse_loss(D0[m], batch["true_cost"][m])
        b = torch.arange(Bk, device=D.device)
        extra = extra + F.mse_loss(D[b, batch["s"], batch["t"]], batch["true_dist"])
    return ce + d_weight * extra, ce, extra


@torch.no_grad()
def evaluate(model, model_kind, K, H, T, n, rng, R, table, ident, device):
    Bk = 64
    correct = torch.zeros(2, device=device)
    total = 0
    dist_err = 0.0
    for _ in range(0, n, Bk):
        b = min(Bk, n - total)
        batch = make_batch(b, K, H, rng, R, table, ident, device)
        logits, D, _ = model(batch["cost"], batch["op"], batch["edge"], batch["diag"],
                             batch["s"], batch["t"], T)
        pred = logits.argmax(-1)
        correct += (pred == batch["target"]).float().sum(0)
        if D is not None:
            bb = torch.arange(b, device=device)
            dist_err += (D[bb, batch["s"], batch["t"]] - batch["true_dist"]).abs().sum().item()
        total += b
    acc = (correct / total).cpu().tolist()
    both = float((correct.sum() / (2 * total)).item())
    return dict(acc_block0=acc[0], acc_block1=acc[1], acc_both=both,
                dist_mae=dist_err / total)


def build_model(name, device):
    if name in ("st", "edge1hop", "softmin"):
        return STModel(variant=name).to(device)
    if name == "mlp":
        return MLPPayloadModel().to(device)
    if name == "vector":
        return VectorLoopModel().to(device)
    raise ValueError(name)


def train(args):
    device = torch.device(args.device)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    R = cube_rotations()
    table = mult_table(R)
    ident = identity_index(R)
    model = build_model(args.model, device)
    if hasattr(model, "tau"):
        model.tau = args.tau
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    d_weight = 0.0 if (args.no_d_supervision or args.model == "vector") else args.d_weight
    # Controls that advance one hop per step must be given the same hop budget
    # as the doubling models (T=2 => 4 hops).  Otherwise the comparison is not
    # hop-matched.
    onehop = args.model in ("edge1hop", "vector")
    train_T_eff = 4 if onehop else args.train_T
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        H = rng.randint(1, 4)
        K = rng.randint(max(6, H + 1), 12)
        batch = make_batch(args.batch, K, H, rng, R, table, ident, device)
        loss, ce, extra = batch_loss(model, batch, train_T_eff, d_weight, args.model)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        if step % args.log_every == 0 or step == 1:
            print(f"[{args.model}] step {step:5d} H={H} K={K} loss={loss.item():.4f} "
                  f"ce={ce.item():.4f} d={extra.item():.4f} "
                  f"|g|={gnorm.item():.3f} ({time.time()-t0:.1f}s)", flush=True)

    results = {"model": args.model, "no_d_supervision": bool(args.no_d_supervision),
               "steps": args.steps, "train_T_eff": train_T_eff, "eval": {}}
    # Same test-time step budget for every model: the doubling model covers
    # 2^T hops in T steps, the one-hop controls cover T.  The controls are
    # trained with T=4 (enough to fit the H<=4 training set), then evaluated at
    # the same T grid; their failure at H=32, T=5 is the doubling advantage.
    for H, K in ((2, 6), (4, 8), (4, 12), (8, 20), (16, 20), (16, 48), (32, 48)):
        for T in (1, 2, 3, 4, 5):
            if not onehop and (1 << T) < H:
                continue
            r = evaluate(model, args.model, K, H, T, args.eval_n, rng, R, table,
                         ident, device)
            results["eval"][f"H{H}_K{K}_T{T}"] = r
            print(f"  H={H:2d} K={K:2d} T={T}  acc_both={r['acc_both']*100:6.2f}% "
                  f"(b0={r['acc_block0']*100:6.2f}, b1={r['acc_block1']*100:6.2f}) "
                  f"dist_mae={r['dist_mae']:.4f}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")
    return results


# --------------------------------------------------------------------------
# smoke test: group, quaternion, and the exact operator on a hand-built graph
# --------------------------------------------------------------------------
def smoke():
    device = torch.device("cpu")
    R = cube_rotations()
    table = mult_table(R)
    ident = identity_index(R)
    assert torch.allclose(R[ident], torch.eye(3))

    # quaternion round-trip: q=(1,0,0,0) -> I
    q = torch.tensor([[1.0, 0, 0, 0]])
    assert torch.allclose(quat_to_rot(q)[0], torch.eye(3), atol=1e-6)

    # hand-built graph: 0-1-2-3-4 backbone, node 5 distractor
    K, H = 6, 4
    D0 = torch.full((1, K, K), D_MAX)
    V0 = torch.zeros(1, K, K, 2, 3, 3)
    edge = torch.zeros(1, K, K, dtype=torch.bool)
    diag = torch.eye(K, dtype=torch.bool).unsqueeze(0)
    for l in range(H):
        edge[0, l, l + 1] = True
        D0[0, l, l + 1] = COST_VALUES[0]
    edge[0, 0, 5] = True
    D0[0, 0, 5] = COST_VALUES[4]
    edge[0, 5, 1] = True
    D0[0, 5, 1] = COST_VALUES[4]
    for i in range(K):
        D0[0, i, i] = 0.0
        V0[0, i, i] = torch.eye(3)
    g = [rng_idx for rng_idx in (3, 7, 11, 19)]
    for l, gi in enumerate(g):
        V0[0, l, l + 1, 0] = R[gi]
        V0[0, l, l + 1, 1] = R[(gi * 5 + 3) % 24]

    m = STModel(variant="st").to(device)
    D, V = D0, V0
    for _ in range(2):
        D, V = m._square(D, V, D0, V0)
    assert abs(D[0, 0, H].item() - H * COST_VALUES[0]) < 1e-3, D[0, 0, H]
    tgt0 = torch.eye(3)
    tgt1 = torch.eye(3)
    for gi in g:
        tgt0 = tgt0 @ R[gi]
        tgt1 = tgt1 @ R[(gi * 5 + 3) % 24]
    assert (V[0, 0, H, 0] - tgt0).abs().max() < 1e-4
    assert (V[0, 0, H, 1] - tgt1).abs().max() < 1e-4
    print("operator test: D and V exact through 2 squarings (4 hops)")

    # edge1hop must not reach 4 hops in 2 steps
    me = STModel(variant="edge1hop").to(device)
    De, Ve = D0, V0
    for _ in range(2):
        De, Ve = me._square(De, Ve, D0, V0)
    assert De[0, 0, H].item() > 1e3, De[0, 0, H]
    print("edge1hop test: 4-hop pair unreachable after 2 one-hop steps (expected)")

    # random model forward finite, both payload types
    for name in ("st", "edge1hop", "softmin", "mlp", "vector"):
        model = build_model(name, device)
        rng = random.Random(0)
        batch = make_batch(4, 8, 4, rng, R, table, ident, device)
        logits, D, D0_ = model(batch["cost"], batch["op"], batch["edge"],
                               batch["diag"], batch["s"], batch["t"],
                               2 if name != "vector" else 4)
        assert torch.isfinite(logits).all(), name
        print(f"forward ok: {name} logits={tuple(logits.shape)}")
    print("smoke: all checks passed")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="st",
                    choices=["st", "edge1hop", "softmin", "mlp", "vector"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--train-T", type=int, default=2)
    ap.add_argument("--d-weight", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--no-d-supervision", action="store_true")
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="toys/runs_toy1/st.json")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    else:
        train(args)


if __name__ == "__main__":
    main()
