"""Diagnostic: is the toy-2 AUROC a learned signal, or a property of the
analytic constraint residual?

Reported AUROC = soft-residual score vs hard-residual label, both computed from
the same model state.  This script evaluates that number for an *untrained*
encoder (random z) and for shuffled labels, on the exact test split of the main
run, plus the ground-truth localisation AUROC (Sudoku).
"""
import argparse
import json
import random
from types import SimpleNamespace

import torch

import toy2_primal_dual as tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sudoku", choices=["sudoku", "sat"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--mode", default="open",
                    choices=["open", "full", "primal", "no_zero_lock", "soft"])
    ap.add_argument("--out", default="")
    ap.add_argument("--neural-v", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cpu")
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)  # same init as the trained runs
    data_args = SimpleNamespace(dataset=a.dataset, n_sol=200, per_sol=2,
                                n_sat=200, sat_n_train=8, sat_n_ood=12)
    _, te = tp.make_tasks(data_args, rng, dev)
    model = tp.Model(te, neural_v=a.neural_v).to(dev)
    ev = SimpleNamespace(mode=a.mode, eps=1e-3, eta=0.3, beta=0.9, gamma=0.5, lam_max=5.0)
    res = {"dataset": a.dataset, "mode": a.mode, "untrained": True,
           "n_test": int(te.x.shape[0]), "eval": {}}
    for T_ in (4, 8, 16, 32, 64):
        r = tp.evaluate(model, te, T_, ev)
        r.pop("board_ok", None)
        res["eval"][f"T{T_}"] = r
        print(f"untrained {a.dataset} mode={a.mode} T={T_:2d}: "
              f"acc_cells={r['acc_cells']*100:6.2f}% "
              f"acc_full={r['acc_full']*100:5.2f}% viol={r['viol_rate']*100:6.2f}% "
              f"auroc_score={r['auroc_hard_score']} "
              f"auroc_score_hard={r['auroc_score_hard']} "
              f"auroc_analytic={r['auroc_analytic']} "
              f"auroc_vs_truth={r['auroc_vs_truth']}")
    a4 = res["eval"]["T4"]["acc_cells"]
    a8 = res["eval"]["T8"]["acc_cells"]
    a64 = res["eval"]["T64"]["acc_cells"]
    res["acc_ratio_8_to_64"] = (a64 / a8) if a8 > 0 else None
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"wrote {a.out} (acc ratio T64/T8 = {res['acc_ratio_8_to_64']})")
    z, _, _ = model(te, a.T, ev)
    p = torch.softmax(z, -1)
    e = te.clamp_e(tp.ste_hard(p))
    nv = te.violation(e).norm(dim=-1).flatten()
    label = nv < 1e-6
    nv_soft = te.violation(te.clamp_e(p)).norm(dim=-1).flatten()
    g = torch.Generator().manual_seed(1)
    perm = torch.randperm(label.numel(), generator=g)
    print(f"  shuffled-label AUROC={tp.auroc(-nv_soft[perm], label)}")


if __name__ == "__main__":
    main()
