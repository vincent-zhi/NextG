"""Assemble runs/TOY_REPORT.md from the toy-1 / toy-2 JSON results."""
from __future__ import annotations

import json
import os

R1 = ["st", "st_hardcore", "edge1hop", "softmin", "mlp", "vector"]
R2 = ["full", "open", "primal", "no_zero_lock", "soft"]


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def pct(x):
    return "n/a" if x is None else f"{100*x:6.2f}%"


def main():
    lines = ["# Toy 1 / Toy 2 — measured results", ""]

    # ---------------- toy 1 ----------------
    lines += ["## Toy 1 — ST-tropical doubling with an SO(3)^2 payload", "",
              "Accuracy is the fraction of graphs on which BOTH payload blocks "
              "are classified correctly (chance = 4.17%). `dist_mae` is the "
              "absolute error of the learned shortest-path distance.", "",
              "| model | H4 T2 (in-dist) | H8 T3 | H16 T4 | H32 T5 | dist MAE H32 |",
              "|---|---|---|---|---|---|"]
    def keys(onehop):
        # in-distribution column: the one-hop controls trained at T=4 are
        # evaluated at T=4; the extrapolation columns use the same T for all.
        k0 = "H4_K8_T4" if onehop else "H4_K8_T2"
        return (k0, "H8_K20_T3", "H16_K20_T4", "H32_K48_T5")

    for m in R1:
        r = load(f"runs/toy1/{m}.json")
        if not r:
            lines.append(f"| {m} | (pending) | | | | |")
            continue
        e = r["eval"]
        k0, k1, k2, k3 = keys(m in ("edge1hop", "vector"))
        get = lambda k: e.get(k, {})
        lines.append(
            f"| {m} | {pct(get(k0).get('acc_both'))} | "
            f"{pct(get(k1).get('acc_both'))} | "
            f"{pct(get(k2).get('acc_both'))} | "
            f"{pct(get(k3).get('acc_both'))} | "
            f"{get(k3).get('dist_mae', float('nan')):.4f} |")
    st = load("runs/toy1/st.json")
    if st:
        e = st["eval"]
        a16 = e.get("H16_K20_T4", {}).get("acc_both", 0.0)
        a32 = e.get("H32_K48_T5", {}).get("acc_both", 0.0)
        lines += ["", f"**Gate (>=90% at H16/T4 and H32/T5): "
                  f"H16 {pct(a16)}, H32 {pct(a32)} -> "
                  f"{'PASS' if a16 >= 0.9 and a32 >= 0.9 else 'FAIL'}**", ""]

    # ---------------- toy 2 ----------------
    for ds in ("sudoku", "sat"):
        lines += [f"## Toy 2 — {ds}", "",
                  "| mode | acc(T8) | acc(T64) | full-grid | AUROC | retention | viol rate |",
                  "|---|---|---|---|---|---|---|"]
        for m in R2:
            r = load(f"runs/toy2/{ds}_{m}.json")
            if not r:
                lines.append(f"| {m} | (pending) | | | | | |")
                continue
            e = r["eval"]
            a8 = e.get("T8", {}).get("acc_cells")
            a64 = e.get("T64", {}).get("acc_cells")
            au = e.get("T64", {}).get("auroc_hard_score")
            ret = r.get("retention_8_to_64")
            au_s = "n/a" if au is None else f"{au:.3f}"
            ret_s = "n/a" if ret is None else f"{ret:.3f}"
            lines.append(
                f"| {m} | {pct(a8)} | {pct(a64)} | "
                f"{pct(e.get('T64', {}).get('acc_full'))} | "
                f"{au_s} | {ret_s} | "
                f"{pct(e.get('T64', {}).get('viol_rate'))} |")
        full = load(f"runs/toy2/{ds}_full.json")
        if full:
            best_ctrl = 0.0
            for m in ("open", "primal", "no_zero_lock", "soft"):
                r = load(f"runs/toy2/{ds}_{m}.json")
                if r:
                    best_ctrl = max(best_ctrl, r["eval"].get("T8", {}).get("acc_cells", 0.0))
            a = full["eval"].get("T8", {}).get("acc_cells", 0.0)
            au = full["eval"].get("T64", {}).get("auroc_hard_score")
            ret = full.get("retention_8_to_64")
            au_s = "n/a" if au is None else f"{au:.3f}"
            ret_s = "n/a" if ret is None else f"{ret:.3f}"
            lines += ["", f"**Gates**: AUROC {au_s} "
                      f"(>=0.90 {'PASS' if (au or 0) >= 0.9 else 'FAIL'}); "
                      f"retention {ret_s} "
                      f"(>=0.98 {'PASS' if (ret or 0) >= 0.98 else 'FAIL'}); "
                      f"gain {100*(a-best_ctrl):+.1f}pt over best control "
                      f"(>=+15pt {'PASS' if a-best_ctrl >= 0.15 else 'FAIL'}).", ""]

    out = "\n".join(lines) + "\n"
    os.makedirs("runs", exist_ok=True)
    with open("runs/TOY_REPORT.md", "w") as f:
        f.write(out)
    print(out)


if __name__ == "__main__":
    main()
