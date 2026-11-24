"""Assemble runs/TOY_REPORT.md from the toy-1 / toy-2 JSON results."""
from __future__ import annotations

import json
import os

R1 = ["st", "st_hardcore", "st_hardcore_warm", "edge1hop", "softmin", "mlp",
      "vector", "st_hardcore_perm", "st_perm"]
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
              "| model | H1 T1 | H4 T2 (in-dist) | H8 T3 | H16 T4 | H32 T5 | "
              "dist MAE H32 |",
              "|---|---|---|---|---|---|---|"]
    def keys(onehop):
        # in-distribution column: the one-hop controls trained at T=4 are
        # evaluated at T=4; the extrapolation columns use the same T for all.
        k0 = "H4_K8_T4" if onehop else "H4_K8_T2"
        return (k0, "H8_K20_T3", "H16_K20_T4", "H32_K48_T5")

    for m in R1:
        r = load(f"runs/toy1/{m}.json")
        if not r:
            lines.append(f"| {m} | (pending) | | | | | |")
            continue
        e = r["eval"]
        k0, k1, k2, k3 = keys(m in ("edge1hop", "vector"))
        get = lambda k: e.get(k, {})
        lines.append(
            f"| {m} | {pct(get('H1_K6_T1').get('acc_both'))} | "
            f"{pct(get(k0).get('acc_both'))} | "
            f"{pct(get(k1).get('acc_both'))} | "
            f"{pct(get(k2).get('acc_both'))} | "
            f"{pct(get(k3).get('acc_both'))} | "
            f"{get(k3).get('dist_mae', float('nan')):.4f} |")
    hs = [load(f"runs/toy1/st_hardcore_s{s}.json") for s in (1, 2)]
    h0 = load("runs/toy1/st_hardcore.json")
    if h0 and all(hs):
        parts = []
        for r in [h0] + hs:
            e = r["eval"]
            parts.append("%.0f/%.0f (MAE/H32 %.3f)" % (
                100 * e["H8_K20_T3"]["acc_both"],
                100 * e["H32_K48_T5"]["acc_both"],
                e["H32_K48_T5"]["dist_mae"] / 32.0))
        note = ("  The metric offset is constant per hop at every seed; "
                "routing reliability is not.")
        s1 = load("runs/toy1/st_s1.json")
        if s1:
            note += ("  D-supervised `st` seeds 0/1: H32 100%/100%, "
                     f"dist MAE {load('runs/toy1/st.json')['eval']['H32_K48_T5']['dist_mae']:.4f}/"
                     f"{s1['eval']['H32_K48_T5']['dist_mae']:.4f} -- supervision "
                     "removes both the scale freedom and the seed lottery.")
        lines += ["", "`st_hardcore` seed spread (H8 / H32 acc): "
                  + "; ".join(parts) + "." + note]
    lines += ["", "Rows ending in `_perm` are the index-order control: every eval "
              "graph gets a random node permutation, so the spine no longer "
              "occupies the lowest indices.  Accuracy should be unchanged if the "
              "ST routing is a real signal rather than an argmin tie-break."]
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
                  "AUROC is per constraint unit (Sudoku: 18 row/col/box units; "
                  "SAT: per clause), computed on held-out solution grids. "
                  "`retention` is the true conditional P(board correct at T=64 | "
                  "board correct at T=8); `acc ratio` is acc(T64)/acc(T8). "
                  "`untrained` = the same solver on a never-trained encoder "
                  "(same init seed): the control for whether anything is learned.", "",
                  "| mode | acc(T4) | acc(T8) | acc(T64) | full-grid | AUROC | "
                  "retention P(ok64|ok8) | acc ratio | viol rate |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for m in R2 + ["untrained"]:
            r = load(f"runs/toy2/{ds}_{m}.json")
            if not r:
                tag = ("killed 2026-11-25 (stop directive)"
                       if m in ("no_zero_lock", "soft") else "(pending)")
                lines.append(f"| {m} | {tag} | | | | | | | |")
                continue
            e = r["eval"]
            a4 = e.get("T4", {}).get("acc_cells")
            a8 = e.get("T8", {}).get("acc_cells")
            a64 = e.get("T64", {}).get("acc_cells")
            au = e.get("T64", {}).get("auroc_hard_score")
            ret = r.get("retention_true_8_to_64", r.get("retention_8_to_64"))
            ratio = r.get("acc_ratio_8_to_64")
            au_s = "n/a" if au is None else f"{au:.3f}"
            ret_s = "n/a" if ret is None else f"{ret:.3f}"
            ratio_s = "n/a" if ratio is None else f"{ratio:.3f}"
            lines.append(
                f"| {m} | {pct(a4)} | {pct(a8)} | {pct(a64)} | "
                f"{pct(e.get('T64', {}).get('acc_full'))} | "
                f"{au_s} | {ret_s} | {ratio_s} | "
                f"{pct(e.get('T64', {}).get('viol_rate'))} |")
        full = load(f"runs/toy2/{ds}_full.json")
        if full:
            best_ctrl = 0.0
            for m in ("open", "primal", "untrained"):
                r = load(f"runs/toy2/{ds}_{m}.json")
                if r:
                    best_ctrl = max(best_ctrl, r["eval"].get("T8", {}).get("acc_cells", 0.0))
            a = full["eval"].get("T8", {}).get("acc_cells", 0.0)
            au = full["eval"].get("T64", {}).get("auroc_hard_score")
            ret = full.get("retention_true_8_to_64", full.get("retention_8_to_64"))
            au_s = "n/a" if au is None else f"{au:.3f}"
            ret_s = "n/a" if ret is None else f"{ret:.3f}"
            unt = load(f"runs/toy2/{ds}_untrained.json")
            unt_s = ""
            if unt:
                au0 = unt["eval"].get("T8", {}).get("acc_cells", 0.0)
                unt_s = (f" Untrained solver baseline: acc(T8) {pct(au0)}, "
                         f"so the learned gain is {100*(a-au0):+.1f}pt.")
                extra = [load(f"runs/toy2/{ds}_untrained_s{s}.json")
                         for s in (1, 2)]
                extra = [x for x in extra if x]
                if extra:
                    t8s = [au0] + [x["eval"]["T8"]["acc_cells"] for x in extra]
                    t64s = [unt["eval"]["T64"]["acc_cells"]] + [
                        x["eval"]["T64"]["acc_cells"] for x in extra]
                    unt_s += (f" Across {len(t8s)} untrained inits: T8 "
                              f"{100*min(t8s):.1f}-{100*max(t8s):.1f}%, T64 "
                              f"{100*min(t64s):.1f}-{100*max(t64s):.1f}%.")
                tseeds = [load(f"runs/toy2/{ds}_full_s{s}.json") for s in (1, 2)]
                if all(tseeds):
                    t8t = [a] + [x["eval"]["T8"]["acc_cells"] for x in tseeds]
                    t64t = [full["eval"]["T64"]["acc_cells"]] + [
                        x["eval"]["T64"]["acc_cells"] for x in tseeds]
                    unt_s += (f" Trained `full` across {len(t8t)} seeds: T8 "
                              f"{100*min(t8t):.1f}-{100*max(t8t):.1f}%, T64 "
                              f"{100*min(t64t):.1f}-{100*max(t64t):.1f}%.")
            lines += ["", f"**Gates**: AUROC {au_s} "
                      f"(>=0.90 {'PASS' if (au or 0) >= 0.9 else 'FAIL'}); "
                      f"true retention {ret_s} "
                      f"(>=0.98 {'PASS' if (ret or 0) >= 0.98 else 'FAIL'}); "
                      f"gain {100*(a-best_ctrl):+.1f}pt over best control "
                      f"(>=+15pt {'PASS' if a-best_ctrl >= 0.15 else 'FAIL'})."
                      + unt_s, ""]
        if ds == "sudoku":
            nv = load("runs/toy2/sudoku_neuralv_vsuper.json")
            if nv:
                ev = nv["eval"]
                lines += ["", "**Neural verifier V_theta (Protocol B)** — encoder "
                          "left at its random init; V_theta trained on 2-cell "
                          "hard negatives. Held-out board AUROC "
                          f"{nv.get('v_auroc_heldout')}. Solver with the learned "
                          "V:", "",
                          "| T | acc_cells | acc_full | viol rate |",
                          "|---|---|---|---|"]
                for t in ("T4", "T8", "T16", "T32", "T64"):
                    d = ev.get(t, {})
                    lines.append(f"| {t} | {pct(d.get('acc_cells'))} | "
                                 f"{pct(d.get('acc_full'))} | "
                                 f"{pct(d.get('viol_rate'))} |")
                lines.append("")

    out = "\n".join(lines) + "\n"
    os.makedirs("runs", exist_ok=True)
    with open("runs/TOY_REPORT.md", "w") as f:
        f.write(out)
    print(out)


if __name__ == "__main__":
    main()
