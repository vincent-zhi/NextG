"""Assemble the verified Stage 1/2/3 tables into runs/FINAL_REPORT.md."""
from __future__ import annotations

import json
import os


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def pct(x):
    return f"{100*x:.1f}%" if isinstance(x, (int, float)) else "n/a"


def main():
    root = "runs"
    s1 = load(f"{root}/stage1_report.json")
    s2 = load(f"{root}/stage2_report_B2_THIN.json")
    s2f = load(f"{root}/stage2_report_F_VCYCLE.json")
    s3 = load(f"{root}/stage3_report.json")
    lines = ["# Looped-Transformer F/E/G — verified results", ""]

    if s1:
        lines += ["## Stage 1 — Direction F (channel-space multigrid V-cycle)", "",
                  "| model | non-emb params | total params | tok/s | peak VRAM | syn_nll | in-dist acc | OOD@N_train | N sweep (OOD mean) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for m in ("B2_THIN", "B2_NARROW", "F_VCYCLE"):
            if m not in s1:
                continue
            r = s1[m]
            curve = ", ".join(f"{k}:{100*v['ood_mean']:.1f}%"
                              for k, v in r["curve"].items())
            lines.append(
                f"| {m} | {r['nonemb_params']/1e6:.3f}M | {r['params']/1e6:.3f}M | "
                f"{r['tok_per_s']:.0f} | {r['peak_vram_gb']:.2f} GB | "
                f"{r['syn_nll']:.4f} | {pct(r['in_dist_mean'])} | "
                f"{pct(r['ood_at_train'])} | {curve} |")
        v = s1.get("verdict", {})
        lines += ["", f"**Verdict F: {v.get('F','?')}** — "
                  f"F−B2_THIN {100*v.get('F_ood_minus_B2_THIN',0):+.1f}pt, "
                  f"F−B2_NARROW {100*v.get('F_ood_minus_B2_NARROW',0):+.1f}pt, "
                  f"speed {v.get('speed_ratio_vs_B2_THIN',0):.2f}x, "
                  f"Nc10→20 {100*v.get('Nc10_to_Nc20_delta',0):+.1f}pt.", ""]

    for tag, r in (("", s2), (" (F_VCYCLE)", s2f)):
        if not r:
            continue
        lines += [f"## Stage 2 — Direction E{tag}", ""]
        g2a = r.get("gate2a") or r.get("gate2a_supplement")
        if g2a:
            lines += ["| sigma_rel | Pass@1 | Oracle Pass@4 | headroom |",
                      "|---|---|---|---|"]
            for s, g in g2a.items():
                lines.append(f"| {s} | {pct(g['pass1'])} | {pct(g['oracle_pass4'])} | "
                             f"{100*g['headroom']:+.1f}pt |")
        if "gate2b" in r:
            g = r["gate2b"]
            lines += ["", "| row | OOD accuracy |", "|---|---|"]
            order = [("B2_N4_K1", "B2 (N=4, K=1)"),
                     ("B2_N16_K1", "B2 (N=16, K=1) — FLOP match"),
                     ("logit_conf_bo4", "Logit-Conf Best-of-4"),
                     ("E2_verifier_bo4", "E2 Verifier Best-of-4"),
                     ("oracle_pass4", "Oracle Pass@4"),
                     ("E1_auto_depth", "E1 Auto-Depth"),
                     ("B2_N32_K1", "B2 (N=32, K=1)")]
            for k, label in order:
                if k in g:
                    lines.append(f"| {label} | {pct(g[k])} |")
            lines += ["", f"Verifier OOD AUROC {g.get('auroc_verifier',0):.3f} vs "
                      f"Logit-Conf AUROC {g.get('auroc_logit_conf',0):.3f}; "
                      f"overthinking drop {g.get('overthinking_drop',0):.4f}, "
                      f"E1 reduction {100*g.get('e1_drop_reduction',0):.0f}%."]
        lines += ["", f"**Verdict E: {r.get('E','?')}**", ""]

    if s3:
        lines += ["## Stage 3 — Direction G (holographic coefficients, FineWeb-Edu)", "",
                  "| model | non-emb params | weight+opt VRAM | peak VRAM | tok/s | final fw_bpb |",
                  "|---|---|---|---|---|---|"]
        for m in ("DENSE_NARROW", "G_WHT", "B0_FULL"):
            if m not in s3 or "tok_per_s" not in s3[m]:
                continue
            r = s3[m]
            lines.append(f"| {m} | {r['nonemb_params']/1e6:.3f}M | "
                         f"{r['static_vram_mb']:.1f} MB | {r.get('peak_vram_gb',0):.2f} GB | "
                         f"{r['tok_per_s']:.0f} | {r.get('final_fw_bpb','n/a')} |")
        b0 = s3.get("B0_FULL", {})
        if "tok_per_s" not in b0:
            lines.append(f"| B0_FULL (static only, not trained) | "
                         f"{b0.get('nonemb_params',0)/1e6:.3f}M | "
                         f"{b0.get('static_vram_mb',0):.1f} MB | — | — | — |")
        v = s3.get("verdict", {})
        lines += ["", f"**Verdict G: {v.get('G','?')}** — VRAM ratio vs B0 "
                  f"{v.get('static_vram_ratio_vs_B0',0):.2f}x, "
                  f"G−DENSE fw_bpb {v.get('g_minus_dense_bpb')}, "
                  f"speed ratio vs DENSE {v.get('tok_ratio_vs_dense',0):.2f}x.", ""]

    out = "\n".join(lines)
    with open(f"{root}/FINAL_REPORT.md", "w") as f:
        f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
