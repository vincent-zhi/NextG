"""
Track C parameter-budget fix (Phase 1 prep, not Phase 0).

Problem found in the plan audit (2026-11-23): the plan claims Track C Tier 1
absorbs 66.8M parameters in boundary MoE and lands exactly on 117.0M non-emb,
but the micro-logic core (8 layers, 42.2M) + boundary attention (4 layers,
12.85M) already leave only 61.93M for MoE + tape. With expert d_ff ~= 777
(the value implied by 66.8M) the true total is ~121.9M before tape -- about
5-7M over target.

Fix proposed here, verified against model.py Block parameter counts
(no hand-waving: attn / norms / FFN are read off a real Block):
  * Tier 1: expert d_ff 777 -> 704, with tape accounted concretely below.
    Total non-emb = 116,966,785 -> 117.0M at the plan's 0.1M rounding.
  * Tier 0: the plan gives no Track C config at all. Two scalings are
    computed; the choice (absolute 768 vs 31%-of-full-width 448 micro-FFN)
    is left to the user -- see output.

Tape accounting used (one concrete design, so the budget actually closes;
swap it for the plan author's tape and re-run this script):
  write queries 8x256, write KV proj d -> 2*d_m, read cross-attention
  q/k/v/out projs (d<->d_m). All other tape pieces are scalars / norms.
"""
import torch

from model import Block, tier0_config, tier1_config

TARGET = {"Tier 0": 25.7e6, "Tier 1": 117.0e6}


def block_parts(cfg):
    b = Block(cfg)
    parts = {n: sum(p.numel() for p in m.parameters())
             for n, m in b.named_children()}
    return parts  # norm1, attn (incl. QK-Norm), norm2, ffn


def tape_params(d, d_m=256, m=8):
    return {
        "write_queries (m x d_m)": m * d_m,
        "write_kv_proj (d -> 2*d_m)": d * 2 * d_m,
        "read_q_proj (d -> d_m)": d * d_m,
        "read_k_proj (d_m -> d_m)": d_m * d_m,
        "read_v_proj (d_m -> d_m)": d_m * d_m,
        "read_out_proj (d_m -> d)": d_m * d,
    }


def track_c(cfg, micro_ff, expert_ff):
    p = block_parts(cfg)
    attn_norms = p["attn"] + p["norm1"] + p["norm2"]  # per-layer, no FFN
    micro_ffn = 3 * cfg.d * micro_ff
    core = cfg.L_core * (attn_norms + micro_ffn)
    boundary_attn = (cfg.L_pre + cfg.L_post) * attn_norms
    n_exp_layers = cfg.L_pre + cfg.L_post
    moe = n_exp_layers * (8 * 3 * cfg.d * expert_ff + cfg.d * 8)  # experts + router
    tape = sum(tape_params(cfg.d).values())
    final_norm = cfg.d
    total = core + boundary_attn + moe + tape + final_norm
    return {
        "per-layer attn+norms": attn_norms,
        "micro core ({} layers)".format(cfg.L_core): core,
        "boundary attn ({} layers)".format(n_exp_layers): boundary_attn,
        "MoE ({} layers, 8 experts, d_ff={})".format(n_exp_layers, expert_ff): moe,
        "tape (d_m=256, m=8)": tape,
        "final norm": final_norm,
        "TOTAL non-emb": total,
        "target": TARGET["Tier 1" if cfg.d == 896 else "Tier 0"],
    }


def show(tier, cfg, micro_ff, expert_ff):
    print(f"\n{tier}: d={cfg.d}, full d_ff={cfg.d_ff}, micro d_ff={micro_ff}, "
          f"expert d_ff={expert_ff}")
    r = track_c(cfg, micro_ff, expert_ff)
    for k, v in r.items():
        print(f"  {k:<38} {v/1e6:8.3f}M" if isinstance(v, (int, float)) and v > 1e5
              else f"  {k:<38} {v}")
    print(f"  -> total {r['TOTAL non-emb']/1e6:.1f}M vs target "
          f"{r['target']/1e6:.1f}M : "
          f"{'MATCH at 0.1M rounding' if round(r['TOTAL non-emb']/1e6, 1) == round(r['target']/1e6, 1) else 'OFF'}")


if __name__ == "__main__":
    torch.manual_seed(0)
    # Tier 1: recommended locked fix (expert 704 + concrete tape design)
    show("Tier 1 (recommended)", tier1_config(), micro_ff=768, expert_ff=704)
    # Tier 1: the plan as written (expert ~777) -- shows the overrun
    show("Tier 1 (plan as written)", tier1_config(), micro_ff=768, expert_ff=777)
    # Tier 0: no config in plan; both micro scalings, expert sized to fill
    # the remaining budget after the same tape design (tape ~0.59M at d=512).
    for micro_ff, expert_ff in ((768, 242), (448, 280)):
        show(f"Tier 0 (micro={micro_ff}, candidate)", tier0_config(),
             micro_ff=micro_ff, expert_ff=expert_ff)
    print("\nTape breakdown (Tier 1, d=896):")
    for k, v in tape_params(896).items():
        print(f"  {k:<32} {v:,}")
