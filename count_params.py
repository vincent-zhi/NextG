"""
Instantiate Tier 0 / Tier 1 configs and print unique vs effective parameter
counts for B0 / B1 / B2, checked against the plan's section 3A table.

Definitions:
  unique    = distinct parameters stored (tied embedding counted once;
              B2 core counted once).
  effective = parameters *executed* in one forward pass (B2 core counted N
              times; embedding/head counted once since they are tied).
  non-emb   = excluding the (tied) embedding matrix.
"""

import gc

import torch

from model import build, tier0_config, tier1_config

PLAN = {
    "Tier 0": {"unique": (25.7, 42.5), "effective": (64.2, 81.0)},
    "Tier 1": {"unique": (117.0, 146.4), "effective": (351.0, 380.4)},
}


def block_params(module_list):
    return sum(p.numel() for m in module_list for p in m.parameters())


def counts(name, cfg):
    model = build(name, cfg)
    total_unique = sum(p.numel() for p in model.parameters())  # dedups tied weights
    emb = model.embed.weight.numel()
    unique = (total_unique - emb, total_unique)

    if name == "B0":
        eff_nonemb = block_params(model.layers) + model.final_norm.weight.numel()
    elif name == "B1":
        eff_nonemb = (block_params(model.pre_layers) + block_params(model.core_layers)
                      + block_params(model.post_layers) + model.final_norm.weight.numel())
    else:  # B2: core executed N times
        eff_nonemb = (block_params(model.pre_layers)
                      + cfg.N * block_params(model.core_layers)
                      + block_params(model.post_layers)
                      + model.final_norm.weight.numel())
    effective = (eff_nonemb, eff_nonemb + emb)
    del model
    gc.collect()
    return unique, effective


def fmt(pair):
    return f"{pair[0] / 1e6:8.1f}M / {pair[1] / 1e6:8.1f}M"


if __name__ == "__main__":
    torch.manual_seed(0)
    for tier_name, cfg in (("Tier 0", tier0_config()), ("Tier 1", tier1_config())):
        print(f"\n{tier_name}: d={cfg.d}, d_ff={cfg.d_ff}, "
              f"L={cfg.L_pre}/{cfg.L_core}/{cfg.L_post}, N={cfg.N}, L_eff={cfg.L_eff()}")
        print(f"{'model':<6} {'unique (non-emb/total)':<26} {'effective (non-emb/total)':<26}")
        for name in ("B0", "B1", "B2"):
            unique, effective = counts(name, cfg)
            print(f"{name:<6} {fmt(unique):<26} {fmt(effective):<26}")
            if name == "B2":
                plan = PLAN[tier_name]
                got_u = tuple(round(x / 1e6, 1) for x in unique)
                got_e = tuple(round(x / 1e6, 1) for x in effective)
                ok_u = got_u == plan["unique"]
                ok_e = got_e == plan["effective"]
                print(f"       plan B2 unique {plan['unique']} -> {'MATCH' if ok_u else 'MISMATCH got ' + str(got_u)}; "
                      f"effective {plan['effective']} -> {'MATCH' if ok_e else 'MISMATCH got ' + str(got_e)}")
