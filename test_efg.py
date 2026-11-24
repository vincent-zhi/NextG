from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F

from data_pipeline import ASTGenerator, ByteTokenizer, Phase0Data
from model import LoopConfig, B2FlatLoop
from tracks_efg import (EFG_MODELS, FVCycle, build_efg, efg_config,
                        STAGE3_MODELS, build_stage3, stage3_config)


TINY_F = LoopConfig(d=128, d_ff=352, n_heads=2, L_pre=1, L_core=2,
                    L_post=0, vocab_size=1024, N=10)
B, T = 2, 24


def tiny_ids(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, TINY_F.vocab_size, (B, T), generator=g)


def build_tiny(name):
    return build_efg(name, TINY_F)


def activate(model):
    with torch.no_grad():
        for pname, param in model.named_parameters():
            if "out_proj.weight" in pname or "down_proj.weight" in pname:
                param.normal_(std=0.02)


def test_configs():
    for name in EFG_MODELS:
        cfg = efg_config(name, vocab_size=1024)
        assert cfg.L_pre == 1 and cfg.L_core == 4 and cfg.L_post == 0
        assert cfg.d % cfg.n_heads == 0 and cfg.d_head == 64
    f = efg_config("F_VCYCLE", 1024)
    assert f.d == 512 and f.d_ff == 1408 and f.N == 10
    assert efg_config("B2_NARROW", 1024).N == 14
    print("  [PASS] EFG configs: B2_THIN(1/4/0,N=4) B2_NARROW(256,N=14) F_VCYCLE(512/256,N_c=10)")


def test_params():
    b2 = B2FlatLoop(TINY_F)
    f = build_tiny("F_VCYCLE")
    b2_n = sum(p.numel() for p in b2.parameters())
    f_n = sum(p.numel() for p in f.parameters())
    extra = 2 * f.d0 * TINY_F.L_core
    assert f_n - b2_n == extra, (f_n, b2_n, extra)
    assert isinstance(f, FVCycle)
    expect_sc = 1.0 / (10 * math.sqrt(1.0 + 10 * TINY_F.L_core))
    expect_sf = 1.0 / math.sqrt(1.0 + (10 + 1) * TINY_F.L_core)
    assert abs(f.scale_c - expect_sc) < 1e-12 and abs(f.scale_f - expect_sf) < 1e-12
    print(f"  [PASS] F_VCYCLE adds ONLY coarse norms (+{extra} params); scale_c={f.scale_c:.5f} scale_f={f.scale_f:.5f}")


def test_causality(name):
    torch.manual_seed(17)
    model = build_tiny(name).eval()
    activate(model)
    ids = tiny_ids()
    t = 10

    emb = model.embed(ids).detach().clone().requires_grad_(True)
    logits = model(inputs_embeds=emb)
    weights = torch.randn(TINY_F.vocab_size)
    (logits[0, t] * weights).sum().backward()
    grad = emb.grad[0, t + 1]
    assert grad is not None and grad.abs().max().item() == 0.0, f"{name}: future gradient is nonzero"
    assert emb.grad[0, t].abs().max().item() > 0.0, f"{name}: causality check is vacuous"

    ids2 = ids.clone()
    ids2[0, t + 1] = (ids2[0, t + 1] + 7) % TINY_F.vocab_size
    with torch.no_grad():
        out1 = model(ids)
        out2 = model(ids2)
    assert torch.equal(out1[:, : t + 1], out2[:, : t + 1]), f"{name}: past logits changed"
    assert not torch.equal(out1[:, t + 1], out2[:, t + 1]), f"{name}: perturbation had no effect"
    print(f"  [PASS] {name}: exact causal gradient and bitwise-identical past logits")


def loss_for(model, ids):
    logits = model(ids[:, :-1])
    return F.cross_entropy(logits.reshape(-1, TINY_F.vocab_size), ids[:, 1:].reshape(-1))


def test_grad_flow(name):
    torch.manual_seed(7)
    model = build_tiny(name)
    ids = tiny_ids(seed=3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)

    opt.zero_grad()
    loss_for(model, ids).backward()
    for pname, param in model.named_parameters():
        assert param.grad is not None, f"{name}: {pname} has no gradient in phase A"
        assert torch.isfinite(param.grad).all(), f"{name}: {pname} has a non-finite gradient"

    opt.step()
    opt.zero_grad()
    loss_for(model, ids).backward()
    still_zero = []
    for pname, param in model.named_parameters():
        assert param.grad is not None, f"{name}: {pname} has no gradient in phase B"
        assert torch.isfinite(param.grad).all(), f"{name}: {pname} has a non-finite gradient"
        if param.grad.abs().sum().item() == 0.0:
            still_zero.append(pname)
    assert not still_zero, f"{name}: zero gradients after one optimizer step: {still_zero}"

    if name == "F_VCYCLE":
        cn = [n for n, _ in model.named_parameters() if "coarse_norm" in n]
        assert len(cn) == 2 * TINY_F.L_core, f"expected {2 * TINY_F.L_core} coarse norms, got {len(cn)}"
        assert all(model.get_parameter(n).grad is not None for n in cn)
    print(f"  [PASS] {name}: all {len(list(model.parameters()))} parameters receive non-zero finite gradients")


def test_identity_at_init():
    torch.manual_seed(11)
    ref = B2FlatLoop(TINY_F).eval()
    f = build_tiny("F_VCYCLE").eval()
    f.load_state_dict(ref.state_dict(), strict=False)
    ids = tiny_ids(seed=5)
    with torch.no_grad():
        r0 = ref(ids)
        f0 = f(ids)
    diff = (r0 - f0).abs().max().item()
    assert diff < 1e-5, f"F_VCYCLE is not B2-identical at zero-init: {diff}"
    print(f"  [PASS] F_VCYCLE zero-init equals B2 (out_proj/down_proj=0), max|diff|={diff:.3e}")


def test_extrapolation(name):
    torch.manual_seed(13)
    model = build_tiny(name).eval()
    activate(model)
    ids = tiny_ids(seed=9)
    sweep = (2, 5, 10, 20, 40) if name == "F_VCYCLE" else (1, 2, 4, 8, 16, 32)
    rms = {}
    with torch.no_grad():
        for N in sweep:
            _, hidden = model(ids, N=N, return_hidden=True)
            rms[N] = hidden.pow(2).mean().sqrt().item()
    base = next(iter(rms.values()))
    bound = max(10.0 * base, base + 5.0)
    for N, value in rms.items():
        assert math.isfinite(value) and value <= bound, f"{name}: RMS exploded at N={N}: {value}"
    print(f"  [PASS] {name}: extrapolation RMS " + ", ".join(f"N={N}:{v:.4f}" for N, v in rms.items()))


def test_cuda_fp16(name):
    if not torch.cuda.is_available():
        print(f"  [SKIP] {name}: CUDA FP16 unavailable on this host")
        return
    torch.manual_seed(19)
    model = build_tiny(name).cuda()
    activate(model)
    ids = tiny_ids(seed=4).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(ids)
        loss = F.cross_entropy(logits.float().reshape(-1, TINY_F.vocab_size), ids.reshape(-1))
    assert torch.isfinite(logits).all(), f"{name}: non-finite FP16 logits"
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    for pname, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), \
            f"{name}: bad FP16 gradient for {pname}"
    scaler.step(opt)
    scaler.update()
    assert math.isfinite(scaler.get_scale()) and scaler.get_scale() > 0, f"{name}: invalid grad scale"
    print(f"  [PASS] {name}: CUDA FP16 forward/backward finite, grad_scale={scaler.get_scale()}")


def test_families_data():
    tok = ByteTokenizer()
    data = Phase0Data(seed=42, tokenizer=tok, text_corpus=None,
                      synthetic_fraction=1.0, seq_len=128, families=("graph", "ast"))
    ids, targets, mask, syn = data.next_batch(4)
    assert ids.shape == (4, 128) and torch.all(syn == 1)
    assert not torch.any((targets == tok.pad_id) & (mask == 1))
    state = data.state_dict()
    expected = data.next_batch(2)
    restored = Phase0Data(seed=42, tokenizer=tok, text_corpus=None,
                          synthetic_fraction=1.0, seq_len=128, families=("graph", "ast"))
    restored.load_state_dict(state)
    actual = restored.next_batch(2)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    generator = ASTGenerator(random.Random(5), ood=True)
    longest = max(len(tok.encode(e.prompt + e.answer)) + 1
                  for e in (generator._dyck() for _ in range(20)))
    assert longest <= 1024
    print(f"  [PASS] graph+ast-only stream, resume-exact, padding unsupervised, OOD Dyck max={longest}")


def test_stage3_g():
    g = build_stage3("G_WHT", stage3_config("G_WHT", 1024))
    h1, h2 = g.layers[0].attn.q_proj.H1, g.layers[0].attn.q_proj.H2
    e1 = (h1.T @ h1 - torch.eye(16)).abs().max().item()
    e2 = (h2.T @ h2 - torch.eye(32)).abs().max().item()
    assert e1 < 1e-6 and e2 < 1e-6, (e1, e2)
    nonemb = {n: sum(p.numel() for pn, p in build_stage3(n, stage3_config(n, 258))
                     .named_parameters() if "embed" not in pn)
              for n in STAGE3_MODELS}
    assert abs(nonemb["G_WHT"] - 17.063e6) < 0.05e6, nonemb
    assert abs(nonemb["DENSE_NARROW"] - 17.052e6) < 0.05e6, nonemb
    print(f"  [PASS] G_WHT orthogonality H16={e1:.1e} H32={e2:.1e}; "
          f"non-emb params G_WHT={nonemb['G_WHT']/1e6:.3f}M "
          f"DENSE_NARROW={nonemb['DENSE_NARROW']/1e6:.3f}M")

    torch.manual_seed(23)
    model = build_stage3("G_WHT", stage3_config("G_WHT", 1024)).eval()
    ids = tiny_ids(seed=6)
    t = 10
    emb = model.embed(ids).detach().clone().requires_grad_(True)
    logits = model(inputs_embeds=emb)
    weights = torch.randn(1024)
    (logits[0, t] * weights).sum().backward()
    grad = emb.grad[0, t + 1]
    assert grad is not None and grad.abs().max().item() == 0.0, \
        "G_WHT: future gradient is nonzero"
    assert emb.grad[0, t].abs().max().item() > 0.0
    ids2 = ids.clone()
    ids2[0, t + 1] = (ids2[0, t + 1] + 7) % 1024
    with torch.no_grad():
        out1, out2 = model(ids), model(ids2)
    assert torch.equal(out1[:, : t + 1], out2[:, : t + 1])
    print("  [PASS] G_WHT: exact causal gradient and bitwise-identical past logits")

    torch.manual_seed(29)
    model = build_stage3("G_WHT", stage3_config("G_WHT", 1024))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    ids = tiny_ids(seed=8)
    opt.zero_grad()
    loss_for(model, ids).backward()
    for pname, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), pname
    opt.step()
    opt.zero_grad()
    loss_for(model, ids).backward()
    zero = [n for n, p in model.named_parameters() if p.grad.abs().sum().item() == 0.0]
    assert not zero, f"G_WHT zero gradients after one step: {zero}"
    print(f"  [PASS] G_WHT: 2-phase gradient flow across {len(list(model.parameters()))} tensors")

    if torch.cuda.is_available():
        torch.manual_seed(31)
        model = build_stage3("G_WHT", stage3_config("G_WHT", 1024)).cuda()
        ids = tiny_ids(seed=4).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(ids)
            loss = F.cross_entropy(logits.float().reshape(-1, 1024), ids.reshape(-1))
        assert torch.isfinite(logits).all()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        for pname, param in model.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(), pname
        scaler.step(opt); scaler.update()
        print(f"  [PASS] G_WHT: CUDA FP16 finite, grad_scale={scaler.get_scale()}")
    else:
        print("  [SKIP] G_WHT: CUDA FP16 unavailable")


if __name__ == "__main__":
    print("Direction F/E/G mandatory CI")
    test_configs()
    test_params()
    test_families_data()
    for name in ("B2_THIN", "B2_NARROW", "F_VCYCLE"):
        print(f"=== {name} ===")
        test_causality(name)
        test_grad_flow(name)
        test_extrapolation(name)
        test_cuda_fp16(name)
    test_identity_at_init()
    test_stage3_g()
    print("\nALL EFG CI PASSED")
