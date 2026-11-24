"""
Phase 0 mandatory CI tests (plan section 2C), run on CPU at tiny scale.

Tiny test config (deviation from Tier 0/1 sizes, for CPU speed only):
  d=128, d_ff=352, heads=2 (d_h=64), L_pre/L_core/L_post = 2/2/2, V=1024, N=4.
The architecture code paths are identical to the Tier configs.

Run:  .venv/bin/python tests.py
"""

import math

import torch
import torch.nn.functional as F

from model import LoopConfig, build

torch.manual_seed(0)

TINY = LoopConfig(d=128, d_ff=352, L_pre=2, L_core=2, L_post=2, vocab_size=1024, N=4)
B, T = 2, 24


def tiny_ids(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, TINY.vocab_size, (B, T), generator=g)


# --------------------------------------------------------------------------- #
# Test 1: Bit-exact causality
# Perturb token x_{t+1}; grad of loss at position t wrt that token's embedding
# must be EXACTLY 0, and logits at positions <= t must be bitwise unchanged.
# --------------------------------------------------------------------------- #
def test_causality(name):
    model = build(name, TINY).eval()
    ids = tiny_ids()
    t = 10  # loss position; perturbed token is t+1 = 11

    # (a) autograd form of the plan's test: || d L_t / d x_{t+1} ||_inf == 0 exactly
    emb = model.embed(ids).detach().clone().requires_grad_(True)
    logits = model(inputs_embeds=emb)
    w = torch.randn(TINY.vocab_size)
    loss_t = (logits[0, t] * w).sum()
    loss_t.backward()
    g = emb.grad[0, t + 1]
    assert g is not None and g.abs().max().item() == 0.0, f"{name}: causality grad not exactly 0"
    # sanity: grad at position t itself must be non-zero (test is not vacuous)
    assert emb.grad[0, t].abs().max().item() > 0.0, f"{name}: grad at t unexpectedly 0"

    # (b) finite perturbation: change token id at t+1, logits at <= t bitwise equal
    ids2 = ids.clone()
    ids2[0, t + 1] = (ids2[0, t + 1] + 7) % TINY.vocab_size
    with torch.no_grad():
        l1 = model(ids)
        l2 = model(ids2)
    assert torch.equal(l1[:, : t + 1], l2[:, : t + 1]), f"{name}: logits <= t changed"
    assert not torch.equal(l1[:, t + 1], l2[:, t + 1]), f"{name}: perturbation had no effect at t+1"
    print(f"  [PASS] {name}: dL_t/dx_{{t+1}} exactly 0; logits<=t bitwise identical")


# --------------------------------------------------------------------------- #
# Test 2: Gradient flow
# Locked zero-init of final projections (plan 2B) makes grads into the
# pre-projection weights (q/k/v, gate/up, norms) EXACTLY zero on the very first
# backward by construction. So the test is two-phase:
#   Phase A (step 1 backward): every param has non-None, finite grad; the
#     zero-grad set must be exactly the expected pre-projection set, and the
#     final projections themselves (out_proj / down_proj) must get non-zero grad.
#   Phase B (after 1 optimizer step, fresh backward): EVERY parameter must now
#     have non-zero grad (projections have moved off zero, flow is unblocked).
# --------------------------------------------------------------------------- #
PRE_PROJ_MARKERS = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "norm1", "norm2",
                    "q_norm", "k_norm")


def _loss(model, ids):
    logits = model(ids[:, :-1])
    return F.cross_entropy(logits.reshape(-1, TINY.vocab_size), ids[:, 1:].reshape(-1))


def test_grad_flow(name):
    torch.manual_seed(7)
    model = build(name, TINY)
    ids = tiny_ids(seed=3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)

    # Phase A
    opt.zero_grad()
    loss = _loss(model, ids)
    loss.backward()
    zero_names, nonzero_names = [], []
    for n, p in model.named_parameters():
        assert p.grad is not None, f"{name}: {n} grad is None (phase A)"
        assert torch.isfinite(p.grad).all(), f"{name}: {n} grad not finite (phase A)"
        (zero_names if p.grad.abs().sum().item() == 0.0 else nonzero_names).append(n)
    unexpected = [n for n in zero_names if not any(m in n for m in PRE_PROJ_MARKERS)]
    assert not unexpected, f"{name}: unexpected zero-grad params in phase A: {unexpected}"
    missing = [n for n in zero_names if not ("out_proj" in n or "down_proj" in n)]
    assert len(missing) == len(zero_names), f"{name}: a final projection got zero grad in phase A"
    assert len(zero_names) > 0, f"{name}: expected some zero-init-blocked grads in phase A"

    # Phase B: one optimizer step, then a fresh backward must reach everything
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
    loss2 = _loss(model, ids)
    loss2.backward()
    still_zero = []
    for n, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"{name}: {n} bad grad (phase B)"
        if p.grad.abs().sum().item() == 0.0:
            still_zero.append(n)
    assert not still_zero, f"{name}: params still zero-grad after 1 step: {still_zero}"
    print(f"  [PASS] {name}: phase A zero-grad set = {len(zero_names)} pre-projection params "
          f"(as expected from zero-init); phase B: all {len(nonzero_names) + len(zero_names)} "
          f"params have non-zero finite grad")


# --------------------------------------------------------------------------- #
# Test 3: Identity-init equivalence
# Configuration used (documented precisely):
#   B2 with N=1, loop_scale overridden to exactly 1.0, and input re-injection
#   DISABLED (reinject=False). Then the loop update is
#       h <- h + 1.0 * (CoreStack(h) - h) = CoreStack(h),
#   i.e. exactly one single pass of the tied core. Weights are copied from B2
#   into a B1 (same shapes / same state-dict keys) via load_state_dict, so both
#   models hold bit-identical parameters; forwards must agree to < 1e-5.
# --------------------------------------------------------------------------- #
def test_identity_equivalence():
    torch.manual_seed(11)
    b2 = build("B2", TINY).eval()
    b1 = build("B1", TINY).eval()
    b1.load_state_dict(b2.state_dict())
    ids = tiny_ids(seed=5)
    with torch.no_grad():
        ref = b1(ids)
        out = b2(ids, N=1, loop_scale=1.0, reinject=False)
    diff = (ref - out).abs().max().item()
    assert diff < 1e-5, f"identity equivalence failed: max diff {diff}"
    print(f"  [PASS] B2(N=1, loop_scale=1.0, reinject=False) vs B1 single pass: "
          f"max|diff| = {diff:.3e} < 1e-5")


# --------------------------------------------------------------------------- #
# Test 4: Loop-extrapolation smoke test
# NOTE on vacuity: with the locked zero-init projections, a fresh B2 is the
# identity through every block, so this test would pass trivially. To actually
# stress loop dynamics we re-initialize the final projections (out_proj /
# down_proj) with std=0.02 Gaussians for this test only, then sweep N and
# record the RMS of the final hidden state (pre-head). Bounded criterion:
# every RMS finite and RMS(N) <= max(10 * RMS(N=1), RMS(N=1) + 5).
# --------------------------------------------------------------------------- #
def test_loop_extrapolation():
    torch.manual_seed(13)
    model = build("B2", TINY).eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "out_proj.weight" in n or "down_proj.weight" in n:
                p.normal_(std=0.02)
    ids = tiny_ids(seed=9)
    rms = {}
    with torch.no_grad():
        for N in (1, 2, 4, 8, 16, 32):
            _, h = model(ids, N=N, return_hidden=True)
            rms[N] = h.pow(2).mean().sqrt().item()
    base = rms[1]
    bound = max(10.0 * base, base + 5.0)
    for N, v in rms.items():
        assert math.isfinite(v) and v <= bound, f"RMS exploded at N={N}: {v} > {bound}"
    print(f"  [PASS] loop extrapolation RMS by N: " +
          ", ".join(f"N={N}: {v:.4f}" for N, v in rms.items()) +
          f"  (bound {bound:.4f})")
    return rms


# --------------------------------------------------------------------------- #
# Test 5: RMSNorm dtype preservation (round-2 audit regression test)
# The norm computes in fp32 internally but must return the INPUT dtype.
# The previous implementation promoted the tensor to fp32 before checking
# its dtype, so bf16 inputs leaked fp32 into the residual stream under
# autocast and this test failed on the dtype assert.
# --------------------------------------------------------------------------- #
def test_rmsnorm_dtype():
    from model import RMSNorm
    norm = RMSNorm(64)
    x = torch.randn(4, 16, 64, dtype=torch.bfloat16)
    out = norm(x)
    assert out.dtype == torch.bfloat16, f"RMSNorm leaked dtype: {out.dtype}"
    ref = norm(x.float())
    err = (out.float() - ref).abs().max().item()
    assert err < 0.05, f"RMSNorm bf16 vs fp32 reference drift too large: {err}"
    x32 = torch.randn(4, 16, 64, dtype=torch.float32)
    assert norm(x32).dtype == torch.float32
    print(f"  [PASS] RMSNorm returns input dtype (bf16 in -> bf16 out, "
          f"max err vs fp32 ref {err:.4f}); fp32 in -> fp32 out")


if __name__ == "__main__":
    print("Test 1: bit-exact causality")
    for name in ("B0", "B1", "B2"):
        test_causality(name)
    print("Test 2: gradient flow")
    for name in ("B0", "B1", "B2"):
        test_grad_flow(name)
    print("Test 3: identity-init equivalence")
    test_identity_equivalence()
    print("Test 4: loop-extrapolation smoke")
    test_loop_extrapolation()
    print("Test 5: RMSNorm dtype preservation")
    test_rmsnorm_dtype()
    print("\nALL TESTS PASSED")
