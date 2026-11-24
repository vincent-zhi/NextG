from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F

from data_pipeline import ASTGenerator, ByteTokenizer, Phase0Data
from model import LoopConfig
from tracks_sprint import build_sprint


TINY = LoopConfig(d=128, d_ff=352, L_pre=2, L_core=2, L_post=2, vocab_size=1024, N=4)
B, T = 2, 24


def tiny_ids(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, TINY.vocab_size, (B, T), generator=g)


def build_tiny(name):
    return build_sprint(name, TINY, tape_chunk_size=4)


def activate_track(model, name):
    with torch.no_grad():
        for param_name, param in model.named_parameters():
            if "out_proj.weight" in param_name or "down_proj.weight" in param_name:
                param.normal_(std=0.02)
        if name == "TB":
            for module in list(model.q_in) + list(model.q_out):
                module.S.normal_(std=0.02)
        elif name == "TC_TAPE":
            model.tape.read_out_proj.weight.normal_(std=0.02)


def test_causality(name):
    torch.manual_seed(17)
    model = build_tiny(name).eval()
    activate_track(model, name)
    ids = tiny_ids()
    t = 10

    emb = model.embed(ids).detach().clone().requires_grad_(True)
    logits = model(inputs_embeds=emb)
    weights = torch.randn(TINY.vocab_size)
    (logits[0, t] * weights).sum().backward()
    grad = emb.grad[0, t + 1]
    assert grad is not None and grad.abs().max().item() == 0.0, f"{name}: future gradient is nonzero"
    assert emb.grad[0, t].abs().max().item() > 0.0, f"{name}: causality check is vacuous"

    ids2 = ids.clone()
    ids2[0, t + 1] = (ids2[0, t + 1] + 7) % TINY.vocab_size
    with torch.no_grad():
        out1 = model(ids)
        out2 = model(ids2)
    assert torch.equal(out1[:, : t + 1], out2[:, : t + 1]), f"{name}: past logits changed"
    assert not torch.equal(out1[:, t + 1], out2[:, t + 1]), f"{name}: perturbation had no effect"
    print(f"  [PASS] {name}: exact causal gradient and bitwise-identical past logits")


def loss_for(model, ids):
    logits = model(ids[:, :-1])
    return F.cross_entropy(logits.reshape(-1, TINY.vocab_size), ids[:, 1:].reshape(-1))


def test_grad_flow(name):
    torch.manual_seed(7)
    model = build_tiny(name)
    ids = tiny_ids(seed=3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)

    opt.zero_grad()
    loss_for(model, ids).backward()
    zero_names = []
    for param_name, param in model.named_parameters():
        assert param.grad is not None, f"{name}: {param_name} has no gradient in phase A"
        assert torch.isfinite(param.grad).all(), f"{name}: {param_name} has a non-finite gradient"
        if param.grad.abs().sum().item() == 0.0:
            zero_names.append(param_name)

    opt.step()
    opt.zero_grad()
    loss_for(model, ids).backward()
    still_zero = []
    for param_name, param in model.named_parameters():
        assert param.grad is not None, f"{name}: {param_name} has no gradient in phase B"
        assert torch.isfinite(param.grad).all(), f"{name}: {param_name} has a non-finite gradient"
        if param.grad.abs().sum().item() == 0.0:
            still_zero.append(param_name)
    assert not still_zero, f"{name}: zero gradients after one optimizer step: {still_zero}"
    print(f"  [PASS] {name}: all {len(list(model.parameters()))} parameters receive non-zero finite gradients")


def test_identity(name):
    torch.manual_seed(11)
    reference = build_sprint("B2", TINY, tape_chunk_size=4).eval()
    model = build_tiny(name).eval()
    model.load_state_dict(reference.state_dict(), strict=False)
    ids = tiny_ids(seed=5)
    with torch.no_grad():
        ref_logits = reference(ids, N=4)
        logits = model(ids, N=4)
    diff = (ref_logits - logits).abs().max().item()
    assert diff < 1e-5, f"{name}: identity initialization differs by {diff}"
    print(f"  [PASS] {name}: B2 identity at N=4, max|diff|={diff:.3e}")


def test_extrapolation(name):
    torch.manual_seed(13)
    model = build_tiny(name).eval()
    activate_track(model, name)
    ids = tiny_ids(seed=9)
    rms = {}
    with torch.no_grad():
        for N in (1, 2, 4, 8, 16, 32):
            _, hidden = model(ids, N=N, return_hidden=True)
            rms[N] = hidden.pow(2).mean().sqrt().item()
    bound = max(10.0 * rms[1], rms[1] + 5.0)
    for N, value in rms.items():
        assert math.isfinite(value) and value <= bound, f"{name}: RMS exploded at N={N}: {value}"
    print(f"  [PASS] {name}: extrapolation RMS " + ", ".join(f"N={N}:{v:.4f}" for N, v in rms.items()))


def test_cuda_fp16(name):
    if not torch.cuda.is_available():
        print(f"  [SKIP] {name}: CUDA FP16 forward/backward unavailable on this host")
        return
    torch.manual_seed(19)
    model = build_tiny(name).cuda()
    activate_track(model, name)
    ids = tiny_ids(seed=4).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(ids)
        loss = F.cross_entropy(logits.float().reshape(-1, TINY.vocab_size), ids.reshape(-1))
    assert torch.isfinite(logits).all(), f"{name}: non-finite FP16 logits"
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    for param_name, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), \
            f"{name}: bad FP16 gradient for {param_name}"
    scaler.step(opt)
    scaler.update()
    assert math.isfinite(scaler.get_scale()) and scaler.get_scale() > 0, f"{name}: invalid grad scale"
    print(f"  [PASS] {name}: CUDA FP16 forward/backward finite, grad_scale={scaler.get_scale()}")


def test_synthetic_only_data():
    class BombCorpus:
        def next_text(self, _rng):
            raise AssertionError("synthetic-only data touched the text corpus")

    tok = ByteTokenizer()
    data = Phase0Data(seed=42, tokenizer=tok, text_corpus=BombCorpus(),
                      synthetic_fraction=1.0, seq_len=64, families=("ast",))
    ids, targets, mask, syn = data.next_batch(4)
    assert ids.shape == (4, 64) and torch.all(syn == 1)
    assert not torch.any((targets == tok.pad_id) & (mask == 1)), "padding received supervised loss"
    state = data.state_dict()
    expected = data.next_batch(2)
    restored = Phase0Data(seed=42, tokenizer=tok, text_corpus=None,
                          synthetic_fraction=1.0, seq_len=64, families=("ast",))
    restored.load_state_dict(state)
    actual = restored.next_batch(2)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual)), "synthetic-only resume diverged"

    generator = ASTGenerator(random.Random(9), ood=True)
    longest = 0
    for _ in range(20):
        example = generator._dyck()
        longest = max(longest, len(tok.encode(example.prompt + example.answer)) + 1)
    assert longest <= 1024, f"OOD Dyck example exceeds seq_len=1024: {longest}"
    print(f"  [PASS] synthetic-only stream never touches text; padding unsupervised; OOD Dyck max={longest}")


if __name__ == "__main__":
    print("Sprint mandatory CI")
    test_synthetic_only_data()
    for name in ("TB", "TC_TAPE"):
        print(f"=== {name} ===")
        test_causality(name)
        test_grad_flow(name)
        test_identity(name)
        test_extrapolation(name)
        test_cuda_fp16(name)
    print("\nALL SPRINT CI PASSED")
