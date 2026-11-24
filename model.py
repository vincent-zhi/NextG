"""
Shared Pre-Norm backbone + locked baselines B0 / B1 / B2 for the looped-Transformer
experiment plan (Phase 0 harness). Implements plan section 3A exactly:

  - Pre-Norm Transformer, RMSNorm, RoPE (base 100K), SwiGLU FFN, QK-Norm
  - Tied input/output embeddings, V = 32768 by default
  - No biases anywhere (required to reproduce the plan's parameter table)
  - Zero-initialized final projections of every attention / FFN sub-block
    (plan section 2B, Loop-muP floor)

Baselines:
  - B0 Untied-FLOP : L_eff unique dense layers, single pass (no tying, no loop)
  - B1 Untied-Param: L_pre + L_core + L_post unique dense layers, single pass
  - B2 Flat-Loop   : weight-tied core looped N times with input re-injection
        h <- h + (1 / (N * sqrt(L_eff))) * f_core(h, x_pre)
      where f_core(h, x_pre) is the *residual delta* of the tied core stack
      evaluated at (h + x_pre), i.e. CoreStack(h + x_pre) - (h + x_pre), and
      x_pre is the hidden state after the pre-layers. L_eff = L_pre + N*L_core + L_post.

ATTENTION SUBSTITUTION NOTE: the plan specifies FlashAttention-2, which is
GPU-only. We use PyTorch SDPA (torch.nn.functional.scaled_dot_product_attention)
with is_causal=True. This is mathematically equivalent (causal self-attention):
masked positions get exactly 0 attention weight, which preserves the bit-exact
causality property the plan's CI test 1 requires (re-verified after the switch).
An earlier version passed an explicit additive -inf mask; that forced SDPA
onto its math fallback backend (materialising the full T x T attention matrix
and rebuilding the mask on every layer/loop iteration). is_causal=True lets
SDPA pick its fused flash / memory-efficient kernels on GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class LoopConfig:
    d: int = 512
    d_ff: int = 1408
    n_heads: int = 0  # 0 -> d // 64 (plan: head dim d_h = 64)
    L_pre: int = 2
    L_core: int = 4
    L_post: int = 2
    vocab_size: int = 32768
    N: int = 4  # training loop count for B2; also defines L_eff for B0
    rope_base: float = 100_000.0
    # B2 loop controls (defaults = locked plan behaviour)
    reinject: bool = True  # input re-injection of x_pre at every loop iteration
    loop_scale: float | None = None  # None -> 1 / (N * sqrt(L_eff)); set to override

    def __post_init__(self):
        if self.n_heads == 0:
            assert self.d % 64 == 0, "default head sizing needs d divisible by 64"
            self.n_heads = self.d // 64
        assert self.d % self.n_heads == 0

    @property
    def d_head(self) -> int:
        return self.d // self.n_heads

    @property
    def L_unique(self) -> int:
        return self.L_pre + self.L_core + self.L_post

    def L_eff(self, N: int | None = None) -> int:
        N = self.N if N is None else N
        return self.L_pre + N * self.L_core + self.L_post


# Tier configs from plan section 3A
def tier0_config(**overrides) -> LoopConfig:
    cfg = LoopConfig(d=512, d_ff=1408, L_pre=2, L_core=4, L_post=2, N=4)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def tier1_config(**overrides) -> LoopConfig:
    cfg = LoopConfig(d=896, d_ff=2432, L_pre=2, L_core=8, L_post=2, N=4)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 but ALWAYS return the input dtype. The previous
        # version promoted x to fp32 before checking x.dtype, so the cast-back
        # branch never fired and bf16 inputs leaked fp32 into the residual
        # stream under autocast (audit fix, round 2).
        orig_dtype = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        xf = xf * torch.rsqrt(var + 1e-6)
        return (xf * self.weight.float()).to(orig_dtype)


class RotaryEmbedding(nn.Module):
    """RoPE with base 100K (plan section 3A). No parameters."""

    def __init__(self, d_head: int, base: float = 100_000.0, max_seq: int = 8192):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        t = torch.arange(max_seq).float()
        freqs = torch.outer(t, inv_freq)  # (T, d_head/2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, T, d_head)
        T = x.shape[2]
        cos = self.cos[:T].to(x.dtype)  # (T, d_head/2)
        sin = self.sin[:T].to(x.dtype)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rot1 = x1 * cos - x2 * sin
        rot2 = x1 * sin + x2 * cos
        out = torch.stack([rot1, rot2], dim=-1).flatten(-2)
        return out.type_as(x)


def causal_mask(T: int, device, dtype) -> torch.Tensor:
    """Additive causal mask: 0 allowed, -inf masked. Exact zeros after softmax.

    Retained for debugging / math-backend comparisons; Attention.forward uses
    SDPA is_causal=True instead (fused kernels, no materialised mask)."""
    mask = torch.full((T, T), float("-inf"), device=device, dtype=dtype)
    return torch.triu(mask, diagonal=1)


class Attention(nn.Module):
    def __init__(self, cfg: LoopConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.q_proj = nn.Linear(cfg.d, cfg.d, bias=False)
        self.k_proj = nn.Linear(cfg.d, cfg.d, bias=False)
        self.v_proj = nn.Linear(cfg.d, cfg.d, bias=False)
        self.out_proj = nn.Linear(cfg.d, cfg.d, bias=False)
        self.q_norm = RMSNorm(cfg.d_head)  # QK-Norm, per-head
        self.k_norm = RMSNorm(cfg.d_head)
        self.rope = RotaryEmbedding(cfg.d_head, base=cfg.rope_base)
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.weight)  # zero-init final projection (plan 2B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = self.rope(q)
        k = self.rope(k)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: LoopConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d, bias=False)
        nn.init.normal_(self.gate_proj.weight, std=0.02)
        nn.init.normal_(self.up_proj.weight, std=0.02)
        nn.init.zeros_(self.down_proj.weight)  # zero-init final projection (plan 2B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    """Pre-Norm block: x += scale * Attn(Norm(x)); x += scale * FFN(Norm(x))."""

    def __init__(self, cfg: LoopConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, residual_scale: float = 1.0) -> torch.Tensor:
        x = x + residual_scale * self.attn(self.norm1(x))
        x = x + residual_scale * self.ffn(self.norm2(x))
        return x


def run_stack(layers: nn.ModuleList, x: torch.Tensor, residual_scale: float = 1.0) -> torch.Tensor:
    for layer in layers:
        x = layer(x, residual_scale=residual_scale)
    return x


# --------------------------------------------------------------------------- #
# Base LM: tied embeddings + final norm + head
# --------------------------------------------------------------------------- #
class BaseLM(nn.Module):
    def __init__(self, cfg: LoopConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.final_norm = RMSNorm(cfg.d)
        # lm_head is tied: implemented as F.linear(h, embed.weight), no extra params.

    def embed_tokens(self, input_ids: torch.Tensor | None, inputs_embeds: torch.Tensor | None):
        if inputs_embeds is not None:
            return inputs_embeds
        return self.embed(input_ids)

    def head(self, h: torch.Tensor) -> torch.Tensor:
        return F.linear(self.final_norm(h), self.embed.weight)

    def _finish(self, h, return_hidden: bool):
        logits = self.head(h)
        if return_hidden:
            return logits, h
        return logits


# --------------------------------------------------------------------------- #
# B1 Untied-Param: L_pre + L_core + L_post unique layers, single pass
# --------------------------------------------------------------------------- #
class B1UntiedParam(BaseLM):
    name = "B1"

    def __init__(self, cfg: LoopConfig):
        super().__init__(cfg)
        self.pre_layers = nn.ModuleList([Block(cfg) for _ in range(cfg.L_pre)])
        self.core_layers = nn.ModuleList([Block(cfg) for _ in range(cfg.L_core)])
        self.post_layers = nn.ModuleList([Block(cfg) for _ in range(cfg.L_post)])

    def forward(self, input_ids=None, inputs_embeds=None, return_hidden=False, **_):
        h = self.embed_tokens(input_ids, inputs_embeds)
        h = run_stack(self.pre_layers, h)
        h = run_stack(self.core_layers, h)
        h = run_stack(self.post_layers, h)
        return self._finish(h, return_hidden)


# --------------------------------------------------------------------------- #
# B0 Untied-FLOP: L_eff unique dense layers, single pass
# --------------------------------------------------------------------------- #
class B0UntiedFLOP(BaseLM):
    name = "B0"

    def __init__(self, cfg: LoopConfig):
        super().__init__(cfg)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.L_eff())])

    def forward(self, input_ids=None, inputs_embeds=None, return_hidden=False, **_):
        h = self.embed_tokens(input_ids, inputs_embeds)
        h = run_stack(self.layers, h)
        return self._finish(h, return_hidden)


# --------------------------------------------------------------------------- #
# B2 Flat-Loop: tied core looped N times, input re-injection, Loop-muP scaling
# --------------------------------------------------------------------------- #
class B2FlatLoop(BaseLM):
    name = "B2"

    def __init__(self, cfg: LoopConfig):
        super().__init__(cfg)
        self.pre_layers = nn.ModuleList([Block(cfg) for _ in range(cfg.L_pre)])
        self.core_layers = nn.ModuleList([Block(cfg) for _ in range(cfg.L_core)])
        self.post_layers = nn.ModuleList([Block(cfg) for _ in range(cfg.L_post)])

    def loop_scale_value(self, N: int | None = None) -> float:
        # Locked extrapolation protocol (audit fix, round 2): the scale is a
        # property of the TRAINED model, computed from the training loop
        # count cfg.N. Iterating more/fewer loops at eval time must not
        # change the update rule, otherwise an extrapolation curve measures a
        # family of differently-scaled models instead of one model iterated.
        if self.cfg.loop_scale is not None:
            return self.cfg.loop_scale
        n_train = self.cfg.N
        return 1.0 / (n_train * math.sqrt(self.cfg.L_eff(n_train)))

    def core_delta(self, inp: torch.Tensor) -> torch.Tensor:
        """f_core residual delta: CoreStack(inp) - inp (branches at scale 1.0;
        the Loop-muP factor is applied once per loop iteration, per plan 3A)."""
        return run_stack(self.core_layers, inp) - inp

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        return_hidden=False,
        N: int | None = None,
        loop_scale: float | None = None,
        reinject: bool | None = None,
    ):
        N = self.cfg.N if N is None else N
        reinject = self.cfg.reinject if reinject is None else reinject
        if loop_scale is None:
            loop_scale = self.loop_scale_value()

        h = self.embed_tokens(input_ids, inputs_embeds)
        h = run_stack(self.pre_layers, h)
        x_pre = h
        for _ in range(N):
            inp = h + x_pre if reinject else h
            h = h + loop_scale * self.core_delta(inp)
        h = run_stack(self.post_layers, h)
        return self._finish(h, return_hidden)


BASELINES = {"B0": B0UntiedFLOP, "B1": B1UntiedParam, "B2": B2FlatLoop}


def build(name: str, cfg: LoopConfig) -> nn.Module:
    return BASELINES[name](cfg)
