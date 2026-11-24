from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import (
    B0UntiedFLOP,
    B2FlatLoop,
    LoopConfig,
    RMSNorm,
    run_stack,
)


# --------------------------------------------------------------------------- #
# Direction F / E / G model configs (sprint handoff, Stage 1)
#
# All three share the same data mix (synthetic graph+ast, 60M tokens) and the
# same parameter/FLOP accounting conventions as the Phase-0 baselines.
# --------------------------------------------------------------------------- #
EFG_MODELS = ("B2_THIN", "B2_NARROW", "F_VCYCLE")


def efg_config(name: str, vocab_size: int) -> LoopConfig:
    if name == "B2_THIN":
        # Parameter/width control: same d as F, 1/4/0 unique layers, N=4.
        return LoopConfig(d=512, d_ff=1408, n_heads=8, L_pre=1, L_core=4,
                          L_post=0, vocab_size=vocab_size, N=4)
    if name == "B2_NARROW":
        # Compute-matched (deep-loop) control: half width, N=14 deep loop.
        return LoopConfig(d=256, d_ff=704, n_heads=4, L_pre=1, L_core=4,
                          L_post=0, vocab_size=vocab_size, N=14)
    if name == "F_VCYCLE":
        # Matryoshka multigrid: full width d=512 with nested d0=256 coarse
        # sub-network. cfg.N is the training coarse-loop count N_c.
        return LoopConfig(d=512, d_ff=1408, n_heads=8, L_pre=1, L_core=4,
                          L_post=0, vocab_size=vocab_size, N=10)
    raise ValueError(f"unknown EFG model {name!r}")


# --------------------------------------------------------------------------- #
# Coarse (Matryoshka) attention / FFN: shared weights sliced to top-left d0
# --------------------------------------------------------------------------- #
def _coarse_attn(attn, x: torch.Tensor, d0: int, n_heads_c: int, d_head: int):
    """Causal attention at width d0 using the top-left [d0, d0] slices.

    q/k/v/out projections reuse the first d0 input *and* d0 output rows/cols of
    the full-width weights. QK-Norm and RoPE are reused unchanged (d_head is
    identical between the full and coarse views)."""
    B, T, _ = x.shape
    q = F.linear(x, attn.q_proj.weight[:d0, :d0]).view(B, T, n_heads_c, d_head).transpose(1, 2)
    k = F.linear(x, attn.k_proj.weight[:d0, :d0]).view(B, T, n_heads_c, d_head).transpose(1, 2)
    v = F.linear(x, attn.v_proj.weight[:d0, :d0]).view(B, T, n_heads_c, d_head).transpose(1, 2)
    q = attn.q_norm(q)
    k = attn.k_norm(k)
    q = attn.rope(q)
    k = attn.rope(k)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = out.transpose(1, 2).contiguous().view(B, T, d0)
    return F.linear(out, attn.out_proj.weight[:d0, :d0])


def _coarse_ffn(ffn, x: torch.Tensor, d0: int, d_ff0: int):
    """SwiGLU at width d0 using gate/up[:d_ff0, :d0] and down[:d0, :d_ff0]."""
    g = F.linear(x, ffn.gate_proj.weight[:d_ff0, :d0])
    u = F.linear(x, ffn.up_proj.weight[:d_ff0, :d0])
    return F.linear(F.silu(g) * u, ffn.down_proj.weight[:d0, :d_ff0])


class FVCycle(B2FlatLoop):
    """Direction F: Matryoshka multigrid V-cycle in the tied core.

    Full-width state carries the fine residual stream. The first d0 channels
    form the coarse state that is looped N_c times through the shared core
    weights' top-left sub-network, then spliced back and given ONE full-width
    polish pass. Loop-muP scales are closed-form properties of the trained
    N_c (never re-derived from the eval N_c).
    """

    name = "F_VCYCLE"

    def __init__(self, cfg: LoopConfig):
        super().__init__(cfg)
        self.d0 = cfg.d // 2
        self.d_ff0 = cfg.d_ff // 2
        self.n_heads_c = self.d0 // cfg.d_head
        self.N_c_train = cfg.N
        self.N_f = 1
        assert self.d0 % cfg.d_head == 0, "coarse width must be a multiple of d_head"
        assert self.d_ff0 > 0
        # ONLY new parameters: dedicated coarse pre-norms (full norms are d-wide
        # and cannot be applied to the d0-wide coarse stream).
        self.coarse_norm1 = nn.ModuleList([RMSNorm(self.d0) for _ in range(cfg.L_core)])
        self.coarse_norm2 = nn.ModuleList([RMSNorm(self.d0) for _ in range(cfg.L_core)])
        # Closed-form Loop-muP scales, frozen at the trained N_c.
        self.scale_c = 1.0 / (self.N_c_train * math.sqrt(1.0 + self.N_c_train * cfg.L_core))
        self.scale_f = 1.0 / math.sqrt(1.0 + (self.N_c_train + 1) * cfg.L_core)

    # -- coarse core stack (width d0) ------------------------------------- #
    def _coarse_block(self, i: int, layer, x: torch.Tensor) -> torch.Tensor:
        x = x + _coarse_attn(layer.attn, self.coarse_norm1[i](x),
                             self.d0, self.n_heads_c, self.cfg.d_head)
        x = x + _coarse_ffn(layer.ffn, self.coarse_norm2[i](x), self.d0, self.d_ff0)
        return x

    def coarse_core_delta(self, inp: torch.Tensor) -> torch.Tensor:
        h = inp
        for i, layer in enumerate(self.core_layers):
            h = self._coarse_block(i, layer, h)
        return h - inp

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        return_hidden=False,
        N: int | None = None,
        loop_scale: float | None = None,
        reinject: bool | None = None,
    ):
        N_c = self.N_c_train if N is None else N

        h = self.embed_tokens(input_ids, inputs_embeds)
        h_pre = run_stack(self.pre_layers, h)            # full width (L_pre)
        x_c = h_pre[..., : self.d0]                      # coarse slice

        # (1) coarse deep loop with coarse re-injection of x_c
        h_c = x_c
        for _ in range(N_c):
            h_c = h_c + self.scale_c * self.coarse_core_delta(h_c + x_c)

        # (2) splice coarse state back into the fine residual stream
        h = torch.cat([h_c, h_pre[..., self.d0:]], dim=-1)

        # (3) single full-width polish pass with fine re-injection of x_pre
        for _ in range(self.N_f):
            inp = h + h_pre
            h = h + self.scale_f * (run_stack(self.core_layers, inp) - inp)

        h = run_stack(self.post_layers, h)               # L_post (0)
        return self._finish(h, return_hidden)


def build_efg(name: str, cfg: LoopConfig) -> nn.Module:
    if name in ("B2_THIN", "B2_NARROW"):
        return B2FlatLoop(cfg)
    if name == "F_VCYCLE":
        return FVCycle(cfg)
    raise ValueError(f"unknown EFG model {name!r}")


# --------------------------------------------------------------------------- #
# Direction G (Stage 3): holographic basis + native coefficient-space training
#
# Weight matrices are parameterised in the Sylvester-Hadamard eigenbasis:
#     W = H_512  blockdiag(C_1..C_4)  H_512,      H_512 = H_16 (x) H_32
# with H_512 symmetric unitary.  Only the 4 x 128 x 128 block-diagonal
# coefficients C are trainable and receive optimizer state.
# --------------------------------------------------------------------------- #
STAGE3_MODELS = ("DENSE_NARROW", "G_WHT")


def stage3_config(name: str, vocab_size: int) -> LoopConfig:
    if name == "DENSE_NARROW":
        return LoopConfig(d=256, d_ff=768, n_heads=4, L_pre=0, L_core=20,
                          L_post=0, vocab_size=vocab_size, N=1)
    if name == "G_WHT":
        return LoopConfig(d=512, d_ff=1536, n_heads=8, L_pre=0, L_core=20,
                          L_post=0, vocab_size=vocab_size, N=1)
    if name == "B0_FULL":
        return LoopConfig(d=512, d_ff=1408, n_heads=8, L_pre=2, L_core=4,
                          L_post=2, vocab_size=vocab_size, N=4)
    raise ValueError(f"unknown Stage-3 model {name!r}")


def _sylvester(n: int) -> torch.Tensor:
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(H.shape[0])


class HoloBasis(nn.Module):
    """Normalized Sylvester-Hadamard factorisation H_512 = H16 (x) H32.

    Zero parameters, zero gradients, zero optimizer state: the matrices are
    non-trainable buffers.  fwht is exact (orthogonal, self-inverse) and costs
    two tiny matmuls, 0.094x the FLOPs of a dense 512x512 projection."""

    def __init__(self, dim: int = 512, n1: int = 16, n2: int = 32):
        super().__init__()
        assert dim == n1 * n2
        self.dim, self.n1, self.n2 = dim, n1, n2
        self.register_buffer("H1", _sylvester(n1), persistent=False)
        self.register_buffer("H2", _sylvester(n2), persistent=False)

    def fwht(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(*shape[:-1], self.n1, self.n2)
        x = torch.matmul(x, self.H2.to(x.dtype))
        x = torch.matmul(self.H1.to(x.dtype), x)
        return x.reshape(*shape)


class HoloSquareProj(HoloBasis):
    """512 -> 512: W x = H (blockdiag(C) (H x)), C: (nb, b, b)."""

    def __init__(self, d: int = 512, b: int = 128, std: float | None = 0.04):
        super().__init__(d)
        assert d % b == 0
        self.nb, self.b = d // b, b
        C = torch.empty(self.nb, b, b)
        if std is None:  # zero-init final projection
            nn.init.zeros_(C)
        else:
            nn.init.normal_(C, std=std)
        self.C = nn.Parameter(C)

    def from_basis(self, hx: torch.Tensor) -> torch.Tensor:
        shape = hx.shape[:-1]
        z = hx.reshape(-1, self.nb, self.b)
        z = torch.einsum("nij,ijk->nik", z, self.C)
        return self.fwht(z.reshape(*shape, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.from_basis(self.fwht(x))


class HoloRectUp(HoloBasis):
    """512 -> 1536 = 3 chunks of 512, each H (blockdiag(C_j) (H x))."""

    def __init__(self, d: int = 512, d_out: int = 1536, b: int = 128,
                 std: float = 0.04):
        super().__init__(d)
        assert d_out % d == 0 and d % b == 0
        self.nb, self.b, self.chunks = d // b, b, d_out // d
        C = torch.empty(self.chunks, self.nb, b, b)
        nn.init.normal_(C, std=std)
        self.C = nn.Parameter(C)

    def from_basis(self, hx: torch.Tensor) -> torch.Tensor:
        shape = hx.shape[:-1]
        z = hx.reshape(-1, self.nb, self.b)
        outs = []
        for j in range(self.chunks):
            y = torch.einsum("nij,ijk->nik", z, self.C[j]).reshape(*shape, self.dim)
            outs.append(self.fwht(y))
        return torch.cat(outs, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.from_basis(self.fwht(x))


class HoloRectDown(HoloBasis):
    """1536 -> 512: per-chunk H, sum of chunk blockdiag maps, H."""

    def __init__(self, d: int = 512, d_in: int = 1536, b: int = 128):
        super().__init__(d)
        assert d_in % d == 0 and d % b == 0
        self.nb, self.b, self.chunks = d // b, b, d_in // d
        self.C = nn.Parameter(torch.zeros(self.chunks, self.nb, b, b))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:-1]
        xc = x.reshape(*shape, self.chunks, self.dim)
        hc = self.fwht(xc)                       # (..., chunks, 512)
        z = torch.zeros(*shape, self.dim, dtype=x.dtype, device=x.device)
        for j in range(self.chunks):
            z = z + torch.einsum("nij,ijk->nik",
                                 hc[..., j, :].reshape(-1, self.nb, self.b),
                                 self.C[j]).reshape(*shape, self.dim)
        return self.fwht(z)


class HoloAttention(nn.Module):
    def __init__(self, cfg: LoopConfig):
        super().__init__()
        assert cfg.d == 512 and cfg.d_head == 64
        self.n_heads, self.d_head = cfg.n_heads, cfg.d_head
        self.q_proj = HoloSquareProj(cfg.d, std=0.04)
        self.k_proj = HoloSquareProj(cfg.d, std=0.04)
        self.v_proj = HoloSquareProj(cfg.d, std=0.04)
        self.out_proj = HoloSquareProj(cfg.d, std=None)  # zero-init
        self.q_norm = RMSNorm(cfg.d_head)
        self.k_norm = RMSNorm(cfg.d_head)
        from model import RotaryEmbedding
        self.rope = RotaryEmbedding(cfg.d_head, base=cfg.rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        hx = self.q_proj.fwht(x)                 # shared by q, k, v
        q = self.q_proj.from_basis(hx).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj.from_basis(hx).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj.from_basis(hx).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q = self.rope(self.q_norm(q))
        k = self.rope(self.k_norm(k))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class HoloSwiGLU(nn.Module):
    def __init__(self, cfg: LoopConfig):
        super().__init__()
        assert cfg.d == 512 and cfg.d_ff == 1536
        self.gate_proj = HoloRectUp(cfg.d, cfg.d_ff, std=0.04)
        self.up_proj = HoloRectUp(cfg.d, cfg.d_ff, std=0.04)
        self.down_proj = HoloRectDown(cfg.d, cfg.d_ff)  # zero-init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = self.gate_proj.fwht(x)              # shared by gate, up
        g = self.gate_proj.from_basis(hx)
        u = self.up_proj.from_basis(hx)
        return self.down_proj(F.silu(g) * u)


class HoloBlock(nn.Module):
    def __init__(self, cfg: LoopConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d)
        self.attn = HoloAttention(cfg)
        self.norm2 = RMSNorm(cfg.d)
        self.ffn = HoloSwiGLU(cfg)

    def forward(self, x: torch.Tensor, residual_scale: float = 1.0) -> torch.Tensor:
        x = x + residual_scale * self.attn(self.norm1(x))
        x = x + residual_scale * self.ffn(self.norm2(x))
        return x


class GWhT(B0UntiedFLOP):
    """Direction G: 20 unique holographic layers (no tying, no loop)."""

    name = "G_WHT"

    def __init__(self, cfg: LoopConfig):
        super().__init__(cfg)
        self.layers = nn.ModuleList([HoloBlock(cfg) for _ in range(cfg.L_core)])


def build_stage3(name: str, cfg: LoopConfig) -> nn.Module:
    if name == "DENSE_NARROW":
        return B0UntiedFLOP(cfg)
    if name == "G_WHT":
        return GWhT(cfg)
    if name == "B0_FULL":
        from model import B0UntiedFLOP as _B0
        return _B0(cfg)
    raise ValueError(f"unknown Stage-3 model {name!r}")
