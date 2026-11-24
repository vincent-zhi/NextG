from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import BASELINES, B2FlatLoop, LoopConfig, RMSNorm, run_stack


class BlockCayley(nn.Module):
    N_BASIS = 5

    def __init__(self, d: int, block_size: int = 32):
        super().__init__()
        if d % block_size != 0:
            raise ValueError(f"d={d} must be divisible by block_size={block_size}")
        self.d = d
        self.b = block_size
        self.n_blocks = d // block_size
        self.S = nn.Parameter(torch.zeros(self.N_BASIS, self.n_blocks, block_size, block_size))
        self.register_buffer("eye", torch.eye(block_size).unsqueeze(0), persistent=False)

    def _matrix(self, s: float, ds: float, device: torch.device) -> torch.Tensor:
        with torch.autocast(device_type=device.type, enabled=False):
            basis = torch.tensor(
                [1.0, s, ds, math.sin(math.pi * s), math.cos(math.pi * s)],
                device=device,
                dtype=torch.float32,
            )
            M = torch.einsum("m,mkij->kij", basis, self.S.float())
            A = M - M.transpose(-1, -2)
            I = self.eye.to(device=device, dtype=torch.float32)
            return torch.linalg.solve(I + A, I - A)

    def forward(self, x: torch.Tensor, s: float, ds: float) -> torch.Tensor:
        B, T, D = x.shape
        Q = self._matrix(s, ds, x.device).to(x.dtype)
        xb = x.reshape(B * T, self.n_blocks, self.b).transpose(0, 1)
        out = torch.bmm(xb, Q.transpose(-1, -2))
        return out.transpose(0, 1).contiguous().reshape(B, T, D)

    @torch.no_grad()
    def deviation_from_identity(self, s: float, ds: float, device: torch.device) -> float:
        Q = self._matrix(s, ds, device)
        I = self.eye.to(device=device, dtype=torch.float32)
        return (Q - I).pow(2).sum().sqrt().item() / math.sqrt(self.d)


class TrackBCayley(B2FlatLoop):
    name = "TB"

    def __init__(self, cfg: LoopConfig, block_size: int = 32):
        super().__init__(cfg)
        self.q_in = nn.ModuleList([BlockCayley(cfg.d, block_size) for _ in range(cfg.L_core)])
        self.q_out = nn.ModuleList([BlockCayley(cfg.d, block_size) for _ in range(cfg.L_core)])

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        return_hidden=False,
        N: int | None = None,
        loop_scale: float | None = None,
        reinject: bool | None = None,
        **_kwargs,
    ):
        N = self.cfg.N if N is None else N
        reinject = self.cfg.reinject if reinject is None else reinject
        loop_scale = self.loop_scale_value() if loop_scale is None else loop_scale

        h = self.embed_tokens(input_ids, inputs_embeds)
        h = run_stack(self.pre_layers, h)
        x_pre = h
        ds = 1.0 / max(1, N)
        for t in range(N):
            s = (t + 0.5) * ds
            inp = h + x_pre if reinject else h
            x = inp
            for l_idx, layer in enumerate(self.core_layers):
                xr = self.q_in[l_idx](x, s, ds)
                dx = layer(xr) - xr
                x = x + self.q_out[l_idx](dx, s, ds)
            h = h + loop_scale * (x - inp)
        h = run_stack(self.post_layers, h)
        return self._finish(h, return_hidden)

    @torch.no_grad()
    def rotation_stats(self, N: int | None = None) -> dict:
        N = self.cfg.N if N is None else N
        device = self.embed.weight.device
        ds = 1.0 / max(1, N)
        in_vals, out_vals = [], []
        for t in range(N):
            s = (t + 0.5) * ds
            in_vals.append(sum(m.deviation_from_identity(s, ds, device) for m in self.q_in) / len(self.q_in))
            out_vals.append(sum(m.deviation_from_identity(s, ds, device) for m in self.q_out) / len(self.q_out))
        return {
            "q_in_dev": [round(v, 4) for v in in_vals],
            "q_out_dev": [round(v, 4) for v in out_vals],
        }


class ChunkCausalTape(nn.Module):
    def __init__(self, d: int, d_m: int = 256, m: int = 8, n_heads: int = 4, chunk_size: int = 128):
        super().__init__()
        if d_m % n_heads != 0:
            raise ValueError(f"d_m={d_m} must be divisible by n_heads={n_heads}")
        self.d = d
        self.d_m = d_m
        self.m = m
        self.n_heads = n_heads
        self.d_h = d_m // n_heads
        self.chunk_size = chunk_size

        self.write_norm = RMSNorm(d)
        self.write_queries = nn.Parameter(torch.randn(m, d_m) * 0.02)
        self.write_kv_proj = nn.Linear(d, 2 * d_m, bias=False)

        self.read_norm = RMSNorm(d)
        self.read_q_proj = nn.Linear(d, d_m, bias=False)
        self.read_k_proj = nn.Linear(d_m, d_m, bias=False)
        self.read_v_proj = nn.Linear(d_m, d_m, bias=False)
        self.read_out_proj = nn.Linear(d_m, d, bias=False)

        self.decay_logit = nn.Parameter(torch.full((d_m,), 2.0))

        for lin in (self.write_kv_proj, self.read_q_proj, self.read_k_proj, self.read_v_proj):
            nn.init.normal_(lin.weight, std=0.02)
        nn.init.zeros_(self.read_out_proj.weight)

    def write(self, h: torch.Tensor) -> torch.Tensor | None:
        B, T, D = h.shape
        C = self.chunk_size
        K = T // C
        if K == 0:
            return None
        hc = self.write_norm(h[:, : K * C]).reshape(B, K, C, D)
        k, v = self.write_kv_proj(hc).chunk(2, dim=-1)
        BK = B * K
        k = k.reshape(BK, C, self.n_heads, self.d_h).transpose(1, 2)
        v = v.reshape(BK, C, self.n_heads, self.d_h).transpose(1, 2)
        q = (
            self.write_queries.to(dtype=k.dtype)
            .view(1, self.m, self.n_heads, self.d_h)
            .expand(BK, -1, -1, -1)
            .transpose(1, 2)
        )
        slots = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return slots.transpose(1, 2).contiguous().reshape(B, K, self.m, self.d_m)

    def read(self, h: torch.Tensor, tape_history: list[torch.Tensor]) -> torch.Tensor:
        if not tape_history:
            return torch.zeros_like(h)
        B, T, _ = h.shape
        C = self.chunk_size
        slots_per_chunk = len(tape_history) * self.m
        K = tape_history[0].shape[1]
        M = torch.stack(tape_history, dim=2).reshape(B, K * slots_per_chunk, self.d_m)

        q = self.read_q_proj(self.read_norm(h)).reshape(B, T, self.n_heads, self.d_h).transpose(1, 2)
        k = self.read_k_proj(M).reshape(B, K * slots_per_chunk, self.n_heads, self.d_h).transpose(1, 2)
        v = self.read_v_proj(M).reshape(B, K * slots_per_chunk, self.n_heads, self.d_h).transpose(1, 2)

        zero_slot = k.new_zeros(B, self.n_heads, 1, self.d_h)
        k = torch.cat([zero_slot, k], dim=2)
        v = torch.cat([zero_slot, v], dim=2)

        chunk_ends = (torch.arange(K, device=h.device) + 1) * C - 1
        slot_ends = chunk_ends.repeat_interleave(slots_per_chunk)
        pos = torch.arange(T, device=h.device)
        allow = pos[:, None] >= slot_ends[None, :]
        sink_col = torch.ones(T, 1, dtype=torch.bool, device=h.device)
        attn_mask = torch.cat([sink_col, allow], dim=1)[None, None, :, :]

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        out = out.transpose(1, 2).contiguous().reshape(B, T, self.d_m)
        return self.read_out_proj(out)


class TrackCTape(B2FlatLoop):
    name = "TC_TAPE"

    def __init__(self, cfg: LoopConfig, chunk_size: int = 128, d_m: int = 256, m: int = 8):
        super().__init__(cfg)
        self.tape = ChunkCausalTape(cfg.d, d_m=d_m, m=m, n_heads=4, chunk_size=chunk_size)

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        return_hidden=False,
        N: int | None = None,
        loop_scale: float | None = None,
        reinject: bool | None = None,
        tape_knockout: bool = False,
        **_kwargs,
    ):
        N = self.cfg.N if N is None else N
        reinject = self.cfg.reinject if reinject is None else reinject
        loop_scale = self.loop_scale_value() if loop_scale is None else loop_scale

        h = self.embed_tokens(input_ids, inputs_embeds)
        h = run_stack(self.pre_layers, h)
        x_pre = h
        tape_history: list[torch.Tensor] = []
        gamma = torch.sigmoid(self.tape.decay_logit).to(h.dtype)

        for t in range(N):
            if t > 0 and not tape_knockout and tape_history:
                h = h + loop_scale * self.tape.read(h, tape_history)
            inp = h + x_pre if reinject else h
            h = h + loop_scale * self.core_delta(inp)
            if t < N - 1 and not tape_knockout:
                new_slots = self.tape.write(h)
                if new_slots is not None:
                    if tape_history:
                        tape_history = [slot * gamma for slot in tape_history]
                    tape_history.append(new_slots)

        h = run_stack(self.post_layers, h)
        return self._finish(h, return_hidden)


def build_sprint(name: str, cfg: LoopConfig, tape_chunk_size: int = 128) -> nn.Module:
    if name == "TB":
        return TrackBCayley(cfg)
    if name == "TC_TAPE":
        return TrackCTape(cfg, chunk_size=tape_chunk_size)
    return BASELINES[name](cfg)
