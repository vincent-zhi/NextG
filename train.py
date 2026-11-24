"""
Phase 0 training entry for the locked baselines B0 / B1 / B2 (plan 1, 2B, 3C).

Locked Loop-muP protocol (plan 2B), no per-run deviations:
  * AdamW, beta1=0.9, beta2=0.95, weight decay 0.1 on >=2D weights ONLY
    (1D Norm gains get 0.0 - round-2 audit fix), gradient clip 1.0
  * Cosine schedule with 5% warmup decaying to a 10% LR floor
  * Batch size: 256K tokens/step at Tier 0 (262144); 512K at Tier 1
  * Seed 42 first; the only sanctioned variation is the 3-point LR sweep
    {0.5x, 1.0x, 2.0x}, run SEPARATELY per model (B1 and B2 get their own
    sweeps - B2's Loop-muP residual scale makes B1's LR non-transferable).

Data: data_pipeline.Phase0Data (locked 80/20 FineWeb-Edu / synthetic token
mix, answer-only loss on synthetic examples per plan 4C). Evaluation uses a
FIXED batch set drawn once from a document-disjoint holdout corpus (2% of
documents), so every eval measures the same items, before and after resume.

Batch geometry on a 24GB A10 (Tier 0, d=512, L_eff=20, seq 2048):
  micro-batch = 8 sequences x 2048 tokens = 16K tokens per forward/backward;
  grad accumulation 262144 / 16384 = 16 micro-steps gives exactly the locked
  256K-token batch. Loss uses F.cross_entropy(ignore_index=-100,
  reduction="sum") - no materialised fp32 logits copy - and each step's
  gradient is normalised by the step's TOTAL supervised-token count, so the
  optimised loss and the logged loss are the same quantity.

Checkpoints: every --ckpt-every steps (and at the end) a full training
state is written to <ckpt-dir>/step-XXXXXXX.pt plus <ckpt-dir>/latest.pt:
model, optimizer, step, tokens seen, CPU + CUDA RNG states, and the
complete data-stream state, so --resume continues the identical token
stream. Checkpoints are always loaded with map_location="cpu" first
(round-2 audit fix: loading straight onto CUDA crashed set_rng_state).

Usage:
  # LR sweep point (short), per model: B2 at 1x for 0.15B tokens
  python train.py --model B2 --lr 3e-3 --max-tokens 150000000 \
      --data local --text-dir fw_text --tokenizer bpe \
      --tokenizer-file tokenizer.json --ckpt-dir runs/b2_lr2
  # Full Tier-0 run: 1.5B tokens
  python train.py --model B2 --lr 3e-3 --max-tokens 1500000000 ...
  # Resume after a session ends (instances are time-limited):
  python train.py --model B2 --lr 3e-3 --max-tokens 1500000000 \
      --ckpt-dir runs/b2_lr2 --resume
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
import shutil
import threading
import time

import torch
import torch.nn.functional as F

from data_pipeline import (ASTGenerator, EntityKBGenerator,
                           GraphSatGenerator, Phase0Data, SYNTHETIC_FAMILIES,
                           VTrackingGenerator, get_tokenizer, make_eval_data,
                           make_text_corpus)
from model import LoopConfig, tier0_config
from tracks_efg import (EFG_MODELS, STAGE3_MODELS, build_efg,
                        build_stage3, efg_config, stage3_config)
from tracks_sprint import build_sprint

BASE_LR_DEFAULT = 3e-3  # sweep centre; {0.5x, 1.0x, 2.0x} per model (plan 2B)


def parse_tokens(s) -> int:
    """Accept 1500000000, 1.5e9, 1500M, 0.15B, ..."""
    if isinstance(s, (int, float)):
        return int(s)
    s = str(s).strip().lower()
    mult = 1
    if s.endswith("b"):
        mult, s = 1_000_000_000, s[:-1]
    elif s.endswith("m"):
        mult, s = 1_000_000, s[:-1]
    elif s.endswith("k"):
        mult, s = 1_000, s[:-1]
    return int(float(s) * mult)


def lr_at(step: int, total_steps: int, base_lr: float,
          warmup_frac: float = 0.05, min_frac: float = 0.1) -> float:
    """Plan 2B: linear warmup for 5% of the run, then cosine decay from
    base_lr to a floor of 10% of base_lr."""
    warmup = max(1, int(warmup_frac * total_steps))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total_steps - warmup)
    p = min(1.0, p)
    cos = 0.5 * (1.0 + math.cos(math.pi * p))
    return base_lr * (min_frac + (1.0 - min_frac) * cos)


def _ctx(amp_dtype: torch.dtype | None):
    if amp_dtype is None:
        return torch.autocast("cpu", enabled=False)
    return torch.autocast("cuda", dtype=amp_dtype)


def chunked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                          reduction: str = "sum", chunk_tokens: int = 2048):
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    if reduction == "sum":
        total = torch.zeros((), device=logits.device, dtype=torch.float32)
        for start in range(0, flat_targets.numel(), chunk_tokens):
            end = start + chunk_tokens
            total = total + F.cross_entropy(
                flat_logits[start:end].float(), flat_targets[start:end],
                ignore_index=-100, reduction="sum")
        return total
    losses = torch.empty(flat_targets.shape, device=logits.device, dtype=torch.float32)
    for start in range(0, flat_targets.numel(), chunk_tokens):
        end = start + chunk_tokens
        losses[start:end] = F.cross_entropy(
            flat_logits[start:end].float(), flat_targets[start:end],
            ignore_index=-100, reduction="none")
    return losses.view_as(targets)


def train_loss_sum(model, batch, device, amp_dtype: torch.dtype | None,
                   N: int | None = None):
    ids, targets, mask, _syn = (t.to(device, non_blocking=True)
                                for t in batch)
    ntok = int(batch[2].sum())
    tgt = targets.masked_fill(mask == 0, -100)
    with _ctx(amp_dtype):
        logits = model(ids) if N is None else model(ids, N=N)
    return chunked_cross_entropy(logits, tgt, reduction="sum"), ntok


class _Prefetcher:
    """One-batch-lookahead producer thread for the training data stream.

    A worker thread calls data.next_batch() ahead of the training loop so
    Python-side example generation / tokenisation overlaps GPU compute.
    A single producer keeps stream order and RNG consumption unchanged.
    Produced batches are pinned for async H2D copies.

    Checkpoint exactness: before producing each batch the worker snapshots
    data.state_dict(); the snapshot travels with the batch. The state that
    resumes the stream exactly after the last CONSUMED batch is the snapshot
    attached to the next unconsumed batch, exposed via resume_state()."""

    def __init__(self, data: Phase0Data, micro_batch: int):
        self._data = data
        self._micro_batch = micro_batch
        self._q: "queue.Queue" = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._pending = None  # (batch, pre_state) fetched, not yet consumed
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                pre_state = self._data.state_dict()
                batch = self._data.next_batch(self._micro_batch)
                if torch.cuda.is_available():
                    batch = tuple(t.pin_memory() for t in batch)
            except Exception as e:  # surfaced in the main thread by _fetch
                self._error = e
                self._q.put(None)
                return
            self._q.put((batch, pre_state))

    def _fetch(self):
        if self._pending is None:
            item = self._q.get()
            if item is None:
                raise RuntimeError("data producer failed") from self._error
            self._pending = item

    def next(self):
        self._fetch()
        batch, _ = self._pending
        self._pending = None
        return batch

    def resume_state(self) -> dict:
        self._fetch()
        return self._pending[1]

    def close(self):
        self._stop.set()


@torch.no_grad()
def evaluate_split(model, eval_batches, device, amp_dtype: torch.dtype | None,
                   byte_len: torch.Tensor) -> dict:
    """Fixed-set evaluation with split metrics (round-2 audit fix):
    FineWeb NLL + BPB on holdout text tokens, synthetic-answer NLL on
    synthetic supervised tokens, and the combined supervised mean."""
    model.eval()
    fw_sum = syn_sum = 0.0
    fw_cnt = syn_cnt = 0
    fw_bytes = 0
    bl = byte_len.to(device)
    for batch in eval_batches:
        ids, targets, mask, syn = (t.to(device, non_blocking=True)
                                   for t in batch)
        with _ctx(amp_dtype):
            logits = model(ids)
        losses = chunked_cross_entropy(logits, targets, reduction="none")
        fw_m = (mask == 1) & (syn == 0)
        syn_m = (mask == 1) & (syn == 1)
        fw_sum += (losses * fw_m).sum().item()
        syn_sum += (losses * syn_m).sum().item()
        fw_cnt += int(fw_m.sum())
        syn_cnt += int(syn_m.sum())
        fw_bytes += int(bl[targets[fw_m]].sum())
    model.train()
    return {
        "eval_loss": (fw_sum + syn_sum) / max(1, fw_cnt + syn_cnt),
        "fw_nll": fw_sum / max(1, fw_cnt),
        "fw_bpb": fw_sum / max(1, fw_bytes) / math.log(2),
        "syn_nll": syn_sum / max(1, syn_cnt),
        "fw_tokens": fw_cnt,
        "syn_tokens": syn_cnt,
    }


def token_byte_lengths(tok) -> torch.Tensor:
    """UTF-8 byte length of each token id (EOS counts 0). For byte-level
    BPE, per-token decoded lengths sum exactly to the text byte count."""
    lens = [0 if i == tok.eos_id
            else len(tok.decode([i]).encode("utf-8", "ignore"))
            for i in range(tok.vocab_size)]
    return torch.tensor(lens, dtype=torch.long)


def build_probes(n_in: int = 16, n_ood: int = 8, families=None):
    """Fixed synthetic probe set for exact-match accuracy (in-distribution
    + OOD dial), generated once from dedicated seeds."""
    gens = {
        "vtrack": VTrackingGenerator,
        "ast": ASTGenerator,
        "graph": GraphSatGenerator,
        "kb": EntityKBGenerator,
    }
    if families is not None:
        gens = {k: v for k, v in gens.items() if k in families}
    probes = []
    for fam, cls in gens.items():
        g_in = cls(__import__("random").Random(777))
        g_ood = cls(__import__("random").Random(888), ood=True)
        for _ in range(n_in):
            ex = g_in.sample()
            probes.append((ex.prompt, ex.answer, fam, False))
        for _ in range(n_ood):
            ex = g_ood.sample()
            probes.append((ex.prompt, ex.answer, fam, True))
    return probes


@torch.no_grad()
def probe_accuracy(raw_model, probes, tok, device, amp_dtype: torch.dtype | None,
                   max_new: int = 40) -> dict:
    """Greedy-decode each probe prompt on the RAW (uncompiled) model and
    score exact match per family, in-distribution and OOD."""
    raw_model.eval()
    stats: dict = {}
    for prompt, answer, fam, ood in probes:
        x = torch.tensor([tok.encode(prompt)], device=device)
        gen: list[int] = []
        with _ctx(amp_dtype):
            for _ in range(max_new):
                logits = raw_model(x)
                nxt = int(logits[0, -1].argmax())
                if nxt == tok.eos_id:
                    break
                gen.append(nxt)
                x = torch.cat([x, torch.tensor([[nxt]], device=device)],
                              dim=1)
        pred = tok.decode(gen).strip()
        key = f"{fam}{'_ood' if ood else ''}"
        c, n = stats.get(key, (0, 0))
        stats[key] = (c + int(pred == answer.strip()), n + 1)
    raw_model.train()
    out = {k: round(c / max(1, n), 3) for k, (c, n) in stats.items()}
    in_c = sum(c for k, (c, n) in stats.items() if not k.endswith("_ood"))
    in_n = sum(n for k, (c, n) in stats.items() if not k.endswith("_ood"))
    out["all"] = round(in_c / max(1, in_n), 3)
    return out


def save_checkpoint(path: str, raw_model, opt, step: int, tokens_seen: int,
                    data_state: dict, args, scaler, loop_rng_state) -> None:
    state = {
        "model": raw_model.state_dict(),
        "optimizer": opt.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available() else None,
        "data_state": data_state,
        "scaler": scaler.state_dict(),
        "loop_rng": loop_rng_state,
        "args": {k: (str(v) if isinstance(v, (os.PathLike,)) else v)
                 for k, v in vars(args).items()},
    }
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)
    latest = os.path.join(os.path.dirname(path), "latest.pt")
    try:
        os.remove(latest)
    except FileNotFoundError:
        pass
    try:
        os.link(path, latest)  # hardlink: instant, no second full copy
    except OSError:
        shutil.copyfile(path, latest)


def main():
    ap = argparse.ArgumentParser(description="Loop-Transformer baseline and sprint training")
    ap.add_argument("--model", required=True,
                    choices=["B0", "B1", "B2", "TB", "TC_TAPE", *EFG_MODELS,
                             *STAGE3_MODELS])
    ap.add_argument("--lr", type=float, default=BASE_LR_DEFAULT,
                    help="peak LR; sweep {0.5x, 1.0x, 2.0x} per model (plan 2B)")
    ap.add_argument("--max-tokens", type=parse_tokens, default=1_500_000_000,
                    help="token budget for this run, e.g. 1.5B or 0.15B for sweep runs")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="optional fixed step budget for smoke tests and exact resume checks")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-tokens", type=parse_tokens, default=262_144,
                    help="locked global batch: 256K tokens/step at Tier 0 (plan 2B)")
    ap.add_argument("--micro-batch", type=int, default=8,
                    help="sequences per forward/backward (8 x 2048 fits a 24GB A10)")
    ap.add_argument("--synthetic-fraction", type=float, default=0.2)
    ap.add_argument("--families", default=",".join(SYNTHETIC_FAMILIES),
                    help="comma-separated synthetic families, e.g. graph,ast "
                         "(default: all four)")
    ap.add_argument("--tape-chunk-size", type=int, default=128)
    ap.add_argument("--rand-loops", action="store_true",
                    help="sample N in (2, 4, 6) per optimizer step for B2/TB")
    ap.add_argument("--grad-accum", type=int, default=None,
                    help="default: batch_tokens / (micro_batch * seq_len)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", choices=["fineweb", "local", "fallback"],
                    default="fineweb")
    ap.add_argument("--text-dir", default=None)
    ap.add_argument("--tokenizer", choices=["byte", "bpe"], default="bpe")
    ap.add_argument("--tokenizer-file", default=None)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--probe-every", type=int, default=5,
                    help="run exact-match probes every Nth eval (and at the end)")
    ap.add_argument("--log-file", default=None)
    ap.add_argument("--resume", nargs="?", const="latest", default=None,
                    help="resume from <ckpt-dir>/latest.pt (or an explicit path)")
    ap.add_argument("--no-prefetch", action="store_true",
                    help="disable the data producer thread (debugging)")
    ap.add_argument("--no-compile", action="store_true",
                    help="disable torch.compile (debugging)")
    ap.add_argument("--tiny", action="store_true",
                    help="smoke-sized config mirroring smoke_train.py (CPU tests)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    device_obj = torch.device(device)
    is_cuda = device_obj.type == "cuda"
    if not 0.0 <= args.synthetic_fraction <= 1.0:
        raise SystemExit("--synthetic-fraction must be between 0 and 1")
    families = tuple(s.strip() for s in args.families.split(",") if s.strip())
    if not families or any(f not in SYNTHETIC_FAMILIES for f in families):
        raise SystemExit(f"--families must be a subset of {SYNTHETIC_FAMILIES}")
    if args.rand_loops and args.model not in ("B2", "TB"):
        raise SystemExit("--rand-loops is only supported for B2 and TB")
    capability = torch.cuda.get_device_capability(device_obj) if is_cuda else (0, 0)
    is_bf16_ok = is_cuda and (torch.version.hip is not None or capability[0] >= 8)
    amp_dtype = torch.bfloat16 if is_bf16_ok else (torch.float16 if is_cuda else None)
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda and not is_bf16_ok)
    if is_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    loop_rng = random.Random(args.seed + 1009)

    tok = get_tokenizer(args.tokenizer, args.tokenizer_file)
    if args.model in EFG_MODELS:
        cfg = efg_config(args.model, tok.vocab_size)
    elif args.model in STAGE3_MODELS:
        cfg = stage3_config(args.model, tok.vocab_size)
    elif args.tiny:
        cfg = LoopConfig(d=128, d_ff=352, L_pre=1, L_core=2, L_post=1,
                         vocab_size=tok.vocab_size, N=2)
    else:
        cfg = tier0_config(vocab_size=tok.vocab_size)
        if args.tokenizer != "bpe" or tok.vocab_size != 32768:
            print(f"WARNING: locked Tier-0 runs use the plan's V=32768 BPE "
                  f"(plan 3A); got tokenizer={args.tokenizer} "
                  f"vocab={tok.vocab_size}. Continuing (harness/test mode).")

    micro_tokens = args.micro_batch * args.seq_len
    if args.batch_tokens % micro_tokens != 0:
        raise SystemExit(
            f"batch-tokens ({args.batch_tokens}) must be a multiple of "
            f"micro-batch x seq-len ({micro_tokens})")
    accum = args.grad_accum or (args.batch_tokens // micro_tokens)
    if args.max_steps is not None and args.max_steps < 1:
        raise SystemExit("--max-steps must be positive")
    total_steps = args.max_steps or max(1, math.ceil(args.max_tokens / args.batch_tokens))

    ckpt_dir = args.ckpt_dir or f"runs/{args.model}-s{args.seed}"
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = args.log_file or os.path.join(ckpt_dir, "train_log.jsonl")
    logf = open(log_path, "a", encoding="utf-8")

    def log(rec: dict):
        rec.setdefault("model", args.model)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(json.dumps(rec), flush=True)

    if args.synthetic_fraction >= 1.0:
        corpus = eval_corpus = None
    else:
        corpus = make_text_corpus(args.data, args.text_dir, subset="train")
        eval_corpus = make_text_corpus(args.data, args.text_dir, subset="eval")
    data = Phase0Data(seed=args.seed, tokenizer=tok, text_corpus=corpus,
                      synthetic_fraction=args.synthetic_fraction,
                      seq_len=args.seq_len, families=families)
    eval_data = make_eval_data(
        tokenizer=tok, text_corpus=eval_corpus,
        synthetic_fraction=args.synthetic_fraction,
        seq_len=args.seq_len, families=families)
    # FIXED eval set: drawn once, identical at every eval and after resume.
    eval_batches = [eval_data.next_batch(args.micro_batch)
                    for _ in range(args.eval_batches)]
    byte_len = token_byte_lengths(tok)
    probes = build_probes(families=families)

    if args.model in EFG_MODELS:
        raw_model = build_efg(args.model, cfg).to(device)
    elif args.model in STAGE3_MODELS:
        raw_model = build_stage3(args.model, cfg).to(device)
    else:
        raw_model = build_sprint(args.model, cfg,
                                 tape_chunk_size=args.tape_chunk_size).to(device)
    decay = [p for p in raw_model.parameters() if p.ndim >= 2]
    no_decay = [p for p in raw_model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=(device == "cuda"))

    start_step, tokens_seen = 0, 0
    if args.resume:
        path = args.resume if args.resume != "latest" \
            else os.path.join(ckpt_dir, "latest.pt")
        # Always land on CPU first: set_rng_state requires a CPU ByteTensor
        # (loading with map_location=cuda crashed GPU resume - audit fix).
        state = torch.load(path, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start_step = state["step"]
        tokens_seen = state["tokens_seen"]
        torch.set_rng_state(state["torch_rng"])
        if device == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        data.load_state_dict(state["data_state"])
        if state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        if state.get("loop_rng") is not None:
            loop_rng.setstate(state["loop_rng"])
        log({"type": "resume", "from": path, "step": start_step,
             "tokens": tokens_seen})

    use_compile = is_cuda and is_bf16_ok and not args.no_compile and not args.tiny
    model = torch.compile(raw_model) if use_compile else raw_model

    prefetcher = None if args.no_prefetch else _Prefetcher(data,
                                                           args.micro_batch)

    n_params = sum(p.numel() for p in raw_model.parameters())
    log({"type": "start", "step": start_step, "tokens": tokens_seen,
         "total_steps": total_steps, "params": n_params, "lr": args.lr,
         "batch_tokens": args.batch_tokens, "micro_batch": args.micro_batch,
         "grad_accum": accum, "seq_len": args.seq_len, "device": str(device),
         "amp_dtype": str(amp_dtype), "grad_scale": scaler.get_scale(),
         "compile": use_compile, "families": list(families)})

    raw_model.train()
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    last_log_t, last_log_tokens = t0, tokens_seen
    done = start_step  # a finished-run resume must not hit an unbound name
    for step in range(start_step, total_steps):
        lr = lr_at(step, total_steps, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        step_N = loop_rng.choice((2, 4, 6)) \
            if args.rand_loops and args.model in ("B2", "TB") else None
        # Pre-draw the whole step: the gradient is normalised by the step's
        # TOTAL supervised tokens, identical to the logged loss (audit fix;
        # previously each micro-batch was weighted equally).
        batches, data_wait = [], 0.0
        for _ in range(accum):
            t_wait = time.time()
            batches.append(prefetcher.next() if prefetcher
                           else data.next_batch(args.micro_batch))
            data_wait += time.time() - t_wait
        ntok_total = sum(int(b[2].sum()) for b in batches)
        loss_sum = None
        for batch in batches:
            ls, _ntok = train_loss_sum(model, batch, device, amp_dtype, N=step_N)
            scaler.scale(ls / max(1, ntok_total)).backward()
            loss_sum = ls.detach() if loss_sum is None \
                else loss_sum + ls.detach()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        tokens_seen += args.batch_tokens
        done = step + 1

        if done % args.log_every == 0 or done == total_steps:
            now = time.time()
            log({"type": "train", "step": done, "tokens": tokens_seen,
                 "loss": (loss_sum / max(1, ntok_total)).item(), "lr": lr,
                 "loop_N": step_N, "grad_scale": scaler.get_scale(),
                 "tok_per_s": round(
                     (tokens_seen - last_log_tokens) / max(1e-9, now - last_log_t)),
                 "peak_vram_gb": round(
                     torch.cuda.max_memory_allocated() / 2**30, 2)
                 if device == "cuda" else 0.0,
                 "data_wait_s": round(data_wait, 3),
                 "elapsed_s": round(now - t0, 1)})
            last_log_t, last_log_tokens = now, tokens_seen
        if done % args.eval_every == 0 or done == total_steps:
            ev = evaluate_split(model, eval_batches, device, amp_dtype,
                                byte_len)
            rec = {"type": "eval", "step": done, "tokens": tokens_seen}
            rec.update({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in ev.items()})
            if (done // args.eval_every) % args.probe_every == 0 \
                    or done == total_steps:
                rec["syn_exact_acc"] = probe_accuracy(
                    raw_model, probes, tok, device, amp_dtype)
            log(rec)
        if done % args.ckpt_every == 0 or done == total_steps:
            path = os.path.join(ckpt_dir, f"step-{done:07d}.pt")
            data_state = prefetcher.resume_state() if prefetcher \
                else data.state_dict()
            save_checkpoint(path, raw_model, opt, done, tokens_seen,
                            data_state, args, scaler, loop_rng.getstate())
            log({"type": "ckpt", "step": done, "tokens": tokens_seen,
                 "path": path})
        if tokens_seen >= args.max_tokens:
            break

    if prefetcher:
        prefetcher.close()
    log({"type": "done", "step": done, "tokens": tokens_seen,
         "elapsed_s": round(time.time() - t0, 1)})
    logf.close()


if __name__ == "__main__":
    main()
