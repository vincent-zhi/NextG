from __future__ import annotations

import argparse
import contextlib
import random

import torch

from data_pipeline import ASTGenerator, Example, GraphSatGenerator, VTrackingGenerator, get_tokenizer
from model import tier0_config
from tracks_sprint import build_sprint


@torch.no_grad()
def eval_exact(model, examples, tok, device, amp_dtype, max_new=24, **forward_kwargs):
    hits = 0
    autocast = contextlib.nullcontext() if amp_dtype is None \
        else torch.autocast("cuda", dtype=amp_dtype)
    for example in examples:
        x = torch.tensor([tok.encode(example.prompt)], device=device)
        generated = []
        with autocast:
            for _ in range(max_new):
                logits = model(x, **forward_kwargs)
                next_token = int(logits[0, -1].argmax())
                if next_token == tok.eos_id:
                    break
                generated.append(next_token)
                x = torch.cat([x, torch.tensor([[next_token]], device=device)], dim=1)
        hits += int(tok.decode(generated).strip() == example.answer.strip())
    return round(hits / max(1, len(examples)), 3)


def sample_vtrack_fixed_V(V, S=24, n=32, seed=2026):
    rng = random.Random(seed + V)
    names = [chr(ord("a") + i) for i in range(26)] + [f"v{i}" for i in range(26)]
    names = names[:V]
    examples = []
    for _ in range(n):
        state = {name: rng.randint(0, 99) for name in names}
        lines = ["#vtrack mod 100"] + [f"{name} = {state[name]}" for name in names]
        for _ in range(S):
            target = rng.choice(names)
            source = rng.choice(names)
            op = rng.choice(["+", "-"])
            literal = rng.randint(1, 25)
            lines.append(f"{target} = {source} {op} {literal}")
            state[target] = (state[source] + literal if op == "+" else state[source] - literal) % 100
        query = rng.choice(names)
        prompt = "\n".join(lines) + f"\n? {query}\n"
        examples.append(Example(prompt, str(state[query]), "vtrack"))
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model", required=True, choices=["B2", "TB", "TC_TAPE"])
    parser.add_argument("--tokenizer", default=None, choices=["byte", "bpe"])
    parser.add_argument("--tokenizer-file", default=None)
    parser.add_argument("--tape-chunk-size", type=int, default=None)
    parser.add_argument("--max-new", type=int, default=24)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    saved_args = state.get("args", {})
    tokenizer_kind = args.tokenizer or saved_args.get("tokenizer", "byte")
    tokenizer_file = args.tokenizer_file or saved_args.get("tokenizer_file")
    chunk_size = args.tape_chunk_size or saved_args.get("tape_chunk_size", 128)
    tok = get_tokenizer(tokenizer_kind, tokenizer_file)
    if saved_args.get("tiny"):
        from model import LoopConfig
        cfg = LoopConfig(d=128, d_ff=352, L_pre=1, L_core=2, L_post=1,
                         vocab_size=tok.vocab_size, N=2)
    else:
        cfg = tier0_config(vocab_size=tok.vocab_size)

    amp_dtype = None
    if torch.device(device).type == "cuda":
        capability = torch.cuda.get_device_capability(device)
        amp_dtype = torch.bfloat16 if torch.version.hip is not None or capability[0] >= 8 \
            else torch.float16
    model = build_sprint(args.model, cfg, tape_chunk_size=chunk_size).to(device).eval()
    model.load_state_dict(state["model"])

    graph_ood = [GraphSatGenerator(random.Random(900 + i), ood=True).sample() for i in range(40)]
    ast_ood = [ASTGenerator(random.Random(950 + i), ood=True).sample() for i in range(40)]

    print(f"\n=== Extrapolation Curve ({args.model}, max_new={args.max_new}) ===")
    curve = {}
    for N in (2, 4, 8, 16, 32):
        graph_acc = eval_exact(model, graph_ood, tok, device, amp_dtype, args.max_new, N=N)
        ast_acc = eval_exact(model, ast_ood, tok, device, amp_dtype, args.max_new, N=N)
        curve[N] = {"graph_ood": graph_acc, "ast_ood": ast_acc}
        print(f"  N={N:2d} | graph_ood={graph_acc:.3f} | ast_ood={ast_acc:.3f}")
    for family in ("graph_ood", "ast_ood"):
        values = [curve[N][family] for N in curve]
        drop = curve[32][family] - max(values)
        print(f"  {family} overthinking_drop(acc32-max_acc)={drop:+.3f}")

    if args.model == "TB":
        print("  Cayley rotation norms (N=4):", model.rotation_stats(4))
        print("  Cayley rotation norms (N=16):", model.rotation_stats(16))

    print(f"\n=== V-Scaling Curve ({args.model}, S=24, N=4) ===")
    for V in (4, 8, 12, 16, 24, 32):
        examples = sample_vtrack_fixed_V(V=V, S=24, n=32)
        accuracy = eval_exact(model, examples, tok, device, amp_dtype, args.max_new, N=4)
        line = f"  V={V:2d} | acc={accuracy:.3f}"
        if args.model == "TC_TAPE":
            knockout = eval_exact(model, examples, tok, device, amp_dtype, args.max_new,
                                  N=4, tape_knockout=True)
            line += f" | knockout={knockout:.3f} | delta_KO={accuracy - knockout:+.3f}"
        print(line)


if __name__ == "__main__":
    main()
