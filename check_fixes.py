"""Round-2 audit fix checks for data_pipeline.py (not part of plan CI)."""
import json
import os
import random
import tempfile
import time

import data_pipeline as dp

# --- 1. Holdout disjointness on a synthetic local corpus ------------------ #
tmp = tempfile.mkdtemp()
docs = []
for i in range(200):
    docs.append({"text": (f"Document number {i}. " * 30) + f"Unique tail {i}."})
with open(os.path.join(tmp, "a.jsonl"), "w") as f:
    for d in docs:
        f.write(json.dumps(d) + "\n")
tr = dp.LocalFilesTextCorpus(tmp, subset="train")
ev = dp.LocalFilesTextCorpus(tmp, subset="eval")
tr_keys = {(tr._file[i], tr._off[i]) for i in range(len(tr))}
ev_keys = {(ev._file[i], ev._off[i]) for i in range(len(ev))}
assert not (tr_keys & ev_keys), "train/eval corpora overlap!"
assert len(tr) + len(ev) == 200, (len(tr), len(ev))
print(f"[PASS] holdout disjoint: train={len(tr)} eval={len(ev)} "
      f"(eval share {len(ev) / 200:.0%})")

# --- 2. 2-SAT entailment: every 'x_k = ?' answer holds in ALL sat masks --- #
g = dp.GraphSatGenerator(random.Random(5))
checked = 0
for _ in range(400):
    ex = g._twosat()
    if " = ?" not in ex.prompt:
        continue
    lines = ex.prompt.split("\n")
    clause_str = lines[1]
    var = int(lines[2].split("=")[0].strip()[1:])
    clauses = []
    for part in clause_str.split(" & "):
        lits = part.strip("()").split(" | ")
        parsed = []
        for lit in lits:
            pos = not lit.startswith("~")
            parsed.append((int(lit.lstrip("~x")), pos))
        clauses.append(parsed)
    n = max(v for c in clauses for v, _ in c)

    def val(lit, mask):
        v, pos = lit
        bit = (mask >> (n - v)) & 1
        return bit == 1 if pos else bit == 0

    sats = [m for m in range(1 << n)
            if all(val(a, m) or val(b, m) for a, b in clauses)]
    assert sats, "entailed question on an UNSAT instance"
    vals = {(m >> (n - var)) & 1 for m in sats}
    assert vals == {int(ex.answer)}, \
        f"ambiguous 2sat answer: vals={vals} answer={ex.answer}"
    checked += 1
assert checked >= 30, f"too few entailed samples checked: {checked}"
print(f"[PASS] 2-SAT entailment: {checked} variable questions all entailed")

# --- 3. Dyck OOD generation is bounded ------------------------------------ #
g2 = dp.ASTGenerator(random.Random(9), ood=True)
t0 = time.time()
worst = 0
for _ in range(30):
    ex = g2._dyck()
    worst = max(worst, len(ex.prompt) + len(ex.answer))
dt = time.time() - t0
assert dt < 20, f"dyck OOD too slow: {dt:.1f}s for 30"
print(f"[PASS] Dyck OOD bounded: 30 examples in {dt:.2f}s, "
      f"longest prompt+answer {worst} chars")

# --- 4. Stream determinism + state round-trip ------------------------------ #
d1 = dp.Phase0Data(seed=42, seq_len=128)
b1 = d1.next_batch(2)
st = d1.state_dict()
b2 = d1.next_batch(2)
d2 = dp.Phase0Data(seed=42, seq_len=128)
d2.load_state_dict(st)
b2r = d2.next_batch(2)
import torch
assert all(torch.equal(x, y) for x, y in zip(b2, b2r)), "state round-trip diverged"
print("[PASS] Phase0Data state_dict round-trip reproduces the stream")

print("\nALL FIX CHECKS PASSED")
