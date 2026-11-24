"""
Phase 0 data pipeline (plan section 3C): locked 80/20 hybrid token stream.

  * 80% general pretraining text: FineWeb-Edu.
  * 20% anti-shortcut algorithmic / structural corpus, programmatically
    generated, zero train/test overlap by construction (fresh random draws),
    split uniformly across the four families named in plan section 3C:

      1. Variable-State Execution (V-Tracking)
      2. Hierarchical AST / Dyck-k + Nested Expression Evaluation
      3. Multi-Hop Graph Search & Constraint Satisfaction
         (shortest path / 2-SAT / Sudoku-lite)
      4. Fact-Logic Separation Probe (Synthetic Entity KB)

Conventions (mirroring smoke_train.py / model.py):
  * Everything is rendered as plain text and encoded with ONE shared
    tokenizer, so text and synthetic data live in the same token-id space
    and every track sees the exact same stream (plan 3C: "exact same token
    stream across all tracks").
  * Per plan 4C (Implicit CoT Leakage), synthetic examples carry a
    prompt/answer split and the loss mask supervises ONLY the answer tokens
    (plus the terminating EOS). Text tokens are all supervised.
  * Deterministic seeding: a single master seed (default 42, the locked
    Phase 0 seed) drives independent random.Random streams for mixing, text
    selection, and each synthetic family. The full stream state can be
    snapshotted / restored (state_dict / load_state_dict) for checkpoint
    resume in train.py.

Tokenizers:
  * "byte" (default, dependency-free): UTF-8 bytes -> ids 0..255, EOS=256,
    PAD=257, vocab_size=258. Used for local CPU tests and as a fallback.
  * "bpe": wraps a HuggingFace `tokenizers` tokenizer.json loaded from
    --tokenizer-file (requires the `tokenizers` package, installable on the
    training instance). The locked Tier-0/Tier-1 runs use the plan's
    V = 32768 BPE vocab (plan 3A); train.py asserts the tokenizer vocab
    matches the model vocab for locked runs.

FineWeb-Edu on the training instance (modelscope.cn reachable, HF not):
    pip install modelscope datasets
    # Option A (ModelScope mirror of the HF dataset):
    #   from modelscope.msdatasets import MsDataset
    #   MsDataset.load('HuggingFaceFW/fineweb-edu', subset_name='sample-10BT',
    #                  split='train')
    # Option B: snapshot-download the dataset repo and read the parquet /
    #   jsonl shards from disk, then point train.py at them with
    #   --data local --text-dir <dir of .txt/.jsonl shards>.
  FineWebEduCorpus below tries (A) via modelscope, then HF `datasets`
  streaming, and raises with instructions if neither is available. For
  offline work, train.py --data fallback uses a small built-in corpus of
  generic paragraphs (deterministic, highly repetitive) purely so the
  harness can be smoke-tested on CPU; it is NOT training data.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

import torch

# --------------------------------------------------------------------------- #
# Tokenizers
# --------------------------------------------------------------------------- #
EOS_ID = 256
PAD_ID = 257


class ByteTokenizer:
    """UTF-8 byte tokenizer: ids 0..255 are raw bytes, 256=EOS, 257=PAD."""

    name = "byte"
    vocab_size = 258
    eos_id = EOS_ID
    pad_id = PAD_ID

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(b for b in ids if 0 <= b < 256).decode("utf-8", errors="replace")


class BPETokenizer:
    """Wraps a HF `tokenizers` tokenizer.json (the plan's V=32768 BPE)."""

    name = "bpe"

    def __init__(self, path: str):
        try:
            from tokenizers import Tokenizer
        except ImportError as e:
            raise RuntimeError(
                "BPE tokenizer requested but the `tokenizers` package is not "
                "installed. On the training instance: pip install tokenizers"
            ) from e
        self._tok = Tokenizer.from_file(path)
        self.vocab_size = self._tok.get_vocab_size()
        eos = self._tok.token_to_id("<|endoftext|>")
        self.eos_id = eos if eos is not None else self.vocab_size - 1
        self.pad_id = self.eos_id

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids) -> str:
        return self._tok.decode(list(ids))


def get_tokenizer(kind: str = "byte", path: str | None = None):
    if kind == "byte":
        return ByteTokenizer()
    if kind == "bpe":
        if not path:
            raise ValueError("--tokenizer bpe requires --tokenizer-file <tokenizer.json>")
        return BPETokenizer(path)
    raise ValueError(f"unknown tokenizer kind: {kind}")


# --------------------------------------------------------------------------- #
# Examples
# --------------------------------------------------------------------------- #
@dataclass
class Example:
    prompt: str
    answer: str
    family: str  # "text", "vtrack", "ast", "graph", "kb"

    @property
    def supervised_from(self) -> int:
        """Char index where the supervised (answer) span starts."""
        return len(self.prompt)


def encode_example(tok, ex: Example) -> tuple[list[int], list[int]]:
    """Return (token_ids, supervised_flags) with EOS appended (supervised).

    flags[i] == 1 iff token i is an answer token (or the final EOS), i.e.
    predicting token i is a supervised loss position. Prompt tokens are 0,
    which implements plan 4C's answer-only loss for synthetic tasks. Text
    examples are fully supervised (answer == whole text, prompt == "")."""
    prompt_ids = tok.encode(ex.prompt)
    answer_ids = tok.encode(ex.answer)
    ids = prompt_ids + answer_ids + [tok.eos_id]
    flags = [0] * len(prompt_ids) + [1] * (len(answer_ids) + 1)
    return ids, flags


# --------------------------------------------------------------------------- #
# Synthetic family 1: Variable-State Execution (V-Tracking), plan 3C.1
# Train: V in [4,8], S in [8,16]. OOD eval: V in {12,16,24,32}, S in {32,64}.
# Semantics are self-contained: header declares all values are mod 100.
# --------------------------------------------------------------------------- #
_VAR_NAMES = [chr(ord("a") + i) for i in range(26)] + [f"v{i}" for i in range(26)]


class VTrackingGenerator:
    family = "vtrack"

    def __init__(self, rng: random.Random, ood: bool = False):
        self.rng = rng
        self.ood = ood

    def sample(self) -> Example:
        rng = self.rng
        if self.ood:
            V = rng.choice([12, 16, 24, 32])
            S = rng.choice([32, 64])
        else:
            V = rng.randint(4, 8)
            S = rng.randint(8, 16)
        names = _VAR_NAMES[:V]
        state = {}
        lines = ["#vtrack mod 100"]
        for n in names:
            state[n] = rng.randint(0, 99)
            lines.append(f"{n} = {state[n]}")

        def expr_value() -> tuple[str, int]:
            """Random small expression over defined vars; returns (text, value)."""
            kind = rng.random()
            if kind < 0.35:
                lit = rng.randint(0, 20)
                return str(lit), lit
            if kind < 0.60:
                v = rng.choice(names)
                return v, state[v]
            a = rng.choice(names)
            op = rng.choice(["+", "-", "*"])
            if op == "*":
                lit = rng.randint(2, 5)
                return f"{a} * {lit}", state[a] * lit
            lit = rng.randint(0, 30)
            val = state[a] + lit if op == "+" else state[a] - lit
            return f"{a} {op} {lit}", val

        for _ in range(S):
            tgt = rng.choice(names)
            if rng.random() < 0.25:  # branching step
                cond_var = rng.choice(names)
                thresh = rng.randint(10, 90)
                gt = state[cond_var] > thresh
                t_txt, t_val = expr_value()
                e_txt, e_val = expr_value()
                lines.append(
                    f"if {cond_var} > {thresh}: {tgt} = {t_txt} else: {tgt} = {e_txt}"
                )
                state[tgt] = (t_val if gt else e_val) % 100
            else:
                txt, val = expr_value()
                lines.append(f"{tgt} = {txt}")
                state[tgt] = val % 100
        q = rng.choice(names)
        prompt = "\n".join(lines) + f"\n? {q}\n"
        return Example(prompt=prompt, answer=str(state[q]), family=self.family)


# --------------------------------------------------------------------------- #
# Synthetic family 2: Hierarchical AST / Dyck-k + nested expressions, plan 3C.2
# Train depth D <= 6; OOD D in {8, 12, 16}. Three sub-kinds: nested
# expression evaluation, Dyck bracket completion, AST scope resolution.
# --------------------------------------------------------------------------- #
class ASTGenerator:
    family = "ast"
    _BRACKETS = [("(", ")"), ("[", "]"), ("{", "}")]

    def __init__(self, rng: random.Random, ood: bool = False):
        self.rng = rng
        self.ood = ood

    def sample(self) -> Example:
        kind = self.rng.random()
        if kind < 0.4:
            return self._expr()
        if kind < 0.7:
            return self._dyck()
        return self._scope()

    # -- nested expression evaluation ------------------------------------- #
    def _expr_tree(self, depth: int):
        rng = self.rng
        if depth == 0 or rng.random() < 0.3:
            return rng.randint(0, 20)
        op = rng.choice(["+", "-", "*"])
        if op == "*":  # keep magnitudes bounded: one side a leaf
            leaf = rng.randint(2, 9)
            sub = self._expr_tree(depth - 1)
            return (op, leaf, sub) if rng.random() < 0.5 else (op, sub, leaf)
        return (op, self._expr_tree(depth - 1), self._expr_tree(depth - 1))

    @staticmethod
    def _eval(node) -> int:
        if isinstance(node, int):
            return node
        op, l, r = node
        lv, rv = ASTGenerator._eval(l), ASTGenerator._eval(r)
        return {"+" : lv + rv, "-": lv - rv, "*": lv * rv}[op]

    def _render(self, node, depth: int) -> str:
        if isinstance(node, int):
            return str(node)
        op, l, r = node
        o, c = self._BRACKETS[depth % 3]
        return f"{o}{self._render(l, depth + 1)} {op} {self._render(r, depth + 1)}{c}"

    def _expr(self) -> Example:
        rng = self.rng
        D = rng.choice([8, 12, 16]) if self.ood else rng.randint(2, 6)
        for _ in range(50):
            tree = self._expr_tree(D)
            val = self._eval(tree)
            if abs(val) <= 99999:
                break
        prompt = f"#expr\n{self._render(tree, 0)} = ?\n"
        return Example(prompt=prompt, answer=str(val), family=self.family)

    # -- Dyck-k completion ------------------------------------------------- #
    def _dyck_seq(self, depth: int, max_depth: int) -> list[str]:
        # Pair budget (audit fix, round 2): the branching recursion is
        # exponential in max_depth (OOD depth 16 could explode). Each opened
        # pair spends 1 budget; when it runs out, no new pairs open, but
        # already-open pairs still close, so the sequence stays balanced.
        rng = self.rng
        out = []
        for _ in range(rng.randint(1, 3)):
            if self._dyck_budget <= 0:
                break
            self._dyck_budget -= 1
            o, c = self._BRACKETS[rng.randrange(3)]
            out.append(o)
            if depth < max_depth:
                out.extend(self._dyck_seq(depth + 1, max_depth))
            out.append(c)
        return out

    def _dyck(self) -> Example:
        rng = self.rng
        D = rng.choice([8, 12, 16]) if self.ood else rng.randint(2, 6)
        open_of = {c: o for o, c in self._BRACKETS}
        close_of = {o: c for o, c in self._BRACKETS}
        for _ in range(50):
            self._dyck_budget = 400 if self.ood else 150
            seq = self._dyck_seq(1, D)
            if len(seq) < 2:
                continue
            cut = rng.randint(1, len(seq) - 1)
            prefix = seq[:cut]
            stack = []
            ok = True
            for ch in prefix:
                if ch in close_of:
                    stack.append(ch)
                elif stack and stack[-1] == open_of[ch]:
                    stack.pop()
                else:
                    ok = False
                    break
            if ok and stack:
                answer = "".join(close_of[o] for o in reversed(stack))
                prompt = "#dyck\n" + "".join(prefix) + " ?\n"
                return Example(prompt=prompt, answer=answer, family=self.family)
        return self._expr()  # pragma: no cover - retry budget exhausted

    # -- AST scope resolution ---------------------------------------------- #
    def _scope(self) -> Example:
        rng = self.rng
        pool = ["x", "y", "z", "w"]
        lines = ["#scope"]
        stack: list[dict] = [{}]
        max_depth = rng.choice([8, 12]) if self.ood else rng.randint(2, 4)
        n_stmt = rng.randint(8, 18) if not self.ood else rng.randint(24, 40)
        for _ in range(n_stmt):
            r = rng.random()
            if r < 0.22 and len(stack) < max_depth:
                lines.append("{")
                stack.append({})
            elif r < 0.34 and len(stack) > 1:
                lines.append("}")
                stack.pop()
            else:
                # bias toward shadowing: often reuse a var defined in an
                # enclosing scope
                defined_outer = [v for d in stack[:-1] for v in d]
                if defined_outer and rng.random() < 0.5:
                    var = rng.choice(defined_outer)
                else:
                    var = rng.choice(pool)
                val = rng.randint(0, 99)
                lines.append(f"{var} = {val}")
                stack[-1][var] = val
        # query a variable visible at the current (innermost) point
        visible = {}
        for d in stack:
            visible.update(d)
        if not visible:
            return self._expr()
        q = rng.choice(list(visible))
        prompt = "\n".join(lines) + f"\n? {q}\n"
        return Example(prompt=prompt, answer=str(visible[q]), family=self.family)


# --------------------------------------------------------------------------- #
# Synthetic family 3: Multi-Hop Graph Search & Constraint Satisfaction,
# plan 3C.3: shortest path / 2-SAT / Sudoku-lite.
# --------------------------------------------------------------------------- #
class GraphSatGenerator:
    family = "graph"

    def __init__(self, rng: random.Random, ood: bool = False):
        self.rng = rng
        self.ood = ood

    def sample(self) -> Example:
        kind = self.rng.random()
        if kind < 0.4:
            return self._shortest_path()
        if kind < 0.7:
            return self._twosat()
        return self._sudoku()

    def _shortest_path(self) -> Example:
        rng = self.rng
        n = rng.randint(10, 14) if self.ood else rng.randint(6, 10)
        nodes = [chr(ord("A") + i) for i in range(n)]
        adj = {u: {} for u in nodes}
        edges = set()

        def add(u, v, w):
            adj[u][v] = adj[v][u] = w
            edges.add((min(u, v), max(u, v), w))

        for i in range(1, n):  # spanning tree -> graph is connected
            add(nodes[i], nodes[rng.randrange(i)], rng.randint(1, 9))
        for _ in range(rng.randint(n // 2, n)):
            u, v = rng.sample(nodes, 2)
            if v not in adj[u]:
                add(u, v, rng.randint(1, 9))
        s, t = rng.sample(nodes, 2)
        # Dijkstra
        import heapq

        dist = {s: 0}
        pq = [(0, s)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, 1 << 30):
                continue
            for v, w in adj[u].items():
                nd = d + w
                if nd < dist.get(v, 1 << 30):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        elines = [f"{u} - {v} : {w}" for u, v, w in sorted(edges)]
        prompt = "#graph\n" + "\n".join(elines) + f"\ndist {s} {t} ?\n"
        return Example(prompt=prompt, answer=str(dist[t]), family=self.family)

    def _twosat(self) -> Example:
        rng = self.rng
        n = rng.randint(6, 10) if self.ood else rng.randint(4, 8)
        m = rng.randint(n, 2 * n)
        clauses = []
        for _ in range(m):
            a, b = rng.sample(range(1, n + 1), 2)
            clauses.append(((a, rng.random() < 0.5), (b, rng.random() < 0.5)))

        def lit_val(lit, mask):
            var, pos = lit
            bit = (mask >> (n - var)) & 1
            return bit == 1 if pos else bit == 0

        sat_masks = [mask for mask in range(1 << n)
                     if all(lit_val(a, mask) or lit_val(b, mask)
                            for a, b in clauses)]

        def fmt(lit):
            var, pos = lit
            return f"x{var}" if pos else f"~x{var}"

        prompt = "#2sat\n" + " & ".join(
            f"({fmt(a)} | {fmt(b)})" for a, b in clauses
        )
        # Entailment fix (audit fix, round 2): with multiple satisfying
        # assignments, "x_k = ?" has a unique correct answer ONLY for
        # variables whose value is the same in EVERY satisfying assignment.
        # The old code answered from the first mask, silently supervising
        # ambiguous questions. Non-entailed cases fall back to SAT?.
        entailed = []
        if sat_masks:
            for var in range(1, n + 1):
                vals = {(m >> (n - var)) & 1 for m in sat_masks}
                if len(vals) == 1:
                    entailed.append((var, vals.pop()))
        if not sat_masks or rng.random() < 0.5 or not entailed:
            prompt += "\nSAT?\n"
            return Example(prompt=prompt,
                           answer="SAT" if sat_masks else "UNSAT",
                           family=self.family)
        var, bit = rng.choice(entailed)
        prompt += f"\nx{var} = ?\n"
        return Example(prompt=prompt, answer=str(bit), family=self.family)

    def _sudoku(self) -> Example:
        rng = self.rng

        def solve(grid, limit=2):
            sols = []

            def bt(pos=0):
                if len(sols) >= limit:
                    return
                if pos == 16:
                    sols.append([row[:] for row in grid])
                    return
                r, c = divmod(pos, 4)
                if grid[r][c] != 0:
                    bt(pos + 1)
                    return
                used = set(grid[r]) | {grid[i][c] for i in range(4)} | {
                    grid[2 * (r // 2) + i][2 * (c // 2) + j]
                    for i in range(2) for j in range(2)
                }
                for d in range(1, 5):
                    if d not in used:
                        grid[r][c] = d
                        bt(pos + 1)
                        grid[r][c] = 0

            bt()
            return sols

        for _ in range(100):
            grid = [[0] * 4 for _ in range(4)]
            # seed a full solution by solving the empty grid with shuffled
            # candidate order via random givens on the diagonal blocks
            for br, bc in ((0, 0), (2, 2)):
                vals = [1, 2, 3, 4]
                rng.shuffle(vals)
                for i in range(2):
                    for j in range(2):
                        grid[br + i][bc + j] = vals[2 * i + j]
            sols = solve([row[:] for row in grid], limit=1)
            if not sols:
                continue
            full = sols[0]
            cells = [(r, c) for r in range(4) for c in range(4)]
            rng.shuffle(cells)
            puzzle = [row[:] for row in full]
            blanks = []
            for r, c in cells[: rng.randint(5, 8)]:
                puzzle[r][c] = 0
                blanks.append((r, c))
            if len(solve([row[:] for row in puzzle], limit=2)) != 1:
                continue  # require unique solution
            blanks.sort()
            lines = [
                " ".join(str(puzzle[r][c]) if puzzle[r][c] else "." for c in range(4))
                for r in range(4)
            ]
            prompt = "#sudoku4\n" + "\n".join(lines) + "\nblanks ?\n"
            answer = " ".join(str(full[r][c]) for r, c in blanks)
            return Example(prompt=prompt, answer=answer, family=self.family)
        return self._shortest_path()  # pragma: no cover


# --------------------------------------------------------------------------- #
# Synthetic family 4: Fact-Logic Separation Probe (Synthetic Entity KB),
# plan 3C.4: arbitrary entity-attribute pairs + multi-hop relational queries.
# --------------------------------------------------------------------------- #
_SYL = ["ka", "zor", "mi", "ta", "lu", "rex", "an", "or", "bel", "shu",
        "vi", "da", "nor", "ek", "tu", "pha", "om", "is", "ur", "al"]
_COLORS = ["red", "blue", "green", "amber", "violet", "teal", "ivory",
           "crimson", "ochre", "slate"]
_CITIES = ["Porto", "Luma", "Vant", "Ostrov", "Kemi", "Dalo", "Riga",
           "Sion", "Tavro", "Ulm", "Brest", "Narva"]
_JOBS = ["baker", "pilot", "weaver", "judge", "miner", "scribe",
         "farmer", "sailor", "mason", "healer"]
_ATTRS = ["color", "city", "job"]
_RELATIONS = ["friend", "mentor"]


class _KB:
    """A synthetic knowledge base: entities with attributes and relations."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.names: list[str] = []
        self.attr: dict[str, dict[str, str]] = {}
        self.rel: dict[str, dict[str, str]] = {}

    def new_entity(self) -> str:
        rng = self.rng
        while True:
            name = (rng.choice(_SYL) + rng.choice(_SYL)).capitalize()
            if rng.random() < 0.4:
                name += str(rng.randint(2, 99))
            if name not in self.attr:
                break
        self.names.append(name)
        self.attr[name] = {
            "color": rng.choice(_COLORS),
            "city": rng.choice(_CITIES),
            "job": rng.choice(_JOBS),
        }
        self.rel[name] = {}
        return name

    def fact_lines(self, name: str) -> list[str]:
        a = self.attr[name]
        lines = [
            f"{name}'s color is {a['color']}.",
            f"{name} lives in {a['city']}.",
            f"{name}'s job is {a['job']}.",
        ]
        for rel, other in self.rel[name].items():
            lines.append(f"{name}'s {rel} is {other}.")
        return lines


class EntityKBGenerator:
    family = "kb"

    def __init__(self, rng: random.Random, ood: bool = False,
                 n_distractor_facts: int = 20):
        self.rng = rng
        self.ood = ood
        self.n_distractor_facts = n_distractor_facts

    def sample(self) -> Example:
        rng = self.rng
        kb = _KB(rng)
        hops = rng.randint(2, 4) if self.ood else rng.randint(1, 3)
        chain = [kb.new_entity()]
        rels = []
        for _ in range(hops):
            rel = rng.choice(_RELATIONS)
            nxt = kb.new_entity()
            kb.rel[chain[-1]][rel] = nxt
            rels.append(rel)
            chain.append(nxt)
        attr = rng.choice(_ATTRS)
        # distractor entities / facts
        facts = []
        for e in chain:
            facts.extend(kb.fact_lines(e))
        n_distr_entities = max(1, self.n_distractor_facts // 4)
        for _ in range(n_distr_entities):
            e = kb.new_entity()
            if rng.random() < 0.5 and kb.names[:-1]:
                kb.rel[e][rng.choice(_RELATIONS)] = rng.choice(kb.names[:-1])
            facts.extend(kb.fact_lines(e))
        rng.shuffle(facts)
        # query phrase: attribute of r_h(...(r_1(e0))) reading inside-out
        phrase = chain[0]
        for rel in rels:
            phrase = f"the {rel} of {phrase}"
        attr_q = {"color": "color", "city": "city", "job": "job"}[attr]
        prompt = "#kb\n" + "\n".join(facts) + f"\nWhat is the {attr_q} of {phrase}?\n"
        return Example(prompt=prompt, answer=kb.attr[chain[-1]][attr],
                       family=self.family)


def make_static_kb(seed: int = 42, n_pairs: int = 50_000, n_queries: int = 200):
    """Build the plan 3C.4 static probe: ~n_pairs entity-attribute/relation
    facts plus multi-hop queries with ground-truth answers (for evaluation).
    Returns (facts_text, [(query, answer), ...])."""
    rng = random.Random(seed)
    kb = _KB(rng)
    per_entity = 5  # 3 attributes + up to 2 relations
    n_entities = max(10, n_pairs // per_entity)
    for _ in range(n_entities):
        kb.new_entity()
    for e in kb.names:
        for rel in _RELATIONS:
            if rng.random() < 0.8:
                kb.rel[e][rel] = rng.choice(kb.names)
    facts = []
    for e in kb.names:
        facts.extend(kb.fact_lines(e))
    queries = []
    for _ in range(n_queries):
        e = rng.choice(kb.names)
        hops = rng.randint(1, 3)
        phrase, cur, ok = e, e, True
        for _ in range(hops):
            rel = rng.choice(_RELATIONS)
            if rel not in kb.rel[cur]:
                ok = False
                break
            cur = kb.rel[cur][rel]
            phrase = f"the {rel} of {phrase}"
        if not ok:
            continue
        attr = rng.choice(_ATTRS)
        queries.append((f"What is the {attr} of {phrase}?", kb.attr[cur][attr]))
    return "\n".join(facts), queries


# --------------------------------------------------------------------------- #
# Text corpora (the 80% stream)
# --------------------------------------------------------------------------- #
_FALLBACK_PARAGRAPHS = [
    "The history of the steam engine begins in the workshops of the eighteenth century, where miners needed a reliable way to pump water from deep shafts. Early engines were enormous and inefficient, burning coal at a tremendous rate, yet they transformed what a single factory could produce in a day.",
    "To sort a list of numbers, one classical approach repeatedly scans the list and swaps neighbouring elements that are out of order. Although simple to describe, this method requires a quadratic number of comparisons in the worst case, so faster algorithms divide the input into smaller pieces and merge the sorted results.",
    "Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide into oxygen and glucose. The light-dependent reactions take place in the thylakoid membranes of chloroplasts, where pigments absorb photons and drive the production of ATP and NADPH.",
    "When a central bank raises its policy interest rate, borrowing becomes more expensive throughout the economy. Mortgages, business loans, and government debt all reprice, savings accounts begin to pay more, and the currency often strengthens as foreign capital seeks the higher yield.",
    "The treaty was signed in the autumn after long negotiations over borders, tariffs, and the status of the northern ports. Delegates argued for weeks about a single clause concerning fishing rights, and the final text reflected a compromise that satisfied neither side completely but ended the blockade.",
    "A hash function maps data of arbitrary size to a fixed-size value. Good hash functions distribute their outputs uniformly, change completely when the input changes by a single bit, and make it infeasible to recover the input from the output or to find two inputs that collide.",
    "Coral reefs support an extraordinary diversity of marine life despite growing in nutrient-poor tropical waters. The reef structure is built by colonies of tiny polyps that secrete calcium carbonate, and the whole ecosystem depends on a symbiosis between the polyps and photosynthetic algae.",
    "In a typical experiment, the researcher first states a hypothesis, then designs a procedure that can falsify it. Measurements are recorded with their uncertainties, control groups are compared against treatment groups, and the analysis is preregistered so that the conclusions cannot be shaped by the data after the fact.",
    "def merge(left, right):\n    out = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            out.append(left[i]); i += 1\n        else:\n            out.append(right[j]); j += 1\n    return out + left[i:] + right[j:]",
    "The orchestra opened with a quiet passage for strings alone, the melody passing from the violas to the cellos before the woodwinds entered. By the final movement the full brass section had joined, and the hall seemed to vibrate with the closing chords as the conductor held the silence for a long moment.",
    "River deltas form where a river carrying sediment reaches a body of standing water and slows down. The deposited material builds new land over centuries, creating wetlands that filter water, shelter young fish, and protect coastlines from storm surges, though they are vulnerable to sea-level rise.",
    "Every graph with more edges than vertices contains a cycle. This simple fact underlies many algorithms: a depth-first search that encounters an already-visited neighbour has found a cycle, and a topological ordering exists exactly when the directed graph has no cycles at all.",
]


class FallbackTextCorpus:
    """Built-in generic paragraphs for offline harness tests only."""

    def __init__(self, subset: str = "all"):
        texts = list(_FALLBACK_PARAGRAPHS)
        if subset == "eval":
            texts = texts[-2:]
        elif subset == "train":
            texts = texts[:-2]
        self.texts = texts

    def next_text(self, rng: random.Random) -> str:
        return rng.choice(self.texts)


class LocalFilesTextCorpus:
    """Random access over .txt files (or {"text": ...} .jsonl) in a directory.

    Lazy / index-based: __init__ scans the files ONCE and records, for every
    usable document, (file, byte offset, byte length) in flat arrays; the
    document text itself is read from disk on demand in next_text(). The
    previous implementation held every document as a Python string, which
    OOM-killed training on the 28 GB instance (6.6 GB of text = 1.45M
    strings, loaded twice: train corpus + eval corpus). The index preserves
    the original document order exactly, and next_text() consumes the RNG
    exactly like rng.choice() did (a single _randbelow call), so seeded
    streams are bit-identical to the old implementation."""

    _INDEX_CACHE: dict = {}
    HOLDOUT_MOD = 50  # doc index % 50 == 49 -> held out for eval (2%)

    def __init__(self, directory: str, subset: str = "all"):
        import glob
        import os
        from array import array

        key = (os.path.abspath(directory), subset)
        cached = self._INDEX_CACHE.get(key)
        if cached is not None:
            self._paths, self._kind, self._file, self._off, self._len = cached
            self._handles: dict = {}
            return

        def keep(doc_idx: int) -> bool:
            # Deterministic document-level holdout (audit fix, round 2):
            # train and eval corpora over the same directory are DISJOINT.
            if subset == "all":
                return True
            held = (doc_idx % self.HOLDOUT_MOD) == (self.HOLDOUT_MOD - 1)
            return held if subset == "eval" else not held

        paths: list[str] = []
        kind = array("b")   # 0 = raw .txt paragraph, 1 = .jsonl line
        file = array("I")   # index into paths (was "b": capped at 127 files)
        off = array("q")
        length = array("i")
        doc_idx = 0
        for path in sorted(glob.glob(os.path.join(directory, "*.txt"))):
            fi = len(paths)
            paths.append(path)
            with open(path, "rb") as f:
                data = f.read()
            pos = 0
            for para in data.split(b"\n\n"):
                if len(para.decode("utf-8", "ignore")) > 200:
                    if keep(doc_idx):
                        kind.append(0)
                        file.append(fi)
                        off.append(pos)
                        length.append(len(para))
                    doc_idx += 1
                pos += len(para) + 2
        for path in sorted(glob.glob(os.path.join(directory, "*.jsonl"))):
            fi = len(paths)
            paths.append(path)
            with open(path, "rb") as f:
                while True:
                    o = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    try:
                        txt = json.loads(line).get("text", "")
                    except json.JSONDecodeError:
                        continue
                    if len(txt) > 200:
                        if keep(doc_idx):
                            kind.append(1)
                            file.append(fi)
                            off.append(o)
                            length.append(len(line))
                        doc_idx += 1
        if not len(off):
            raise RuntimeError(f"no usable text found in {directory}")
        self._paths, self._kind, self._file, self._off, self._len = \
            paths, kind, file, off, length
        self._INDEX_CACHE[key] = (paths, kind, file, off, length)
        self._handles = {}

    def __len__(self):
        return len(self._off)

    def _read(self, i: int) -> str:
        fi = self._file[i]
        h = self._handles.get(fi)
        if h is None:
            h = open(self._paths[fi], "rb")
            self._handles[fi] = h
        h.seek(self._off[i])
        raw = h.read(self._len[i])
        if self._kind[i] == 0:
            return raw.decode("utf-8")
        return json.loads(raw).get("text", "")

    def next_text(self, rng: random.Random) -> str:
        return self._read(rng.randrange(len(self._off)))


class FineWebEduCorpus:
    """Streaming FineWeb-Edu (sample-10BT) via ModelScope or HF datasets.

    See the module docstring for the instance setup. The iterator position
    is NOT part of the pipeline state_dict: on checkpoint resume the text
    stream restarts (mixing/synthetic streams resume exactly; token counts
    and the LR schedule are unaffected)."""

    def __init__(self, subset: str = "sample-10BT"):
        self._buf: list[str] = []
        self._it = self._open(subset)

    @staticmethod
    def _open(subset: str):
        try:  # Option A: ModelScope mirror
            from modelscope.msdatasets import MsDataset

            ds = MsDataset.load("HuggingFaceFW/fineweb-edu", subset_name=subset,
                                split="train")
            return iter(ds)
        except Exception:
            pass
        try:  # Option B: HF datasets (streaming)
            from datasets import load_dataset

            ds = load_dataset("HuggingFaceFW/fineweb-edu", subset,
                              split="train", streaming=True)
            return iter(ds)
        except Exception as e:
            raise RuntimeError(
                "Could not open FineWeb-Edu. On the training instance run:\n"
                "  pip install modelscope datasets\n"
                "and use --data fineweb (ModelScope mirror), or download the "
                "dataset shards and use --data local --text-dir <dir>."
            ) from e

    def next_text(self, rng: random.Random) -> str:
        while True:
            if not self._buf:
                try:
                    row = next(self._it)
                except StopIteration:
                    raise RuntimeError("FineWeb-Edu stream exhausted")
                text = row["text"] if isinstance(row, dict) else str(row)
                # chunk long documents into paragraph blocks
                self._buf = [p for p in text.split("\n\n") if len(p) > 200] or [text]
            txt = self._buf.pop()
            if len(txt) > 200:
                return txt


def make_text_corpus(kind: str, text_dir: str | None = None,
                     subset: str = "all"):
    if kind == "fallback":
        return FallbackTextCorpus(subset=subset)
    if kind == "local":
        if not text_dir:
            raise ValueError("--data local requires --text-dir")
        return LocalFilesTextCorpus(text_dir, subset=subset)
    if kind == "fineweb":
        # Streaming corpus: no document index, so no holdout split; eval on
        # --data fineweb remains a separate-seed stream (documented limit).
        # Formal Phase 0 runs use --data local, which IS holdout-split.
        return FineWebEduCorpus()
    raise ValueError(f"unknown data kind: {kind}")


# --------------------------------------------------------------------------- #
# The mixed, packed token stream
# --------------------------------------------------------------------------- #
SYNTHETIC_FAMILIES = ("vtrack", "ast", "graph", "kb")


class Phase0Data:
    """Locked 80/20 mix (plan 3C) packed into fixed-length sequences.

    Plan 3C defines the mix over the token stream ("80% General
    Pretraining Corpus / 20% Anti-Shortcut Corpus", exact same stream for
    every track). Round-2 audit fixes:
      * The mix is controlled by TOKEN share via a deficit controller that
        tracks tokens emitted from each source and always feeds the source
        behind its target. (The old per-example coin flip weighted a
        ~2000-token text document the same as a ~50-token synthetic item,
        so the realised stream share was nowhere near 80/20.)
      * Synthetic examples are packed WHOLE: one that does not fit the
        rest of the current sequence is deferred intact to the next
        sequence instead of being sliced across the boundary.
      * next_batch returns a 4th tensor, syn_mask, marking every token
        belonging to a synthetic example (prompt + answer), so evaluation
        can split FineWeb NLL from synthetic-answer NLL exactly.

    next_batch(batch_size) -> (input_ids, targets, loss_mask, syn_mask),
    each (batch_size, seq_len); loss_mask[b, t] supervises the prediction
    of targets[b, t] (== input at t+1 in the packed stream)."""

    TEXT_CHUNK = 512  # text tokens per fill decision (controller granularity)

    def __init__(self, seed: int = 42, tokenizer=None, text_corpus=None,
                 synthetic_fraction: float = 0.2, seq_len: int = 2048,
                 ood: bool = False, families: tuple[str, ...] = SYNTHETIC_FAMILIES):
        self.tok = tokenizer if tokenizer is not None else ByteTokenizer()
        self.text_corpus = text_corpus if text_corpus is not None \
            else (None if synthetic_fraction >= 1.0 else FallbackTextCorpus())
        self.synthetic_fraction = synthetic_fraction
        self._families = tuple(families)
        self.seq_len = seq_len
        self.mix_rng = random.Random(seed)
        self.text_rng = random.Random(seed + 1)
        self._gens = {
            "vtrack": VTrackingGenerator(random.Random(seed + 101), ood=ood),
            "ast": ASTGenerator(random.Random(seed + 102), ood=ood),
            "graph": GraphSatGenerator(random.Random(seed + 103), ood=ood),
            "kb": EntityKBGenerator(random.Random(seed + 104), ood=ood),
        }
        self._text_buf_ids: list[int] = []
        self._text_buf_flags: list[int] = []
        self._pending_syn: tuple[list[int], list[int]] | None = None
        self._tok_text = 0   # tokens emitted from the text corpus
        self._tok_syn = 0    # tokens emitted from synthetic generators

    # -- example production ------------------------------------------------ #
    def next_example(self) -> Example:
        if self.synthetic_fraction >= 1.0 or self.mix_rng.random() < self.synthetic_fraction:
            fam = self.mix_rng.choice(self._families)
            return self._gens[fam].sample()
        return Example(prompt="", answer=self.text_corpus.next_text(self.text_rng),
                       family="text")

    def _fill_text(self, need: int):
        while len(self._text_buf_ids) < need:
            ids, flags = encode_example(
                self.tok,
                Example(prompt="",
                        answer=self.text_corpus.next_text(self.text_rng),
                        family="text"))
            self._text_buf_ids.extend(ids)
            self._text_buf_flags.extend(flags)

    def _take_text(self, n: int):
        self._fill_text(n)
        ids = self._text_buf_ids[:n]
        flags = self._text_buf_flags[:n]
        del self._text_buf_ids[:n]
        del self._text_buf_flags[:n]
        self._tok_text += n
        return ids, flags

    def _want_syn(self) -> bool:
        if self.synthetic_fraction >= 1.0:
            return True
        total = self._tok_text + self._tok_syn
        if total == 0:
            return False  # start with text; controller engages immediately
        return (self._tok_syn / total) < self.synthetic_fraction

    def _next_sequence(self):
        L1 = self.seq_len + 1
        ids: list[int] = []
        flags: list[int] = []
        syn: list[int] = []
        while len(ids) < L1:
            remaining = L1 - len(ids)
            if self._pending_syn is not None:
                p_ids, p_flags = self._pending_syn
                if len(p_ids) <= remaining:
                    ids += p_ids
                    flags += p_flags
                    syn += [1] * len(p_ids)
                    self._tok_syn += len(p_ids)
                    self._pending_syn = None
                    continue
                if self.synthetic_fraction >= 1.0:
                    ids += [self.tok.pad_id] * remaining
                    flags += [0] * remaining
                    syn += [1] * remaining
                    break
                # does not fit: fill the tail with text, keep it pending
                t_ids, t_flags = self._take_text(remaining)
                ids += t_ids
                flags += t_flags
                syn += [0] * remaining
                break
            if self._want_syn():
                fam = self.mix_rng.choice(self._families)
                e_ids, e_flags = encode_example(
                    self.tok, self._gens[fam].sample())
                if len(e_ids) > L1:  # pathological single item: hard truncate
                    e_ids, e_flags = e_ids[:L1], e_flags[:L1]
                if len(e_ids) <= remaining:
                    ids += e_ids
                    flags += e_flags
                    syn += [1] * len(e_ids)
                    self._tok_syn += len(e_ids)
                else:
                    self._pending_syn = (e_ids, e_flags)
                continue
            take = min(remaining, self.TEXT_CHUNK)
            t_ids, t_flags = self._take_text(take)
            ids += t_ids
            flags += t_flags
            syn += [0] * take
        return ids, flags, syn

    # -- batching ------------------------------------------------------------ #
    def next_batch(self, batch_size: int):
        seqs = [self._next_sequence() for _ in range(batch_size)]
        ids = torch.tensor([s[0] for s in seqs], dtype=torch.long)
        flags = torch.tensor([s[1] for s in seqs], dtype=torch.long)
        syn = torch.tensor([s[2] for s in seqs], dtype=torch.long)
        return ids[:, :-1], ids[:, 1:], flags[:, 1:], syn[:, 1:]

    # -- checkpoint state ---------------------------------------------------- #
    def state_dict(self) -> dict:
        return {
            "mix_rng": self.mix_rng.getstate(),
            "text_rng": self.text_rng.getstate(),
            "gen_rng": {k: g.rng.getstate() for k, g in self._gens.items()},
            "text_buf_ids": list(self._text_buf_ids),
            "text_buf_flags": list(self._text_buf_flags),
            "pending_syn": self._pending_syn,
            "tok_text": self._tok_text,
            "tok_syn": self._tok_syn,
        }

    def load_state_dict(self, state: dict):
        self.mix_rng.setstate(state["mix_rng"])
        self.text_rng.setstate(state["text_rng"])
        for k, st in state["gen_rng"].items():
            self._gens[k].rng.setstate(st)
        self._text_buf_ids = list(state["text_buf_ids"])
        self._text_buf_flags = list(state["text_buf_flags"])
        self._pending_syn = state["pending_syn"]
        self._tok_text = state["tok_text"]
        self._tok_syn = state["tok_syn"]


def make_eval_data(seed: int = 1234, **kwargs) -> Phase0Data:
    """Held-out stream with its own seed (never trained on)."""
    return Phase0Data(seed=seed, **kwargs)


# --------------------------------------------------------------------------- #
# Standalone stats / demo
# --------------------------------------------------------------------------- #
def _stats(n_per_family: int = 60):
    tok = ByteTokenizer()
    rngs = {fam: random.Random(42 + i) for i, fam in enumerate(SYNTHETIC_FAMILIES)}
    gens = {
        "vtrack": VTrackingGenerator(rngs["vtrack"]),
        "ast": ASTGenerator(rngs["ast"]),
        "graph": GraphSatGenerator(rngs["graph"]),
        "kb": EntityKBGenerator(rngs["kb"]),
    }
    for fam, gen in gens.items():
        plen, alen = [], []
        sample = None
        for _ in range(n_per_family):
            ex = gen.sample()
            sample = ex
            plen.append(len(tok.encode(ex.prompt)))
            alen.append(len(tok.encode(ex.answer)))
        print(f"[{fam}] n={n_per_family} prompt_tokens mean={sum(plen)/len(plen):.1f} "
              f"max={max(plen)} | answer_tokens mean={sum(alen)/len(alen):.1f} "
              f"max={max(alen)}")
        print(f"  sample prompt: {sample.prompt[:160]!r}")
        print(f"  sample answer: {sample.answer!r}")
    data = Phase0Data(seed=42, seq_len=64)
    ids, tgt, mask, syn = data.next_batch(4)
    print(f"[mix] batch shapes: input {tuple(ids.shape)} target {tuple(tgt.shape)} "
          f"mask {tuple(mask.shape)} | supervised fraction "
          f"{mask.float().mean().item():.3f} (expect ~0.2*answer-only + 0.8*full)")


if __name__ == "__main__":
    _stats()
