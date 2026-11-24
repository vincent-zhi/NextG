# F/E/G staged falsification — results handoff (2026-11-24 23:10 GMT+8)

Status at write time: **Stage 1 complete, Stage 2 complete (early-stopped by its
pre-registered gates), Stage 3 in progress** (`DENSE_NARROW` ~28% trained,
`G_WHT` queued). Everything below is measured unless explicitly marked
"pending".

Repo: `/root/work/loop-verify-sprint` (no git repo; files are the source of
truth). GPU: one V100-SXM2-16GB (SM70), torch 2.5.1+cu124, fp16 + GradScaler,
`--no-compile` on SM70. Disk: 86 GB free, `runs/` currently 9.8 GB.

---

## 0. Protocol as implemented (all gates pre-registered by the user)

| Stage | Direction | Gate (short) | Verdict |
|---|---|---|---|
| 1 | F — channel-space Matryoshka V-cycle | OOD beats BOTH controls by >=3pt, Nc 10→20 holds, tok/s >=0.90x B2_THIN | **RED** |
| 2 | E — latent verifier best-of-K | base woken (acc(N) > acc(N/2)), then Gate 2A Oracle Pass@4 headroom >=5pt, then E2 >=3pt over N=16 and logit-conf, AUROC >=0.70 | **RED (BASE_SLEEPING + NO_DIVERSITY_HEADROOM)** |
| 3 | G — holographic coefficients, real FineWeb-Edu | static w+opt VRAM >=3.5x lower than B0_FULL, fw_bpb >=0.02 better than DENSE_NARROW, speed floor | **pending** (only static VRAM measured: 3.74x, passes) |

---

## 1. Stage 1 (F) — FULL RESULTS

### 1.1 Training (identical budget for all three)

`max_tokens=60M`, 916 steps × 65,536 tokens = 60,030,976 tokens; `seq_len=1024`,
`micro_batch=8`, `lr=3e-3`, `seed=42`, `--tokenizer byte` (V=258),
`synthetic_fraction=1.0`, `families=("graph","ast")` 50/50, pre-registered
crucible `L_pre=1, L_core=4, L_post=0`.

| model | non-emb params | total params | wall time | tok/s (median last 20) | peak VRAM | final syn_nll |
|---|---|---|---|---|---|---|
| B2_THIN (N=4) | 16.0626M | 16.1947M | 1415 s | 48,250 | 5.21 GB | 0.5253 |
| B2_NARROW (N=14) | 4.0175M | 4.0836M | 2614 s | 27,470 | 8.42 GB | 0.5444 |
| F_VCYCLE (Nc=10) | 16.0646M | 16.1967M | 1931 s | 31,516 | 7.70 GB | **0.5057** |

Core-FLOP accounting verified independently (T=512 attention convention, the same
convention as the handoff): B2_THIN 119.5M/token, B2_NARROW 119.3M/token,
F_VCYCLE 85.2M coarse + 29.9M fine = 115.1M/token. Parameter accounting:
F adds exactly `2 * d0 * L_core = 2048` coarse-norm params over B2_THIN.

### 1.2 Frozen OOD crucible

Fixed probe set: per family 40 in-distribution (`GraphSatGenerator(Random(777))`,
`ASTGenerator(Random(778))`) and 80 OOD (`Random(888)` / `Random(889)`, `ood=True`)
→ 160 OOD + 80 in-dist. Greedy decode, `max_new=24`, byte tokenizer.

| model | in-dist acc (graph/ast/mean) | OOD@N_train (graph/ast/mean) | N sweep, OOD mean |
|---|---|---|---|
| B2_THIN | 22.5 / 40.0 / **31.25%** | 31.25 / 20.0 / **25.625%** @N=4 | 1:21.9, **2:25.6, 4:25.6**, 8:23.8, 16:21.3, 32:15.0 |
| B2_NARROW | 25.0 / 45.0 / **35.0%** | 30.0 / 18.75 / **24.375%** @N=14 | 4:20.6, 7:23.8, **14:24.4**, 28:23.8 |
| F_VCYCLE | 20.0 / 40.0 / **30.0%** | 27.5 / 20.0 / **23.75%** @Nc=10 | 2:23.1, 5:23.1, **10:23.8, 20:23.8, 40:23.8** |

Per-family OOD at N_train: B2_THIN 31.25/20.0, B2_NARROW 30.0/18.75,
F_VCYCLE 27.5/20.0 (graph/ast).

### 1.3 Verdict

`F_ood − B2_THIN = −1.875pt`, `F_ood − B2_NARROW = −0.625pt`,
`speed = 0.653x B2_THIN`, `Nc10→Nc20 = 0.0pt`.
**F = RED.** (It is also the only model whose NLL improves over controls, but
that does not transfer OOD.)

### 1.4 Important side finding: the core is still sleeping

The thin-boundary crucible did not fix the sleeping core. `B2_THIN` accuracy at
N=2 and N=4 is *exactly* 41/160 (25.625%) with identical per-family counts, even
though 62/160 OOD predictions do change between N=2 and N=4 (checked with
`max_new=16`). So the loop changes outputs without changing correctness at this
budget. This is why Stage 2's pre-registered base gate fails. Accuracy is also
non-monotone in N (N=1 21.9 -> N=2/4 25.6 -> N=32 15.0).

---

## 2. Stage 2 (E) — EARLY-STOPPED, NO VERIFIER TRAINED

### 2.1 Base checkpoint gate (pre-registered)

B2_THIN: `acc(N_train=4) = 25.625%`, `acc(N_train/2=2) = 25.625%` → strict
`>` fails. `acc >= 15%` holds. → **BASE_SLEEPING**, so by protocol the verifier
is not trained and E is RED; proceed to Stage 3.

### 2.2 Supplementary Gate 2A (diagnostic only, 160 fresh OOD prompts,
`GraphSat(Random(4242), ood=True)` + `AST(Random(4243), ood=True)`, 80 each,
N=4, K=4 = clean + 3 perturbed trajectories)

Perturbation: deterministic `torch.Generator` noise of shape `(1, max_len, d)`
drawn once per trajectory and sliced `[:, :T]` (bitwise identical across decoding
steps), injected at the end of loop t=0, scaled by `sigma_rel * RMS(h)` per
position.

| sigma_rel | Pass@1 | Oracle Pass@4 | headroom |
|---|---|---|---|
| 0.02 | 17.50% | 17.50% | +0.00pt |
| 0.05 | 17.50% | 17.50% | +0.00pt |
| 0.10 | 17.50% | 17.50% | +0.00pt |
| 0.20 | 17.50% | 18.125% | **+0.625pt** |

Max headroom 0.625% << 5% → Gate 2A would fail independently.
**E = RED (BASE_SLEEPING + NO_DIVERSITY_HEADROOM).**

### 2.3 What is implemented but NOT executed (ready to run on a woken base)

`eval_stage2_e.py` contains the full Gate 2B: 1,500-prompt trajectory collection
(6 trajectories at N=4 + depths {2,4,8,16}), `LatentVerifier`
(RMSNorm(2d)->Linear(2d,d/2)->SiLU->Dropout(0.1)->Linear(d/2,1), ~0.52M params),
BCE(-E,y) + margin ranking on same-prompt (correct,wrong) pairs, 20 epochs
AdamW(1e-3, 1e-2), 90/10 prompt-level split, then the six-row FLOP-matched
comparison and OOD AUROC. The full path was smoke-tested end-to-end with
`--n-collect 20 --epochs 1 --force-gate2a` on the step-450 partial checkpoint
(it ran; numbers meaningless). One bug class was fixed along the way (device
placement of labels/pair indices).

---

## 3. Stage 3 (G) — IN PROGRESS

### 3.1 Data

`/mnt/systemdisk/fw_mini/shard0.jsonl`: 120,000 documents, 582.7 MB, 25 s to
stream from the ModelScope mirror via `stage3_prepare_fw.py` (HF is unreachable;
`datasets` + `modelscope` deps installed). `LocalFilesTextCorpus` applies the
document-level 98/2 holdout (`doc_idx % 50 == 49` → eval). Byte tokenizer,
`synthetic_fraction=0.0`.

### 3.2 Architecture (implemented in `tracks_efg.py`, CI-passing)

`G_WHT`: 20 unique layers, d=512, d_ff=1536, 8 heads, block size b=128.
Square projections are `W = H512 · blockdiag(C1..C4) · H512` with
`H512 = H16 ⊗ H32`, both normalized Sylvester Hadamard buffers (non-trainable,
zero optimizer state). FFN gate/up map 512→3×512 chunks, down sums the three
chunk maps. `fwht` = reshape (...,16,32) → `matmul(H32)` → `matmul(H16)` →
reshape. Init std `0.02·sqrt(512/128)=0.04` for q/k/v/gate/up; zero init for
out/down. Coefficients are applied with `einsum("nij,ijk->nik")` (bmm over a
(4,128,128) stack does not broadcast; fixed).

CI (`test_efg.py`, all PASS):
- `||H16^T H16 − I||∞ = 0.0`, `||H32^T H32 − I||∞ = 5.96e-8`
- exact causal gradient (future ∂L/∂x = 0.0) and bitwise-identical past logits
- 2-phase gradient flow across 222 tensors
- CUDA FP16 finite, grad_scale 65536

### 3.3 Exact parameter and static-VRAM measurements (already final)

Measured by building the model, one real optimizer step, summing
parameter + AdamW moment bytes (fp32). Reproduce with
`python -c "import eval_stage3_g as E; print(E.static_vram_mb('G_WHT'))"`.

| model | non-emb params | weight+opt VRAM | +grad | ratio vs B0_FULL |
|---|---|---|---|---|
| DENSE_NARROW (d=256, d_ff=768, 20L) | 17.0524M | 195.9 MB | 261.2 MB | — |
| G_WHT (d=512, d_ff=1536, 20L, b=128) | 17.0629M | 196.8 MB | 262.4 MB | — |
| B0_FULL (d=512, d_ff=1408, 20L) | 64.2490M | 736.8 MB | 982.4 MB | **3.74x** |

→ the >=3.5x static-VRAM gate **passes**.

### 3.4 Throughput warning (important for interpreting the gate)

Isolated fwd+bwd micro-benchmark (micro_batch=8, seq 1024, one optimizer step per
micro-batch): DENSE_NARROW 227 ms → 36.3k tok/s; G_WHT 735 ms → 11.1k tok/s
(**0.31x**). NVIDIA profiler: 24.5% of G's time is the two `fwht` matmuls
(1,680 tiny `volta_fp16_s884gemm_64x64` calls per 3 steps, 289 us each), 12.9%
the block-diagonal einsums, 8.6% attention, ~25% elementwise (RMSNorm upcasts,
RoPE, SiLU). Under real training with 8-micro-batch accumulation DENSE_NARROW
reaches ~80k tok/s; scaling G the same way implies ~22-24k tok/s (**~0.28x
DENSE_NARROW**), i.e. the speed floor (0.70x DENSE or 0.85x B0_FULL) is very
likely to fail even if fw_bpb wins. Replacing the 2-stage fwht with a single
precomputed 512x512 Hadamard matmul was benchmarked at 0.19 ms vs 0.37 ms per
call but raises FLOPs 10x for that op; not adopted (deviates from the
pre-registered operator and its FLOP claim).

### 3.5 Training status at handoff

- `DENSE_NARROW`: **DONE** — 2,289 steps / 150,011,904 tokens in 2000.6 s
  (33.4 min), 78,648 tok/s (median last 20), peak 3.3 GB, final train loss
  1.1760. fw_bpb on the fixed 2% holdout: 2.4614 @250, 2.1928 @500,
  **1.761 @2289**. This is the bar G_WHT must beat by >=0.02 (i.e. <=1.741).
- `G_WHT`: started 23:36, 2,289 steps / 150M tokens, 17.195M total params.
  **Live throughput 25,713 tok/s = 0.327x DENSE_NARROW**, peak 5.94 GB
  (step 50 at 128.3 s; ETA ~01:12). This fails the DENSE-relative speed floor
  (needs >=0.70x); only the B0-relative branch (>=0.85x B0_FULL) remains, and
  B0_FULL is only launched if G's fw_bpb beats DENSE's 1.761.
- `B0_FULL`: runs only if G's final fw_bpb < DENSE_NARROW's (protocol); static
  VRAM is already measured regardless.
- Result file: `runs/stage3_report.json` (written by `eval_stage3_g.py`).

---

## 4. Files and how to reproduce

| file | purpose |
|---|---|
| `tracks_efg.py` | `FVCycle`, `GWhT`/HoloBasis/Holo* projections, configs for B2_THIN/B2_NARROW/F_VCYCLE/DENSE_NARROW/G_WHT/B0_FULL |
| `test_efg.py` | Stage-1 + Stage-3 CI (causality, 2-phase grads, FP16, params, orthogonality) |
| `train.py` | unified trainer (Phase-0 loop); accepts the new model names |
| `run_stage1_f.sh` | CI + 3 × 60M-token runs + `eval_stage1_f.py` |
| `eval_stage1_f.py` | frozen 160-OOD sweep + verdict |
| `eval_stage2_e.py` | Gate 2A/2B: perturbation forward, batched exact-length decode, verifier, six-row eval |
| `stage3_prepare_fw.py` | builds the 120k-doc local FineWeb-Edu shard |
| `run_stage3_g.sh` | CI + DENSE_NARROW + G_WHT (+conditional B0_FULL) + `eval_stage3_g.py` |
| `eval_stage3_g.py` | static VRAM, peak VRAM, tok/s, fw_bpb curve, verdict |
| `make_report.py` | assembles `runs/FINAL_REPORT.md` |

Reproduce Stage 1:
```bash
bash run_stage1_f.sh
```
Reproduce the Stage-2 supplement (protocol stops before Gate 2B):
```bash
python eval_stage2_e.py --ckpt runs/stage1_b2_thin --model B2_THIN   # stops at base gate
python - <<'PY'  # supplementary Gate 2A only (see runs/stage2_gate2a_supplement.json)
...
PY
```
Reproduce Stage 3:
```bash
python stage3_prepare_fw.py          # if the shard is missing
bash run_stage3_g.sh
```

Logs: `logs/stage1_master.log`, `logs/stage1_eval.log`,
`logs/stage3_master.log`, `logs/stage3_*_{eval}.log`; per-run JSONL at
`runs/<run>/train_log.jsonl`; checkpoints + AdamW state at
`runs/<run>/latest.pt` / `step-XXXXXXX.pt` (186 MB for 16M-param models).

---

## 5. Open questions / suggested next steps

1. **Why is the core still asleep?** N=2 vs N=4 changes 39% of predictions but
   zero accuracy. Investigate (a) the frozen Loop-muP scale
   `1/(N·sqrt(1+N·L))` making the per-iteration delta too small at 60M tokens,
   (b) whether grads into the core are dominated by the zero-init out/down
   projections, (c) deeper token budgets — the sleeping-core diagnosis may simply
   need >100M tokens on graph/ast.
2. **F fits better but generalizes worse** (lowest syn_nll 0.5057, worst OOD).
   Distinguish overfitting of the coarse pathway from a genuinely weaker coarse
   representation; a 2x-token rerun with the same crucible would separate them.
3. **Stage 2 is untested on a woken base.** `eval_stage2_e.py` is ready; run it
   only after a checkpoint passes `acc(N)>acc(N/2)` and OOD >= 15%.
4. **G speed.** If G's fw_bpb wins, the scientific claim (coefficient-space
   training is quality-neutral at 3.77x compression) can still hold while the
   speed gate fails. A fused fwht + fused block-diagonal kernel, or BF16 on
   sm80+, is the obvious remedy; eager PyTorch on SM70 is the bottleneck.
5. **Tokenizer.** Byte-level V=258 was used everywhere, which keeps fw_bpb exact
   and embeddings tiny. Results may differ with the plan's BPE V=32768.
6. **Probe-set sensitivity.** 80 OOD items per family gives ±~5pt binomial
   error; the 1.9pt F deficit is inside one sigma of the 160-item estimate, so
   treat F's OOD ordering as suggestive, not decisive; the speed and
   Nc-extrapolation failures are not noise-limited.

## 6. Deviations / decisions worth recording

- Stage 2 was **not** run (protocol); only the Gate-2A diagnostic was run, and
  its numbers are labelled supplementary.
- Gate 2A used 80 OOD items per family (160 total) rather than 40+40; this is a
  stricter version of the handoff's "80 held-out graph_ood + ast_ood prompts".
- `eval_stage1_f.py`'s in-distribution probe is 40 per family (80 total).
- Stage-3 `micro_batch=8` per the handoff; larger batches were not substituted.
- Static VRAM is reported as weights + AdamW moments; peak VRAM is from the
  trainer.
