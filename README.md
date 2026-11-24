# Loop-Verify — Phase 0 Harness for the Looped-Transformer Experiment Plan

Implements the shared Pre-Norm backbone, the three locked baselines (B0/B1/B2),
and the Option-1 feasibility sprint tracks (`TB` and `TC_TAPE`) with synthetic-only
data, V100 FP16 support, mandatory sprint CI, and crucible evaluation.

## V100 setup

Use Python 3.11/3.12. For a V100 with driver support for CUDA 12.6:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
pip install tokenizers
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

If `nvidia-smi` shows a driver limited to CUDA 12.1–12.4, use
`torch==2.5.1` with the `cu124` wheel instead. Do not install the latest
CUDA 13 wheel on Volta/SM70.

## Run the V100 sprint

```bash
bash sprint.sh check   # environment + tests.py + test_sprint.py
bash sprint.sh smoke   # 20-step runs, resume to step 30, VRAM/grad-scale gate
bash sprint.sh train   # B2, TB --rand-loops, TC_TAPE at 50M tokens
bash sprint.sh eval    # N/V curves, knockout, rotation stats
```

Useful controls:

```bash
MICRO_BATCH=16 bash sprint.sh train        # only after confirming a 32GB V100
TOKENIZER=bpe TOKENIZER_FILE=tokenizer.json bash sprint.sh all
bash sprint.sh b2rand                      # random-loop confound control
bash sprint.sh extend                      # resume all three models to 100M
```

The script never deletes existing checkpoints. Training automatically resumes
when `<run-dir>/latest.pt` exists.

## Files

- `model.py` — shared backbone plus B0/B1/B2 baselines.
- `tracks_sprint.py` — Track B Cayley and Track C chunk-causal Tape.
- `data_pipeline.py` — synthetic generators, tokenizers, and mixed/synthetic-only streams.
- `train.py` — baseline/sprint training with AMP, GradScaler, and exact resume state.
- `tests.py` / `test_sprint.py` — Phase 0 and sprint CI gates.
- `eval_crucible.py` — N extrapolation, V scaling, knockout, and rotation diagnostics.
- `sprint.sh` — server-side check/smoke/train/eval workflow.
- `count_params.py` / `smoke_train.py` — Phase 0 parameter and CPU smoke utilities.

## Substitution: SDPA instead of FlashAttention-2

The plan specifies FlashAttention-2, which is GPU-only. On CPU, attention uses
PyTorch SDPA with an explicit additive causal mask (0 on/below the diagonal,
−inf above). This is mathematically equivalent causal self-attention, and
because masked logits are exactly −inf → exactly 0 softmax weight, the
bit-exact causality property required by CI test 1 is preserved (verified:
perturbing token x_{t+1} leaves logits at positions ≤ t bitwise identical and
∂L_t/∂x_{t+1} exactly 0). When this harness moves to GPU (Phase 0 training on
ROCm), swap the SDPA call for FlashAttention-2 / SDPA flash backend without
changing any math.

## How each CI test is configured

1. **Bit-exact causality** — run against all of B0/B1/B2 in the autograd form
   of the plan's formula, ‖∂L_t/∂x_{t+1}‖∞ = 0 *exactly*, plus a finite
   perturbation check (bitwise-equal logits at positions ≤ t).
2. **Gradient flow** — two-phase (see ambiguity #3 below). Phase A: after the
   first backward, every parameter has a non-None, finite grad, and the
   zero-grad set is exactly the pre-projection weights. Phase B: after one
   optimizer step, a fresh backward gives non-zero grad to *every* parameter.
3. **Identity-init equivalence** — B2 configured as **N=1, loop_scale
   overridden to exactly 1.0, and input re-injection disabled
   (reinject=False)**. The loop update then reduces to
   h ← h + 1.0·(CoreStack(h) − h) = CoreStack(h), i.e. one single pass of the
   tied core. Weights are copied into a B1 via `load_state_dict` (identical
   keys/shapes), and forwards agree with max|diff| = 0.0 (< 1e-5 required).
4. **Loop-extrapolation smoke** — final-projection weights are first
   re-initialized with std=0.02 Gaussians (see ambiguity #4), then hidden-state
   RMS is recorded at N ∈ {1,2,4,8,16,32} and asserted finite and
   ≤ max(10·RMS(N=1), RMS(N=1)+5). Measured RMS: N=1: 0.0794, N=2: 0.0785,
   N=4: 0.0772, N=8: 0.0761, N=16: 0.0752, N=32: 0.0745.

## Ambiguities in the plan and what was chosen

1. **Loop-μP scaling placement.** §2B says "scale every recurrent residual
   output by 1/(N√L_eff)", while the locked B2 equation in §3A applies the
   factor once to the whole core output per loop iteration. These differ
   (per-branch scaling composes nonlinearly). **Chosen: the §3A equation,
   literally** — one factor per loop iteration applied to the core's total
   residual delta; core branches run at internal scale 1.0. Pre/post
   (non-recurrent) residuals are unscaled.
2. **Meaning of f_core(h, x_pre).** §3A writes h ← h + c·f_core(h, x_pre)
   without defining f_core's return. **Chosen:** f_core returns the core
   stack's *residual delta* evaluated at the re-injected input,
   CoreStack(h + x_pre) − (h + x_pre), where x_pre is the hidden state after
   the pre-layers. This is the only reading under which the identity test
   (test 3) can hold exactly.
3. **Zero-init vs. gradient-flow conflict.** §2B's locked zero-init of final
   projections makes gradients into all pre-projection weights (q/k/v,
   gate/up, norms) *exactly zero on the first backward, by construction* —
   §2C test 2's "non-zero gradients reach every module on step 1" cannot hold
   literally for a zero-init model. Resolved with the two-phase test above:
   step-1 flow is verified to be blocked *only* where zero-init predicts, and
   full flow is verified after one optimizer step.
4. **Loop-extrapolation test would be vacuous at locked init.** With
   zero-init projections every block is the identity, so RMS is constant in N
   for a fresh model. The test therefore re-initializes the final projections
   (std=0.02) *for this test only*, so loop dynamics are actually stressed.
   Training init remains zero-init as locked.
5. **Plan test 3 is written for Tracks B/C vs. Flat-Loop.** In Phase 0 (no
   tracks yet) it is applied to B2 itself: B2 in identity configuration must
   reproduce the single-pass tied-core forward (a B1 holding the same
   weights). The same harness function can be reused for Tracks B/C in
   Phase 1.
6. **QK-Norm granularity unspecified.** Implemented as a per-head RMSNorm
   over d_head=64 applied to q and k after reshaping (2×64 params/layer —
   negligible; the plan's parameter table reproduces either way).
7. **Biases.** The plan never mentions biases; its parameter table only
   reproduces with **no biases anywhere** (all Linear layers bias=False),
   which is what is implemented. A final RMSNorm (d params) is included in
   non-embedding counts; it is invisible at the table's 0.1M rounding.
8. **Tiny-scale deviations for CPU.** Tests use d=128, d_ff=352, L=2/2/2,
   V=1024; the smoke run uses d=128, L=1/2/1, V=256, N=2. Code paths are
   identical to the Tier configs; `count_params.py` uses the real Tier 0/1
   configs.
9. **Smoke-training data.** Pure uniform-random tokens are unlearnable
   (optimal loss = ln V at every step), so smoke data is random-token
   sequences generated by iterating a fixed random permutation of the vocab
   (x_{t+1} = perm[x_t]) — random tokens, but with deterministic first-order
   structure a correct harness must be able to fit. LR is a constant 3e-3;
   the plan's cosine schedule and 3-point LR sweep are for real Tier runs,
   not a 30-step smoke test. All other optimizer settings are the locked ones.

## Verified results (2026-11-24, this VM, CPU)

- `tests.py`: **all 4 tests PASS** for B0/B1/B2 (details in test 4 above;
  identity-equivalence max|diff| = 0.0).
- `count_params.py`: B2 matches the plan table **exactly** at 0.1M rounding
  on both tiers — Tier 0: unique 25.7M/42.5M, effective 64.2M/81.0M;
  Tier 1: unique 117.0M/146.4M, effective 351.0M/380.4M. Cross-checks:
  B0 unique = B2 effective and B1 unique = B2 unique on both tiers.
- `smoke_train.py`: mean loss first-3 → last-3 steps: B0 4.96 → 0.077,
  B1 4.82 → 0.067, B2 4.80 → 0.067. All decreased.
