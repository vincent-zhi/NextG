# NextG — research snapshot

This repository is a snapshot of the **looped-transformer F/E/G falsification
sprint** and the **toy-1 / toy-2 operator tests** run on a single
V100-SXM2-16GB. Everything here is text: source, logs, and JSON results.
Model checkpoints (`.pt`, 186–318 MB each) and the 583 MB FineWeb-Edu mini
shard are intentionally excluded (see `.gitignore`).

## Layout

| path | contents |
|---|---|
| `TRACK_C_LOCKED.md`, `README.md` | the original locked Track-C protocol and environment notes |
| `model.py`, `tracks_sprint.py`, `tracks_efg.py`, `train.py` | model / trainer implementations (B0/B1/B2, F V-cycle, G holographic) |
| `data_pipeline.py`, `stage3_prepare_fw.py` | synthetic graph/AST corpora + FineWeb-Edu shard prep |
| `test_efg.py`, `tests.py`, `test_sprint.py` | CI: causality, two-phase grads, Hadamard orthogonality, FP16 finiteness |
| `eval_stage1_f.py`, `eval_stage2_e.py`, `eval_stage3_g.py` | frozen evaluation crucibles |
| `run_stage1_f.sh`, `run_stage3_g.sh`, `sprint.sh`, `run_full.sh` | end-to-end run scripts |
| `make_report.py` | assembles `runs/FINAL_REPORT.md` from the JSON reports |
| `toys/toy1_semigroup.py` | ST-tropical semigroup doubling, `SO(3)^2` payload, 5 model variants |
| `toys/toy2_primal_dual.py` | primal-dual constraint inversion on 6x6 Sudoku and unique non-UP 3-SAT |
| `logs/` | raw stdout/stderr of every stage |
| `runs/RESULTS_HANDOFF.md` | **the full verified-results handoff** (start here) |
| `runs/*_report.json` | machine-readable Stage 1/2/3 verdicts |
| `runs/*/train_log.jsonl`, `runs/*/run.log` | per-step training logs and eval logs |

## Verified status at snapshot time (2026-11-25 00:15 CST)

- **Stage 1 (F, channel-space Matryoshka V-cycle): RED.**
  OOD 23.75% vs 25.625% (B2_THIN) and 24.375% (B2_NARROW); speed 0.653x;
  Nc 10 -> 20 delta 0.0 pt.
- **Stage 2 (E, latent verifier): RED (BASE_SLEEPING + NO_DIVERSITY_HEADROOM).**
  `acc(N=4)=acc(N=2)=25.625%`; max Oracle Pass@4 headroom 0.625 pt (< 5 pt).
- **Stage 3 (G, holographic coefficients, FineWeb-Edu): in progress.**
  - Static VRAM: 3.74x lower than B0_FULL (gate >= 3.5x **passes**).
  - DENSE_NARROW done: fw_bpb **1.761** @ step 2289 (the bar for G).
  - G_WHT training: live throughput ~0.33x DENSE_NARROW, so the speed gate
    has already failed on the DENSE-relative branch; only the `fw_bpb`
    comparison and the B0-relative speed branch remain.
- **Toy 1**: operator identity and the `edge1hop` control are verified
  analytically (`--smoke`); the learned end-to-end comparison is running.
- **Toy 2**: uniqueness / non-UP generators, zero-lock, and forward/loss smoke
  checks pass (`--smoke`); the full 5-mode comparison is running.

## Reproduce

```bash
# Stage 1
bash run_stage1_f.sh
# Stage 3 (requires the FineWeb shard; see stage3_prepare_fw.py)
bash run_stage3_g.sh
# Toy operator / solver checks (CPU, ~3 s / ~30 s)
python toys/toy1_semigroup.py --smoke
python toys/toy2_primal_dual.py --smoke
```

All gates were pre-registered; results in `runs/` are not renegotiated.
