# Toy 1 / Toy 2 — measured results

## Toy 1 — ST-tropical doubling with an SO(3)^2 payload

Accuracy is the fraction of graphs on which BOTH payload blocks are classified correctly (chance = 4.17%). `dist_mae` is the absolute error of the learned shortest-path distance.

| model | H1 T1 | H4 T2 (in-dist) | H8 T3 | H16 T4 | H32 T5 | dist MAE H32 |
|---|---|---|---|---|---|---|
| st | n/a | 100.00% | 100.00% | 100.00% | 100.00% | 0.0005 |
| st_hardcore | 100.00% |  95.00% |  77.75% |  52.75% |  20.50% | 14.2555 |
| st_hardcore_warm |  23.25% |  24.75% |  26.00% |  23.25% |  21.75% | 13.6315 |
| edge1hop | n/a | 100.00% |   3.25% |   3.25% |   5.00% | 9984.0000 |
| softmin | n/a | 100.00% | 100.00% | 100.00% | 100.00% | 0.3263 |
| mlp | n/a |   4.50% |   4.50% |   5.00% |   3.50% | 0.0001 |
| vector |  42.25% |   5.50% |   3.25% |   5.75% |   4.75% | 0.0000 |
| st_hardcore_perm | 100.00% |  99.25% |  93.00% |  85.50% |  57.00% | 13.7240 |
| st_perm | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.1313 |

`st_hardcore` seed spread (H8 / H32 acc): 78/20 (MAE/H32 0.445); 100/100 (MAE/H32 0.453); 25/5 (MAE/H32 0.473).  The metric offset is constant per hop at every seed; routing reliability is not.  D-supervised `st` seeds 0/1: H32 100%/100%, dist MAE 0.0005/0.0001 -- supervision removes both the scale freedom and the seed lottery.

Rows ending in `_perm` are the index-order control: every eval graph gets a random node permutation, so the spine no longer occupies the lowest indices.  Accuracy should be unchanged if the ST routing is a real signal rather than an argmin tie-break.

**Gate (>=90% at H16/T4 and H32/T5): H16 100.00%, H32 100.00% -> PASS**

## Toy 2 — sudoku

AUROC is per constraint unit (Sudoku: 18 row/col/box units; SAT: per clause), computed on held-out solution grids. `retention` is the true conditional P(board correct at T=64 | board correct at T=8); `acc ratio` is acc(T64)/acc(T8). `untrained` = the same solver on a never-trained encoder (same init seed): the control for whether anything is learned.

| mode | acc(T4) | acc(T8) | acc(T64) | full-grid | AUROC | retention P(ok64|ok8) | acc ratio | viol rate |
|---|---|---|---|---|---|---|---|---|
| full |  38.61% |  47.52% |  73.43% |   7.14% | 0.996 | n/a | 1.545 |  61.51% |
| open |  22.61% |  22.61% |  22.61% |   0.00% | 1.000 | n/a | 1.000 |  96.03% |
| primal |  36.14% |  40.92% |  66.50% |  17.86% | 0.831 | n/a | 1.625 |  45.44% |
| no_zero_lock | killed 2026-11-25 (stop directive) | | | | | | | |
| soft | killed 2026-11-25 (stop directive) | | | | | | | |
| untrained |  34.82% |  43.07% |  77.56% |  10.71% | 0.999 | n/a | 1.801 |  55.56% |

**Gates**: AUROC 0.996 (>=0.90 PASS); true retention n/a (>=0.98 FAIL); gain +4.5pt over best control (>=+15pt FAIL). Untrained solver baseline: acc(T8)  43.07%, so the learned gain is +4.5pt. Across 3 untrained inits: T8 43.1-47.1%, T64 74.8-77.6%. Trained `full` across 3 seeds: T8 46.5-47.5%, T64 73.4-77.2%.


**Neural verifier V_theta (Protocol B)** — encoder left at its random init; V_theta trained on 2-cell hard negatives. Held-out board AUROC 1.0. Solver with the learned V:

| T | acc_cells | acc_full | viol rate |
|---|---|---|---|
| T4 |  27.89% |   0.00% |  98.02% |
| T8 |  31.19% |   0.00% |  93.65% |
| T16 |  31.02% |   0.00% |  95.24% |
| T32 |  31.68% |   0.00% |  91.87% |
| T64 |  29.54% |   0.00% |  92.26% |

## Toy 2 — sat

AUROC is per constraint unit (Sudoku: 18 row/col/box units; SAT: per clause), computed on held-out solution grids. `retention` is the true conditional P(board correct at T=64 | board correct at T=8); `acc ratio` is acc(T64)/acc(T8). `untrained` = the same solver on a never-trained encoder (same init seed): the control for whether anything is learned.

| mode | acc(T4) | acc(T8) | acc(T64) | full-grid | AUROC | retention P(ok64|ok8) | acc ratio | viol rate |
|---|---|---|---|---|---|---|---|---|
| full | (pending) | | | | | | | |
| open |  48.67% |  48.67% |  48.67% |   0.00% | 1.000 | n/a | 1.000 |  12.24% |
| primal | (pending) | | | | | | | |
| no_zero_lock | killed 2026-11-25 (stop directive) | | | | | | | |
| soft | killed 2026-11-25 (stop directive) | | | | | | | |
| untrained |  57.67% |  55.67% |  52.67% |   4.00% | 0.972 | n/a | 0.946 |  13.36% |
