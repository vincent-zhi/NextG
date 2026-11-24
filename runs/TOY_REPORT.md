# Toy 1 / Toy 2 — measured results

## Toy 1 — ST-tropical doubling with an SO(3)^2 payload

Accuracy is the fraction of graphs on which BOTH payload blocks are classified correctly (chance = 4.17%). `dist_mae` is the absolute error of the learned shortest-path distance.

| model | H4 T2 (in-dist) | H8 T3 | H16 T4 | H32 T5 | dist MAE H32 |
|---|---|---|---|---|---|
| st | 100.00% | 100.00% | 100.00% | 100.00% | 0.0005 |
| st_hardcore | (pending) | | | | |
| edge1hop | 100.00% |   3.25% |   3.25% |   5.00% | 9984.0000 |
| softmin | (pending) | | | | |
| mlp | (pending) | | | | |
| vector | (pending) | | | | |

**Gate (>=90% at H16/T4 and H32/T5): H16 100.00%, H32 100.00% -> PASS**

## Toy 2 — sudoku

| mode | acc(T8) | acc(T64) | full-grid | AUROC | retention | viol rate |
|---|---|---|---|---|---|---|
| full |  47.52% |  73.43% |   7.14% | 0.996 | 1.545 |  61.51% |
| open | (pending) | | | | | |
| primal | (pending) | | | | | |
| no_zero_lock | (pending) | | | | | |
| soft | (pending) | | | | | |

**Gates**: AUROC 0.996 (>=0.90 PASS); retention 1.545 (>=0.98 PASS); gain +47.5pt over best control (>=+15pt PASS).

## Toy 2 — sat

| mode | acc(T8) | acc(T64) | full-grid | AUROC | retention | viol rate |
|---|---|---|---|---|---|---|
| full | (pending) | | | | | |
| open | (pending) | | | | | |
| primal | (pending) | | | | | |
| no_zero_lock | (pending) | | | | | |
| soft | (pending) | | | | | |
