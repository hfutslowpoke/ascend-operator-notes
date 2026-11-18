# 按本机证据

由 `tools/build_index.py` 从 `index/catalog.json` 生成；不要手改。

`pass` 只表示本机 910B3 构建/运行通过，不外推到 910C。

## `pass`

| 实现 | 层 | 条目 | 说明 |
|---|---|---|---|
| `BE4VER-PACO-BATCHED-JACOBI` | `general` | [README](../general/linear-algebra/be4ver-svd/README.md) | locally-validated on Ascend 910B3; see local_validation |
| `BE4VER-QDWH-SDC-FUSED` | `general` | [README](../general/linear-algebra/be4ver-svd/README.md) | locally-validated on Ascend 910B3; see local_validation |
| `KS2026-ASC910B-GROUPEDTOPK` | `competition` | [README](../competition/kernelswift-2026/README.md) | locally-validated on Ascend 910B3; see local_validation |
| `KS2026-ASC910B-SINKHORN` | `competition` | [README](../competition/kernelswift-2026/README.md) | locally-validated on Ascend 910B3; see local_validation |
| `MFG2026-FFT2D` | `competition` | [README](../competition/mfg-2026/ops-fft/README.md) | local 910B3 PASS: build CANN 9.1.0; 32x32 C2C impulse vs analytic DFT, best 0.1537 ms, max_abs 9.086799e-08, bad=0 |

## `pass-negative`

| 实现 | 层 | 条目 | 说明 |
|---|---|---|---|
| `XAYAH-ASC910B-MATMUL-NEG` | `general` | [README](../general/elementwise/52xayah-triton-kernels/README.md) | locally-validated negative on Ascend 910B3; see local_validation |
| `XAYAH-ASC910B-SOFTMAX-NEG` | `general` | [README](../general/elementwise/52xayah-triton-kernels/README.md) | locally-validated negative on Ascend 910B3; see local_validation |

## `partial`

| 实现 | 层 | 条目 | 说明 |
|---|---|---|---|
| `GITCODE-LINALG-LSTSQ` | `competition` | [README](../competition/elec-ops-inspection/README.md) | locally validated on small single-batch square cases; larger/batched cases failed; see local_validation |

## `fail`

| 实现 | 层 | 条目 | 说明 |
|---|---|---|---|
| `GITCODE-RECOMPUTE-WU` | `competition` | [README](../competition/cann-ops-competitions/README.md) | locally compiled but runtime failed on Ascend 910B3/CANN 9.1.0; author number not reproduced |
| `KS2026-ASC910B-FUSEDMOE` | `competition` | [README](../competition/kernelswift-2026/README.md) | local compile failed on Ascend 910B3; author number not reproduced |

## `offline-only`

| 实现 | 层 | 条目 | 说明 |
|---|---|---|---|
| `GITCODE-CONCAT` | `competition` | [README](../competition/cann-ops-competitions/README.md) | offline layout/address tests passed locally; no NPU build/run |

## `not-run`

| 实现 | 层 | 条目 | 说明 |
|---|---|---|---|
| `HANSEN-2026POWER-FIRE-R2` | `competition` | [README](../competition/hansen-2026-power/README.md) | not-reproduced; static source and public score only |
| `HANSEN-2026POWER-SPLINE-R1` | `competition` | [README](../competition/hansen-2026-power/README.md) | not-reproduced; static source and public score only |
| `HANSEN-2026POWER-TREE-R1` | `competition` | [README](../competition/hansen-2026-power/README.md) | not-reproduced; static source and public score only |
| `MAMBA2-ASCENDC-CHUNKMIX` | `general` | [README](../general/attention/mamba-ascendc/README.md) | not-reproduced; author device event/profiler |
| `MFG2026-CYLINDER-QUERY` | `competition` | [README](../competition/mfg-2026/driving-sdk/README.md) | not-run; A2-targeted generic path; build pending |
| `MFG2026-FFT1D` | `competition` | [README](../competition/mfg-2026/ops-fft/README.md) | building block only; not an end-to-end official-problem run |
| `MFG2026-FPS` | `competition` | [README](../competition/mfg-2026/driving-sdk/README.md) | not-run; docs A2/A3/950 |
| `MFG2026-GNN-GATHER-COO` | `competition` | [README](../competition/mfg-2026/ops-gnn/README.md) | not-run; 950-only reference |
| `MFG2026-GNN-GRID-CLUSTER` | `competition` | [README](../competition/mfg-2026/ops-gnn/README.md) | not-run; 950-only reference |
| `MFG2026-GNN-RANDOM-WALK` | `competition` | [README](../competition/mfg-2026/ops-gnn/README.md) | not-run; 950-only reference |
| `MFG2026-GNN-SEGMENT-MIN-CSR` | `competition` | [README](../competition/mfg-2026/ops-gnn/README.md) | Supporting alt project build+run PASS on its own 8x2048 sample; main ops-gnn implementation still not-run; official int64/arg_out/empty-0/first-tie semantics not covered by the alt project. |
| `MFG2026-GNN-SPMM` | `competition` | [README](../competition/mfg-2026/ops-gnn/README.md) | not-run; A2/A3 arch22 exists |
| `MFG2026-MATH-COMPLEX-ABS` | `competition` | [README](../competition/mfg-2026/ops-math/README.md) | not-run; 950-only reference |
| `MFG2026-MATH-COMPLEX-EXP-BLOCKS` | `competition` | [README](../competition/mfg-2026/ops-math/README.md) | building block composition; no end-to-end official-problem run |
| `MFG2026-MATH-STACK-BALL-QUERY` | `competition` | [README](../competition/mfg-2026/ops-math/README.md) | not-run; A2 config exists |
| `MFG2026-RADIUS` | `competition` | [README](../competition/mfg-2026/driving-sdk/README.md) | not-run; docs A2/A3/950 |
| `MFG2026-SOLVER-COMPLEX-DIV` | `competition` | [README](../competition/mfg-2026/ops-solver/README.md) | not-run; 950 SIMT reference |
| `MFG2026-SPARSE-GATHER` | `competition` | [README](../competition/mfg-2026/ops-sparse/README.md) | not-run; A2 arch22 exists |
| `MFG2026-SPARSE-SPGEMM` | `competition` | [README](../competition/mfg-2026/ops-sparse/README.md) | not-run; A2/A3 arch22 exists |
| `MFG2026-SPARSE-SPMM` | `competition` | [README](../competition/mfg-2026/ops-sparse/README.md) | not-run; A2/A3 arch22 exists |
| `PARITY-CAPACITY-CHUNK` | `general` | [README](../general/equivariant/parity-ictc-o3/README.md) | not-reproduced; author report and docs only |
| `PARITY-MESH-ELECTROSTATICS` | `general` | [README](../general/equivariant/parity-ictc-o3/README.md) | not-reproduced; author report and docs only |
| `PARITY-O3-PACKED-TP` | `general` | [README](../general/equivariant/parity-ictc-o3/README.md) | not-reproduced; author report and docs only |
| `VLLMHUST-LINEAR-SWIGLU` | `general` | [README](../general/transformer/vllm-hust-opset/README.md) | not-reproduced; external report only |
