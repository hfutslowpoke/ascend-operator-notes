# 按算子族

由 `tools/build_index.py` 从 `index/catalog.json` 生成；不要手改。

| 算子族 | 实现 | 层 | 条目 | 本机 910B3 |
|---|---|---|---|---|
| `attention-backward` | `STARCUP-2026-A-DENSELIGHTNING` | `competition` | [README](../competition/starcup-2026-a-group/dense-lightning-indexer-grad-kl-loss/README.md) | `platform-only` |
| `attention-backward` | `STARCUP-2026-FINAL-SFAG` | `competition` | [README](../competition/starcup-2026-final/sparse-flash-attention-grad/README.md) | `platform-only` |
| `attention-recompute-gemm` | `GITCODE-RECOMPUTE-WU` | `competition` | [README](../competition/cann-ops-competitions/README.md) | `fail` |
| `batched-least-squares` | `GITCODE-LINALG-LSTSQ` | `competition` | [README](../competition/elec-ops-inspection/README.md) | `partial` |
| `candidate-decode-fusion` | `HANSEN-2026POWER-FIRE-R2` | `competition` | [README](../competition/hansen-2026-power/README.md) | `not-run` |
| `complex-elementwise` | `MFG2026-MATH-COMPLEX-ABS` | `competition` | [README](../competition/mfg-2026/ops-math/README.md) | `not-run` |
| `complex-elementwise` | `MFG2026-MATH-COMPLEX-EXP-BLOCKS` | `competition` | [README](../competition/mfg-2026/ops-math/README.md) | `not-run` |
| `complex-elementwise` | `MFG2026-SOLVER-COMPLEX-DIV` | `competition` | [README](../competition/mfg-2026/ops-solver/README.md) | `not-run` |
| `concat-layout` | `GITCODE-CONCAT` | `competition` | [README](../competition/cann-ops-competitions/README.md) | `offline-only` |
| `equivariant-tensor-product` | `PARITY-O3-PACKED-TP` | `general` | [README](../general/equivariant/parity-ictc-o3/README.md) | `not-run` |
| `fourier-transform` | `MFG2026-FFT1D` | `competition` | [README](../competition/mfg-2026/ops-fft/README.md) | `not-run` |
| `fourier-transform` | `MFG2026-FFT2D` | `competition` | [README](../competition/mfg-2026/ops-fft/README.md) | `pass` |
| `gemm` | `XAYAH-ASC910B-MATMUL-NEG` | `general` | [README](../general/elementwise/52xayah-triton-kernels/README.md) | `pass-negative` |
| `geometric-argmin` | `HANSEN-2026POWER-TREE-R1` | `competition` | [README](../competition/hansen-2026-power/README.md) | `not-run` |
| `graph-gather` | `MFG2026-GNN-GATHER-COO` | `competition` | [README](../competition/mfg-2026/ops-gnn/README.md) | `not-run` |
| `graph-gather` | `MFG2026-SPARSE-GATHER` | `competition` | [README](../competition/mfg-2026/ops-sparse/README.md) | `not-run` |
| `graph-random-walk` | `MFG2026-GNN-RANDOM-WALK` | `competition` | [README](../competition/mfg-2026/ops-gnn/README.md) | `not-run` |
| `graph-segment-reduction` | `MFG2026-GNN-SEGMENT-MIN-CSR` | `competition` | [README](../competition/mfg-2026/ops-gnn/README.md) | `not-run` |
| `graph-spline-conv` | `HANSEN-2026POWER-SPLINE-R1` | `competition` | [README](../competition/hansen-2026-power/README.md) | `not-run` |
| `late-interaction` | `SHANGHE-2026-PRELIM-BMMS` | `competition` | [README](../competition/shanghe-2026-prelim/batchmatmulmaxsum/README.md) | `platform-only` |
| `linear-activation-fusion` | `VLLMHUST-LINEAR-SWIGLU` | `general` | [README](../general/transformer/vllm-hust-opset/README.md) | `not-run` |
| `long-range-electrostatics` | `PARITY-MESH-ELECTROSTATICS` | `general` | [README](../general/equivariant/parity-ictc-o3/README.md) | `not-run` |
| `memory-scheduled-graph-message` | `PARITY-CAPACITY-CHUNK` | `general` | [README](../general/equivariant/parity-ictc-o3/README.md) | `not-run` |
| `moe-expert-compute` | `KS2026-ASC910B-FUSEDMOE` | `competition` | [README](../competition/kernelswift-2026/README.md) | `fail` |
| `moe-routing` | `KS2026-ASC910B-GROUPEDTOPK` | `competition` | [README](../competition/kernelswift-2026/README.md) | `pass` |
| `normalization` | `KS2026-ASC910B-SINKHORN` | `competition` | [README](../competition/kernelswift-2026/README.md) | `pass` |
| `point-cloud-clustering` | `MFG2026-GNN-GRID-CLUSTER` | `competition` | [README](../competition/mfg-2026/ops-gnn/README.md) | `not-run` |
| `point-cloud-query` | `MFG2026-CYLINDER-QUERY` | `competition` | [README](../competition/mfg-2026/driving-sdk/README.md) | `not-run` |
| `point-cloud-query` | `MFG2026-MATH-STACK-BALL-QUERY` | `competition` | [README](../competition/mfg-2026/ops-math/README.md) | `not-run` |
| `point-cloud-query` | `MFG2026-RADIUS` | `competition` | [README](../competition/mfg-2026/driving-sdk/README.md) | `not-run` |
| `point-cloud-sampling` | `MFG2026-FPS` | `competition` | [README](../competition/mfg-2026/driving-sdk/README.md) | `not-run` |
| `polar-eigendecomposition` | `BE4VER-QDWH-SDC-FUSED` | `general` | [README](../general/linear-algebra/be4ver-svd/README.md) | `pass` |
| `softmax` | `XAYAH-ASC910B-SOFTMAX-NEG` | `general` | [README](../general/elementwise/52xayah-triton-kernels/README.md) | `pass-negative` |
| `sparse-dense-matmul` | `MFG2026-GNN-SPMM` | `competition` | [README](../competition/mfg-2026/ops-gnn/README.md) | `not-run` |
| `sparse-dense-matmul` | `MFG2026-SPARSE-SPMM` | `competition` | [README](../competition/mfg-2026/ops-sparse/README.md) | `not-run` |
| `sparse-softmax` | `STARCUP-2026-A-SPARSESOFTMAX` | `competition` | [README](../competition/starcup-2026-a-group/sparse-softmax/README.md) | `platform-only` |
| `sparse-sparse-matmul` | `MFG2026-SPARSE-SPGEMM` | `competition` | [README](../competition/mfg-2026/ops-sparse/README.md) | `not-run` |
| `ssm-chunked-scan` | `MAMBA2-ASCENDC-CHUNKMIX` | `general` | [README](../general/attention/mamba-ascendc/README.md) | `not-run` |
| `svd` | `BE4VER-PACO-BATCHED-JACOBI` | `general` | [README](../general/linear-algebra/be4ver-svd/README.md) | `pass` |
