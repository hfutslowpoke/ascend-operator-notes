# 按算子族

由 `tools/build_index.py` 从 `index/catalog.json` 生成；不要手改。

| 算子族 | 实现 | 层 | 条目 | 本机 910B3 |
|---|---|---|---|---|
| `attention-recompute-gemm` | `GITCODE-RECOMPUTE-WU` | `competition` | [README](../competition/cann-ops-competitions/README.md) | `fail` |
| `batched-least-squares` | `GITCODE-LINALG-LSTSQ` | `competition` | [README](../competition/elec-ops-inspection/README.md) | `partial` |
| `candidate-decode-fusion` | `HANSEN-2026POWER-FIRE-R2` | `competition` | [README](../competition/hansen-2026-power/README.md) | `not-run` |
| `concat-layout` | `GITCODE-CONCAT` | `competition` | [README](../competition/cann-ops-competitions/README.md) | `offline-only` |
| `equivariant-tensor-product` | `PARITY-O3-PACKED-TP` | `general` | [README](../general/equivariant/parity-ictc-o3/README.md) | `not-run` |
| `gemm` | `XAYAH-ASC910B-MATMUL-NEG` | `general` | [README](../general/elementwise/52xayah-triton-kernels/README.md) | `pass-negative` |
| `geometric-argmin` | `HANSEN-2026POWER-TREE-R1` | `competition` | [README](../competition/hansen-2026-power/README.md) | `not-run` |
| `graph-spline-conv` | `HANSEN-2026POWER-SPLINE-R1` | `competition` | [README](../competition/hansen-2026-power/README.md) | `not-run` |
| `linear-activation-fusion` | `VLLMHUST-LINEAR-SWIGLU` | `general` | [README](../general/transformer/vllm-hust-opset/README.md) | `not-run` |
| `long-range-electrostatics` | `PARITY-MESH-ELECTROSTATICS` | `general` | [README](../general/equivariant/parity-ictc-o3/README.md) | `not-run` |
| `memory-scheduled-graph-message` | `PARITY-CAPACITY-CHUNK` | `general` | [README](../general/equivariant/parity-ictc-o3/README.md) | `not-run` |
| `moe-expert-compute` | `KS2026-ASC910B-FUSEDMOE` | `competition` | [README](../competition/kernelswift-2026/README.md) | `fail` |
| `moe-routing` | `KS2026-ASC910B-GROUPEDTOPK` | `competition` | [README](../competition/kernelswift-2026/README.md) | `pass` |
| `normalization` | `KS2026-ASC910B-SINKHORN` | `competition` | [README](../competition/kernelswift-2026/README.md) | `pass` |
| `polar-eigendecomposition` | `BE4VER-QDWH-SDC-FUSED` | `general` | [README](../general/linear-algebra/be4ver-svd/README.md) | `pass` |
| `softmax` | `XAYAH-ASC910B-SOFTMAX-NEG` | `general` | [README](../general/elementwise/52xayah-triton-kernels/README.md) | `pass-negative` |
| `ssm-chunked-scan` | `MAMBA2-ASCENDC-CHUNKMIX` | `general` | [README](../general/attention/mamba-ascendc/README.md) | `not-run` |
| `svd` | `BE4VER-PACO-BATCHED-JACOBI` | `general` | [README](../general/linear-algebra/be4ver-svd/README.md) | `pass` |
