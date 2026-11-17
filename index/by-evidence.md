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
| `PARITY-CAPACITY-CHUNK` | `general` | [README](../general/equivariant/parity-ictc-o3/README.md) | not-reproduced; author report and docs only |
| `PARITY-MESH-ELECTROSTATICS` | `general` | [README](../general/equivariant/parity-ictc-o3/README.md) | not-reproduced; author report and docs only |
| `PARITY-O3-PACKED-TP` | `general` | [README](../general/equivariant/parity-ictc-o3/README.md) | not-reproduced; author report and docs only |
| `VLLMHUST-LINEAR-SWIGLU` | `general` | [README](../general/transformer/vllm-hust-opset/README.md) | not-reproduced; external report only |
