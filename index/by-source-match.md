# 按官方源码匹配级别

由 `tools/build_index.py` 从 `index/catalog.json` 生成；不要手改。

只列出制造专场的 15 题映射与 2 个基础件；级别定义见 `competition/mfg-2026/README.md`。

> 同一题可能有多个候选实现（例如 SpmmSum 的 ops-gnn/ops-sparse）。`pass` 只表示本机测试通过，不一定是官方全 shape。

## `direct`

| 实现 | 官方题 | 条目 | 本机 |
|---|---|---|---|
| `MFG2026-FPS` | 点云计算：FurthestPointSample | [README](../competition/mfg-2026/driving-sdk/README.md) | `not-run` |
| `MFG2026-GNN-SEGMENT-MIN-CSR` | 图计算：SegmentMinCsr | [README](../competition/mfg-2026/ops-gnn/README.md) | `not-run` |

## `adapt`

| 实现 | 官方题 | 条目 | 本机 |
|---|---|---|---|
| `MFG2026-CYLINDER-QUERY` | 机器人抓取：CylinderQuery | [README](../competition/mfg-2026/driving-sdk/README.md) | `not-run` |
| `MFG2026-FFT2D` | 仿真计算：FourierTransform2D | [README](../competition/mfg-2026/ops-fft/README.md) | `pass` |
| `MFG2026-GNN-SPMM` | 图计算：SpmmSum | [README](../competition/mfg-2026/ops-gnn/README.md) | `not-run` |
| `MFG2026-MATH-STACK-BALL-QUERY` | 点云计算：BallQuery | [README](../competition/mfg-2026/ops-math/README.md) | `not-run` |
| `MFG2026-RADIUS` | 点云计算：Radius | [README](../competition/mfg-2026/driving-sdk/README.md) | `not-run` |
| `MFG2026-SPARSE-SPGEMM` | 图计算：Spspmm | [README](../competition/mfg-2026/ops-sparse/README.md) | `not-run` |
| `MFG2026-SPARSE-SPMM` | 图计算：SpmmSum | [README](../competition/mfg-2026/ops-sparse/README.md) | `not-run` |

## `reference`

| 实现 | 官方题 | 条目 | 本机 |
|---|---|---|---|
| `MFG2026-GNN-GATHER-COO` | 图计算：GatherCoo | [README](../competition/mfg-2026/ops-gnn/README.md) | `not-run` |
| `MFG2026-GNN-GRID-CLUSTER` | 点云计算：GridCluster | [README](../competition/mfg-2026/ops-gnn/README.md) | `not-run` |
| `MFG2026-GNN-RANDOM-WALK` | 图计算：RandomWalk | [README](../competition/mfg-2026/ops-gnn/README.md) | `not-run` |
| `MFG2026-MATH-COMPLEX-ABS` | 仿真计算：ComplexAbs | [README](../competition/mfg-2026/ops-math/README.md) | `not-run` |
| `MFG2026-SOLVER-COMPLEX-DIV` | 仿真计算：ComplexDiv | [README](../competition/mfg-2026/ops-solver/README.md) | `not-run` |

## `building-block`

| 实现 | 官方题 | 条目 | 本机 |
|---|---|---|---|
| `MFG2026-FFT1D` | 仿真计算：FourierTransform2D | [README](../competition/mfg-2026/ops-fft/README.md) | `not-run` |
| `MFG2026-MATH-COMPLEX-EXP-BLOCKS` | 仿真计算：ComplexExp | [README](../competition/mfg-2026/ops-math/README.md) | `not-run` |

## `not-applicable`

| 实现 | 官方题 | 条目 | 本机 |
|---|---|---|---|
| `MFG2026-SPARSE-GATHER` | 图计算：GatherCoo | [README](../competition/mfg-2026/ops-sparse/README.md) | `not-run` |
