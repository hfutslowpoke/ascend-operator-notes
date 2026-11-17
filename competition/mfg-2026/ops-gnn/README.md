# ops-gnn：GatherCoo / SegmentMinCsr / RandomWalk / SpmmSum / GridCluster

> 层级：`competition` · 赛事：2026 制造行业算子专场 · 对应赛题：06、07、08、09、12 · 本机 910B3：`not-run`

## 1. 来源登记

| 项 | 值 |
|---|---|
| 仓库 | https://gitcode.com/cann/ops-gnn |
| 维护者 | CANN 开源社区 |
| 固定 commit | `ea0b18a1d3e97f1ae2315df69338bb51c9f96466`（2026-11-17） |
| 默认分支 | `master` |
| 许可证 | CANN Open Software License Agreement Version 2.0 |
| 本地缓存 | `repository-research-cache/mfg-2026/ops-gnn/` |
| 去重 | 与 ops-sparse 的 SpMM/SpGEMM 不同：ops-gnn 是 PyTorch 图算子封装，含 segment/gather/random_walk |

## 2. 硬件支持矩阵（关键）

`README.md` 和 `docs/zh/api_reference.md` 明确：

| 架构 | 可用算子 |
|---|---|
| A2/A3（arch22） | `segment_coo`、`segment_csr`、`spmm`、`bspmm`、`sddmm_csr`、`scatter_softmax`、`scatter_log_softmax` |
| 950（arch35） | `gather_coo`、`random_walk`、`radius`、`knn`、`nearest`、`grid_cluster`、`fps`、`matrix_exp` |

因此对 910B：

- **能直接参考/构建**：`segment_csr`（07）、`spmm`（09）；
- **只能看算法**：`gather_coo`（06）、`random_walk`（08）、`grid_cluster`（12）、`radius`/`fps`（11/13 的 950 版本）；
- `ops-gnn` 的 `radius`/`fps` 与 DrivingSDK 重复时，优先 DrivingSDK（A2 文档更明确）。

## 3. 源码定位

| 内容 | 路径/行 |
|---|---|
| 支持矩阵/API 总览 | `docs/zh/api_reference.md:178-199` |
| `gather_coo` 语义/约束 | `docs/zh/api_reference.md:533-563` |
| `gather_coo` kernel（slab/通用） | `csrc/npu/gather_coo/op_kernel/arch35/gather_coo_kernel.h:24-110` |
| `segment_csr` 模板/min/max/sum | `csrc/npu/segment_csr/op_kernel/arch22/segment_csr.cpp:22-61` |
| `segment_csr` int64 行拷贝 | `csrc/npu/segment_csr/op_kernel/arch22/segment_csr_common.h:35-61` |
| `random_walk` 语义 | `docs/zh/api_reference.md:688-730` |
| `random_walk` launch 参数 | `csrc/npu/random_walk/op_kernel/arch35/random_walk.h:18-33` |
| `random_walk` kernel | `csrc/npu/random_walk/op_kernel/arch35/random_walk.cpp`（本次 raw 403，路径由 tree 确认） |
| `grid_cluster` 网格量化/线性 ID | `csrc/npu/grid_cluster/op_kernel/arch35/grid_cluster.cpp:15-120,135-221` |
| `spmm` reduce 语义 | `csrc/npu/common/spmm/spmm_utils.h:210-241` |
| `radius` 语义（950） | `docs/zh/api_reference.md:441-472` |

机器索引：[source.json](source.json)。

本地源码副本：[`source/`](source/)（按上游路径保存，字节与固定 commit 一致；`source.json` 的 `local_path` 指向对应文件。上游原文的相对链接假设整仓，本地不保证可解析）。

## 4. 机制

### 4.1 SegmentMinCsr（07，A2/A3）

- 模板 `SegmentReduceDevice<T, SegmentReduce, HasArg>` 同时覆盖 `SegmentMin/Max/Sum`；
- 每个 unit 是一个 `(task, feature tile)`：`task` 是一个 CSR 行/分段，`tileK` 是特征列宽；
- 流式读 `tileRows × tileK` 的源行，按 `indptr` 在 UB 中归约；
- min/max 走极值路径，sum 走有序浮点路径；空段按 dtype 填最小值；
- 多 AIV 按 unit 切分，`ProcessSingleUnit`/`ProcessOutputGroup` 处理单输出/输出分组。

### 4.2 SpmmSum（09，A2/A3）

- `SpmmReduce` 枚举 sum/max/min/mean；`SpmmReducer` 给 half 等类型初值和 apply；
- CSR 行分组 + `copy_lhs` 消息传递；`spmm_utils.h` 用 `SpmmHostShape`/`CheckSpmmCsr` 做形状/索引校验；
- 与 `ops-sparse` 的 SpMM 相比，这层更贴近 PyTorch `torch_sparse` 接口。

### 4.3 GatherCoo（06，950 only）

- `GatherCooKernel` 支持 slab 快路径（连续重复行）和通用 tile；
- `ProcessOutputSlab`/`ProcessOriginalOutputSlab` 把重复行块一次写出，`ProcessGenericTile` 逐行处理；
- 纯前向位拷贝，无反向；`index` 必须 int64、非降序。

### 4.4 RandomWalk（08，950 only）

- Philox 计数器 RNG（`common/philox.h`），按 COO 图采样；
- 支持 uniform 与 node2vec 的 p/q 权重；`coalesced` 控制输入是否已按源节点排序；
- 输出节点序列 `[S, walk_length+1]`，可选边序列。

### 4.5 GridCluster（12，950 only）

- 每维 `floor((x-origin)/size)` 得到网格索引，再按各维 radix 组合成 int64 线性 ID；
- 对 subnormal/极小 size 做缩放和倒数快路径（`GridIsSubnormal`/`GridScaleTiny`/`GridIndexReciprocal`）；
- 3D 用 `Grid3Simt`，更高维用 `GridWideBody`；多核按点切分。

## 5. 910B 适用性与证据

- 只有 `segment_csr`/`spmm` 有 arch22 路径；其余为 arch35-only，**不能直接构建到 910B**。
- 本批未构建/未运行：`local_status=not-run`；没有作者性能数字。
- 构建 `ops-gnn` 需要 PyTorch/torch_npu + CANN 9.1 + CMake；全量构建较重，可先跑 `test/segment_csr/arch22` 与 `test/spmm/arch22`。
- 950-only 算子只能作为算法参考，不能在 910B3 上宣称通过。

## 6. 负例/风险

- **架构错配**：`gather_coo`/`random_walk`/`grid_cluster` 只有 arch35；直接抄到 910B 会编译失败或语义不完整。
- **GatherCoo 索引**：必须 int64、非降序；重复索引/空 Tensor/非连续输入要按文档。
- **RandomWalk 随机性**：必须固定 seed；p/q、coalesced、自环/重复边语义要与题面一致。
- **GridCluster**：极小 size/负坐标/int64 溢出/空点集；线性 ID 的维序必须与题面一致。
- **Segment**：空段填充最小值；NaN/极值语义；`indptr` 非降序。
- **Spmm**：A2 dtype 组合比 950 少；`CSR_ALG3` 只支持非转置；行分块/布局影响性能。

## 7. 复用判断

**优先复用**：Segment CSR 的模板化 min/max/sum、流式行 tile、空段填充；SpMM 的 reduce 枚举/行分组。

**需要验证**：910B3 上 `segment_csr`/`spmm` 的 arch22 构建与精度；`segment_min_csr` 的 NaN/空段语义。

**暂不使用**：`gather_coo`/`random_walk`/`grid_cluster` 的 arch35 代码直接上 910B；只能作为算法参考，移植需重写 SIMT/Vector 路径。

## 8. 下一步

1. 构建 `ops-gnn` 的 A2 wheel（`CANN_TARGET=a2`），先跑 `test/segment_csr/arch22` 与 `test/spmm/arch22`；
2. 用 PyTorch/scatter 参考对照 min/max/sum 与空段；
3. 对 950-only 算子，记录“不可在 910B 构建”的负例，不强行移植；
4. 若题面要求 GatherCoo/RandomWalk/GridCluster，另开 910B 实现，不复用 arch35 二进制。
