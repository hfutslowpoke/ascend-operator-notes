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
| 本地源码 | `competition/mfg-2026/ops-gnn/source/` |
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

### 2.1 官方接口校正（2026-11-17）

| 赛题 | 官方接口 | 源码差异 |
|---|---|---|
| 06 `GatherCoo` | `src (M,K)` fp32/fp16/int32/int8，`index (E) int32`，`out (E,K)`，允许重复 | `reference`：ops-gnn 是 int64、rank1–8、arch35；官方是 int32 2D，910B 无 arch22 |
| 07 `SegmentMinCsr` | `src (E,K)`，`indptr int64`，`out/arg_out`，空段 `0/E`，首遇 tie-break | `direct`：arch22 `segment_min_csr` 已覆盖 arg_out/空段/tie-break |
| 08 `RandomWalk` | `rowptr/col/seed int64`，`rand (B,L)` float32 必选，p/q，孤立点停留 | `reference`：源码 arch35 用 Philox，不接受外部 rand |
| 09 `SpmmSum` | `rowptr int64`、`col int32`、`value/mat` fp32/fp16，空行 0 | `adapt`：源码是通用 CSR 聚合，需把 value/mat 映射到 op/reduce |
| 12 `GridCluster` | `pos (N,D)`，`size/start/end (D)`，`D∈{2,3}`，`stride[0]=1` | `reference`：源码 stride 顺序一致，但仅 arch35 |

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

### 3.1 补充来源：`HIT1920/torch_scatter_npu`（A2/A3 Segment）

- 仓库：https://gitcode.com/HIT1920/torch_scatter_npu
- 固定 commit：`0199ab0d836ea07e26d2914179a22bf3c793b62e`（2026-09-09）
- 许可证：根目录 LGPL-3.0；`Segment_Coo/` 为 CANN Open Software License Agreement 2.0。
- 为什么有用：
  1. `Segment_Csr/SegmentMinCsr` 是一个独立的 msOpGen/AclNN 工程，明确支持 Atlas A2/A3（910B/910C），有 host tiling、kernel、最小调用示例，比整仓 ops-gnn 更适合做 910B3 的快速构建验证；
  2. `Segment_Coo` 的 arch2201 kernel 提供 A2/A3 的 segment min/max 归约和 `(value, arg)` 输出，可作为 argmin/并列处理的第二套实现参考。
- 与官方语义的差距（不能直接提交）：
  - `indptr` 是 **int32**，官方要求 **int64**；
  - 只输出 `out`，**没有 `arg_out`**，官方要求 `out` + `arg_out (N,K) int64`；
  - 空段填 dtype 最大值，官方要求 `out=0`、`arg_out=E`；
  - `Segment_Coo` 的相等极值保留**最后一次**出现（last occurrence），官方要求**首遇**（严格小于才更新）；
  - `Segment_Coo` 是 COO/index 归约，不是官方 CSR 行切片。
- 本地副本：`supporting/torch_scatter_npu/`（15 个文件，含 LICENSE、两份 README、kernel/host/tiling、AclNN 调用示例、Segment_Coo arch2201 min kernel）；来源/边界见 [`PROVENANCE.md`](supporting/torch_scatter_npu/PROVENANCE.md)。
- 复用判断：只把它当**构建/调用骨架**和 kernel 写法参考；官方候选仍是本仓 `ops-gnn` 的 arch22 `segment_min_csr`，并需补齐 int64、arg_out、空段 0/E、首遇 tie-break。
- 本机状态：`not-run`；没有在 910B3 构建该仓。

## 4. 机制

### 4.1 SegmentMinCsr（07，A2/A3）

- 模板 `SegmentReduceDevice<T, SegmentReduce, HasArg>` 同时覆盖 `SegmentMin/Max/Sum`；
- 每个 unit 是一个 `(task, feature tile)`：`task` 是一个 CSR 行/分段，`tileK` 是特征列宽；
- 流式读 `tileRows × tileK` 的源行，按 `indptr` 在 UB 中归约；
- min/max 走极值路径，sum 走有序浮点路径；官方空段 `out=0`、`arg_out=E`；源码 `HasArg` 路径实现了 arg 输出与空段哨兵，需核对 `M=E`；
- 多 AIV 按 unit 切分，`ProcessSingleUnit`/`ProcessOutputGroup` 处理单输出/输出分组。

上游核对：pyg-lib `segment_min_csr` CPU 实现（commit `d13e206`，`segment_csr_kernel.cpp:279-404`）使用 `arg_out=E` 哨兵、`out` 初值 `numeric_limits::max()`、严格 `<` 首遇、空段 `out=0`，与官方逐条一致；详见 [算法参考核对](../../../docs/research/MFG2026_ALGORITHM_REFERENCES_20261117.md#6-segmentmincsrpyg-lib-源码与官方逐条一致)。这加强了 `direct` 分级，但仍需本机 910B3 构建。

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
- 本批未构建/未运行：`local_status=not-run`；没有作者性能数字。补充来源 `HIT1920/torch_scatter_npu` 同样未在本机构建。
- 构建 `ops-gnn` 需要 PyTorch/torch_npu + CANN 9.1 + CMake；全量构建较重，可先跑 `test/segment_csr/arch22` 与 `test/spmm/arch22`。
- 950-only 算子只能作为算法参考，不能在 910B3 上宣称通过。

## 6. 负例/风险

- **架构错配**：`gather_coo`/`random_walk`/`grid_cluster` 只有 arch35；直接抄到 910B 会编译失败或语义不完整。
- **GatherCoo 索引**：源码要求 int64、非降序、rank1–8；官方要求 int32 `[0,M)`、允许重复、2D `[M,K]`；两者不是同一契约。
- **RandomWalk 随机性**：官方要求外部 `rand` 输入，保证逐位可比；源码 Philox 计数器 RNG 不满足该契约。
- **GridCluster**：官方 `stride[0]=1`（dim0 最快），源码一致；但源码仅 arch35，910B 无实现。
- **Segment**：官方空段 `out=0`、`arg_out=E`；源码 `HasArg` 路径实现，需核对 `E` 与 `M` 的对应；`indptr` 非降序。`HIT1920/torch_scatter_npu` 的 SegmentMinCsr 空段填最大值、无 arg_out、int32 indptr，不能直接当官方实现。
- **Spmm**：A2 dtype 组合比 950 少；`CSR_ALG3` 只支持非转置；行分块/布局影响性能。

## 7. 复用判断

**优先复用**：Segment CSR 的模板化 min/max/sum、arg_out、流式行 tile、空段 0/哨兵；SpMM 的 reduce 枚举/行分组；`torch_scatter_npu` 的最小 A2/A3 msOpGen 工程可作构建骨架。

**需要验证**：910B3 上 `segment_min_csr`/`spmm` 的 arch22 构建与精度；官方 arg_out/空段/首遇 tie-break；SpmmSum 的 value/mat 映射。

**暂不使用**：`gather_coo`/`random_walk`/`grid_cluster` 的 arch35 代码直接上 910B；只能作为算法参考，移植需重写 SIMT/Vector 路径。

## 8. 下一步

1. 先用 `supporting/torch_scatter_npu/Segment_Csr/.../install.sh` 在 910B3 构建最小 SegmentMinCsr 工程，确认工具链；再构建 `ops-gnn` 的 A2 wheel（`CANN_TARGET=a2`），先跑 `test/segment_csr/arch22` 与 `test/spmm/arch22`；
2. 用 PyTorch/scatter 参考对照 min/max/sum、arg_out 与空段；重点验证官方首遇 tie-break 与 `E` 哨兵；
3. 对 950-only 算子，记录“不可在 910B 构建”的负例，不强行移植；
4. 官方已明确要求 GatherCoo/RandomWalk/GridCluster；需另写 910B 实现，不复用 arch35 二进制。
