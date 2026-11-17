# ops-sparse：SpmmSum / Spspmm / Gather（部分 GatherCoo）

> 层级：`competition` · 赛事：2026 制造行业算子专场 · 对应赛题：06、09、10 · 本机 910B3：`not-run`

## 1. 来源登记

| 项 | 值 |
|---|---|
| 仓库 | https://gitcode.com/cann/ops-sparse |
| 维护者 | CANN 开源社区 |
| 固定 commit | `8fae00b8832bf31f1ce26059680bce67c6026c4f`（2026-11-17） |
| 默认分支 | `master` |
| 许可证 | CANN Open Software License Agreement Version 2.0 |
| 本地源码 | `competition/mfg-2026/ops-sparse/source/` |
| 去重 | 与 ops-gnn 的 PyTorch 层不同：ops-sparse 是 aclsparse Generic API/C++ 库 |

## 2. 对应赛题

### 2.0 官方题面校正（2026-11-17）

| 赛题 | 官方接口 | 源码差异 |
|---|---|---|
| 09 `SpmmSum` | `rowptr int64`、`col int32`、`value/mat` fp32/fp16、`out (M,K)`、空行 0、col 升序 | `adapt`：`aclsparseSpMM` 是通用 `alpha·A·B+beta·C`，需 alpha=1/beta=0 与 CSR 描述符 |
| 10 `Spspmm` | COO `indexA/valueA/indexB/valueB`，输出 COO 固定槽位 `U=m·n`，未命中 `(-1,-1)/0` | `adapt`：源码 `SpGEMM` 是 CSR 输入/输出 + 动态 nnz，需 COO↔CSR 与槽位层 |
| 06 `GatherCoo` | `src (M,K)`、`index int32`、`out (E,K)` | 不可用：`aclsparseGather` 是 dense gather，不是 COO 行扩展 |


### 2.1 SpmmSum（09）

```text
C = alpha * op(A) * op(B) + beta * C
```

- A 为 CSR 稀疏矩阵，B/C 为稠密矩阵；
- 支持 opA/opB 非转置/转置/共轭转置（算法组合有限制）；
- A2/A3 支持 FP32、FP16、BF16、COMPLEX64；`FP32_HIGH_PRECISION` 用 Kahan 补偿；
- 本题只关心 sum，但库支持 sum/max/min/mean（ops-gnn 层）与 SpMM 的通用乘法。

### 2.2 Spspmm（10）的核心：CSR SpGEMM

```text
C' = alpha * op(A) * op(B) + beta * C
```

- 这里是 SpGEMM 核心的 CSR 实现：A/B/C 均为 CSR；输出 CSR 列索引严格升序、无重复坐标；数值抵消的显式零保留；
- **官方 Spspmm 不是这个接口**：官方输入/输出是 COO，输出固定槽位 `U=m·n`，未命中槽位 `(-1,-1)/0`；必须加转换层；
- 多阶段：WorkEstimation → EstimateMemory → Compute（symbolic + numeric）→ Copy；
- A2/A3/950 支持。

### 2.3 Gather（06 的部分）

- `aclsparseGather` / `aclsparseGatherRaw`：按稀疏索引从稠密向量取值；
- 有 arch22 路径，但它是 **dense gather**，不是 COO 行扩展；不能直接当 `GatherCoo`。

## 3. 源码定位

| 内容 | 路径/行 |
|---|---|
| SpMM 语义/约束/算法 | `sparse/spmm/README.md:1-120` |
| SpMM host：dtype/行分块/布局 | `sparse/spmm/arch22/spmm_host.cpp:201-330` |
| SpMM group axpy kernel | `sparse/spmm/arch22/kernels/spmm_kernel_group_axpy.h` |
| SpMM fp32 Kahan | `sparse/spmm/arch22/kernels/spmm_kernel_compensated_math.h` |
| SpGEMM 多阶段/约束 | `sparse/spgemm/README.md:1-40` |
| SpGEMM host：symbolic/compute | `sparse/spgemm/arch22/spgemm_host.cpp:654-...`、`:1066-...` |
| SpGEMM symbolic kernel | `sparse/spgemm/arch22/spgemm_symbolic_kernel.cpp` |
| SpGEMM numeric/merge | `sparse/spgemm/arch22/spgemm_numeric_kernel.cpp`、`spgemm_merge_row_fastpaths.h` |
| Gather 语义/约束 | `sparse/gather/README.md:1-60` |
| Gather host（arch22） | `sparse/gather/arch22/gather_host.cpp` |

机器索引：[source.json](source.json)。

本地源码副本：[`source/`](source/)（按上游路径保存，字节与固定 commit 一致；`source.json` 的 `local_path` 指向对应文件。上游原文的相对链接假设整仓，本地不保证可解析）。

## 4. 机制

### 4.1 SpMM

- CSR 行分块：按 dtype、B/C 布局（行/列主序）、opA/opB 选择 `SelectSpmmRowBlock`；
- 行组 axpy：同一行的非零元对 B 的多个列做 group axpy，减少重复索引/循环；
- ALG1 通用布局、ALG2 紧凑行主序快路径、ALG3 固定原始行序（确定性）；
- FP32 高精度用 Kahan 补偿求和；复杂/半精度用对应路径；
- 预处理（可选）做行重排 + 分箱，加速后续调用。

### 4.2 SpGEMM

- WorkEstimation 统计每行/总中间乘积数 `numProds`；
- EstimateMemory 规划 workspace（ALG2/ALG3 的 chunkFraction）；
- Compute 先 symbolic 写 `rowOffsets`，再 numeric 做多路 merge、重复坐标相加、列索引排序；这是 CSR 输出语义，官方 COO 固定槽位需另写输出层；
- Copy 把内部列索引/values 写入调用方 CSR；
- 输出结构严格升序、无重复；显式零保留。

### 4.3 Gather

- 稀疏索引 `y[X.indices[i]-idxBase]` 写入 `X.values[i]`；
- `GatherRaw` 跳过描述符创建/销毁，适合高频小 gather；
- idxBase 0/1；索引固定 int32；与 COO 行扩展不是同一语义。

## 5. 910B 适用性与证据

- SpMM/SpGEMM 明确 A2/A3（arch22）；Gather 有 arch22；本批未构建/未运行：`local_status=not-run`。
- 构建 `ops-sparse` 需要 CANN 9.1 + CMake，产物为 aclsparse 库/测试；可先跑 `test/spmm`、`test/spgemm` 的 A2 用例。
- 没有作者性能数字；README 主要讲 API/约束。

## 6. 负例/风险

- **SpMM 算法选择**：CSR_ALG3 只支持非转置；不同布局/alg 的性能差异大；A2 dtype 组合比 950 少。
- **SpGEMM**：`numProds` 不是最终 `nnz(C)`；输出 CSR 需要动态分配/更新；大 nnz 可能超 INT32；多阶段状态机不能跳步。官方 Spspmm 还要求 COO 定槽位 `U=m·n≤250000` 和未命中 `(-1,-1)/0`。
- **Gather**：不是 `GatherCoo`；只有 dense gather，不处理 COO 行扩展/重复行 slab。
- **零维/空行**：SpMM A2 允许 M/K/N=0、nnz=0；SpGEMM 的输出结构/显式零语义要按 README。
- **构建成本**：全库较大，A2 测试可能依赖 CANN 版本；不要把编译成功当性能。

## 7. 复用判断

**优先复用**：SpMM 的行分块/group axpy/Kahan；SpGEMM 的 symbolic+numeric+merge 多阶段（作为 Spspmm 的 CSR 核心）；Gather 的 raw fast path（仅 dense gather）。

**需要验证**：910B3 上 A2 构建；SpMM 的 alg/布局 crossover；SpGEMM 小图/大图的 workspace/nnz。

**暂不使用**：把 Gather 当 GatherCoo；把 950 的 SpGEMM 确定性实现直接搬 910B（需确认 arch22 分支）。

## 8. 下一步

1. 构建 `ops-sparse` A2，跑 `test/spmm` 与 `test/spgemm` 的最小用例；
2. 用小图（对角/链/随机）验证 SpGEMM 输出结构、`numProds/nnz`、显式零；
3. SpMM 先 fp32，再 fp16/bf16/complex64；记录 alg/布局/行分块对时延的影响；
4. Gather 只作为 dense gather 参考，GatherCoo 另找/另写 910B 实现；
5. Spspmm 先写 COO↔CSR + 固定槽位输出层，再复用 SpGEMM 核心。
