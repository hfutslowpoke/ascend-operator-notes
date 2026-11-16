# GitCode 实现的三类可迁移机制：任务批处理、布局优先搬运、补偿数值

本文从 GitCode 两个官方竞赛仓的 3 份实现提炼机制，面向 910B/910C 的固定 Shape Case 优化。源码对应固定 commit：

- `cann/cann-ops-competitions@5126fead3327c7544ed4becff93775505071334c`
  - `recompute_w_u`（Delicate02）
  - `Concat`（BatchLions）
- `cann/elec-ops-inspection@dfe55921db351f71cfb8c96f795fd4472bd6bd47`
  - `LinalgLstsq`（XDU_XDU-BTD）

性能数字区分作者 `O+A` 与本机 `V`；本机只验证了 Lstsq 的三个小尺寸单 batch 方阵，其余是编译/运行负例或离线测试。

## 1. 机制 A：按 head/chunk 合并任务，复用共享 K，做多槽 Vector/Cube 流水

### 1.1 源码

`recompute_w_u`：

- `op_kernel/recompute_wu.h:43-50`：`MAX_CHUNKS_PER_TASK=2`、`MAX_BATCH_CHUNKS_PER_TASK=4`、`MAX_HEADS_PER_TASK=4`、`PIPELINE_DEPTH=5`、`GROUP_PIPELINE_DEPTH=3`；
- `:591-684`：batch chunks，按 group head 复用 K（`:642-647`），按 head/chunk 展开 `A*beta` 与 `A*beta*exp(g)`；
- `:686-792`：D128 group pipeline，3 个 group slot、4 个 head 一组；
- `:1025-1105`：Cube `Mmad + Fixpipe`，D128 四槽（`:1107-1147`）；
- `:1313-1395`：Cube 侧 group pipeline，共享 L0B K tile（`:1359-1363`）；
- `op_host/recompute_wu_tiling.cpp:204-345`：按 `groupSize/taskCount/blockDim/keyDim/valueDim` 选 task 粒度；
- `op_host/recompute_wu_tiling.cpp:346-390`：按 `pipelineDepth/chunksPerTask/headsPerTask` 算 workspace。

### 1.2 改变了什么

- 同一个 GQA group 的 K 只打包一次，多个 query head 复用；
- 一个任务覆盖最多 4 个 head × 4 个 chunk，减少任务启动/同步次数；
- Vector 侧预处理与 Cube 侧 GEMM 通过多槽 workspace 流水，减少串行等待；
- D128 四槽把共享 K 固定在 L0B，减少重复 `MTE2`；
- staged output 处理非 16 对齐的 K/V，避免末尾残块卡死。

算术量仍是 `A @ (v*beta)` 与 `A @ (k*beta*e^g)`；减少的是**重复读 K、任务启动、跨核等待、尾部搬运**。

### 1.3 适用条件

- GQA/MQA，多个 query head 共享 key；
- `D=128` 或 `256`，`BT=64`，`B=1`；
- 能接受单 kernel 内复杂 flag/workspace；
- 目标 CANN 与作者 8.5.0 接近，或愿意逐 key 做移植回归。

### 1.4 代价与负例

- 10 个 tiling key，每个 key 的 workspace/flag 都不同；全 key 回归成本高；
- workspace 随 `pipelineDepth × chunksPerTask × headsPerTask` 增长；
- 本机 CANN 9.1.0 编译成功但运行 `507057`（AI Core kernel execution failed），5 个 shape 全失败；显式 `KERNEL_TASK_TYPE` 也未修复；
- 因此“编译成功”不能作为可运行/可复用证据；必须先复跑作者 CANN 8.5，或定位 9.1 的 ABI/同步差异。

## 2. 机制 B：先把 shape 归一成 `rows×n_i`，再按布局选搬运策略

### 2.1 源码

`Concat`：

- `op_host/concat.cpp:55-263`：Host 取动态输入、归一 `rows/nOut/lens`，处理负轴、零长度、单路非空；
- `:240-262`：单路非空且行 ≤128B 时，把 `rows` 并进列，走整块拷贝；
- `:267-330`：策略选择（按行/按列/DMA 拼装/Gather 拼装/输出线性切分）；
- `:440-532`：列带、`CheckConcatFinal` 收敛/拒绝；
- `op_host/s9_concat_layout.h:96-117,163-241,355-435,603-696`：UB 预算、核映射、带区间、`PlanConcatStage`；
- `op_host/s9_tiling_calc.h:493-620,752-997,1235-1410`：`MaxInputsPerBand`、`GatherFitsBands`、`CalcConcatLens`、`CalcConcatBandPlan`、`CheckConcatFinal`；
- `op_kernel/concat.cpp:264-630`：Kernel 初始化、核/带/布局；
- `:909-1071`：直写带路径；
- `:1075-1445`：Gather 拼装路径（连续读 + 偏移表 + 一次连续写）；
- `:1427-1500`：tile 主循环。

### 2.2 改变了什么

- 多路小片段不再“每路每行一条小 DMA”，而是按带内相交路连续读、Gather 拼成整行、一次连续写；
- 列带让一行只占 `bw·esz`，同样 UB 能装更多行，读 DMA 变大；
- 单路非空窄行整块拷贝，精确等价，作者实测 1.54×–674×（910B1，行宽 ≤128B）；
- L2 默认 `NORMAL`，作者实测小 shape 11.10→8.44 µs、大 shape 37.74→26.62 µs；
- Host/Kernel 共用纯 C++ 布局头，避免“Host 说放得下、Kernel 越界”。

### 2.3 适用条件

- 动态输入列表、最多 256 路；
- 行数/列宽/路数变化大，单一策略会退化；
- 能接受复杂 UB 预算和 Host/Kernel 共享布局；
- 目标 910B，CANN 8.5 附近。

### 2.4 代价与负例

作者自己的 A/B 给出了多条负结果：

- 均衡列带（`S9_BAL=1`）在 G16 246 路上慢 **1.35%**：关键路径由最宽带决定，空转核不浪费带宽；
- 边界路裁剪（`S9_TRIM`）在三个形状上慢 8.0%/7.9%/1.0%：少读的字节抵不上跨步 DMA；
- 整块取 TilingData 头慢 **8%**：连续标量头本来就在同一 cache line；
- 早期 strategy 5（按路×行段 + 大块 DMA）在 246 路上慢 **4.33×**，代码已删除；
- 整块拷贝只对窄行有利，≥256B 行会慢 11%–19%；
- 本机只跑了离线测试（215,242 + 32,681 + 8,247 项通过），没有 NPU 构建/运行。

## 3. 机制 C：按 shape 选 solver + 补偿算术 + 迭代精化

### 3.1 源码

`LinalgLstsq`：

- `op_host/linalg_lstsq.cpp:24-91`：shape 校验、blockDim、tiling key、workspace；
- `op_kernel/linalg_lstsq.cpp:25-90`：TwoSum/TwoProduct/补偿 Divide；
- `:223-335`：按 `SHAPE` 分派 tiny/packed/200/256/generic；
- `op_kernel/lstsq_compensated_tile_product.h:19-40`：高低位拆分乘积；
- `op_kernel/lstsq_shared_refinement.h:19-112`：tiny/64/方阵迭代精化；
- `op_kernel/lstsq_shared_residual.h:17-330`：残差补偿归约；
- `op_kernel/qr/lstsq_qr_cube_compute.h:17-266`：QR/多项式/Consumer；
- `op_kernel/square/lstsq_square256_solve.h:17-228`：256 方阵列三角求解；
- `op_kernel/tiny/lstsq_tiny15_fast_compute.h:17-303`：15 阶快路 + 主元保护；
- `op_kernel/lstsq_shared_workspace.h:17-451`：workspace/输出/残差调度。

### 3.2 改变了什么

- 小尺寸（15/16/50/64）走专用快路，减少通用 solver 的固定开销；
- 200/256 走 Cube QR + 4 batch 共享因子，多个 RHS 复用分解；
- 补偿算术把 FP32 乘加拆成 high/low，降低消元/回代的抵消误差；
- 迭代精化用原方程残差修正解，允许快路在条件数变差时仍满足精度；
- 主元/有限性/秩保护在不满足时回退，避免“快但错”。

### 3.3 适用条件

- FP32、列满秩、`M>=N`、`K<=64`；
- 小尺寸单 batch 方阵：本机已验证 16/50/64；
- 能接受 shape 专用代码和每 key 回归；
- 需要比朴素 FP32 更稳的消元/回代。

### 3.4 代价与负例

- 作者 README 没有逐 case 时延/榜单，无法证明“高性能”；
- CANN 9.1 需要两处 `template` 兼容性 patch 才能编译；
- 本机多 batch（16×16 b=2、50×50 b=4）、矩形（64×32、64×50、100×64）、200/256 均崩溃：
  - 小/矩形：`507035` vector core exception / invalid GM address；
  - 200/256：`507057` aicore exception / MTE invalid GM address；
- 因此“小尺寸 PASS”不能外推到全 shape；矩形残差布局/workspace 仍可能是本机驱动或版本差异。

## 4. 横向比较

| 维度 | recompute_w_u | Concat | LinalgLstsq |
|---|---|---|---|
| 主瓶颈 | K 重复读、Vector/Cube 串行、尾部搬运 | 多路小 DMA、写碎片、UB 容量 | 消元/回代的固定开销与舍入 |
| 核心动作 | head/chunk 合并 + GQA K 复用 + 多槽流水 | 归一 rows×n_i + 按布局选路 + Gather/分带 | shape 专用 solver + 补偿算术 + 精化 |
| 省掉什么 | 重复 K 打包、任务启动、跨核等待 | 小 DMA、碎片写、重复读 | 通用 solver 固定开销、部分舍入误差 |
| 增加什么 | workspace、flag、10 个 key | UB 账、偏移表、Host/Kernel 共享布局 | 代码体量、每 key 回归、补偿指令 |
| 本机结论 | 编译成功、运行失败 | 离线测试通过、未上 NPU | 小尺寸单 batch PASS，大/多 batch 失败 |

## 5. 选择表

| 现象 | 优先机制 | 先验证什么 |
|---|---|---|
| 多头共享 K、K 重复读 | head/chunk 合并 + L0B 复用 | K 打包次数、L0B 命中、workspace |
| Vector 预处理与 Cube GEMM 串行 | 多槽 workspace + 跨核 flag | 槽数、flag 死锁、尾部 shape |
| 多路小片段拼输出 | 归一 rows×n_i + Gather 拼装 | 偏移表、32B 对齐、UB 账 |
| 单路非空窄行 | 整块连续拷贝 | 行宽阈值、输出覆盖 |
| 形状变化大 | 列带/按行/按列多策略 | 每策略的 crossover、负载均衡 |
| FP32 消元抵消 | TwoSum/TwoProduct + 残差补偿 | 误差、指令数、UB |
| 快路条件数退化 | 迭代精化 + 主元/秩保护 | 残差、回退率、额外 FLOPs |

## 6. 消融设计

1. **recompute_w_u**：固定 T/H/Hg/D，扫 `headsPerTask=1/2/4`、`chunksPerTask=1/2/4`、`pipelineDepth=2/5`；记录 K 打包次数、workspace、event 时延、是否 `507057`。
2. **Concat**：固定 dtype/路数，扫 `bands=1/2/4/8`、`gBuf=1/2/3`、`S9_TRIM=0/1/2`、`S9_PRE=0/1`；记录读/写字节、tileG、event 时延；必须包含宽行和窄行两类。
3. **LinalgLstsq**：固定 M/N/K，扫 single/batch、方阵/矩形、16/50/64/200/256；记录每 key 的编译、运行、误差、workspace；先确认 `residuals` 布局。
4. **跨版本**：同一份源码在 CANN 8.5/9.1 各构建一次，记录编译错误、运行错误、数值差异；不要把 9.1 的失败直接归因于算法。
5. **离线测试**：Concat 的地址/布局测试可作为 Host/Kernel 共享布局的模板；但必须补 NPU 精度/性能。

## 7. 证据边界

- `recompute_w_u`：作者 22/22、第 1 名、逐 case µs 是 `O+A`；本机编译成功是 `V`，运行失败也是 `V`。
- `Concat`：作者第十名/348.33 µs 是 `O+A`；本机离线测试是 `V`（CPU）；NPU 是 `N`。
- `LinalgLstsq`：作者说明是 `S+L+A`，无性能数字；本机小尺寸 PASS 是 `V`，大/多 batch 失败也是 `V`。
- 三份实现都不能直接外推到 910C；CANN 9.1 的失败也不能证明作者实现错误。
