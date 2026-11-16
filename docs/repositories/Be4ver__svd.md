# Be4ver/svd：批量单边 Jacobi SVD 与 QDWH+SDC 融合

## 1. 来源登记

| 项 | 值 |
|---|---|
| 仓库 | `https://github.com/Be4ver/svd` |
| 作者 | ZhuShengRui |
| 场景 | 生产/实验型 Ascend SVD 与极分解算子（非比赛提交） |
| 访问日期 | 2026-11-16 |
| 默认分支 | `main` |
| 固定 commit | `b02796511ea62861c647a56b880da15b6a07b3a0` |
| 主要 DSL | C++ / Ascend C（AscendC CMake，目标 `ascend910b3`） |
| 算子族 | 批量 Jacobi SVD、QDWH 极分解、对称谱分治（SDC）、叶块 Jacobi |
| 许可证 | 未声明 LICENSE/COPYING |
| 本地缓存 | `repository-research-cache/Be4ver_svd` |
| 与既有归档关系 | 新来源；与 GroupedMatmulGelu 无重复 |
| 本报告范围 | 深读并本机复测 `paco-jacobi-svd` 与 `qdwh_sdc_fused_jacobi` 两份实现 |

### 1.1 环境

| 项 | 作者环境 | 本机复测 |
|---|---|---|
| 设备 | Ascend 910B3（`paco`/QDWH 报告均写 910B3） | Ascend 910B3，NPU 2 |
| CANN | 仓库未统一固定；QDWH 的 `PROMPT_gemm_merge.md:39-40` 记录过 CANN 8.2.RC1 | 9.1.0 |
| 编译器 | Ascend C / `ccec`，CMake 目标 `ascend910b3` | gcc 11.4.0 + CANN 9.1.0 `ccec` |
| 计时 | `paco`：主机高精度时钟包住 launch + stream 同步；QDWH：`aclrtEventElapsedTime` | 同作者口径 |
| 内存 | 910B3，单 die；UB 预算 192 KiB | 同 |

作者没有在仓库中给出统一的 CANN 版本文件，`paco` 的 readme 只要求 source `ascend-toolkit/set_env.sh`。因此本机 CANN 9.1.0 的复测是“同 SoC、不同工具链版本”的验证，不是严格同环境复现。

### 1.2 本机构建与运行

两个算子都用仓库自带 CMake 在独立副本中重新构建，未修改第三方源码：

```text
cmake 3.22.1
gcc 11.4.0
CANN 9.1.0
SOC ascend910b3
```

构建与运行命令、原始日志见 [`docs/research/LOCAL_910B_VALIDATION_20261116.md`](../research/LOCAL_910B_VALIDATION_20261116.md) 第 2.6–2.7 节。核心结果：

| 实现 | 形状 | 本机 | 作者 | 正确性 |
|---|---|---:|---:|---|
| PACO 批量 Jacobi | 512×(32×32)，FP32 | **6.174 ms** | 7.061 ms | 512/512 PASS；max σ abs `3.242e-05` |
| QDWH+SDC 融合 | 128×128，FP32，cond=1e2 | **4.288 ms** | 4.392 ms | QDWH/SDC/sigma PASS |
| QDWH+SDC 融合 | 512×512，同上 | **20.698 ms** | 20.540 ms | PASS |
| QDWH+SDC 融合 | 1024×1024，同上 | **48.475 ms** | 46.839 ms | PASS |

## 2. 实现一：PACO 批量单边 Jacobi SVD

### 2.1 语义与适用范围

`paco-jacobi-svd/readme.md` 描述的是“小 shape + 大 batch”的批量单边 Jacobi：

- 输入 `A_raw(rows×cols)`，FP32，批量 `batch`；
- device 侧 pad 到 `rowsPad×colsPad`：行按 64 对齐（AIC 分块），列按 32 对齐（Cube fractal 16，AIV 只需 8）；
- 每个 AIV 分到 `perCore=ceil(batch/coreNum)` 个矩阵；
- 对每个矩阵做单边 Jacobi 行旋转，使各行两两正交；行范数即奇异值；
- AIC 在所有 AIV 完成后分摊矩阵，用 `MatmulCustom(isTransposeB)` 算 `G=B·Bᵀ`；主机取对角线开根得到 `sigma`。

约束（`jacobi_svd_tiling.cpp:11-25,27-41`、`jacobi_svd_custom.cpp:363-371`）：

- `rows <= 128`、`cols <= 256`、`batch > 0`；
- 列 pad 到 32；行 pad 到 64；
- `maxSweeps` 默认 30；
- 当前设备入口只支持 `KERNEL_TYPE_MIX_AIC_1_2`。

### 2.2 固定源码位置

| 内容 | 路径/行 |
|---|---|
| 单对行旋转、点积/范数、就地更新 | `paco-jacobi-svd/jacobi_svd_custom.cpp:130-170` |
| K 个矩阵按行交错载入/写回 | 同上 `:215-249` |
| K 个矩阵合并 `Mul` + `WholeReduceSum` | 同上 `:251-303` |
| 主机分路：小 batch 单矩阵 / 大 batch K 合并 | 同上 `:435-482` |
| AIC 计算 `G=B·Bᵀ`、写 sigma | 同上 `:501-538` |
| pad/列对齐/tiling | `paco-jacobi-svd/jacobi_svd_tiling.cpp:11-25,56-71` |
| 作者说明与性能表 | `paco-jacobi-svd/readme.md` |
| 64×64 tournament 研究 | `paco-jacobi-svd/docs/64x64-optimization-study-2026-11-12.md:1-18,62-100` |

关键文件 SHA256：

```text
jacobi_svd_custom.cpp  c9ca7615ced974caec1085e5da3ddd060b4759f1c87c03c8836256fc92bf2a1c
kernel_v8.cpp          1496d8411ca5939efd19a2c95272a2602d321b0a38547b6a7dbaae2daeef81e8
64x64 study            d963b05aeaddd8c7551faf1b43fcf964c2f4b55fa67cedd36a11c5b247e6bd25
```

### 2.3 机制：跨矩阵交错 + 合并归约

`LoadMulti` 把 K 个矩阵按行交错存放（`jacobi_svd_custom.cpp:215-232`）：

```text
tile[(i*K + k)*colsW] = A_k 的第 i 行
```

这样同一个 `(i,j)` 行对上，K 个矩阵的数据在 UB 中连续：

- 一次 `Mul(prod3_, u, u, n)` 覆盖 K 个矩阵的 `u²`；
- 一次 `WholeReduceSum(red_, prod3_, colsW, K, ...)` 得到 K 个 `|u|²`；
- 同理得到 K 个 `|v|²` 和 K 个 `<u,v>`；
- 角度仍逐矩阵用标量算（代码注释说明矢量角度链只在 `count=64` 时正确，见 `readme.md` 的“已验证的坑”）；
- 旋转用 `Muls/Axpy/Adds`，每对只保留一次 `Adds` 拷回。

K 的选择（`jacobi_svd_custom.cpp:440-445`）：

```text
kUbBudget = 120 KiB
perMatBytes = (rows + 4) * colsWork * 4
K = min(kUbBudget / perMatBytes, 64)
```

这减少的是**短向量指令数和归约启动次数**，不是 Jacobi 的算术复杂度。`readme.md` 明确记录：K 再大（140 KiB）收益为负；列 pad 到 64 会让 32 列矩阵带宽翻倍。

### 2.4 机制：64×64 tournament 原型（未合入）

`docs/64x64-optimization-study-2026-11-12.md:62-100` 给出另一条更强的路径，但它只是研究补丁 `docs/research-64x64/tournament.patch`，**不在当前主路径**：

- 64 行共有 2016 个无序对；字典序每 sweep 串行做 2016 对；
- tournament 把行分成 A/B 两组，每轮 32 对互不共享行，63 轮覆盖全部 2016 对；
- 一轮内 3 条 `WholeReduceSum(repeat=32)` 得到 32 组 a/b/c；角度链固定 64 lane（前 32 有效）；
- 每轮 4 条行更新指令（`Mul`/`MulAddDst`），整个 sweep 252 条，而不是 8064 条；
- 用第二个 UB 矩阵 ping-pong，5 段 `Adds(...,0)` 完成行重排。

作者实测（同一设备、相邻运行、event 中位数）：

| 输入 | batch | 原实现 | tournament | 加速 |
|---|---:|---:|---:|---:|
| 标准正态 | 1 | 3.62260 ms | 0.85056 ms | 4.26× |
| 标准正态 | 40 | 3.94784 ms | 0.94690 ms | 4.17× |
| cond≈1e3 | 40 | 4.86842 ms | 1.41022 ms | 3.45× |
| cond≈1e6 | 40 | 6.51702 ms | 1.87254 ms | 3.48× |

同一份研究还给出反例：

- 只把整张矩阵常驻 UB、保持字典序和标量角度：batch=1 3.62284→3.50662 ms，batch=40 3.94800→3.81564 ms，仅 3.2–3.4%；说明主要收益来自“轮内 32 对融合”，不是“常驻”。
- 只把角度链向量化、保留 K≤8 的单对组织：batch=1 14.2394 ms（比 3.62 ms 慢），batch=256 18.3270 ms（比约 16 ms 慢）；固定成本需要更多独立配对来摊薄。
- 先集中算标量系数、再循环更新：batch=256 15.9976→20.7030 ms，更慢。
- 硬截断 sweep：10 sweeps 在 cond=1e6 上小奇异值相对误差明显变差；作者保留 30 sweeps 默认。

### 2.5 证据判断

- **可复现**：当前通用路径在 910B3 + CANN 9.1.0 上编译运行，32×32×512 为 6.174 ms，512/512 通过；作者 7.061 ms，同量级。
- **不可直接引用**：4.26× 是未合入的研究补丁；不能写成“Be4ver 当前代码 4.26×”。
- **没有 baseline**：作者没有给 PyTorch/厂商库同环境 baseline，6.174 ms 只能与作者 7.061 ms 比较，不能换算成相对 baseline 的加速比。
- **精度边界**：作者验收用 σ `atol=rtol=1e-4`、正交/重构 `1e-3`；cond=1e6 时小奇异值相对误差可达 1.5%，绝对容差会掩盖它。

## 3. 实现二：QDWH + SDC 融合

### 3.1 语义与适用范围

`qdwh_sdc_fused_jacobi/README.md` 描述的是：

```text
A = U H,  H = (A^T A)^{1/2}（M>=N 时）
```

- QDWH 迭代求极分解 `U,H`；
- SDC 在 device 内递归分裂 `H`，直到叶块 `n <= threshold`（默认 64）；
- 每个叶块用 64×64 单边 Jacobi 求奇异值；
- 整个 QDWH + SDC + 叶块 Jacobi 在一次 kernel launch 内完成；
- 计算期间 host 与 device 没有中间传输；host 只在开始上传 A、结束下载 leaves/sigma。

约束（README、`fused_main.cpp:108-121`、`fused_batch_cfg.hpp:17-83`）：

- `16 <= blockDim <= 64` 且为偶数；
- `16 <= threshold <= 64`；
- `1 <= maxIter <= 64`；
- `mPad + nPad <= kQrMaxQd = 6144`（QR panel 点积区容量）；
- `n >= 2816` 时 `maxIter=4` 不够，需要至少 6；
- AIV UB 预算 192 KiB，`kSdcBatchMax` 由 `static_assert` 约束为 15。

### 3.2 固定源码位置

| 内容 | 路径/行 |
|---|---|
| 融合 kernel 入口（只做 Init + RunAiv/RunCube） | `qdwh_sdc_fused_jacobi/qdwh_sdc_fused_custom.cpp:13-29` |
| Phase 1 QDWH（AIV） | `.../fused_qdwh_sdc.hpp:89-146` |
| Phase 1 QDWH（Cube） | 同上 `:153-188` |
| Phase 2 SDC 分组 AIV（λ/A0 + batch QDWH + tail） | 同上 `:386-477` |
| Phase 2 SDC 分组 Cube | 同上 `:481-580` |
| 阶段 barrier / RunAiv / RunCube | 同上 `:733-825` |
| `H± = Qᵀ H Q` 一次 GEMM、跳过显式对称化 | `.../sdc_node.hpp:331-397` |
| `λ=trace(H)/n` 跨 AIV 分布式归约 | `.../vec.hpp:397-453` |
| QDWH `Xt += Q2·(κQ1)ᵀ` 的 atomic 累加 | `.../qr_custom.cpp:215-300` |
| UB 预算与 `kSdcBatchMax=15` | `.../fused_batch_cfg.hpp:17-83` |
| 64×64 叶块 Jacobi | `.../jacobi_leaf.hpp:36-134` |
| host 单次 malloc/launch/event 计时 | `.../fused_main.cpp:280-354,435-436` |

关键文件 SHA256：

```text
fused_qdwh_sdc.hpp  8326fbad62acaf8c2bca93f11ef3c8af96a24710f889acc839233ad258edba40
sdc_node.hpp        6609010fc88662090866024144af737a6e91733f558aa6483bddd8ba603d39e7
vec.hpp             984a7c51f42e949cd01f185d18eec8aeecb9dd5bb56ca08d3d8f2b63faf024c5
qr_custom.cpp       62b2b5b689511ca42665742f797ed056208c328377ecfedb65106cb17a201a13
fused_batch_cfg.hpp 05733d49e81edaecd4d99948307fbb0632d01f798289f4a599bfa428131a750e
README.md           59b6a473f776b27576e8c02ee2adc4b4034a6bcf7edf344a9b92066e488be3ff
```

### 3.3 机制一：一次 launch 完成 QDWH + SDC + 叶块 Jacobi

旧流程是两次 host main / 两个 kernel：

```text
qdwh_main: A -> qdwh_custom -> U,H -> h.bin/u.bin
sdc_dev_main: H -> sdc_dev_custom -> leaves
```

融合后（`README.md`、`fused_main.cpp:290-324`）：

```text
fused_main: 上传 A 一次 -> launch qdwh_sdc_fused_custom 一次 -> 下载 leaves
```

关键点：

- `H` 直接由 QDWH 最后一步写进 SDC arena 的 `slot0`（`fused_qdwh_sdc.hpp:103-104,144`），不落 `h.bin`；
- SDC 节点队列、`H±` 搬运、叶块收集都在 device 内完成（`fused_qdwh_sdc.hpp:196-477`）；
- QDWH Phase 1 与 SDC Phase 2 用 `PhaseBarrier` 在 kernel 内同步（`:733-743`）；
- 计算期间 host 不参与数据传输；结束后的 D2H 只为验证/输出。

作者证据：

| 形状 | 融合 kernel | 独立 QDWH+SDC + 独立 Jacobi | 节省 |
|---|---:|---:|---:|
| 64×64/batch1 | 4.102 ms | 6.278 ms | 2.176 ms（34.7%） |
| 128×128 | 6.428 ms（回归记录） | — | — |
| 1024×1024 | 46.839 ms | — | — |

64×64 的 34.7% 来自 `docs/performance_64x64.md`；1024×1024 的 46.839 ms 来自 `benchmark_square_sizes_20261115/1024/run.log`，与更早的 `benchmark_1024x1024_run1/kernel.log` 的 51.599 ms 不是同一轮优化（后者 batch≤8，前者 batch≤15）。

本机复测 128/512/1024 分别为 4.288/20.698/48.475 ms，均通过 QDWH/SDC/sigma 校验；与作者数字接近，说明融合路径在 CANN 9.1.0 上可执行。

### 3.4 机制二：同层 SDC 节点 batch QDWH

QDWH Phase 2 原来一个节点调一次 split，每轮都要付“逐行 DataCopyPad + 跨核 flag + QR panel 循环”的固定成本。融合版把 device 节点队列头最多 `kSdcBatchMax` 个节点打包成一组，做一次 batch QDWH（`fused_qdwh_sdc.hpp:386-423,481-500`；`sdc_node.hpp:71-167`）。

`kSdcBatchMax` 不是拍脑袋：

- `kQrColumns=2`、`kQrMaxQd=6144`、`kQrChunkSize=8192`；
- 固定 UB 占用 `aQueueIn/bQueueIn/aQueueOut/tauQueueOut`；
- 点积区 `columns * kQrMaxQd`；
- `tQueueDst` 每 batch 256 float；
- `householder_v` 每矩阵 `8*columns + 512` float；
- `QdwhPrepare` 的定长数组 B+1<=16；
- 三个上界取最小，并用 `static_assert` 兜底，得到 15（`fused_batch_cfg.hpp:33-83`）。

作者实测（README）：

| 输入 | batch=8 | batch=15 | 分组数 |
|---|---:|---:|---:|
| 1024×1024 | 50.15 ms | 49.96 ms | 6→6 |
| 1999×1999 | 135.92 ms | 132.60 ms | 9→8 |
| 2048×2048 | 136.09 ms | 134.97 ms | 9→8 |
| 3072×3072 | 401.7 ms | 403.5 ms | 11→8 |

解释：

- 1024 从 51.599 ms（batch≤8 的早期 run）到 46.839 ms（batch≤15）主要是固定开销摊薄；
- 3072² 的分组数从 11 降到 8，时间不变，说明 kernel 已经计算受限（QR 串行链 / GEMM），继续合并分组无效；
- batch 已到 UB 天花板（195,840/196,608 B，仅剩 768 B），不能再加。

### 3.5 机制三：跳过显式对称化

数学上 `H+ = Q+ᵀ H Q+`、`H- = Q-ᵀ H Q-` 本来就对称；GEMM 只引入约 1e-7 的舍入不对称。原实现每个节点做一次 `H ← 0.5(H+Hᵀ)` 的标量两遍读写，是 Phase 2 的 AIC scalar 大头。

`sdc_node.hpp:379-397` 用 `static constexpr bool kSymmetrize = false` 跳过它。作者记录（README）：

| | kernel | eig_err | sym_max | 结果 |
|---|---:|---:|---:|---|
| 打开 | 160.8 ms | 2.2e-07 | 0.0 | PASS |
| 关闭 | **134.3 ms** | **2.0e-07** | 0.0 | PASS |

这是“去掉数学上冗余的同步/标量工作”的典型正例，但条件是：

- `H` 的对称性由算法保证；
- GEMM 的不对称在容差内；
- 仍要检查 `sym_max`、`eig_err`，不能默认所有 shape 都安全。

### 3.6 机制四：λ 的分布式归约

原实现 40 个 AIV 各自把 `H` 的整条对角线读一遍（n 次 scalar GM 读 × 40）。`vec.hpp:397-453` 改成：

- 行按 `blockNum` 平均切段，每个 AIV 只读自己那一段；
- 每个核把部分和写到 GM 连续数组的第 `blockIdx` 个 float；
- 一次 `AivBarrier`；
- 所有核把数组整段读回 UB，用 `WholeReduceSum` 归约。

作者实测 2048²：`aiv_scalar` 21007→19917 µs（−5.2%），kernel 132.76→132.29 ms；求和顺序改变会让 λ 末位不同，从而改变分裂树（2048² leaves 47→46），误差仍在 1.5e-06 量级。

这适合“每个核都重复读同一段标量”的场景；代价是额外 GM 部分和、barrier 和一次全核读回。若标量读已经命中 cache，收益会很小（README 也记录了另外两类“零收益”的 scalar/同步优化）。

### 3.7 机制五：一次 GEMM 得到 `QᵀH`

`sdc_node.hpp:331-361` 把原来 4 组 GEMM（`QT=Q+ᵀ`、`Tmp=QT·H`、`QT=Q-ᵀ`、`Tmp=QT·H`）压成一次：

```text
TmpAll = Qᵀ · H
TmpAll[0:k, :]   = Q+ᵀ H
TmpAll[k:n, :]   = Q-ᵀ H
```

做法是让 A 直接用 Q 的紧凑行主序，`isTransposeA` 让 Cube 按转置形态取块，在 L1→L0 做 `enTranspose`；K/M/N 都跑满 `nPad`（padding 行/列由 AIV 写成 0），末尾残块不再是问题。

收益（README）：少 2 次 GEMM 启动 + 2 次 `AicBarrier`，不再写 `QT`，省 `(kPad+m2Pad)*n` 个 float 的 Fixpipe 写回。代价是：

- 依赖 Q 的 padding 恒为 0；
- 依赖 `isTransposeA` 在整块 128 下可用，末尾残块会卡死；
- 多读的 H 行必须乘 0，arena 必须提前清 0（`fused_qdwh_sdc.hpp:108-117`）。

## 4. 证据与判断

### 4.1 作者证据

- PACO：`benchmark_32x32_batch512_run1/run.log` 7.061 ms；`verify.log` 512/512 PASS，max σ abs `3.242e-05`，最差 B 正交 `9.345e-07`，最差 Vt 正交 `1.025e-06`。
- QDWH：`benchmark_square_sizes_20261115` 记录 128/512/1024/2048 = 4.392/20.540/46.839/120.302 ms，全部 PASS；`benchmark_1024x1024_run1` 记录 51.599 ms（早期 batch≤8）和 `eig_err=2.367e-07`、`trace_err=5.047e-09`、`sigma max abs=1.3838629e-06`。
- 64×64 融合：4.102 ms vs 6.278 ms（−34.7%），sigma max abs 1.481e-6。
- 64×64 tournament：4.26×（batch1）、4.17×（batch40），但只是研究补丁。

### 4.2 本机证据

见 [`LOCAL_910B_VALIDATION_20261116.md`](../research/LOCAL_910B_VALIDATION_20261116.md)：

- PACO 32×32×512：6.174 ms，512/512 PASS；
- QDWH 128/512/1024：4.288/20.698/48.475 ms，QDWH/SDC/sigma 全部 PASS；
- QDWH 1000×800、seed=42 连跑 3 次：36.812/38.104/36.953 ms，`sigma.bin` 三次 SHA256 相同，QDWH/SDC/sigma 全 PASS（作者的 atomic 非确定性在本机未触发）；
- 两个算子都在 CANN 9.1.0 上重新构建成功，未改源码。

### 4.3 证据级别

| 项 | 级别 |
|---|---|
| 源码/数据流/布局 | `S+L` |
| 作者 benchmark/精度日志 | `O+A` |
| 本机编译/运行（PACO、QDWH） | `V` |
| 64×64 tournament 4.26× | `O+A`，研究补丁，未合入 |
| 1000×800 atomic 非确定性 | `A`，本机未复现该输入 |
| 910C / 其它 CANN | `N` |

## 5. 适用条件

1. **FP32、小/中矩阵、批量或单矩阵**：PACO 适合 `rows<=128, cols<=256` 的批量；QDWH 适合方阵/长矩阵，但受 `mPad+nPad<=6144` 限制。
2. **UB 预算明确**：PACO 的 K 由 120 KiB 预算推导；QDWH 的 batch 由 192 KiB UB 推导，余量只有 768 B。换 SoC/CANN 必须重新算。
3. **能接受多 sweep 迭代**：Jacobi 的收敛依赖输入条件数；cond=1e6 需要更多 sweep，硬截断会伤小奇异值。
4. **能接受非确定性**：QDWH 的 atomic 累加会让 `H` 末位随运行变化；SDC 在特征值接近时可能放大。对 golden 逐位敏感的任务要先改成确定性归约。
5. **一次性融合的复杂度可控**：QDWH+SDC 的 flag 协议、arena 布局、padding 清零、`isTransposeA` 残块限制都很强；没有完整回归不要轻易移植。
6. **许可证**：仓库未声明 LICENSE；本库只引用和分析，不复制源码。

## 6. 负例、退化与未知风险

1. **4.26× 不是当前代码成绩**：tournament 只在 `docs/research-64x64/tournament.patch`，当前主路径没有它；本机验证的是通用路径。
2. **atomic 非确定性 + 特征值间隙放大**：`QdwhPmat` 用 atomic 累加，`H` 每次有 ~1e-6 的最低位数差异；SDC 的 `H±` 在 λ 贴近特征值时把噪声放大 `~||H||/gap ≈ 2000` 倍。作者记录 1000×800、seed=42 时 σ 误差 1.5e-5~3.1e-4 且逐次不同；同 shape 换 seed 1/2/3 又稳定在 ~2e-6。这是算法条件数问题，不是回归。本机用同一输入连跑 3 次，`sigma.bin` SHA256 三次相同、QDWH/SDC/sigma 全 PASS，说明该缺陷在当前 CANN 9.1.0/调度下没有触发；不能据此否定作者记录，也不能说已经修复。
3. **小奇异值相对误差**：cond=1e6、batch=40 时 tournament 最小奇异值最大相对误差约 1.50%，原实现约 0.912%；现有 `atol=1e-4` 会允许这种误差。
4. **硬截断 sweep 不稳**：10 sweeps 在 cond=1e6 上误差明显变差；作者保留 30。
5. **向量角度链只在 count=64 时正确**：K=1/4/10/13/26 下 `CompareScalar/Select/Sqrt/Div` 结果不可信；所以 K 合并路径仍用标量角度循环。
6. **UB 超限直接失败**：`>192KB` 会 `aclError 507015`、输出全 0；QDWH 的 batch 已经到 195,840/196,608 B。
7. **列 pad 过宽会变慢**：PACO 列 pad 到 64（而不是 32）会让 32 列矩阵带宽翻倍。
8. **没有 LICENSE/README（根目录）**：只有一个 commit；来源、版权和再分发边界不明确。

## 7. 复用建议

### 7.1 优先复用

- **批量小矩阵的跨矩阵交错布局**：把 K 个矩阵的同一条行放在连续 UB，合并 `Mul/Reduce`；适合批量 SVD、批量 QR、批量小 GEMM 的预处理。
- **单 kernel 多阶段融合**：QDWH + SDC + 叶块 Jacobi 一次 launch，H 不落盘；适合“中间矩阵只被下一个阶段消费”的分解流水线。
- **UB 预算驱动的 batch 上限**：用 `static_assert` 把 `kSdcBatchMax` 绑到实际缓冲布局，避免“host 放行、device 越界”。
- **去掉数学冗余的对称化**：当 `H±` 由正交相似变换保证对称时，跳过 `0.5(H+Hᵀ)`；必须用 `sym_max/eig_err` 回归。
- **分布式标量归约**：每个核只读自己那一段对角线，部分和落 GM，barrier 后 `WholeReduceSum`；适合大 n 的 trace/λ。

### 7.2 需要验证后再用

- **64×64 tournament**：先把补丁合入独立分支，按 `cond=1e2/1e4/1e6`、batch=1/8/40 做完整回归；确认 63 轮配对的行重排在目标 CANN 上仍正确。
- **QDWH batch=15**：换 910C/其它 UB 容量后重新推导；3072² 已经显示分组数下降但时间不变，不能假设 batch 越大越快。
- **atomic 消除**：把 `QdwhPmat` 改成确定性树归约/补偿求和，或在 SDC 分裂前做一次高精度对称化；先用 1000×800、seed=42 做逐次复现测试。
- **CANN 9.1.0 的数值差异**：本机 1024 比作者慢 3.5%，Sinkhorn 也出现更大误差；换版本要重新跑精度。

### 7.3 暂不使用

- 把 4.26× 写成 Be4ver 当前代码成绩；
- 把 51.599 ms 写成当前 1024×1024 成绩（当前是 46.839 ms）；
- 在 cond=1e6、小奇异值敏感的任务里直接用 `atol=1e-4` 验收；
- 在没有 LICENSE 的情况下复制/再分发源码；
- 把 910B3 的 UB 预算、batch=15、K 值直接搬到 910C 或 950。
