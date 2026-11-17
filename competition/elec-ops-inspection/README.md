# GitCode `cann/elec-ops-inspection`：批量最小二乘 LinalgLstsq 深读

## 1. 来源登记

| 项 | 值 |
|---|---|
| 仓库 | `https://gitcode.com/cann/elec-ops-inspection` |
| 维护者 | CANN 开源社区；提交归原参赛队 |
| 场景 | 昇腾 AI 行业创新大赛——2026 电力行业算子专场（预赛/决赛） |
| 访问日期 | 2026-11-16 |
| 默认分支 | `master` |
| 固定 commit | `dfe55921db351f71cfb8c96f795fd4472bd6bd47`（2026-11-16 19:09:03 +08:00） |
| 主要 DSL | Ascend C |
| 许可证 | 根 `LICENSE` 为 Apache-2.0；提交文件头也标注 Apache-2.0 |
| 本地缓存 | `repository-research-cache/gitcode-selected/elec`（选中文件）；完整树为 `repository-research-cache/GitCode_elec-ops-inspection` |
| 与既有归档关系 | 是既有 GitHub `Hansenwuwuwu/ascend-c-competition-archive` 的**上游官方仓**；按去重规则不重复计为独立算法来源；本轮新增 `lstsq` 实现 |
| 本报告范围 | 深读 `XDU_XDU-BTD/code/lstsq`（南工喵喵/不太对，西安电子科技大学） |

### 1.1 与 Hansen 归档的关系

Hansen 仓库归档了同一电力行业赛的 Fire/Spline/Tree 等提交，并已在本库第一批深读。`cann/elec-ops-inspection` 是赛事官方仓，包含更多队伍和算子（`lstsq`、`pinv`、`eigh`、`det`、`solve`、`swiglu` 等）。因此：

- 不把 `cann/elec-ops-inspection` 当作新的独立算法来源；
- 只把它作为 `lstsq` 的**原始官方来源**和 Hansen 归档的 upstream 校验；
- 旧报告中的 Fire/Spline/Tree 仍归 Hansen 归档来源，不重复计数。

## 2. 语义与约束

来源：`op_host/linalg_lstsq.cpp:24-91`、`README.md:1-13`、`op_kernel/tiling_key_linalg_lstsq.h`。

```text
A: [..., M, N] FP32
B: [..., M, K] FP32
X: [..., N, K] FP32
residuals: 方阵时为空；M>N 时 [..., K]
```

求解列满秩、`M >= N` 的最小二乘 `A X = B`，`driver="gels"`。约束：

- `1 <= M <= 1024`；
- `1 <= N <= min(M, 512)`；
- `1 <= K <= 64`；
- 前置 batch 维每维 1–32，`batches` 不超过 uint32；
- 只支持 FP32；
- 输出 `solution` 形状 `[..., N, K]`；`residuals` 在 `M==N` 时为空，`M>N` 时形状 `[..., K]`。

## 3. 固定源码位置

| 内容 | 路径/行 |
|---|---|
| Host 入口、shape 校验、blockDim、workspace | `op_host/linalg_lstsq.cpp:24-91` |
| InferShape / InferDataType / OpDef | 同上 `:96-143` |
| 补偿算术（TwoSum/TwoProduct/Divide） | `op_kernel/linalg_lstsq.cpp:25-90` |
| kernel 入口与 shape 分派 | 同上 `:122-335` |
| 分块补偿乘积（high/low 拆分） | `op_kernel/lstsq_compensated_tile_product.h:17-40` |
| 迭代精化（tiny/64/方阵） | `op_kernel/lstsq_shared_refinement.h:19-112` |
| 残差计算/补偿归约 | `op_kernel/lstsq_shared_residual.h:17-330` |
| QR 因子/多项式/Consumer | `op_kernel/qr/lstsq_qr_cube_compute.h:17-266` |
| QR 三角求解/RHS | `op_kernel/qr/lstsq_qr_cube_solve.h:17-280` |
| 256 方阵求解/列三角 | `op_kernel/square/lstsq_square256_solve.h:17-228` |
| 15 阶 tiny 快路/主元保护 | `op_kernel/tiny/lstsq_tiny15_fast_compute.h:17-303` |
| workspace/输出/残差调度 | `op_kernel/lstsq_shared_workspace.h:17-451` |
| 模板 key（BIG × SHAPE） | `op_kernel/tiling_key_linalg_lstsq.h:10-30` |
| 提交说明 | `README.md:1-13` |

关键 SHA256 见 [`index/catalog.json`](../../index/catalog.json)；本报告不手抄长哈希。

## 4. 机制

### 4.1 按 shape 选专用求解器，而不是一个 kernel 通吃

Host 用两组条件选 tiling key（`op_host/linalg_lstsq.cpp:57-78`）：

```text
large = (M==N && (128<=N<=200 || N==256 || ((N==50||N==64) && K==2)))
sharedQr    = M==N==256 && K==3 && batches<=4 && batches*4<=AIC
sharedQr200 = M==N==200 && K==4 && batches<=4 && batches*4<=AIC
implicitTiny16 = M==N==16 && K==1 && ...
shape = 10/11 (tiny16), 8 (sharedQr200), 7 (sharedQr),
        5 (tiny15), 6 (tiny16), 0 (generic), 1 (200/K4),
        2 (256/K3), 3 (50/K2), 9 (64/K2)
```

Kernel 侧按 `SHAPE` 分派（`op_kernel/linalg_lstsq.cpp:223-335`）：

- `SHAPE=7/8`：`RunMixedQr<256,3>` / `<200,4>`，Cube QR + 4 个 batch 共享因子；
- `SHAPE=1/2`：`KernelLargePipe200/256`，分块 LU/QR + 列三角；
- `SHAPE=3/9`：50/64 阶 packed 快路；
- `SHAPE=5/6`：15/16 阶 tiny 快路；
- `SHAPE=10/11`：16 阶 implicit tiny；
- `SHAPE=0/4`：通用方阵/矩形回退。

这减少的是“通用 solver 的固定开销/无效分支”，代价是代码体量和每 key 的独立回归。

### 4.2 补偿算术：TwoSum/TwoProduct + 高低位拆分

`op_kernel/linalg_lstsq.cpp:53-80`：

```text
Sum(a,b):    s = fl(a+b); v = fl(s-a);
             lo = fl(fl(a-fl(s-v)) + fl(b-v))
Product(a,b): h = fl(a*b); ah=High(a); bh=High(b);
             al=fl(a-ah); bl=fl(b-bh);
             e = fl(fl(fl(ah*bh-h)+fl(ah*bl))+fl(al*bh));
             lo = fl(e+fl(al*bl))
Divide(a,b):  seed=a/b; 2 次 Newton/补偿修正; b==0 -> QuietNaN
```

`CompensatedTileProduct`（`lstsq_compensated_tile_product.h:19-40`）把每个 tile 的乘积拆成 `high/low`，用 7 条 `Mul/Sub/Add` 得到补偿后的低部分，再参与归约。

`lstsq_shared_residual.h` 进一步做残差补偿归约（`MergeCounterResidualHalves<1024/512/...>`，`:92-97`），并用 `WriteCompensatedResidual` 写回（`:115`）。

这改善的是 FP32 消元/回代中的抵消误差，代价是每条乘加变成多条向量指令、寄存器/UB 压力上升。

### 4.3 迭代精化：用原方程残差修正解

`lstsq_shared_refinement.h`：

- `RefineTinyRows`（`:68-90`）：对 tiny 路径做 2 步残差修正，每步重新算 `r = b - A x`，解 `A d = r`，再 `x += d`；
- `RefineSquareRows`（`:93-112`）：对方阵路径按列算残差，`SolveInPlaceWithFactors` 回代，再累加；
- `RefineSmallSingle64`（`:28-33`）：64 阶单矩阵路径。

这用额外 FLOPs 换精度，适合“快路可能因条件数/主元退化”的场景；但残差本身也要补偿，否则会引入新的舍入。

### 4.4 主元/有限性/秩保护

- tiny 15 路径用 `pivotLower/pivotUpperBits` 检查主元范围、`rankSafe` 检查 15 个主元是否都有效（`lstsq_tiny15_fast_compute.h:94-159`）；不满足则回退 legacy；
- `scalar_guard::QuietNaN` 处理除零（`linalg_lstsq.cpp:68-69`）；
- Host 检查 `M/N/K/batches` 上限和 dtype（`:24-56`）；
- `sharedQr` 只在 `batches<=4` 且 `batches*4<=AIC` 时启用，避免多 batch 抢核。

### 4.5 workspace 与输出布局

Host 估算：

```text
elements = 2*M*N + M*K + 3*N*K + 2*N
perBatch = ceil(elements/16)*16
workspace = libApi + batches * (shared? K : 1) * perBatch * 4 + 64
```

（`op_host/linalg_lstsq.cpp:86-91`）

`lstsq_shared_workspace.h` 负责：

- `BuildAugmented64`（`:19-67`）构造增广矩阵；
- `EmitSquareDma` / `LoadRows200` / `Rows64ToColumns`（`:68-115`）搬运与重排；
- `ProcessFixedSquare`（`:149-173`）按 task 调 `SolveCooperativeTask`/`Refine`；
- `CopyAtomicRhs50/64`（`:191-257`）处理重复/原子 RHS 输出；
- `CheckOutputSchedule`（`:427-451`）判断输出是否与 cache line 共享，必要时把 blockDim 降到 1。

这部分是“按最终消费者布局生产”，减少中间转置/重复清零。

## 5. 作者证据与局限

`XDU_XDU-BTD/code/lstsq/README.md` 只给实现思路（5 条），**没有逐 case 时延、baseline、warmup/重复次数或榜单分数**。`preliminary/problems/README.md` 只有“预赛 problem 算子目录”，没有题面细节。因此：

- 本实现只能记为 `S+L+A`（源码、固定 commit、作者说明），没有 `O` 原始性能记录；
- 不能给出“比 PyTorch/numpy 快多少”；
- 作者的精度/性能结论需要在目标环境重新跑。

## 6. 本机构建与 NPU 验证

### 6.1 构建

源码副本：`local-validation-20261116/gitcode_lstsq`，CANN 9.1.0 + gcc 11.4.0 + cmake 3.22.1。

第一次构建失败：

```text
op_kernel/qr/lstsq_qr_vector_common.h:67:39: error:
use 'template' keyword to treat 'ReinterpretCast' as a dependent template name
op_kernel/qr/lstsq_qr_vector_common.h:72:41: same
```

这是 CANN 9.1 的 clang 对依赖模板名更严格；第三方源码在 CANN 8.5 下可编译。为了让本机验证继续，只在**本地副本**改了两处：

```cpp
kernel.all_.template Zq6<int32_t>()
kernel.trans_[...].template Zq6<int32_t>()
```

未修改第三方仓库原文，也未把 patch 写进本库源码。改后构建成功，生成：

```text
build-local/libcust_opapi.so
build-local/op_kernel/ascendc_kernels/binary/ascend910b/linalg_lstsq/LinalgLstsq_c04cee5cafcd9476761557bb8347387f.o
build-local/op_host/libcustom_ascendc_cust_optiling.so
```

日志：[`gitcode_lstsq_build2.log`](evidence/gitcode_lstsq_build2.log)。

### 6.2 NPU 运行

运行方式：自写 ACLNN 驱动（不属于第三方仓库），输入 `A[i,j] = 2 (i==j) + 0.05/(1+|i-j|)`，`B=A·1`，期望 `X=1`；每个 shape 跑 5 次，取最好的一次；所有运行经 `bin/npu-lock`。

| shape (M×N, K, batch) | 结果 | 最好时延 | max abs |
|---|---|---:|---:|
| 16×16, K=1, b=1 | **PASS** | 0.0265 ms | 1.192e-07 |
| 50×50, K=2, b=1 | **PASS** | 0.0301 ms | 3.576e-07 |
| 64×64, K=2, b=1 | **PASS** | 0.0395 ms | 3.576e-07 |
| 16×16, K=1, b=2 | FAIL `507035` | — | — |
| 50×50, K=2, b=4 | FAIL `507035` | — | — |
| 64×32, K=2, b=1 | FAIL `507035` | — | — |
| 64×50, K=2, b=1 | FAIL `507035` | — | — |
| 100×64, K=1, b=1 | FAIL `507035` | — | — |
| 200×200, K=4, b=1 | FAIL `507057` | — | — |
| 256×256, K=3, b=1 | FAIL `507057` | — | — |

错误摘要：

```text
64x32: vector core exception; scalar instruction accesses an invalid GM address
200/256: aicore exception; MTE accesses an invalid GM address or cross-device timeout
```

判断：

- **小尺寸单 batch 方阵**（16/50/64）在 CANN 9.1 上可运行且通过数值检查；
- **多 batch、矩形、200/256 shared-QR** 路径在 CANN 9.1 上崩溃；可能是 Host shape/残差布局、workspace 大小或 CANN 版本 ABI 差异，尚未定位到单条语句；
- 不能把作者在 CANN 8.5 上的（未公开）成绩当作本机结果；
- 也不能因为小尺寸 PASS 就认为所有 shape 都可用。

日志：

- [`gitcode_lstsq_M16_N16_K1_B1.log`](evidence/gitcode_lstsq_M16_N16_K1_B1.log)
- [`gitcode_lstsq_M50_N50_K2_B1_r.log`](evidence/gitcode_lstsq_M50_N50_K2_B1_r.log)
- [`gitcode_lstsq_M64_N64_K2_B1_f.log`](evidence/gitcode_lstsq_M64_N64_K2_B1_f.log)
- [`gitcode_lstsq_M64_N32_K2_B1_r.log`](evidence/gitcode_lstsq_M64_N32_K2_B1_r.log)
- [`gitcode_lstsq_M256_N256_K3_B1_f.log`](evidence/gitcode_lstsq_M256_N256_K3_B1_f.log)

## 7. 适用条件、负例与复用判断

适用：

- FP32、列满秩、`M>=N`、`K<=64` 的批量最小二乘；
- 小尺寸（15/16/50/64）单 batch 方阵：本机已验证可用；
- 能接受 shape 专用路径和每 key 独立回归；
- 需要 FP32 补偿算术/迭代精化提高精度时。

负例/风险：

- **没有作者性能数据**：无法证明“高性能”；
- **CANN 9.1 只验证了小尺寸单 batch**：多 batch/矩形/200/256 在本机崩溃；
- **两处 `template` 兼容性 patch**：CANN 9.1 不能直接编译原文件；
- **shape 专用代码多**：每个 key 的 workspace/主元/残差布局都不同，换 shape 必须重跑；
- **`residuals` 可选但矩形路径可能依赖**：本机矩形仍失败，不能确认我们的残差布局与算子期望完全一致；
- **不支持非 FP32、非 `gels`、`M<N`、`M>1024`、`N>512`、`K>64`**。

复用判断：

- **优先复用**：补偿 TwoSum/TwoProduct、残差补偿归约、按 shape 选 tiny/packed/QR/LU 的分层思路、主元/秩保护；
- **需要验证**：每个 shape key 在本机 CANN/驱动上的可用性；矩形路径的 `residuals` 布局；多 batch 的 workspace/blockDim；
- **暂不使用**：在未复现作者环境前，不把该实现当作通用 Lstsq 模板；不把 16×16 的 0.0265 ms 外推到 200/256 或 910C。

## 8. 去重与计数

- `cann/elec-ops-inspection` 是 Hansen 归档的上游官方仓，**不增加独立算法来源数**；本轮新增 `lstsq` 一份实现。
- 与既有 15 份实现不重复：既有第一批的 Hansen 深读是 Fire/Spline/Tree，没有 Lstsq。
- 与本报告同仓的 `pinv/eigh/det/solve/swiglu` 等未深读，留作候选。
