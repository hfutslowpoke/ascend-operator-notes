# GitCode `cann/cann-ops-competitions`：GDN recompute_w_u 与 S9 Concat 深读

## 1. 来源登记

| 项 | 值 |
|---|---|
| 仓库 | `https://gitcode.com/cann/cann-ops-competitions` |
| 维护者 | CANN 开源社区；各赛事提交归原参赛队 |
| 场景 | 2026 CANN 算子天梯赛、S8/S9 算子挑战赛、高校/行业赛题与作品归档 |
| 访问日期 | 2026-11-16 |
| 默认分支 | `master` |
| 固定 commit | `5126fead3327c7544ed4becff93775505071334c`（2026-11-16 20:16:27 +08:00） |
| 主要 DSL | Ascend C |
| 许可证 | 根 `LICENSE` 为 MIT（版权年份/持有人仍为占位符）；S9 `Concat` 提交目录另有 Apache-2.0 `LICENSE`；各提交可能不同 |
| 本地缓存 | `repository-research-cache/gitcode-selected/ops`（按固定 commit 下载的选中文件）；完整树为 `repository-research-cache/GitCode_cann-ops-competitions` |
| 与既有归档关系 | 与既有 26 份 GroupedMatmulGelu 公开 PR **同仓但不同赛事/算子**；不重复计数 26 份 PR，也不把本仓当作新的独立算法来源 |
| 本报告范围 | 深读 `recompute_w_u`（Delicate02，天梯赛 7 月）与 `Concat`（BatchLions，S9）两份实现 |

### 1.1 GitCode 访问方式

上一轮记录 `git clone` 返回 `gnutls_handshake() failed`。本轮定位到原因是环境变量 `https_proxy=http://127.0.0.1:29670`：GitHub 经该代理可访问，GitCode 经代理超时；**直连 GitCode 正常**。

```text
curl --noproxy '*' https://gitcode.com/...        # HTTP 200
env -u http_proxy -u https_proxy ... git ls-remote # 5126fead...
```

因此本报告的所有 GitCode 内容都通过直连或 `raw.gitcode.com` 固定 commit URL 获取。代理配置未写入仓库，也未修改系统设置。

## 2. 实现一：`recompute_w_u`（Delicate02）

### 2.1 语义与约束

来源：`01_official/cann-ops-ladder-2026/July/recompute_w_u/README.md`、提交者 `README.md`。

```text
u = (A · diag(beta)) @ v
w = (A · diag(beta) · exp(g)) @ k
```

这是 Qwen3.5/Qwen3-next 的 GDN（Gated Delta Net）Prefill 阶段 chunk-wise 算法的一部分：先做 WY 分解/下三角求解，再重算 `w/u`。`chunk_offsets` 在运行时给出变长序列的分块边界。

关键约束（提交 README 与 host tiling）：

- `B=1`，`T` 可变（题面给 1k–256k），`H` 1–128，`Hg` 1–32，`D` 32–256，`BT=64`；
- 输入 `A[B,H,T,BT]`、`k[B,T,Hg,D]`、`v[B,T,H,D]`、`beta[B,H,T]` 为 BF16；`g[B,H,T]` 为 FP32；`chunk_offsets` 为 int32；
- 输出 `w[B,H,T,D]`、`u[B,H,T,D]` 为 BF16；
- `H % Hg == 0`，GQA 的 key 头按 `h / (H/Hg)` 复用；
- 精度要求：相对误差 ≤1e-5 或绝对误差 ≤1e-6（题面）。

### 2.2 固定源码位置

| 内容 | 路径/行 |
|---|---|
| kernel 入口、10 个 tiling key 分派 | `.../Delicate02/code/op_kernel/recompute_wu.cpp:1-70` |
| 模板参数/10 个模式常量 | `.../code/op_kernel/recompute_wu_tiling_key.h:10-43` |
| tiling 数据结构 | `.../code/op_kernel/recompute_wu_tiling_data.h:10-29` |
| 常量、workspace 布局、GQA/chunk 上限 | `.../code/op_kernel/recompute_wu.h:19-83` |
| 直读 K / compact workspace 判据 | 同上 `:143-199` |
| Vector 侧 `Process`、batch chunks、D128 group pipeline | 同上 `:228-268,591-700,686-792` |
| Cube 侧 `Mmad + Fixpipe`、D128 四槽、group pipeline | 同上 `:1025-1105,1313-1395` |
| host 路径判据、tiling key、workspace | `.../code/op_host/recompute_wu_tiling.cpp:46-67,204-345,346-390,408-441` |
| 作者性能表 | `.../Delicate02/README.md:60-76` |
| ACLNN 示例 | `.../code/examples/test_aclnn_recompute_wu.cpp:1-280` |

关键文件 SHA256（固定 commit 下载副本）：

```text
recompute_wu.h          49954f374ae8c1bb01cfd4c0f7497bec337ef7d573b109f9a3c8a463e78eafe9
recompute_wu_tiling.cpp 519b3b027b4eebb24b1e7515237c25f988bd25f379969ce503c94a632c5796b1
README.md               070b35f7818d2822e1f1cb89d0ca6cc368a1d168fa132e528d54979639c48204
```

Concat 关键文件：

```text
op_host/concat.cpp        8c51610aeee84cfb1e7c4c2dac8b3f82d7e7294cd72072752a772065a0f00224
op_kernel/concat.cpp      58e308ccff10c36d7428841e5786ce994c69707f236c391e6038b589e15eb3bb
op_host/s9_concat_layout.h 0228a138666f45586a4eb4162b8bed5950d0ec733bf2767189a2c3f067c01e65
```

完整清单以 [`catalog/external-repositories.json`](../../catalog/external-repositories.json) 的 `source_files[].sha256` 为准。

### 2.3 机制

#### A. 按 head/chunk 合并任务，GQA 共享 K

`ProcessBatchChunks`（`recompute_wu.h:591-684`）把一个 `(chunk group, group head, head batch)` 作为一个任务：

- `groupSize = H / Hg`，一个 group head 对应多个 query head；
- 同一个 group head 的 `k` 矩阵只打包一次（`:642-647`），后续 head 复用；
- `A * beta` 与 `A * beta * exp(g)` 的向量预处理按 head/chunk 展开（`:635-651`）；
- 一次最多 `MAX_BATCH_CHUNKS_PER_TASK=4` 个 chunk、`MAX_HEADS_PER_TASK=4` 个 head（`:43-45`）。

这减少的是 **K 的重复 GM 读取和右矩阵打包次数**，不是 GEMM 的算术量。

#### B. 多槽 workspace + Vector/Cube 跨核流水

Vector 侧按 `PIPELINE_DEPTH=5` 或 `2` 维护多个 workspace 槽（`:46-51,270-275`），Cube 侧用 `CrossCoreSetFlag/WaitFlag` 消费：

- Vector 写完 `w` 的 A/K 准备区后发 `W_READY`；
- Cube 算完 `w` 后发 `DONE`，Vector 再准备 `u`；
- `u` 用 `U_READY`，Cube 算完发 `DONE`；
- D128 group pipeline 额外用 `GROUP_PIPELINE_DEPTH=3` 和 group 级 flag（`:47-48,686-792,1313-1395`）。

这减少了 Vector 与 Cube 的串行等待，代价是 workspace 增大、flag 协议复杂。

#### C. `Mmad + Fixpipe` 与 D128 四槽

`RecomputeWuCubeMmad`（`:1025-1105`）直接使用 `Mmad`/`Fixpipe`，而不是只依赖 Matmul 高阶 API：

- `SetFixpipeNz2ndFlag(1,0,0)` 让 Fixpipe 直接写 ND（`:1051-1053`）；
- D128 路径只需要一半的 B/C slot，因此把两个 chunk 的共享 K 矩阵放在 slot 0/1，再用 slot 2/3 做 head batch（`:1107-1147`）；
- `RunCubeCachedL0B`（`:1359-1363`）把共享 K tile 固定在 L0B，多个 head 复用；
- staged output 把非 16 对齐的 `K/V` 先写 workspace，再按 16 对齐搬运（`:175-191,834-885`）。

#### D. 运行时路径选择，不依赖测试点编号

host `TilingFunc`（`:204-441`）按以下量选路：

- `groupSize`、`taskCount`、`blockDim`、`keyDim/valueDim`；
- 是否 `uniformFullChunks`、`batchChunkCandidate`、`chunksPerTask`；
- 是否 `useDirectKeyPath`、`useOfficialMmad`、`usePackedD128HeadBatch`、`useD128GroupPipeline`、`useStagedOutput`、`usePackedWide`、`useBatchChunks`；
- 10 个 tiling key：`NORMAL/MDL/OFFICIAL_MMAD/NORMAL_PACKED_WIDE/BATCH_CHUNKS/OFFICIAL_MMAD_D128_DIRECTV/D128_GROUP4/D128_GROUP_PIPELINE/SMALL_GROUP_DIRECTV/STAGED_OUTPUT`（`recompute_wu_tiling_key.h:10-20`）。

作者明确说“所有条件均基于维度、任务波次和片上存储容量，不依赖测试点编号或固定完整 shape”（提交 README:26）。

### 2.4 作者证据

| 项 | 值 |
|---|---|
| 平台 | Ascend 910B3 |
| CANN | 8.5.0（提交 README:8,64） |
| 评测 | CANNJudge 提交 `119991`：22/22 Pass；同源码重投 `121028`：22/22 Pass |
| 榜单 | 2026-07-31 快照排名第 1（动态榜单，可能变化） |
| 计时 | CANNJudge 设备侧 Task Duration，单位 µs |
| 22 个测试点 | 最小 18.00 µs；最大 2884.56 µs；其余见 README:68-74 |
| 输入 | 题面 shape B=1, T 可变, H/Hg/D 可变, chunk_offsets 变长 |

这些是作者环境/平台记录（`O+A`），不是本机复现。

### 2.5 本机构建与运行

在独立副本 `local-validation-20261116/gitcode_recompute_wu` 中，用 CANN 9.1.0 + gcc 11.4.0 + cmake 3.22.1 重新构建，未修改源码：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cd code
bash build.sh -j8
```

结果：**构建成功**，生成 `custom_opp_ubuntu_aarch64.run`，kernel 二进制
`RecomputeWu_dc22c0ec8550d8469dfa43564b045d22.o`，包内 `version.info` 记录
`custom_opp_compiler_version=9.1.0`。

运行：用自写的 ACLNN 驱动（不属于第三方仓库）加载 `ASCEND_CUSTOM_OPP_PATH`，输入全 1、`g=1`、合法 `chunk_offsets`，解析期望 `u=chunk_len`、`w=chunk_len*e`。尝试了 5 个 shape：

| shape | 结果 |
|---|---|
| T=256, H=8, Hg=8, D=128 | `507057` AI Core kernel execution failed |
| T=256, H=8, Hg=2, D=128 | `507057` |
| T=256, H=8, Hg=2, D=256 | `507057` |
| T=10016, H=8, Hg=8, D=128 | `507057` |
| T=10016, H=8, Hg=2, D=128 | `507057` |

错误摘要：

```text
fault kernel_name=RecomputeWu_dc22c0ec8550d8469dfa43564b045d22_7
[DFX_INFO]AI Core kernel execution failed ... retCode=0x26
rtStreamSynchronize: ErrCode=507057, desc=[suspect remote error]
```

尝试过给 10 个 tiling key 显式加 `KERNEL_TASK_TYPE(..., KERNEL_TYPE_MIX_AIC_1_2)`（本地兼容性实验，未改第三方仓库原文），仍失败。因此结论是：

- **编译层面**：CANN 9.1.0 可以编译该 CANN 8.5.0 提交；
- **运行层面**：在当前 910B3/CANN 9.1.0 上，至少 D128 group pipeline 路径会启动后崩溃；不能把作者 22/22 当作本机复现；
- 失败根因尚未定位到单条指令/UB 越界；可能是 CANN 9.1 的编译/同步/UB 行为差异，也可能是本驱动的输入/输出布局与算子隐含假设不一致。复用时必须先按作者 CANN 8.5 环境复跑，再逐路径移植。

原始日志：

- [`gitcode_recompute_build.log`](../../our_work/evidence/local-910b-20261116/gitcode_recompute_build.log)
- [`gitcode_recompute_run3.log`](../../our_work/evidence/local-910b-20261116/gitcode_recompute_run3.log)

### 2.6 适用条件与负例

适用：

- BF16 GDN/GQA 的 `recompute_w_u`，变长 chunk；
- `B=1`、`H/Hg/D` 在题面范围；
- 能接受单 kernel 内 Vector/Cube 流水、复杂 workspace 和 flag 协议；
- 目标 CANN 与作者环境接近（8.5.0），且能按 shape 覆盖所有 tiling key。

负例/风险：

- **不是通用 GEMM**：语义绑定 `A` 的 chunk 结构和 GQA key 复用；换布局要重写打包。
- **B=1 硬约束**：host 直接拒绝 `B!=1`（`recompute_wu_tiling.cpp:197-199`）。
- **`BT=64` 硬约束**：`chunkSize != 64` 直接失败（`:210-212`）。
- **K/V ≤256**：`:212-214`。
- **CANN 版本敏感**：本机编译成功但运行 `507057`；不能把“编译通过”当“可运行”。
- **动态榜单**：第 1 名是 2026-07-31 快照，不是永久成绩。
- **没有本机性能数字**：运行失败，不能给出本机 µs 或加速比。

### 2.7 复用判断

优先复用：

- **按 group head 合并 K、按 head batch 复用 L0B**：适合 GQA/多头共享 key 的 GEMM 类算子；
- **多槽 workspace + 跨核 flag 的生产者/消费者流水**：适合 Vector 预处理 + Cube 矩阵乘的融合；
- **`Mmad + Fixpipe` 直接写 ND、D128 四槽**：适合 D=128 的小矩阵批量乘；
- **运行时按 shape/task wave 选 tiling key**：适合多 shape 赛题，但必须做全 key 回归。

暂不直接复用：

- 在 CANN 9.1/CANN 8.5 未对齐前，不把该 kernel 作为 910C 或新 CANN 的模板；
- 不把 22/22 或第 1 名当作本机正确性/性能证明；
- 不在 `B!=1`、`BT!=64`、`D>256` 的场景直接套用。

## 3. 实现二：S9 `Concat`（BatchLions）

### 3.1 语义与证据

`Concat` 沿 `dim` 拼接动态数量的输入张量，等价 `torch.cat`；支持 float16/float32/int8/int32，负轴、零长度分片、最多 256 路（aclnn 动态张量列表上限）。

作者证据（`README.md:3-9`）：

- 团队 BatchLions；S9 算子挑战赛 Concat；
- 版本 v45-tight；原始提交包 SHA256 `c11963b8...cab0a`；
- 最终榜单第十名，五档合计 **348.33 µs**，64 次提交；
- 数据取自最终赛事榜单截图，**本次合仓未重新运行隐藏用例**。

这是 `O+A`：有最终榜单数字，但没有逐 case 原始日志/计时；不能当本机复现。

### 3.2 固定源码位置

| 内容 | 路径/行 |
|---|---|
| Host 入口、平台核数、动态输入 | `op_host/concat.cpp:55-263` |
| 单路非空窄行整块拷贝（`S9_COPY1`） | 同上 `:240-262` |
| 策略选择/`S9_STRAT`、已删除 strategy 5 的负结果 | 同上 `:267-330` |
| 列带、最终配置校验、`CheckConcatFinal` | 同上 `:440-532` |
| AIC+AIV 混合、tiling key、workspace | 同上 `:625-680` |
| 纯算术布局：UB 预算、核映射、带区间 | `op_host/s9_concat_layout.h:96-117,163-241,355-435,603-696` |
| Host 选路/带数/`CalcConcatLens`/`GatherFitsBands`/`CheckConcatFinal` | `op_host/s9_tiling_calc.h:493-620,635-656,752-997,1235-1410` |
| Kernel 主体、核/带/布局 | `op_kernel/concat.cpp:264-630` |
| 直写带路径 `DirectBanded` | 同上 `:909-1071` |
| Gather 拼装路径 `GatherAssembled` | 同上 `:1075-1445` |
| 离线测试入口 | `tests/run.sh:1-20`；`tests/twin_concat.cpp:1-1371`；`tests/test_concat_layout.cpp:1-835`；`tests/test_single_band_budget.cpp:1-27` |

### 3.3 机制

#### A. 先把 shape 归一成 `rows × n_i`，再选搬运策略

Host 把输入归一为：

```text
rows = ∏shape[:dim]
n_i  = ∏shape_i[dim:]
nOut = Σn_i
```

每个输入占输出的列区间 `[off_i, off_i+n_i)`。然后按 `rows`、`nOut`、路数、对齐、UB 容量选择：

- 按行切（默认，burst 最长）；
- 按列切（行切用不满核时）；
- DMA 拼装 / Gather 拼装；
- 按输入分核；
- 输出线性切分（`rows==1`）；
- AIC+AIV 混合整块搬运（`rows==1` 且 `S9_AIC` 开启）。

`TilingKey = 元素字节宽度`（1/2/4/8）；`+16` 表示混合核（`op_host/concat.cpp:663-671`）。

#### B. Gather 拼装：连续读 + 偏移表 + 一次连续写

`GatherAssembled`（`op_kernel/concat.cpp:1075-1445`）：

- 每条输入按带内裁剪后的列区间发一条连续 `DataCopyPad` 到 UB 暂存；
- 构建字节偏移表，用 `Gather` 按元素粒度把多路拼成整行；
- 再一条连续 `DataCopyPad` 写出；
- int8 借道 half 走 Gather（`S9GatherAs<int8_t>`，`:67-80`），因为 Atlas A2 的 Gather 不支持 int8；
- 偏移表一次摊开 `tileG` 行，每个 tile 只发一次 Gather（`:476-488`）；
- 单带且行首尾相接时，把多行写合并成一块（`oneWriteOn`）。

这解决的是“每路每行一条小 DMA、写碎片化”的问题；代价是 UB 暂存/偏移表和 `Gather` 的向量开销。

#### C. 列带切分：读侧变大、写侧变碎

`MapConcatCore` / `BandRange`（`s9_concat_layout.h:163-241`）把核排成 `bands × (blockDim/bands)` 网格：

- 每个核只处理输出的一段列带；
- 一条带里只与相交的输入路做 DMA；
- 一行只占 `bw·esz` 而不是 `nOut·esz`，同样 UB 能装更多行，读 DMA 变大 `bands` 倍；
- 代价是写从整行连续变成 `bands` 段。

作者做了大量 A/B：

- 均衡切分（`S9_BAL=1`）在 G16 4096×16384 f16 246 路上比旧法慢 **1.35%**，因为关键路径由最宽带决定，空转核不浪费带宽（`op_host/concat.cpp:173-215`）；
- 边界路裁剪（`S9_TRIM`）在三个形状上分别慢 8.0%、7.9%、1.0%，机理是“少读的字节抵不上跨步 DMA 的开销”（`:189-213`）；
- 整块取 TilingData 头（`S9_PRE=1`）在 128×256 f16 9 路上慢 **8%**（`:642-648`）；
- 早期“按（路×行段）切 + 大块 DMA”的 strategy 5 在 246 路上慢 **4.33×**，代码已删除（`:278-289`）。

#### D. L2 模式与混合核

- 默认输入/输出 L2 都 `NORMAL`；作者实测小 shape 11.10→8.44 µs、大 shape 37.74→26.62 µs，`PERSISTENT` 反而更慢（`op_host/concat.cpp:317-330`）；
- `rows==1` 时可用 AIC+AIV 混合搬整块；AIC 只搬 32B 对齐连续块，AIV 处理另一半（`:625-660`）。

### 3.4 离线验证（本机）

`bash tests/run.sh` 不需要 CANN/NPU，本机通过：

```text
Concat 来源映射与地址回放：215242 项通过
布局、对齐与 Host/Kernel 布局一致性：32681 项通过
独立累计实际暂存需求：18000 组随机配置中 8247 个可行布局通过
```

日志：[`gitcode_concat_offline_tests.log`](../../our_work/evidence/local-910b-20261116/gitcode_concat_offline_tests.log)。

这些测试证明地址映射、输出覆盖、UB 布局/对齐和 Host/Kernel 布局一致性；**不证明** NPU 编译、精度或性能。本机没有构建/运行 Concat 的 NPU 包。

### 3.5 适用条件与负例

适用：

- 动态输入列表、最多 256 路、多种 dtype 的 Concat；
- 行数/列宽/路数变化大，需要按 shape 选搬运策略；
- 能接受复杂的 UB 预算和 Host/Kernel 共享布局头；
- 目标 910B（作者在 910B4/910B1 上量测）。

负例/风险：

- **不是所有形状都快**：整块拷贝只对窄行（≤128B）有利；宽行会慢 11%–19%；
- **分带不总是收益**：列带切分增加写碎片，均衡切分反而慢 1.35%；
- **裁剪不总是收益**：三个形状都变慢；
- **拼装路前提严格**：需要 16/32 位 dtype、nOut 上限、UB 容量、32B 对齐；不满足会降级或触发 507035；
- **aclnn 动态张量列表上限 256**：超过到不了算子；
- **榜单第十名是最终快照**：没有逐 case 日志，不能复算 348.33 µs；
- **CANN 8.5.0 目标**：CANN 9.1 的编译/运行未验证。

### 3.6 复用判断

优先复用：

- **Host/Kernel 共享纯 C++ 布局头**：`s9_concat_layout.h` 把 UB 预算、核映射、带区间、tileG 只写一份，Host 和 Kernel 都调用；适合任何“布局复杂、Host/Kernel 易漂移”的搬运算子；
- **先归一成 `rows×n_i`，再按 shape 选策略**：比“一个 kernel 通吃”更稳；
- **Gather 拼装 + 偏移表摊开 tileG 行**：适合多路小片段拼连续输出；
- **单路非空窄行整块拷贝**：精确等价，收益可达 1.54×–674×（作者 910B1 行宽扫描）；
- **最终配置统一校验/收敛/留痕**：`CheckConcatFinal` 防止“Host 说放得下、Kernel 越界”。

暂不直接复用：

- 不把 348.33 µs 当本机成绩；
- 不在宽行/大 nOut 上默认开整块拷贝或边界裁剪；
- 不在未做 UB 实账的情况下改 `blockDim/bands/gBuf/aBuf`；
- 不把 CANN 8.5 的编译结果直接搬到 9.1/910C。

## 4. 同仓候选（未深读）

| 实现 | 目录 | 观察 | 本轮决定 |
|---|---|---|---|
| S9 `IndexAdd`（VibeThinking） | `.../S9/submissions/IndexAdd/VibeThinking` | README 说明按 `outer/axis/inner` 选路、行缓存、分组累加、重复索引；无性能数字 | 候选；后续可与 Scatter/Gather 专题一起深读 |
| S9 `Greater`（多队） | `.../S9/submissions/Greater/*` | 逐元素比较，机制简单，性能上限低 | 候选；不作为高性能主来源 |
| 天梯赛其它月份/算子 | `.../cann-ops-ladder-2026/{June,August,September}` | 目录存在，部分只有题面/`.gitkeep`；需逐月筛选 | 候选；下一轮按“有提交+有日志”筛 |
| 2025 挑战赛/高校赛 | `01_official/cann-challenge-2025`、`03_university` | 算子族多，但与 910B/2026 目标不完全一致 | 候选；先不扩范围 |

## 5. 证据级别

| 项 | 级别 |
|---|---|
| 仓库/固定 commit/源码路径 | `S+L` |
| recompute_wu 作者 22/22、第 1 名、逐 case µs | `O+A` |
| recompute_wu 本机编译成功 | `V` |
| recompute_wu 本机运行 | `V`（失败，507057） |
| Concat 作者第十名/348.33 µs | `O+A` |
| Concat 本机离线布局/地址测试 | `V`（CPU） |
| Concat 本机 NPU 编译/运行 | `N` |
| 910C / 其它 CANN | `N` |

## 6. 去重与计数

- 本仓与既有 26 份 GroupedMatmulGelu 公开 PR 同属 `cann/cann-ops-competitions`；本报告只新增 **2026 天梯赛 recompute_w_u** 与 **S9 Concat** 两份实现，不重复登记 26 份 PR。
- 因此本仓在 catalog 中登记为“同一官方仓的新赛事实现”，**不增加独立算法来源数**；新增实现数为 2。
- 两份实现都不与既有 15 份实现重复：一份是 GDN 注意力重算，一份是多输入 Concat 搬运。
