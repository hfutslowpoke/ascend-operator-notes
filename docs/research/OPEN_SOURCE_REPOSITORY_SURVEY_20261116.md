# 公开比赛与高性能算子仓库调研（2026-11-16，第一/二批 + GitCode 补充）

状态：第一/二批 + GitCode 补充来源筛选与源码深读完成；纯 Triton 任务、Be4ver 的两个 Ascend C 算子和 GitCode Lstsq 小尺寸已在本机 Ascend 910B3 复测。Parity-LRX 未在本机构建。未向比赛平台提交，未修改当前算子工程或 ACW 框架。除明确标注“本机复测”的数字外，本文中的“加速比”都是原作者/原报告在各自环境下的数字。

> 2026-11-17 增量：新增 **6 个制造专场官方/生产仓**（ops-fft、DrivingSDK、ops-gnn、ops-sparse、ops-math、ops-solver）和 **15 份实现**，覆盖 15 道制造赛题中的 13 道（2 道未找到公开源码）。当前全库为 **20 个仓库记录、18 个独立算法来源、15 个深读仓库、33 份实现**。本文以下第 1–8 节保留 2026-11-16 批次的原始口径，制造专场增量见 [competition/mfg-2026/README.md](../../competition/mfg-2026/README.md)。

## 1. 结论摘要

- 本轮登记 **14 个仓库记录**，其中 **12 个独立算法来源**；**9 个仓库记录**完成源码级深读（含 1 个负例；GitCode 2 个记录与既有官方基线/Hansen 上游重叠，不新增独立来源），覆盖 **18 份实现**、至少 15 类算子族：检测框解码融合、图样条卷积、点到线段最近距离、MoE 路由与专家计算、分组 TopK、Sinkhorn 归一化、Mamba-2 分块扫描、线性层+SwiGLU 融合、Softmax/Matmul 负例、批量 Jacobi SVD、QDWH 极分解/SDC、O(3) 张量积、长程静电、容量调度图消息、GDN 注意力重算、多输入 Concat、批量最小二乘。
- 与既有归档的 26 份 GroupedMatmulGelu 公开 PR **没有重复**。既有官方 `cann-ops-competitions` 快照仍作为基线引用，本轮不重复计数；GitCode 本轮已直连成功，补充了 2 个官方仓、3 份新实现（详见 [GitCode 补充调研](GITCODE_SOURCE_SURVEY_20261116.md)）。
- 第一批最值得后续验证的 5 项机制见第 6 节：按权重索引先聚合、选择性高精度修复、分块扫描+消费者布局+跨核流水、去中间张量的生产者消费者融合、按规模在“稠密全专家”和“动态分发”之间选路；第二批新增矩阵内配对重排、单 kernel 多阶段融合、有界 stencil/容量调度三组候选；GitCode 补充新增任务批处理/GQA K 复用、布局优先搬运、补偿算术+迭代精化三组候选。
- 最强的新证据来自：
  - `mantuoluozk/kernelswift-2026-triton-kernels`：Ascend 910B1、CANN 9.0.0、Triton-Ascend 3.2.1，官方评测器 `warmup=200`、`repeat=500`、中位数，10/10 正确性通过；本轮深读其中 3 题。
  - `So-cean/mamba-ascendc`：Ascend 910B3，device event p50，同平台 PyTorch/Triton-Ascend/AscendC 三层对照，并有 msprof 阶段分解；本轮深读 ChunkMix。
  - `vLLM-HUST/vllm-hust-opset`：Ascend 910B2，算子级 10 warmup/50 同步样本，端到端 3 轮匹配；本轮深读 Linear+SwiGLU Graph 融合。
  - `Be4ver/svd`：Ascend 910B3，PACO 批量 Jacobi 与 QDWH+SDC 融合；作者有设备日志/精度验证，本轮在 CANN 9.1.0 上重新构建并复测。
  - `Parity-LRX/ICTC-O3-HP-Ascend`：Ascend 910B3/CANN 9.0.0，packed O(3) TP、mesh 静电、capacity chunk；MIT，性能协议完整，但仓库只有文档/图、无逐点原始日志。
  - GitCode `cann/cann-ops-competitions`：Delicate02 `recompute_w_u` CANNJudge 22/22、2026-07-31 第 1 名；BatchLions S9 `Concat` 最终第十名、五档 348.33 µs。
  - GitCode `cann/elec-ops-inspection`：XDU_XDU-BTD `LinalgLstsq`，补偿算术+迭代精化；作者无逐 case 性能数字，本机小尺寸单 batch 通过。
- **本机 910B3 复测**：GroupedTopk PASS 2.670×（作者 2.478×）、Sinkhorn PASS 13.226×（作者 9.203×）、52Xayah Softmax/Matmul 负例 0.288×/0.242×稳定复现；Be4ver PACO 32×32×512 为 6.174 ms（作者 7.061 ms，512/512 PASS），QDWH+SDC 128/512/1024 为 4.288/20.698/48.475 ms（作者 4.392/20.540/46.839 ms，全 PASS）；GitCode Lstsq 16/50/64 单 batch PASS（best 0.0265/0.0301/0.0395 ms），但 batch>1/矩形/200/256 失败；GitCode recompute_w_u 编译成功但运行 507057；FusedMoE 在当前 TA 3.2.0/CANN 9.1.0 上 UB 溢出，作者 11.446× 未复现。详见 [本机复测记录](../validation/LOCAL_910B_VALIDATION_20261116.md)。
- 证据最弱的深读项是 `Hansenwuwuwu/ascend-c-competition-archive` 的三份 2026 电力行业算子：有官方公示分数和名次，但没有逐 case 时延，也没有本机编译/运行；因此只把它当作“获奖源码的结构证据”，不把分数换算成加速比。
- `52Xayah/triton-ascend-kernels` 的 Softmax/Matmul 是明确负例：原作者批量计时约 0.140 ms vs PyTorch 0.035 ms、0.699 ms vs 0.163 ms；本机逐 sample 协议为 0.288×/0.242×。它提醒我们，手写 Triton 不会自动超过厂商库，必须先把 tile、数据布局、融合和启动开销算清楚。

## 2. 范围、搜索与证据方法

### 2.1 搜索范围

- 日期：2026-11-16。
- 平台：GitHub 公开仓库为主；GitCode 官方比赛仓/行业仓在补充轮已直连成功。上一轮 `git clone` 的 `gnutls_handshake() failed` 由环境代理导致，`--noproxy '*'` 或去掉 `https_proxy` 后可正常访问；详见 [GitCode 补充调研](GITCODE_SOURCE_SURVEY_20261116.md)。
- 检索词（实际使用或组合使用）：
  - `ascend c competition`
  - `cann operator competition`
  - `ascend operator`
  - `ascendc`
  - `triton ascend`
  - `KernelSwift 2026`
  - `2026 电力 算子`
  - `Mamba AscendC`
  - `vLLM Ascend operator`
  - `SVD Ascend`
- 发现方式：GitHub repository search、仓库 README/许可证/提交历史、原始源码、作者报告、官方评测器。搜索引擎和榜单只用于发现，技术结论回到固定 commit 的源码或原始报告。

### 2.2 证据分级

本文统一使用以下标记，避免把“作者说快”与“本机测到快”混在一起：

| 标记 | 含义 |
|---|---|
| `S` | 静态源码观察：路径、符号、数据流、同步、容量等可在固定 commit 核对 |
| `A` | 作者声明：README、注释、设计文档中的解释或数字 |
| `O` | 原始平台/评测记录：官方公示分数、官方评测器输出、原始 benchmark 报告 |
| `L` | 本地缓存/固定版本核对：clone 后 commit、文件哈希、路径存在 |
| `V` | 本机 Ascend 910B3 复测：通过正确性并记录时延，或记录编译/运行失败 |
| `N` | 本批未复现：本环境没有编译/运行/ profiler/平台提交 |

一个结论可以同时有 `S+L`，但没有 `V` 或 `O` 之外的实验证据时，不写成“已验证性能”。本批：Hansen/Mamba/vLLM-HUST/Parity 仍为 `N`；KernelSwift、52Xayah、Be4ver 的部分实现为 `V`。

### 2.3 本地缓存

第三方源码缓存在工作区外：

```text
/workspace/acw/var/private/repository-research-cache/
```

缓存只用于阅读和行号定位，不进入 Git，不包含账号、令牌、构建产物或 profiler 原始目录。固定 commit 和关键文件 SHA256 见 [来源登记](../../index/catalog.json)。

## 3. 去重、筛选与淘汰

### 3.1 与既有归档的关系

既有私仓保存了 26 份 2026 CANN 挑战赛上合赛区 GroupedMatmulGelu 公开 PR、248 个原始文件，以及我们的最终源码、组件、证据和复盘。本轮：

- 不重复登记这 26 份 PR；
- 不把官方 `cann/cann-ops-competitions` 的同一批快照当成新仓库；
- 不把 fork、镜像、同一源码的二次搬运算作独立来源；
- 只在新仓库的算子族、机制或证据与既有归档不同时计入。

### 3.2 14 个来源记录（其中 12 个独立）的取舍

| # | 来源 | 类型/年份 | 本轮决定 | 理由 |
|---:|---|---|---|---|
| 1 | `Hansenwuwuwu/ascend-c-competition-archive` | 2026 电力行业算子，Ascend C | **深读** | 9 题 27 份公开提交 + SwiGLU 3 份补充；源码完整，含题面、哈希、逐行审计；与 GroupedMatmulGelu 不同题 |
| 2 | `mantuoluozk/kernelswift-2026-triton-kernels` | 2026 KernelSwift，Triton | **深读** | 四平台 10 题，Ascend 910B 有官方评测器、完整环境、正确性和中位数成绩 |
| 3 | `So-cean/mamba-ascendc` | 生产型 Ascend C，Mamba-2 | **深读** | 910B3 原生前向/反向，三层实现对照 + msprof + 明确限制；算子族新 |
| 4 | `vLLM-HUST/vllm-hust-opset` | 生产型 vLLM Ascend，Triton | **深读** | 小而完整：算子级/端到端/负例/回退边界齐全；有原始外部报告 |
| 5 | `52Xayah/triton-ascend-kernels` | 教学型 Triton，A2 | **深读（负例）** | 三个算子、明确负结果；用于校准“手写 Triton 不一定快”的判断 |
| 6 | `Gxj230958/TritonAscendBench` | 迁移基准，101 个生产 kernel | 候选 | 正确性/来源冻结做得强，但作者明确说 910B 性能尚未接通；不能当性能证据 |
| 7 | `elstainniles/triton-operator-benchmarks` | Triton 基准集合 | 候选 | 条目多、测试细，但主要是 FlagGems/Liger/FLA 导入，且历史环境是 950PR；独立算法与 910B 证据不足 |
| 8 | `Yunle-Lee/CANN-Operator-test-competition-personal-solution` | 比赛提交/测试模板 | 淘汰（不计高性能） | 主要是 Add/Mul/Pow/Cumsum 的 aclnn 测试代码和赛事目录，没有 Ascend C 高性能 kernel 或时延 |
| 9 | `Be4ver/svd` | Ascend SVD，C++/Ascend C | **深读（第二批）** | PACO Jacobi + QDWH/SDC 两份实现，有 910B3 设备日志和精度验证；本轮在 CANN 9.1.0 上重新构建，PACO/QDWH 共 4 组运行全 PASS；根目录无 LICENSE |
| 10 | `Parity-LRX/ICTC-O3-HP-Ascend` | 生产型 O(3)/MACE，Ascend C | **深读（第二批）** | 910B3 上 3.90× forward / 3.85× fwd+VJP / 4.76× training / 1.90× MD 等作者证据，MIT；深读 packed TP、mesh 静电、capacity chunk 三份实现；本机未构建 |
| 11 | `U-Illll/ascend-eigh-operator` | 单算子 eigh，Ascend C | 候选 | 语义、tiling、双精度路径清楚，但只有单算子、没有性能数据 |
| 12 | `asfsdsf/S6` | 单算子 Fmax，Ascend C | 淘汰 | 只有 6 个源文件和 LICENSE，无 README、无测试、无性能证据；不足以支撑“高性能”判断 |
| 13 | GitCode `cann/cann-ops-competitions` | 2026 天梯赛/S9，Ascend C | **深读（GitCode 补充）** | 与既有官方基线同仓、不同赛事/算子；深读 recompute_w_u 与 Concat；不新增独立来源 |
| 14 | GitCode `cann/elec-ops-inspection` | 2026 电力行业，Ascend C | **深读（GitCode 补充）** | Hansen 归档的上游官方仓；新增 LinalgLstsq 实现；不新增独立来源 |

因此“14 个仓库记录、12 个独立算法来源、9 个深读仓库记录、18 份实现”的口径是：GitCode 两个官方/上游记录已登记并深读，但按去重规则不重复增加独立来源。

另外见过但未计入 12 个独立来源的：

- `WildPickledPepper/ascend-rowcol-int8-quant`：只有任务书、benchmark/golden 脚本和配置，未看到 kernel 源文件；暂不纳入。
- `m-rui001/ascend-c-operator-dojo`：59 轮学习/对照笔记和检查表，偏训练材料，不是独立的高性能算子实现；暂不纳入。
- `CosmosMind-ai/PyPTO-Ops`、`Luoshu-Intelligent-Computing/ascend-310p-kernels`：前者是新 DSL 的算子库，后者是 310P；与 910B/910C 目标差异大，留作后续专题。

### 3.3 官方基线

| 基线 | 固定版本 | 说明 |
|---|---|---|
| `cann/cann-ops-competitions` | `776f2e552a125d3c210b497385f8b57292ebdc92`（既有归档） | 26 份 GroupedMatmulGelu 公开 PR 已在本库 `competition/grouped-matmul-gelu/prs`；本轮不重复抓取 |
| 本库 GroupedMatmulGelu 最终源码 | 见 `competition/grouped-matmul-gelu/our-submission` 与 `competition/grouped-matmul-gelu/catalog/our-work.json` | 历史成绩 58.54 分、第 11、整包 Pass；不是本轮新证据 |

## 4. 仓库对比

| 来源 | 比赛/场景 | 主要 DSL | 算子族 | 固定 commit | 许可证 | 证据强度 | 本轮用途 |
|---|---|---|---|---|---|---|---|
| Hansen archive | 2026 电力行业算子 | Ascend C | 解码融合、图卷积、几何搜索、SwiGLU、线代 | `37e07aba5273e280d2d227d8764c5e49b7eb3eae` | 未声明 | 官方分数 + 静态源码；无逐 case 时延 | 深读 3 题 |
| mantuo KernelSwift | 2026 KernelSwift 赛道一 | Triton / Triton-Ascend | MoE、TopK、Attention、归一化、增强 | `ac937591f5517b0dfc7c172252f0f27caf772d34` | 未声明 | 官方评测器 + 910B 环境 + 正确性 | 深读 3 题 |
| So-cean mamba | 生产型 Mamba-2 | Ascend C + Triton 对照 | SSM/扫描、归一化、门控 | `445a5c001dc3cc09fcec649824f2330634d753f9` | Apache-2.0 | device event + msprof + 同平台三层对照 | 深读 ChunkMix |
| vLLM-HUST opset | 生产型 vLLM Ascend | Triton + FX pass | 线性层+SwiGLU、持久 MatMul | `58acc93cfb39838b1ef66c42fb4279db6e7040ba` | MIT | 算子级 + 端到端 + 负例 | 深读 Linear+SwiGLU |
| 52Xayah kernels | 教学/实验 | Triton-Ascend | Softmax、MatMul、Vector Add | `6785cbbaf470dbc126102dcdedb1c9e028d7dd63` | 未声明 | 作者粗略计时；明确负结果 | 负例 |
| Gxj TAB | 迁移基准 | Triton-Ascend | 101 个 vLLM/SGLang kernel | `8a395c87f60a5e0e97f4a48cb16dcc6129e7e850` | Apache-2.0 | 正确性/来源强；性能未接通 | 候选 |
| elstainniles | Triton 基准 | Triton | 逐元素、归一化、FLA、Liger | `e6cf7c979f4b09968c59f25a15a542d282c9c7f2` | Apache-2.0 | 950PR 历史数据 + 大量导入 | 候选 |
| Yunle-Lee | CANN 测试赛 | C++/aclnn | Add/Mul/Pow/Cumsum 测试 | `a0c1c8a8ba8ac203afe28f2f7d9c75b9c06ab595` | MIT | 测试模板；无 kernel/时延 | 淘汰 |
| Be4ver svd | Ascend SVD | C++/Ascend C | Jacobi SVD、QDWH/SDC | `b02796511ea62861c647a56b880da15b6a07b3a0` | 未声明 | 作者设备日志 + 精度；本机 910B3/CANN 9.1.0 构建并复测 4 组 | 深读 2 题 |
| Parity ICTC-O3 | 生产型 O(3)/MACE | Ascend C | 等变张量积、长程静电、MD | `ca9ccc4ea9e2e9d21e48daeb45925f2bc4382be8` | MIT | 910B3 多组作者数据 + 完整协议；无逐点原始日志 | 深读 3 题 |
| GitCode CANN 竞赛仓 | 2026 天梯赛/S9 | Ascend C | GDN 重算、Concat、IndexAdd 等 | `5126fead3327c7544ed4becff93775505071334c` | MIT（根；Concat 提交 Apache-2.0） | CANNJudge/最终榜单 + 源码；本机 recompute 编译成功/运行失败，Concat 离线测试 | 深读 2 题 |
| GitCode 电力行业仓 | 2026 电力行业 | Ascend C | 批量最小二乘、线性代数 | `dfe55921db351f71cfb8c96f795fd4472bd6bd47` | Apache-2.0 | 源码 + 作者说明；无逐 case 时延；本机小尺寸单 batch PASS | 深读 1 题 |
| U-Illll eigh | 单算子 | Ascend C | 对称特征分解 | `fa2275fa4b2cee005e6868ce23266686eafa7a1b` | MIT | 源码/契约；无性能 | 候选 |
| asfsdsf S6 | 单算子 | Ascend C | Fmax 广播 | `0454168359b4749058b61432bf15b3945b44efa9` | MIT | 源码极小；无文档/性能 | 淘汰 |

## 5. 深读实现索引

| ID | 来源 | 算子 | 关键源码（固定 commit） | 环境 | 原始证据 | 报告 |
|---|---|---|---|---|---|---|
| `HANSEN-2026POWER-FIRE-R2` | Hansen archive | FireCandidateDecodeFusion (rank2) | `.../FireCandidateDecodeFusion/rank2_submission_1176342/code/op_kernel/device_fire.h:105-143,272-365` | 910B4 / CANN 9.0.0.0 / 20 AIC（比赛环境） | 公示分 83.26；无逐 case 时延 | [Hansen 报告](../../competition/hansen-2026-power/README.md) |
| `HANSEN-2026POWER-SPLINE-R1` | Hansen archive | SplineConv (rank1) | `.../SplineConv/rank1_submission_1174398/code/op_kernel/spline_conv_compact.h:86-94,200-225` | 同上 | 公示分 89.16；无逐 case 时延 | 同上 |
| `HANSEN-2026POWER-TREE-R1` | Hansen archive | TreeLineMinDistance (rank1) | `.../TreeLineMinDistance/rank1_submission_1175341/code/op_kernel/tree_pairs.h:323-515,752-854` | 同上 | 公示分 91.85；无逐 case 时延 | 同上 |
| `KS2026-ASC910B-GROUPEDTOPK` | mantuo | GroupedTopk | `platforms/ascend910b/task01_grouped_topk/solution.py:8-89` | 910B1 / CANN 9.0.0 / TA 3.2.1 | 作者 0.701666→0.283144 ms，2.478×；本机 2.670× PASS | [KernelSwift 报告](../../competition/kernelswift-2026/README.md) |
| `KS2026-ASC910B-FUSEDMOE` | mantuo | FusedMoE | `.../task02_fused_moe/solution.py:8-87` | 同上 | 作者 6.942960→0.606584 ms，11.446×；本机编译失败（UB） | 同上 |
| `KS2026-ASC910B-SINKHORN` | mantuo | hc_split_sinkhorn | `.../task08_hc_split_sinkhorn/solution.py:67-89` | 同上 | 作者 3.032301→0.329475 ms，9.203×；本机 13.226× PASS | 同上 |
| `MAMBA2-ASCENDC-CHUNKMIX` | So-cean | Mamba-2 SSD ChunkMix | `mamba_ascendc_ops/mamba2_ssd_chunk_mix/op_kernel/mamba2_ssd_chunk_mix.cpp:1016-1211` | 910B3 / CANN 8.2.RC1 / TA 3.2.0 | 作者 0.632 ms vs 6.287/4.305；H256 31.888 ms；本机未构建 | [Mamba 报告](../../general/attention/mamba-ascendc/README.md) |
| `VLLMHUST-LINEAR-SWIGLU` | vLLM-HUST | Linear+SwiGLU | `.../linear_swiglu_kernel.py:35-66` | 910B2 / CANN 9.0.0（报告）/ 9.1.0-beta.3（兼容清单） | 作者算子 −15.93~17.92%；e2e TPOT −3.82%；本机未搭建 | [OPset 报告](../../general/transformer/vllm-hust-opset/README.md) |
| `XAYAH-ASC910B-SOFTMAX-NEG` | 52Xayah | Softmax | `src/softmax.py:8-41` | A2 / CANN 9.2.0 / TA 3.2.2 | 作者 0.140 vs 0.035 ms；本机逐 sample 0.288× | [52Xayah 报告](../../general/elementwise/52xayah-triton-kernels/README.md) |
| `XAYAH-ASC910B-MATMUL-NEG` | 52Xayah | Matmul | `src/matmul.py:8-89` | 同上 | 作者 0.699 vs 0.163 ms；本机逐 sample 0.242× | 同上 |
| `BE4VER-PACO-BATCHED-JACOBI` | Be4ver | 批量单边 Jacobi SVD | `paco-jacobi-svd/jacobi_svd_custom.cpp:130-170,215-303,435-538` | 910B3 / CANN 未固定（QDWH 文档记 8.2.RC1） | 作者 7.061 ms；本机 6.174 ms，512/512 PASS | [Be4ver 报告](../../general/linear-algebra/be4ver-svd/README.md) |
| `BE4VER-QDWH-SDC-FUSED` | Be4ver | QDWH+SDC 融合 | `qdwh_sdc_fused_jacobi/fused_qdwh_sdc.hpp:89-188,386-500` | 同上 | 作者 128/512/1024=4.392/20.540/46.839 ms；本机 4.288/20.698/48.475 ms，全 PASS | 同上 |
| `PARITY-O3-PACKED-TP` | Parity | packed O(3) TP | `ictc_o3/backends/ascend/packed_tensor_product.py:60-165` | 910B3 / CANN 9.0.0 / torch 2.7.1 | 作者 3.90× forward、3.85× fwd+VJP；本机未构建 | [Parity 报告](../../general/equivariant/parity-ictc-o3/README.md) |
| `PARITY-MESH-ELECTROSTATICS` | Parity | mesh 静电 assignment | `ictc_o3/ascendc/op_kernel/electrostatics.cpp:22-119` | 同上 | 作者 q+μ+Q +6.8%/+10.7%；particle −40.6%/−51.2%；1.90× MD | 同上 |
| `PARITY-CAPACITY-CHUNK` | Parity | capacity chunk | `ictc_o3/backends/ascend/capacity_message.py:7-80` | 同上 | 作者 196,608 atoms 2.43×、52.68→43.55 GiB | 同上 |
| `GITCODE-RECOMPUTE-WU` | GitCode `cann/cann-ops-competitions` / Delicate02 | GDN 注意力重算 | `.../recompute_wu.h:143-199,591-792,1025-1395` | 910B3 / CANN 8.5.0 | 作者 CANNJudge 22/22、7 月第 1 名；本机编译成功但运行 507057 | [CANN 竞赛仓报告](../../competition/cann-ops-competitions/README.md) |
| `GITCODE-CONCAT` | GitCode `cann/cann-ops-competitions` / BatchLions | S9 多输入 Concat | `.../Concat/BatchLions/op_kernel/concat.cpp:909-1445` | 910B / CANN 8.5.0 | 作者最终第十名、五档 348.33 µs；本机离线 215,242+32,681+8,247 项通过，未上 NPU | 同上 |
| `GITCODE-LINALG-LSTSQ` | GitCode `cann/elec-ops-inspection` / XDU_XDU-BTD | 批量最小二乘 | `.../lstsq/op_host/linalg_lstsq.cpp:24-91` | 910B / CANN 未固定 | 作者无性能数字；本机 16/50/64 单 batch PASS，多 batch/矩形/200/256 失败 | [电力仓报告](../../competition/elec-ops-inspection/README.md) |

### 5.1 本机 910B3 复测摘要

环境：Ascend 910B3（NPU 2）、CANN 9.1.0、torch 2.10.0+cpu、torch_npu 2.10.0、Triton-Ascend 3.2.0；所有 NPU 运行经 `bin/npu-lock`。

| 实现 | 结果 | 本机数字 | 作者数字 | 判断 |
|---|---|---:|---:|---|
| GroupedTopk | PASS | 0.593500→0.222255 ms，2.670× | 0.701666→0.283144 ms，2.478× | 方向一致 |
| Sinkhorn | PASS | 3.616230→0.273420 ms，13.226×；max diff 6.765e-06 | 3.032301→0.329475 ms，9.203×；max diff 1.19e-07 | 方向一致 |
| FusedMoE | 编译失败 | UB 2359296 bits > 1572864 | 6.942960→0.606584 ms，11.446× | 当前环境不可移植 |
| 52Xayah Softmax | PASS（负例） | 0.334385 vs 0.096430 ms，0.288× | 0.140 vs 0.035 ms，0.25× | 负例稳定 |
| 52Xayah Matmul | PASS（负例） | 0.902870 vs 0.218790 ms，0.242× | 0.699 vs 0.163 ms，0.23× | 负例稳定 |
| Be4ver PACO 32×32×512 | PASS | 6.174 ms；512/512 | 7.061 ms | 同量级；当前通用路径 |
| Be4ver QDWH 128×128 | PASS | 4.288 ms | 4.392 ms | 基本一致 |
| Be4ver QDWH 512×512 | PASS | 20.698 ms | 20.540 ms | 基本一致 |
| Be4ver QDWH 1024×1024 | PASS | 48.475 ms | 46.839 ms | 慢 3.5% |
| GitCode Lstsq 16×16, K=1, b=1 | PASS | 0.0265 ms；max abs 1.19e-07 | 作者无时延 | 小尺寸单 batch 可运行 |
| GitCode Lstsq 50×50, K=2, b=1 | PASS | 0.0301 ms；max abs 3.58e-07 | 作者无时延 | 同上 |
| GitCode Lstsq 64×64, K=2, b=1 | PASS | 0.0395 ms；max abs 3.58e-07 | 作者无时延 | 同上 |
| GitCode Lstsq 多 batch/矩形/200/256 | FAIL | 507035/507057 | 作者无时延 | 不能外推 |
| GitCode recompute_w_u | 编译成功，运行失败 | 507057；5 个 shape 全失败 | 22/22、第 1 名 | 编译≠可运行 |
| GitCode Concat | 离线 PASS | 215,242+32,681+8,247 项 | 第十名、348.33 µs | 未上 NPU |

完整命令、原始输出和兼容性 workaround 见 [本机 910B3 复测记录](../validation/LOCAL_910B_VALIDATION_20261116.md) 与 [GitCode 补充调研](GITCODE_SOURCE_SURVEY_20261116.md)。

## 6. 最值得后续验证的 11 项机制

> 以下“预期改变”是机制层面的假设，不是加速比承诺。每项都给出可证伪的验证方法。

### M1 先按离散权重索引聚合，再乘权重（SplineConv rank1）

- **源码定位**：`SplineConv/rank1_submission_1174398/code/op_kernel/spline_conv_compact.h:200-225` 按 `wi` 把 `x_src * basis` 累加到 `agg[wi*C]`，`:86-94` 再对 4 个权重矩阵做一次 `4C × 16` 的向量乘。
- **适用条件**：权重索引只由离散样条 cell 决定、basis 对边线性；目标节点有多个入边；`C` 和 4 个 slot 能常驻 UB；本题聚合专用路径主要是 `ci=32/128, co=16, dim=2, degree=1, kernel=2`，compact 文件另有 `ci=8` 的 single 特化。
- **预期改变的关键路径**：每个目标节点、入度 `d`，通用逐边投影约 `4*C*16*d` MAC；重排后约 `4*C*d + 4*C*16` MAC。`d=4` 时约 3.2× 少的投影工作；权重装载从 `4d` 次降为 4 次。`d=1` 时反而略增开销。
- **代价**：需要 `4C` 聚合缓冲、目标节点链表/桶、额外清零和一次矩阵向量乘；浮点求和顺序改变，可能影响容差。
- **可证伪验证**：在同一图数据上按入度分桶（1、2、4、8、16），分别测逐边投影和先聚合路径；若 `d≤1.5` 没有收益或数值超差，机制不成立；若 `d≥4` 仍无收益，瓶颈不在投影 MAC，而在图遍历/搬运。

### M2 选择性高精度修复，而不是全量高精度（Fire rank2）

- **源码定位**：`FireCandidateDecodeFusion/rank2_submission_1176342/code/op_kernel/device_fire.h:105-143` 只在 `|2s-0.5+gx| < 1/16` 且 `stride` 超过阈值时，用软件 53 位 exp + 正确舍入倒数重算；调用点在 `:320-327`。
- **适用条件**：主路径是 FP32 向量化 sigmoid/解码；只有少数 lane 发生中心坐标的灾难性抵消；能先做一次廉价的最大值/幅度筛选；题目允许逐 lane 分支。
- **预期改变的关键路径**：大多数 lane 保持 `exp + 1/(1+·)` 的向量路径；敏感 lane 才付出标量 GM 读、软件 exp、倒数。它减少的是“为了极少敏感 lane 把整张量升到高精度”的工作，不是减少 sigmoid 本身。
- **代价**：一次 reduction + 标量循环；如果敏感 lane 比例高，标量路径会反噬；软件 exp 的代码体积和分支也会增加。
- **可证伪验证**：构造 `gx` 使 `2s-0.5+gx` 落在 0 附近，统计触发比例；分别测“无修复”“全量高精度”“选择性修复”的正确性和时延。若触发比例 >5–10% 或选择性修复仍超差，机制不划算/不充分。

### M3 分块扫描 + 消费者原生布局 + 跨核流水（Mamba-2 ChunkMix）

- **源码定位**：`mamba_ascendc_ops/mamba2_ssd_chunk_mix/design.md:20-57` 给出 `CB=C@B^T`、`W=tril(CB*exp(dA_i-dA_j))`、`y_diag=W@xdt`、`chunk_states=B^T@(xdt*exp(...))`；`op_kernel/mamba2_ssd_chunk_mix.cpp:169-210` 做 decay/mask，`:1016-1211` 做 AIC/AIV 双槽流水；`mamba2_dynamic_matmul.h:6-26,211-254` 复用一套 L1/L0/L0C 处理三种 GEMM。
- **适用条件**：序列足够长，分块开销能被摊薄；`P=N=chunk` 为 64/128；`H%G==0`；Cube 和 Vector 能并行；UB 能容纳所选 chunk 的输入/权重/中间。
- **预期改变的关键路径**：把长度 `L` 的串行扫描变成 `L/chunk` 个 chunk 的状态传递 + 每个 chunk 内的三角矩阵乘；把 335 个 PyTorch kernel / 20 个 Triton kernel 压到 3 个设备 kernel；`[N,P]` 状态布局让 `B^T@weightedX` 直接写消费者布局，省掉转置。
- **代价**：chunk 内多出 `O(L*chunk)` 的三角工作；12.06 GiB 的跨 kernel 中间物化（H256）；StateEpilogue 仍然 memory-bound；T128 的 UB 已到 182.8 KiB，不能双缓冲。
- **可证伪验证**：固定 `B,H,P,N`，扫 `L=512..16384` 与 `chunk=32/64/128`，记录 kernel 数、跨 kernel 字节数、StateEpilogue 占比；若短序列或小 `H` 没有收益，或 chunk 减半后中间物化反而下降但总时延不变，说明瓶颈不在扫描依赖。

### M4 去中间张量的生产者消费者融合（Linear+SwiGLU / Sinkhorn）

- **源码定位**：`vllm_hust_operator_optimizations/optimizations/linear_swiglu_kernel.py:35-66` 同时保留 gate/up 两个 FP32 accumulator，循环内只加载一次 A，最后 `gate*sigmoid(gate)*up` 并只写 `[M,N]`；`linear_swiglu_graph.py:34-50` 在 `M<=128` 走融合，否则回退。Sinkhorn 的 `task08_hc_split_sinkhorn/solution.py:67-89` 把 20 轮行列归一化留在寄存器。
- **适用条件**：被消除的中间张量确实占主导；激活是逐元素；两个 accumulator 能同时放进 CBUF/寄存器；M 小到能覆盖固定开销；不需要量化/ACLNN 路径。
- **预期改变的关键路径**：Linear+SwiGLU 省掉 `[M,2N]` 的写回+重读和一次 `npu_swiglu` 启动，GEMM FLOPs 不变；Sinkhorn 省掉 20 轮 × 多个小 kernel 和中间 tensor。
- **代价**：两个 FP32 accumulator 让 CBUF 压力翻倍；M 大时直接 overflow 或算错（报告明确说 M>128 回退）；首次 JIT 冷启动可能为负；Sinkhorn 的标量化只适合 4×4，矩阵变大就退化为大量标量。
- **可证伪验证**：对 Linear+SwiGLU 扫 `M=1/32/128/256/512`，同时记录 CBUF、kernel 数、算子时延和端到端 TPOT；若 M=128 的算子收益 <5% 或 e2e 收益 <1%，说明中间张量不是主瓶颈。对 Sinkhorn 扫 `hc_mult=4/8/16`，若标量化版本在 8×8 上仍快，才说明可扩展。

### M5 稠密全专家 vs 动态分发（FusedMoE）

- **源码定位**：`task02_fused_moe/solution.py:8-42` 对所有 8 个专家算 gate/up 和 down，`:46-63` 只在最后按 top-2 路由；参考实现 `reference.py:49-75` 用 Python 循环、布尔 mask 和每个专家的小 GEMM，只算被选中的 2 个专家。
- **适用条件**：专家数 `E` 很小（8）、token 数 `T` 很小（83）、`H/I` 很小（128/64）；启动、布尔分发、权重转换的固定开销远大于多算 4 倍的专家 FLOPs；权重能按 tile 转换而不物化完整副本。
- **预期改变的关键路径**：把“动态分发 + 8 次小 GEMM + 完整权重 cast”换成 3 个规则 kernel；计算量约增加到 4 倍（8/2 专家），但去掉了 Python 循环、mask 和大量 launch。
- **代价**：`O(E*T*H*I)` 的稠密计算和 `[E,T,I]` 中间缓冲；E 或 T 变大时，浪费的专家计算会吞掉所有收益。
- **可证伪验证**：固定 `H/I`，把 `E` 从 8 扫到 64/128、`T` 从 83 扫到 1024/8192；分别记录 kernel-only 和端到端。若 E=64 时加速比消失，说明“小 E”是必要条件；若 kernel-only 仍快但端到端不快，说明收益主要来自 launch 消除而非 kernel 本身。

### 6.6 第二批新增候选机制

第一批的 5 项仍是主清单；第二批补 3 组与前面不重叠的机制。

#### M6 矩阵内配对重排：把“每对一次标量反馈”改成“每轮 32 对一次向量更新”

- **源码定位**：`paco-jacobi-svd/docs/64x64-optimization-study-2026-11-12.md:62-100` 的 tournament 原型；补丁在 `docs/research-64x64/tournament.patch`，未合入主路径。
- **适用条件**：小矩阵（64×64）、batch 不大、单矩阵能常驻 UB；行数能被 2 整除；能接受旋转顺序改变导致 sweep 数/浮点结果变化。
- **预期改变的关键路径**：每个 sweep 从 2016 个串行对、每对 4 条行更新（8064 条）降到 63 轮 × 4 条（252 条），角度链按 32 lane 批量；算术复杂度仍是 O(n³/sweep)。
- **代价**：需要第二个 UB 矩阵做 ping-pong，5 段连续 `Adds` 重排；固定 64 lane 角度链在小 batch 更慢；条件数高时需要更多 sweep。
- **可证伪验证**：把补丁合入独立分支，按 `cond=1e2/1e4/1e6`、batch=1/8/40 跑完整校验和 event 计时；若 batch=1 没有 ≥2× 或 cond=1e6 小奇异值相对误差明显变差，机制不成立。

#### M7 单 kernel 多阶段融合：中间矩阵只被下一阶段消费时不落 GM

- **源码定位**：`qdwh_sdc_fused_jacobi/fused_qdwh_sdc.hpp:89-188,196-477`、`fused_main.cpp:280-354`；H 直接写 SDC arena slot0，Phase1/Phase2 用 `PhaseBarrier` 衔接。
- **适用条件**：多阶段分解（QDWH→SDC→叶块 Jacobi）的中间矩阵 H 只被下一阶段读；UB/arena 能容纳；能实现跨核 flag/phase barrier；不需要 host 中途检查。
- **预期改变的关键路径**：省掉两次 host main、两个 kernel、H/U 落盘和 H2D/D2H；作者 64×64 4.102 vs 6.278 ms（−34.7%），1024² 46.839 ms；本机 128/512/1024 全 PASS。
- **代价**：一个巨大 MIX kernel，flag 协议复杂；UB 余量只有 768 B；某阶段失败会拖垮整包；调试/回归成本高。
- **可证伪验证**：对同一分解分别做“两次 launch + H 落盘”和“单 launch 融合”，记录 kernel 数、H2D/D2H 字节、event 时延、UB 占用和失败模式；若中间传输占比 <5% 或融合后编译失败，融合不划算。

#### M8 有界 stencil/容量调度：按规模/通道/内存选 schedule，而不是单 kernel 通吃

- **源码定位**：`ictc_o3/ascendc/op_kernel/electrostatics.cpp:22-119`（每 atom×channel tile，stencil 权重/导数在寄存器/UB）；`ictc_o3/backends/ascend/electrostatics.py:22-85`（native/vectorized/particle/CANN 选路）；`ictc_o3/backends/ascend/capacity_message.py:65-80`（按内存自动 chunk）。
- **适用条件**：mesh/图算子有 3D stencil 或大边中间；channel 可 tile；能按 N/通道/adjoint 目标/内存选不同 schedule；能接受 atomic 非确定性或 chunk 重算。
- **预期改变的关键路径**：避免 `[N,stencil,S]` 和全边中间物化；作者 scalar q 额外推理 37.78→18.43 ms（−51.2%），196,608 atoms capacity 11.30→4.65 s（2.43×），显存 52.68→43.55 GiB。
- **代价**：多套 schedule 的维护/验证成本；atomic scatter 非确定；chunk 重算增加 FLOPs；小 N/训练/高阶要走回退。
- **可证伪验证**：固定算子和输入，扫 `N`、`channels`、`stencil order`、`adjoint target`，分别记录每套 schedule 的时延/显存/数值；若 crossover 与作者相反，或回退路径不比单 schedule 快，说明选路规则不成立。

### 6.7 GitCode 补充候选机制

#### M9 head/chunk 合并 + GQA K 复用 + 多槽流水

- **源码定位**：GitCode `cann/cann-ops-competitions` 的 `recompute_wu.h:591-684,686-792,1025-1395`；host 选路 `recompute_wu_tiling.cpp:204-390`。
- **适用条件**：GQA/MQA、多个 query head 共享 K；`BT=64`、`D=128/256`；能接受复杂 workspace/flag。
- **预期改变的关键路径**：K 只打包一次、最多 4 head×4 chunk 合并、Vector/Cube 多槽流水；减少重复 K 读、任务启动和跨核等待。
- **代价**：10 个 tiling key、workspace 增长、CANN 版本敏感；本机 CANN 9.1 运行 507057。
- **可证伪验证**：固定 T/H/Hg/D，扫 `headsPerTask=1/2/4`、`chunksPerTask=1/2/4`、`pipelineDepth=2/5`；记录 K 打包次数、workspace、event 时延；若 head/chunk 合并没有减少 K 读或反而更慢，机制不成立。

#### M10 布局优先的 Concat：先归一 rows×n_i，再选 Gather/直写/分带

- **源码定位**：GitCode `Concat/BatchLions/op_host/concat.cpp:55-263,267-330,440-532`；`op_kernel/concat.cpp:909-1071,1075-1445`；`s9_concat_layout.h:96-117,163-241,355-435,603-696`。
- **适用条件**：动态输入列表、最多 256 路、行宽/路数变化大；能接受 UB 预算和 Host/Kernel 共享布局。
- **预期改变的关键路径**：把每路每行的小 DMA 改成带内连续读 + Gather 拼整行 + 一次写；列带增大读 burst；窄行整块拷贝。
- **代价**：写碎片、偏移表、UB 暂存；作者的均衡分带/裁剪/整块取头分别慢 1.35%/1–8%/8%；本机只做离线测试。
- **可证伪验证**：固定 dtype/路数，扫 `bands=1/2/4/8`、`gBuf`、`S9_TRIM=0/1/2`、`S9_PRE=0/1`，分宽行/窄行记录读/写字节和 event 时延；若宽行或大 nOut 没有收益，选路规则不成立。

#### M11 shape 专用 solver + 补偿算术 + 迭代精化

- **源码定位**：GitCode `cann/elec-ops-inspection` 的 `linalg_lstsq.cpp:25-90,223-335`；`lstsq_compensated_tile_product.h:19-40`；`lstsq_shared_refinement.h:19-112`；`lstsq_shared_residual.h:17-330`。
- **适用条件**：FP32、列满秩、`M>=N`、`K<=64`；小尺寸/方阵/批量 RHS；能接受每 key 回归。
- **预期改变的关键路径**：tiny/packed/QR/LU 专用路径减少通用 solver 固定开销；TwoSum/TwoProduct + 残差补偿降低抵消误差；迭代精化用原方程修正。
- **代价**：代码体量大、每 key 独立 workspace；本机多 batch/矩形/200/256 崩溃；作者无性能数字。
- **可证伪验证**：固定 M/N/K，扫 16/50/64/200/256、single/batch、方阵/矩形；记录每 key 的编译、运行、误差、workspace；若补偿算术没有改善误差或明显变慢，机制不成立。

## 7. 证据局限与未复现项

1. **本机验证覆盖两批，但不是全部**。GroupedTopk、Sinkhorn、Be4ver PACO/QDWH 通过；FusedMoE 编译失败；52Xayah 两题负例复现。Hansen Ascend C、Mamba-2 自定义 OPP、Parity 原生库、vLLM-HUST/vLLM 未在本机编译/运行，仍为 `N`。
2. **本机不是作者环境**。本机是 910B3/CANN 9.1.0/TA 3.2.0，作者分别是 910B1/CANN 9.0.0/TA 3.2.1（KernelSwift）、910B3/CANN 8.2.RC1/TA 3.2.0（Mamba）、910B2/CANN 9.0.0（vLLM）、910B3/CANN 未固定（Be4ver）、910B3/CANN 9.0.0（Parity）。只能比较机制方向，不能把绝对时延或加速比横比。
3. **Hansen archive 没有逐 case 时延**。只有公示分数和名次，且分数不是跨题可比的加速比；不能把 91.85 分写成“比 83.26 快 10%”。
4. **Mantuo 的作者数字与本地数字不同**。GroupedTopk/Sinkhorn 方向一致，但本机 reference/candidate 都更快，且 Sinkhorn 误差更大；不能把 2.478×/9.203× 当作本机复现。
5. **FusedMoE 的 11.446× 未复现**。本机 TA 3.2.0/CANN 9.1.0 在原始 BLOCK_M=32、num_warps=1/4、BLOCK_M=16 和 num_stages=1 副本上都 UB 溢出；这是编译兼容性失败，不是性能结论。
6. **So-cean 的数字是作者环境**。910B3、CANN 8.2.RC1、TA 3.2.0；跨设备 A100 对照是同 shape、同 feature，但不同架构，不能把 70.82% 当作“910B3 与 A100 的普遍比例”。950PR 当前 HEAD 还有 grouped fast-path 回归。
7. **vLLM-HUST 的环境记录不完全一致**。原始报告写 CANN 9.0.0，兼容清单写 9.1.0-beta.3；证据快照的 source commit 是 `33d7813...`，当前仓库 HEAD 是 `58acc93...`。本轮只记录这两个事实，不把当前 HEAD 当成已复测。
8. **52Xayah 的计时口径不同**。作者用 `time.time()` 包住 20/50 次调用后统一同步；本机逐 sample 同步的绝对值更高（含同步/launch），但负例方向一致。不能把两种协议的绝对时延直接比较。
9. **Be4ver 的作者环境未固定 CANN**。QDWH 文档记录过 8.2.RC1，本机用 9.1.0；PACO 作者 7.061 ms、本机 6.174 ms，QDWH 1024 本机慢 3.5%。同 SoC 不同工具链，不能称为严格复现。
10. **Be4ver 的 4.26× 是研究补丁**。64×64 tournament 只在 `docs/research-64x64/tournament.patch`，当前主路径没有它；本机验证的是通用路径。
11. **Be4ver 的 atomic 非确定性是环境相关风险**。作者记录 1000×800、seed=42 的奇异值误差 1.5e-5~3.1e-4 且逐次不同；本机用同一输入连跑 3 次，`sigma.bin` SHA256 三次相同、全 PASS，说明该输入在当前 CANN 9.1.0/调度下没有触发。不能据此否定作者记录，也不能说已经修复；复用前要在目标环境重复运行。
12. **Parity 没有逐点原始日志**。仓库有性能协议、图和说明，但没有逐点 benchmark JSON/CSV；3.90×/4.76×/1.90× 只能作为作者报告，不能当精确复现。
13. **许可证不完整**。Hansen、mantuo、52Xayah、Be4ver 没有声明 LICENSE；本库不复制其源码，只做引用、行号定位和分析。公开可读不等于可再分发。
14. **未验证规则合规**。即使源码来自获奖提交，也不能证明当前比赛规则允许同样的融合、近似、私有 API 或数据布局。
15. **GitCode recompute_w_u 编译成功但运行失败**。CANN 9.1.0 上 5 个 shape 均报 507057；作者环境是 CANN 8.5.0。不能把编译成功当可运行，也不能把作者 22/22 当本机复现。
16. **GitCode Lstsq 只验证了小尺寸单 batch**。16/50/64 方阵 PASS；多 batch、矩形、200/256 在 CANN 9.1 上崩溃（507035/507057）。不能外推到全 shape 或 910C。
17. **GitCode Concat 只做了离线测试**。215,242+32,681+8,247 项地址/布局/UB 测试通过，但没有 NPU 构建、精度或性能数据；作者 348.33 µs 是最终榜单截图，不是本机复现。

## 8. 后续建议

按“能直接服务未来 910B/910C 固定 Shape Case 优化”排序：

1. **Be4ver 的 64×64 tournament**：把 `tournament.patch` 合入独立分支，按 cond=1e2/1e4/1e6、batch=1/8/40 做完整回归；这是当前最强的“指令组织”候选，但还不是当前代码成绩。
2. **Be4ver 的 atomic 确定性化**：把 `QdwhPmat` 的 atomic 累加改成确定性树归约/补偿求和，或改 SDC 分裂路径；先用 1000×800、seed=42 复现“逐次不同”，再验证修复。
3. **Parity 的 packed TP / mesh 静电**：在本机构建 `libictc_ascendc.so`（需要匹配 CANN/torch_npu），先跑 CPU FP64 小案例和单算子，再跑 910B3；重点验证 grouped TP 的小 E/小 C crossover 和 mesh 的 N/channel 选路。
4. **Gxj230958/TritonAscendBench**：选 1 个 L2/L3 kernel 做“上游 Triton 语义 → Triton-Ascend”迁移对照，重点不是速度，而是正确性/边界/来源冻结方法。
5. **U-Illll/ascend-eigh-operator**：单算子 eigh，适合与 Be4ver 的 SVD 一起看“双精度/双 limb、tridiagonal reduction、back-transform”在 910B 上的资源取舍。
6. **mantuoluozk 其余 7 题**：优先 Task04（watchdog 拆分）、Task10（UB 超限拆分）、Task03/06（弱收益），把“融合不一定快”的边界补全。
7. **GitCode recompute_w_u**：在 CANN 8.5.0 复跑作者 shape，确认 22/22；再定位 CANN 9.1 的 507057 是 ABI、同步、UB 还是本驱动布局问题。
8. **GitCode Concat**：补动态张量列表 ACLNN 驱动和官方 case，先做 NPU 编译/精度，再复算 348.33 µs 或给出本机时延。
9. **GitCode Lstsq**：确认矩形 `residuals` 布局和 workspace，再逐 key 验证多 batch/200/256；不要只用 16×16 的结果外推。

## 9. 文件入口

- 来源登记：[`index/catalog.json`](../../index/catalog.json)
- 本机复测：[`docs/research/LOCAL_910B_VALIDATION_20261116.md`](../validation/LOCAL_910B_VALIDATION_20261116.md)
- GitCode 补充：[`docs/research/GITCODE_SOURCE_SURVEY_20261116.md`](GITCODE_SOURCE_SURVEY_20261116.md)
- 机器摘要：[`docs/validation/local-910b-validation-20261116.json`](../validation/local-910b-validation-20261116.json)
- 仓库报告：[`index/README.md`](../../index/README.md)
- 跨实现经验：[`docs/experiences/README.md`](../experiences/README.md)
- 重点仓库：
  - [Hansen 2026 电力行业算子归档](../../competition/hansen-2026-power/README.md)
  - [KernelSwift 2026 Ascend 910B](../../competition/kernelswift-2026/README.md)
  - [Mamba-2 AscendC](../../general/attention/mamba-ascendc/README.md)
  - [vLLM-HUST OPset](../../general/transformer/vllm-hust-opset/README.md)
  - [52Xayah 负例](../../general/elementwise/52xayah-triton-kernels/README.md)
  - [Be4ver SVD](../../general/linear-algebra/be4ver-svd/README.md)
  - [Parity ICTC-O3](../../general/equivariant/parity-ictc-o3/README.md)
  - [GitCode CANN 竞赛仓](../../competition/cann-ops-competitions/README.md)
  - [GitCode 电力行业仓](../../competition/elec-ops-inspection/README.md)
