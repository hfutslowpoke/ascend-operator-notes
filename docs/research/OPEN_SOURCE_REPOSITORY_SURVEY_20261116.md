# 公开比赛与高性能算子仓库调研（2026-11-16，第一批）

状态：第一批来源筛选与源码深读完成。未在 NPU 上编译或运行任何第三方代码，未向比赛平台提交，未修改当前算子工程或 ACW 框架。本文中的“加速比”全部是原作者/原报告在各自环境下的数字，不是本机复现。

## 1. 结论摘要

- 本轮筛选 **12 个独立仓库/来源**，其中 **5 个**完成源码级深读，覆盖 **10 份实现**、至少 8 类算子族：检测框解码融合、图样条卷积、点到线段最近距离、MoE 路由与专家计算、分组 TopK、Sinkhorn 归一化、Mamba-2 分块扫描、线性层+SwiGLU 融合、Softmax/Matmul 负例。
- 与既有归档的 26 份 GroupedMatmulGelu 公开 PR **没有重复**。既有官方 `cann-ops-competitions` 快照仍作为基线引用，本轮不重复计数；GitCode 本轮 TLS 握手失败，未新增抓取。
- 最值得后续验证的 5 项机制见第 6 节：按权重索引先聚合、选择性高精度修复、分块扫描+消费者布局+跨核流水、去中间张量的生产者消费者融合、按规模在“稠密全专家”和“动态分发”之间选路。
- 最强的新证据来自：
  - `mantuoluozk/kernelswift-2026-triton-kernels`：Ascend 910B1、CANN 9.0.0、Triton-Ascend 3.2.1，官方评测器 `warmup=200`、`repeat=500`、中位数，10/10 正确性通过；本轮深读其中 3 题。
  - `So-cean/mamba-ascendc`：Ascend 910B3，device event p50，同平台 PyTorch/Triton-Ascend/AscendC 三层对照，并有 msprof 阶段分解；本轮深读 ChunkMix。
  - `vLLM-HUST/vllm-hust-opset`：Ascend 910B2，算子级 10 warmup/50 同步样本，端到端 3 轮匹配；本轮深读 Linear+SwiGLU Graph 融合。
- 证据最弱的深读项是 `Hansenwuwuwu/ascend-c-competition-archive` 的三份 2026 电力行业算子：有官方公示分数和名次，但没有逐 case 时延，也没有本机编译/运行；因此只把它当作“获奖源码的结构证据”，不把分数换算成加速比。
- `52Xayah/triton-ascend-kernels` 的 Softmax/Matmul 是明确负例：Triton 版本分别约 0.140 ms vs PyTorch 0.035 ms、0.699 ms vs 0.163 ms。它提醒我们，手写 Triton 不会自动超过厂商库，必须先把 tile、数据布局、融合和启动开销算清楚。

## 2. 范围、搜索与证据方法

### 2.1 搜索范围

- 日期：2026-11-16。
- 平台：GitHub 公开仓库为主；GitCode 官方比赛仓尝试过一次，`git clone` 返回 `gnutls_handshake() failed`，本轮未继续重试。
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
| `N` | 未复现：本环境没有编译、运行、profiling 或平台提交 |

一个结论可以同时有 `S+L`，但没有 `N` 之外的实验证据时，不写成“已验证性能”。本轮所有实现都是 `N`。

### 2.3 本地缓存

第三方源码缓存在工作区外：

```text
/workspace/acw/var/private/repository-research-cache/
```

缓存只用于阅读和行号定位，不进入 Git，不包含账号、令牌、构建产物或 profiler 原始目录。固定 commit 和关键文件 SHA256 见 [来源登记](../../catalog/external-repositories.json)。

## 3. 去重、筛选与淘汰

### 3.1 与既有归档的关系

既有私仓保存了 26 份 2026 CANN 挑战赛上合赛区 GroupedMatmulGelu 公开 PR、248 个原始文件，以及我们的最终源码、组件、证据和复盘。本轮：

- 不重复登记这 26 份 PR；
- 不把官方 `cann/cann-ops-competitions` 的同一批快照当成新仓库；
- 不把 fork、镜像、同一源码的二次搬运算作独立来源；
- 只在新仓库的算子族、机制或证据与既有归档不同时计入。

### 3.2 12 个来源的取舍

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
| 9 | `Be4ver/svd` | Ascend SVD，C++/Ascend C | 候选（下一批优先） | PACO Jacobi + QDWH/SDC 两份实现，有 910 设备日志和精度验证；根目录无 LICENSE、无 README，留待下一批 |
| 10 | `Parity-LRX/ICTC-O3-HP-Ascend` | 生产型 O(3)/MACE，Ascend C | 候选（下一批优先） | 910B3 上 3.90×/4.76× 等强证据，MIT；体量大，本轮未深读 |
| 11 | `U-Illll/ascend-eigh-operator` | 单算子 eigh，Ascend C | 候选 | 语义、tiling、双精度路径清楚，但只有单算子、没有性能数据 |
| 12 | `asfsdsf/S6` | 单算子 Fmax，Ascend C | 淘汰 | 只有 6 个源文件和 LICENSE，无 README、无测试、无性能证据；不足以支撑“高性能”判断 |

另外见过但未计入 12 个来源的：

- `WildPickledPepper/ascend-rowcol-int8-quant`：只有任务书、benchmark/golden 脚本和配置，未看到 kernel 源文件；暂不纳入。
- `m-rui001/ascend-c-operator-dojo`：59 轮学习/对照笔记和检查表，偏训练材料，不是独立的高性能算子实现；暂不纳入。
- `CosmosMind-ai/PyPTO-Ops`、`Luoshu-Intelligent-Computing/ascend-310p-kernels`：前者是新 DSL 的算子库，后者是 310P；与 910B/910C 目标差异大，留作后续专题。

### 3.3 官方基线

| 基线 | 固定版本 | 说明 |
|---|---|---|
| `cann/cann-ops-competitions` | `776f2e552a125d3c210b497385f8b57292ebdc92`（既有归档） | 26 份 GroupedMatmulGelu 公开 PR 已在本库 `third_party/shanghe_final`；本轮不重复抓取 |
| 本库 GroupedMatmulGelu 最终源码 | 见 `our_work/final_submission` 与 `catalog/our-work.json` | 历史成绩 58.54 分、第 11、整包 Pass；不是本轮新证据 |

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
| Be4ver svd | Ascend SVD | C++/Ascend C | Jacobi SVD、QDWH/SDC | `b02796511ea62861c647a56b880da15b6a07b3a0` | 未声明 | 设备日志 + 精度；无 README | 候选 |
| Parity ICTC-O3 | 生产型 O(3)/MACE | Ascend C | 等变张量积、长程静电、MD | `ca9ccc4ea9e2e9d21e48daeb45925f2bc4382be8` | MIT | 910B3 多组强数据 + 协议 | 候选 |
| U-Illll eigh | 单算子 | Ascend C | 对称特征分解 | `fa2275fa4b2cee005e6868ce23266686eafa7a1b` | MIT | 源码/契约；无性能 | 候选 |
| asfsdsf S6 | 单算子 | Ascend C | Fmax 广播 | `0454168359b4749058b61432bf15b3945b44efa9` | MIT | 源码极小；无文档/性能 | 淘汰 |

## 5. 深读实现索引

| ID | 来源 | 算子 | 关键源码（固定 commit） | 环境 | 原始证据 | 报告 |
|---|---|---|---|---|---|---|
| `HANSEN-2026POWER-FIRE-R2` | Hansen archive | FireCandidateDecodeFusion (rank2) | `.../FireCandidateDecodeFusion/rank2_submission_1176342/code/op_kernel/device_fire.h:105-143,272-365` | 910B4 / CANN 9.0.0.0 / 20 AIC（比赛环境） | 公示分 83.26；无逐 case 时延 | [Hansen 报告](../repositories/Hansenwuwuwu__ascend-c-competition-archive.md) |
| `HANSEN-2026POWER-SPLINE-R1` | Hansen archive | SplineConv (rank1) | `.../SplineConv/rank1_submission_1174398/code/op_kernel/spline_conv_compact.h:86-94,200-225` | 同上 | 公示分 89.16；无逐 case 时延 | 同上 |
| `HANSEN-2026POWER-TREE-R1` | Hansen archive | TreeLineMinDistance (rank1) | `.../TreeLineMinDistance/rank1_submission_1175341/code/op_kernel/tree_pairs.h:323-515,752-854` | 同上 | 公示分 91.85；无逐 case 时延 | 同上 |
| `KS2026-ASC910B-GROUPEDTOPK` | mantuo | GroupedTopk | `platforms/ascend910b/task01_grouped_topk/solution.py:8-89` | 910B1 / CANN 9.0.0 / TA 3.2.1 | 0.701666→0.283144 ms，2.478×，官方评测器 | [KernelSwift 报告](../repositories/mantuoluozk__kernelswift-2026-triton-kernels.md) |
| `KS2026-ASC910B-FUSEDMOE` | mantuo | FusedMoE | `.../task02_fused_moe/solution.py:8-87` | 同上 | 6.942960→0.606584 ms，11.446× | 同上 |
| `KS2026-ASC910B-SINKHORN` | mantuo | hc_split_sinkhorn | `.../task08_hc_split_sinkhorn/solution.py:67-89` | 同上 | 3.032301→0.329475 ms，9.203× | 同上 |
| `MAMBA2-ASCENDC-CHUNKMIX` | So-cean | Mamba-2 SSD ChunkMix | `mamba_ascendc_ops/mamba2_ssd_chunk_mix/op_kernel/mamba2_ssd_chunk_mix.cpp:1016-1211` | 910B3 / CANN 8.2.RC1 / TA 3.2.0 | 0.632 ms vs 6.287/4.305；H256 31.888 ms | [Mamba 报告](../repositories/So-cean__mamba-ascendc.md) |
| `VLLMHUST-LINEAR-SWIGLU` | vLLM-HUST | Linear+SwiGLU | `.../linear_swiglu_kernel.py:35-66` | 910B2 / CANN 9.0.0（报告）/ 9.1.0-beta.3（兼容清单） | 算子 −15.93~17.92%；e2e TPOT −3.82% | [OPset 报告](../repositories/vLLM-HUST__vllm-hust-opset.md) |
| `XAYAH-ASC910B-SOFTMAX-NEG` | 52Xayah | Softmax | `src/softmax.py:8-41` | A2 / CANN 9.2.0 / TA 3.2.2 | 0.140 ms vs PyTorch 0.035 ms | [52Xayah 报告](../repositories/52Xayah__triton-ascend-kernels.md) |
| `XAYAH-ASC910B-MATMUL-NEG` | 52Xayah | Matmul | `src/matmul.py:8-89` | 同上 | 0.699 ms vs PyTorch 0.163 ms | 同上 |

## 6. 最值得后续验证的 5 项机制

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

## 7. 证据局限与未复现项

1. **没有任何本机 NPU 实验**。本轮没有编译第三方工程、没有运行 profiler、没有执行比赛平台提交脚本。所有性能数字都是作者/原报告/官方公示记录。
2. **Hansen archive 没有逐 case 时延**。只有公示分数和名次，且分数不是跨题可比的加速比；不能把 91.85 分写成“比 83.26 快 10%”。
3. **Mantuo 的数字是作者环境**。910B1、CANN 9.0.0、Triton-Ascend 3.2.1，官方评测器 `warmup=200/repeat=500`、中位数、每次 sample 同步。没有本机重跑；不能外推到 910C 或不同 CANN。
4. **So-cean 的数字是作者环境**。910B3、CANN 8.2.RC1、TA 3.2.0；跨设备 A100 对照是同 shape、同 feature，但不同架构，不能把 70.82% 当作“910B3 与 A100 的普遍比例”。950PR 当前 HEAD 还有 grouped fast-path 回归。
5. **vLLM-HUST 的环境记录不完全一致**。原始报告写 CANN 9.0.0，兼容清单写 9.1.0-beta.3；证据快照的 source commit 是 `33d7813...`，当前仓库 HEAD 是 `58acc93...`。本轮只记录这两个事实，不把当前 HEAD 当成已复测。
6. **52Xayah 的计时很粗**。`time.time()` 包住 20/50 次调用后统一同步，没有逐 sample 同步、没有中位数、没有环境隔离；只能当负例方向，不能当精确数字。
7. **许可证不完整**。Hansen、mantuo、52Xayah、Be4ver 没有声明 LICENSE；本库不复制其源码，只做引用、行号定位和分析。公开可读不等于可再分发。
8. **未验证规则合规**。即使源码来自获奖提交，也不能证明当前比赛规则允许同样的融合、近似、私有 API 或数据布局。

## 8. 下一批建议

按“能直接服务未来 910B/910C 固定 Shape Case 优化”排序：

1. **Be4ver/svd**：线性代数族，PACO Jacobi 与 QDWH/SDC 两份实现，有 910 设备日志和精度验证；先补 LICENSE/来源核验，再深读 QDWH/SDC 的 QR 串行链与 UB 预算。
2. **Parity-LRX/ICTC-O3-HP-Ascend**：910B3 上强性能证据，MIT；重点看 O(3) 张量积的对称性剪枝、长程静电的 mesh/FFT 分块、HCCL 重叠。体量大，建议只选 1–2 个 kernel。
3. **Gxj230958/TritonAscendBench**：选 1 个 L2/L3 kernel 做“上游 Triton 语义 → Triton-Ascend”迁移对照，重点不是速度，而是正确性/边界/来源冻结方法。
4. **U-Illll/ascend-eigh-operator**：单算子 eigh，适合与 Be4ver 的 SVD 一起看“双精度/双 limb、tridiagonal reduction、back-transform”在 910B 上的资源取舍。
5. **mantuoluozk 其余 7 题**：优先 Task04（watchdog 拆分）、Task10（UB 超限拆分）、Task03/06（弱收益），把“融合不一定快”的边界补全。

## 9. 文件入口

- 来源登记：[`catalog/external-repositories.json`](../../catalog/external-repositories.json)
- 仓库报告：[`docs/repositories/README.md`](../repositories/README.md)
- 跨实现经验：[`docs/experiences/README.md`](../experiences/README.md)
- 三个重点仓库：
  - [Hansen 2026 电力行业算子归档](../repositories/Hansenwuwuwu__ascend-c-competition-archive.md)
  - [KernelSwift 2026 Ascend 910B](../repositories/mantuoluozk__kernelswift-2026-triton-kernels.md)
  - [Mamba-2 AscendC](../repositories/So-cean__mamba-ascendc.md)
  - [vLLM-HUST OPset](../repositories/vLLM-HUST__vllm-hust-opset.md)
  - [52Xayah 负例](../repositories/52Xayah__triton-ascend-kernels.md)
