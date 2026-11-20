# Agent 阅读指南

本指南给后续 agent/开发者一条**最短学习路径**：先建立两层知识库的地图，再按目标选一条路线，最后回到条目源码和本机证据。不要从 423 个文件里随机翻。

## 1. 十分钟建立地图

按顺序读：

1. [README](../README.md)：两层结构、当前规模、本机验证摘要。
2. [docs/ARCHITECTURE.md](ARCHITECTURE.md)：条目契约、证据等级、910B3 验证协议。
3. [index/README.md](../index/README.md)：机器索引怎么读、`local_status` 含义。
4. [index/cards.md](../index/cards.md)：35 份实现的“一句话机制 + 先读文件 + 本机状态 + 边界”。
5. 选下面一条阅读路线。

如果只做一次检索，直接读 [index/cards.json](../index/cards.json) 或 [index/by-family.md](../index/by-family.md)；cards 是给 agent 的入口，不替代条目 README。

## 2. 阅读卡怎么用

每张 card 包含：

| 字段 | 用途 |
|---|---|
| `id` / `title` / `family` | 定位实现，不要只按算子名字猜语义 |
| `one_line` | 机制摘要；完整机制在条目 README 的“机制”节 |
| `local_status` | `pass` / `partial` / `fail` / `offline-only` / `not-run` / `platform-only` |
| `source_files` | 上游路径 + 行号；配合 `source.json` 的哈希/URL |
| `conditions` | 适用条件；不满足就不要套 |
| `negative_cases` | 已知退化/失败/版本敏感 |
| `evidence` | 本机日志路径；没有就是没跑 |

规则：card 只负责“让你决定要不要读”，**结论以条目 README + source.json + evidence 为准**。

## 3. 阅读路线

### 路线 A：GroupedMatmulGelu 竞赛机制（约 2–3 小时）

适合：要理解分组 GEMM、窄输出、专家调度、生产消费流水。

1. [competition/grouped-matmul-gelu/README.md](../competition/grouped-matmul-gelu/README.md)（总览/目录/证据边界）
2. [analysis/OPEN_SOURCE_REVIEW.md](../competition/grouped-matmul-gelu/analysis/OPEN_SOURCE_REVIEW.md)（领先 PR 的 4 类机制）
3. [analysis/RETROSPECTIVE.md](../competition/grouped-matmul-gelu/analysis/RETROSPECTIVE.md)（哪些收益没兑现、为什么）
4. [analysis/PLATFORM_EVIDENCE.md](../competition/grouped-matmul-gelu/analysis/PLATFORM_EVIDENCE.md)（SID/分数/口径）
5. [evidence/main-five-pmu-roofline.md](../competition/grouped-matmul-gelu/evidence/main-five-pmu-roofline.md)（PMU/roofline）
6. [docs/experiences/work_reduction_vs_fixed_overhead.md](experiences/work_reduction_vs_fixed_overhead.md)
7. [docs/experiences/layout_and_buffer_lifecycle.md](experiences/layout_and_buffer_lifecycle.md)

带着问题读：

- K=1 的有限值域表为什么只对“当前调用内实际值”有效？换输入会怎样？
- N=1 窄输出和普通 GEMM 尾部有什么不同？
- `group_list` 真实负载调度与平均切分的差异在哪里？
- 哪些收益是局部的，为什么没有兑现为整包耗时？

### 路线 B：线性代数（SVD / Lstsq / eigh）（约 3 小时）

适合：要做批量 SVD、最小二乘、特征分解，关心数值稳定性和 shape 专用化。

1. [general/linear-algebra/be4ver-svd/README.md](../general/linear-algebra/be4ver-svd/README.md)（PACO + QDWH/SDC）
2. [competition/elec-ops-inspection/README.md](../competition/elec-ops-inspection/README.md)（LinalgLstsq）
3. [general/linear-algebra/ascend-eigh/README.md](../general/linear-algebra/ascend-eigh/README.md)（候选，先看缺口）
4. [docs/experiences/iterative_linear_algebra_and_fusion.md](experiences/iterative_linear_algebra_and_fusion.md)
5. [docs/experiences/numerical_selective_precision.md](experiences/numerical_selective_precision.md)

带着问题读：

- 批量矩阵是交错存放还是逐矩阵？Jacobi 轮内怎么配对？
- QDWH+SDC 单 kernel 融合省掉了哪些 Host/Device 往返？atomic 非确定性怎么处理？
- Lstsq 的 TwoSum/TwoProduct、残差补偿、迭代精化分别解决什么误差？
- 哪些 shape 在本机通过，哪些失败？不要用 16×16 外推 200/256。

### 路线 C：布局与搬运算子（Concat / Transpose 类）（约 1.5 小时）

适合：多输入拼接、动态 tensor list、UB 预算、Gather/分带。

1. [competition/cann-ops-competitions/README.md](../competition/cann-ops-competitions/README.md) 的 `Concat` 部分
2. [docs/experiences/gitcode_operator_mechanisms.md](experiences/gitcode_operator_mechanisms.md) 的“布局优先搬运”
3. [docs/experiences/layout_and_buffer_lifecycle.md](experiences/layout_and_buffer_lifecycle.md)
4. [index/cards.md](../index/cards.md) 的 `GITCODE-CONCAT`

带着问题读：

- 为什么先把 shape 归一成 `rows × n_i`，而不是直接按输入循环？
- Gather 拼装/直写带/按输入切核分别在什么条件下更优？
- 作者哪些 A/B 是负结果（均衡分带、边界裁剪、整块取头）？
- 离线地址/布局测试通过，能证明什么、不能证明什么？

### 路线 D：注意力/SSM/Transformer（约 2 小时）

适合：Mamba-2、GDN 重算、Linear+SwiGLU、长序列/chunk。

1. [general/attention/mamba-ascendc/README.md](../general/attention/mamba-ascendc/README.md)
2. [competition/cann-ops-competitions/README.md](../competition/cann-ops-competitions/README.md) 的 `recompute_w_u` 部分
3. [general/transformer/vllm-hust-opset/README.md](../general/transformer/vllm-hust-opset/README.md)
4. [docs/experiences/shape_specialization_and_fallbacks.md](experiences/shape_specialization_and_fallbacks.md)
5. [docs/experiences/work_reduction_vs_fixed_overhead.md](experiences/work_reduction_vs_fixed_overhead.md)

带着问题读：

- chunk 扫描的消费者布局是什么？中间张量是否必须落 GM？
- GQA 的 K 复用减少的是读次数、打包次数还是算术量？
- 融合一定更快吗？作者的负例/回退条件是什么？
- recompute_w_u 在本机为什么编译成功但运行 507057？不要把它当可运行模板。

### 路线 E：数值稳健性与选择性精度（约 2 小时）

适合：FP32 消元/回代、SVD/Least Squares、需要高精度但不想全量双精度。

1. [docs/experiences/numerical_selective_precision.md](experiences/numerical_selective_precision.md)
2. [competition/elec-ops-inspection/README.md](../competition/elec-ops-inspection/README.md)
3. [general/linear-algebra/be4ver-svd/README.md](../general/linear-algebra/be4ver-svd/README.md)
4. [competition/hansen-2026-power/README.md](../competition/hansen-2026-power/README.md) 的 Fire 部分

带着问题读：

- 哪些值/ lane 需要高精度，哪些可以快？
- TwoSum/TwoProduct 的补偿项怎么进入归约，代价是多少？
- 迭代精化用原方程残差修正，什么时候会引入新误差？
- 主元/秩/有限性检查不满足时怎么回退？

### 路线 F：基准、迁移与负例（约 1 小时）

适合：要把上游 Triton kernel 迁到 Ascend，或需要正确性/性能基准。

1. [general/benchmarks/tritonascendbench/README.md](../general/benchmarks/tritonascendbench/README.md)
2. [general/benchmarks/triton-operator-benchmarks/README.md](../general/benchmarks/triton-operator-benchmarks/README.md)
3. [general/elementwise/52xayah-triton-kernels/README.md](../general/elementwise/52xayah-triton-kernels/README.md)
4. [skills/operator-pattern-lookup/SKILL.md](../skills/operator-pattern-lookup/SKILL.md)

带着问题读：

- 哪些是原创，哪些是 FlagGems/Liger/FLA 导入？
- 正确性通过和性能通过是两件事；作者是否明确“910B 性能未接通”？
- 手写 Triton 在什么情况下反而更慢？负例的稳定复现说明了什么？

### 路线 G：2026 制造行业算子专场（15 题）

适合：要参加/复现制造专场，或需要图计算、点云、复数/FFT 的行业算子。

1. [competition/mfg-2026/README.md](../competition/mfg-2026/README.md)（15 题总表、2/6/5/2 覆盖口径）
2. [official/specs.md](../competition/mfg-2026/official/specs.md)（官方 shape/dtype/属性/约束）
3. [problems.md](../competition/mfg-2026/problems.md)（逐题官方规格/源码/差异）
4. [MFG2026_ALGORITHM_REFERENCES_20261117.md](../docs/research/MFG2026_ALGORITHM_REFERENCES_20261117.md)（pyg-lib/torch_cluster/AnyGrasp 对 tie/空段/first-K 的核对）
5. [adapters.md](../competition/mfg-2026/adapters.md)（6 个 adapt 题的改造步骤与验收清单）
6. [reference/](../competition/mfg-2026/reference/README.md)（CPU 语义参考与 9 项测试；非官方 golden）
7. [driving-sdk](../competition/mfg-2026/driving-sdk/README.md)（FPS 最接近 direct；Radius/CylinderQuery 需适配官方定槽位/first-K）
8. [ops-fft](../competition/mfg-2026/ops-fft/README.md)（官方要求 2..8192；源码仅 32/64/128，32×32 本机 PASS 是子集）
9. [ops-gnn](../competition/mfg-2026/ops-gnn/README.md)（SegmentMinCsr direct；GatherCoo/RandomWalk/GridCluster 950-only）
10. [ops-sparse](../competition/mfg-2026/ops-sparse/README.md)（SpmmSum adapt；Spspmm 官方是 COO 固定槽位；Gather 不是 GatherCoo）
11. [ops-math](../competition/mfg-2026/ops-math/README.md)（ComplexAbs 950-only；BallQuery 官方单半径/严格 `<`；StackBallQuery 有 910B 配置）
12. [ops-solver](../competition/mfg-2026/ops-solver/README.md)（ComplexDiv 算法参考，950 SIMT）

补充来源放在对应条目的 `supporting/`，在 `source.json` 里标记为 `supporting_sources`：`ops-gnn/supporting/torch_scatter_npu/` 是 SegmentMinCsr 的 A2/A3 构建骨架，`driving-sdk/supporting/ops-gnn-fps-950/` 是 FP16/tie-break 对照。它们都**不是**官方题解候选，不能改变 `direct/adapt/reference/none` 分级。

补充：[制造专场 10 个小点](experiences/manufacturing_operator_edges.md)（比较符、空段/空行、首遇 tie、固定槽位、`rot^T`、归一化、int32/int64）。

带着问题读：

- 15 题里哪 2 题 direct、哪 6 题 adapt、哪 5 题 reference、哪 2 题 none？为什么“有 A2 路径”不等于“能直接提交”？
- 为什么 `ball_query` README 写 A2 √、代码却只有 arch35？`stack_ball_query` 能不能替代？
- `CylinderQuery` 的 A2 支持为什么没有文档单列？构建前要确认什么？
- `SpmmSum` 在 ops-gnn 与 ops-sparse 两套实现有什么不同？Gather 为什么不能替代 GatherCoo？
- FFT 的 32×32 PASS 能证明什么、不能证明什么？官方要求到 8192，源码只到 128，差距在哪里？

### 路线 H：给知识库新增来源（agent 维护者）

适合：要把一个新竞赛仓/通用算子纳入本库。

1. [skills/competition-repo-intake/SKILL.md](../skills/competition-repo-intake/SKILL.md)
2. [docs/ARCHITECTURE.md](ARCHITECTURE.md) 的条目契约/准入/证据等级
3. [index/entry-schema.json](../index/entry-schema.json)
4. [docs/MIGRATION_20261117.md](MIGRATION_20261117.md)（旧路径→新路径，避免链接漂移）
5. [skills/ascend-910b-local-validation/SKILL.md](../skills/ascend-910b-local-validation/SKILL.md)

## 4. 读条目时的固定顺序

1. **来源登记**：commit、许可、赛事/上下文、去重关系。
2. **语义与约束**：是不是同一个数学对象；dtype/layout/边界。
3. **源码定位**：路径/符号/行号；打开 `source.json` 核对哈希。
4. **机制**：减少什么工作/等待，增加什么代价。
5. **作者证据**：baseline、计时边界、warmup/repeat、硬件、原始位置。
6. **本机验证**：`V` 只代表 910B3；失败码也要看。
7. **适用条件与负例**：不满足就不要套。
8. **复用判断**：优先复用/需要验证/暂不使用。

## 5. 常见误读

- 把 `O+A`（榜单/作者）当 `V`（本机复现）。
- 把 910B3 的小 shape 结果外推到 910C 或全 shape。
- 看到“编译成功”就认为可运行（recompute_w_u 是反例）。
- 看到“融合/单 kernel”就认为一定更快（作者有负例）。
- 只看机制，不看负例/容量/版本敏感。
- 把导入/基准集合算成独立算法来源。

## 6. 输出学习结论时

建议用这个模板，方便下一个人/agent 接手：

```markdown
## 我读了什么
- 条目/章节：
- 源码 commit/路径：

## 机制
- 减少的工作/等待：
- 增加的成本：

## 证据
- 作者/平台：
- 本机 910B3：
- 未验证：

## 能否复用
- 适用条件：
- 负例/风险：
- 下一步验证：
```
