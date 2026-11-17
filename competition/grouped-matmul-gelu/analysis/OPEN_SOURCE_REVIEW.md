# 公开最终源码技术分析

## 研究范围与证据

完整归档上合赛区 26 份公开最终提交 PR（248 个文件），固定每份 head commit，并逐文件验证 API 提供的 Git blob ID。详见 [总目录](../catalog/INDEX.md)。优先深读当时平台前三名的关键路径，同时对照我们的最终源码、PMU 和负实验。其余提交保留完整源码供检索，没有声称逐条审计了所有分支。

| 队伍 / PR | 11 月 16 日平台分数 | 主要阅读对象 |
|---|---:|---|
| TeamLiquid / [3062](https://gitcode.com/cann/cann-ops-competitions/pull/3062) | 77.21 | 实际分组 LPT、有限值域查表、活跃专家 B 预处理、多平面 L0C |
| Lizzop / [3027](https://gitcode.com/cann/cann-ops-competitions/pull/3027) | 74.88 | 同 kernel LUT、N=1 Cube 窄输出、容量与复用阈值 |
| WHAleX / [3064](https://gitcode.com/cann/cann-ops-competitions/pull/3064) | 68.95 | 底层 streaming engine、非二次幂 K 面板、容量约束 |

分数来自同次平台快照，源码来自相应队伍公开 PR。没有平台提供的公开 PR 源码哈希与代表 SID 的直接绑定，因而“该队源码中的机制导致了具体多少分”仍未经单项消融验证。第三方代码此次未编译、未上板；以下明确区分静态机制、代价推导与历史硬件证据。

## 1. K=1：把重复非线性计算变成当前调用内的有限值域生产

TeamLiquid 的 [`outer_table_gm`](../prs/pr-3062/sjtu_team-liquid/code/groupmatmulgelu/kernel.asc#L11887) 对 K=1 使用一个很有迁移价值的分解：

```
T[e, v, n] = FP16(GELU(FP32(INT32(v * W[e, 0, n])) * scale[e, n]))
Y[m, n]   = T[actual_expert(m), INT8_X[m, 0], n]
```

`v` 遍历完整 INT8 值域 -128…127。表由本次权重和 scale 在 Device 上生成；实际 group_list 决定非空专家和每行归属。生产完毕后同一 kernel 内全核同步，再读取本次 X 查表输出。不存在持久化到下次调用的结果缓存。

原实现每个输出做非线性，代价近似 O(MN) 次 GELU；查表路径约 O(256×活跃专家数×N) 次 GELU，加 O(MN) 输出搬运。平均每专家行数远大于 256 时，**减少的是昂贵工作量**，比单纯更换几条 GELU 指令有更大潜力。N、M/E、表容量、L2 复用、实际查表行读放大共同决定盈亏。

该实现按 16 个值、512 列生成表，尾列补齐到 32；输出阶段两个 UB 行槽以 MTE3→MTE2 credit 保护复用，仍按原始行序写出连续 ND。Host 使用 K、N、M/E 和表容量上限选择算法族，不读取 X/W/scale/group_list 数据。

Lizzop 的 [`Int8OuterLookup`](../prs/pr-3027/hfut_Lizzop/code/groupmatmulgelu/kernel.asc#L9970) 独立采用同类方法，`Build<3> → SyncAll → Gather<4>` 处于一次 MIX kernel，Host 明确用两次以上值域复用作为门槛。两份实现的分块、存储占用和阈值不同，值得做同环境消融。

**迁移条件：** 输入某个维度使原算子退化为有限域映射；当前调用的其它参数可共享；表生成成本可摊销。可能适用于量化激活、低基数索引和重复小参数非线性。**边界：** 任意 K 的值域会指数增长；表太大或 M 小会更慢。运行时表不是硬编码固定输出，但赛事对“预先计算”的解释需按具体规则确认，本次不作人工合规认定。

## 2. N=1：先问是否还应该用通用矩阵乘

TeamLiquid 的 [`vector_dot`](../prs/pr-3062/sjtu_team-liquid/code/groupmatmulgelu/kernel.asc#L11783) 把窄输出当作按行点积：实际 GL 查找专家，AIV 直接读取一行 X 和一列 W，完成整数归约后再做 FP32 scale、GELU、最终 half。

这里 INT8 转 half 用于精确表示原始整数输入，不是把反量化结果在 GELU 前降为 half。INT8 乘积可在 FP32 中精确表示；代码再转 INT32 做树形求和，避免长 FP32 累加改变整数点积。分析精度时必须看转换所处的计算阶段，不能只搜索 `half` 关键字。

Lizzop 则在足够大 M、长 K 且容量满足时优先采用窄输出 Cube 路径（Host 约 10078 行）：B 补到 16 列，M32 tile，A/B 在 L1 驻留。由此得到的设计问题不是“Vector 一定快”，而是：

- N 方向 padding 浪费与 Cube 吞吐的平衡；
- 每行重复读取 W 与按专家驻留的平衡；
- AIV 点积是否抢占 GELU/输出所需的执行资源；
- Cube 路径额外的布局、FIX、中间 C、跨核同步是否能摊销。

以后先做这两类算法的成本比较，再调 tile，不默认所有矩阵形状都走一套 GEMM。

## 3. 实际 group_list 的调度，比 M/E 平均数更重要

TeamLiquid [`aggregate_lpt::BuildRows`](../prs/pr-3062/sjtu_team-liquid/code/groupmatmulgelu/kernel.asc#L7064) 在 Device 中读取真实累计 GL，生成每专家完整行块和尾块。按 16 行对齐的工作量从大到小安排任务，使用小根堆把下一任务分配给当前最轻 owner，再生成每 owner 的连续描述符区间。

它把专家、原始行、有效行数编码成任务描述符。调度重排的是内部任务顺序，最终写回仍使用原始行坐标；不等于重排输出或假设均分。代价包含元数据生成和发布，适合长尾/不均衡足以覆盖调度成本的输入族，小输入不该盲目套用。

我们的 C1 v2 同样证明了分组分布是重要变量：[本地对照](../evidence/c1-v2-local-screen.md) 中，均匀 official2 14.955→15.005 µs 基本不变，random9181 29.550→19.590 µs。真实工作量最大 owner 的完整 K 轮数从 5 降为 2，PMU 等待链缩短。平台一次观察未胜过较早较优载体，故未把它晋升为平台最优。

可迁移方法是建立“实际专家段 → 工作描述符 → 每 owner 负载 → 原始输出归属”的闭环。Host 可依据容量和上界选方案；实际 GL 相关计算必须留在 Device。

## 4. 面向 Cube 的 B 表示与活跃地址寿命

TeamLiquid [`aggregate_lpt_prepare::PackB`](../prs/pr-3062/sjtu_team-liquid/code/groupmatmulgelu/kernel.asc#L6565) 先从真实 GL 提取活跃专家，为其 B 生成消费所需 NZ 表示。ND→NZ 搬到 L1，再按流水写入 GM；使用多个 staging 槽和最后读者 credit，避免等所有工作结束才复用存储。

它解决的是“后续多次消费是否值得支付一次布局生产”的问题。收益必须扣除预处理、尾部补零、同步和额外 GM/L2 事务；活跃专家少、复用高更有利。我们已有多种 prepared-B，因此不能把“打包 B”本身视为别人独有的答案，需要比较生产粒度、首包到达、持续供给和最后消费者何时释放空间。

以后对每个缓冲明确三件事：ready 由哪个生产阶段发出；哪个动作是最后一次读取；哪个 credit 允许下一代覆盖。不同缓冲不必在同一个全局迭代点退休。实际例子见 [R3 v4 生命周期](../evidence/r3-v4-lifetimes.json)。

## 5. 多平面 L0C：让长 K 和多 N 消费协同设计

TeamLiquid [`ultra_aggregate`](../prs/pr-3062/sjtu_team-liquid/code/groupmatmulgelu/kernel.asc#L10783) 根据 L0C 容量确定能保留几个输出平面。把完整 K 的主体和尾部组织为对多个 N 平面分阶段生产，保持 INT32 累加直到 true-K 全部完成，再 FIX 给 AIV。

这里值得学的是联动：A 在 L1 的寿命、多 N 面板复用、L0C 槽数、FIX 排空与 AIV 消费一起决定调度，而非只增加一层双缓冲。更多 L0C 槽会限制 tile 大小；更多在途 C 会扩大地址域和消费压力。应以全 Task 的关键链为目标，而不是仅追求 Cube 时间变小。

## 6. K 面板不必只选 256/512/1024

WHAleX [`NativeRewrite`](../prs/pr-3064/fdu_alex_zeyize_hhm/code/groupmatmulgelu/kernel.asc#L8457) 包含底层搬运、MMAD、FIX 和跨核信号控制。Host 长 K 路径约 11256 行明确测试：

```
BM=128, BN=256, panelK=640
2 × (BM+BN) × panelK = 491520 B <= L1
```

容量允许时用 640 深度双缓冲面板，否则回退 512。640 可能减少 K 阶段切换次数且维持双缓冲；实际收益仍需考虑对齐、尾段、L0 分块和搬运效率。其源码还有 576/704/768 等 tile 组合，提示应从资源不等式枚举可行形状，再测完整流水，而非只扫二次幂。

该文件同时保留高层 Matmul 和 native engine，部分路线含不同精度/转换策略。没有据此认定每个分支遵循相同数学顺序，未来复用前应独立确认。

## 与我们的差距：现在能下什么结论

公开 5 点同次快照显示我们的 C0/C1/C3/C4 比榜首慢约 9.05%/11.80%/16.48%/5.79%，C2 更快。隐藏 15 点无本次公开耗时，无法把 18.67 分差精确拆给算法。

静态源码揭示了我们应更早系统比较的**算法族覆盖**：K=1 有限域映射、N=1 两种引擎、实际工作量调度、面板容量枚举、多 N/L0C 生命周期联合设计。这些是可检验的差距假设。它们比“领先者只是省了几条指令”更符合工作量和数据流层面的解释，但尚不能代替同输入、同构建、同平台消融。

下一次优先建立这些算法族的通用最小实现与可复现实验，再决定何处需要精细流水。不要直接复制他人的条件分支去匹配已有 case，也不要把平台 Pass 当成对所有输入正确性或规则合规的证明。

