# GroupedMatmulGelu 竞赛基线（2026 CANN 上合赛区决赛）

> 层级：`competition` · 状态：`deep-read`（历史归档）· 本机 910B3：`platform-only`

这是本库的起点：2026 CANN 挑战赛上合赛区决赛 GroupedMatmulGelu 的公开 PR 归档、我们的实际参赛源码、分析、PMU 证据和复盘。

## 1. 规模与身份

| 项 | 值 |
|---|---|
| 赛事 | 2026 CANN 挑战赛区域决赛，上合赛区 |
| 公开 PR | 26 份，248 个原始文件 |
| 固定上游 | `cann/cann-ops-competitions@776f2e552a125d3c210b497385f8b57292ebdc92` |
| 我们的最终源码 | `our-submission/kernel.asc` |
| SHA256 | `161ac0ff2ab1920ca3de3e4f5d8b1b4d19a9a720c2781dcbc5c065f989b0b887` |
| SID | `6af7d84c6bd0f9be3716f4c6` |
| 2026-11-16 榜单观察 | 58.54 分、第 11、整包 Pass |

这组数字是历史快照，不是当前通用算子性能，也不是 910C 结论。

## 2. 目录

| 路径 | 内容 |
|---|---|
| [prs/](prs/) | 26 份公开 PR 的固定 commit 快照，保留原始路径、LICENSE、PROVENANCE、SHA256/Git blob |
| [official/](official/) | 官方题面、提交规则、LICENSE、PROVENANCE |
| [our-submission/](our-submission/) | 我们的 `kernel.asc`、CMake、sample/bench/verify 驱动、本地 reference |
| [components/](components/) | 最终版组件和未晋升候选；候选成绩不能替代最终源码 |
| [analysis/OPEN_SOURCE_REVIEW.md](analysis/OPEN_SOURCE_REVIEW.md) | 领先 PR 的有限值域表、窄输出归约、真实分组调度、生产消费流水 |
| [analysis/RETROSPECTIVE.md](analysis/RETROSPECTIVE.md) | 成功方向、负实验、为什么局部收益没有兑现 |
| [analysis/PLATFORM_EVIDENCE.md](analysis/PLATFORM_EVIDENCE.md) | 源码/SID/成绩对应、公开点对照、缺失字段 |
| [analysis/RULES_AND_MATH.md](analysis/RULES_AND_MATH.md) | 数学顺序、单 kernel/Host-Device 边界、赛事约束 |
| [catalog/](catalog/) | 26 PR 的 `public-final-sources.json`、`submissions.csv`、`our-work.json`、人读索引 |
| [evidence/](evidence/) | PMU、roofline、负实验、平台排名快照、最终构建记录 |
| [source.json](source.json) | 本条目的机器来源清单 |

> `official/.../CONTRIBUTING.md` 保留原始快照，其中 CANNJudge 链接带双重括号；为不改变归档哈希，未修正该第三方文件。

## 3. 公开 PR 归档

- 每份 PR 固定到 head commit，逐文件保存 SHA256 和 Git blob ID；
- 采集时 PR 仍为 open；公开提交、平台 Pass、人工核验是三件不同的事；
- 归档阶段没有重新编译/运行第三方源码；本库不把任何 PR 的榜单成绩当成本机复现；
- 26 份 PR 是“竞赛基线”，不计入后续 35 份外部实现。

人读索引：[catalog/INDEX.md](catalog/INDEX.md)；机器索引：[catalog/public-final-sources.json](catalog/public-final-sources.json)。

## 4. 我们的提交

- `our-submission/kernel.asc` 与最终 SID 对应，SHA256 已固定；
- `main.asc`、`bench.asc`、`verify_batch.asc`、`CMakeLists.txt` 来自当时的 trial 工作区，源码逐字节一致；
- `reference.py` 是我们的本地参考实现，不是官方 golden；
- 未包含 NPU 二进制、输入张量、输出文件；归档任务没有重新编译/运行 NPU。

细节：[our-submission/README.md](our-submission/README.md)；来源/状态：[catalog/our-work.json](catalog/our-work.json)。

## 5. 可复用机制（摘要）

优先复用的方向：

1. **K=1 当前调用内有限值域查表**：只对实际出现的输入值建表，减少重复计算；
2. **N=1 点积/窄输出引擎**：避免把窄输出当成普通 GEMM 的尾部；
3. **真实 `group_list` 工作调度**：按实际分组负载分核，而不是平均切分；
4. **活跃专家预处理 + 多平面 L0C**：把生产者/消费者和缓冲生命周期一起设计；
5. **事件退休/缓冲复用**：在 UB/L1/L0C 之间明确最后读者，减少无效同步。

适用边界：

- 这些机制绑定 GroupedMatmulGelu 的语义、shape 族和 910B 存储层次；
- 不能直接外推到 910C、其它 dtype/layout 或通用 GEMM；
- 竞赛隐藏用例/评分时期变化会影响“分数”，不要把历史名次当通用性能。

## 6. 证据边界

| 项 | 级别 |
|---|---|
| 26 PR 源码/路径/哈希 | `S+L` |
| 官方题面/规则 | `S` |
| 我们的 SID/平台 Pass/榜单观察 | `O` |
| 公开源码机制分析 | `S+L+A` |
| 本机 910B3 重跑 | 只有历史 PMU/实验证据；本轮重构未重新编译/运行 |
| 910C | `N` |

完整验证记录见 [../../docs/validation/LOCAL_910B_VALIDATION_20261116.md](../../docs/validation/LOCAL_910B_VALIDATION_20261116.md) 与 [evidence/](evidence/)。

## 7. 复用判断

**作为竞赛机制与工程流程的参考库，不作为通用算子模板。** 新算子应先判断语义/布局/存储层次是否真的相似，再挑选机制；不要直接复制 `kernel.asc` 或把 58.54 分当成本项目标成绩。
