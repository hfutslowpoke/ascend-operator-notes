# 知识库入口

知识库回答：**有什么、是什么、在哪、证据是什么。**

它包含三类内容：

1. **公开赛题**：官方题面、shape/dtype/属性/约束、测试点数；
2. **开源仓源码**：固定 commit、路径/行号、SHA256、本地快照；
3. **我们自己的实现**：kernel/host/驱动、构建与运行日志、平台 SID/榜单。

## 1. 按需求找

| 你要找 | 入口 |
|---|---|
| 制造专场 15 题 + 源码分级 + 本机状态 | [`../index/by-cannjudge.md`](../index/by-cannjudge.md) |
| 制造专场官方规格 | [`../competition/mfg-2026/official/specs.md`](../competition/mfg-2026/official/specs.md) |
| 制造专场逐题差异/缺口 | [`../competition/mfg-2026/problems.md`](../competition/mfg-2026/problems.md) |
| 按算子族 | [`../index/by-family.md`](../index/by-family.md) |
| 按本机证据 | [`../index/by-evidence.md`](../index/by-evidence.md) |
| 按官方源码匹配级别 | [`../index/by-source-match.md`](../index/by-source-match.md) |
| 机器索引（ACW 读取） | [`../index/catalog.json`](../index/catalog.json) |
| 阅读卡 | [`../index/cards.md`](../index/cards.md) |

## 2. 知识库子库

| 子库 | 内容 | 入口 |
|---|---|---|
| 竞赛知识 | GroupedMatmulGelu、CANN 竞赛仓、电力行业、Hansen、KernelSwift、制造专场 | [`../competition/README.md`](../competition/README.md) |
| 通用知识 | SVD/eigh、Mamba-2、vLLM-HUST、O(3)/mesh、Triton 基准、CANNBot | [`../general/README.md`](../general/README.md) |
| 官方题面 | 制造专场 15 题公开题面与 specs | [`../competition/mfg-2026/official/README.md`](../competition/mfg-2026/official/README.md) |
| 索引 | catalog/cards/by-family/by-evidence/by-source-match/by-cannjudge | [`../index/README.md`](../index/README.md) |

## 3. 我们自己的实现

| 实现 | 位置 | 状态 |
|---|---|---|
| GroupedMatmulGelu 参赛 kernel | [`../competition/grouped-matmul-gelu/our-submission/`](../competition/grouped-matmul-gelu/our-submission/) | 历史竞赛提交；SID/榜单快照见 `analysis/PLATFORM_EVIDENCE.md` |
| 制造专场 SegmentMinCsr 补充骨架 | [`../competition/mfg-2026/ops-gnn/supporting/torch_scatter_npu/`](../competition/mfg-2026/ops-gnn/supporting/torch_scatter_npu/) | 第三方 A2/A3 骨架；BUILD+RUN PASS 仅示例，非官方语义 |
| 制造专场主实现 | 暂无 | `competition/mfg-2026/*/source/` 是公开源码，不是我们的提交 |

> “我们自己的实现”与“公开源码”必须分开标注。前者看 `our-submission/`/`own/`，后者看 `source/` + `source.json`。

## 4. 每个条目怎么读

固定顺序：

1. `README.md` 的“来源登记”：commit、许可、赛事/上下文、去重关系；
2. “赛题语义/约束”：是不是同一个数学对象，dtype/layout/边界；
3. “源码定位”：路径/符号/行号；
4. `source.json`：`local_path`、SHA256、固定 commit；
5. `evidence/`：本机 910B3 日志；没有日志就是 `not-run`；
6. “负例/风险”：架构、版本、shape、dtype、语义差异；
7. “复用判断”：直接复用、适配、只读算法、不要用。

## 5. 知识和经验、工具的边界

| 内容 | 归属 |
|---|---|
| 官方题面、测试点数、源码、哈希、本机日志 | 知识库 |
| 跨题规律、平台规则、负例、crossover、输入构造方法 | 经验库 [`../playbook/`](../playbook/) |
| 索引生成、归档校验、CANNJudge 客户端/打包/提交 | 工具库 [`../toolkit/`](../toolkit/) |

索引（`index/`）是知识库的视图，不是独立库。

## 6. 新增已结束比赛

按 [`INTAKE.md`](INTAKE.md) 走。先固定题面、源码、自己的实现和平台证据，再写经验；不要反过来。
