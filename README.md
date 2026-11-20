# 昇腾算子知识库（ACW 外部知识层）

> **Agent 入口**：[AGENT_START_HERE.md](AGENT_START_HERE.md) —— 一屏判断“我要找什么、去哪”。
>
> 三层库：[knowledge/](knowledge/README.md)（知识） · [playbook/](playbook/README.md)（经验/规则） · [toolkit/](toolkit/README.md)（工具）。
>
> **访问**：仓库已公开，匿名可读/可 clone；`push`、改设置、改可见性仍需 collaborator 权限或 token。边界见 [SECURITY.md](SECURITY.md)。

本库由三层组成：**知识库**（`competition/`、`general/`、`index/`）、**经验库**（`docs/experiences/`、`docs/validation/`、`docs/research/`）、**工具库**（`tools/`、`skills/`、CANNJudge 工具）。三层边界见 [LIBRARIES.md](LIBRARIES.md)。

知识库内部再分两层，供 ACW 做算子设计、复用判断和本机验证：

- **[competition/](competition/README.md)**：竞赛高性能 PR/实现，按赛事组织；
- **[general/](general/README.md)**：CANNBot 及其它非竞赛来源，按算子族组织。

每个准入条目都有：`README.md`（语义、机制、作者证据、适用条件、负例、复用判断）、`source.json`（固定 commit、源码路径/行号/哈希）、`evidence/`（本机 910B3 日志或明确的未验证说明）。

## 快速入口

| 入口 | 用途 |
|---|---|
| [AGENT_START_HERE.md](AGENT_START_HERE.md) | 给 agent 的一屏导航：按需求路由到三层库 |
| [knowledge/README.md](knowledge/README.md) | 知识库入口：公开赛题、开源仓、我们自己的实现 |
| [playbook/README.md](playbook/README.md) | 经验/规则入口：平台精度、输入构造、本机 vs 平台 |
| [toolkit/README.md](toolkit/README.md) | 工具入口：索引/校验/CANNJudge/提交/验证 |
| [competition/README.md](competition/README.md) | 竞赛库：GroupedMatmulGelu、CANN 竞赛仓、电力行业、Hansen、KernelSwift、制造专场、上合 2026、星辰杯 2026 |
| [playbook/topics/README.md](playbook/topics/README.md) | 通用经验专题：小 case、团队取包、隐藏 case、语义覆盖、Cube/Vector、对齐尾块 |
| [general/README.md](general/README.md) | 通用库：SVD/eigh、Mamba-2、vLLM-HUST、O(3)/mesh、Triton 基准、CANNBot |
| [index/README.md](index/README.md) | 统一索引、按算子族/按证据/按官方匹配级别检索 |
| [index/catalog.json](index/catalog.json) | ACW 直接读取的机器索引（24 仓库记录、39 份实现） |
| [index/cards.md](index/cards.md) | 39 张阅读卡：一句话机制、源码入口、本机状态、边界 |
| [index/by-source-match.md](index/by-source-match.md) | 制造专场按 direct/adapt/reference/none/building-block 检索 |
| [competition/mfg-2026/official/specs.md](competition/mfg-2026/official/specs.md) | 2026 制造专场 15 题官方 shape/dtype/属性/约束 |
| [index/by-cannjudge.md](index/by-cannjudge.md) | 15 题 × CANN Judge 交叉索引：题目 ID、测试点数、源码分级、本机状态 |
| [docs/CANNJUDGE_INTEGRATION.md](docs/CANNJUDGE_INTEGRATION.md) | 两层算子库 + CANNJudge 工具/题面/提交的联合视图 |
| [LIBRARIES.md](LIBRARIES.md) | 知识库/经验库/工具库三层边界与目录 |
| [docs/experiences/README.md](docs/experiences/README.md) | 跨实现经验：工作量/布局/数值/形状专用化 + 制造专场 10 个小点 |
| [docs/AGENT_READING_GUIDE.md](docs/AGENT_READING_GUIDE.md) | 给后续 agent 的十分钟地图 + 阅读路线 |
| [skills/README.md](skills/README.md) | 给 ACW/agent 的检索、入库、本机验证工作流 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 两层架构、条目契约、证据等级、验证协议 |
| [docs/MIGRATION_20261117.md](docs/MIGRATION_20261117.md) | 旧结构到新结构的迁移记录 |

## 当前规模

| 项 | 数量 |
|---|---:|
| 仓库记录 | 24（18 个独立公开算法来源 + 4 个我们的已结束比赛提交 + 2 个同仓不同赛事） |
| 深读仓库 | 15（含 1 个负例） |
| 实现条目 | 39 |
| 竞赛条目 | 10 个赛事/基线（含制造专场、上合 2026、星辰杯 2026）+ 1 个淘汰记录 |
| 通用条目 | 8 个算子/基准 + CANNBot 接入位 + 1 个淘汰记录 |
| 本机 910B3 有证据的实现 | 11 / 39：5 通过、2 负例通过、1 部分通过、2 失败、1 仅离线；制造专场 1/15 已跑（FourierTransform2D 32×32）；上合/星辰 4 个条目为 `platform-only` |

> 去重口径：GitCode `cann/cann-ops-competitions` 与既有 26 份 GroupedMatmulGelu 官方基线同仓不同赛事；`cann/elec-ops-inspection` 是 Hansen 归档的上游官方仓。两者都登记并深读，但不重复增加“独立算法来源”数。

## 2026-11-20 上合 / 星辰杯入库

- [上合 2026 初赛](competition/shanghe-2026-prelim/README.md)：BatchMatmulMaxSum，平台 24/661、64.44；Cube BMM + AIV max/sum，按 shape 分 tiny/direct/flat/trans/split-K 多条路径。
- [上合 2026 决赛](competition/shanghe-2026-final/README.md)：GroupedMatmulGelu，复用既有 [grouped-matmul-gelu](competition/grouped-matmul-gelu/README.md)，新增 [效率分析](competition/grouped-matmul-gelu/analysis/EFFICIENCY_20261120.md)。
- [星辰杯 2026 A 组](competition/starcup-2026-a-group/README.md)：Sparse Softmax（2/161、84.36）、DenseLightning（15/94、28.83）；mHcPre 团队 18/106、65.01，但按“宁缺毋滥”未入库，只保留 [excluded](competition/excluded.md) 记录。
- [星辰杯 2026 决赛](competition/starcup-2026-final/README.md)：SparseFlashAttentionGrad（6/28、73.77）。
- 通用经验新增 [playbook/topics/](playbook/topics/README.md)：小 case / launch-bound、团队取包、隐藏 case 泛化、语义覆盖、Cube/Vector、对齐尾块。

> 这 5 个条目是 `platform-only`：有 SID/排名/逐 case 时间和源码，但**没有本机 910B3 重跑**；不能写成“本机通过”。

## 2026-11-17 制造专场补充

- [2026 制造行业算子专场：15 题源码地图](competition/mfg-2026/README.md)：6 个官方/生产仓、15 份实现映射、2 个 910B 基础件、79 份选中源码快照；官方题面已落库到 [official/](competition/mfg-2026/official/README.md)；6 个 adapt 题改造步骤见 [adapters.md](competition/mfg-2026/adapters.md)；上游 tie/空段/first-K 核对见 [算法参考](docs/research/MFG2026_ALGORITHM_REFERENCES_20261117.md)；搜索/去重/缺口见 [制造专场源码调研](docs/research/MFG2026_SOURCE_SURVEY_20261117.md)；
- 按官方语义重分级：**2 题 direct**（SegmentMinCsr、FurthestPointSample）、**6 题 adapt**（FourierTransform2D、SpmmSum、Spspmm、Radius、BallQuery、CylinderQuery）、**5 题 reference**（ComplexAbs、ComplexDiv、GatherCoo、RandomWalk、GridCluster）、**2 题 none**（ComplexExp、FusedPropStep）；
- 8 题有 A2/910B 路径，但其中 6 题接口/布局/后处理需改；没有任何一题是“原样可直接提交”；
- ComplexExp/FusedPropStep 官方公式和 dtype 已补；仍无公开单 kernel 实现，但已加入 910B 基础件：ComplexExp 可用 exp/sin/cos + ComplexV3，FusedPropStep 可用 fft1_d/fft2_d（复数）或 rfft1_d/irfft1_d（FP32 实数）+ 逐点乘组合；
- 本机已跑 1/15：`FourierTransform2D`（`ops-fft/fft2_d`）在 910B3 通过 **32×32 C2C 子集**，best 0.1537 ms、max_abs 9.086799e-08；官方要求 2..8192 + inverse，尚未覆盖。其余 14 份 `local_status=not-run`，没有把文档支持或 950 实现写成 910B 通过。

## 本机验证摘要

设备：Ascend 910B3（物理 NPU 2，容器内 ACL logical 0）；CANN 9.1.0；gcc 11.4.0；cmake 3.22.1；所有设备运行经 `bin/npu-lock`。

| 实现 | 结果 | 关键数字 |
|---|---|---|
| KernelSwift GroupedTopk | PASS | 0.5935→0.2223 ms，2.670× |
| KernelSwift Sinkhorn | PASS | 3.6162→0.2734 ms，13.226× |
| KernelSwift FusedMoE | 编译失败 | UB 2359296 bits > 1572864 |
| Be4ver PACO 32×32×512 | PASS | 6.174 ms，512/512 |
| Be4ver QDWH 128/512/1024 | PASS | 4.288/20.698/48.475 ms |
| GitCode Lstsq 16/50/64 单 batch | 部分 PASS | best 0.0265/0.0301/0.0395 ms；多 batch/矩形/200/256 失败 |
| GitCode recompute_w_u | 编译成功、运行失败 | 5 个 shape 全部 `507057` |
| GitCode Concat | 仅离线测试 | 215,242+32,681+8,247 项通过 |
| 52Xayah Softmax/Matmul | 负例通过 | 0.288×/0.242× |
| 制造 SegmentMinCsr 补充骨架 | BUILD+RUN PASS（示例） | 8×2048、int32 indptr、4 段含空段、golden==output |
| 制造 CPU 参考 | PASS（CPU） | 9 项语义测试 |

> 补充骨架与 CPU 参考不计入 39 份实现的 910B3 通过数；SegmentMinCsr 骨架不是官方语义，主 `ops-gnn` 实现仍为 `not-run`。

完整命令、日志和失败码见 [docs/validation/LOCAL_910B_VALIDATION_20261116.md](docs/validation/LOCAL_910B_VALIDATION_20261116.md)。

## 证据规则

- `S+L`：固定源码路径/行号；`A`：作者说明；`O`：官方榜单/平台；`V`：本机 910B3；`N`：未验证。
- **`O+A` 不是 `V`**：作者/平台成绩不等于本机复现。
- **编译成功不等于可运行**：recompute_w_u 是明确反例。
- **小 shape 通过不等于全 shape 通过**：Lstsq 是明确反例。
- **910B3 结果不外推 910C**：没有 910C 数据就保持 `N`。
- 负例、失败码和未披露字段必须保留，不补造。

## 历史基线

`competition/grouped-matmul-gelu/` 保存 2026 CANN 挑战赛上合赛区决赛的 26 份公开 PR、248 个原始文件、官方题面/规则，以及我们的实际参赛源码：

- 源码：[competition/grouped-matmul-gelu/our-submission/kernel.asc](competition/grouped-matmul-gelu/our-submission/kernel.asc)
- SHA256：`161ac0ff2ab1920ca3de3e4f5d8b1b4d19a9a720c2781dcbc5c065f989b0b887`
- SID：`6af7d84c6bd0f9be3716f4c6`
- 2026-11-16 只读榜单观察：58.54 分、第 11、整包 Pass

这组数字是历史快照，不是当前通用算子性能。分析入口：[公开源码分析](competition/grouped-matmul-gelu/analysis/OPEN_SOURCE_REVIEW.md)、[比赛复盘](competition/grouped-matmul-gelu/analysis/RETROSPECTIVE.md)、[平台证据](competition/grouped-matmul-gelu/analysis/PLATFORM_EVIDENCE.md)。

## 维护

```bash
python3 tools/build_index.py      # 生成层索引、source.json、by-family/by-evidence
python3 tools/verify_archive.py   # 哈希、JSON、Python、链接、敏感模式
git diff --check
```

新增条目的步骤见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 和 [skills/competition-repo-intake/SKILL.md](skills/competition-repo-intake/SKILL.md)。本库只做研究和本地验证，不自动向竞赛平台提交，也不修改 ACW 框架或当前算子工程。
