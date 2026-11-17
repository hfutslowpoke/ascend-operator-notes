# 昇腾算子知识库（ACW 外部知识层）

本库把公开竞赛实现、生产算子库和 CANNBot 知识整理成两层结构，供 ACW 做算子设计、复用判断和本机验证：

- **[competition/](competition/README.md)**：竞赛高性能 PR/实现，按赛事组织；
- **[general/](general/README.md)**：CANNBot 及其它非竞赛来源，按算子族组织。

每个准入条目都有：`README.md`（语义、机制、作者证据、适用条件、负例、复用判断）、`source.json`（固定 commit、源码路径/行号/哈希）、`evidence/`（本机 910B3 日志或明确的未验证说明）。

## 快速入口

| 入口 | 用途 |
|---|---|
| [competition/README.md](competition/README.md) | 竞赛库：GroupedMatmulGelu、CANN 竞赛仓、电力行业、Hansen、KernelSwift、制造专场 |
| [general/README.md](general/README.md) | 通用库：SVD/eigh、Mamba-2、vLLM-HUST、O(3)/mesh、Triton 基准、CANNBot |
| [index/README.md](index/README.md) | 统一索引、按算子族/按证据检索 |
| [index/catalog.json](index/catalog.json) | ACW 直接读取的机器索引（20 仓库记录、35 份实现） |
| [index/cards.md](index/cards.md) | 35 张阅读卡：一句话机制、源码入口、本机状态、边界 |
| [competition/mfg-2026/official/specs.md](competition/mfg-2026/official/specs.md) | 2026 制造专场 15 题官方 shape/dtype/属性/约束 |
| [docs/AGENT_READING_GUIDE.md](docs/AGENT_READING_GUIDE.md) | 给后续 agent 的十分钟地图 + 阅读路线 |
| [skills/README.md](skills/README.md) | 给 ACW/agent 的检索、入库、本机验证工作流 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 两层架构、条目契约、证据等级、验证协议 |
| [docs/MIGRATION_20261117.md](docs/MIGRATION_20261117.md) | 旧结构到新结构的迁移记录 |

## 当前规模

| 项 | 数量 |
|---|---:|
| 仓库记录 | 20（18 个独立算法来源） |
| 深读仓库 | 15（含 1 个负例） |
| 实现条目 | 35 |
| 竞赛条目 | 6 个赛事/基线（含制造专场）+ 1 个淘汰记录 |
| 通用条目 | 8 个算子/基准 + CANNBot 接入位 + 1 个淘汰记录 |
| 本机 910B3 有证据的实现 | 11 / 35：5 通过、2 负例通过、1 部分通过、2 失败、1 仅离线；制造专场 1/15 已跑（FourierTransform2D 32×32） |

> 去重口径：GitCode `cann/cann-ops-competitions` 与既有 26 份 GroupedMatmulGelu 官方基线同仓不同赛事；`cann/elec-ops-inspection` 是 Hansen 归档的上游官方仓。两者都登记并深读，但不重复增加“独立算法来源”数。

## 2026-11-17 制造专场补充

- [2026 制造行业算子专场：15 题源码地图](competition/mfg-2026/README.md)：6 个官方/生产仓、15 份实现映射、2 个 910B 基础件、75 份选中源码快照；官方题面已落库到 [official/](competition/mfg-2026/official/README.md)；6 个 adapt 题改造步骤见 [adapters.md](competition/mfg-2026/adapters.md)；上游 tie/空段/first-K 核对见 [算法参考](docs/research/MFG2026_ALGORITHM_REFERENCES_20261117.md)；搜索/去重/缺口见 [制造专场源码调研](docs/research/MFG2026_SOURCE_SURVEY_20261117.md)；
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
