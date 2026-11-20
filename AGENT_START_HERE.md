# Agent 从这里开始

这是一份给后续 agent 的**一屏导航**。先判断你要找什么，再进对应入口；不要从 `competition/` 或 `docs/` 里随机翻。

> **公开只读**：仓库已设为 Public，匿名可读/可 clone；写入、改设置、改可见性仍需授权。边界见 [SECURITY.md](SECURITY.md)。

## 30 秒判断

| 你要做的事 | 先读 | 再进 |
|---|---|---|
| 开发/编译/打包/提交/查榜单 | [toolkit/README.md](toolkit/README.md) | `tools/`、`tools/competition/`、ACW `bin/cann-*` |
| 找跨题经验、平台规则、负例、输入构造 | [playbook/README.md](playbook/README.md) | `playbook/topics/`、`playbook/rules/`、`docs/experiences/` |
| 找某道题、某个算子、某个开源仓、我们自己的实现 | [knowledge/README.md](knowledge/README.md) | `competition/`、`general/`、`index/` |
| 看本机 910B3 到底跑过什么 | [index/by-evidence.md](index/by-evidence.md) | 各条目 `evidence/`、`docs/validation/` |
| 把一场已结束的比赛入库 | [knowledge/INTAKE.md](knowledge/INTAKE.md) | `skills/competition-repo-intake/` |
| 看仓库整体分层和边界 | [LIBRARIES.md](LIBRARIES.md) | `docs/ARCHITECTURE.md` |

## 仓库只有三层

```text
toolkit/    开发和提交工具：索引、校验、CANNJudge 客户端/打包/提交、agent skills
playbook/   跨题经验与规则：平台精度、输入构造、本机 vs 平台、机制/负例
knowledge/  知识库：公开赛题、开源仓源码、我们自己的每算子实现与证据
```

`index/` 是知识库的索引，不是第四层；`by-cannjudge.md` 是索引视图，也不是第四层。

## 典型检索路径

### “我要做制造专场 05 FourierTransform2D”

```text
knowledge/README.md
  → competition/mfg-2026/official/specs.md
  → index/by-cannjudge.md（题号→slug→候选）
  → competition/mfg-2026/ops-fft/README.md
  → playbook/README.md（先看平台精度/输入构造规则）
  → toolkit/README.md（本机验证、打包、dry-run）
```

### “我要找某个开源仓的某段实现”

```text
knowledge/README.md
  → index/catalog.json / index/by-family.md
  → competition/<contest>/<repo>/README.md 或 general/<family>/<repo>/README.md
  → 条目 source.json 看 commit/路径/行号/SHA
  → source/ 看本地快照
```

### “小 case 只有 1 ms，怎么优化？”

```text
playbook/topics/small-shape-launch-bound.md
  → 判据：是否 launch-bound
  → 做法：去 TPipe/去 tiling、raw UB、固定 lane、减少 block
  → 证据：Sparse Softmax 2/161；mHcPre（未入库，仅经验案例）
```

### “团队排名为什么不在我的账号上？”

```text
playbook/topics/team-ranking-and-package-selection.md
  → 查 team_id，不只看 user_id
  → latest/best 两种聚合
  → 证据：mHcPre 团队第 18 来自队友 gao（未入库）
```

### “本地精度没过，但性能很好，要不要放弃？”

```text
playbook/rules/platform-evaluation.md
```

核心规则：**平台最终结果是准绳；本地精度是筛查，不是一票否决。** 只有本地失败能确定平台一定过不了（编译/运行/越界/NaN/确定性语义错）才直接放弃；单纯本地数值差要按平台容差、输入构造和平台回读再判断。

## 新比赛入库

等你给已结束赛题时，按 [knowledge/INTAKE.md](knowledge/INTAKE.md) 走：

```text
competition/<contest-slug>/
  README.md
  official/                 # contest.json + problems.json + 题面
  <problem-entry>/
    README.md
    source/                 # 我们的 kernel/host/驱动
    source.json             # 生成：SID/detail/文件哈希
    evidence/               # detail-summary/cases.csv/ranking-summary/sid
    candidates/             # 可选：未采用/未计时的版本
```

不要先写“经验总结”，先把题面、源码、自己的实现和平台证据固定下来；经验从这些事实里再提炼。

## 当前状态（2026-11-20）

```text
知识库：24 个仓库记录、39 份实现、39 张阅读卡、21 份官方题面（制造 15 + 上合/星辰 6）
经验库：6 个通用专题 + 7 篇跨实现经验 + 制造 10 个小点 + 平台规则
工具库：索引/校验/官方题面/CANNJudge 客户端/打包/提交/skills
制造分级：2 direct / 6 adapt / 5 reference / 2 none
本机：制造主实现 1/15 FFT 32×32 子集 PASS；Segment 骨架示例 PASS；CPU 参考 9 项 PASS
平台提交：本库未记录，未 POST
910C：无数据
```

## 不要这样开始

- 不要先读 `competition/grouped-matmul-gelu/prs/` 的 26 份 PR；先看 `analysis/` 和条目 README。
- 不要把 `index/cards.md` 当完整知识；它只是入口卡。
- 不要把作者 `O+A` 当本机 `V`；不要把本地 `pass` 当平台通过。
- 不要把 `playbook/` 的经验当 golden；每条经验都写了适用条件和负例。
- 不要在仓库里找 Cookie/账号/隐藏测试数据；它们不在三层库里。
