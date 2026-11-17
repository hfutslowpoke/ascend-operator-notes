---
name: competition-repo-intake
description: 筛选公开竞赛仓库，固定 commit、去重、深读实现、做 910B3 验证，并按两层知识库契约建立 competition 条目。
---

# Competition Repo Intake

## 何时使用

- 发现一个新的竞赛仓/PR/榜单，想判断是否值得纳入；
- 需要把某个竞赛实现从“看过”变成“可追溯、可复用、可验证”的条目；
- 需要更新 `competition/catalog.json`、`source.json` 和条目 README。

## 输入

```text
repo_url:        公开仓库/PR/官方归档 URL
competition:     赛事名称、年份、赛区/赛道
candidate:       算子/队伍/PR 编号
scope:           本轮要读几个实现、是否做本机构建
hardware:        作者环境 + 本机环境（本机只有 910B3）
```

## 步骤

### 1. 确认来源资格

- 公开可访问；不是 fork/镜像/二次搬运（与既有来源重复时只登记关系，不重复计数）。
- 有真实 kernel/Host 源码，不是只有题面、报名页、aclnn 测试脚手架。
- 能固定 commit（`git ls-remote` 或 raw URL 的 commit），记录默认分支和访问日期。
- 记录许可证；未声明许可证时只做引用/行号分析，不复制源码。

### 2. 去重

对照 `index/catalog.json`：

- 同一官方仓的不同赛事/算子：可以新增实现，但不增加“独立算法来源”；
- 同一赛事的归档仓与官方仓：选一个作为来源，另一个只写关系；
- 同一实现的 fork/镜像：不新增条目。

### 3. 选实现

优先选：

- 有作者/平台性能证据或逐 case 日志；
- 机制与现有条目互补（不是又一个同构 GEMM）；
- 能在 910B3 上构建/运行，或能明确记录失败码；
- 代码量可控，能定位 Host/kernel/入口。

不选：

- 只有测试 harness、没有 kernel；
- 只有榜单截图、没有源码；
- 语义/环境完全无法确认。

### 4. 深读

按 `docs/ARCHITECTURE.md` 的条目契约写：

1. 来源登记（URL、作者、赛事、commit、许可、缓存、去重）；
2. 语义与约束（数学式、shape/dtype/layout、边界、未知项）；
3. 源码定位（路径、符号、行号、Host 分路、kernel 入口）；
4. 机制（切分、布局、搬运、驻留、同步、融合、数值）；
5. 作者证据（baseline、计时边界、warmup/repeat、硬件、原始位置）；
6. 本机验证（见 `ascend-910b-local-validation` skill）；
7. 适用条件与负例；
8. 复用判断。

### 5. 建目录

```text
competition/<event>/<entry>/
├── README.md
├── source.json        # 由 tools/build_index.py 生成
└── evidence/          # 原始日志/结果 JSON
```

赛事/条目命名用小写 kebab-case；不要把新赛事塞进 `grouped-matmul-gelu/`。

### 6. 更新机器索引

在 `index/catalog.json` 增加/更新：

- `repositories[]`：`tier=competition`、`entry_path`、`commit`、`license`、`status`；
- `implementations[]`：`operator_family`、`source_files`（路径/行号/SHA256/URL）、`conditions`、`negative_cases`、`reuse_decision`、`local_status`、`local_validation`；
- 淘汰项：`status=excluded`，在 `competition/excluded.md` 写原因。

### 7. 生成与校验

```bash
python3 tools/build_index.py
python3 tools/verify_archive.py
git diff --check
```

检查：

- `source.json` 的 commit/路径/哈希与 catalog 一致；
- 条目 README 的链接有效；
- 本机日志已放进 `evidence/`，路径写入 catalog；
- 失败也是有证据，不要删。

### 8. 提交

- 分支建议：`research/<topic>-<date>` 或 `architecture/<topic>-<date>`；
- commit message 写清：来源、固定 commit、读了哪些实现、本机结果/失败；
- 不自动向竞赛平台提交；不修改 ACW 框架或当前算子工程。

## 输出模板

```markdown
## 本轮纳入
| 条目 | 赛事 | commit | 实现 | 证据 | 本机 |
|---|---|---|---|---|---|

## 去重/未计入
- ...

## 本机结果
- PASS/FAIL/未跑，命令与日志路径

## 下一轮
- 未定位的失败码、未覆盖 shape、需要 CANN 8.5/910C 的项
```

## 边界

- 不把“公开 PR”自动等同于“规则合规/人工审核通过”；
- 不把作者 22/22 或榜单第 1 当成本机复现；
- 不为了凑数把测试脚手架/重复搬运算成独立来源；
- 不在未确认许可时复制第三方源码。
