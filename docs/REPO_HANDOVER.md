# 昇腾算子知识库：公开仓库交接与继续维护

编写日期：2026-11-17；公开化更新：2026-11-20。本文可单独转交给新环境的开发者或 agent，说明仓库身份、两层结构、验证/入库流程和回退方式。

## 1. 仓库身份

| 项 | 内容 |
|---|---|
| GitHub 仓库 | https://github.com/hfutslowpoke/ascend-operator-notes |
| Git 地址 | `https://github.com/hfutslowpoke/ascend-operator-notes.git` |
| 所有者 | `hfutslowpoke` |
| 可见性 | **Public**（匿名可读/可 clone；写入仍需 collaborator 权限或 token） |
| 默认分支 | `main` |
| 当前重构分支 | `architecture/two-tier-knowledge-base-20261117` |
| 用途 | ACW 的外部算子知识层：竞赛机制、通用算子、证据、索引和 skills |

`main` 与 `architecture/two-tier-knowledge-base-20261117` 现在指向同一版三层结构；`research/open-source-operators-20261116` 保留重构前的历史。旧版 GroupedMatmulGelu 归档仍在 `competition/grouped-matmul-gelu/`。

公开化时已把 mHcPre 的源码和候选包从 `main` 历史中移除（团队最佳源码不可见、个人版优化不足），只保留官方题面、团队排名/SID/分数和 [competition/excluded.md](../competition/excluded.md) 的说明。GitHub 仍可能在一段时间内保留不可达对象；不要把旧 commit 当成公开内容。

## 2. 两层结构

```text
competition/   竞赛高性能 PR/实现，按赛事组织
general/       CANNBot 及其它非竞赛来源，按算子族组织
index/         统一机器索引、按族/按证据视图
skills/        给 ACW/agent 的检索、入库、验证工作流
docs/          方法论、经验、验证记录、迁移记录
tools/         build_index.py、verify_archive.py
```

每个准入条目：

```text
<entry>/
├── README.md      # 语义、源码、机制、作者证据、本机验证、负例、复用判断
├── source.json    # 固定 commit、源码路径/行号/哈希（生成）
└── evidence/      # 原始日志/结果 JSON；没有则明确写未验证
```

详细契约见 [docs/ARCHITECTURE.md](ARCHITECTURE.md)。

## 3. 已保存内容

### 3.1 GroupedMatmulGelu 历史基线

- 26 份公开最终 PR、248 个原始文件，逐文件 SHA256 + Git blob ID；
- 官方题面、提交规则、LICENSE、PROVENANCE；
- 我们的实际参赛源码 `competition/grouped-matmul-gelu/our-submission/kernel.asc`：
  - SHA256 `161ac0ff2ab1920ca3de3e4f5d8b1b4d19a9a720c2781dcbc5c065f989b0b887`
  - SID `6af7d84c6bd0f9be3716f4c6`
  - 2026-11-16 榜单观察：58.54 分、第 11、整包 Pass
- 公开源码分析、复盘、平台证据、PMU/负实验。

这些是历史快照，不是当前通用算子性能，也不是 910C 结论。

### 3.2 外部调研

- 24 个仓库记录、18 个独立公开算法来源、15 个深读仓库、39 份实现；
- 竞赛层：CANN 竞赛仓、电力行业仓、Hansen、KernelSwift、上合 2026、星辰杯 2026；
- 通用层：Be4ver SVD、eigh 候选、Mamba-2、vLLM-HUST、Parity O(3)/mesh、52Xayah 负例、两个 Triton 基准、CANNBot 接入位；
- 本机 910B3 证据：KernelSwift 2 PASS/1 编译失败、Be4ver 4 组 PASS、Lstsq 3 组部分 PASS、recompute_w_u 编译成功/运行失败、Concat 仅离线；上合/星辰 4 个条目为 `platform-only`。

完整数字见 [README](../README.md)、[index/catalog.json](../index/catalog.json)、[docs/validation/LOCAL_910B_VALIDATION_20261116.md](validation/LOCAL_910B_VALIDATION_20261116.md)。

## 4. 导航

| 路径 | 内容 |
|---|---|
| [README.md](../README.md) | 全库入口 |
| [competition/README.md](../competition/README.md) | 竞赛库入口 |
| [general/README.md](../general/README.md) | 通用库入口 |
| [index/README.md](../index/README.md) | 索引使用说明 |
| [index/catalog.json](../index/catalog.json) | ACW 直接读取的机器索引 |
| [skills/README.md](../skills/README.md) | ACW skills |
| [docs/ARCHITECTURE.md](ARCHITECTURE.md) | 架构/条目契约/证据等级 |
| [docs/MIGRATION_20261117.md](MIGRATION_20261117.md) | 旧结构→新结构映射、备份哈希 |
| [docs/PATTERN_COOKBOOK.md](PATTERN_COOKBOOK.md) | 可迁移设计手册 |
| [docs/TOOLS_AND_REPRODUCIBILITY.md](TOOLS_AND_REPRODUCIBILITY.md) | 工具、计时、复现边界 |
| [docs/experiences/README.md](experiences/README.md) | 跨实现经验 |
| [docs/research/OPEN_SOURCE_REPOSITORY_SURVEY_20261116.md](research/OPEN_SOURCE_REPOSITORY_SURVEY_20261116.md) | 全量调研总览 |
| [docs/validation/LOCAL_910B_VALIDATION_20261116.md](validation/LOCAL_910B_VALIDATION_20261116.md) | 本机复测记录 |

## 5. 新环境首次配置

需要 Git 和 Python 3。只浏览/维护资料不需要 NPU；运行算子需另备 CANN/驱动/硬件。

**只读（推荐给其它会话/agent）**：公开仓库无需登录、无需 token。

```bash
git clone https://github.com/hfutslowpoke/ascend-operator-notes.git
cd ascend-operator-notes
git checkout architecture/two-tier-knowledge-base-20261117

python3 tools/build_index.py --check
python3 tools/verify_archive.py
```

也可以直接读 GitHub 网页或 `raw.githubusercontent.com`；匿名 API 有速率限制（约 60 次/小时）。

**写入（仅授权成员）**：需要 GitHub 账号的 collaborator 权限和 token。

```bash
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
git push
```

上传需要 `can_push=true`。仓库是 Public，但 `push`/`PATCH`/删除分支仍然要求授权；不要把密码、token 或 `hosts.yml` 放进仓库或交接文本。若要限制 `main` 的直接推送，可另行启用 GitHub ruleset/branch protection。

如新环境尚无 Git 身份：

```bash
git config user.name "your-name"
git config user.email "your-email@users.noreply.github.com"
```

## 6. 如何继续增加内容

### 6.1 竞赛条目

1. 按 [skills/competition-repo-intake/SKILL.md](../skills/competition-repo-intake/SKILL.md) 筛选、固定 commit、去重、深读；
2. 建 `competition/<event>/<entry>/README.md`、`evidence/`；
3. 在 `index/catalog.json` 增加 `tier=competition` 的 repository/implementation；
4. 运行：

```bash
python3 tools/build_index.py
python3 tools/verify_archive.py
git diff --check
```

### 6.2 通用条目

1. 按 [general/cannbot/INGESTION.md](../general/cannbot/INGESTION.md) 判断内容类型；
2. 建 `general/<family>/<entry>/README.md`、`evidence/`；
3. 在 `index/catalog.json` 增加 `tier=general` 记录；
4. 同样运行 build/verify。

### 6.3 证据要求

- 本机运行必须经 `/workspace/acw/bin/npu-lock`；
- 记录 910B3 / CANN 9.1.0 / gcc / cmake / shape / dtype / 计时口径 / 重复次数；
- 编译成功≠可运行；小 shape PASS≠全 shape PASS；
- 失败码、fault kernel、尝试过的兼容 patch 必须保留；
- 910B3 结果不外推 910C；作者/平台数字与本机数字分开写。

## 7. 许可与复制边界

- 优先引用固定 commit + 路径/行号 + SHA256，不复制第三方源码；
- `competition/grouped-matmul-gelu/prs/` 是已归档的公开 PR 快照，保留原始 LICENSE/PROVENANCE；
- `competition/mfg-2026/*/source/` 是 CANN 官方仓/DrivingSDK 的固定 commit 片段，分别按 CANN Open Software License 2.0 / Apache-2.0 保留头部与许可证全文，见 [competition/mfg-2026/THIRD_PARTY_LICENSES.md](../competition/mfg-2026/THIRD_PARTY_LICENSES.md)；
- 未声明许可证的仓库（Hansen、KernelSwift、52Xayah、Be4ver）只做引用/行号分析，源码缓存在工作区外，不复制进本库；
- 本地兼容 patch 只放工作区外副本，不写入第三方仓库和本库；
- 公开可读不等于允许再许可；本库没有为整个集合设置统一许可证。

## 8. 历史与回退

| 项 | 值 |
|---|---|
| 重构前 HEAD | `8e4d486a6cca7c7f58fa5dd8c54ab591d8d94aed` |
| 重构前分支 | `research/open-source-operators-20261116` |
| 工作区备份 | `/workspace/acw/var/private/repository-research-cache/kb-restructure-20261117/worktree-before.tar.gz` |
| 备份 SHA256 | `26f33fe2e7564c1ae598bdab83056ed97f72118f903bffce98126327d0f184b2` |

回退方式：

- Git：`git checkout 8e4d486 -- .` 或从该 commit 建新分支；
- 备份：解压 tar 到临时目录，对照 `file-map.json` 恢复；
- 迁移细节见 [docs/MIGRATION_20261117.md](MIGRATION_20261117.md)。

按本次重构约定，推送验证成功后删除本地暂存备份；Git 历史仍保留旧结构。

## 9. 不要做

- 不修改 ACW 框架或当前算子工程；
- 不自动向竞赛平台提交；
- 不把作者/平台成绩写成本机复现；
- 不把 910B3 结果外推到 910C；
- 不在未确认许可时复制第三方源码；
- 不手改生成文件（`competition/catalog.json`、`general/catalog.json`、各 `source.json`、`index/by-*.md`），要改就改 `index/catalog.json` 后重新生成。
