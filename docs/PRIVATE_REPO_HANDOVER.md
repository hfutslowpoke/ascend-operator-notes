# Ascend 算子经验私仓：迁移与继续提交说明

编写日期：2026-11-16。本文可单独转交给新环境的开发者或 agent，说明现有仓库及继续维护方式，不规定后续多 agent 的组织架构。

## 1. 仓库身份与用途

| 项目 | 内容 |
|---|---|
| GitHub 仓库 | https://github.com/hfutslowpoke/ascend-operator-notes |
| Git 地址 | `https://github.com/hfutslowpoke/ascend-operator-notes.git` |
| 所有者 | `hfutslowpoke` |
| 可见性 | **Private**，建立与首次上传后均已通过 GitHub API 核对 |
| 默认分支 | `main` |
| 建立日期 | 2026-11-16 |
| 初始归档基线 | `6fe1293c979c5e1438102f577e8536ba28f19964`；后续请以远端最新提交为准 |
| 用途 | 长期保存昇腾算子源码、性能优化经验、工具及可追溯证据，供后续算子开发复用 |

仓库最初整理了 2026 CANN 挑战赛上合赛区决赛 GroupedMatmulGelu 的资料。比赛已经结束，当前仓库的提交工具和开发计划属于历史资料；读取本文不代表恢复旧比赛的优化、NPU 实验或平台提交任务。

本仓库可以继续扩展到新算子、新硬件或新比赛。新项目应记录自己的语义、规则、环境与证据，不直接继承旧题目的约束和分路。

## 2. 已经保存的内容

### 2.1 公开源码与题目

- **26 份公开最终提交 PR，248 个原始文件**，包含 TeamLiquid、Lizzop、WHAleX 等队伍的源码及附带说明。
- 每份 PR 固定到具体 head commit；原始文件逐一验证 Git blob ID，并保存 SHA256。
- 官方题目、提交规则和原始 LICENSE。
- 下载的是相关 PR 的提交文件快照，并未把每个参赛者个人仓库的全部历史、分支或其它项目装入本库。

这些公开 PR 在采集时仍为 open；公开提交、平台 Pass 和人工核验通过是不同的证据。归档阶段没有重新编译或运行第三方源码。

### 2.2 我们自己的最终代码与候选

实际最终榜单代表保存在 `our_work/final_submission/kernel.asc`：

```text
SHA256: 161ac0ff2ab1920ca3de3e4f5d8b1b4d19a9a720c2781dcbc5c065f989b0b887
SID:    6af7d84c6bd0f9be3716f4c6
平台创建时间: 2026-11-14 03:41:00.774 UTC
2026-11-16 榜单观察: 58.54 分，第 11，整包 Pass
```

目录还包含当时的 CMake、sample/bench/verify 驱动和本地 reference。reference 是我们的本地实现，不是从官方获得的 golden。最终代表通过源码哈希、SID 和服务器时间对应确认，区别于较早交接日志中的母包。

`our_work/components` 另存 C1、R3 的选中组件和后续候选；各文件的状态见该目录 README。**未提交候选、仅本地更快的版本，均未被写成平台最优版。**

### 2.3 技术经验与工具

- 公开源码分析：K=1 当前调用内有限值域表、N=1 点积/窄输出引擎、真实 group_list 工作调度、活跃专家预处理、多平面 L0C 和缓冲生命周期。
- 我们的比赛复盘：成功方向、负实验、为什么局部收益没有兑现为完整耗时、平台评分时期变化和集成问题。
- PMU 与多级 roofline：区分容量、活跃地址集合、逻辑请求和物理事务；按实际关键 owner 解释等待和重叠。
- 实际用过的构建、数值、计时、profiling、ELF 审查、组件提取、提交、排名快照、日志登记和 NPU 串行锁工具。

完整 SDK、驱动、所有输入数据、所有实验、大体积 PROF、NPU 二进制以及账号凭据未打包。部分历史工具和报告引用旧机器绝对路径，复用时需适配。

## 3. 目录导航与推荐阅读

| 路径 | 内容与用途 |
|---|---|
| `README.md` | 全库入口 |
| `docs/OPEN_SOURCE_REVIEW.md` | 领先队公开源码的关键机制、适用条件和证据边界 |
| `docs/RETROSPECTIVE.md` | 比赛经验、负结果和工程教训 |
| `docs/PATTERN_COOKBOOK.md` | 语义、成本模型、生产消费、事件退休等可迁移方法 |
| `docs/TOOLS_AND_REPRODUCIBILITY.md` | 历史环境、工具用途、计时口径与复现限制 |
| `docs/PLATFORM_EVIDENCE.md` | 最终源码/SID/成绩对应及同次公开点对照 |
| `docs/RULES_AND_MATH.md` | 本次比赛数学顺序与官方约束 |
| `docs/GITHUB_QUICKSTART.md` | 面向初学者的 GitHub 使用说明 |
| `catalog/INDEX.md`、`submissions.csv` | 26 份公开 PR 的导航 |
| `catalog/public-final-sources.json` | 第三方固定 commit、原路径、哈希和获取来源 |
| `catalog/our-work.json` | 我们代码/工具/精选证据的来源、状态和哈希 |
| `third_party/official` | 官方题目、规则、LICENSE |
| `third_party/shanghe_final/pr-*` | 原样保存的公开参赛源码及 `PROVENANCE.json` |
| `our_work/final_submission` | 我们实际选中的最终源码与构建驱动 |
| `our_work/components` | 选中组件和未晋升候选 |
| `our_work/evidence` | 精选本地/PMU/平台证据和负实验 |
| `tools/analysis` | 历史分析、数值、计时与组件工具 |
| `tools/competition` | 历史平台客户端、轮询、日志协调、NPU 锁 |
| `tools/acquisition` | 公开 PR 固定版本重新获取工具，不包含账号凭据 |
| `tools/verify_archive.py` | 离线校验源码身份、JSON/Python 语法和部分敏感信息模式 |
| `MANIFEST-SHA256.json` | 归档文件的整体哈希清单，不包含它自身 |
| `THIRD_PARTY.md` | 来源、作者与许可说明 |

新环境先读 [README](https://github.com/hfutslowpoke/ascend-operator-notes/blob/main/README.md)、[公开源码分析](https://github.com/hfutslowpoke/ascend-operator-notes/blob/main/docs/OPEN_SOURCE_REVIEW.md)、[复盘](https://github.com/hfutslowpoke/ascend-operator-notes/blob/main/docs/RETROSPECTIVE.md)，再按当前算子选择相关源码与工具，无需先通读所有历史文件。

2026-11-16 新增的外部调研入口：[公开算子仓库调研总览](research/OPEN_SOURCE_REPOSITORY_SURVEY_20261116.md)、[来源登记](../catalog/external-repositories.json)、[仓库深读索引](repositories/README.md)、[跨实现经验](experiences/README.md)。这些是外部参考，不改变本库 GroupedMatmulGelu 的历史快照、成绩或规则。

## 4. 新环境首次配置

### 4.1 登录 GitHub 并克隆

需要 Git、GitHub CLI（`gh`）和 Python 3。只浏览和维护资料不需要 NPU；运行历史算子需另外准备对应 CANN/驱动和硬件环境。

在新环境终端执行：

```bash
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
gh auth status

git clone https://github.com/hfutslowpoke/ascend-operator-notes.git
cd ascend-operator-notes

gh api repos/hfutslowpoke/ascend-operator-notes \
  --jq '{repository: .full_name, private: .private, can_push: .permissions.push}'
git status
python3 tools/verify_archive.py
```

浏览器授权时登录有权限的 GitHub 账号。账号可以是所有者，也可以是受邀且有相应权限的协作者；上传需要 `can_push=true`。仓库应保持 `private=true`。

当前旧服务器的登录状态不会随 Git clone 转移；新环境需独立授权。不要把 GitHub 密码、令牌或旧服务器的 `hosts.yml` 放到仓库或交接文本中。浏览器设备码过期时重新运行登录命令即可。

Git 提交还需要作者姓名和邮箱。若新环境尚未配置，在克隆目录设置：

```bash
git config user.name "hfutslowpoke"
git config user.email "hfutslowpoke@users.noreply.github.com"
```

上面是本次归档采用的身份；实际使用其它账号时换成自己的姓名和提交邮箱。GitHub Settings → Emails 显示的个人 noreply 地址可用于保护邮箱隐私。

参考：[GitHub CLI 登录](https://cli.github.com/manual/gh_auth_login)、[克隆仓库](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)。

### 4.2 已克隆过的环境

工作区无未提交改动时：

```bash
git checkout main
git pull --ff-only origin main
python3 tools/verify_archive.py
```

有未提交改动时先保存到自己的分支和 commit，再同步 main。远端有新提交导致 push 被拒绝时，先 fetch 并合并或 rebase；不要为覆盖远端而强推 main。

## 5. 如何继续增加内容

### 5.1 保留原始快照，新增项目独立归档

`third_party`、`our_work/final_submission` 和已登记的历史组件/证据按固定快照保留。需要改造某份实现时复制到新项目目录，记录原来源和父本哈希；适配后的代码不要冒充原始提交。

下面是**未来可新增的目录示例，初始仓库没有预建**：

```text
operators/<operator_name>/
  README.md                 # 语义、一般参数族、入口、环境、当前版本
  src/                      # Host / Kernel / Tiling
  tests/                    # 输入生成、数值边界、验证驱动
  experiments/<experiment_id>/
    PLAN.md                 # 假设、预计改变的关键路径和代价
    RESULT.md               # 结果、结论、负例和证据等级
    metadata.json           # 源码/构建/输入/设备/平台身份
    measurements/           # 精选计时与必要 PMU

docs/experiences/            # 跨算子可复用经验
catalog/operators.json      # 新项目索引，实际需要时再建
```

后续架构可以选择其它目录，只需在 README 或索引中解释入口。新算子没有平台结果时写清“未测”；有本地收益但无平台收益时分别登记。

### 5.2 一次实验建议保留什么

- 算子数学语义、适用参数族、硬件/SDK/编译器身份；
- 父本和候选的源码 SHA256、实际构建入口、ELF 身份；
- 输入生成器、种子与真实分组分布，数值/guard/输入只读结果；
- 完整 Task 计时协议、对照顺序、分布统计，必要的关键 owner PMU；
- 若有平台测试，记录 SID、逐点时间、提交与观测时间、评分时期；
- 成功/负结果、机制解释、适用条件和下一步。

区分静态推导、本地计时、PMU、平台单次和平台重复证据。不要只记录最小耗时或只保留正结果。大体积输入、构建产物和 PROF 可以放外部存储，仓库保留地址与哈希。

## 6. 如何提交到 GitHub

以下示例新增一篇经验文档；修改源码或工具时替换为实际路径。

### 6.1 建立分支并编辑

```bash
git checkout main
git pull --ff-only origin main
python3 tools/verify_archive.py
git checkout -b docs/new-operator-notes

# 编辑或新增 docs/experiences/new-operator.md
git status
git diff
git add docs/experiences/new-operator.md
```

上面示例中的文件需要先实际创建。按路径添加本次内容，避免把不相关实验和临时文件一起提交。

### 6.2 更新整体哈希清单

新增或修改已登记文件会改变整体哈希。先确认改动范围符合预期，再在仓库根目录执行下面命令。它仅统计 Git 当前已跟踪/已暂存的文件，不把整个磁盘或未加入 Git 的实验数据写入清单。

```bash
python3 - <<'PY'
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess

root = Path(subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], text=True).strip())
paths = subprocess.check_output(
    ["git", "ls-files", "-z"], cwd=root).split(b"\0")
files = {}
for raw in paths:
    if not raw:
        continue
    name = raw.decode("utf-8")
    if name == "MANIFEST-SHA256.json":
        continue
    path = root / name
    files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
document = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "algorithm": "sha256",
    "files": dict(sorted(files.items())),
}
(root / "MANIFEST-SHA256.json").write_text(
    json.dumps(document, ensure_ascii=False, indent=2) + "\n")
PY

git add MANIFEST-SHA256.json
python3 tools/verify_archive.py
git diff --cached --stat
git diff --cached
```

整体清单和原始来源清单作用不同：

- `MANIFEST-SHA256.json` 记录当前归档文件，可随合理的新内容更新；
- `catalog/public-final-sources.json`、`catalog/our-work.json` 还固定了历史原件身份。改写旧原件后单纯刷新整体清单仍会校验失败，应保留原件、将改版另存。

不要通过重写原始来源哈希掩盖意外修改。校验脚本还包含部分敏感文件/令牌模式检查，但不代替检查本次实际 diff。

### 6.3 保存版本并上传

```bash
git commit -m "docs: record new operator experiment"
git push -u origin HEAD
```

上传分支后，可由后续架构的集成负责人合并，或用 GitHub Pull Request：

```bash
gh pr create --base main \
  --title "记录新算子的实验与经验" \
  --body "补充实现来源、完整调用对照与适用条件。"
```

通过网页或既有集成流程合并后，新环境再拉取 main。单人维护的小改动也可直接在 main 执行对应 add/commit/push；多人或多 agent 同时写入时，分支更容易保持改动归属。

## 7. 与后续多 agent 架构衔接

本文不选择模型、agent 职责、消息协议或任务调度器。接入本库时只需约定：

1. 所有任务能定位仓库和当前基线 commit，明确自己负责的目录或文件。
2. 并行写入采用独立 clone、分支或 worktree；避免多个 agent 同时操作同一个 Git index、checkout 或 commit。
3. 谁负责汇总 README/索引/整体清单，谁负责合并到 main，应有明确归属；整理清单时以集成后的实际文件为准。
4. 分享实际 NPU 时统一锁路径；独立工作区并不能隔离同一设备上的计时干扰。
5. 向仓库交付可复用源码、环境说明、结果和证据；当前任务队列与凭据由新环境自己的系统管理。

若使用 Git worktree，示例为：

```bash
git worktree add -b experiments/new-operator ../ascend-notes-new-operator main
```

worktree 有独立文件和 index，但共享同一 Git 仓库的对象与部分元数据；需要完全独立 Git 配置时使用独立 clone。参考 [Git 官方 worktree 文档](https://git-scm.com/docs/git-worktree)。

## 8. 来源、许可和环境注意事项

- 私仓继续保持 Private；凭据留在新环境的授权系统，仓库保存源码和证据。
- 第三方源码保留作者、原始说明、固定 commit 和许可信息；新增公开代码同样记录来源。具体说明见 `THIRD_PARTY.md`。
- 历史工具有旧绝对路径和旧接口；先看入口和依赖，再选择性迁移。比赛后台提交工具作为经验保留，不在新环境自动启动。
- 旧目标为 910C 单 die、arch22、20 AIC/40 AIV。换芯片时重新确认容量、吞吐和 API，不能直接套用旧 tiling、架构能力和理论下界。
- 本库不要求配置 GitHub Actions，也未提供自动执行历史 NPU/平台提交脚本的 CI；后续需要时由新项目自行建立。

## 9. 可直接交给新环境的开场说明

> 请先登录并克隆私仓 https://github.com/hfutslowpoke/ascend-operator-notes ，读取 `docs/PRIVATE_REPO_HANDOVER.md` 和 README，核对 main 的最新 commit，并运行 `python3 tools/verify_archive.py`。仓库已保存 26 份 GroupedMatmulGelu 公开最终提交、我们的实际最终源码、PMU/roofline 分析、比赛复盘与工具。请将它作为后续算子开发的参考与经验库，按照当前任务和你们自己的多 agent 架构接入。保留原始快照与来源；新算子、改造代码和新实验另建目录，记录源码/环境/测量身份与证据等级，更新索引和整体哈希清单后提交分支并集成。GitHub 私密访问需在新环境独立授权，旧服务器凭据不随仓库迁移。
