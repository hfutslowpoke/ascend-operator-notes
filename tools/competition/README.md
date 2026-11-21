# CANN 算子比赛起步包（2026-10-30 r1）

面向 CannJudge 类平台（`cannjudge.cn`）上的 Ascend C / AscendC 算子开发比赛。把“登录、找题、下载模板、开发、验证、提交、看结果”这一套流程，以及这类比赛最容易踩的坑，压缩成一份可直接带走的工具包。

## 先看这里

| 你的问题 | 先读 |
|---|---|
| 新会话怎么开工 | `docs/00-NEW-SESSION-PROMPT.md` |
| 怎么登录、怎么拿到比赛和题目 ID | `docs/01-platform-login-and-problem-discovery.md` |
| 怎么提交、提交格式是什么、怎么防重复 | `docs/02-submission-workflow.md` |
| 本地怎么编译、运行、看 tiling key | `docs/03-local-dev-build-run.md` |
| 这类算子开发最常见的坑 | `docs/04-common-pitfalls.md` |
| 平台计时、评分、本地怎么对齐 | `docs/05-performance-and-scoring.md` |
| 哪些事绝对不能做 | `docs/06-compliance-and-redlines.md` |
| 提交前逐项检查 | `docs/07-submit-checklist.md` |
| 报错怎么排查 | `docs/08-troubleshooting.md` |
| 2026-10-30 公开赛事/题目快照 | `docs/09-live-snapshot-20261030.md` |

工具：

```text
tools/cann_client.py       通用平台客户端：账号、赛事、题目、模板、历史、详情、榜单
tools/cann_submit.py       通用提交器：模板校验、全局 ledger、账号额度、SID 落盘、只 GET 轮询
tools/cann_quota.py        账号额度预测：50 次 / 12:00→12:00 北京时间，跨题目统计
tools/cann_package.py      按模板生成可复现提交 zip / payload，不联网提交
tools/check_env.sh         检查 CANN、NPU、编译器、profiler 是否就绪
tools/npu_lock.sh          用 flock 串行化实际设备运行（编译/CPU 校验不持锁）
```

## 5 分钟快速开始

### 1. 登录并保存 Cookie

浏览器打开 <https://cannjudge.cn/auth/login>，用比赛账号登录（账号密码 / GitCode / 华为账号，以赛事要求为准）。

登录后打开开发者工具：Application / Storage → Cookies → `https://cannjudge.cn`，复制 `cannjudge_auth` 的值。然后：

```bash
python3 tools/cann_client.py set-cookie A
python3 tools/cann_client.py accounts
python3 tools/cann_client.py check-auth --problem <PROBLEM_ID>
```

`set-cookie` 是隐藏输入，不会把 Cookie 写进命令历史；账号文件默认在 `~/.config/cannjudge/accounts.json`，权限 `0600`。

### 2. 找比赛、找题目

```bash
# 列出公开赛事
python3 tools/cann_client.py contests

# 用赛事 slug 解析 contestId，并列出题目
python3 tools/cann_client.py resolve <contest-slug>
python3 tools/cann_client.py problems <contest-slug>

# 已知赛事 ID 时
python3 tools/cann_client.py problems <CONTEST_ID>
```

页面地址通常是 slug，API 要的是 24 位 ObjectId。不要把另一场比赛的题目 ID 拿来复用。

### 3. 下载模板、确认可编辑文件

```bash
python3 tools/cann_client.py template <PROBLEM_ID> --account A
python3 tools/cann_client.py download-template <PROBLEM_ID> template.zip --account A
```

重点看 `editable: true` 的文件和它们的相对路径。提交时 `files[].path` 必须与模板路径一致；不要把 ZIP 整包 POST 上去。

### 4. 本地开发与验证

至少完成：

- 编译通过，实际 tiling key / kernel 名与预期一致；
- 标准输入、边界输入、随机/模糊输入都过精度门；
- 输入只读、输出确实写回 GM；
- 正确处理重复索引、非法索引、空块、尾部、RoPE、KV merge 等题目语义；
- 用 `npu_lock.sh` 串行跑**实际设备运行**（编译/CPU 校验可并行），记录源码 SHA、构建产物 SHA、设备频率和完整 Task 时间。

### 5. 先 dry-run，再提交

```bash
# 默认只校验，不 POST
python3 tools/cann_submit.py \
  --problem-id <PROBLEM_ID> --account A \
  --source-dir ./my_submission --tag my-first-check

# 确认 dry-run 输出后，真正提交（显式授权）
CANNJUDGE_ALLOW_SUBMIT=1 python3 tools/cann_submit.py \
  --problem-id <PROBLEM_ID> --account A \
  --source-dir ./my_submission --tag my-first-submit --submit
```

提交器会：

1. 拉模板，核对可编辑文件；
2. 对 `path + 内容` 做稳定 SHA；
3. 查全局 ledger，拒绝重复源码（本地安全策略）；
4. 查该账号当前 12:00→12:00 窗口、目标赛事内全部题目的提交数；
5. POST 前先落盘 `post_started`；
6. 拿到 `submissionId` 后立即落盘，再只 GET 轮询；
7. 保存最终 detail 和公开用例时间。

ledger 默认在 `~/.local/state/cannjudge/submit-ledger`，所有会话共享；可用 `CANNJUDGE_SUBMIT_LEDGER` 改路径。提交时同一 ledger 上持 `flock`，只防止并发重投，不做额外 pacing/cooldown。`submit_background.py` 是 2026 比赛专用的账号轮换脚本，不是通用提交路径。

### 6. 看结果

```bash
python3 tools/cann_client.py history <PROBLEM_ID> --account A
python3 tools/cann_client.py detail <SUBMISSION_ID> --account A
python3 tools/cann_client.py ranking <PROBLEM_ID> --account A
```

## 这个包不包含什么

- **不包含任何 Cookie、Token、密码、账号私有配置。**
- **不包含某道题的答案源码或平台测试数据。**
- 不保证平台 API 永远不变；页面/接口变化时以官方页面和实际返回为准。

## 使用边界

这是一份“比赛工程起步包”，不是规则解释器。最终以目标赛事的官方题面、模板、提交说明和规则为准。尤其是：

- 不同赛事的题目模板可能完全不同（单 `kernel.asc`、`custom_template`、host/kernel 多文件）；
- 不同赛事的提交额度、日界线、队伍绑定、采用哪次成绩的规则不同；
- 平台计时口径、评分公式、隐藏点数量不同；
- 本地 910B3/910B4 与平台设备可能存在频率、L2、调度差异。

## 目录

```text
.
├── README.md
├── docs/
│   ├── 00-NEW-SESSION-PROMPT.md
│   ├── 01-platform-login-and-problem-discovery.md
│   ├── 02-submission-workflow.md
│   ├── 03-local-dev-build-run.md
│   ├── 04-common-pitfalls.md
│   ├── 05-performance-and-scoring.md
│   ├── 06-compliance-and-redlines.md
│   ├── 07-submit-checklist.md
│   ├── 08-troubleshooting.md
│   └── 09-live-snapshot-20261030.md
├── official/
│   └── 星辰杯算子开发规则_v1.0.md
├── examples/
│   └── accounts.example.json
└── tools/
    ├── cann_client.py
    ├── cann_submit.py
    ├── cann_package.py
    ├── check_env.sh
    └── npu_lock.sh
```
