# CannJudge 工具使用文档

所有命令默认在仓库根目录执行：

```bash
cd ascend-operator-notes
source cannjudge/cann.env.example
```

## 1. 环境变量

| 变量 | 默认 | 用途 |
|---|---|---|
| `CANNJUDGE_API` | `https://cannjudge.cn/api` | 平台 API |
| `CANNJUDGE_ACCOUNT_FILE` | `~/.config/cannjudge/accounts.json` | 账号文件，0600 |
| `CANNJUDGE_USE_PROXY` | `0` | 是否使用代理 |
| `NPU_LOCK` | 优先沿用 ACW 的锁，其次在本用户运行目录下 | NPU 串行锁 |
| `ACW_SUBMIT_LOCK` | 同一运行目录的 `cann-submit.lock` | 平台 POST 串行锁 |
| `ACW_SUBMIT_LEDGER` | ACW 台账或 `~/.local/state/cannjudge/submit-ledger` | 跨工作目录共享提交记录 |
| `ACW_ALLOW_SUBMIT` | `0` | 1 才允许 `cann_submit.py --submit` |

账号文件格式（**不要把真实 Cookie 提交到仓库**）：

```json
[
  {
    "label": "B",
    "uid": "<uid>",
    "cookie": "<cannjudge_auth cookie>",
    "usable_for_submit": true
  }
]
```

`cann_client.py accounts` 只打印 label/UID/过期时间，不打印 Cookie。
设置/更新 Cookie：

```bash
python3 cannjudge/tools/cann_client.py set-cookie B
```

## 2. 只读查询：cann_client.py

```bash
# 账号
python3 cannjudge/tools/cann_client.py accounts

# 赛事与题目
python3 cannjudge/tools/cann_client.py contests
python3 cannjudge/tools/cann_client.py problems gcc_2026_mfg
python3 cannjudge/tools/cann_client.py resolve gcc_2026_mfg segmentmincsr

# 题目/模板
python3 cannjudge/tools/cann_client.py problem <problemId>
python3 cannjudge/tools/cann_client.py template <problemId> --account B
python3 cannjudge/tools/cann_client.py download-template <problemId> /tmp/template.zip --account B

# 历史/详情/榜单
python3 cannjudge/tools/cann_client.py history <problemId> --account B
python3 cannjudge/tools/cann_client.py detail <SID> --account B
python3 cannjudge/tools/cann_client.py ranking <problemId> --account B

# 原始 GET（只读）
python3 cannjudge/tools/cann_client.py get "/problems/<problemId>" --account B
```

`history` 返回该账号在该题目的全部提交；`detail` 返回单次提交的逐 case 结果。
`ranking` 的 `testcases[].tbest` 和 `rows[].result[]` 是计算分数的输入。

## 3. 额度：cann_quota.py

```bash
# 所有配置了 Cookie 的账号，当前 12:00 窗口
python3 cannjudge/tools/cann_quota.py --contest gcc_2026_mfg

# 单账号 + 逐题明细
python3 cannjudge/tools/cann_quota.py --contest gcc_2026_mfg --account B --details

# JSON
python3 cannjudge/tools/cann_quota.py --contest gcc_2026_mfg --json

# 回看历史窗口（排查 429）
python3 cannjudge/tools/cann_quota.py --contest gcc_2026_mfg \
  --account B --details --at '2026-11-18T16:37:00+08:00'
```

输出：

```text
account total   eff limit remain last             reset
B           6     6    50     44 11-18 16:25      11-19 12:00
  - segmentmincsr: 6  (last 11-18 16:25)
  last submission 11.8 minutes ago
  window: 2026-11-18T12:00:00+08:00 -> 2026-11-19T12:00:00+08:00
```

字段：

| 字段 | 含义 |
|---|---|
| `total` | 窗口内全部提交数 |
| `effective` | 排除 `Judge Interrupted` 后的计数 |
| `limit` | 默认 50 |
| `remaining` | `limit - effective` |
| `last_submission` | 窗口内最后一次提交 |
| `reset` | 下一个 12:00（北京时间） |

> `cann_quota.py` 永远不 POST，只读历史。

## 3.1 分数/headroom：cann_score.py

```bash
# 保存 detail/ranking 后离线计算
python3 cannjudge/tools/cann_client.py detail <SID> --account B > detail.json
python3 cannjudge/tools/cann_client.py ranking <pid> --account B > ranking.json

python3 cannjudge/tools/cann_score.py \
  --detail detail.json --ranking ranking.json --target 60

# 只给 ranking 时用 --sid 选自己的行
python3 cannjudge/tools/cann_score.py \
  --ranking ranking.json --sid <SID> --target 60 --json
```

输出：

| 字段 | 含义 |
|---|---|
| `score` | 当前 case 分数 |
| `gain_2x` | 时间减半，总分增加多少 |
| `max_gain` | 到 100 分还剩多少 |
| `factor_to_50` | 还要快多少倍到 50 分（1.5×tbest） |
| `required_ratio_for_target` | 其他 case 不变时，该 case 到目标总分所需倍率 |
| `target_state` | `already_enough` / `needs_improvement` / `impossible_alone` |
| `all_at_1_5x` | 所有 case 到 1.5× 的总分 |
| `all_at_1_3x` | 所有 case 到 1.3× 的总分 |

注意：`impossible_alone` 表示“只靠这一个 case 达不到目标”，不是“总分不可能”；通常需要多个 case 同时改善。

## 4. 提交：cann_submit.py

### 4.1 dry-run

```bash
python3 cannjudge/tools/cann_submit.py \
  --problem-id 6af1cdaa6bd0f9be37a1afe1 \
  --account B \
  --source-dir /workspace/gcc_2026_mfg_ops/p07-segmentmincsr-continuation \
  --tag p07-v2d-r1
```

dry-run 会：

1. 拉取题目模板；
2. 只读取 `editable=true` 的文件；
3. 计算 `source_digest`；
4. 查重；
5. 检查账号当前 12:00 窗口额度；
6. 打印计划并写 `*.plan.json`；
7. **不 POST**。

### 4.2 正式提交

```bash
ACW_ALLOW_SUBMIT=1 python3 cannjudge/tools/cann_submit.py \
  --problem-id 6af1cdaa6bd0f9be37a1afe1 \
  --account B \
  --source-dir /workspace/gcc_2026_mfg_ops/p07-segmentmincsr-continuation \
  --tag p07-v2d-r1 --submit
```

`--submit` 需设置 `ACW_ALLOW_SUBMIT=1` 或兼容变量 `CANNJUDGE_ALLOW_SUBMIT=1`。
`CANNJUDGE_SUBMIT_LOCK` 和 `CANNJUDGE_SUBMIT_LEDGER` 可用于沿用旧环境的同一把锁与台账。
POST 成功后打印：

```text
accepted sid=<SID>
```

实际提交在全局锁内核对账户额度、去重、POST 和保存 SID。锁在轮询前释放。
`--no-poll` 在保存 SID 后直接返回。账户额度核对失败会阻止实际 POST。

### 4.3 常用参数

| 参数 | 说明 |
|---|---|
| `--problem-id` | 题目 ObjectId |
| `--account` | 账号 label |
| `--source-dir` | 本地工程根目录 |
| `--tag` | 本次提交标签，写入账本 |
| `--submit` | 真正 POST；省略则 dry-run |
| `--no-poll` | 不等待终态 |
| `--poll-interval` | 轮询间隔秒，默认 3 |
| `--poll-timeout` | 轮询超时秒，默认 3600 |
| `--ledger-dir` | 账本目录，默认使用上表的共享目录 |
| `--max-per-day` | 本地额度上限，默认 50 |
| `--quota-scope` | `account`（默认，所有题目）或 `problem`（旧行为） |
| `--contest` | 账号级额度所属赛事；省略时从题目推断 |
| `--reset-hour` | 北京时间刷新小时，默认 12 |
| `--allow-duplicate` | 允许忽略旧版去重记录。已有受理记录保留，仍需人工核对 |
| `--compat` | `auto` 为旧 host/kernel 模板补兼容字段 |
| `--file path=local` | 覆盖模板中的某个文件，可重复 |

### 4.4 账本

每题记录位于台账目录的 `<problem_id>/`。

```text
<digest>.plan.json       # dry-run 计划：源码哈希、文件列表、额度窗口
<digest>.accepted.json   # POST 记录：SID、状态、429 原始 body/headers
<digest>.detail.json     # 最后一次 GET 详情
ledger.jsonl             # 每次受理的追加日志
```

`accepted.json` 关键字段：

```json
{
  "status": "accepted",
  "submission_id": "<SID>",
  "platform_status": "Runtime Error",
  "post_error_metadata": {
    "body": "...",
    "headers": {"retry-after": "..."}
  }
}
```

`status` 可能是：

```text
post_started
accepted
platform_terminal
poll_deadline
rate_limited_no_post
post_error_check_history
ambiguous_do_not_retry
```

### 4.5 429

收到 429 时工具会：

- 不记录 SID；
- 写 `status=rate_limited_no_post`；
- 保存 `retry_after_seconds`（如果平台给了）和原始 body/headers；
- 退出，不自动重试。

不要：

- 立刻换账号再提交；
- 改 tag/包名重传；
- 每几分钟重试。

先运行：

```bash
python3 cannjudge/tools/cann_quota.py --contest gcc_2026_mfg --account <label> --details
```

确认是额度还是冷却。

## 5. NPU 锁：npu_lock.sh

```bash
# 直接运行
bash cannjudge/tools/npu_lock.sh <command> [args...]

# 或设置 NPU_LOCK
NPU_LOCK=/workspace/acw/var/run/npu.lock \
  bash cannjudge/tools/npu_lock.sh python3 test_local.py
```

规则：

- 所有 NPU 运行必须串行；
- 嵌套调用会复用 `ACW_NPU_LOCK_HELD=1`，不会死锁；
- 不要在锁外直接跑 `python3 test_local.py`。

## 独立打包和本地评测

```bash
python3 cannjudge/tools/cann_package.py --problem-id "$problem_id" --account "$account" \
  --source-dir "$source_dir" --out candidate.zip
bash cannjudge/tools/check_env.sh

run_dir=$(mktemp -d)
python3 cannjudge/tools/cann_eval.py build --binding "$binding" --source "$source_dir" --run-dir "$run_dir"
python3 cannjudge/tools/cann_eval.py check --binding "$binding" --source "$source_dir" --run-dir "$run_dir"
```

打包和 `cann-submit` 使用同一套可编辑文件选择、UTF-8 校验和源码摘要逻辑。评测绑定由算子工程提供。JSON 绑定可直接读，YAML 绑定需要 PyYAML。CPU 编译和打包可以并行。校验、计时、采样和诊断使用同一 NPU 锁。自定义探针先加载当前 CANN 环境，再运行下列命令。

```bash
bash cannjudge/tools/npu_lock.sh timeout --kill-after=5s 120s \
  python3 cannjudge/tools/acl_probe.py --device 0
```

`cann_eval.py` 支持 `--case-id`、`--cann-home` 和实际物理/逻辑设备号。正确性见运行目录的 `raw-*.json`，以 `cases[].passed` 为准。存在 ACW 时沿用 `/workspace/acw/var/run/`。独立环境使用 `${XDG_RUNTIME_DIR:-/tmp}/cannjudge-$(id -u)/`。

## 6. 与 ACW 的配合

ACW solo 任务：

```bash
/workspace/acw/bin/acw solo platform --refresh
/workspace/acw/bin/acw solo submit --tag p07-v2d-r1 --account B
/workspace/acw/bin/acw solo exp add --hypothesis "..." --verdict keep
```

`acw solo submit` 会调用同一套提交逻辑，并把结果写入
`/workspace/acw/var/state/solo/<task_id>.json`。

## 7. 排错

| 现象 | 检查 |
|---|---|
| `missing credential for account` | `accounts` 是否显示该 label；`set-cookie` |
| `missing editable files` | 模板路径与 `--source-dir` 是否一致 |
| `duplicate source digest` | 查共享台账 `<pid>/<digest>.accepted.json`；不要盲目 `--allow-duplicate` |
| `local quota guard` | 用 `cann_quota.py` 看 12:00 窗口；不要换账号绕过 |
| 429 | 停止；记录 body/headers；等冷却 |
| `ambiguous_do_not_retry` | 先查 `history`，按摘要/时间匹配，不要重发 |
| 平台 `Runtime Error` / `507057` | 本地 CANN 版本差异；先做最小改动 smoke，不要整包重写 |
| 编译通过但平台失败 | 检查平台 CANN 9.0 与本地 9.1 的 API/DMA 差异 |

## 8. 最小命令清单

```bash
# 额度
python3 cannjudge/tools/cann_quota.py --contest gcc_2026_mfg --account B --details

# 查询
python3 cannjudge/tools/cann_client.py history <pid> --account B
python3 cannjudge/tools/cann_client.py ranking <pid> --account B
python3 cannjudge/tools/cann_client.py detail <SID> --account B

# dry-run
python3 cannjudge/tools/cann_submit.py --problem-id <pid> --account B \
  --source-dir <dir> --tag <tag>

# POST
ACW_ALLOW_SUBMIT=1 python3 cannjudge/tools/cann_submit.py --problem-id <pid> \
  --account B --source-dir <dir> --tag <tag> --submit
```

## 9. 相关文档

- [RULES.md](RULES.md)
- [RUNBOOK.md](RUNBOOK.md)
- [PAYLOAD_CONTRACT.md](PAYLOAD_CONTRACT.md)
- [DEVELOPMENT.md](DEVELOPMENT.md)
- [SECURITY.md](SECURITY.md)
