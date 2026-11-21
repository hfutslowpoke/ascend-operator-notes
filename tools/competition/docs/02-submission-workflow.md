# 提交流程与提交器

## 1. 平台提交的不是 ZIP

CannJudge 前端提交的是 **可编辑文件的 JSON 列表**：

```json
{
  "problemId": "<24位 ObjectId>",
  "userId": "<账号 UID>",
  "files": [
    {"path": "kernel.asc", "content": "完整源码文本"}
  ]
}
```

`path` 必须与模板返回的可编辑文件路径一致。不要把整个工程 ZIP 直接 POST 到 `/submissions/submit`。

部分旧 host/kernel 工程还会读取四个兼容字段：

```json
{
  "host_cpp": "...",
  "kernel_cpp": "...",
  "tiling_h": "...",
  "tiling_key_h": "..."
}
```

新模板不一定需要；旧模板可能需要。`cann_submit.py --compat auto` 会按路径尝试填充，但**提交后必须回读详情核对**，不能只看 ZIP。

## 2. 提交前必须确认

1. `problemId` 是目标题目，不是另一场比赛的同名题目；
2. `userId` 与 Cookie 是同一个账号；
3. 模板 `editable` 列表与本地文件一一对应；
4. 每个文件是完整源码，不是 diff、不是压缩包、不是空文件；
5. 源码 SHA 与本地验证过的版本一致；
6. 输入只读、输出写回 GM、没有 host 数值计算；
7. 没有 hash/白名单/针对具体测试用例的快路径；
8. 特化分支对声明域内任意合法输入正确，且有正确 fallback；
9. 没有把调试用的 `return`、空 kernel、纯拷贝、结果重放留在最终版；
10. 该源码没有在本题目提交过（全局 ledger 去重，本地策略；平台是否允许重复以官方规则为准）。

## 3. 使用 `cann_submit.py`

### 3.1 默认 dry-run

```bash
python3 tools/cann_submit.py \
  --problem-id <PROBLEM_ID> \
  --account A \
  --source-dir ./my_submission \
  --tag my-candidate-r1
```

默认不 POST，只做：

- 拉模板；
- 核对 `editable`；
- 读本地文件；
- 计算稳定 SHA；
- 检查**全局 ledger** 是否重复（所有会话共享，不是工作区里的 `.cann-submit`）；
- 检查该账号在**当前 12:00→12:00 北京时间窗口内、全部题目**的提交数；
- 打印文件列表、字节数、digest、payload 大小。

### 3.2 真正提交

```bash
CANNJUDGE_ALLOW_SUBMIT=1 python3 tools/cann_submit.py \
  --problem-id <PROBLEM_ID> \
  --account A \
  --source-dir ./my_submission \
  --tag my-candidate-r1 \
  --submit
```

`CANNJUDGE_ALLOW_SUBMIT=1`（或 `ACW_ALLOW_SUBMIT=1`）是本地授权保护，不是平台规则；默认 dry-run，不会 POST。成功后会先看到：

```text
accepted sid=<SUBMISSION_ID>
```

随后默认每 3 秒 GET 一次，直到终态。也可以 `--no-poll` 只拿 SID，之后手动查：

```bash
python3 tools/cann_client.py detail <SUBMISSION_ID> --account A
```

### 3.3 文件路径不匹配

如果模板路径是 `op_kernel/foo.cpp`，而本地目录结构不同：

```bash
# 方式一：把本地目录整理成模板路径
# 方式二：显式覆盖
python3 tools/cann_submit.py ... \
  --file op_kernel/foo.cpp=./src/foo.cpp
```

`--file` 可以重复。工具仍会按模板 path 提交。

### 3.4 旧 host/kernel 模板的兼容字段

```bash
python3 tools/cann_submit.py ... --compat auto
```

工具会尝试：

- `host_cpp` ← 第一个 `op_host/*.cpp`
- `kernel_cpp` ← 第一个 `op_kernel/*.cpp`
- `tiling_h` ← 第一个 `*tiling*.h`（排除 tiling_key）
- `tiling_key_h` ← 第一个 `*tiling_key*.h`

如果模板不是这个结构，或详情回读发现字段不对，改用 `--compat none`（默认）并只提交 `files`。**兼容字段填错可能让评测走错路径。**

### 3.5 重复源码

工具按 `path + \0 + content + \0` 计算 SHA-256，并写入**全局 ledger**（所有会话/账号共享）：

```text
~/.local/state/cannjudge/submit-ledger/<PROBLEM_ID>/<digest>.json
~/.local/state/cannjudge/submit-ledger/<PROBLEM_ID>/<digest>.accepted.json
~/.local/state/cannjudge/submit-ledger/<PROBLEM_ID>/ledger.jsonl
```

可用 `CANNJUDGE_SUBMIT_LEDGER` 或 `ACW_SUBMIT_LEDGER` 改路径；不要用工作区里的 `.cann-submit`，否则不同会话互相看不见。旧工作区已有 `.cann-submit` 时，先把它移到全局路径（或临时用 `--ledger-dir` 指向旧目录）再提交，避免把已提交的 digest 当成新包。提交时同一 ledger 上还会持有 `flock`，防止两个会话同时 POST 同一 digest。

如果同一 digest 已存在，默认拒绝。这是**本地安全策略**，不是 CANNJudge 的硬规则；确实需要复投时：

```bash
--allow-duplicate
```

但同一源码重复提交通常没有性能收益，只会消耗额度；平台侧也可能按重复包处理。

### 3.6 额度与 429

- **平台规则**：每个账号 50 次 / 北京时间 12:00→12:00 窗口；工具默认按目标赛事内全部题目统计，具体以官方规则为准；
- 工具默认按账号全部题目统计（`--quota-scope account`），不是只数当前题；
- `--max-per-day 50` 是本地镜像保护；`--max-per-day 0` 可关闭；
- 不同赛事额度/重置时间不同，按官方说明调整 `--reset-hour` 和上限；
- 429 时工具会保存 `retry_after_seconds`，不重试；
- 可单独查额度：

```bash
python3 tools/cann_quota.py --contest <CONTEST> --account A --details
```

- 等额度恢复后再提交；
- 不要用换账号、改 ZIP 名、改时间戳等方式绕过额度或重复限制。账号轮换只应在赛事允许的队伍账号范围内使用。

### 3.7 网络中断 / 没有拿到 SID

工具在 POST 前先写 `post_started`，POST 后如果拿不到 `submissionId` 会标记 `ambiguous_do_not_retry`，并提示先查历史。

正确做法：

1. 查该账号提交历史；
2. 按源码 SHA、创建时间、题目匹配；
3. 如果平台已受理，拿已有 SID 查详情；
4. 只有确认没有受理，才考虑重新提交。

**POST 之后只 GET，不重发。**

## 4. 提交状态

常见终态：

| 状态 | 含义 | 处理 |
|---|---|---|
| `Pass` | 通过 | 看 `valid`、逐用例时间、精度 |
| `Failed` | 失败 | 看 `result[].msg` |
| `Wrong Answer` | 数值错误 | 查精度、边界、语义 |
| `Compile Error` | 编译失败 | 查模板路径、宏、API、编译日志 |
| `Runtime Error` | 运行错误 | 查设备初始化、workspace、越界、同步 |
| `Time Limit Exceeded` | 超时 | 查死循环、同步、任务量 |
| `System Error` | 平台/环境错误 | 保留 SID，稍后重查或联系赛方 |

`valid=true` 与 `Pass` 要分开看：有的赛事会出现“用例 Pass 但提交无效（违规/源码不一致/非最终版）”。

## 5. 结果字段

`GET /submissions/<SID>?userId=<UID>` 常见字段：

- `status`
- `valid`
- `result[]`
  - `testcase_id`
  - `testcase_status`
  - `time`
  - `best_time`
  - `precision_ratio`
  - `msg`（可能包含 profiler 的 `Task Duration` 文本）

注意：

- `precision_ratio=1` 不等于误差为零；
- `result[].msg` 可能挂在相邻用例上，不能机械地把消息行号当成用例编号；
- 平台 `time` 的单位、聚合方式以题目/日志为准；
- 保存原始 detail，后续用脚本离线复核。

## 6. 提交后回读

至少做三件事：

1. 保存 `accepted.json`（SID、账号、problemId、源码 SHA、时间）；
2. 保存最终 `detail.json`（完整平台响应）；
3. 下载提交工程 ZIP，核对可编辑源码 SHA 与本地一致：

```bash
python3 tools/cann_client.py download '/submissions/<SID>/package' submitted.zip --account A
```

如果平台返回的源码与本地不一致，先解决传输/兼容字段问题，不要继续优化。

## 7. 推荐提交节奏

- 本地验证通过 → dry-run → 提交 1 次；
- 拿到终态后，保存 detail 和源码 SHA；
- 需要对比时，尽量同期穿插父本/候选；
- 不要把“本地最好”直接当“线上当前最好”；
- 采用哪次成绩、是否覆盖，按赛事规则和队伍约定执行；
- 失败包也保留 SID 和日志，便于追溯。

## 8. 最小命令

```bash
# 1) 只校验
python3 tools/cann_submit.py --problem-id <PID> --account A \
  --source-dir ./src --tag r1-check

# 2) 提交（显式授权）
CANNJUDGE_ALLOW_SUBMIT=1 python3 tools/cann_submit.py --problem-id <PID> --account A \
  --source-dir ./src --tag r1 --submit

# 3) 手动看结果
python3 tools/cann_client.py detail <SID> --account A

# 4) 看历史和榜单
python3 tools/cann_client.py history <PID> --account A
python3 tools/cann_client.py ranking <PID> --account A
```
