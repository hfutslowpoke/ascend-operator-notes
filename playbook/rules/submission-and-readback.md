# 提交与回读：dry-run、SID、detail、ranking

## 1. 默认 dry-run

```bash
# 只校验，不 POST
bin/cann-submit \
  --problem-id <PID> --account A \
  --source-dir ./my_submission --tag mfg-r1
```

dry-run 至少核对：

- 模板 `editable` 路径与本地文件一一对应；
- 文件是完整源码，不是 diff/zip/空文件；
- 源码 SHA 与本地验证版本一致；
- payload 大小/字段；
- 是否重复源码；
- 当天额度。

## 2. 打包（可选）

```bash
bin/cann-package \
  --problem-id <PID> --account A \
  --source-dir ./my_submission --out submission.zip \
  --payload-json payload.json
```

`cann-package` 只生成 zip + manifest，不 POST。manifest 里的 SHA 可用于复盘。

## 3. 真正提交

只有用户明确授权后才 POST：

```bash
ACW_ALLOW_SUBMIT=1 bin/cann-submit \
  --problem-id <PID> --account A \
  --source-dir ./my_submission --tag mfg-r1 --submit
```

`bin/cann-submit` 在 `ACW_ALLOW_SUBMIT!=1` 时拒绝 `--submit`；不要绕过。

## 4. SID 与状态机

```text
dry-run
  → POST（授权后）
  → submissionId/SID
  → 只 GET 轮询
  → detail（status/valid/result[]/Task Duration）
  → ranking（如需）
```

如果 POST 后没拿到 SID：

- 先查 `history`/本地 ledger；
- 不要直接重发；
- `ambiguous_do_not_retry` 表示可能已受理；
- 只有确认平台没受理，才考虑重提。

## 5. 重复与额度

- 同一 `path+content` 的 SHA 默认拒绝；
- 同一源码重复提交通常没有性能收益；
- 默认 `--max-per-day 50`，以赛事规则为准；
- 429 保存 `retry_after_seconds`，不换账号/不改时间戳绕过；
- 额度/重复限制是硬规则，不是“优化点”。

## 6. 回读

```bash
bin/cann-client history <PID> --account A
bin/cann-client detail <SID> --account A
bin/cann-client ranking <PID> --account A
```

保存：

- SID、提交时间、账号标签（不保存 Cookie）；
- `status`/`valid`；
- `result[]` 逐用例状态、失败码；
- `Task Duration`、频率、Block Dim；
- 排名/分数快照。

## 7. 失败处理

| 状态 | 先看 | 处理 |
|---|---|---|
| `Compile Error` | 模板路径、宏、API、编译日志 | 修构建，不重投同包 |
| `Runtime Error` | 设备初始化、workspace、越界、同步 | 本机复现/缩小 shape |
| `Wrong Answer` | 逐用例、容差、边界、tie/填充 | 按平台 golden 修，不用本地结果反驳 |
| `TLE` | 死循环、同步、任务量 | 看 profile，不要盲目重提 |
| `System Error` | 平台/环境 | 保留 SID，稍后重查/联系赛方 |

## 8. 证据归档

```text
own/platform/
  sid.txt
  detail.json
  ranking.json
  submission.zip.sha256
  payload.json.sha256
```

日志/JSON 原样保存；不要只留摘要。

## 9. 红线

- 不在仓库/日志/提交包写 Cookie/Token；
- 不绕过隐藏测试数据、额度、重复限制；
- 不伪造 SID/榜单；
- 不把本地 `pass` 写平台 `Pass`；
- 不提交已知结构性错误的代码“碰运气”。
