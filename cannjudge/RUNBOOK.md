# CannJudge 操作手册

本手册按“每天开始 → 提交前 → 提交 → 提交后 → 异常”的顺序执行。
规则细节见 [RULES.md](RULES.md)，工具参数见 [TOOLS.md](TOOLS.md)，载荷见 [PAYLOAD_CONTRACT.md](PAYLOAD_CONTRACT.md)。

## 0. 每天开始

```bash
cd ascend-operator-notes
source cannjudge/cann.env.example

# 当前 12:00 窗口，所有账号
python3 cannjudge/tools/cann_quota.py --contest gcc_2026_mfg

# 单账号 + 逐题明细
python3 cannjudge/tools/cann_quota.py --contest gcc_2026_mfg --account B --details
```

确认：

- [ ] 窗口起点/终点是北京时间 12:00；
- [ ] `total/effective/remaining` 覆盖该账号所有题目；
- [ ] `last_submission` 距现在足够久（经验至少 30 分钟，保守 60 分钟）；
- [ ] 账号 Cookie 未过期（`accounts` 只显示 label/UID/过期时间）。

> 不要按自然日 00:00 统计；不要把 429 直接当成额度耗尽。

## 1. 提交前 10 项检查

1. [ ] 题目 ID 正确；
2. [ ] 源码目录是当前候选，不是旧 build；
3. [ ] 本地 `test_local` / 扩展对拍通过；
4. [ ] 只改了一个主要因素，且有 baseline；
5. [ ] `source_digest` 与预期一致；
6. [ ] 同 digest 没有已受理记录；
7. [ ] 当前窗口剩余额度 > 0；
8. [ ] 上次提交时间足够久；
9. [ ] dry-run `payload_bytes < 609980`，只含 editable 文件；
10. [ ] 用户已明确授权 POST（或任务书明确授权）。

## 2. dry-run

```bash
python3 cannjudge/tools/cann_submit.py \
  --problem-id <pid> \
  --account B \
  --source-dir /path/to/project \
  --tag <problem>-<round>-<hypothesis>
```

检查 `plan.json`：

```bash
python3 -m json.tool .cann-submit/<pid>/<digest>.plan.json | less
```

- `files[]` 是否只含 editable；
- `payload_bytes` 是否超限；
- `compat_fields` 是否符合预期；
- `quota_window_start/end` 是否是当前 12:00 窗口；
- `source_digest` 是否与本地 `sha256` 一致。

## 3. 正式提交

```bash
ACW_ALLOW_SUBMIT=1 python3 cannjudge/tools/cann_submit.py \
  --problem-id <pid> \
  --account B \
  --source-dir /path/to/project \
  --tag <tag> --submit
```

规则：

- 只 POST 一次；
- POST 后只 GET 轮询；
- 第一件事是保存 SID；
- 不因为“还没出结果”重复 POST；
- 不用 `--allow-duplicate` 绕过冷却。

## 4. 提交后

```bash
# 终态/逐 case
python3 cannjudge/tools/cann_client.py detail <SID> --account B > detail.json

# 榜单/tbest
python3 cannjudge/tools/cann_client.py ranking <pid> --account B > ranking.json

# 逐 case 分数/headroom
python3 cannjudge/tools/cann_score.py \
  --detail detail.json --ranking ranking.json --target <target>
```

记录：

- SID、状态、提交时间；
- 逐 case time/tbest/status；
- 当前总分/名次/前三；
- 额度窗口；
- `keep` / `revert` / `iterate` / `stop`；
- 下一条可执行动作。

## 5. 429

```text
1) 查 12:00 窗口额度；
2) 看 last_submission 距现在多久；
3) 保存 429 body/headers；
4) 停止，不连续重试；
5) 不换账号绕过；
6) 等冷却或人工确认。
```

不要：

- 每 5 分钟重试；
- 用另一个账号提交同一源码来绕过冷却；
- 改 tag/包名重新 POST；
- 在 `ambiguous_do_not_retry` 时盲目重发。

## 6. POST 无 SID（ambiguous）

如果 POST 返回成功但没有 `submissionId`：

```text
status = ambiguous_do_not_retry
```

处理：

1. 不要重发；
2. 查该账号该题 history；
3. 按源码摘要/时间/题目匹配；
4. 如果找到 SID，补记；
5. 如果找不到，记录为 `ambiguous`，人工对账；
6. 下一次提交前先确认额度/冷却。

## 7. 平台失败 triage

| 状态 | 先做什么 | 不要做什么 |
|---|---|---|
| `Wrong Answer` | 保留首个错误位置/形状；本地最小复现 | 先调性能 |
| `Runtime Error` | 保留错误码/失败起始 case；本地同 shape smoke | 整包重写 |
| `Compile Error` | 本地同 CANN 版本/同模板编译 | 只改代码不看版本 |
| `TLE` | 看是否单 case 爆炸；先做时间分解 | 盲目加大 tile |
| `Skipped` | 看前一个失败 case | 当成 Pass |
| `Judge Interrupted` | 查 history；可能不计额度 | 立即重发 |

## 8. 回退

回退到最后一个平台 Pass 的版本：

```text
SID / source digest / kernel sha256
```

步骤：

1. 从源码归档/候选目录恢复；
2. 本地 `test_local` + 扩展对拍；
3. 平台 dry-run 确认 digest；
4. 只提交一次；
5. 记录回退原因和对比。

## 9. 账号/凭据

- 账号文件权限 0600；
- 不打印 Cookie；
- 不把账号文件放进仓库/日志/聊天；
- Cookie 过期用 `set-cookie` 更新；
- 一个账号一个额度池；
- 不假设“换账号就能绕过限流”。

## 10. 每日结束

- [ ] 记录当天所有 SID/状态/分数；
- [ ] 更新当前最优与 fallback；
- [ ] 记录 429/冷却/额度；
- [ ] 写下一条可执行动作；
- [ ] 不把未验证的本地结果写成平台成绩；
- [ ] 不留下未对账的 POST。

## 11. 相关

- [RULES.md](RULES.md)
- [TOOLS.md](TOOLS.md)
- [PAYLOAD_CONTRACT.md](PAYLOAD_CONTRACT.md)
- [SECURITY.md](SECURITY.md)
- [DEVELOPMENT.md](DEVELOPMENT.md)
