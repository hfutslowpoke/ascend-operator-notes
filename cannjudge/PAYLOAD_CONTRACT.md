# 模板与载荷契约

本页说明 `cann_submit.py` 如何从平台模板构造提交载荷。
平台行为可能变化；以平台返回的模板和 `plan.json` 为准。

## 1. 模板

提交前，工具向平台请求该题/该账号的模板：

```text
GET /problems/<problemId>/template?...
```

模板返回 `files[]`，每项至少包含：

| 字段 | 含义 |
|---|---|
| `path` | 仓库内相对路径，例如 `op_kernel/foo.cpp` |
| `editable` | 是否允许提交；只有 `true` 的文件进入载荷 |
| `content` | 模板默认内容（可能为空） |

规则：

- 只提交 `editable=true` 的文件；
- 非 editable 文件（构建脚本、模板声明等）不进入载荷；
- `--all-files` 只用于本地检查，禁止与 `--submit` 同用；
- 本地路径必须与模板 `path` 一致；
- 用 `--file path=local` 覆盖某个模板路径时，`local` 可以是相对 `--source-dir` 或绝对路径。

## 2. 源码读取

对每个选中的模板文件：

1. 读取本地文件；
2. 必须是 UTF-8 文本；
3. 不能包含 NUL 字节；
4. 记录 `bytes` 与 `sha256`；
5. 内容进入载荷。

失败时直接退出，不提交部分文件。

## 3. 载荷

载荷是 JSON，包含文件路径/内容/兼容字段。
平台上限：

```text
PAYLOAD_LIMIT = 609980 bytes
```

工具在 dry-run 的 `plan.json` 里记录：

```json
{
  "problem_id": "<pid>",
  "account": "<label>",
  "tag": "<tag>",
  "source_digest": "<sha256>",
  "payload_bytes": 12345,
  "compat_fields": [],
  "files": [
    {
      "path": "op_kernel/foo.cpp",
      "bytes": 1234,
      "sha256": "<sha256>"
    }
  ],
  "submit": false
}
```

`source_digest` 的计算：

```text
for each file sorted by path:
  sha256(path + "\0" + content + "\0")
```

同一 `source_digest` 视为同一源码；默认拒绝重复 POST。

## 4. 旧模板兼容字段

一些旧 host/kernel 模板要求额外字段。`--compat auto` 会按路径猜测并填充：

| 字段 | 匹配 |
|---|---|
| `host_cpp` | `op_host/*.cpp` |
| `kernel_cpp` | `op_kernel/*.cpp` |
| `tiling_key_h` | 路径含 `tiling_key` 的 `.h` |
| `tiling_h` | 路径含 `tiling` 且不含 `tiling_key` 的 `.h` |

只在模板确实要求时使用；`compat_fields` 会写入 `plan.json`，提交后要回读 detail 确认。

## 5. dry-run 检查清单

`plan.json` 必须确认：

- [ ] `problem_id` 正确；
- [ ] `account` 正确；
- [ ] `tag` 可识别（题目/轮次/假设）；
- [ ] `source_digest` 与预期一致；
- [ ] `payload_bytes` < 609980；
- [ ] `files[]` 只含 editable 文件；
- [ ] 每个文件的 `sha256` 与本地一致；
- [ ] `compat_fields` 为空，或确实是旧模板需要；
- [ ] 额度窗口剩余 > 0；
- [ ] 同 digest 没有已受理记录；
- [ ] 用户已明确授权 POST。

## 6. 提交后

- 保存 `accepted.json`（SID、状态、429 原文）；
- 轮询到终态；
- 保存 `detail.json`（逐 case time/tbest/status）；
- 用 `cann_score.py` 计算分数/headroom；
- 更新 `ledger.jsonl` 和 case 记录；
- 不重复 POST 同一 digest，除非明确 `--allow-duplicate` 且写清原因。

## 7. 禁止

- 不提交非 editable 文件；
- 不提交二进制/非 UTF-8/含 NUL 的文件；
- 不提交超过 609980 字节的载荷；
- 不把 Cookie、账号文件、构建产物、`.npy`/`.bin` 放进源码目录；
- 不修改模板路径来“绕过”缺失文件；
- 不在没有 dry-run 的情况下 POST。

## 8. 相关

- [TOOLS.md](TOOLS.md)
- [RUNBOOK.md](RUNBOOK.md)
- [cannjudge/tools/cann_submit.py](tools/cann_submit.py)
