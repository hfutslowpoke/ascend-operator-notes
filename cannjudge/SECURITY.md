# 安全边界

本工具包会接触平台 Cookie 和提交账本。以下规则不可协商。

## 1. 凭据

- Cookie/Token 只放在 `CANNJUDGE_ACCOUNT_FILE` 指向的文件；
- 文件权限 0600；
- 仓库、日志、提交包、聊天中都不出现 Cookie/Token；
- `cann_client.py accounts` 只打印 label/UID/过期时间；
- 不把 `accounts.json`、`.env`、`hosts.yml`、`*.pem`、`*.key` 提交进 git；
- 不使用来源不明的 Cookie/账号。

## 2. 日志

- 平台原始响应只摘录必要字段（SID、状态、逐 case、错误码）；
- 429 原文可以保存到本地账本，但提交进仓库前必须脱敏；
- 本地日志去掉用户名、绝对临时路径、内网地址；
- 不记录 Cookie、Authorization、完整 header；
- 错误信息里如果包含敏感值，先脱敏再打印。

## 3. 提交账本

`.cann-submit/` 只保存在本地，不提交进仓库。
账本可以包含：

- SID、tag、source digest；
- 文件路径/哈希/字节数；
- 平台状态、逐 case；
- 429 的状态/时间/必要 header。

不应包含：

- Cookie/Token；
- 账号文件全文；
- 平台原始响应全文（除非本地私有且已脱敏）；
- 构建产物/大文件。

## 4. 共享

- 共享账本前先脱敏；
- 不共享账号文件；
- 不共享带 Cookie 的终端输出/截图；
- 不在公开 issue/PR 里贴平台原始响应；
- 不把内部路径/用户名带进公开仓库。

## 5. 反滥用

- 429 后不连续重试；
- 不换账号绕过冷却；
- 不改包名/时间戳/标签重复 POST；
- 不伪造 SID/分数；
- 不自动批量提交；
- 不在没有用户授权时 POST。

## 6. 事件处理

如果怀疑凭据泄露：

1. 立即停止使用该账号；
2. 更新 Cookie/撤销会话；
3. 检查账本/日志是否包含敏感值；
4. 清理本地缓存；
5. 记录事件时间、影响范围、处理动作；
6. 不把泄露的凭据写进 issue/聊天。

## 7. 相关

- [DEVELOPMENT.md](DEVELOPMENT.md)
- [RUNBOOK.md](RUNBOOK.md)
- [PROVENANCE.md](PROVENANCE.md)
