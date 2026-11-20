# 公开访问与安全边界

## 1. 读写模型

| 操作 | 是否需要授权 |
|---|---|
| 浏览 GitHub 网页、`raw.githubusercontent.com`、匿名 `git clone` | 否 |
| 匿名 GitHub API | 否，但有速率限制（约 60 次/小时） |
| `git push`、修改仓库设置、删除分支、改可见性 | 是；需要 collaborator 权限或 token |

仓库公开不会让任何人获得写权限。GitHub 的权限模型仍按账号/团队控制；本库没有把 token、cookie、`hosts.yml` 或账号文件放进 Git。

## 2. 本库不包含什么

- 不包含 CANNJudge 的 cookie/token、GitCode token、SSH 私钥或云厂商密钥；
- 不包含隐藏测试数据或 `/api/testcases/<id>` 的绕过结果；
- 不包含本机 910C 结论；910B3 结果只在 `docs/validation/` 和条目 `evidence/`；
- 不把作者/平台成绩写成本机复现；
- mHcPre 的源码/候选包未进入公开树和 `main` 历史，只保留团队排名、SID、分数和 `competition/excluded.md` 说明。

## 3. 第三方内容

- `competition/grouped-matmul-gelu/prs/` 是公开 PR 快照，保留原始 LICENSE/PROVENANCE；
- `competition/mfg-2026/*/source/` 是 CANN/DrivingSDK 固定 commit 片段，许可证见 [`competition/mfg-2026/THIRD_PARTY_LICENSES.md`](competition/mfg-2026/THIRD_PARTY_LICENSES.md)；
- 未声明许可证的仓库只做引用/行号分析，源码不进入本库；
- 本库没有统一许可证；公开可读不等于允许再许可。再分发第三方片段时请保留上游版权头和许可证。

## 4. 证据边界

平台排名、SID、逐 case 时间来自 CANNJudge 只读查询；源码来自对应提交详情或用户提供的 zip。没有在本机重跑的条目标为 `platform-only` 或 `not-run`。README 中的“平台 Pass”只代表平台评测算例，不代表题面全集已实现。

## 5. 报告问题

如果发现误公开的凭据、个人数据或许可问题，请通过 GitHub issue 或 `@hfutslowpoke` 联系维护者，并注明文件路径。不要在本库提交真实 token、cookie 或隐藏测试数据。
