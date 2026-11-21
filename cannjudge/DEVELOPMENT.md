# CannJudge 工具开发规范

本目录是仓库内的独立工具包，目标是：**默认安全、可离线测试、可审计、不会泄露凭据**。
它不是 ACW 框架的一部分，也不修改 ACW 或当前算子工程。

## 1. 环境

| 项 | 要求 |
|---|---|
| Python | 3.10+（本目录用 `list[...]`、`X | None`） |
| 依赖 | 平台工具用标准库。读取 YAML 绑定时需 PyYAML |
| 平台 | Linux；NPU 运行另需 `npu-lock` |
| 网络 | 只有 `cann_client.py` 的只读查询/提交需要网络；测试必须离线 |

## 2. 目录

```text
cannjudge/
├── README.md             # 入口
├── RULES.md              # 平台规则
├── TOOLS.md              # 工具参数
├── DEVELOPMENT.md        # 本文件
├── PROVENANCE.md         # 来源/修改/许可
├── PAYLOAD_CONTRACT.md   # 模板/载荷契约
├── RUNBOOK.md            # 操作手册
├── SECURITY.md           # 凭据/日志/共享边界
├── tools/                # 可执行工具
├── tests/                # 离线测试
└── examples/             # 快速开始
```

## 3. 代码约定

- 每个工具一个 `main()`，可直接 `python3 file.py ...`；
- 默认行为必须安全：查询只读，提交默认 dry-run；
- 真正 POST 需要显式 `--submit` + `ACW_ALLOW_SUBMIT=1`；
- 错误用非零退出码；不要 `except: pass`；
- 输出同时照顾人读和机器读：表格给人，`--json` 给脚本；
- 时间统一用带时区的 ISO-8601；额度窗口用北京时间；
- 不打印 Cookie、Token、账号文件全文；
- 不在代码里写死账号、UID、SID、题目 ID；
- 不把 `accepted` 当 `Pass`；只有终态才写成绩。

## 4. 新增工具

1. 先写用途、输入、输出、失败模式；
2. 默认 dry-run / 只读；
3. 纯函数尽量拆出来，便于离线测试；
4. 在 `tests/` 加至少一个不联网的测试；
5. 更新 `README.md` / `TOOLS.md` 的目录和命令；
6. 如果复制了 ACW 代码，更新 `PROVENANCE.md`；
7. 运行：

```bash
python3 -m unittest discover -s cannjudge/tests -v
python3 tools/update_manifest.py
python3 tools/verify_archive.py
git diff --check
```

## 5. 测试要求

- 测试不能读真实 `accounts.json`；
- 测试不能访问网络；
- 测试不能 POST；
- 使用 `tempfile.TemporaryDirectory()` 构造临时源码；
- 对纯函数测边界：空输入、非法 UTF-8、NUL、超限、重复摘要；
- 对额度窗口测 12:00 边界、UTC 转换、半开区间；
- 对 score 测 `time=tbest`、`1.5×`、`2.25×`、目标总分；
- 对 429/ambiguous 只测解析和状态，不发请求。

## 6. 分支与提交

- 分支：`cannjudge`（规则/工具），不要混入 `competition/` 实现改动；
- 一个提交只做一件事；
- 提交信息用 `feat(cannjudge): ...`、`fix(cannjudge): ...`、`docs(cannjudge): ...`；
- 不提交 `.cann-submit/`、`accounts.json`、`.env`、日志、构建产物；
- 修改规则时同时更新 `RULES.md` 和 `tests/`。

## 7. 评审清单

- [ ] 默认 dry-run；
- [ ] 显式提交确认；
- [ ] 账号级 12:00 额度；
- [ ] 429 不自动重试、不换账号；
- [ ] 同源摘要去重；
- [ ] SID 优先；
- [ ] `accepted` ≠ `Pass`；
- [ ] 不打印/不保存 Cookie；
- [ ] 测试离线；
- [ ] 文档与代码一致。

## 8. 已知基线

`python3 tools/verify_archive.py` 在本次补充前已有若干历史链接/缺失日志错误（旧 evidence 日志被 `.gitignore` 排除）。
新增内容必须不引入**新的**断链或敏感模式；提交前对比错误列表，不要把历史错误当成自己引入的。

## 9. 相关

- [README.md](README.md)
- [RULES.md](RULES.md)
- [TOOLS.md](TOOLS.md)
- [PAYLOAD_CONTRACT.md](PAYLOAD_CONTRACT.md)
- [RUNBOOK.md](RUNBOOK.md)
- [SECURITY.md](SECURITY.md)
