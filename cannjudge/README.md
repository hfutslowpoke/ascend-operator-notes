# CannJudge 规则与提交工具

本目录只做两件事：

1. 记录我们对 CannJudge 提交规则、额度、冷却、状态和计分口径的**可复核结论**；
2. 提供**默认 dry-run、可审计、不会泄露 Cookie** 的提交与额度工具。

它不存放算子实现、不存放比赛源码、不自动提交。
算子知识仍在仓库的 `competition/`、`general/`、`index/`。

## 目录

```text
cannjudge/
├── README.md             # 本文件
├── RULES.md              # 额度、12:00 刷新、冷却、429、状态、计分
├── TOOLS.md              # 工具参数、账本格式、排错
├── DEVELOPMENT.md        # 环境、代码约定、测试、提交
├── PROVENANCE.md         # 来源、修改、哈希、许可边界
├── PAYLOAD_CONTRACT.md   # 模板/载荷契约、UTF-8/NUL、609980 上限
├── RUNBOOK.md            # 每日额度、提交、429、回退、对账
├── SECURITY.md           # Cookie/日志/账本共享边界
├── cann.env.example      # 只含路径/开关，不含 Cookie
├── tools/
│   ├── cann_client.py    # 只读查询：账号/赛事/题目/模板/历史/详情/榜单
│   ├── cann_quota.py     # 按 12:00->12:00 北京时间统计账号所有题目额度
│   ├── cann_submit.py    # 模板校验 + 去重 + 额度预检 + dry-run/提交 + SID 轮询
│   ├── cann_score.py     # 逐 case 分数、gain_2x、factor_to_50、目标差距
│   ├── cann_package.py   # 官方模板可编辑文件打包
│   ├── cann_eval.py      # 独立构建、校验、计时、PMU
│   ├── acl_probe.py      # ACL 设备及 stream 初始化探针
│   ├── check_env.sh      # 环境自检
│   └── npu_lock.sh       # 本机 NPU 串行锁
├── tests/
│   ├── test_quota_window.py
│   ├── test_score.py
│   └── test_submit_guards.py
└── examples/
    └── quickstart.sh
```

## 快速开始

```bash
cd ascend-operator-notes
source cannjudge/cann.env.example

# 1) 看所有账号当前额度（北京时间 12:00 刷新）
python3 cannjudge/tools/cann_quota.py --contest gcc_2026_mfg

# 2) 只看 B，带逐题明细
python3 cannjudge/tools/cann_quota.py --contest gcc_2026_mfg --account B --details

# 3) 提交前先 dry-run（不会 POST）
python3 cannjudge/tools/cann_submit.py \
  --problem-id <problemId> \
  --account B \
  --source-dir /path/to/project \
  --tag my-r1

# 4) dry-run 通过后再 POST
ACW_ALLOW_SUBMIT=1 python3 cannjudge/tools/cann_submit.py \
  --problem-id <problemId> \
  --account B \
  --source-dir /path/to/project \
  --tag my-r1 --submit
```

> 注意：上面的 `ACW_ALLOW_SUBMIT` 只是本项目沿用的显式确认变量；`cann_submit.py` 本身在 `--submit` 时要求它，防止误 POST。

提交后看逐 case 分数/headroom：

```bash
python3 cannjudge/tools/cann_score.py \
  --detail detail.json \
  --ranking ranking.json \
  --target 60
```

`detail.json` / `ranking.json` 用 `cann_client.py detail/ranking` 保存；`cann_score.py` 只读本地文件，不联网。

## 规则摘要

- 每个账号每天 **50 次**；
- 额度在**北京时间 12:00** 刷新，不是 00:00；
- 账号之间独立，冷却也独立；
- 每次提交独立，不按队伍共享；
- 计数必须覆盖该账号当天的**所有题目**；
- 429 不等于额度耗尽，先查 12:00 窗口；
- 429 后停止，不连续重试，不换账号绕过；
- 同一源码最多 2 次 POST；
- 平台 POST 后只 GET 回读，不重发。

详见 [RULES.md](RULES.md)。

## 安全边界

- 账号 Cookie 只在 `CANNJUDGE_ACCOUNT_FILE` 指向的 0600 文件里；
- 工具只打印 label/UID/过期时间，不打印 Cookie；
- 提交账本只保存 SID、源码摘要、文件哈希和平台状态；
- 默认 dry-run；只有显式 `--submit` + `ACW_ALLOW_SUBMIT=1` 才会 POST；
- 仓库不保存 `accounts.json`、`.env`、Cookie、Token 或构建产物。

## 与 ACW 的关系

ACW 的 `acw solo submit` 内部也调用这套提交逻辑，并把 SID、平台终态、逐 case 时间写入
`var/state/solo/<task_id>.json`。本目录是**可独立使用、可脱离 ACW 运行**的规则与工具版本。

## 文档

| 文档 | 用途 |
|---|---|
| [RULES.md](RULES.md) | 额度/12:00/429/状态/计分 |
| [TOOLS.md](TOOLS.md) | 工具参数/账本/排错 |
| [RUNBOOK.md](RUNBOOK.md) | 每日操作、提交前后、异常、回退 |
| [PAYLOAD_CONTRACT.md](PAYLOAD_CONTRACT.md) | 模板/载荷、UTF-8/NUL、609980 上限 |
| [DEVELOPMENT.md](DEVELOPMENT.md) | 开发规范、离线测试、提交 |
| [PROVENANCE.md](PROVENANCE.md) | 来源/哈希/许可边界 |
| [SECURITY.md](SECURITY.md) | Cookie/日志/共享边界 |

## 验证

```bash
python3 -m unittest discover -s cannjudge/tests -v
python3 tools/verify_archive.py
```
