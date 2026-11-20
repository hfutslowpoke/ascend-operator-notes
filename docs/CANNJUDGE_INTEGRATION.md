# CANNJudge 与两层算子库的联合视图

本库其实有三块内容，之前容易分开看：

1. **两层算子库**：`competition/`（竞赛实现）与 `general/`（通用/CANNBot 来源）；
2. **CANNJudge 官方层**：`competition/mfg-2026/official/` 的公开题面、结构化 specs、题目 ID/测试点数；
3. **CANNJudge 工具层**：仓库内 `tools/competition/` 的可移植起步包，以及 ACW 根目录 `tools/cannjudge/` + `bin/cann-*` 的在线客户端/提交器。

三者不是三套独立资料，而是一条链：

```text
CANNJudge 赛事/题目 API
        │  公开题面、slug、problemId、测试点数、容差
        ▼
competition/mfg-2026/official/  ──►  index/catalog.json 的 official_problem
        │                                   │
        │                                   ├─► source_match（direct/adapt/reference/none）
        │                                   ├─► local_status（910B3 证据）
        │                                   └─► 条目 README / source.json
        ▼
adapters.md / reference/ / evidence/  ──►  910B3 本机验证
        │
        ▼
cann-package → cann-submit dry-run → （用户授权）--submit → detail/ranking
        │
        └──────────────► 回填 evidence、catalog、复盘（闭环）
```

## 1. 每层回答什么问题

| 问题 | 先看 | 说明 |
|---|---|---|
| 这题官方到底要什么？ | `competition/mfg-2026/official/specs.md`、`descriptions/` | shape/dtype/属性/约束/容差，来自 CANN Judge 公开 API |
| 题目 ID、slug、测试点数？ | `index/by-cannjudge.md`、`official/problems.json` | 15 题的 `problemId`、canonical name、total/ranking 测试点 |
| 有没有可复用源码？ | `index/by-source-match.md`、各条目 README | `direct/adapt/reference/none/building-block` |
| 现有源码差在哪？ | `competition/mfg-2026/problems.md`、`adapters.md` | 逐题差异和改造步骤 |
| 怎么验证边界？ | `reference/`、`evidence/`、`skills/ascend-910b-local-validation/` | CPU 语义参考 + 910B3 日志 |
| 怎么提交？ | `tools/competition/` 或 `bin/cann-*` | dry-run 默认，POST 需显式授权 |
| 平台结果怎么回填？ | `detail`/`ranking` + evidence | 保留 SID、逐用例时间、失败原因 |

## 2. 制造专场联合地图

[`index/by-cannjudge.md`](../index/by-cannjudge.md) 是机器生成的联合表，列包括：

- CANN Judge 题目/URL、slug、题面；
- 官方分级（direct/adapt/reference/none）；
- 候选实现（候选级别 + 本机状态）；
- 测试点数（总/排名）。

当前快照：

```text
contest_id: 6aed4926a5348dcef5dd8d0c
title:      昇腾AI行业创新大赛——2026制造行业算子专场（预赛）
problems:   15
分级:       2 direct / 6 adapt / 5 reference / 2 none
本机:       制造专场主实现仅 FFT 32×32 子集 PASS；其余 not-run
补充:       SegmentMinCsr 骨架 BUILD+RUN PASS（示例，非官方语义）
参考:       CPU 语义测试 9 项 PASS（非 NPU）
平台提交:   本库未记录制造专场 POST
```

## 3. CANNJudge 工具层怎么用

### 3.1 发现题目

便携包在仓库内：

```bash
python3 tools/competition/tools/cann_client.py resolve gcc_2026_mfg
python3 tools/competition/tools/cann_client.py problems gcc_2026_mfg
```

ACW 环境有包装入口：

```bash
bin/cann-client resolve gcc_2026_mfg
bin/cann-client problems gcc_2026_mfg
```

`resolve`/`problems` 是公开信息；模板、历史、详情、榜单需要账号 Cookie。

### 3.2 打包与 dry-run

```bash
# 只生成可复现 zip + manifest，不 POST
bin/cann-package \
  --problem-id <PID> --account A \
  --source-dir ./my_submission --out submission.zip

# 默认只校验：模板路径、文件 SHA、payload、重复/额度
bin/cann-submit \
  --problem-id <PID> --account A \
  --source-dir ./my_submission --tag mfg-r1
```

### 3.3 真正提交

只有用户明确授权后才 POST：

```bash
ACW_ALLOW_SUBMIT=1 bin/cann-submit \
  --problem-id <PID> --account A \
  --source-dir ./my_submission --tag mfg-r1 --submit
```

`bin/cann-submit` 会在没有 `ACW_ALLOW_SUBMIT=1` 时拒绝 `--submit`；这是硬门禁，不是提示。

### 3.4 回读结果

```bash
bin/cann-client history <PID> --account A
bin/cann-client detail <SID> --account A
bin/cann-client ranking <PID> --account A
```

回读后应回填：

- `status`/`valid`/`result[]`；
- 逐用例 `Task Duration`、频率、Block Dim；
- 失败码（如 `507057`）和是否为本机/CANN 版本差异；
- SID、源码 SHA、提交时间、是否重复。

## 4. 工具层与规则层

CANNJudge 的规则摘要见 `tools/competition/docs/06-compliance-and-redlines.md`。与本库最相关的几条：

- 不能按具体测试用例/hash/白名单分支；按 shape/tiling key 通用分派可以，但域内任意输入必须正确；
- Host 只能做 tiling/workspace/memcpy/launch/校验，不能做数值计算；
- 不能结果重放、空 kernel、纯搬运；
- 特化分支必须有真实可用的 fallback；
- 最终提交不能留调试返回/针对 case 的注释。

本库的 `adapters.md` 和 `reference/` 就是为“域内正确 + 可验证”服务的；`reference/` 只用于自测，不参与评分路径。

## 5. 安全边界

- Cookie/Token 只放 `~/.config/cannjudge/accounts.json` 或 `var/private/`，权限 `0600`；不进入仓库、日志、提交包。
- 不绕过隐藏测试数据；`/api/testcases/<id>` 返回 403，本库没有尝试绕过。
- 不把本地 PASS 写成平台通过；不伪造 SID/榜单。
- 不把 950/arch35 结果写成 910B；不把作者 `O+A` 写成本机 `V`。
- 提交默认 dry-run；POST 必须显式授权。

## 6. 推荐闭环

```text
1. CANNJudge 拉题面/测试点数
2. 在 `index/by-cannjudge.md` 找 slug → 官方分级 → 候选实现
3. 读 adapters/reference，写适配层
4. 910B3 本机验证（npu-lock）
5. cann-package 生成包
6. cann-submit dry-run，核对路径/SHA/payload
7. 用户授权后 submit；拿 SID
8. detail/ranking 回读；更新 evidence/catalog
```

这条链的“事实源”分工：

- **官方题面/测试点数**：CANNJudge 公开 API；
- **源码/哈希**：GitCode 固定 commit；
- **本机结果**：910B3 日志；
- **平台结果**：CANNJudge 提交详情/榜单。

四者不能互相替代；这也是本库一直保留 `S+L`、`O+A`、`V`、`N` 分级的原因。

## 7. 入口汇总

- 联合索引：`index/by-cannjudge.md`
- 官方规格：`competition/mfg-2026/official/specs.md`
- 逐题差异：`competition/mfg-2026/problems.md`
- 适配蓝图：`competition/mfg-2026/adapters.md`
- CPU 参考：`competition/mfg-2026/reference/`
- 本机证据：`competition/mfg-2026/evidence/`
- 平台工具：`tools/competition/README.md`、ACW `bin/cann-*`
- 规则摘要：`tools/competition/docs/06-compliance-and-redlines.md`
