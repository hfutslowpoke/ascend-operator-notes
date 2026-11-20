# 已结束比赛入库流程

当一场比赛结束、用户提供赛题/我们自己的代码后，按本流程入库。目标不是先写“经验总结”，而是先把**题面、源码、我们自己的实现、平台证据**固定下来。

## 1. 目录骨架

```text
competition/<contest-slug>/
├── README.md                  # 总览：题目映射、来源、我们自己的实现、平台结果
├── official/
│   ├── problems.json          # 公开 API/页面快照（题目 ID、测试点数）
│   ├── specs.json             # 结构化 shape/dtype/属性/约束/容差
│   ├── descriptions/          # 逐题完整题面
│   └── README.md              # 分级/关键校正/未获取项
├── <source-entry>/            # 每个开源仓一个子条目
│   ├── README.md
│   ├── source.json
│   └── source/                # 选中源码快照（按上游路径）
├── own/                       # 我们自己的实现（可选，多个算子可分目录）
│   ├── kernel/                # kernel/host/tiling
│   ├── driver/                # 本地驱动/测试
│   ├── evidence/              # 构建/运行日志、误差、时延
│   └── platform/              # SID、detail、ranking、提交包哈希
└── evidence/
    └── README.md              # 本机/平台证据总账
```

如果比赛已经有固定提交仓（如 `cann/elec-ops-inspection`），可以保留其目录结构；但 `official/`、`own/`、`evidence/` 的边界不要混。

## 2. 七步

### Step 1：抓公开题面

- 用公开页面/API，不绕过隐藏测试数据；
- 每题的 `canonical_name`/slug、`problem_id`、`total_testcase_count`、`ranking_testcase_count`、容差必须落库；
- 题面原文放 `official/descriptions/`，结构化摘要放 `official/specs.json`；
- 公开 API 里如果包含测试 shape/范围（如点云题），原样保留，不要改写成“经验”。

### Step 2：固定开源仓

- 每个仓固定 commit，不跟 HEAD；
- 记录 license、默认分支、访问日期；
- 只复制选中源码文件到 `source/`，保留上游路径；`source.json` 写 `path/lines/sha256/url/local_path`；
- 整仓不复制；需要整仓时放工作区外缓存，不提交。

### Step 3：把题目映射到源码

每题至少给出：

```text
official_problem: contest_id / problem_id / slug / total_testcases / ranking_testcases
source_match: direct / adapt / reference / none
candidate: 仓库 + 文件 + 行号
mismatch: 接口/layout/shape/dtype/后处理差异
decision: 直接复用 / 适配 / 只读算法 / 不用
```

`direct` 只给“语义一致 + 目标 SoC 路径存在”的实现；有 A2 路径但接口不同，写 `adapt`；只有 950/arch35 或算法 helper，写 `reference`。

### Step 4：归档我们自己的实现

- `own/kernel/`：kernel、host、tiling、CMake/构建脚本；
- `own/driver/`：本地测试驱动、输入生成、golden；
- `own/evidence/`：构建日志、运行日志、误差、时延、失败码；
- `own/platform/`：CANNJudge SID、detail JSON、ranking 快照、提交包 SHA；
- 每个文件写 SHA256；日志保留原始换行/失败信息，不改写。

### Step 5：写总览 README

`competition/<contest-slug>/README.md` 至少包含：

1. 赛事、时间、官方算力、评测方式；
2. 题目总表：题号、slug、官方分级、候选源码、我们自己的实现、本机/平台状态；
3. 来源仓表：commit、license、覆盖题目；
4. 我们自己的实现：路径、环境、平台 SID、成绩/失败；
5. 本机 910B3 证据：哪些跑过、哪些没跑、哪些失败；
6. 关键语义校正：接口/layout/tie/填充/归一化；
7. 负例/风险：架构、版本、shape、dtype、平台差异；
8. 下一步：待验证项、待修项、是否提交。

### Step 6：更新索引

- 在 `index/catalog.json` 增加仓库/实现/证据；
- `official_problem` 填 `contest_id/problem_id/canonical_name/testcase counts`；
- 运行：

```bash
python3 tools/build_index.py
python3 tools/build_mfg_official.py   # 仅制造专场；其它赛事按其生成器
python3 tools/build_index.py --check
python3 tools/verify_archive.py
git diff --check
```

### Step 7：清理

- 删除工作区外的临时 clone、构建目录、安装目录、下载缓存；
- 仓库只保留选中源码、日志、哈希、驱动；
- 如果日志被 `.gitignore` 忽略，加 reviewed exception 并确认 `git status` 里出现。

## 3. 每题检查表

```text
[ ] contest_id 与 problem_id 是目标比赛，不是同名题
[ ] slug/canonical_name 与官方一致
[ ] 官方题面/规格/容差已落库
[ ] 测试点数（总/排名）已记录
[ ] source_match 有依据（不是“看起来像”）
[ ] 候选源码 commit/路径/行号/SHA 固定
[ ] 我们自己的实现与公开源码分开
[ ] 本机日志有设备/CANN/命令/shape/dtype/误差/时延
[ ] 平台结果有 SID/detail/ranking；没有就写未提交
[ ] 负例/失败码保留
[ ] 没有 Cookie/账号/隐藏测试数据
[ ] catalog/index/manifest 已更新
```

## 4. 经验什么时候写

等 Step 1–6 完成后，再把跨题规律写进 `playbook/`：

- 平台精度 vs 本地精度；
- 输入构造方法；
- 本地失败是否足以否定方案；
- 机制/crossover/负例。

经验必须能回到具体题目的源码、日志或平台 detail；不能回到证据的，只写成待验证假设。
