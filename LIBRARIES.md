# 三层库：知识库、经验库、工具库

本仓库不是“两个算子库 + 一个 CANNJudge 工具”，而是三层：

| 层 | 目录 | 存什么 | 不存什么 |
|---|---|---|---|
| **知识库** | [`knowledge/README.md`](knowledge/README.md)；底层 `competition/`、`general/`、`index/`、`competition/mfg-2026/official/` | 源码快照、固定 commit/哈希、算子语义、官方题面/规格、条目 README、机器索引/阅读卡 | 不存放未验证的“经验结论”；不把 `pass` 当平台通过 |
| **经验库** | [`playbook/README.md`](playbook/README.md)；底层 `docs/experiences/`、`competition/*/analysis/`、`docs/validation/`、`docs/research/` | 跨实现机制、负例、阈值/crossover、复盘、验证记录、调研结论 | 不替代源码事实；作者 `O+A` 不写成本机 `V` |
| **工具库** | [`toolkit/README.md`](toolkit/README.md)；底层 `tools/`、`tools/competition/`、`skills/` | 索引生成、归档校验、CANNJudge 客户端/打包/提交、agent 工作流 | 不保存 Cookie、账号、隐藏测试数据；不自动 POST |

一句话：

- 知识库回答**“有什么、是什么、在哪”**；
- 经验库回答**“为什么这样做、什么时候不成立”**；
- 工具库回答**“怎么生成、怎么验证、怎么提交”**。

## 1. 知识库

```text
competition/                 竞赛知识：赛事、公开 PR、源码映射
  mfg-2026/                  制造专场 15 题
    official/                官方题面、specs、problems.json、descriptions
    <entry>/README.md        语义、机制、差异、复用判断
    <entry>/source.json      固定 commit、路径、行号、SHA256
    <entry>/source/          选中源码快照
general/                     通用知识：CANNBot、SVD、Mamba、vLLM-HUST 等
index/
  catalog.json               主索引（仓库/实现/官方题面/证据）
  cards.md                   阅读卡
  by-family.md               按算子族
  by-evidence.md             按本机证据
  by-source-match.md         制造专场按 direct/adapt/reference/none
  by-cannjudge.md            制造专场 × CANN Judge 交叉索引
```

`by-cannjudge.md` 是**知识库索引的一个视图**，不是第四个库；它把官方题面、题目 ID/测试点数、源码分级和本机状态放在一张表里。

## 2. 经验库

```text
playbook/topics/              按问题选的一页专题：小 case、团队取包、隐藏 case、语义覆盖、Cube/Vector、对齐尾块
playbook/rules/               平台精度、输入构造、本机 vs 平台、提交回读
docs/experiences/
  work_reduction_vs_fixed_overhead.md
  layout_and_buffer_lifecycle.md
  numerical_selective_precision.md
  shape_specialization_and_fallbacks.md
  iterative_linear_algebra_and_fusion.md
  equivariant_and_mesh_operators.md
  gitcode_operator_mechanisms.md
  manufacturing_operator_edges.md
docs/research/                外部来源调研、算法参考核对
docs/validation/              本机 910B3 验证总账
competition/*/analysis/       赛事复盘、公开源码分析、平台证据
```

经验库的每条结论要能回到源码/日志/题面；不能回到证据的“经验”只能作为待验证假设。

## 3. 工具库

```text
tools/
  build_index.py              生成 catalog/cards/source.json/by-*
  build_mfg_official.py       生成 specs.md/problems.md/by-cannjudge.md
  verify_archive.py           哈希、JSON、Python、链接、敏感模式校验
  competition/                CANNJudge 便携起步包（client/package/submit/docs）
skills/                       agent 工作流：检索、入库、910B 验证
```

ACW 环境另有：

```text
bin/cann-client              查询赛事/题目/模板/历史/榜单
bin/cann-package             按模板生成 zip/payload，不 POST
bin/cann-submit              dry-run 默认；POST 需 ACW_ALLOW_SUBMIT=1
tools/cannjudge/             对应的 Python 实现
```

## 4. 三层的流转

```text
CANNJudge 公开题面/测试点数
        │
        ▼
知识库：official/ → catalog.official_problem → source_match/local_status
        │
        ▼
经验库：adapters、reference、负例、crossover、验证记录
        │
        ▼
工具库：build_index → 本机验证 → cann-package → cann-submit dry-run
        │
        └────────────► 平台 detail/ranking → 回填知识库/经验库
```

## 5. 边界规则

1. **索引不是库**：`index/` 下的 `by-*.md`、`cards.md` 都是视图；`by-cannjudge.md` 也一样。
2. **证据不是结论**：`evidence/` 是条目的证据附件；`docs/validation/` 是跨条目验证记录，不替代条目 `local_status`。
3. **官方不是本机**：CANNJudge 的 `O`（题面/榜单）与本机 `V`（910B3 日志）分开写。
4. **工具不产生事实**：`cann-submit` 只执行提交；是否通过以平台 `detail/ranking` 为准。
5. **经验必须可证伪**：每条经验至少给出适用条件、负例或验证方法。
6. **不提交私有数据**：Cookie/账号/隐藏测试数据不进入三层库。

## 6. 当前规模

```text
知识库：24 个仓库记录、39 份实现、39 张阅读卡、21 份官方题面（制造 15 + 上合/星辰 6）
经验库：6 个通用专题 + 7 篇跨实现经验 + 制造 10 个小点 + 本机验证总账
工具库：索引/校验/官方题面/CANNJudge 客户端/打包/提交/agent skills
```

详细架构见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)；迁移记录见 [`docs/MIGRATION_20261117.md`](docs/MIGRATION_20261117.md)。
