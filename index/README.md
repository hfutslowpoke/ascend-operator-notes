# 统一索引

`index/catalog.json` 是本库唯一的机器可读主索引。ACW、agent 或脚本应优先读它，而不是解析 Markdown。

## 文件

| 文件 | 用途 |
|---|---|
| [catalog.json](catalog.json) | 20 个仓库记录、35 份实现、2 个淘汰记录；含 `tier`、`entry_path`、固定 commit、源码路径/行号/哈希、作者证据、适用条件、负例、复用判断、本机验证、制造题官方规格/`source_match` |
| [entry-schema.json](entry-schema.json) | 条目/来源字段契约；新增字段先改 schema |
| [by-family.md](by-family.md) | 按算子族检索（生成） |
| [by-evidence.md](by-evidence.md) | 按本机证据状态检索（生成） |
| [cards.md](cards.md) | 35 张人读阅读卡（生成） |
| [cards.json](cards.json) | 35 张机器可读阅读卡（生成） |
| `../competition/catalog.json` | 竞赛层过滤视图（生成） |
| `../general/catalog.json` | 通用层过滤视图（生成） |
| 各条目 `source.json` | 单条目的固定来源与实现清单（生成）；`supporting_sources` 是补充来源，不当官方候选 |
| `../competition/mfg-2026/official/specs.json` | 2026 制造专场 15 题官方规格（shape/dtype/属性/约束/容差） |

## 当前规模

| 层 | 仓库记录 | 深读 | 实现 | 淘汰 |
|---|---:|---:|---:|---:|
| `competition` | 11 | 10 | 26 | 1 |
| `general` | 9 | 5 | 9 | 1 |
| 合计 | 20 | 15 | 35 | 2 |

> `competition/grouped-matmul-gelu` 是 26 份公开 PR 的历史基线，不计入 35 份新深读实现；它的源码/哈希索引在 `competition/grouped-matmul-gelu/catalog/`。

## 证据状态

`implementations[].local_status` 取值：

| 值 | 含义 |
|---|---|
| `pass` | 本机 910B3 构建/运行通过，数值检查通过 |
| `pass-negative` | 本机复现了作者/预期的负结果（例如手写 Triton 比基线慢） |
| `partial` | 部分 shape/路径通过，其余失败或未覆盖 |
| `fail` | 本机构建或运行失败 |
| `offline-only` | 只做了 CPU 离线测试，没有 NPU 构建/运行 |
| `not-run` | 未在本机构建/运行 |
| `pending` | CANNBot 等尚未接入 |

完整证据等级（`S/L/A/O/V/N`）见 [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md)。

## ACW 查询示例

```python
import json
cat = json.load(open("index/catalog.json"))
# 找通用库里所有线性代数实现
for impl in cat["implementations"]:
    if impl["tier"] == "general" and impl["operator_family"] == "svd":
        print(impl["id"], impl["entry_path"], impl["local_status"])
```

或直接按族浏览 [by-family.md](by-family.md)，按本机证据浏览 [by-evidence.md](by-evidence.md)。

## 新增/修改流程

1. 改 `index/catalog.json`（加仓库/实现/源码文件/证据/负例/本机结果）。
2. 新建或更新条目 `README.md`、`evidence/`。
3. 运行 `python3 tools/build_index.py` 生成层索引、`source.json`、`by-*.md`。
4. 运行 `python3 tools/verify_archive.py`，再 `git diff --check`。
5. 不要手改生成文件；不要在没有本地结果时把 `local_status` 写成 `pass`。
