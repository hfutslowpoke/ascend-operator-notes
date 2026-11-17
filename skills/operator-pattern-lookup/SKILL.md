---
name: operator-pattern-lookup
description: 按瓶颈或算子族检索可复用的昇腾算子机制，并给出证据等级、适用条件、负例和 910B3 验证计划。
---

# Operator Pattern Lookup

## 何时使用

- 已知当前算子的瓶颈（MTE2、Vector、Cube、同步、UB 容量、尾部搬运、数值误差），想找别人怎么解；
- 已知算子族（GEMM/Grouped GEMM、SVD/eigh/Lstsq、Concat/Transpose、MoE、SSM、归一化、等变张量积），想快速定位候选实现；
- 需要判断某个“加速比”能不能信、能不能迁到 910B/910C。

## 输入

```text
operator_family: 例如 grouped-gemm / svd / concat / attention-recompute
bottleneck:      例如 mte2 / vector / cube / sync / ub-capacity / tail / numerical
shape_dtype:     目标 shape、dtype、layout（尽量具体）
target:          910b3 / 910c / unknown
constraints:     是否允许改布局、是否必须单 kernel、是否允许近似
```

## 步骤

1. **读机器索引**：`index/catalog.json`。
   - 按 `tier`（competition/general）、`operator_family`、`status`、`local_status` 过滤。
   - 不要只用 `operator` 字符串匹配；同一算子可能有不同语义。
2. **读人读索引**：`index/by-family.md`、`index/by-evidence.md`，确认候选条目和本机状态。
3. **打开条目**：`<entry_path>/README.md`，重点读：
   - “语义与约束”：是不是同一个数学对象、dtype/layout；
   - “机制”：它减少的是什么工作/等待，增加什么代价；
   - “适用条件与负例”：容量门槛、退化、版本敏感；
   - “复用判断”：优先复用/需要验证/暂不使用。
4. **核对源码**：`<entry_path>/source.json` 的 commit、路径、行号、SHA256；必要时打开原始 URL。README 与代码冲突时以代码为准。
5. **看证据**：`<entry_path>/evidence/` 和 `index/catalog.json` 的 `local_validation`。
   - `O+A` 只能作为作者/平台声称；
   - `V` 只代表本机 910B3；
   - `N` 不要补造。
6. **跨条目综合**：`docs/experiences/*.md` 里有跨实现机制（工作量削减、布局/缓冲生命周期、数值/选择性精度、形状专用化、迭代线性代数、等变/mesh、GitCode 三类机制）。
7. **输出验证计划**：每个候选机制给出一个可证伪实验（固定 shape、扫参数、记录时延/误差/失败码）。

## 输出模板

```markdown
## 结论
- 候选机制 A：来自 <entry>，证据 <S/L/A/O/V/N>，适用条件 ...
- 候选机制 B：...

## 不推荐
- 机制 C：原因（负例/容量/版本/证据不足）

## 910B3 验证计划
| 机制 | 固定输入 | 扫描变量 | 指标 | 失败判据 |
|---|---|---|---|---|
| A | ... | ... | event 时延、max abs | 没有收益或误差超阈值 |

## 证据边界
- 哪些是作者数字，哪些是本机数字；
- 910C 是否有数据（没有就写 N）。
```

## 边界

- 不把竞赛榜单/作者数字当本机结果；不把 910B3 结果外推 910C。
- 不直接复制第三方源码；需要改造时，复制到工作区外或新项目目录，记录父本 commit/哈希。
- 不忽略负例；如果条目只有成功记录，标为“证据不足”。
- 不把“机制相似”当成“可以直接替换”；先确认语义、dtype/layout、同步/容量约束。
