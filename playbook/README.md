# 经验库入口

经验库回答：**为什么这样做、什么时候不成立、平台和本机冲突时听谁的。**

它不是 golden，也不是源码事实；每条经验都应能回到具体源码、日志、题面或平台 detail。

## 1. 先看规则

| 规则 | 解决的问题 |
|---|---|
| [平台精度与最终结果](rules/platform-evaluation.md) | 本地精度没过，是否要放弃高性能方案；平台最终结果怎么用 |
| [平台输入构造](rules/input-construction.md) | 没有隐藏测试数据时，怎么构造 shape/边界/图/点云输入 |
| [本机 vs 平台](rules/local-vs-platform.md) | 本地 910B3 失败/通过，哪些能外推，哪些不能 |
| [提交与回读](rules/submission-and-readback.md) | dry-run、SID、detail、ranking、额度、重复包 |

## 1.1 通用专题（先看这里）

按“我遇到的是什么问题”选，不按赛题选：

| 专题 | 解决的问题 |
|---|---|
| [小 case / launch-bound 优化](topics/small-shape-launch-bound.md) | 0.5–2 ms、不同 shape 时间接近常数，怎么减固定开销 |
| [团队赛取包与账号矩阵](topics/team-ranking-and-package-selection.md) | 团队排名为什么不在我的账号上；latest/best 怎么选 |
| [隐藏 case 泛化](topics/hidden-case-generalization.md) | 可见 case 高分，最终分为什么低 |
| [语义覆盖](topics/semantic-coverage.md) | 平台 Pass 了，题面的属性/模式是否都实现 |
| [Cube + Vector 分工](topics/cube-vector-split.md) | 什么时候用 Cube，什么时候纯 Vector |
| [对齐、尾块与 padding](topics/alignment-and-tail-handling.md) | 非对齐 K/N、尾块、max padding |

完整列表与写法约定见 [`topics/README.md`](topics/README.md)。

## 2. 跨实现经验

按主题读，不要按“某队用了这个技巧”直接照搬：

| 文档 | 主题 |
|---|---|
| [work_reduction_vs_fixed_overhead.md](../docs/experiences/work_reduction_vs_fixed_overhead.md) | 什么时候少算，什么时候多算但更规则 |
| [layout_and_buffer_lifecycle.md](../docs/experiences/layout_and_buffer_lifecycle.md) | AoS/SoA、消费者布局、双槽、`FIX_M` |
| [numerical_selective_precision.md](../docs/experiences/numerical_selective_precision.md) | 选择性高精度、half 粗筛、eps/顺序 |
| [shape_specialization_and_fallbacks.md](../docs/experiences/shape_specialization_and_fallbacks.md) | Host 分路、编译期 key、fail-closed |
| [iterative_linear_algebra_and_fusion.md](../docs/experiences/iterative_linear_algebra_and_fusion.md) | 批量矩阵交错、轮内配对、单 kernel 融合 |
| [equivariant_and_mesh_operators.md](../docs/experiences/equivariant_and_mesh_operators.md) | 分组 GEMM、有界 stencil、容量调度 |
| [gitcode_operator_mechanisms.md](../docs/experiences/gitcode_operator_mechanisms.md) | GQA K 复用、布局优先搬运、补偿算术 |
| [manufacturing_operator_edges.md](../docs/experiences/manufacturing_operator_edges.md) | 制造专场 10 个小点：比较符/tie/空段/槽位/`rot^T` |

## 3. 经验怎么写

每条至少包含：

```text
现象/瓶颈：
做法：
改变了什么（算术/搬运/同步/布局）：
适用条件：
负例/反例：
证据（源码行号/日志/O+V）：
验证设计（怎么证伪）：
```

没有负例和适用条件的“经验”，只能写成假设，不能写成规则。

## 4. 证据边界

- 作者 `O+A` 不等于本机 `V`；
- 本机 `V` 不等于平台 `O`；
- 平台 `O` 也不能自动外推到另一个 SoC/CANN；
- 经验库不替代知识库的源码/哈希，也不替代工具库的执行日志。
