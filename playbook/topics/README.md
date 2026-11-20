# 通用经验专题

> 每个专题只回答一类问题，控制在半页到一页。规则是“默认判断”，专题是“具体怎么做”。
> 每条都带适用条件、负例和证据；没有负例的结论只能算假设。

## 专题

| 专题 | 回答的问题 | 典型证据 |
|---|---|---|
| [小 case / launch-bound 优化](small-shape-launch-bound.md) | 0.5–2 ms、不同 shape 时间几乎一样时，怎么优化？ | Sparse Softmax 2/161；mHcPre |
| [团队赛取包与账号矩阵](team-ranking-and-package-selection.md) | 团队排名为什么不在我的账号上？latest/best 怎么选？ | mHcPre 团队第 18 来自队友 `gao` |
| [隐藏 case 泛化](hidden-case-generalization.md) | 可见 case 1.1x，为什么最终分只有 58？ | GroupedMatmul 76.16→58.54；Final 81.30→73.77 |
| [语义覆盖](semantic-coverage.md) | 平台 Pass 了，题面的 RoPE/mask/变长/bf16 都实现了吗？ | DenseLightning 硬编码 hasRope=0；Sparse Softmax 缺 bf16 |
| [Cube + Vector 分工](cube-vector-split.md) | 什么时候用 Cube，什么时候纯 Vector？AIC/AIV 怎么排？ | BatchMatmul、GroupedMatmul、Final、mHcPre |
| [对齐、尾块与 padding](alignment-and-tail-handling.md) | 非对齐 K/N、尾块、max padding 怎么处理？ | BatchMatmul、GroupedMatmul、Sparse Softmax |

## 怎么用

1. 先读 `../rules/` 判断平台/本机/提交边界；
2. 再按问题选一个专题；
3. 专题里的“证据”只说明某题在某平台的表现，不是通用 golden；
4. 新经验先写成专题草稿，至少补一个负例/反例，再进 `docs/experiences/` 的深度长文。

## 不做什么

- 不写“万能优化清单”；
- 不把某个赛题的精确 shape 分支当通用做法；
- 不把平台分数/本机时间混成同一个证据等级；
- 不写没有适用条件的“技巧”。
