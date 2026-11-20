# 规则索引

这些规则是跨题目的默认判断，不替代具体题面。

| 规则 | 先读 |
|---|---|
| 平台精度是最终结果，但本地精度不是一票否决 | [platform-evaluation.md](platform-evaluation.md) |
| 没有隐藏数据时怎么构造合法/边界/随机输入 | [input-construction.md](input-construction.md) |
| 本机 910B3 与平台 910B 的差异、哪些能外推 | [local-vs-platform.md](local-vs-platform.md) |
| dry-run、SID、detail、ranking、额度、重复包 | [submission-and-readback.md](submission-and-readback.md) |

使用顺序：

1. 先看题面/官方 tolerance（知识库）；
2. 按 `input-construction.md` 造本地用例；
3. 按 `local-vs-platform.md` 判断本地结果；
4. 按 `platform-evaluation.md` 决定是否带差异提交；
5. 按 `submission-and-readback.md` 走 dry-run/提交/回读。
