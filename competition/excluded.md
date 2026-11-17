# 竞赛库淘汰记录

本页只记录“看起来像竞赛来源、但没有形成高性能算子条目”的仓库。机器记录仍在 [../index/catalog.json](../index/catalog.json)，`status=excluded`。

## Yunle-Lee / CANN-Operator-test-competition-personal-solution

| 项 | 值 |
|---|---|
| URL | https://github.com/Yunle-Lee/CANN-Operator-test-competition-personal-solution |
| 固定 commit | `a0c1c8a8ba8ac203afe28f2f7d9c75b9c06ab595` |
| 上下文 | CANN 算子测试挑战赛（2026） |
| 许可 | MIT |
| 内容 | 主要是 Add/Mul/Pow/Cumsum 的 aclnn 测试代码和赛事目录 |

淘汰原因：

- 没有 Ascend C 高性能 kernel，也没有逐 case 时延；
- 更像测试脚手架/参赛模板，不足以支撑“高性能算子实现”判断；
- 没有可提炼的 tiling/布局/数值机制。

保留价值：如果未来需要 aclnn 测试框架，可以作为参考；不计入竞赛实现数量。

## 其它未计入的竞赛候选

- 只有题面、报名页、榜单截图，没有公开源码/提交的赛事目录；
- fork、镜像、二次搬运的 PR（与已登记来源重复）；
- 只有测试用例、没有 kernel 的“参赛模板”。

这些不单独建条目；如后续拿到固定 commit 和源码，再按 [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) 的条目契约重新评估。
