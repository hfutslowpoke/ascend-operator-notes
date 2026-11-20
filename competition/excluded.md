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

## 星辰杯 A 组 mHcPre（我们的提交，未入库）

| 项 | 值 |
|---|---|
| 赛事 | 中国电信星辰杯·高校 AI 算子开发挑战赛 A 组 |
| 题目 | mHcPre（`6a7c191ca52e0f540a89ca9e`） |
| 团队排名 | 18 / 106，65.01 |
| 团队最佳 SID | `6aba2b61694b590c3cca0422`（submitter `gao`） |
| C 个人 SID | `6aba1cdd694b590c3cbf2a78`（detail 496336，Pass，约 59.43） |
| C 个人最新 | `497262`，Time Limit Exceeded（latest 覆盖 Pass） |
| 候选 zip | `MhcPre_submission_497378.zip`，SHA256 `7685a9cc4d32c27bbde292c1e11bc92ad0e5a04893dc58df97dbcc35c609917c` |

未入库原因：

- 团队最佳源码对 C 账号返回 403，无法核对/复现；
- C 的源码优化一般（2 case 2.44/3.22 ms，best 1.82/2.48），不是高性能示例；
- zip 497378 只差 30 行 hPost 提前写回，没有平台计时，不能证明收益；
- 保留价值主要是“团队取包”“latest 覆盖”“小 K 纯 Vector”三个经验点，
  已写进 [playbook/topics/](../playbook/topics/)，不需要一个完整的知识库条目。

> zip 本体未随仓库公开；本页只保留文件名和 SHA256，避免把未验证的候选包当成正式提交。

## 其它未计入的竞赛候选

- 只有题面、报名页、榜单截图，没有公开源码/提交的赛事目录；
- fork、镜像、二次搬运的 PR（与已登记来源重复）；
- 只有测试用例、没有 kernel 的“参赛模板”。

这些不单独建条目；如后续拿到固定 commit 和源码，再按 [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) 的条目契约重新评估。
