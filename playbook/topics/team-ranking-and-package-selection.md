# 团队赛取包与账号矩阵

> 一句话：组队赛的排名通常取**队伍中任一队员的 latest 或 best 包**。只看一个账号，
> 会把“团队有成绩”误判成“没有成绩”；只盯 latest，又会把好包被调试包覆盖。

## 1. 两种聚合方式

| 模式 | 含义 | 风险 |
|---|---|---|
| `latest` | 队伍每个成员的最新提交参与聚合/排名 | 一次 TLE/调试包可能覆盖之前的 Pass |
| `best` | 队伍取历史最好提交 | 需要知道“最好”是分数最好还是时间最好 |

提交前必须确认题目的 `ranking_submission_mode`，不要假设。

## 2. 查排名时同时查 user 和 team

```text
user_id  → 个人提交历史
team_id  → 队伍排名（submitter 可能是别人）
```

排名行里通常同时有 `user`、`team`、`submission_id`。如果只看 `user_id`，会漏掉：

- 队友的包；
- 团队取 best 时别人的更高分；
- 团队取 latest 时别人的最新包。

## 3. 本次案例

- **mHcPre**：团队第 18、65.01 分来自 `gao` 的 `6aba2b61694b590c3cca0422`；
  C/D 都不在前 100。C 个人最新 497262 是 TLE，latest 覆盖了 496336 的 Pass。
- **星辰杯 A 其他题**：Sparse Softmax 的团队第 2 来自 `slowpoke`；
  DenseLightning 第 15、决赛第 6 来自 `feixiangtian16`。同一队伍，不同题由不同账号贡献。
- **上合**：初赛/决赛的排名行里团队名是 `slowpoke 的队伍`，与星辰杯的
  `hfutslowpoke 的队伍` 不同；跨赛事不要拿 team_name 硬匹配，优先用 team_id/成员 UID。

## 4. 做法

1. 建一张“成员 × 题目”的提交账本：账号、SID、时间、状态、分数、是否团队最佳。
2. 每次查排名，先按 `team_id` 找团队行，再回到成员账号找源码。
3. 团队最佳源码不可见时，**只记录排名/SID/分数**，不要拿自己的源码冒充团队最佳。
4. latest 模式下，正式包提交后不要再用调试包/未验证包覆盖；如果必须试，先确认能回滚。
5. 比赛结束前留一次“最终包”提交，并保存 SID、detail、ranking 快照。

## 5. 负例

- 只看 C 账号，会得出“mHcPre 没成绩”的错误结论；
- 只看个人最新提交，会忽略队友的更高分包；
- 把 `slowpoke 的队伍` 和 `hfutslowpoke 的队伍` 当成两个不相干队伍，会漏掉上合/星辰杯的关联；
- latest 模式下用 TLE 包覆盖 Pass，直接把个人排名打没。

## 6. 检查清单

```text
[ ] 是否查了 team_id/team_name，而不只是 user_id？
[ ] 是否记录了团队最佳包的 submitter 和 SID？
[ ] 是否确认 ranking_submission_mode=latest/best？
[ ] 团队最佳源码不可见时，是否明确标注“只有证据、没有源码”？
[ ] 是否保存了最终包的 SID/detail/ranking 快照？
[ ] 是否避免用调试包覆盖正式包？
```
