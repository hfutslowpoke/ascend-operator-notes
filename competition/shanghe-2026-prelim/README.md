# 2026 CAPP 上合赛区初赛

> 层级：`competition` · 赛事：2026 年 CANN 挑战赛上合赛区（初赛） · 题数：1 · 本机：`not-run`

## 1. 赛事

| 项 | 值 |
|---|---|
| contest_id | `6a9a9741bf41025d601345c7` |
| slug | `op_challenge_shanghe_prelim` |
| 时间 | 2026-09-04 16:00 → 2026-10-17 10:00 UTC |
| 官方题面 | <https://cannjudge.cn/public/op_challenge_shanghe_prelim> |
| 官方数据 | [`official/`](official/)（contest/problems/题面/完整 problem JSON） |

## 2. 题目

| 题 | 平台 | 官方题面 | 我们的条目 | 平台排名 |
|---|---|---|---|---:|
| BatchMatmulMaxSum | `6a9aa054bf41025d6014f3ef` | [desc](official/descriptions/batchmatmulmaxsum.md) | [batchmatmulmaxsum/](batchmatmulmaxsum/) | 24 / 661 |

## 3. 平台结果

| 项 | 值 |
|---|---|
| 账号 | `slowpoke` |
| 团队 | `slowpoke 的队伍` |
| SID | `6ad314c2193e507d8e5e08a4` |
| score | 64.44 |
| 状态 | Pass / valid=true |
| 可见 case | 15 / 15 |
| 可见总时间 | 396.75 ms |

逐 case 与效率分析见 [batchmatmulmaxsum/README.md](batchmatmulmaxsum/README.md)。

## 4. 一句话结论

Cube BMM + AIC/AIV max/sum 融合，按 shape 走 tiny/direct/flat/trans/split-K/cached-N
多条路径；小 shape 到 1.3 ms，大 shape 到 94 ms，整体约为第一名的 1.19 倍总时间。
早期纯 Vector 版本（zip 278917）在平台上是 Runtime Error，说明 bmm 规模上去后必须用 Cube。
