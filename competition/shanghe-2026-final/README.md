# 2026 CAPP 上合赛区决赛

> 层级：`competition` · 赛事：2026 年 CANN 挑战赛上合赛区（决赛） · 题数：1 · 本机：`platform-only`

## 1. 赛事

| 项 | 值 |
|---|---|
| contest_id | `6ae2a86a1dac5d235bf45ae8` |
| slug | `op_challenge_shanghe_final` |
| 时间 | 2026-10-30 09:00 → 2026-11-14 03:45 UTC |
| 官方题面 | <https://cannjudge.cn/public/op_challenge_shanghe_final> |
| 官方数据 | [`official/`](official/) |

## 2. 题目

| 题 | 平台 | 官方题面 | 我们的条目 | 平台排名 |
|---|---|---|---|---:|
| GroupedMatmulGelu | `6ae3fad11dac5d235b148620` | [desc](official/descriptions/groupmatmulgelu.md) | [../grouped-matmul-gelu/](../grouped-matmul-gelu/)（已有） | 11 / 32 |

> 上合决赛的源码、SID、平台证据已经在 [`../grouped-matmul-gelu/`](../grouped-matmul-gelu/)。
> 本目录不复制第二份源码，只补官方题面和链接。
> 这次新增的效率分析见
> [`../grouped-matmul-gelu/analysis/EFFICIENCY_20261120.md`](../grouped-matmul-gelu/analysis/EFFICIENCY_20261120.md)。

## 3. 平台结果

| 项 | 值 |
|---|---|
| 账号 | `slowpoke` |
| 团队 | `slowpoke 的队伍` |
| SID | `6af7d84c6bd0f9be3716f4c6` |
| score | 58.54 |
| 状态 | Pass / valid=true |
| 可见 case | 5 / 20 |
| 可见总时间 | 138.77 ms |

## 4. 一句话结论

INT8 Cube + FP32 反量化 + GELU 融合，AIC/AIV 4 槽流水，9 个 tile 的 host 代价模型，
非对齐 K/N 做 pack；可见 5 case 平均 76.16，但隐藏 15 case 反推只有 52.67，
说明按公开 shape 的精确分支没有覆盖隐藏 shape。完整分析见
[`../grouped-matmul-gelu/analysis/EFFICIENCY_20261120.md`](../grouped-matmul-gelu/analysis/EFFICIENCY_20261120.md)。
