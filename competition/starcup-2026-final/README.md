# 中国电信星辰杯 · 高校 AI 算子开发挑战赛 决赛

> 层级：`competition` · 赛事：星辰杯决赛 · 题数：1 · 本机：`not-run`

## 1. 赛事

| 项 | 值 |
|---|---|
| contest_id | `6abcb2fa694b590c3c300a2e` |
| slug | `ct_starcup_aiop_final` |
| 时间 | 2026-10-09 00:00 → 2026-10-31 15:59 UTC |
| 官方题面 | <https://cannjudge.cn/public/ct_starcup_aiop_final> |
| 官方数据 | [`official/`](official/) |

## 2. 题目

| 题 | 平台 | 官方题面 | 我们的条目 | 平台排名 / 分 |
|---|---|---|---|---:|
| SparseFlashAttentionGrad | `6ac8bc7c694b590c3c96728a` | [desc](official/descriptions/sparseflashattentiongrad.md) | [sparse-flash-attention-grad/](sparse-flash-attention-grad/) | 6 / 28, 73.77 |

## 3. 平台结果

| 项 | 值 |
|---|---|
| 账号 | `feixiangtian16` |
| 团队 | `hfutslowpoke 的队伍` |
| SID | `6ae610371dac5d235b1e9b2e` |
| score | 73.77 |
| 状态 | Pass / valid=true |
| 可见 case | 10 / 15 |
| 可见总时间 | 1847.24 ms |

## 4. 一句话结论

三阶段融合（dP/dV → dS → dQ/dK）+ 稀疏 gather/scatter + KV merge + L1 双缓冲；
可见 10 case 平均 81.30，但隐藏 5 case 反推只有 58.71，说明隐藏 shape 的 tiling key
选择仍不够好。源码是 token 压缩形式，可提炼机制，不宜逐行当教学样例。
