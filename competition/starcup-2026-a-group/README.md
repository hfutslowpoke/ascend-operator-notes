# 中国电信星辰杯 · 高校 AI 算子开发挑战赛 A 组

> 层级：`competition` · 赛事：星辰杯 A 组 · 题数：3 · 本机：`not-run`

## 1. 赛事

| 项 | 值 |
|---|---|
| contest_id | `6a7bf036a52e0f540a88df6e` |
| slug | `ct_starcup_aiop_g1` |
| 时间 | 2026-08-11 16:00 → 2026-09-29 06:15 UTC |
| 官方题面 | <https://cannjudge.cn/public/ct_starcup_aiop_g1> |
| 官方数据 | [`official/`](official/) |

## 2. 题目与平台结果

| 题 | 平台 | 官方题面 | 我们的条目 | 团队排名 / 分 |
|---|---|---|---|---:|
| Sparse Softmax（简单） | `6a7c17b6a52e0f540a89c0ce` | [desc](official/descriptions/sparsesoftmax.md) | [sparse-softmax/](sparse-softmax/) | **2 / 161, 84.36** |
| Dense Lightning Indexer Grad KL Loss（中等） | `6a7c1839a52e0f540a89c463` | [desc](official/descriptions/denselightningindexergradklloss.md) | [dense-lightning-indexer-grad-kl-loss/](dense-lightning-indexer-grad-kl-loss/) | 15 / 94, 28.83 |
| mHcPre（困难） | `6a7c191ca52e0f540a89ca9e` | [desc](official/descriptions/mhcpre.md) | **未入库**（优化不足，见 [excluded](../excluded.md)） | 18 / 106, 65.01（队友 `gao`） |

> **mHcPre 为什么不入库**：团队第 18、65.01 分来自 `gao` 的
> `6aba2b61694b590c3cca0422`，不是 C/D；`gao` 的源码对 C 账号返回 403。
> C 个人最新 497262 是 TLE，latest 模式覆盖了 496336 的 Pass。C 的 zip 497378
> 有 hPost 提前写回，但优化效果一般且无平台计时。按“宁缺毋滥”原则，这题只保留
> 团队成绩和 `excluded` 记录，不把 C 的源码当高性能条目入库。
> zip 497378 **未入库**（SHA256 `7685a9cc...`）；需要时由维护者从中转区单独提供。

## 3. 两题一句话 + 一个未入库

- **Sparse Softmax**：launch-bound 小 case；raw UB 去 TPipe/去 tiling 读取，固定 8 lane
  all-pairs + BlockReduce；5 case 1.40–1.48 ms，第 2 名。风险是精确 shape 入口和 bf16 缺失。
- **DenseLightning**：复用 softmaxMax/Sum，KL+全部梯度融合，UB arena，head split；
  但纯 Vector 做 D=128 GEMM，小 case 比 best 慢 7–12 倍；host 还硬编码
  `hasRope=0/sparseMode=0/useSeqLens=0`，题面的 RoPE/mask/变长未实现。
- **mHcPre（未入库）**：团队最佳 65.01 来自 `gao`；C 的源码优化不足、团队最佳源码不可见，
  只作为“团队取包 / latest 覆盖 / 小 K 纯 Vector”的经验案例，不作为知识库实现。

## 4. 证据边界

- 平台排名/分数/逐 case 时间来自 CANNJudge 只读查询；
- 源码来自提交详情；Sparse Softmax 494890、DenseLightning 495694 与用户 zip 逐文件一致；
- 未在本机 910B3 重跑；未看隐藏 case 输入；不声称 910C 结果。
