# 竞赛知识库

这里保存公开竞赛中的高性能 PR/实现。目标不是复刻某次提交，而是提炼可迁移的机制：任务切分、布局、搬运、片上驻留、同步、融合、数值策略，以及它们在特定 shape/硬件上的边界。

## 条目

| 条目 | 赛事/上下文 | 实现 | 证据 | 本机 910B3 |
|---|---|---|---|---|
| [grouped-matmul-gelu](grouped-matmul-gelu/README.md) | 2026 CANN 挑战赛上合赛区决赛 | 26 份公开 PR + 我们的 `kernel.asc` | `O+A` + 历史 `V` | 58.54 分、第 11、整包 Pass（历史快照） |
| [cann-ops-competitions](cann-ops-competitions/README.md) | 2026 CANN 天梯赛 / S9 算子挑战赛 | Delicate02 `recompute_w_u`、BatchLions `Concat` | `O+A` | recompute 编译成功但运行 `507057`；Concat 仅离线测试 |
| [elec-ops-inspection](elec-ops-inspection/README.md) | 2026 电力行业算子专场（官方仓） | XDU_XDU-BTD `LinalgLstsq` | `S+L+A` | 16/50/64 单 batch PASS；多 batch/矩形/200/256 失败 |
| [hansen-2026-power](hansen-2026-power/README.md) | 2026 电力行业算子专场（公开归档） | Fire rank2、SplineConv rank1、TreeLineMinDistance rank1 | `O+A` | 未构建/未运行 |
| [kernelswift-2026](kernelswift-2026/README.md) | 2026 KernelSwift 算子创新大赛 | GroupedTopk、FusedMoE、Sinkhorn | `O+A` | GroupedTopk/Sinkhorn PASS；FusedMoE 编译失败 |
| [mfg-2026](mfg-2026/README.md) | 2026 制造行业算子专场（15 题） | 6 个官方/生产仓、15 份实现映射、2 个 910B 基础件、75 份源码快照、15 份官方题面、6 题适配蓝图 | `O+S+L+A` + 本机 `V`（FFT 子集） | 官方语义分级 2 direct / 6 adapt / 5 reference / 2 none；FourierTransform2D 仅 32×32 子集 PASS |

淘汰记录见 [excluded.md](excluded.md)。机器索引见 [catalog.json](catalog.json)（由 `tools/build_index.py` 生成），全库索引见 [../index/README.md](../index/README.md)。

## 准入规则

一个竞赛条目要进入本库，至少满足：

1. **公开且可固定**：有公开 URL、默认分支/提交、固定 commit；fork、镜像、重复搬运不算独立来源。
2. **有真实实现**：不是只有题面、报名页、aclnn 测试脚手架；要能定位 Host/kernel 源码。
3. **有证据**：作者/平台成绩、逐 case 日志、源码审计或本机复现，至少有一类；未披露的字段写“未知”。
4. **有复用价值**：能提炼机制、适用条件和负例，而不是“某队拿了名次”本身。
5. **不冒充通用结论**：竞赛榜单/隐藏用例成绩只属于该赛事；不能写成生产算子性能。

## 怎么读一个条目

1. 先看 `README.md` 的“来源登记”和“证据级别”，确认 commit、硬件、CANN、计时口径。
2. 再看“机制”和“适用条件/负例”，判断它解决的是你的瓶颈，还是只在特定 shape 上有效。
3. 打开 `source.json` 核对源码路径/行号和哈希，避免 README 与代码漂移。
4. 看 `evidence/` 里的原始日志；没有本机结果就按 `N` 处理。
5. 最后看“复用判断”：优先复用、需要验证、暂不使用分别是哪些条件。

## 当前缺口

- 制造专场 15 题：官方题面已落库；按官方语义 2 题 direct、6 题 adapt、5 题 reference、2 题 none；`FourierTransform2D` 仅在 910B3 通过 32×32 C2C 子集（官方要求 2..8192 + inverse），其余 14 份未跑；
- `recompute_w_u` 未在 CANN 8.5.0 复跑，`507057` 根因未定位；
- `Concat` 未构建 NPU 包，动态张量列表 ACLNN 驱动缺失；
- `LinalgLstsq` 多 batch/矩形/200/256 未通过，矩形 `residuals` 布局未确认；
- Hansen、KernelSwift 的未运行实现仍只有作者证据；
- 没有 910C、msprof/PMU 的跨代对比。
