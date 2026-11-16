# 外部仓库深读报告索引

本目录保存第一/二批与 GitCode 补充共 9 个仓库记录的源码级深读（其中 1 个负例、2 个 GitCode 官方/上游记录）。按去重规则，GitCode 的 `cann/cann-ops-competitions` 与 `cann/elec-ops-inspection` 分别与既有官方基线/Hansen 归档重叠，不重复计为独立算法来源；但新增 3 份实现。每份报告都包含：

- 固定 commit、许可证、缓存位置、与既有归档的去重关系；
- 精确源码路径/行号、符号、Host 分路与执行入口；
- 语义、shape/dtype/layout、环境；
- 性能机制、工作量模型、证据级别；
- 适用条件、负例、退化条件；
- 对本项目的复用判断。

| 仓库 | 报告 | 深读实现 | 证据强度 |
|---|---|---|---|
| `Hansenwuwuwu/ascend-c-competition-archive` | [报告](Hansenwuwuwu__ascend-c-competition-archive.md) | Fire rank2、SplineConv rank1、TreeLineMinDistance rank1 | 官方公示分数 + 静态源码；无逐 case 时延 |
| `mantuoluozk/kernelswift-2026-triton-kernels` | [报告](mantuoluozk__kernelswift-2026-triton-kernels.md) | GroupedTopk、FusedMoE、Sinkhorn | 910B1 官方评测器 + 本机 910B3 复测（2 PASS，1 编译失败） |
| `So-cean/mamba-ascendc` | [报告](So-cean__mamba-ascendc.md) | Mamba-2 SSD ChunkMix | 910B3 device event/msprof；作者数字，未本机复现 |
| `vLLM-HUST/vllm-hust-opset` | [报告](vLLM-HUST__vllm-hust-opset.md) | Linear+SwiGLU Graph 融合 | 910B2 算子级/端到端外部报告；未本机复现 |
| `52Xayah/triton-ascend-kernels` | [报告](52Xayah__triton-ascend-kernels.md) | Softmax、Matmul（负例） | 作者粗略计时 + 本机 910B3 负例复现 |
| `Be4ver/svd` | [报告](Be4ver__svd.md) | PACO 批量 Jacobi、QDWH+SDC 融合 | 作者 910B3 设备日志 + 本机 CANN 9.1.0 构建复测（4 组全 PASS） |
| `Parity-LRX/ICTC-O3-HP-Ascend` | [报告](Parity-LRX__ICTC-O3-HP-Ascend.md) | packed O(3) TP、mesh 静电、capacity chunk | 作者 910B3/CANN 9.0.0 报告 + 完整协议；无逐点原始日志，本机未构建 |
| `cann/cann-ops-competitions`（GitCode） | [报告](GitCode__cann-ops-competitions.md) | Delicate02 `recompute_w_u`、BatchLions S9 `Concat` | 作者 CANNJudge 22/22、第 1 名/第十名；本机 recompute 编译成功但运行 507057；Concat 离线测试通过、未上 NPU |
| `cann/elec-ops-inspection`（GitCode） | [报告](GitCode__elec-ops-inspection.md) | XDU_XDU-BTD `LinalgLstsq` | 作者只有实现说明、无性能数字；本机 2 行兼容 patch 后编译成功，16/50/64 单 batch PASS，多 batch/矩形/200/256 失败 |

候选仓库（未深读，原因见总览）：

- `Gxj230958/TritonAscendBench`：正确性/来源冻结强，性能未接通；
- `elstainniles/triton-operator-benchmarks`：导入多、950PR 历史数据；
- `Yunle-Lee/CANN-Operator-test-competition-personal-solution`：测试模板，无高性能 kernel；
- `U-Illll/ascend-eigh-operator`：单算子 eigh，无性能数据；
- `asfsdsf/S6`：单算子 Fmax，无文档/性能；
- GitCode S9 `IndexAdd`（VibeThinking）：有实现说明、无性能数字；
- GitCode S9 `Greater` / `Transpose` / `SquareSumV1`：机制简单/偏搬运，暂不作为高性能主来源。

总览与方法：[`docs/research/OPEN_SOURCE_REPOSITORY_SURVEY_20261116.md`](../research/OPEN_SOURCE_REPOSITORY_SURVEY_20261116.md)。
GitCode 补充：[`docs/research/GITCODE_SOURCE_SURVEY_20261116.md`](../research/GITCODE_SOURCE_SURVEY_20261116.md)。
本机复测：[`docs/research/LOCAL_910B_VALIDATION_20261116.md`](../research/LOCAL_910B_VALIDATION_20261116.md)。
