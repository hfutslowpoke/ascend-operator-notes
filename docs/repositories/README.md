# 外部仓库深读报告索引

本目录保存第一批 5 个仓库的源码级深读。每份报告都包含：

- 固定 commit、许可证、缓存位置、与既有归档的去重关系；
- 精确源码路径/行号、符号、Host 分路与执行入口；
- 语义、shape/dtype/layout、环境；
- 性能机制、工作量模型、证据级别；
- 适用条件、负例、退化条件；
- 对本项目的复用判断。

| 仓库 | 报告 | 深读实现 | 证据强度 |
|---|---|---|---|
| `Hansenwuwuwu/ascend-c-competition-archive` | [报告](Hansenwuwuwu__ascend-c-competition-archive.md) | Fire rank2、SplineConv rank1、TreeLineMinDistance rank1 | 官方公示分数 + 静态源码；无逐 case 时延 |
| `mantuoluozk/kernelswift-2026-triton-kernels` | [报告](mantuoluozk__kernelswift-2026-triton-kernels.md) | GroupedTopk、FusedMoE、Sinkhorn | 910B1 官方评测器；作者数字，未本机复现 |
| `So-cean/mamba-ascendc` | [报告](So-cean__mamba-ascendc.md) | Mamba-2 SSD ChunkMix | 910B3 device event/msprof；作者数字，未本机复现 |
| `vLLM-HUST/vllm-hust-opset` | [报告](vLLM-HUST__vllm-hust-opset.md) | Linear+SwiGLU Graph 融合 | 910B2 算子级/端到端外部报告；未本机复现 |
| `52Xayah/triton-ascend-kernels` | [报告](52Xayah__triton-ascend-kernels.md) | Softmax、Matmul（负例） | 作者粗略计时；明确负结果 |

候选仓库（未深读，原因见总览）：

- `Gxj230958/TritonAscendBench`：正确性/来源冻结强，性能未接通；
- `elstainniles/triton-operator-benchmarks`：导入多、950PR 历史数据；
- `Yunle-Lee/CANN-Operator-test-competition-personal-solution`：测试模板，无高性能 kernel；
- `Be4ver/svd`：SVD，910 设备日志，下一批优先；
- `Parity-LRX/ICTC-O3-HP-Ascend`：O(3)/MACE，910B3 强数据，下一批优先；
- `U-Illll/ascend-eigh-operator`：单算子 eigh，无性能数据；
- `asfsdsf/S6`：单算子 Fmax，无文档/性能。

总览与方法：[`docs/research/OPEN_SOURCE_REPOSITORY_SURVEY_20261116.md`](../research/OPEN_SOURCE_REPOSITORY_SURVEY_20261116.md)。
