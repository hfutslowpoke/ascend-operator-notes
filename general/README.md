# 通用算子库

这里保存非竞赛来源：生产库、框架算子、独立 Ascend C/Triton 实现、基准集合，以及后续接入的 CANNBot 知识。和竞赛库不同，通用库更看重语义稳定、接口边界、数值策略和跨版本可维护性，而不是某次榜单名次。

## 条目

| 条目 | 算子族 | 来源/类型 | 证据 | 本机 910B3 |
|---|---|---|---|---|
| [be4ver-svd](linear-algebra/be4ver-svd/README.md) | 线性代数 | 生产/实验型 Ascend SVD | 作者 910B3 日志 + `V` | PACO 6.174 ms；QDWH 128/512/1024 = 4.288/20.698/48.475 ms，全 PASS |
| [ascend-eigh](linear-algebra/ascend-eigh/README.md) | 线性代数 | 单算子 Ascend C | `S+L+A` | 未构建；作者无性能数据 |
| [mamba-ascendc](attention/mamba-ascendc/README.md) | 注意力/SSM | 生产型 Mamba-2 | `O+A` | 未构建 |
| [vllm-hust-opset](transformer/vllm-hust-opset/README.md) | Transformer | 生产型 vLLM Ascend 算子集 | `O+A` | 未搭建依赖链 |
| [parity-ictc-o3](equivariant/parity-ictc-o3/README.md) | 等变/科学计算 | 生产型 O(3)/MACE | `O+A` | 未构建 |
| [52xayah-triton-kernels](elementwise/52xayah-triton-kernels/README.md) | 逐元素/负例 | 教学/实验型 Triton-Ascend | `A+V` | Softmax 0.288×、Matmul 0.242×，负例稳定 |
| [tritonascendbench](benchmarks/tritonascendbench/README.md) | 迁移/正确性基准 | 101 个生产 kernel | `S+L` | 未跑；作者明确 910B 性能未接通 |
| [triton-operator-benchmarks](benchmarks/triton-operator-benchmarks/README.md) | 基准集合 | FlagGems/Liger/FLA 导入 | `S+L` | 未跑；历史环境 950PR |
| [cannbot](cannbot/README.md) | 知识接入位 | CANNBot 知识/工具参考 | 待接入 | 待定 |

淘汰记录见 [excluded.md](excluded.md)。机器索引见 [catalog.json](catalog.json)（由 `tools/build_index.py` 生成），全库索引见 [../index/README.md](../index/README.md)。

## 准入规则

1. **非竞赛或竞赛来源不明确**：竞赛高性能 PR 放 `competition/`；来源不明但只有单算子的，先放通用库候选。
2. **有算子内容**：语义、实现、接口、tiling、数值策略至少有一项能定位到源码/权威文档；工具/API 页面只作参考。
3. **有证据或明确缺口**：作者性能、本机复现、正确性基准至少有一类；没有就标 `candidate`，不能写成“高性能”。
4. **不是重复搬运**：FlagGems/Liger/FLA 等导入不算独立算法来源；只作为基准/迁移参考。
5. **许可清楚**：未声明许可证的仓库只做引用和行号定位，不复制源码；源码缓存留在工作区外。

## CANNBot 的定位

CANNBot 适合补充：

- 算子语义、接口、dtype/layout、tiling 约束；
- Ascend C API 的可用性/限制；
- 性能调优建议、常见错误和版本差异。

但 CANNBot 的通用 API 页面、工具说明不算“高性能算子实现”，不计入 33 份实现的数量。接入规则见 [cannbot/INGESTION.md](cannbot/INGESTION.md)。

## 怎么读一个条目

1. 先看“语义与约束”，确认它和你要实现的算子是不是同一数学对象、同一 dtype/layout。
2. 看“源码定位”与 `source.json`，确认不是只有 README/论文。
3. 看“作者证据”和“本机验证”是否分开；不要把作者 910B3 数字当成本机 910C 数据。
4. 看“负例/退化”，尤其是版本敏感、UB 容量、动态 shape、atomic 非确定性。
5. 最后看“复用判断”：哪些机制可以迁移，哪些接口/依赖不能直接搬。
