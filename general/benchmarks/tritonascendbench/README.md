# TritonAscendBench（候选，正确性/迁移参考）

> 层级：`general` · 状态：`candidate` · 本机 910B3：`not-run`

## 1. 来源登记

| 项 | 值 |
|---|---|
| 仓库 | https://github.com/Gxj230958/TritonAscendBench |
| 作者 | Xingjian Guo |
| 年份 | 2026 |
| 固定 commit | `8a395c87f60a5e0e97f4a48cb16dcc6129e7e850`（2026-10-20） |
| 默认分支 | `main` |
| 许可证 | Apache-2.0 |
| 上下文 | 迁移基准（非比赛） |
| 本地缓存 | `repository-research-cache/Gxj230958_TritonAscendBench` |
| 去重 | 与 `elstainniles/triton-operator-benchmarks` 不同：本仓强调 vLLM/SGLang 生产 kernel 的迁移与正确性冻结 |

## 2. 内容与价值

作者登记的规模是 101 个 vLLM/SGLang kernel，其中 83 个 verified、18 个 ceiling-accepted。真正有价值的是：

- 固定上游来源/版本，记录 kernel 语义和测试输入；
- 对“Triton 语义 → Triton-Ascend”迁移做正确性对照；
- 明确区分 verified 与 ceiling-accepted，不把后者当完全通过。

作者同时明确：**910B 性能尚未接通**。因此本仓当前只能作为正确性/迁移参考，不能作为性能证据。

## 3. 已知限制

- 未在本机构建或运行；`local_status=not-run`。
- 101 个 kernel 中哪些能在 CANN 9.1.0/TA 3.2.0 上编译，尚未逐项验证。
- 性能计时未接通；即使后续跑通，也要重新定义 warmup/重复/统计口径。
- 部分 kernel 可能依赖 vLLM/SGLang 的特定输入布局，不能只看 shape 相同就复用。

## 4. 适用场景

- 需要把一个上游 Triton kernel 迁到 Triton-Ascend 时，先用它的测试/参考做语义对照；
- 需要检查边界 dtype/layout 时，作为用例来源；
- 作为“正确性优先、性能后补”的工程模板，而不是“最快实现”榜单。

## 5. 待办

1. 选 1 个与 ACW 当前算子接近的 kernel，固定上游版本和输入；
2. 在 910B3/TA 3.2.0 上编译，记录失败码/UB 溢出；
3. 跑作者提供的 correctness 用例，记录 max abs/相对误差；
4. 再补本机计时（warmup/repeat/median），与基线分开写；
5. 如果只通过正确性、没有性能收益，保持 `candidate`，不要升级为“高性能来源”。

## 6. 复用判断

**优先复用其测试/来源冻结方法，不直接复用其性能结论。** 对 ACW 来说，它是迁移正确性基准，不是算子性能库。
