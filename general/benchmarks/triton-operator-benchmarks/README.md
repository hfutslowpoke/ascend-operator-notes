# triton-operator-benchmarks（候选，基准集合）

> 层级：`general` · 状态：`candidate` · 本机 910B3：`not-run`

## 1. 来源登记

| 项 | 值 |
|---|---|
| 仓库 | https://github.com/elstainniles/triton-operator-benchmarks |
| 作者 | zhaojingkai |
| 年份 | 2026 |
| 固定 commit | `e6cf7c979f4b09968c59f25a15a542d282c9c7f2`（2026-11-09） |
| 默认分支 | `main` |
| 许可证 | Apache-2.0（导入的 FlagGems/Liger/FLA 保留原许可） |
| 上下文 | Triton 算子基准集合（非比赛） |
| 本地缓存 | `repository-research-cache/elstainniles_triton-operator-benchmarks` |
| 去重 | 主要是 FlagGems/Liger/FLA 导入，**不计独立算法来源** |

## 2. 内容与价值

作者登记 61 个条目，覆盖 elementwise、normalization、FLA、Liger 等；测试和条目组织较细，适合作为：

- Triton-Ascend 的基准/回归用例来源；
- 上游 FlagGems/Liger/FLA 语义的对照；
- 检查“同一 kernel 在不同 dtype/layout 下”的边界。

但它的主体是导入/复测，不是原创算法；历史环境是 Ascend 950PR，不是 910B。因此不能把其中的性能数字直接搬到 910B/910C。

## 3. 已知限制

- 未在本机构建或运行；`local_status=not-run`。
- 历史环境 950PR 与当前 910B3 不同代，编译/性能都可能不成立。
- FlagGems/Liger/FLA 各自许可不同，复用时要逐个核对，不能只看根 LICENSE。
- 条目多，容易把“收录数量”误当成“高性能实现数量”；索引里按导入/复测处理。

## 4. 适用场景

- 需要找某类算子的测试用例/边界输入；
- 需要对照上游 Triton 实现和多 dtype 行为；
- 作为 ACW 回归集的一部分，而不是性能 ranking。

## 5. 待办

1. 选 2–3 个与当前 ACW 算子相关的条目，确认上游来源和许可；
2. 在 910B3/TA 3.2.0 上编译，记录哪些条目因 API/UB/dtype 失败；
3. 只对通过正确性检查的条目补本机计时；
4. 把“导入/复测”与“原创”在索引里分开，避免重复计数。

## 6. 复用判断

**作为基准/用例参考，不作为独立算法来源或性能证明。** 如果某个条目能在 910B3 上稳定跑通，可以升级为具体算子的对照实现，但仍需单独建条目和证据。
