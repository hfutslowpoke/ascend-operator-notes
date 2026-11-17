# 制造专场：独立 CPU 参考实现

本目录用 NumPy 实现官方题面的边界语义，用于测试适配层和回答 tie/空段/first-K 问题。

**它不是官方隐藏 golden，也不是 Ascend 实现**；不能把这里的 PASS 当作竞赛评测通过。它只对应公开题面 + 上游参考实现核对（见 [算法参考核对](../../../docs/research/MFG2026_ALGORITHM_REFERENCES_20261117.md)）。

## 覆盖

| 函数 | 题 | 关键语义 |
|---|---|---|
| `segment_min_csr` | 07 | `arg_out=E`、空段 `out=0`、严格 `<` 首遇 |
| `furthest_point_sample` | 13 | 首点 0、`npoint` 轮、`argmax` 首最大值 = 最小索引 tie |
| `radius` | 11 | 定槽位、扫描序 first-K、`d²<=r²`、`(-1,-1)`、`r=0` 无命中 |
| `ball_query` | 14 | 严格 `d²<r²`、first-K、不足重复首个、零命中全 0 |
| `cylinder_query` | 15 | `rot^T·Δ`、严格边界、first-K、不足重复首个、零命中全 0 |
| `spmm_sum` | 09 | CSR×dense、空行 0、FP32 累加 |
| `spspmm` | 10 | COO×COO、固定槽位 `U=m·n`、保留零和坐标、行主序 |

## 运行

```bash
cd competition/mfg-2026/reference
python3 test_reference.py
```

当前结果：9 个测试全 PASS；日志在 [`../evidence/mfg_reference_tests_20261117.log`](../evidence/mfg_reference_tests_20261117.log)。

## 限制

- `radius` 的 `ignore_same_index` 按官方文字比较全局索引；如果 `ptr_x/ptr_y` 不同，需按题面/官方实现再确认全局索引是否对齐。
- `furthest_point_sample` 的 FP16 提供 `per_dim` 与 `fp32` 两种模式；官方只写“按输入 dtype/保持 half 精度”，最终必须用官方 `mx_driving` 逐位确认。
- `spspmm` 只实现小规模 COO 字典累加；没有 SpGEMM 的 workspace/多阶段语义，不能用于性能评估。
- 所有函数都假设输入已满足题面的单调/范围/无重复约束；越界检查只做最小防御。
- 只覆盖公开题面的语义，不包含隐藏测试数据/官方容差外的特殊行为。
