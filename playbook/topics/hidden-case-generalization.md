# 隐藏 case 泛化：可见 case 高分不等于最终高分

> 一句话：比赛最终分通常包含不可见 case。只用公开/可见 shape 调优，会在隐藏 shape 上
> 掉一个档；要用 **shape family + 通用 fallback**，而不是只写精确 shape 分支。

## 1. 现象

平台分数是逐 case 分数的平均：

```text
case_score = 100 / (1 + log_1.5(t / T))
final_score = mean(case_score)
```

如果可见 case 很少，可以用最终分反推隐藏组均值：

```text
hidden_avg = (final_score * total - visible_avg * visible) / hidden_count
```

本次两例：

| 题目 | 可见组 | 最终分 | 隐藏组（反推） |
|---|---:|---:|---:|
| GroupedMatmulGelu | 76.16（5） | 58.54 | **52.67**（15） |
| SparseFlashAttentionGrad | 81.30（10） | 73.77 | **58.71**（5） |

隐藏组比可见组低 20+ 分，不是“稍微慢一点”。

## 2. 为什么精确分支会翻车

- 可见 case 命中 `if (k == 1024 && n == 3*256)`、`tasks==1 && axis==8`、
  `n4/d16/k64` 等专用路径；
- 隐藏 shape 不满足条件，落到通用路径；
- 通用路径可能是标量循环、无 Cube、无 pack、无 split，和专用路径差 2–10 倍；
- host 代价模型/启发式只对公开 shape 校准过。

## 3. 做法

1. **按 shape family 实现**：例如 `1≤axis≤8` 一条实例、`axis>4096` 一条 streaming；
   不要每个精确 shape 一条。
2. **保留通用 fallback**：专用路径不命中时，fallback 不能是指数级慢的标量循环；
   至少要有 Cube/Vector 的基本分块。
3. **扰动测试**：在公开 shape 附近构造 `±1`、非 2 幂、尾块、空段、不同 batch/head；
   不要只跑官方示例。
4. **host 代价模型**：用 rounded rows/cols/K、traffic、setup、epilogue 估成本，
   选 tile/kernel；不要只按 `if shape == X`。
5. **记录覆盖矩阵**：哪些 shape 族已验证，哪些只跑了示例，哪些完全没跑。
6. **不要用可见 case 的 1.0x 外推最终分**；最终分以平台为准。

## 4. 负例

- GroupedMatmulGelu：可见 5 case 1.16x，最终 58.54；隐藏 15 case 平均 52.67。
- SparseFlashAttentionGrad：可见 10 case 1.108x，最终 73.77；隐藏 5 case 平均 58.71。
- Sparse Softmax：`PATH6–9` 是精确 shape；`PATH11/12` 是 family，后者更值得复用。
- BatchMatmul：15/15 全可见，所以 score 与可见均值完全一致；这不代表其它题也一样。

## 5. 检查清单

```text
[ ] 是否知道 total/visible/hidden case 数？
[ ] 是否用最终分反推过隐藏组均值？
[ ] 是否有 shape family，而不是只有精确 shape？
[ ] fallback 是否至少是 Cube/Vector 分块，而不是标量循环？
[ ] 是否在公开 shape 附近做了扰动/尾块/空段测试？
[ ] 是否记录了“已验证/未验证/完全没跑”的 shape 矩阵？
[ ] 是否避免把可见 case 的 t/T 当成最终 score？
```
