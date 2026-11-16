# 跨实现经验索引

本目录只放“能回到原始源码/结果”的跨算子经验，不复制 tile 数值、排行榜或 README 总结。每条经验都区分：

- `S`：静态源码事实；
- `M`：数学/布局推导；
- `A`：作者声明；
- `O`：原始评测/平台记录；
- `V`：本机 Ascend 910B3 复测；
- `N`：本环境未复现。

当前首批：

| 文档 | 主题 | 主要来源 |
|---|---|---|
| [工作量与固定开销](work_reduction_vs_fixed_overhead.md) | 什么时候减少算法工作量，什么时候用更多计算换掉启动/分发/中间张量 | SplineConv、GroupedTopk、TreeLineMinDistance、FusedMoE、Sinkhorn、Linear+SwiGLU、Mamba-2 |
| [布局与缓冲生命周期](layout_and_buffer_lifecycle.md) | AoS/SoA、消费者原生布局、双槽、跨核事件、UB/CBUF/L0C | Fire、Tree、Spline、Mamba-2、Linear+SwiGLU |
| [数值与选择性精度](numerical_selective_precision.md) | FP32 中间、选择性高精度修复、half 粗筛风险、softmax 单调性、tie/eps | Fire、Tree、GroupedTopk、Sinkhorn |
| [形状专用化与回退](shape_specialization_and_fallbacks.md) | Host 分路、编译期 key、fail-closed、路径覆盖与代码膨胀 | Hansen 三题、KernelSwift、Mamba-2、vLLM-HUST、52Xayah |

使用建议：

1. 先读当前任务书和评测协议，再看对应主题；
2. 不要从“某队用了这个技巧”直接推出“当前 shape 也会快”；
3. 每条经验都给了可证伪的验证方法，优先做能推翻假设的实验；
4. 任何第三方成绩都不是本机复现，引用时保留原环境、输入和计时口径。
