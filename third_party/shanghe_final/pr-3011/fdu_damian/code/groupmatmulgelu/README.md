# groupmatmulgelu

提交团队: Damian 的队伍
提交者: 孙长东


## 算子实现介绍

基于 Ascend C，以单 Kernel 融合分组 INT8 矩阵乘、FP32 反量化和 GELU，按实际分组调度多核任务，结合流水并行与数据复用加速，输出 FP16 结果。
