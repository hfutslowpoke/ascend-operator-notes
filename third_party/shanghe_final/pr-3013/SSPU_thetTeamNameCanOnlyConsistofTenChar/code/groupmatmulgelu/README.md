# groupmatmulgelu

提交团队: 实测队名只能有十个字
提交者: gcw_E6y3NKaG


## 算子实现介绍

该算子将分组矩阵乘、逐列缩放和GELU激活融合为单个NPU kernel。根据矩阵规模选择Cube或Vector计算路径，通过分块调度、片上缓存复用及计算与搬运重叠减少开销，最终输出FP16结果。
