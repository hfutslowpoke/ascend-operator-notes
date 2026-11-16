# groupmatmulgelu

提交团队: Lizzop
提交者: 李子豪


## 算子实现介绍

本算子采用 Ascend C 原生开发，Host 侧仅完成参数校验、Shape 推导、容量约束下的 Tiling 选择、Workspace 管理和单次Kernel 启动；全部矩阵计算、量化缩放、Tanh-GELU 激活及结果写回均在同一个 Device Kernel 内完成。计算阶段根据运行时Shape 选择 Cube 与 Vector 协同的分块流水路径，使用 GM Workspace 暂存必要的输入布局和中间结果，并通过片上缓冲与同步提升数据复用；针对 N=1 和 K=1 的通用退化场景使用专用向量化路径，其中 K=1 路径在 Kernel 内依据当前权重和缩放动态生成 INT8 查表数据。实现支持动态输入值、权重、缩放系数和 group_list，正确处理空专家、尾块及跨专家偏移，采用完整 INT32 累加后转 FP32/FP16，保证确定性、输入只读和单 Kernel 执行约束。
