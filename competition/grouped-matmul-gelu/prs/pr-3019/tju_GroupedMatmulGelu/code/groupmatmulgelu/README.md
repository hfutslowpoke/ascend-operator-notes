# groupmatmulgelu

提交团队: 老板来份忘情牛肉题
提交者: 颜宇熙


## 算子实现介绍

实现的 GroupedMatmulGelu 算子需完成以下核心计算：
1. 变长分组矩阵乘：x 和 weight 均为 INT8，矩阵乘采用 INT32 累加。每个专家的 K、N 相同，组内 token 数 M_e 可以不同。
2. per-channel 反量化：每个专家、每个输出通道使用独立的 FP32 系数 scale[e,n]，沿该专家的 M_e 维广播。
3. tanh GELU 激活：对反量化的 FP32 中间结果计算固定 tanh 近似定义的 GELU。
4. 输出转换：激活结束后才转换为 FP16，保持输入 token 的分组排列顺序。
