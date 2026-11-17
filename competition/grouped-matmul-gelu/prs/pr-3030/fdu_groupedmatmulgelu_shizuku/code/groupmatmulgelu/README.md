# groupmatmulgelu

提交团队: shizuku的队伍
提交者: 李翔然


## 算子实现介绍

**Cube 算完整 INT32，Vector 接手后处理。** Cube 将 A、B 分块搬入片上缓冲，沿 K 做 `Mmad` 累加。完整的 INT32 输出块写入环形缓冲，通过跨核事件交给 Vector。Vector 按专家和列读取 `scale`，在 FP32 做反量化和 tanh GELU，最后转 FP16，只写有效行列。窄 N 可走整 B 驻留或单列 Cube 点积，K=1 的大输出可用纯 Vector 外积。
