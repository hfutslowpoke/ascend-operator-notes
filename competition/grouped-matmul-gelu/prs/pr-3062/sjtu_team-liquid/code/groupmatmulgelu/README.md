# groupmatmulgelu

提交团队: TeamLiquid
提交者: LeonArmstrong


## 算子实现介绍

该版基于kernel31，按尺寸和资源分派，设备读取真实分组。深K宽输出先将非空专家权重转为NZ，再以128×256分块、512深面板双缓冲计算；窄输出用1024深面板。K=1宽输出在同次设备调用内构建256种INT8值的GELU结果表，再按实际x整行搬运；其他适合场景采用Martin的Vector路径。全程INT32累加、FP32反量化及GELU、FP16输出，入口4KB对齐，资源不足时回退。
