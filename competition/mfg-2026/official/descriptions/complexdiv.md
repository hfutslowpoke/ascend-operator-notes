## 算子说明

### 算子功能

复数除法广泛用于阻抗与导纳变换、滤波器系数计算、通信信道均衡等场景。CPU `torch.div`（aten::Div）对同 dtype 张量即逐点除法，支持广播。

`ComplexDiv` 算子计算 `(a + b·i) / (c + d·i)`，两个输入与输出必须同 dtype，不做隐式类型提升。

计算规则：

```
out = (a + b·i) / (c + d·i) = (a + b·i)(c - d·i) / (c² + d²)
```

### 输入输出规格

| 张量 | 形状 | 数据类型 | 含义 |
|------|------|---------|------|
| dividend | [N, M] | float16 / float32 / complex64 | 被除数 |
| divisor | [N, M] | 与 dividend 相同 | 除数（非零） |
| out | [N, M] | 与 dividend 相同 | 逐点商 |

dtype 必须一致：float16 ÷ float16 输出 float16，float32 ÷ float32 输出 float32，complex64 ÷ complex64 输出 complex64。不允许 fp32 与 complex64 混用。

维度约束：各输入 N, M ∈ [1, 65536]。分量绝对值不超过 10。

广播：dividend 与 divisor 按 NumPy / torch 广播规则逐点相除——两者 shape 从末维对齐，各维相等或其一为 1；输出 shape 为广播结果。

除数非零；除零行为未定义。

### 属性参数

无。

### 功能示例

除法：dividend = (3 + 4·i)、divisor = (2 - 1·i)（均为 complex64），则 out = 0.4 + 2.2·i。

广播：dividend=[1,1]、divisor=[8,8] 输出 [8,8]；dividend=[2048,2048]、divisor=[1,2048] 输出 [2048,2048]。此时输出所有元素相等，等于 dividend 的单元素值除以 divisor 各元素。

### 任务要求

针对不同 shape、广播组合与输入 dtype，设计算子逻辑，保证精度正确：float32 / complex64 配置满足 atol = 1e-4、rtol = 1e-4，float16 配置满足 atol = 1e-3、rtol = 1e-3。

充分发挥 NPU 带宽能力（逐点算子），向量化处理、多核均分元素；广播输入应按输出 shape 做索引映射。

**禁止事项**：禁止调用覆盖完整语义的现成复数除法接口，禁止回退到 CPU，禁止做隐式类型提升。
