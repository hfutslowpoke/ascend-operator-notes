## 算子说明

### 算子功能

频域法是求解波动传播类偏微分方程（光波传播、声场仿真、电磁 FDTD 的角谱法等）的主流方法：将场量变换到频域，乘以传递函数 `h`，再变换回空间域。传统实现调用三次独立算子（FFT2、逐点乘、IFFT2），频谱中间量需要两次完整的 Global Memory 读写。

`FusedPropStep` 算子以单算子实现完整的频域传播步 `IFFT2(FFT2(field) · h)`，频谱中间量不流出 kernel。

计算规则：

```
S = FFT2(field)                       # 逐 batch 二维正变换
P = S * h                             # 频域逐点乘（h 沿 batch 广播）
out = IFFT2(P)                        # norm=True: 含 1/(N·M) 归一化
                                      # norm=False: 未归一化逆变换（1/(N·M) 因子约定预乘入 h）
```

### 输入输出规格

| 张量 | 形状 | 数据类型 | 含义 |
|------|------|---------|------|
| field | [B, N, M] | float16 / float32 / complex64 | 空间域输入场 |
| h | [Bh, N, M] | 与 field 相同 | 频域传递函数 |
| out | [B, N, M] | 与 field 相同 | 输出场 |

dtype 必须一致：field 与 h 同 dtype，输出同 dtype。实数情形等价于对实序列做 fft / ifft。

维度约束：B ∈ [1, 256]，N, M ∈ [2, 8192] 且均为 2 的幂。分量取值有限，不支持 NaN、+Inf、-Inf，取值在 [-1, 1]。

h 的 shape 约束：h 的 N、M 必须与 field 相同（频域逐点乘的数学约束）；batch 维允许 `Bh = 1` 或 `Bh = B`。`Bh = 1` 时 h 沿 batch 广播（同一传递函数作用于整个 batch），输出 shape 与 field 相同。

### 属性参数

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| norm | bool | False | True 时逆变换含 1/(N·M) 归一化；False 时归一化因子预乘入 h |

### 功能示例

恒等传播：h 为全 1（频域全通）且 `norm=True` 时 `out == field`（误差在浮点舍入内）。`norm=False` 时取 `h` 全 1/(N·M)（即把归一化因子预乘入 h）同样得到 `out == field`。若 h 为频域 δ（仅 h[0,0]=1），输出退化为常数场，每个元素等于 field 的逐 batch 均值（norm=True）。当 field 的 batch 为 B、h 的 batch 为 1 时，上述性质按广播后的每个 batch 分别成立。

### 任务要求

针对不同 B、N、M（2 的幂直到 8192）、dtype 与 h 的 batch 广播组合，设计算子逻辑，保证精度正确：complex64 配置满足 atol = 1e-4、rtol = 1e-4，float32 / float16 配置满足 atol = 1e-3、rtol = 1e-3。

充分发挥 NPU 带宽与多核并行能力：FFT/IFFT 与逐点乘在同一 kernel 内融合，减少频谱中间量的 Global Memory 往返；针对 2 的幂长度使用高基蝶形，batch 维并行划分；h 的 batch 广播按取模索引复用。float16 路径可在 float32 中做蝶形中间量。

**禁止事项**：禁止调用覆盖完整语义的现成融合算子，禁止回退到 CPU，禁止改变归一化约定（norm=False 时不得额外乘 1/(N·M)）。
