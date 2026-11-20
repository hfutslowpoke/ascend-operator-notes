# Cube + Vector 分工与 AIC/AIV 流水

> 一句话：GEMM 主导时，Cube 做矩阵乘、Vector 做 epilogue，两者用 AIC/AIV 和 ring buffer
> 重叠；但小 K/小矩阵-向量场景，Cube 的 setup 可能不划算，纯 Vector 更快。

## 1. 什么时候用

**Cube + Vector**：

- GEMM 是主体（K 较大、M/N 成块）；
- epilogue 是逐元素/归约（dequant、GELU、max、sum、softmax 反向）；
- 可以把 C 放在 GM 的少量槽位，AIC 产、AIV 消。

**纯 Vector**：

- 矩阵-向量级别，K=nD 只有 64/128（如 mHcPre）；
- 极小 shape，Cube setup/L1/L0 成本大于收益；
- 只需要逐元素/归约，没有真正的 GEMM。

## 2. 做法

1. **AIC/AIV 分工**
   - AIC：`MatmulImpl`/`Mmad` 产 C；
   - AIV：读 C → Cast/乘 scale/激活/归约 → 写输出；
   - `CrossCoreSetFlag` / `CrossCoreWaitFlag` 按槽同步。
2. **Ring buffer**
   - 2/4 个 C 槽，AIC 写下一个槽时 AIV 处理上一个；
   - 槽数按 C tile 大小和 UB/L1 预算定；不要太多，否则 cache 抖动。
3. **Pack 非对齐 A/B**
   - K/N 非 32B 对齐时，先 pack 到对齐布局；
   - pack 的成本要和 Cube 的尾块退化比较，不是所有 shape 都值得。
4. **Split-K**
   - M/N 小、K 大、核空闲时，把 K 切给多个核；
   - 部分结果在 GM/UB 归约；注意 max 不能跨 K 提前合并。
5. **Epilogue 复用**
   - 前向已有 `softmaxMax/softmaxSum` 时，反向直接复用，不重算 softmax；
   - GELU 用稳定 FP32 形式，不在激活前降 FP16。
6. **纯 Vector 的小矩阵**
   - `Brcb` 广播系数、`Mul + WholeReduceSum` 做矩阵-向量；
   - 参数（phi/bias/gamma）一次读入 UB，按 block 复用。

## 3. 证据

- **BatchMatmulMaxSum**：Cube BMM + AIV max/sum，split-K、cached-N、transposed-C；
  15 case 64.44 分，约为第一名的 1.19 倍总时间。
- **GroupedMatmulGelu**：INT8 Cube → INT32 C，AIV 反量化+GELU；4 槽 ring；
  可见 5 case 1.16x，但隐藏 15 case 反推 52.67，说明 dispatch 泛化不足。
- **SparseFlashAttentionGrad**：AIC/AIV `MIX_AIC_1_2`，L1 双缓冲，dP/dV/dS/dQ/dK 融合；
  可见 10 case 1.108x，隐藏 5 case 反推 58.71。
- **mHcPre**：K=64/128 的矩阵-向量，纯 Vector 反而合适；Cube setup 不划算。
- **BatchMatmul 早期纯 Vector 版**：平台 Runtime Error；规模上去后纯 Vector bmm 不可行。

## 4. 负例/边界

- 小 shape 用 Cube 可能更慢；先测 setup 成本。
- ring 槽不是越多越好；C tile 太大时，槽数增加会让 UB/L1 不够。
- pack 不是免费；非对齐但很小的矩阵，直接 `DataCopyPad` 可能更便宜。
- split-K 的 partial reduce 有额外 GM 往返；K 不能被 split 整除时要收尾。
- 纯 Vector 只适合小 K/矩阵-向量；D=128 的 GEMM 主导 shape 应优先 Cube。

## 5. 检查清单

```text
[ ] 这个 shape 真的是 GEMM 主导吗？还是矩阵-向量/逐元素？
[ ] Cube 的 setup 成本是否小于收益？
[ ] AIC/AIV 是否有明确的槽位/同步协议？
[ ] ring 槽数是否按 C tile 和 UB/L1 预算定？
[ ] 非对齐 K/N 是 pack 还是 DataCopyPad？有没有 A/B 对比？
[ ] split-K 的 partial reduce 是否计入总时间？
[ ] epilogue 是否复用了前向统计量/参数？
```
