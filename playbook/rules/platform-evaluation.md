# 平台精度与最终结果

## 1. 一句话

**平台最终结果是准绳；本地精度是筛查，不是一票否决。**

但这句话不等于“本地算错也可以提交”。要区分：

- **结构性错误**：本地就知道平台一定过不了；
- **数值差异**：可能来自本地参考、编译器、频率、fp16 舍入或容差口径，不能仅凭本地差异否定高性能方案。

## 2. 平台结果包含什么

CANNJudge 的 `detail` 至少看：

```text
status        Pass / Failed / Wrong Answer / Compile Error / Runtime Error / TLE
valid         提交是否有效（有些赛事 Pass 但 valid=false）
result[]      逐用例状态
Task Duration 设备侧单 Task 时间
频率/Block Dim/Profiler 文本
```

平台精度 = 官方 testcase + 官方 tolerance + 官方输入构造；本地 910B3 只是近似环境。

## 3. 本地失败分类

| 本地现象 | 判断 | 处理 |
|---|---|---|
| 编译失败 | **硬失败** | 先修；平台一定过不了 |
| 运行崩溃、`507035`/`507057`、非法 GM 地址 | **硬失败** | 先修；不要提交 |
| 输出形状/索引结构错、越界、NaN/Inf（题面不允许） | **硬失败** | 先修 |
| 确定性语义错（tie、空段、填充、`rot^T`、固定槽位） | **硬失败** | 先修；平台 golden 一定不一致 |
| 本地数值差很大（数量级/符号/整段错） | **大概率硬失败** | 查公式/布局/精度；不要靠平台碰运气 |
| 本地数值差很小，且在官方 atol/rtol 附近 | **软失败** | 查参考/容差/舍入；可做平台候选，但不能宣称通过 |
| 本地比本地 CPU golden 差，但官方 golden 未公开 | **软失败** | 用公开题面/上游语义核对；平台提交后看 detail |
| 本地时延不好 | **性能问题，不是精度** | 可提交测量；但先确认没有明显退化 |

## 4. 什么时候可以“带着本地差异提交”

同时满足才考虑：

1. 编译/运行/结构/索引/边界都通过；
2. 本地差异是数值级，不是符号/数量级/整段错；
3. 差异能用合法理由解释（fp16 舍入、FP32 累加顺序、参考实现公式差异、容差口径）；
4. 有官方公开 tolerance 或上游参考语义支持；
5. 提交前 dry-run 通过，源码没有 case-specific 分支；
6. 平台 detail 回读后，如果 `Wrong Answer`，按平台结果修，不用本地结果反驳。

不满足以上条件，不要用“高性能”当理由提交已知错误。

## 5. 不要做的事

- 不要因为本地 CPU 参考和 kernel 有一点差异，就直接删掉一个高性能方案；
- 也不要把本地 `pass` 写成平台 `Pass`；
- 不要用本地时延证明平台排名；
- 不要用作者 `O+A` 替代平台 `O`；
- 不要在平台 `Wrong Answer` 后说“本地是对的”而不看 testcase/容差；
- 不要针对具体 testcase/hash 分支来“过平台”。

## 6. 本库案例

- `recompute_w_u`：本机 CANN 9.1 编译成功但 5 个 shape 全部 `507057`，这是**硬失败**，不能提交；作者平台 22/22 是 `O+A`，不是本机 `V`。
- `LinalgLstsq`：16/50/64 单 batch PASS，但多 batch/矩形/200/256 崩溃；这是**范围不足**，不是“本地精度差一点”。
- `52Xayah Softmax/Matmul`：本机负例是性能不如厂商库，不是精度错误；不能拿它否定所有 Triton 方案。
- 制造专场 FFT：32×32 PASS 只是子集；不能推断 8192/inverse 平台通过。

## 7. 报告模板

```text
本地：
  build: pass/fail
  run: pass/fail/error code
  structural checks: shape/index/tie/empty/padding
  numeric: max_abs/max_rel vs 参考
  timing: 口径/warmup/repeat/median
平台：
  SID:
  status/valid:
  per-case:
  Task Duration:
  ranking:
结论：
  本地能证明什么：
  本地不能证明什么：
  平台最终结果：
```
