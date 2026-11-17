# Ascend eigh（候选，未深读）

> 层级：`general` · 状态：`candidate` · 本机 910B3：`not-run`

## 1. 来源登记

| 项 | 值 |
|---|---|
| 仓库 | https://github.com/U-Illll/ascend-eigh-operator |
| 作者 | U-Illll |
| 年份 | 2026 |
| 固定 commit | `fa2275fa4b2cee005e6868ce23266686eafa7a1b`（2026-11-13） |
| 默认分支 | `main` |
| 许可证 | MIT（根 `LICENSE`） |
| 上下文 | 单算子 Ascend C（非比赛） |
| 本地缓存 | `repository-research-cache/U-Illll_ascend-eigh-operator` |
| 去重 | 与 Be4ver SVD、GitCode Lstsq 不重复；同属线性代数，但算法不同 |

## 2. 语义与约束

仓库说明的路径是：tridiagonal reduction → QL/QR → back-transform，目标是 `eigh`（对称/Hermitian 特征分解）。

**未知项**（尚未深读）：

- 支持的 dtype（FP32/FP16/BF16？是否有双精度/双 limb？）；
- shape 范围、batch 维、是否只支持方阵；
- 特征向量是否输出、排序约定、残差/正交性检查；
- Host tiling 的分路阈值和 UB 预算；
- 许可证之外的第三方来源声明。

在深读前，不要把“eigh”这个名字当成完整语义说明。

## 3. 源码与证据

本轮只登记仓库和候选理由，**没有完成源码级深读**：

- 未固定/核对 Host、kernel、tiling、测试文件的行号；
- 作者 README 有实现说明，但没有性能数据（baseline、计时边界、输入、硬件均未披露）；
- 未在本机构建或运行；`local_status=not-run`。

## 4. 为什么保留为候选

- 单算子、MIT、语义清楚，和 Be4ver SVD 可以互补看“特征分解/三对角化/back-transform”在 910B 上的资源取舍；
- 线性代数条目目前只有 SVD 与 Lstsq，eigh 能补一个常见但实现难度高的算子族；
- 代码量相对小，适合做一次完整深读，而不是大规模基准。

## 5. 待办

1. 固定 commit 后列出 `op_host/`、`op_kernel/`、测试入口和关键符号行号；
2. 确认 dtype/shape/batch 约束和数值策略（是否迭代、是否双精度、是否有回退）；
3. 在 910B3/CANN 9.1.0 上构建；如编译器报错，只在工作区外副本加兼容 patch；
4. 用 `torch.linalg.eigh` 做小规模参考，记录 max abs/相对误差、正交性 `||V^T V - I||`；
5. 跑 16/32/64/128 等方阵，记录构建、运行、失败码；
6. 把结果写进 `README.md` 和 `evidence/`，再决定是否升级为 `deep-read`。

## 6. 复用判断

**暂不直接复用。** 先完成源码定位和 910B3 验证；如果只支持小方阵、没有性能数据，它更适合作为 eigh 实现参考，而不是 ACW 的首选模板。
