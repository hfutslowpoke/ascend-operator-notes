# ops-solver：ComplexDiv 算法参考

> 层级：`competition` · 赛事：2026 制造行业算子专场 · 对应赛题：02 `ComplexDiv` · 本机 910B3：`not-run`

## 1. 来源登记

| 项 | 值 |
|---|---|
| 仓库 | https://gitcode.com/cann/ops-solver |
| 维护者 | CANN 开源社区 |
| 固定 commit | `739775a145d7789693f97758f6771bb1b3eb6a8f`（2026-11-17） |
| 默认分支 | `master` |
| 许可证 | CANN Open Software License Agreement Version 2.0 |
| 本地缓存 | `repository-research-cache/mfg-2026/ops-solver/` |
| 去重 | 与 ops-math 的复数算子不重复；这里是复数 LU/QR 求解器内部的除法 |

## 2. 赛题语义

`ComplexDiv`（02）要求复数逐元素除法：

```text
(a + bi) / (c + di) = ((ac + bd) + (bc - ad)i) / (c² + d²)
```

直接套公式容易在 `c²+d²` 溢出/下溢时出错；需要缩放、补偿或 Smith 算法。

## 3. 源码定位

| 内容 | 路径/行 |
|---|---|
| 复数除法 `Divide` | `src/utils/complex_solver_kernel.hpp:61-80` |
| 补偿商 `Quotient` | 同上 `:49-59` |
| 缩放/分母/误差 | 同上 `:67-80` |
| 架构守卫（仅 950 SIMT） | 同上 `:15-18` |
| 复数 LU 迁移说明 | `docs/zh/complex_solver_migration.md:1-20` |

机器索引：[source.json](source.json)。

本地源码副本：[`source/`](source/)（按上游路径保存，字节与固定 commit 一致；`source.json` 的 `local_path` 指向对应文件。上游原文的相对链接假设整仓，本地不保证可解析）。

## 4. 机制

`Divide(a,b)` 的做法：

1. `scale = max(|b.real|, |b.imag|)`；取指数位，把 scale 规整到 2 的幂，避免后续平方溢出/下溢；
2. 把 `a.real/a.imag/b.real/b.imag` 都除以 `scale`；
3. 令 `first=max(|br|,|bi|)`、`second=min(...)`，计算 `denominator = first² + second²`，并用 FMA 记录 `first²`、`second²` 的舍入误差；
4. `Quotient` 先算 `numerator = ar*br + ai*bi`，用 FMA 记录乘积误差；再 `result = numerator/denominator`，用 `remainder = fma(-result,denominator,numerator)+error-result*denominatorError` 修正，最后 `fma(remainder, 1/denominator, result)`；
5. 虚部同理，符号按 `(bc - ad)` 处理。

这是 Smith 缩放 + 补偿商，目标是避免 `c²+d²` 溢出/下溢和大幅值抵消。

## 5. 910B 适用性与证据

- 文件在 `#if __NPU_ARCH__ == 3510` 内，注释明确：`simt_api` 头只在 dav-3510（Ascend950）可编译，dav-c220/dav-m200 会报错；
- 迁移文档也写明复数 LU/求逆接口“仅接受 Ascend950，其他架构返回 ARCH_MISMATCH”；
- 因此这是 **950-only 算法参考**，不是 910B 可调用算子；本批未构建：`local_status=not-run`。

## 6. 负例/风险

- 不是独立 `ComplexDiv` elementwise 算子；只是 LU 面板里的除法 helper。
- 依赖 SIMT（`asc_simt.h`/shuffle/FMA），不能直接搬到 910B Vector。
- 只处理 `aclFloatComplex`（complex64）；complex32/128 需另写。
- 没有性能数据；补偿商指令数比朴素公式多，是否值得要看题面精度/范围。

## 7. 复用判断

**优先复用**：缩放因子、`Quotient` 的 FMA 误差修正、`first/second` 排序避免平方溢出。

**需要验证**：910B 上 Vector 版 `Div`/`Sqrt`/`Mul` 的精度；题面是否要求 complex64 以外的 dtype；是否允许用 double 中间量。

**暂不使用**：直接编译这份 SIMT 代码到 910B；把它当完整算子。

## 8. 下一步

1. 拿到 `ComplexDiv` 题面，确认 dtype、特殊值、精度阈值；
2. 用 910B Vector 写一个逐元素版本，先做缩放 + 朴素商，再加 FMA 补偿；
3. 与 CPU `numpy.complex64` 对照，覆盖 `c²+d²` 接近 0/极大/极小的输入；
4. 记录 max abs/相对误差、指令数、时延；如果补偿没有收益，保留朴素路径。
