# ops-fft：FourierTransform2D（910B 的 fft2_d）

> 层级：`competition` · 赛事：2026 制造行业算子专场 · 对应赛题：05 `FourierTransform2D` · 本机 910B3：`PASS`（32×32 C2C 子集，官方 2..8192 未覆盖，2026-11-17）

## 1. 来源登记

| 项 | 值 |
|---|---|
| 仓库 | https://gitcode.com/cann/ops-fft |
| 维护者 | CANN 开源社区 |
| 固定 commit | `65f5ff1dee32802562a9412789b0c7719ef99269`（2026-11-17） |
| 默认分支 | `master` |
| 许可证 | CANN Open Software License Agreement Version 2.0 |
| 本地源码 | `competition/mfg-2026/ops-fft/source/` |
| 去重 | 与既有 ops-math/ops-gnn/ops-sparse 不重复；FFT 是独立算子族 |

## 2. 赛题语义

> 官方题面（2026-11-17）：输入 `[B,N,M] complex64`，`B∈[1,256]`，`N,M∈[2,8192]` 且均为 2 的幂；`inverse=True` 时逆变换含 `1/(N·M)`；容差 atol=rtol=1e-3。`ops-fft/fft2_d` 只支持 32/64/128 的 C2C，覆盖的是官方 shape 的子集，不能把 32×32 PASS 当全题通过。

二维复数到复数 FFT（C2C）：

```text
Y[k1,k2] = sum_{n1,n2} X[n1,n2] * exp(-2πi*(k1*n1/N1 + k2*n2/N2))
```

接口：

```c
aclfftResult aclfftPlan2d(aclfftHandle* plan, int batch, int nx, int ny, aclfftType type);
aclfftResult aclfftExecC2C(aclfftHandle plan, aclfftComplex* idata, aclfftComplex* odata, int direction);
```

- `type` 仅 `ACLFFT_C2C`；`direction` 为 forward/backward。
- 输入/输出复数按行主序、实虚交错；`nx` 是慢索引（行），`ny` 是快索引（列）。
- 文档明确：**仅 Ascend 910B 支持，950 系列不支持**。

## 3. 源码定位

| 内容 | 路径/行 |
|---|---|
| 产品支持/语义/接口 | `docs/zh/API_Reference/FFT_2D.md:1-70` |
| 2D 执行入口、plan 校验、错误码 | `src/fft2_d/arch32/fft2_d_exec_api.cpp:18-60` |
| 尺寸表、P/Q 矩阵生成、Host 入口 | `src/fft2_d/arch32/dd/fft2_d_dd.cpp:17-29,31-62,64-112` |
| 复数实虚分离、UB/L1/L0 双缓冲、Cube | `src/fft2_d/arch32/dd/fft2_d_dd_kernel.h:50-143,259-420` |
| 公共 DFT/矩阵乘 | `src/common/kernel/fft/fftmatmul.h`、`matrix_dot.h`、`fft_common_core.h` |
| 头文件/API | `src/fft2_d/fft2_d.h`、`include/cann_ops_fft.h` |

机器索引：[source.json](source.json)。

### 3.1 910B 补充基础件：`fft1_d`

`ops-fft` 同一 commit 还有 `fft1_d`（arch32，910B C2C 1D），能补 `fft2_d` 的尺寸缺口：

| 内容 | 路径/行 |
|---|---|
| 1D 尺寸分派 | `source/src/fft1_d/arch32/fft1_d_exec_api.cpp:29-153` |
| n ≤ 256：DFT 路径 | `source/src/fft1_d/arch32/dft/fft1_d_dft.cpp` |
| 512..16384：FFTB 路径 | `source/src/fft1_d/arch32/fft_b/fft1_d_b.cpp` |
| 更大 n：FFTN 路径 | `source/src/fft1_d/arch32/fft_n/fft1_d_n.cpp` |
| 1D 接口/支持范围 | `source/docs/zh/API_Reference/FFT_1D.md` |

用法边界：`fft1_d` 是**单维** C2C；官方 2D FFT 需要按行/列各做一次 1D FFT，再处理转置/布局与 `inverse` 的 `1/(N·M)`。它能把官方 `N,M∈[2,8192]` 的单维长度覆盖到（n≤256 DFT、512..16384 FFTB），但不是现成的 2D FFT，也不是 FusedPropStep 的融合实现。

本地源码副本：[`source/`](source/)（按上游路径保存，字节与固定 commit 一致；`source.json` 的 `local_path` 指向对应文件。上游原文的相对链接假设整仓，本地不保证可解析）。

## 4. 机制

1. **2D DFT 拆成两次矩阵乘**：Host 生成 P（N1×N1 DFT 矩阵）和 Q（N2×N2 DFT 矩阵），设备侧按 `P·X·Q` 做二维变换；这比直接 O(N²) 逐点求和更适合 Cube。
2. **复数实虚分离**：输入复数实虚交错存在 GM；kernel 用 `vreducev2` 按 pattern 取 real/imag 到两个 UB 平面，矩阵乘后再合并写回。
3. **尺寸专用**：`CalcBatchNumsPerLoop` 只接受 `{32,64,128}×{32,64,128}` 的组合，并按尺寸选择每轮 batch 数（32×32 可 8 个 batch，128×128 只能 1 个）；其它尺寸直接拒绝。
4. **多级缓冲**：L1 256KB、L0A/L0B 32KB、L0C 64KB 的 ping-pong；batch 循环 + 多核切分；P/Q 常驻/复用。
5. **Host 生成旋转因子**：P/Q 元素由 `cos/sin(2πij/N)` 在 Host 计算，再 H2D；设备不做三角函数表生成。

减少的是：逐点 DFT 的 O(N²) 乘加和重复三角函数；代价是 Host P/Q 的 O(N²) 内存、固定尺寸限制、复数拆分/合并搬运。

## 5. 910B 适用性与本机证据

- 文档明确 910B √、950 ×；代码有 `arch32`（A2/910B）路径，`src/fft2_d/CMakeLists.txt` 按 SoC 选 arch 目录。
- **本机 PASS**：Ascend 910B3（物理 NPU 2 / ACL logical 0）、CANN 9.1.0、gcc 11.4.0、cmake 3.22.1。构建命令 `./build.sh --ops=fft2_d --soc=Ascend910B --pkg -j8`，生成 `cann-910b-ops-fft_9.0.0_linux-aarch64.run`（SHA256 `1aa58979…d0e92`）。
- 验证输入：32×32、batch=1、complex64，冲激放在 `(3,7)`；解析 DFT 为 `exp(-2πi(3k1/32+7k2/32))`。不依赖 `torch.fft`，避免参考实现自身的精度/版本差异。
- 结果：5 次顺序执行（无单独 warmup）`1.2524 / 0.1880 / 0.1574 / 0.1544 / 0.1537 ms`，**best 0.1537 ms**；`max_abs=9.086799e-08`、`max_rel=9.086799e-08`、`bad=0`。计时口径是 host `steady_clock` 包住 `aclfftExecC2C + aclrtSynchronizeStream`，含 launch/sync 开销。
- 注意：这只证明 32×32 forward 的 C2C 精度/可运行性；64/128、batch>1、backward、P/Q setup 成本仍未验证。`fft1_d` 可补单维 2..8192 的基础件，但 2D 组合/逆变换归一化仍需实现。仓库 README 的已验证 CANN 是 9.0.0-beta.2，本机 9.1.0 构建/运行通过，但不代表其它 CANN 版本。
- 证据：
  - [构建日志](../evidence/ops-fft_fft2d_build_20261117.log)
  - [32×32 运行日志](../evidence/ops-fft_fft2d_32x32_run_20261117.log)
  - [验证驱动](../evidence/fft2d_verify.cpp)

## 6. 负例/风险

- **尺寸受限**：源码只支持 32/64/128 的组合；官方要求 2..8192 的 2 的幂，源码缺少 2/4/8/16 与 256..8192。任意 N、质数 N、非 2 的幂都不行。
- **只支持 C2C**：没有 R2C/C2R；如果赛题要实数输入，需要自己补。
- **只支持 910B**：950 明确不支持；不能把这份代码当 950/910C 模板。
- **Host P/Q 开销**：N=128 时 P/Q 各 128×128 复数矩阵，Host 生成 + H2D 不是零成本；小 batch 可能被 setup 主导。
- **作者没有性能数据**：README 只讲功能/接口；本机只测了 32×32 单 batch，不能外推 64/128 或大 batch。
- **plan/stream 生命周期**：`aclfftPlan2d`/`aclfftDestroy`/`aclfftSetStream` 必须正确配对；错误码在 exec 入口有校验。

## 7. 复用判断

**优先复用**：2D DFT 的 P/Q 矩阵乘分解、复数实虚分离、尺寸专用 batch 循环、L1/L0 双缓冲。

**已在本机验证**：CANN 9.1.0 构建通过；32×32 单 batch forward 精度/时延通过（best 0.1537 ms）。

**仍需验证**：官方 2..8192 的 shape、batch>1、inverse（含 `1/(N·M)` 归一化）；`fft1_d` 行/列组合的布局/精度；P/Q Host 生成是否成为小 batch 瓶颈；多次调用下 plan 复用与 stream 绑定。

**暂不使用**：950/910C；任意尺寸；需要 R2C/C2R 的场景。

## 8. 下一步

1. 用 `fft1_d` 行/列组合扩到官方 2..8192 的 2 的幂、batch>1，并核对 inverse 归一化；仍用解析冲激或独立 CPU DFT 对照；
2. 跑 backward，确认 `ACLFFT_BACKWARD` 的符号/归一化口径；
3. 把 plan 创建、P/Q 生成/H2D、exec 分开计时，确认小 batch 的 setup 占比；
4. 若后续换 CANN 版本失败，保留失败日志，不把 9.1.0 的结果外推到其它版本。
