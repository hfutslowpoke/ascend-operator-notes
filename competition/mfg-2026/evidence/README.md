# 制造专场本机验证记录

当前状态：**1 份本机 PASS、14 份未跑**；另有 1 份补充骨架构建 PASS、运行待锁。

| 实现 | 状态 | 设备/版本 | 结果 | 证据 |
|---|---|---|---|---|
| 05 `FourierTransform2D` / `ops-fft/fft2_d` | **PASS** | Ascend 910B3（物理 NPU 2 / ACL logical 0）、CANN 9.1.0、gcc 11.4.0、cmake 3.22.1 | 32×32 batch=1 C2C，5 次 best **0.1537 ms**，`max_abs=9.086799e-08`，`bad=0` | [构建日志](ops-fft_fft2d_build_20261117.log)、[运行日志](ops-fft_fft2d_32x32_run_20261117.log)、[驱动](fft2d_verify.cpp) |
| 07 补充骨架 `HIT1920/torch_scatter_npu` SegmentMinCsr | **BUILD PASS** | Ascend 910B3 工具链、CANN 9.1.0、gcc 11.4.0、cmake 3.22.1 | `install.sh -v Ascend910B3` 成功，生成 `custom_opp_ubuntu_aarch64.run`；AclNN 用例在 `bin/npu-lock` 排队 | [构建说明](segment_min_csr_alt_20261117.md)、[构建日志](segment_min_csr_alt_build_20261117.log) |
| 其余 14 份 | `not-run` | — | — | 见各条目 README |

## FFT 构建与运行细节

- 构建命令：`./build.sh --ops=fft2_d --soc=Ascend910B --pkg -j8`；
- 产物：`cann-910b-ops-fft_9.0.0_linux-aarch64.run`，SHA256 `1aa58979203dc6b55c954236239b3bbd620e425c45a62f4dc1b1bb76a10d0e92`；
- 安装：包安装到独立路径，运行时链接构建目录的 `libcann_ops_fft.so`；
- 驱动：`fft2d_verify.cpp`，32×32、batch=1、complex64，冲激放在 `(3,7)`，解析 DFT 为 `exp(-2πi(3k1/32+7k2/32))`；
- 计时：host `steady_clock` 包住 `aclfftExecC2C + aclrtSynchronizeStream`，含 launch/sync；
- 5 次：`1.2524 / 0.1880 / 0.1574 / 0.1544 / 0.1537 ms`（best 0.1537 ms）；
- 运行经 `bin/npu-lock` 串行。

## 计划顺序

| 优先级 | 条目 | 目标 | 状态/风险 |
|---:|---|---|---|
| 1 | `ops-fft/fft2_d` | 910B 构建 + 32/64/128 精度/时延 | 32×32 PASS；64/128、batch>1、backward 待跑 |
| 2 | `driving-sdk/radius` | 单算子调用 + CPU 邻居对照 | 未跑；全量构建重、torch_npu 版本 |
| 3 | `driving-sdk/furthest_point_sampling` | 索引序列对照 + 时延 | 未跑；总元素上限、分片 |
| 4 | `driving-sdk/cylinder_query` | 候选/排序结果对照 + 时延 | 未跑；A2 支持未单列 |
| 5 | `ops-math/stack_ball_query` | 910B 配置构建 + GEIR 用例 | 未跑；输入是堆叠布局 |
| 6 | `ops-gnn/segment_csr` | min/max/sum + 空段 | 未跑；全量 wheel 构建 |
| 7 | `ops-sparse/spmm`、`spgemm` | 小图结构/数值 | 未跑；多阶段 workspace/构建 |

950-only 的 `ComplexAbs`、`ComplexDiv`、`GatherCoo`、`RandomWalk`、`GridCluster` 不在 910B 上强构建；只做算法参考。

## 日志命名

```text
evidence/<entry>_<op>_build.log
evidence/<entry>_<op>_<shape>_run.log
evidence/<entry>_<op>_<shape>_verify.log
```

每个日志要带：

- 设备：物理 NPU 2 / ACL logical 0；
- CANN、gcc、cmake、torch/torch_npu；
- 命令、shape/dtype、参考实现、误差阈值；
- 计时口径（ACLNN launch+sync / event / msprof）；
- 失败码、fault kernel、兼容 patch。

## 硬规则

- 所有 NPU 运行必须 `bin/npu-lock`；
- 编译成功 ≠ 可运行；小 shape PASS ≠ 全 shape PASS；
- 910B3 结果不外推 910C；
- 作者/文档数字与本机数字分开写；
- 失败/负例保留，不删日志。
