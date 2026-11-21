---
name: ascend-910b-local-validation
description: 在 Ascend 910B3/CANN 9.1.0 上构建、运行、计时和记录昇腾算子，强制 npu-lock、证据分级和负例保留。
---

# Ascend 910B Local Validation

## 何时使用

- 要把一个外部实现从“作者说快”变成“本机跑过/失败过”；
- 需要补 `evidence/`、`local_status`、条目 README 的“本机验证”章节；
- 需要判断编译失败、运行失败、精度不符、性能不达预期分别怎么记。

## 环境基线

| 项 | 本机值 |
|---|---|
| SoC | Ascend 910B3（**没有 910C**） |
| 物理设备 | NPU 2（`npu-smi info`） |
| ACL logical device | 0（`aclrtSetDevice(0)`） |
| CANN | 9.1.0（`/usr/local/Ascend/cann-9.1.0`） |
| gcc / cmake | 11.4.0 / 3.22.1 |
| NPU 串行锁 | `/workspace/acw/bin/npu-lock`（**只包设备运行**；编译/CPU 校验不持锁） |

不要从仓库名/作者环境推断本机兼容性；910B3 结果不能写成 910C。

## 步骤

### 1. 准备源码副本

- 第三方源码只放工作区外缓存或独立副本，例如 `repository-research-cache/local-validation-<date>/<name>/`；
- 需要兼容性 patch 时，只改副本，记录文件/行号/原因；不要改第三方仓库原文，也不要把 patch 写进本库；
- 构建前加载 CANN：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
```

### 2. 构建

- 用仓库自带 `build.sh`/CMake，记录完整命令、目标 SoC、产物；
- 保存构建日志到 `evidence/<name>_build.log`；
- 如果失败，记录编译器错误、缺失头文件、UB 溢出、API 版本差异；不要只写“编译失败”。

### 3. 运行

- 所有 NPU 命令经锁：

```bash
cd /workspace/acw
bin/npu-lock bash -lc '
  source /usr/local/Ascend/cann-9.1.0/set_env.sh
  <环境变量与运行命令>
'
```

- 记录：shape/dtype/layout、输入生成方式、参考实现、误差阈值；
- 保存 stdout/stderr 到 `evidence/<name>_<shape>_run.log`；
- 失败时记录 `aclrtSynchronizeStream`/`aclGetRecentErrMsg`、错误码（如 `507035`/`507057`）、fault kernel；
- 不要吞掉第一次运行的懒加载时间；区分首次运行和 warm run。

### 4. 计时

必须写清口径：

| 口径 | 含义 | 不能怎么用 |
|---|---|---|
| ACLNN launch + sync | 含 Host 调用/同步，适合端到端单算子 | 不能当纯 kernel 时间 |
| device event | 设备侧 kernel 区间 | 不能含未同步的 Host 工作 |
| msprof / Task Duration | 平台/工具口径 | 不能和 ACLNN 时延直接横比 |

记录 warmup、重复次数、统计（best/median）、是否逐 sample 同步。首次运行含懒加载时，不能把第一个值当稳定结果。

### 5. 记录到知识库

1. 原始日志放 `<entry>/evidence/`；
2. `index/catalog.json` 的 `implementations[].local_status` 用枚举：
   `pass` / `pass-negative` / `partial` / `fail` / `offline-only` / `platform-only` / `not-run` / `pending`；
3. `local_validation` 写环境、命令、shape、精度、时延、失败码、patch；
4. 条目 README 的“本机验证”写结论和日志链接；
5. 运行：

```bash
python3 tools/build_index.py
python3 tools/verify_archive.py
```

## 判定规则

- **编译成功 ≠ 可运行**：编译通过但运行 `507057`，`local_status=fail`。
- **小 shape PASS ≠ 全 shape PASS**：只覆盖部分路径时用 `partial`。
- **负例通过 ≠ 实现优秀**：手写 Triton 比基线慢，`pass-negative` 也是有效证据。
- **作者数字 ≠ 本机数字**：分别写，不计算跨环境加速比。
- **910B3 ≠ 910C**：没有 910C 数据就写 `N`。
- **失败要保留**：错误码、fault kernel、尝试过的 patch 都是后续判断依据。

## 输出模板

```markdown
## 环境
- 910B3 / CANN 9.1.0 / gcc 11.4.0 / cmake 3.22.1 / npu-lock

## 构建
- 命令：
- 结果：成功/失败
- 产物：
- 日志：

## 运行
| shape | dtype/layout | 结果 | best/median | max abs | 失败码 |
|---|---|---|---|---|---|

## 结论
- local_status:
- 适用/不适用：
- 未验证：

## 证据
- evidence/...
```

## 边界

- 不自动提交竞赛平台；不修改 ACW 框架或当前算子工程。
- 不保存 token、密码、私钥、完整 PROF、NPU 二进制。
- 不在没有日志的情况下写“本机通过”。
- 不把一次 best-of-5 写成稳定中位数。
