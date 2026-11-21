# 本地开发、编译、运行与 profiling

## 1. 环境检查

```bash
bash tools/check_env.sh
```

它只做只读检查：

- `npu-smi info` 是否有设备；
- `CANN_HOME` / `ASCEND_HOME_PATH`；
- `cmake`、`bisheng`、`msprof` 是否在 PATH；
- Python 版本；
- 当前目录和基本系统信息。

不要在没有检查设备占用的情况下直接跑 NPU 性能测试。

## 2. 典型工程结构

单 kernel 模板可能只有：

```text
kernel.asc
CMakeLists.txt
main.asc
run.sh
```

host + kernel 模板可能是：

```text
CMakeLists.txt
op_host/
  <op>.cpp
op_kernel/
  <op>.cpp
  <op>_tiling.h
  tiling_key_<op>.h
```

以模板实际返回为准。**提交时只提交 `editable: true` 的文件。**

## 3. 编译

单 kernel 模板通常按模板自带 `run.sh` / `CMakeLists.txt` 构建：

```bash
bash run.sh
```

host + kernel 工程常用 CMake ASC：

```bash
cmake -S . -B build \
  -DASCEND_COMPUTE_UNIT=ascend910b \
  -DASC_DIR="$CANN_HOME/lib64/cmake"
cmake --build build -j8
```

关键环境变量：

```bash
export CANN_HOME=/home/developer/Ascend/cann-9.0.0
export ASCEND_HOME_PATH="$CANN_HOME"
export ASCEND_CUSTOM_OPP_PATH="$PWD/build/tmp/vendors/custom"
export LD_LIBRARY_PATH="$CANN_HOME/lib64:$LD_LIBRARY_PATH"
```

不同 CANN 版本/模板的变量名可能不同，以模板和 `set_env.sh` 为准。

## 4. 运行

先跑模板自带的最小用例：

```bash
bash run.sh
# 或
./build/<target>
```

host 侧必须检查每个 ACL 返回值：

- `aclrtSetDevice`
- `aclrtCreateStream`
- `GetWorkspaceSize`
- `aclrtMalloc`
- kernel launch
- `aclrtSynchronizeStream`
- `aclrtMemcpy`
- 释放

失败时不要只看最后一行；从第一个非 0 返回值开始查。

## 5. NPU 串行锁

多个 agent/进程同时跑 NPU 会污染计时、增加争用，甚至导致初始化错误。所有**设备运行**建议统一用：

```bash
bash tools/npu_lock.sh <your-command> <args...>
```

或：

```bash
export NPU_LOCK=/mnt/workspace/sfag-locks/npu.lock
bash tools/npu_lock.sh ./run_bench.sh
```

编译、静态检查、CPU 参考实现可以并行；**设备运行串行**。

## 6. 计时口径

至少区分四种时间：

| 口径 | 来源 | 用途 |
|---|---|---|
| host event elapsed | `aclrtEvent` + `aclrtSynchronizeStream` | 本地快速对比，包含 launch/同步边界 |
| profiler Task Duration | `msprof` / 平台 profiling 文本 | 更接近单 Task 设备时间 |
| 完整算子窗口 | 一次调用内所有 kernel/task | 多 kernel 时不能只看一个子任务 |
| 平台成绩 | 平台提交详情 | 最终裁决 |

不要直接把本地 event 时间、`msprof` Task、平台成绩互相换算。记录：

- 设备型号、AIC/AIV 核数；
- 实际频率（`npu-smi info -t freq` 或 profiler info）；
- CANN 版本、编译器版本；
- 源码 SHA、构建产物 SHA；
- 实际 tiling key / kernel 名 / block dim；
- workspace 大小与是否复用；
- 输入分布、seed、有效边数/重复度；
- 预热次数、计时次数、丢弃策略、中位数/均值。

## 7. profiling

常用：

```bash
msprof --application="./your_app" --output=./prof
```

或使用模板/框架自带脚本。重点看：

- Task Duration；
- AIC / AIV 的 Cube、Vector、MTE1/2/3、Scalar、Fixpipe busy；
- L2 hit/miss；
- GM→L1 / GM→UB 请求量；
- 实际 block dim；
- kernel 名和 tiling key。

注意：

- busy 时间不能简单相加；
- 平均 busy 不等于最慢核；
- Task Duration 与各核 busy 的差值包含启动、同步、尾部、调度；
- 频率变化会改变重叠关系，不只是等比缩放；
- 多 kernel 时要把完整调用窗口算进去。

## 8. 正确性验证

至少覆盖：

1. **官方样例 / 模板用例**
2. **形状边界**：最小 shape、最大 shape、非对齐尾块、单行/单列、空索引、全重复、全非法、极端重复度
3. **数值边界**：0、负数、极大/极小、NaN/Inf（如果题目允许）
4. **语义边界**：RoPE、KV merge、mask、重复索引、非法索引、稀疏模式
5. **随机/模糊输入**：多个 seed，与 CPU oracle 比较
6. **输入只读**：运行前后输入 buffer 逐字节一致
7. **输出完整性**：不能被后续 kernel 覆盖、不能只写 UB 不写 GM
8. **多次运行**：冷启动/热启动、连续调用、不同输入顺序

建议保存：

```text
tests/
  case-<name>.json       # shape、dtype、attrs、seed
  case-<name>-out.bin    # 实际输出
  case-<name>-ref.bin    # CPU/参考输出
  summary.json           # max_abs/max_rel/worst、nonfinite、stress_bad
```

## 9. 最小 AscendC kernel 骨架

具体 API 以题目模板和 CANN 版本为准，但结构通常是：

```cpp
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void my_kernel(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    // 1. 解析 tiling
    // 2. 计算 blockIdx / subBlockIdx 的任务范围
    // 3. InitBuffer / 分配 UB、L1、L0
    // 4. DataCopy / Load / Mmad / Vector
    // 5. 同步、搬出、写回 GM
}
```

host 侧：

```cpp
// 1. 校验 shape/dtype/attr
// 2. 计算 tiling
// 3. GetWorkspaceSize
// 4. 分配 workspace
// 5. launch kernel
// 6. synchronize
```

不要把数值计算放到 host；host 只做 tiling、workspace、memcpy、launch、校验。

## 10. 构建产物绑定

每次实验记录：

- 源码文件列表 + SHA-256；
- 编译命令、CANN 版本、编译器版本；
- 生成的 `.so` / kernel binary SHA；
- 实际加载的 `ASCEND_CUSTOM_OPP_PATH`；
- 实际 tiling key、kernel 名、block dim；
- 运行日志和 profiler 目录。

否则很容易出现“改了源码但跑的是旧包”“平台实际走的是另一条分路”的误判。

## 11. 建议工作区结构

```text
workspace/
├── src/                  # 当前候选源码
├── tests/                # 正确性用例
├── bench/                # 计时脚本
├── build/                # 构建产物
├── logs/                 # 编译/运行/profiler 日志
├── results/              # 结构化结果 JSON/CSV
└── NOTES.md              # 每轮假设、改动、证据、结论
```

提交 ledger 不在工作区：默认 `~/.local/state/cannjudge/submit-ledger`，所有会话共享；不要在每个工作区各建一份 `.cann-submit`。每轮只改一个主要因素；失败也保留记录，避免重复踩坑。
