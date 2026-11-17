# 实际最终提交源码

`kernel.asc` SHA256：`161ac0ff2ab1920ca3de3e4f5d8b1b4d19a9a720c2781dcbc5c065f989b0b887`。

SID `6af7d84c6bd0f9be3716f4c6`；接受时间 2026-11-14 03:41:00.774 UTC。

这是从实际提交任务和 11 月 16 日榜单 SID 对应确认的版本。`main.asc`、`bench.asc`、`verify_batch.asc`、`CMakeLists.txt` 来自该源码当时的 trial 工作区，源码逐字节一致；`reference.py` 是我们的本地参考实现，不是从官方取得的 golden。

未包含 build-npu、输入张量、输出文件和 NPU 二进制。此归档任务没有重新编译或运行 NPU。原构建成功记录位于 [actual-final-build.json](../evidence/actual-final-build.json)。

`CMakeLists.txt` 使用 `find_package(ASC REQUIRED)`；应先配置对应 CANN 工具链。sample/bench 的参数约定直接查看各驱动 `main` 函数，不保证历史绝对路径在新机器存在。

