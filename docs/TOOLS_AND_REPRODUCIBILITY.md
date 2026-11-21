# 工具、环境与复现边界

## 环境身份

历史目标为 Ascend 910C 单 die，arch22 / `dav-2201`，20 AIC、40 AIV；常用记录频率 1650 MHz。L1 512 KiB、L0A/B 各 64 KiB、L0C 128 KiB、UB 192 KiB。具体 CANN、驱动和编译器版本应由原构建日志核对；不要把后来的技能库版本反写为比赛当时环境。

Ascend C/CANN SDK 提供 `ascendc_library`、ACL runtime、tiling API 和 platform 查询。arch35 的 L0C→UB、Reg/VF 能力不能直接移植到 arch22。官方模板接口/设备约束见 [题目](../competition/grouped-matmul-gelu/official/03_university/CANN-ops-challenge-2026-shanghe/final/problems/GroupedMatmulGelu.md)。

## 实际使用的工具

| 阶段 | 归档工具 | 作用与注意点 |
|---|---|---|
| 环境检查 | `tools/competition/tools/check_env.sh` | CANN、Python、设备依赖；版本需现场保存 |
| NPU 互斥 | `tools/competition/tools/npu_lock.sh` | `flock` 只包实际设备运行（run/bench/profile）；编译/CPU 校验并行，不占锁；独立工作区仍共享设备锁 |
| 构建/调用 | `competition/grouped-matmul-gelu/our-submission/CMakeLists.txt`、`main.asc`、`bench.asc` | 完整 Task 驱动，需先适配相对路径和 CANN 安装 |
| 数值/边界 | `acc_suite.py`、`regress.py`、`verify_case.py`、`verify_batch.asc`、`reference.py` | 实际 GL、输出毒化/guard、输入只读、不同数值分布 |
| 普通计时 | `bench_all.py`、`bench_gen.py` | 完整 kernel、多组输入和配对，脚本可能引用原工作区 |
| 深采集 | `prof_op.py`、msprof / msopprof | Task、AIC/AIV Pipe、UB、Memory、Cache，采集超时清理 |
| 分层模型 | `model_owned_roofline.py` | owner 级别有效工作/数据量，模型值不是实测带宽 |
| 二进制闭包 | `audit_device_elf.py` | 检查实际 ELF 入口、资源和被编译组件 |
| 组件提取 | `component_tool.py` | token/brace-aware 提取、逆向恢复与差异范围证明；不替代编译 |
| 登录/模板/提交 | `cann_client.py`、`cann_package.py`、`cann_submit.py` | 历史接口工具，凭据外置；全局 ledger、账号 12:00 窗口额度；比赛结束后不执行 POST |
| 后台结果 | `submit_background.py` | accepted SID 持久化、配额/冷却、结果轮询、成绩归属 |
| 排名快照 | `scoring_snapshot.py`、`ranking_archive.py`、`watch_platform_ranking.py` | 同期榜单、服务器时间/SID 对应、留存变化 |
| 评分对照 | `compare_score_snapshots.py` | 防止跨评分时期简单比总分 |
| 公共登记 | `point_updates.py` | 先追加日志、再更新 owner 索引；锁与旧 SHA 防覆盖 |
| 分路检查 | `dispatch_policy.py` | 辅助检查参数族分路，不构成全部规则或数值证明 |

表中分析 Python 文件位于 [tools/analysis](../tools/analysis)，比赛客户端位于 [tools/competition/tools](../tools/competition/tools)。starter kit 原始说明是历史快照，冷却策略、接口状态和配额可能已经改变；本次没有重新提交或修改比赛服务。

## 计时口径不能混用

比赛曾使用普通完整调用 32 warmup / 16 target、多进程中位数、短 AB/BA 或 ABBA；PMU 通常是单独 replay。C1 v2 的短筛选是每窗口 1 first + 8 warmup + 8 target、两个方向，并非长时间稳态性能保证。每份 evidence 的协议应单独读取。

后续技能库 `ops-profiling` 推荐 10 组 warmup→target、每例中位数等标准协议，这是未来复用建议，不是对历史实验口径的改写。不要把 PMU Task 减普通 Task 当成一个固定测量开销，也不要把 first-call 简单称为受控 cold-cache。

## 用过的分析技能

`ascendc-performance-best-practices`、`ascendc-tiling-design`、`ascendc-perf-optimize`、`ascendc-api-best-practices`、`npu-arch`、`ascendc-sync-audit`、`ascendc-precision-debug`、`ascendc-crash-debug`、`ascendc-runtime-debug`、`ops-profiling`、`ascendc-env-check`。

这些技能帮助查询架构/API 限制、分析 tiling、同步与 PMU。它们不是比赛成绩证据；最终必须回到具体源码、数值测试和平台结果。本库未复制整套后续更新的技能库，避免把版本变化混入比赛记录。

## 复现路径

1. 在新机器安装对应 CANN/驱动、CMake、Python、NumPy、ACL 开发依赖，记录设备/单 die 身份。
2. 阅读最终目录 `CMakeLists.txt`。这是当时 sample/bench/verify_batch 驱动，构建需要比赛模板环境和正确工具链；若路径发生变化先迁移。
3. 先跑数值及边界，再做完整 Task 对照；第三方代码也先确认路线、精度与 workspace 寿命。
4. 单独采 PMU，保留对应输入、源码/ELF 哈希和关键核数据，不将 profiler 的数值直接替换普通计时。

此次归档仅验证源文件完整性、JSON/Python 语法、来源和敏感文件排除，没有重建 NPU 二进制。未包含原始测试数据、所有数千次实验、巨型 PROF、设备 binary。原报告中的绝对路径是历史来源；若报告引用了未选入的小文件，需回到旧工作区，不应理解为本库已包含全部闭包。

## 今后保存一次实验的最小记录

建议一个目录保存：问题与一般输入族、父本/候选源码 SHA、Host 选择、完整构建结果、运行入口/ELF 身份、固定输入生成器和种子、数值/guard、普通 Task、必要的 PMU 字段、平台 SID/逐点耗时/时期、结论与反例。

大文件可以另存对象存储或压缩包，但索引、哈希和关键报告必须保留。仓库中保存逻辑和证据，不保存凭据、浏览器状态及未初始化输出内容。

