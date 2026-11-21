# 新会话启动提示词（可直接复制）

把下面这段发给新的 agent/会话，再把比赛链接、题目链接和目标补上。

```text
你接手一个 CANN/AscendC 算子开发比赛题目。请先读本目录的：
- README.md
- docs/01-platform-login-and-problem-discovery.md
- docs/02-submission-workflow.md
- docs/03-local-dev-build-run.md
- docs/04-common-pitfalls.md
- docs/05-performance-and-scoring.md
- docs/06-compliance-and-redlines.md
- docs/07-submit-checklist.md

比赛/题目：<填赛事 slug 或链接>
目标：<通过精度 + 降低完整 Task 耗时 / 先拿到 Pass>
可用账号：<标签，例如 A>
是否允许平台提交：<是/否；若否只做本地>

工作原则：
1. 先读题面和官方模板，确认 dtype/layout/shape 范围/精度门/特殊语义。
2. 不假设隐藏输入分布；不针对具体 case 写 hash/白名单/快路径。
3. Host 只做 tiling/workspace/memcpy/launch/校验，不做数值计算。
4. 每轮只改一个主要因素；记录源码 SHA、产物 SHA、实际 tiling key、频率。
5. 所有设备运行（run/bench/profile）串行，使用 tools/npu_lock.sh；编译/CPU 校验可并行。
6. 正确性优先：官方样例、边界、随机/fuzz、输入只读、输出完整。
7. 性能用完整 Task + ABBA/多窗口；局部 busy 不等于总时间。
8. 提交前 dry-run；同一源码不重复 POST；拿到 SID 先落盘，之后只 GET。

请先输出：
- 题目理解（输入输出、约束、精度、特殊语义）
- 模板可编辑文件和提交格式
- 本地环境检查结果
- 第一版正确性计划
- 第一轮性能基线计划
```

## 题目 intake 模板

```markdown
# 题目 intake

## 1. 基本信息
- 赛事 / contest slug：
- 题目 / problem slug：
- problemId：
- code_template：
- 评测硬件 / CANN：
- 提交额度 / 采用规则：

## 2. 输入输出
| 名称 | 形状 | dtype | layout | 说明 |
|---|---|---|---|---|

## 3. 约束
- shape 范围：
- dtype/attr：
- 对齐：
- 特殊语义（重复/非法索引、mask、RoPE、KV merge、空块、尾块）：
- 输入只读 / 输出要求：

## 4. 精度门
- 参考实现：
- atol / rtol / matched_ratio / max_abs：
- NaN/Inf 规则：

## 5. 模板
- editable 文件：
- 提交 path：
- 是否需要兼容字段：
- 模板自带测试命令：

## 6. 计划
- 第一版正确路径：
- 性能基线：
- 风险：
```

## 交接模板

```markdown
# 交接

## 当前状态
- 最好版本：
- 源码 SHA：
- 产物 SHA：
- 实际 key / kernel：
- 平台 SID / 成绩：
- 采用/回退：

## 已验证
- 正确性：
- 性能：
- 失败候选：

## 下一步
1.
2.

## 不要重试
- <负结果及原因>
```
