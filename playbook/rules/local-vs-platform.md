# 本机 vs 平台：哪些能外推，哪些不能

## 1. 两套环境

| 项 | 本机 | 平台 |
|---|---|---|
| SoC | Ascend 910B3（物理 NPU 2 / ACL logical 0） | 赛事云端 910B（具体批次/频率以平台为准） |
| CANN | 9.1.0 | 以平台为准（可能是 9.0/9.1/其它） |
| 计时 | host/event/profiler 自选 | 平台 `Task Duration`/评分口径 |
| 输入 | 我们构造 | 官方 case（可能隐藏） |
| 结果 | `V` | `O`（detail/ranking） |

**本机通过 ≠ 平台通过；本机失败要分类。**

## 2. 本机结果分类

### 硬失败（平台大概率也过不了）

- 编译失败；
- 运行崩溃、`507035`/`507057`、非法 GM 地址；
- 输出形状/索引结构错、越界；
- 题面不允许的 NaN/Inf；
- 确定性语义错（tie、空段、填充、固定槽位、`rot^T`）；
- host 参与数值计算、结果重放、case-specific 分支。

### 软失败/不可外推

- 本地 CPU 参考与 kernel 的数值差很小；
- fp16 舍入/累加顺序差异；
- 本地参考公式与官方不同；
- 本地时延高/低；
- 本地 CANN 版本与平台不同；
- 本地只覆盖子集 shape。

### 通过但不可外推

- 只跑 32×32，不代表 8192；
- 只跑 fp32，不代表 fp16；
- 只跑单 batch，不代表多 batch；
- 只跑 forward，不代表 inverse；
- 只跑无空段，不代表空段。

## 3. 时延不可直接比较

本地 event/steel_clock 与平台 Task Duration 的差异来源：

- 频率（本地/平台可能不同）；
- L2 容量、调度、启动/同步；
- 输入分布、workspace 复用；
- 平台采样/聚合方式。

做法：

- 本地用 ABBA/多窗口、固定 warmup/repeat，报中位数/均值/全部样本；
- 平台只看 `Task Duration`/detail 的逐 case；
- 不用本地绝对时间推平台排名；
- 同一版本 A/B 才比较相对收益。

## 4. 提交决策

```text
编译/运行/结构/边界都过？
  ├─ 否：先修，不提交
  └─ 是：
      本地数值差是结构性/数量级？
        ├─ 是：先修
        └─ 否：
            有官方 tolerance/上游语义支持？
              ├─ 否：补证据/保守实现
              └─ 是：dry-run → 提交候选 → 平台 detail 判定
```

平台判 `Wrong Answer` 时，以平台 testcase/容差为准；本地结果不能反驳平台。

## 5. 记录模板

```text
本机：
  SoC/CANN/gcc/cmake:
  shape/dtype/seed/输入生成:
  源码 SHA/产物 SHA:
  build/run/结构/误差/时延:
平台：
  contest/problem/SID:
  status/valid/per-case:
  Task Duration/ranking:
结论：
  哪些是 V，哪些是 O，哪些仍是 N:
```
