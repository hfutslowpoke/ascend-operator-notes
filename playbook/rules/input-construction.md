# 平台输入构造：没有隐藏数据时怎么造用例

目标不是“猜平台隐藏数据”，而是用公开题面 + 合法约束构造**能覆盖语义、能证伪实现**的本地用例。

## 1. 输入来源的优先级

1. 官方公开题面里的 shape/dtype/属性/测试点数；
2. 官方题面里的 case 表（点云题常给 N/M/nsample/范围）；
3. 官方参考实现/上游库的语义（pyg-lib、torch_cluster、AnyGrasp、PointNet++）；
4. 题面的结构化约束（单调、范围、无重复、有限值等）；
5. 自己构造的边界/随机/fuzz 用例。

不要从隐藏 testcase ID、平台返回的片段或他人成绩反推测试数据；也不要把“猜中某个 case”当成方案正确。

## 2. 通用 shape 族

| 维度 | 建议覆盖 |
|---|---|
| batch | 1、2、4；空 batch（如果题面允许） |
| N/M | 1、小、中、大；题面给出的上限/下限 |
| K/特征维 | 1、2/3、8/16、32、64、题面上限 |
| 序列/分段 | 单段、多段、空段、连续空段、全空 |
| 图边数 | E=0/1、小、幂律长尾、hub、孤立点 |
| 幂次长度 | 2/4/8/…/8192；非 2 幂（如果题面允许） |
| nsample/npoint | 1、小于命中数、等于 N、大于命中数（若允许） |

## 3. 按题目类型的边界

### 图/稀疏

- `indptr/rowptr` 单调不减，末位等于 E；E=0 全空图；
- 空行/空段：`out=0`、`arg_out=E`（SegmentMinCsr）；
- 行内 `col` 升序；COO 行主序、无重复；
- 重复索引（GatherCoo 允许；RandomWalk 的边可能重复）；
- hub 节点 + 孤立点；幂律度分布；
- 零和坐标保留（Spspmm）；数值抵消为 0；
- `int32` vs `int64` 索引边界。

### 点云

- 零命中：全 0（Ball/Cylinder）或 `(-1,-1)`（Radius）；
- 命中 1、`nsample-1`、`nsample`、`>nsample`；
- 严格边界：`d²==r²`（Radius 命中，Ball/Cylinder 不命中）；
- 所有点重合；对称点；正方形四点（FPS tie）；
- `ignore_same_index` 开/关；多 batch 不跨 batch；
- fp16 输入：先转 fp32 再算距离/阈值；
- Cylinder：单位 rot、非正交 rot、负行列式 rot；`rot^T·Δ`；
- FPS：`npoint=1`、`npoint=N`；全点重合应重复 0。

### 复数/FFT

- DC、Nyquist、冲激、单频、随机复数；
- forward + inverse；inverse 的 `1/(N·M)`；
- N/M=2、4、8、…、8192；非 2 幂（若支持）；
- 实部/虚部极值；NaN/Inf（题面禁止时不要造）；
- `rfft/irfft` 的 FP32-only 限制。

### 逐元素/复数算子

- 实数/复数 dtype 全覆盖；
- 0、负数、极大/极小、接近抵消；
- 广播/非广播；非连续输入（如果接口允许）；
- fp16 中间精度与输出 dtype。

## 4. 属性/默认值

- 每个属性至少测默认值和边界值；
- `radius=0`（Radius 无命中）、`hmin/hmax` 相等/反向（非法）；
- `nsample=1/N`；`npoint=1/N`；
- `inverse=true/false`；`norm=true/false`；
- 不要用默认值替代题面 case 指定的属性。

## 5. 随机与可复现

```text
seed 固定；记录 seed、生成脚本、shape、dtype、范围、分布；
随机图用幂律/均匀两套；
随机点云用 uniform、normal、重复点、对称点；
不要只跑一次随机；至少多 seed + 固定边界集。
```

## 6. 输入合法性

构造前先校验：

- dtype/layout/rank/最后一维；
- 索引范围、单调性、无重复（题面要求时）；
- 输入有限（题面禁止 NaN/Inf 时）；
- batch 边界不跨；
- 空 tensor/零长度是否符合题面；
- fp16 转 fp32 的位置；
- 输出槽位/填充值。

非法输入不要拿来“证明”实现错；先确认题面是否定义了该输入。

## 7. 本地用例分层

| 层 | 用途 | 是否提交 |
|---|---|---|
| `official-public` | 官方公开 shape/case 表 | 本地必跑 |
| `boundary` | 空段/零命中/边界相等/tie | 本地必跑 |
| `stress` | 大 shape/长尾/hub | 本地按资源 |
| `fuzz` | 随机 seed/极值/非法边缘 | 本地筛选，不直接当平台 golden |

## 8. 反模式

- 只跑官方示例，不跑空段/零命中/边界；
- 用 `torch.randn` 一次随机当全部覆盖；
- 把非法输入（越界 index、非单调 indptr）当合法 case；
- 用本地 CPU 参考的近似公式当官方 golden；
- 为了“过某个 case”加 shape/hash 分支；
- 只测 fp32，不测题面要求的 fp16；
- 只测命中，不测未命中/填充；
- 把 `atol/rtol` 调到能过，而不是按题面容差。
