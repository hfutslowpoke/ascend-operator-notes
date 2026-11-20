# 语义覆盖：平台 Pass 只证明“跑过的 case”

> 一句话：题面里的输入、属性、模式，代码不一定都实现了；平台 Pass 只能证明评测算例
> 覆盖到的那部分。入库/复用前先做“题面 → 代码”覆盖表。

## 1. 典型缺口

- 题面有可选输入，op def 没声明；
- 题面有属性（mask/rope/layout/seq_len），host 直接硬编码成 0；
- 题面有 dtype（bf16），实现只注册 fp16/fp32；
- 题面有变长/空段/边界，代码只跑等长/非空；
- 题面有 KV merge / causal / inverse，代码只实现 forward。

## 2. 本次案例

### DenseLightningIndexerGradKLLoss

官方题面包含：

- `queryRope/keyRope`（Dr=64）与 RoPE 反向；
- `sparseMode=3`（rightDownCausal）；
- `actual_seq_lengths_query/key` 变长。

代码的 op def 只声明 5 个输入 + `scale_value`；host 硬编码：

```text
dRope = 0; hasRope = 0; sparseMode = 0; useSeqLens = 0;
```

平台 7/7 Pass 只说明这 7 个 case 没触发这些路径；不能写成“题面全集已实现”。

### Sparse Softmax

官方题面列了 `num_nodes` 可选输入、bf16；op def 没有 `num_nodes`，且明确不注册 bf16
（工具链拒绝 bf16↔fp32 Cast）。平台 Pass 只证明评测的 index/ptr、fp16/fp32 case 通过。

## 3. 做法

1. 建表：题面的每个输入/输出/属性/模式 × `implemented / tested / missing`。
2. 对 missing 项：
   - 要么实现；
   - 要么 fail closed（明确报错），不要静默当成默认值；
   - 要么在 README 里写“未覆盖”，不要混在“已通过”里。
3. 平台 Pass 后，逐条回填“哪些 case 覆盖了哪些模式”；没有证据的写 `unknown`。
4. 复用别人的代码时，先看 op def/attrs 是否比题面窄；窄了就不能直接当 full solution。

## 4. 检查清单

```text
[ ] op def 的输入/输出/属性是否覆盖题面全部？
[ ] host 是否把某些属性硬编码成 0/默认值？
[ ] dtype 是否全部注册？没有注册的，是否在 README 说明？
[ ] 变长/空段/causal/rope/KV merge 是否有测试？
[ ] 平台 Pass 的 case 是否真的触发过这些模式？
[ ] README 是否区分“已实现/已测试/未覆盖”？
```
