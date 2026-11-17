# SegmentMinCsr 补充工程构建记录（2026-11-17）

## 对象

- 上游：`HIT1920/torch_scatter_npu`
- commit：`0199ab0d836ea07e26d2914179a22bf3c793b62e`
- 子目录：`Segment_Csr/SegmentMinCsr/SegmentMinCsrFrameworkLaunch`
- 用途：A2/A3（910B）最小 msOpGen/AclNN 构建骨架；**不是官方语义实现**。

## 环境

| 项 | 值 |
|---|---|
| 设备 | Ascend 910B3（物理 NPU 2 / ACL logical 0） |
| CANN | 9.1.0（`/usr/local/Ascend/cann-9.1.0`） |
| gcc | 11.4.0 |
| cmake | 3.22.1 |
| Python | 3.12.13（只用于数据生成脚本） |

## 构建

```bash
cd Segment_Csr/SegmentMinCsr/SegmentMinCsrFrameworkLaunch
bash install.sh -v Ascend910B3 -i /usr/local/Ascend/cann-9.1.0
```

结果：`install_rc=0`。

产物：

```text
CustomOp/build_out/custom_opp_ubuntu_aarch64.run
size   225941
sha256 94b1155e7028400ebc4944a4f10ef100872bf2d180ec2bafa5c2f4e36bbccbba
```

构建日志：[segment_min_csr_alt_build_20261117.log](segment_min_csr_alt_build_20261117.log)（SHA256 `efb8efc9b682a371677f92f2248073cebd6f7bae8b6fd3fddb485a748eaddd51`）。

## 证明什么

- 这套 A2/A3 msOpGen 工程能在 CANN 9.1.0 / 910B3 工具链上生成 910B 自定义算子包；
- `op_host` tiling、`op_kernel`、aclnn 接口、`custom_opp_ubuntu_aarch64.run` 打包链路可走通；
- 可作为后续 `ops-gnn` arch22 `segment_min_csr` 构建的对照骨架。

## 不能证明什么

- **不能证明官方语义通过**：该工程 `indptr` 是 int32、只输出 `out`、没有 `arg_out`、空段填 dtype 最大值；官方要求 int64、`out+arg_out`、空段 `0/E`、首遇 tie。
- 不能证明精度/性能；本记录只有构建结果。
- 不能把 910B3 构建外推到 910C 或 950。

## 运行状态

已把 AclNN 用例放入 `bin/npu-lock` 队列；截至本记录写入时仍在等待全局 NPU 锁，尚未得到运行结果。运行日志将另行归档；**在拿到日志前不写 `pass`**。

运行命令（待锁释放后执行）：

```bash
export ASCEND_INSTALL_PATH=/usr/local/Ascend/cann-9.1.0
source <install-path>/vendors/customize/bin/set_env.bash
export ASCEND_RT_VISIBLE_DEVICES=0
cd .../AclNNInvocationNaive
bash run.sh
```

样例只覆盖 `src (8,2048)`、`indptr int32`、`float32`、4 个 segment；即使 PASS，也只证明该工程的示例，不是官方 18 组测试。
