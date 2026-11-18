# SegmentMinCsr 补充工程构建与运行记录（2026-11-17/18）

## 对象

- 上游：`HIT1920/torch_scatter_npu`
- commit：`0199ab0d836ea07e26d2914179a22bf3c793b62e`
- 子目录：`Segment_Csr/SegmentMinCsr/SegmentMinCsrFrameworkLaunch`
- 用途：A2/A3（910B）最小 msOpGen/AclNN 构建/调用骨架；**不是官方语义实现**。

## 环境

| 项 | 值 |
|---|---|
| 设备 | Ascend 910B3（物理 NPU 2 / ACL logical 0） |
| CANN | 9.1.0（`/usr/local/Ascend/cann-9.1.0`） |
| gcc | 11.4.0 |
| cmake | 3.22.1 |
| Python | 3.12.13（只用于数据生成脚本） |

## 构建（2026-11-17）

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

## 运行（2026-11-18）

运行命令：

```bash
/workspace/acw/bin/npu-lock bash -lc '
  export ASCEND_INSTALL_PATH=/usr/local/Ascend/cann-9.1.0
  source <install-path>/vendors/customize/bin/set_env.bash
  export ASCEND_RT_VISIBLE_DEVICES=0
  cd .../AclNNInvocationNaive
  bash run.sh
'
```

样例：`src (8,2048) float32`、`indptr (5,) int32 = [0,0,5,8,8]`、4 个 segment，其中第 0/3 段为空段；`run_rc=0`，输出：

```text
[INFO] Successfully completed SegmentMinCsr test case!
[INFO] out tensor compare pass
INFO: SegmentMinCsr precision PASS!
```

数据/结果哈希：

```text
input/src_single_case.bin        sha256 2d0fa35dcd985aae4c7ca26b9f9f7239ad55047e213f2b08340b70a5a38b8ecb
input/indptr_single_case.bin     sha256 10a74a3993d72f2a0e932e7f16000d93802d8dec93d23cd6025e8fb130f01b75
output/golden_single_case.bin    sha256 a447f88ef3b83c3d1a7b1de7b2a5d73232c65c548181a596696ea2b35d25721d
output/output_y_single_case.bin  sha256 a447f88ef3b83c3d1a7b1de7b2a5d73232c65c548181a596696ea2b35d25721d
```

运行日志：[segment_min_csr_alt_run_20261118.log](segment_min_csr_alt_run_20261118.log)（SHA256 `48b9a979e8bd7b8dead43fc19a8e272eb7badc70619d14e8923f4d43121417b5`）。

## 证明什么

- 这套 A2/A3 msOpGen 工程能在 CANN 9.1.0 / 910B3 工具链上生成 910B 自定义算子包；
- `op_host` tiling、`op_kernel`、aclnn 接口、`custom_opp_ubuntu_aarch64.run` 打包链路可走通；
- AclNN 示例能在 910B3 上执行，`golden == output`，包括第 0/3 段空段的情形；
- 可作为后续 `ops-gnn` arch22 `segment_min_csr` 构建/调用骨架。

## 不能证明什么

- **不能证明官方语义通过**：该工程 `indptr` 是 int32、只输出 `out`、没有 `arg_out`、空段填 dtype 最大值；官方要求 int64、`out+arg_out`、空段 `0/E`、首遇 tie。
- 不能证明官方 18 组 shape、fp16 或性能；样例只有 `8×2048`、float32、4 段。
- 不能把 910B3 结果外推到 910C 或 950。

## 结论

**补充骨架 BUILD+RUN PASS（仅示例用例）**；主 `ops-gnn` 实现仍为 `not-run`，官方语义仍需按 `adapters.md` 与 `docs/research/MFG2026_ALGORITHM_REFERENCES_20261117.md` 改造后验证。
