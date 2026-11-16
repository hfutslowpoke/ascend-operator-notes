# 第三方来源与许可

官方仓库：[cann/cann-ops-competitions](https://gitcode.com/cann/cann-ops-competitions)。归档时主分支固定为 `776f2e552a125d3c210b497385f8b57292ebdc92`。

原始 [LICENSE](third_party/official/LICENSE) 原样保留。该版本 MIT 文本的版权年份/持有人仍为占位符，本库未擅自补写。各 PR 的作者归属、附带说明和可能存在的文件级许可应一并查看；不为未明确覆盖的内容额外授予许可。

公开最终提交位于 `third_party/shanghe_final`。每个 `PROVENANCE.json` 保存 PR、源仓库、固定 head commit、原路径、SHA256 和验证过的 Git blob ID。总表见 [public-final-sources.json](catalog/public-final-sources.json)。代码来自公开 PR，而非受保护的判题提交接口。

公开 PR 在下载时尚未合并。本库用于私密学习与内部检索，不宣称这些代码已获得赛事人工合规确认，也不把相似代码的归属问题当作本次技术归档已经解决。

我们自己的代码和工具放在 `our_work` 与 `tools`。其中可能使用比赛模板及 Ascend C API，相关权利仍属于原权利人。当前没有为整个集合统一设置一个覆盖所有来源的新许可证。

## 2026-11-16 外部算子调研

本轮新增的 12 个外部来源只做固定 commit 引用、源码行号定位和中文分析，**没有把第三方源码复制进本库**。登记见 [catalog/external-repositories.json](catalog/external-repositories.json)，缓存位于工作区外 `repository-research-cache/`。

许可证状态：

- Apache-2.0：`So-cean/mamba-ascendc`、`Gxj230958/TritonAscendBench`、`elstainniles/triton-operator-benchmarks`；
- MIT：`vLLM-HUST/vllm-hust-opset`、`Parity-LRX/ICTC-O3-HP-Ascend`、`U-Illll/ascend-eigh-operator`、`asfsdsf/S6`、`Yunle-Lee/CANN-Operator-test-competition-personal-solution`；
- 未声明许可证：`Hansenwuwuwu/ascend-c-competition-archive`、`mantuoluozk/kernelswift-2026-triton-kernels`、`52Xayah/triton-ascend-kernels`、`Be4ver/svd`。

未声明许可证的仓库仅用于私密学习/内部检索，不复制、不再分发其源码；公开可读不等于允许再许可。Be4ver 的本机构建只在工作区外的源码缓存副本中进行，本库只保存运行日志和哈希，没有加入第三方源码。作者成绩、官方公示分数和本库的分析都不是本机复现，也不构成规则合规或通用正确性证明。

