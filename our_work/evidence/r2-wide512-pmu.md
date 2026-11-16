# 完整生产包的同组流水与请求模型

| 输入 | 版本 | Task µs | 最长原生 AIV / block | 同组 AIC µs | 同组 MTE2 µs | 同组 Cube µs | 同组 FIX µs |
|---|---|---:|---|---:|---:|---:|---:|
| r2_sample | parent9307 | 40.800 | 38.028 / 9 vector1 | 35.058 | 29.044 | 7.247 | 16.353 |
| r2_sample | R2wide | 36.760 | 34.475 / 1 vector0 | 31.125 | 23.828 | 6.893 | 2.567 |
| rg_sh_r2_dirichlet | parent9307 | 44.180 | 41.152 / 16 vector1 | 38.599 | 31.190 | 7.258 | 18.833 |
| rg_sh_r2_dirichlet | R2wide | 42.400 | 39.376 / 16 vector0 | 36.565 | 28.615 | 7.129 | 2.269 |

逐组ready/credit等待、AIV搬运/Vector、请求字节、活跃核和算术强度见 [ANALYSIS.json](ANALYSIS.json)，原始60核计数器仍在各raw目录。

这些是未经重链接、未删除计算的原生产二进制。PMU replay 与平台、OFF分开；最长原生核记录没有统一起始时间，不能直接宣布绝对关键尾。各busy和wait不能相加，也不能把Task减busy当可删除空泡。请求字节为不同存储接口的逻辑请求，不直接当作实际HBM字节；Cube忙时占比不等于峰值MAC利用率。
