三版冻结候选本地比较完成；当前保留 joint，relay 与 relay+FMA 均无本地提速证据。提交仍暂停，0 POST/canonical/补交。

同一 SDK 从三份冻结组件取函数，只有 namespace 与私有启动选择不同；生产源码 SHA 已重新核对。official=official4，random=random2718 的真实 GL 与随机合法数据。每段 first+2 warmup+6 retained；first 是该段首调用，不能称 forced-cold。msprof application ai-core=off，两个短窗口分别顺序 joint/relay/FMA 与 FMA/relay/joint，共108个完整 Task。

| 输入/窗口 | joint first/warm µs | relay first/warm µs | relay变化 | FMA first/warm µs | FMA变化 |
|---|---:|---:|---:|---:|---:|
| uniform/forward | 52.30/18.53 | 56.04/19.56 | +5.56% | 57.04/19.98 | +7.83% |
| random2718/forward | 58.78/27.43 | 60.28/27.69 | +0.95% | 52.40/28.18 | +2.73% |
| uniform/reverse | 53.86/20.00 | 48.84/20.90 | +4.50% | 49.46/21.30 | +6.50% |
| random2718/reverse | 54.08/30.70 | 58.30/30.94 | +0.78% | 52.78/31.20 | +1.63% |

窗口间绝对耗时漂移明显，未合并掩盖；两窗排序一致。random relay差约0.8–0.95%，单独不足强结论，且没有观察到随机点收益；official回退约4.5–5.6%。FMA两个输入都更慢。首调用波动与warm结论分开保留。

只补一对 random2718 joint/relay 短 PMU，频率均1650MHz，Task35.380→35.740µs。相同最长AIV core7/vector0：time32.673→33.064，V11.018→11.661，MTE2 8.916→7.447，MTE3 3.351→4.413，ready5 6.021→6.455，ready6 9.305→7.892µs。AIV平均V9.981→10.711，MTE3 3.505→4.150，vector-stall0.090→0.509µs。最大忙时不能相加；不同PMU重放的Task也不与application绝对耗时混合。

这些计数支持分片增加的V/写回/队列工作抵消了供给等待下降这一方向，但不能断言某个descriptor就是最后关键事件。工具请求TimelineDetail后报Msopt dump kernel failed，仅PipeUtilization和MemoryUB成功，故没有逐指令证据证明前半MTE3完全被后半V覆盖。最初一次精确kernel-name过滤与mangled名称不匹配而无数据，已保留并排除；成功采集取消名称过滤，单进程只有一个版本。没有再补采。

relay与FMA源码、数学边界和生命周期文档保持冻结；本轮没有第四算法方向、full生产构建、平台提交或平台收益宣称。逐调用耗时和PMU原始CSV见 RESULTS.json。建议两新包继续停在候选状态，不依据本地结果替换已Pass joint。
