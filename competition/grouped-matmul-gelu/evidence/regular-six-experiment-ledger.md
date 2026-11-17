<!-- regular6-final-paused -->
# 本轮结束与暂停

root及所有8个子会话已完成当前轮，暂停开发和新提交。

唯一授权集成提交SID `6af7cec56bd0f9be37f49def` 编译失败，未产生任何点耗时。原因是root集成C1 namespace时放到了StageB/StageL0定义之前；不是600秒超时。

已移动定义顺序，完整本地编译通过（75.49s）。修复源码：`/mnt/workspace/groupmatmulgelu-regular-worker/opt/candidate_C1v2_R3v4_on_best20_161ac_orderfix_v2/kernel.asc`，SHA256 `ffd5609c3371f4fe16d90805bf741f9895812a0aa398ed6bcdcfd4425a464a12`。修复包未上NPU/平台、未追加POST，保留为候选。

| 测试点 | 用户提供最低观测 µs | 当前161ac集成 µs | 差距 |
|---|---:|---:|---:|
|1/C0|39.741|43.724|+10.0%|
|2/C1|22.947|23.804|+3.7%|
|4/C3|22.310|25.597|+14.7%|
|5/C4|44.581|46.891|+5.2%|
|8/R3|42.768|43.941|+2.7%|
|12/R2|37.958|39.541|+4.2%|

这些最低值来自不同提交，不能作为同包同时可达到的保证。当前C1/C3/R2/R3 namespace与原四点已选组件token一致；没有误接入上一轮负候选。C0/C3差距明显，但目前证据不足以判定是集成改变或平台测时条件。C0/C4保持既有选用供给实现。

R3v4本地12调用严格验证通过，uniform/random短ABBA约改善1.86%/2.59%；尚无平台收益。C1v2保留强本地随机收益候选，不替换当前平台更低观测的v1。

保留当前母包和公共点索引，不晋升候选。完整证据见同目录RESULT.json、ORDER-FIX.json、ORDER-FIX-BUILD.json、SELECTION.json；公共日志先行追加，无索引变更。

所有本轮NPU实验和本地编译均已退出；遗留已接收SID的只读观察不是开发或新提交。用户明确恢复前不启动下一轮。

<!-- regular6-onecheck-current -->
# 2026-11-14 一次组合平台检验完成

用户仅授权一次候选组合评估；源码1506029b，SID6af7cb756bd0f9be37e896a6，20/20Pass。root与子会话均恢复无新POST，仅登记。

| 测试点 | 组件 | 四点母包734f µs | 本次1506 µs | 选择 |
|---|---|---:|---:|---|
| 2 | C1 expert Work v2 |25.400667|24.314000|保留优先候选；未刷新v1最低23.170667，不晋升索引|
| 4 | C3 pure relay |22.330500|26.697167|不晋升，保留joint|
| 12 | R2 consumer issue |37.957667|42.974333|不晋升，保留原native-wide consumer|

未改动点1/5/8也慢约6–9%，不按这些点修正目标耗时，不把整包分数归因于单点。独立C1v2紧凑候选c4193da1已恢复C3/R2原组件，仅静态登记，未宣称完整包已Pass。

权威结果：delivery/round281/ONECHECK-PLATFORM/{RESULT,SELECTION,C1-ONLY-INTEGRATION-CANDIDATE}.json。各点当前发布选择不变。后续再次提交需要用户授权。

<!-- /regular6-onecheck-current -->

<!-- current-readable-platform-checkpoint -->
# 当前补充：可读单点平台与资源分配

更新UTC：2026-11-14T01:49:38.420043+00:00。机器状态以 ROUND-STATUS.json（`ROUND-STATUS.json`） 为准；下文旧轮次报告保留原证据。

- C0、C4：子会话本轮均已完成，依用户授权暂停这2点开发，保留母包与全部源码/证据；继续C1/C3/R2/R3。最新未改C0/C4观测39.741/45.044167µs，tbest37.420833/41.663833µs；这只是相对优先级，未声称roofline已到。
- R2：可读native-wide f1cc，SID6af7bb4a6bd0f9be37acf10e，20/20Pass，44.254333µs对固定可读母45.354333µs约低2.43%；已按实际目标耗时与同方向本地证据选入公共索引，波动未排除。首attempt明确429未受理，另起job只产生一个acceptedSID。
- C3：联合当前V/下一块ready等待及OQ双buffer，可读5e7039，SID6af7bd136bd0f9be37b36ba7已20/20Pass；目标24.460333µs对母25.340833µs约低3.47%，已登记当前候选，首样尚未宣称稳定收益。
- R3：实际GL加权连续专家段、段内驻B，新SDK随机46.145→30.395µs；可读8204，SID6af7bda36bd0f9be37b5796f已20/20Pass，R3=44.844167µs对固定母51.7945µs，单样低13.42%；已登记当前候选，旧原实现也有45–46µs，不能宣称稳定13.42%。
- C1：独立双CQ及held输出最后读者退休新流水10轻数值调用通过，178KiB UB；原AIC device byteexact，可读c3ef0b已排平台，未获结果前不登记收益。旧fullA/fullB与K256/N128负路线不重做。
- 提交：global0，按账号180秒，60秒首查/20秒后查；12项聚焦离线通过。源码暂停宏编程，只保留必要宏；本轮C0/R2/C3/R3可读提交源均0宏。新5点归独立任务，不跨点开发。

<!-- /current-readable-platform-checkpoint -->

# 当前流水优化与平台选择

负责人 regular6：C0、C1、C3、C4、R2、R3。当前共同母包9307来自已经平台验证的C0/C4宽供给组件。历史总分均保留其原快照，不能跨评分时期直接相减；已验证相同SID、相同耗时记录的总分发生变化。统一比较参见 评分参考（`../round264/SCORING-CHANGE/REPORT.md`）。

## 已登记的公开组件

公共索引 POINT-UPDATES.md（`/mnt/workspace/groupmatmulgelu/POINT-UPDATES.md`） 记录先日志后晋升，用户随后明确同意登记C0/C4。

| 点 | 原e902两次 µs | 9307两次 µs | 当前选择 |
|---|---|---|---|
| C0 | 48.694333 / 49.397667 | 42.520667 / 43.604167 | 改善复现，选9307共享组件 |
| C4 | 48.171 / 49.154167 | 44.580667 / 48.911 | 首次改善、复测回原母区间；选同一共享组件并完整保留波动 |

源码：submit_C0C4_wide_supply_on_e902_v1/kernel.asc（`/mnt/workspace/groupmatmulgelu-regular-integration-sol2-20261114/opt/submit_C0C4_wide_supply_on_e902_v1/kernel.asc`），SHA `9307cda904f88c15dbdad28b32e7710a9de6a7c371f55075748382f8e0bfc3db`。只替换共享namespace，Host/其他13点逐字节保持；full463.70s。短Work≤64将两个N256消费面板合为N512 B供给，A整K驻留，B三槽K256；两N最后MMAD先发再等GM C credit/FIX。长Work还包括已明确记录的499式credit延后，不能称为只改短分路。完整true-K INT32→FP32反量化/GELU→最终half，真实GL和输出行序。

首次结果（`../round254/C0C4-WIDE-PLATFORM/PLATFORM-RESULT.json`）、同源复测（`../round254/C0C4-WIDE-PLATFORM/REPEAT-RESULT.json`）。生产driver原来的两个env-poison请求实际都是0x5a，READY标签已收窄为samepoison重复；逐bit/guard/输入只读与平台Pass仍有效。SDK真实双poison单独有证据。没有把错误标签扩展为全输入初值独立性证明。

## 当前单目标平台与构建

| 方向 | 当前证据 | 选择 |
|---|---|---|
| C1 dual64 UF独立C | Pass60.50，C1=28.8605 µs，原24.214/24.717 | 不晋升；本地IO8/C8也慢约6% |
| C3 输出Y借用 | Pass58.21，C3=27.527167 µs，原22.566833/25.284 | 不晋升，保留444 |
| C4 dual64早FIX旧消费 | Pass59.95，C4=48.164167，原48.171 | 持平，不晋升 |
| R3 splitK初始供给 | 原e902单目标Pass61.22，原母61.36/60.26 | 候选保留；未证明单个隐藏点改善 |
| R2 原448消费+native N512供给 | 第一次Pass但总分身份缺失；同源补测SID6af7a8dd…为Pass63.54、身份匹配 | 相对旧60.42不能直接判收益；新鲜9307对照SID6af7ae73…正在运行 |
| C1 zeroC跨Work联合流水 | Pass60.97，C1=28.027333µs | 慢于母包，不晋升 |
| C0 K128六B槽事件退休 | Pass61.25，C0/C4=43.837667/49.324333µs | 不晋升，保留9307 |
| C4 两M64、AIV各自owner半块 | Pass62.76，C4=48.734333µs，C0=44.554333µs | 目标公开耗时没有改善，不晋升 |
| C0 当前pair发布先于未来B就绪依赖 | 单ProducePairs改动；full466.714s，14次真实双poison生产调用通过 | 一次公开验证就绪，先收回新鲜母包身份结果 |
| C1 整A2驻留、整B1地址复用 | 完整实现和严格数值通过；三个GL共1176Tasks，最终版warm慢29%–52%、IO8慢36%–45% | 负结果闭合，无full/POST |
| C4 K768×N512驻留、INT32部分和归并 | 29次数值、1176Tasks；三GL均慢23%–53% | C中转翻三倍，负结果闭合，无full/POST |

来源与SID见 实时状态（`ROUND-STATUS.json`）。C1/C3负结果、R3观察已经追加公共日志，不将较高整包分数代替公开目标耗时。R3完整ABBA对实际454：uniform28.98→26.99、random46.86→45.36、Dir44.83→43.50µs；此局部证据没有被写成隐藏平台收益。metadata返回20但只有15例记录，隐藏推算保持关闭。

## 联合设计的实际收益和代价

C0宽供给/N256消费的收益出现在本地地址旋转条件，warm几乎不动；平台C0收益已经复现。它改善DMA服务宽度、A复用和Cube/FIX/消费credit的发射关系，不能解释为单纯少发指令。

C1旧UF增加独立C，地址条件改变后暴露负收益。新v2把C写进当前Work自有Y地址，最终half先留UB；不同Work地址域不重叠，下一Work的A/B预取和第一N大部分MMAD可以先发，直到首FIX前再排空上一Work两个末次读者并发flush事件。约162KiB/AIV UB，额外独立C为0。它恢复v1随机输入退化的大部分，但相对原469的完整调用仅约−3%至+4%的混合信号，必须交平台。

R2保留448原AIV、metadata、GM C和任务顺序，供给改为A整K驻留、B三槽K256×N512宽DMA，L1共448KiB；两个N最后MMAD发完再等C credit/FIX。实际448对照下uniform/Dir的IO8/C8分别47.070→43.340、45.555→42.760µs，warm只改善2.56/0.52%。生产验证按原Host实际分路，未把unseen P2/P4强迫进入新组件。

R3将初始一个K512×N512请求拆成两个K256独立ready，首块就绪后立即开始K128 MMAD，后一块并行供给；后续各B块在最后MTE1读完成即复用。不是减少B总字节；增加请求次数换初始依赖缩短。完整A2驻留和另一FP32消费表达共同记录为联合改动。

C3已完成两种联合方案。输出借用+四L1供给的同设备1764次对照，相对已有alias只再改善约0.35%至3.9%，不足以跳过alias已存在的平台负结果。原444任务归属+四L1+chunk8消费另有56次数值、1764个完整Task，三分布均慢6.75%至17.05%，已收束，无full/POST。最小同核PMU：Task22.02→24.52µs，AIV尾19.436→22.070；READY等待确实下降，但Vector9.855→13.512、MTE3 1.568→4.795；双小CQ使末C退休更晚，同核AIC credit等待也升高。busy与wait有重叠，未相加。这个结果否定当前小块资源组织，并不证明联合流水没有空间。

C4两M64 owner路线已完成本地和平台。它让Cube/FIX可交错，每AIV只负责自己的64行半块，并保留广播ready和未拥有半块的空读者credit；真实末读者完成才退槽。短Work保留9307，GM C不增加。平台C4=48.734333µs，没有超过已选择版本的两次观测。另一个K768宽驻留方案降低MTE2忙时，却使C写读从6/6MiB到18/18MiB，FIX和AIV读取及整数归并变成更重的依赖，因此关闭。

C1整A2/整B1路线真正实现了L1地址退休复用：M0 A已搬入A2后L1给整B，B前块最后MTE1读完再覆盖成M1 A。数值正确但最终仍负；完整Work的A2/B2载荷从256/512KiB变成128/1024KiB，总MTE1载荷增加50%，第一次MMAD还要等512KiB B就绪。最终配对PMU Task22.30→26.02µs，MTE2 10.722→7.547、FIX4.199→1.667下降，MTE1 4.404→6.795及AIV READY13.601→17.393上升。该表示未改善最终调用，停止此方案。

## 硬件、证据与下一步

本机20AIC/40AIV，L1 512KiB，L0A/B各64KiB，L0C128KiB，UB192KiB。层级roofline和纯DMA服务只给供给/计算/写回的参照下限；完整Task还受有限槽位、最后读者、地址复用距离和启动依赖约束。AIC和AIV最大busy可能来自不同核，不能拼成一条真实关键链。

纯B测量固定字节：C0 N256/K256固定地址16.332、轮换31.992µs/轮，宽N512为11.749/28.461；窄N128/N64反而更慢。C3新consumer的PMU以同一关键block交叉验证了“减少READY等待仍变慢”，因此选择依据始终是完整调用，再由固定SID平台公开目标耗时验收。

C0新时序原生产包配对PMU：均匀42.88→41.40µs，seed947随机45.96→46.50µs；GM/L1、C写读、Y请求字节不变，单次replay仍是混合证据。见 同组记录（`../round265/C0-PUBLISH-CARRIER-PMU/REPORT.md`）。一次公开测试已安排在新鲜母包结果/榜单归属捕获完成之后自动启动，不重试POST，尚未据此更新索引。

新增R2/R3原生产二进制PMU共8组，均不改二进制、不移除计算。R2原9307→native宽供给，sample40.80→36.76µs、Dir44.18→42.40µs；同组最长AIV所在AIC的MTE2分别29.044→23.828、31.190→28.615µs，请求字节未变。R3原e902→splitK，uniform45.08→39.96、random1709 65.74→58.42µs，同样未减少请求字节。详见 R2同组分析（`../round262/R2-NATIVE-WIDE-CARRIER-PMU/REPORT.md`）、R3同组分析（`../round263/R3-SPLITK-CARRIER-PMU/REPORT.md`）。单次PMU replay不等于平台耗时，也不据此晋升隐藏点。

公开点按用户最新授权绕过600s等待；隐藏点保持正常提交队列。当前先等待已接受的新鲜9307对照获取结果，再发root下一公开测试，避免自己覆盖隐藏对照身份。所有实际NPU调用持 `/tmp/cann-npu.lock`，完整TU一次一个，源码与日志受保留清单保护。平台提交不携带本地验证observer或诊断环境。继续保留新测试case和所有耗时开放后的评测机会。

旧实验、原始反向结果、地址与PMU目录均保留在 NATIVE-ANALYSIS.json（`NATIVE-ANALYSIS.json`） 及 本轮早先报告（`REPORT-before-public-win-20261114.md`），不回填公共日志历史。
