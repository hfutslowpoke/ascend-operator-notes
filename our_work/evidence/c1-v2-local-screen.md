# C1 exact-expert Work v2: minimal local screen closed

The frozen same ELF ran mode10(v1) and mode11(v2). No device/source changes, full build, platform POST, public-index write or third version. User submission pause remains in force.

| Fixture | v1 Task µs | v2 Task µs | Reduction |
|---|---:|---:|---:|
| official2 | 14.955 | 15.005 | -0.33% |
| random_seed9181 | 29.550 | 19.590 | 33.71% |

Each number is the mean of two window medians from one short AB/BA per fixture. Each window has1first+8warmup+8targets,8windows/136actual MIX Tasks fully captured; NPU lock46.125s. No partial cohort was pooled. First-call samples are preserved but not interpreted as controlled cold-cache performance. Uniform AB14.83→14.84 and BA15.08→15.17 are near a tie; random AB29.50→19.65 and BA29.60→19.53 both improve.

Random9181 paired PMU at1650MHz reports diagnostic Task43.10→28.16µs. The longest AIV moves from block1(40.351µs,READY33.613µs) to block5(25.392µs,READY18.515µs). Their corresponding AICs are36.847→21.362µs, MTE232.417→16.116µs; these refer to the respective critical groups, not a same-core subtraction.

| Same core | Metric µs | v1 | v2 |
|---|---|---:|---:|
| 1 | aic_time(us) | 36.847 | 17.455 |
| 1 | aic_mte2_time(us) | 32.417 | 13.567 |
| 1 | aiv_time(us) | 40.351 | 21.665 |
| 1 | aiv_scalar_wait_id5_time(us) | 33.613 | 13.100 |
| 5 | aic_time(us) | 18.071 | 21.362 |
| 5 | aic_mte2_time(us) | 13.375 | 16.116 |
| 5 | aiv_time(us) | 21.112 | 25.392 |
| 5 | aiv_scalar_wait_id5_time(us) | 16.747 | 18.515 |

The full raw per-core AIC/AIV/MTE2/READY, memory and cache rows are retained in the paired JSON. Busy/wait times overlap and are not added into saved wall time. The heavier block5 in v2 is expected when formerly concentrated real Work is spread across owners; critical maxima and final Task both improve in the random case.

Uniform remains16Work/max1; random9181 remains28Work but max5→2complete K rounds per core, while v1 consumer math and UB regions stay identical. This screen supports the intended supply-chain mechanism and keeping v2 as a candidate. It does not establish better production/platform performance. v1 remains the platform-backed reference.

All18prior true90/165 numeric calls, finite outputs, first-sync,512Bguards and4inputRO evidence remain valid for the frozen binary. Static9red+2high candidates are retained separately from the manual lifetime account.

Recommendation: retain this candidate for a future authorized evaluation; stop local work now. The full readable single-target source remains512837B SHA19d0ba44813453353cfdd239c90c133152486a0cb10938443fb936702424d87c. Four-point capacity is out of this task scope.

Machine-readable report: /mnt/workspace/groupmatmulgelu-regular-c1-sol2-20261114/delivery/C1-EXPERT-WORK-V2-LOCAL-SCREEN.json
Timing raw: /mnt/workspace/groupmatmulgelu-regular-c1-sol2-20261114/results/expert-work-v2-short-AB-BA/COMPLETE.json
PMU raw: /mnt/workspace/groupmatmulgelu-regular-c1-sol2-20261114/delivery/C1-EXPERT-WORK-V2-PAIRED-RANDOM-PMU.json
