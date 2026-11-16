# C4 partial768 versus exact9307 — negative closed

The single resident-wide-B / exact INT32 Kpartial SDK is numerically correct but slower in every complete-call cohort. Do not promote. No full production build, POST, public index or other worker source changed. Current9307 is the control; these results are not versus499/494/e902.

## Frozen design

One480KiB L1 arena holds A128×768 and B768×512. Two M64 CO buffers and shared K128 B loads evaluate four M64×N256 quadrants. K2048 is768+768+512 and K1536 is768+768. Each AIV retains four32×256 INT32 sums; every trueK contribution is merged by INT32 Add before any FP32 scale/GELU/final FP16. UB185KiB uses separate32KiB C input,16KiB FP32 scratch,8KiB output and1KiB scale. L1 nextpart supply starts at the current part last MTE1 reader release. GM C and Work ownership retain9307 addresses and realGL traversal. See LIFETIME.md for every reader/overwrite event.

The uniform traffic cost is explicit: A4→8MiB, B24MiB unchanged, C write6→18MiB and C read6→18MiB, final Y3MiB unchanged. A is reread per Npair; the larger B transfer setup does not remove that cost.

## Numerical proof

29 strict calls:9classes × control poison90 / candidate poison90 / candidate poison165, plus2integer preflights. All outputs are bitwise equal to same-SDK9307; the first8real cases also equal frozen9307 outputs. Finite and golden tolerance pass. Each call enforces first stream sync, true output poisoning,512B Y guards and all four inputs unchanged. Cases cover uniform/random/Dirichlet, empty experts, severe skew, M1, odd N256 tail, K1536/K2048, N256/768/1024/1280, and >2^24 partial integer sums across K768/K1536 boundaries with cancellation in the final512. No host runtime numerical work was added.

## Complete-call OFF

Every row is a fresh ABBA[9307,new,new,9307] cohort. Each window includes first+32warm+16target complete MIX calls; median of16target Tasks, then mean of2window medians per arm. All1176Tasks/24windows are captured. IO8 rotates X/W/scale/GL/Y among8real copies while C is one fixed guarded workspace per process. All49address lines per window, distinct-pointer counts, guards/inputRO and final outputs are checked. The address driver relinks the frozen SDK object; all nonsymbol ELF sections, including device fatbins, are byte exact. No old partial timings are pooled.

| GL | IO copies | 9307 µs | partial768 µs | Time increase |
|---|---:|---:|---:|---:|
| uniform | 1 | 29.210 | 43.835 | 50.07% |
| uniform | 8 | 49.705 | 61.405 | 23.54% |
| random5001 | 1 | 34.035 | 52.090 | 53.05% |
| random5001 | 8 | 52.575 | 70.415 | 33.93% |
| dir_extreme | 1 | 35.315 | 53.640 | 51.89% |
| dir_extreme | 8 | 45.330 | 62.875 | 38.71% |

## Minimum uniform PMU pair

Task45.10→59.86µs. This independent replay pair corroborates the negative complete-call result; it is not pooled into OFF. New critical AIV is block5/vector0,57.339µs. For a paired comparison, the following values use block5/vector0 and the same block5/cube0 in both arms.

| Same physical block5 metric | 9307 µs | partial768 µs |
|---|---:|---:|
| AIC total | 37.727 | 54.520 |
| Cube | 15.392 | 15.207 |
| MTE1 | 12.896 | 12.377 |
| MTE2 | 31.855 | 29.748 |
| FIX | 2.538 | 9.616 |
| credit3 wait | 3.827 | 8.967 |
| AIV total | 40.305 | 57.339 |
| Vector | 8.358 | 12.267 |
| AIV MTE2 | 2.640 | 9.206 |
| AIV MTE3 | 0.889 | 2.216 |
| READY5 wait | 33.981 | 41.270 |

The measured GM→L1 request rises28672→32768KiB; L0C→GM rises6144→18432KiB; GM→UB rises6240→18624KiB (including scale/metadata); UB→GM stays3072KiB. MTE2 busy falls about2.11µs on this block despite more A bytes, but AIC total rises16.79µs and AIV total rises17.03µs. The extra partial FIX, C load and exact INT32 merge costs dominate. Wait/busy intervals overlap and are not added as independent savings. No further Kpart/queue/granularity matrix was run.

## Evidence

- COMPLETE.json: frozen identity, disposition and scope
- SOURCE-IDENTITY.json / ADDRESS-IDENTITY.json: exact control and unmodified device object
- NUMERIC.json / FIXTURES.json:29calls, source fixtures, output SHA and guards
- OFF-SUMMARY.json and OFF-*/capture: full fresh Task cohorts and address logs
- PMU-SUMMARY.json and PMU-uniform-mode*/raw: one minimum replay pair
- LIFETIME.md / SYNC-CANDIDATES.md: resource closure and all23scanner candidates retained verbatim

Source SHA256: `8431cff071284077d876ce15973910fe0fad037c542cc2b1fd8bf7ca4993c212`
