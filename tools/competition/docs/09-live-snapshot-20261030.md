# 公开赛事快照（2026-10-30）

> 这是只读 API 快照，只用于帮助新会话快速定位。赛事开关、题目列表、ID 都可能变化；行动前重新执行 `cann_client.py contests` / `problems`。

## 当前 `ongoing=true` 且 `submit_enabled=true` 的赛事

| 赛事 slug | 标题 | contestId | 题目数 |
|---|---|---|---:|
| `2026_electronic` | 昇腾AI行业创新大赛——2026电力行业算子专场 | `6aa3bb4f2d3dd2c5aeab7eef` | 10 |

## `2026_electronic` 题目列表

全部为 `custom_template`，提交时按模板返回的 `editable` 文件路径提交，不要套用单 `kernel.asc` 模板。

| problemId | slug | 标题 |
|---|---|---|
| `6ab0f890b0477ec41e3146a7` | `swiglu` | 电力巡检算子 SwiGLU |
| `6ab0f969b0477ec41e31c112` | `treelinemindistance` | 最小安全距离计算算子 TreeLineMinDistance |
| `6ab0fb08b0477ec41e32a670` | `firecandidatedecodefusion` | 目标候选框解码与置信度融合算子 FireCandidateDecodeFusion |
| `6ab0fd36b0477ec41e33e401` | `safetybeltviolationgeometry` | 安全带判定融合算子 SafetyBeltViolationGeometry |
| `6ab0fdcab0477ec41e342fc6` | `splineconv` | 潮流信息聚合算子 SplineConv |
| `6ab0fe98b0477ec41e34a06e` | `det` | 电力系统线性化分析算子 det |
| `6ab0ff69b0477ec41e3515bc` | `lstsq` | 多工况潮流数据分析算子 lstsq |
| `6ab10035b0477ec41e358e1a` | `pinv` | 潮流计算算子 pinv |
| `6ab10091b0477ec41e35b5cc` | `solve` | 线性修正方程求解算子 solve |
| `6ae1a7d01dac5d235b315a19` | `eigh__v2` | 实对称矩阵谱分解算子 eigh |

## 其他可见赛事（多数 `submit_enabled=false`）

| slug | 标题 | 说明 |
|---|---|---|
| `s1` / `s2` | 算子挑战赛 S1/S2 | 历史题库，当前提交开关关闭 |
| `vector` / `cube` / `fusion` | Vector/Cube/Fusion 算子题目 | 公开练习题库 |
| `ladder_october` | CANN 算子天梯赛（10-11 月赛） | 天梯赛 |
| `ct_starcup_aiop_*` | 星辰杯高校 AI 算子开发挑战赛 | 历史/决赛 |
| `op_challenge_*` | 2026 CANN 挑战赛各赛区 | 初赛/决赛 |
| `ictcompetition` | 华为 ICT 大赛昇腾 AI 算子开发赛道 | 决赛 |

## 重新获取

```bash
python3 tools/cann_client.py contests > contests.json
python3 tools/cann_client.py problems 2026_electronic > electronic-problems.json
python3 tools/cann_client.py resolve 2026_electronic swiglu
```
