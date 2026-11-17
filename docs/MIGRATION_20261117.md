# 迁移记录：单层归档 → 两层知识库（2026-11-17）

本文件记录 2026-11-17 的仓库重构：旧结构按“批次/来源”平铺，新结构分为 `competition/` 与 `general/` 两层，并统一条目契约、机器索引和 ACW skills。

## 1. 重构前状态

| 项 | 值 |
|---|---|
| 分支 | `research/open-source-operators-20261116` |
| HEAD | `8e4d486a6cca7c7f58fa5dd8c54ab591d8d94aed` |
| 文件数 | 423 |
| 顶层目录 | `third_party/`（280）、`our_work/`（64）、`tools/`（41）、`docs/`（29）、`catalog/`（5） |
| 工作区 | 干净 |

## 2. 本地暂存备份

动结构前先做工作区备份（不含 `.git`，历史仍由 Git 保存）：

```text
path:   /workspace/acw/var/private/repository-research-cache/kb-restructure-20261117/worktree-before.tar.gz
sha256: 26f33fe2e7564c1ae598bdab83056ed97f72118f903bffce98126327d0f184b2
tar entries: 553（目录+文件）
worktree files: 423
```

同目录还保存：

- `HEAD-before.txt`、`branch-before.txt`、`git-status-before.txt`；
- `file-map.json`：旧路径 → 新路径的逐文件映射。

按用户要求，仓库更新并推送验证后删除该暂存目录；Git 历史仍可回退到 `8e4d486`。

## 3. 顶层迁移

| 旧路径 | 新路径 | 说明 |
|---|---|---|
| `third_party/shanghe_final/` | `competition/grouped-matmul-gelu/prs/` | 26 份公开 PR 快照 |
| `third_party/official/` | `competition/grouped-matmul-gelu/official/` | 官方题面/规则/LICENSE |
| `our_work/final_submission/` | `competition/grouped-matmul-gelu/our-submission/` | 我们的最终源码 |
| `our_work/components/` | `competition/grouped-matmul-gelu/components/` | 历史组件/候选 |
| `our_work/evidence/` | 按来源拆分（见下） | 不再集中在一个目录 |
| `catalog/external-repositories.json` | `index/catalog.json` | 统一机器索引，新增 `tier`/`entry_path`/`local_status` |
| `catalog/public-final-sources.json` | `competition/grouped-matmul-gelu/catalog/` | 26 PR 机器索引 |
| `catalog/our-work.json` | `competition/grouped-matmul-gelu/catalog/` | 我们代码/证据索引 |
| `catalog/submissions.csv` | `competition/grouped-matmul-gelu/catalog/` | PR 清单 |
| `catalog/INDEX.md` | `competition/grouped-matmul-gelu/catalog/` | 人读 PR 索引 |
| `docs/repositories/*.md` | 对应条目 `README.md` | 9 份仓库深读 |
| `docs/OPEN_SOURCE_REVIEW.md` | `competition/grouped-matmul-gelu/analysis/` | 公开源码分析 |
| `docs/RETROSPECTIVE.md` | `competition/grouped-matmul-gelu/analysis/` | 比赛复盘 |
| `docs/PLATFORM_EVIDENCE.md` | `competition/grouped-matmul-gelu/analysis/` | 平台证据 |
| `docs/RULES_AND_MATH.md` | `competition/grouped-matmul-gelu/analysis/` | 规则/数学顺序 |
| `docs/research/LOCAL_910B_VALIDATION_20261116.md` | `docs/validation/` | 本机验证 |
| `docs/repositories/README.md` | `index/README.md` | 旧仓库索引被统一索引取代 |

## 4. 仓库深读迁移

| 旧报告 | 新条目 |
|---|---|
| `docs/repositories/GitCode__cann-ops-competitions.md` | `competition/cann-ops-competitions/README.md` |
| `docs/repositories/GitCode__elec-ops-inspection.md` | `competition/elec-ops-inspection/README.md` |
| `docs/repositories/Hansenwuwuwu__ascend-c-competition-archive.md` | `competition/hansen-2026-power/README.md` |
| `docs/repositories/mantuoluozk__kernelswift-2026-triton-kernels.md` | `competition/kernelswift-2026/README.md` |
| `docs/repositories/Be4ver__svd.md` | `general/linear-algebra/be4ver-svd/README.md` |
| `docs/repositories/Parity-LRX__ICTC-O3-HP-Ascend.md` | `general/equivariant/parity-ictc-o3/README.md` |
| `docs/repositories/So-cean__mamba-ascendc.md` | `general/attention/mamba-ascendc/README.md` |
| `docs/repositories/vLLM-HUST__vllm-hust-opset.md` | `general/transformer/vllm-hust-opset/README.md` |
| `docs/repositories/52Xayah__triton-ascend-kernels.md` | `general/elementwise/52xayah-triton-kernels/README.md` |

新增候选条目（原先只有 catalog 记录）：

- `general/linear-algebra/ascend-eigh/`
- `general/benchmarks/tritonascendbench/`
- `general/benchmarks/triton-operator-benchmarks/`
- `general/cannbot/`（接入位，无内容）

## 5. 证据迁移

| 旧日志 | 新位置 |
|---|---|
| `environment.txt`、`environment_and_build.txt`、`gitcode_environment_and_build.txt` | `docs/validation/` |
| `local-910b-validation-20261116.json` | `docs/validation/` |
| `grouped_topk_*`、`sinkhorn_*`、`fused_moe_*` | `competition/kernelswift-2026/evidence/` |
| `paco_*`、`qdwh_*` | `general/linear-algebra/be4ver-svd/evidence/` |
| `xayah_*` | `general/elementwise/52xayah-triton-kernels/evidence/` |
| `gitcode_recompute_*`、`gitcode_concat_*` | `competition/cann-ops-competitions/evidence/` |
| `gitcode_lstsq_*` | `competition/elec-ops-inspection/evidence/` |
| 其余 GroupedMatmulGelu PMU/负实验 | `competition/grouped-matmul-gelu/evidence/` |

原始日志内容不改写；只移动路径并同步 `index/catalog.json`、`docs/validation/local-910b-validation-20261116.json` 中的日志字段。

## 6. 新增结构

- `competition/README.md`、`general/README.md`：两层入口与准入规则；
- `competition/excluded.md`、`general/excluded.md`：淘汰记录；
- `index/README.md`、`index/entry-schema.json`、`index/by-family.md`、`index/by-evidence.md`；
- `competition/catalog.json`、`general/catalog.json`、各条目 `source.json`（生成）；
- `skills/`：`operator-pattern-lookup`、`competition-repo-intake`、`ascend-910b-local-validation`；
- `docs/ARCHITECTURE.md`：架构、条目契约、证据等级、验证协议；
- `tools/build_index.py`：索引生成器。

## 7. 实施说明

- 当前文件系统是 overlayfs，`git mv` 对目录返回 `Invalid cross-device link`；改用 `cp -a` + 删除源目录，Git 仍按内容识别为 rename。
- 移动后统一重写 Markdown 相对链接：以旧路径解析目标，映射到新路径，再按新文档位置计算相对路径。
- 历史证据文档中指向未归档工作区/`/mnt/...` 的失效链接，转为“文字 + 代码路径”，避免断链但不伪造目标文件。
- 未修改第三方源码；Lstsq 的本地兼容 patch 仍只在工作区外副本。

## 8. 校验与后续

重构后必须通过：

```bash
python3 tools/build_index.py --check
python3 tools/verify_archive.py
git diff --check
```

校验结果和最终提交记录在本文件末尾；暂存备份在推送成功后删除。

### 8.1 重构提交校验

```text
文件数：461（重构前 423；新增架构/索引/skills/cards，删除旧仓库索引）
python3 tools/build_index.py --check: pass
python3 tools/verify_archive.py:
  ok: true
  public_files=248 own_copied_files=69 python_syntax=26 json=63 errors=[]
Markdown 链接：442 条；本库文档 0 断链；归档第三方 CONTRIBUTING.md 的原始畸形链接保留（见 GroupedMatmulGelu README 说明）
Markdown 表格：179 个；列数不一致 0
git diff --check: pass
```

- [x] `build_index.py --check` 通过
- [x] `verify_archive.py` `ok: true`
- [x] Markdown 链接/表格检查通过（归档第三方原样文件除外）
- [x] `git diff --check` 通过
- [x] 远端分支可访问（`main` 与 `architecture/two-tier-knowledge-base-20261117` 均为 `c1f6283`）
- [x] 本地暂存备份已删除

### 8.2 制造专场增量与清理（2026-11-17）

```text
提交：c1f6283c6f5f5b551a3b0b92778d44d0d32e11d6（批次）；17e3be0586d0fac3e3cdbbf6cec412a28583884f（evidence 日志跟踪修复）
范围：2026 制造行业算子专场 6 个官方/生产仓、15 份实现映射
python3 tools/build_index.py --check: pass
python3 tools/verify_archive.py:
  ok: true
  public_files=248 own_copied_files=69 python_syntax=26 json=69 errors=[]
Markdown 链接：540 条；本库文档 0 断链
Markdown 表格：202 个；列数不一致 0
git diff --check: pass
```

本机 910B3 结果：

```text
ops-fft/fft2_d, CANN 9.1.0, Ascend 910B3 physical NPU 2 / ACL logical 0
build: ./build.sh --ops=fft2_d --soc=Ascend910B --pkg -j8  -> success
run:   32x32 batch=1 complex64 impulse vs analytic DFT
       5 runs, best 0.1537 ms, max_abs=9.086799e-08, max_rel=9.086799e-08, bad=0
```

清理的当前批次临时数据（均在 `repository-research-cache/` 下）：

```text
review_ops-fft, review_ops-gnn, review_ops-sparse, review_ops-math,
review_ops-cv, review_ops-solver, review_DrivingSDK,
review_mat-chem-sim-pred, mfg-2026, local-validation-20261117,
kb-restructure-20261117, mfg/competition/tree HTML, add_mfg_catalog.py
释放：约 21 MiB；仓库内证据/哈希/驱动保留
注：首次提交时 `*.log` 忽略规则使两份 FFT 日志未入 Git；已为 `competition/mfg-2026/evidence/` 加例外并在 17e3be0 跟踪，当前远端 `main` 可检出日志。

制造专场的 36 个唯一选中源码文件（39 条引用）已按上游路径复制到 `competition/mfg-2026/<entry>/source/`；`source.json` 的 `local_path` 与 SHA256 可直接核对，外部缓存删除后仓库仍自包含这些选中文件。
```

更早批次的 `local-validation-20261116/` 与来源缓存未删除；它们不是本批临时产物，保留供对照。清理后 `verify_archive.py` 仍为 `ok: true`。
