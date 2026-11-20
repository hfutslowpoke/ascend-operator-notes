#!/usr/bin/env python3
"""Generate manufacturing-official quick specs and per-problem source mapping.

Source of truth: competition/mfg-2026/official/specs.json
Do not edit the generated Markdown by hand.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPECS = ROOT / "competition/mfg-2026/official/specs.json"
PROBLEMS = ROOT / "competition/mfg-2026/official/problems.json"
CATALOG = ROOT / "index/catalog.json"
SPECS_MD = ROOT / "competition/mfg-2026/official/specs.md"
PROBLEMS_MD = ROOT / "competition/mfg-2026/problems.md"
CANJUDGE_MD = ROOT / "competition/mfg-2026/cannjudge-map.md"

ENTRY = {
    "MFG2026-MATH-COMPLEX-ABS": "ops-math/README.md",
    "MFG2026-MATH-COMPLEX-V3": "ops-math/README.md",
    "MFG2026-SOLVER-COMPLEX-DIV": "ops-solver/README.md",
    "MFG2026-FFT2D": "ops-fft/README.md",
    "MFG2026-FFT1D": "ops-fft/README.md",
    "MFG2026-GNN-GATHER-COO": "ops-gnn/README.md",
    "MFG2026-GNN-SEGMENT-MIN-CSR": "ops-gnn/README.md",
    "MFG2026-GNN-RANDOM-WALK": "ops-gnn/README.md",
    "MFG2026-GNN-SPMM": "ops-gnn/README.md",
    "MFG2026-GNN-GRID-CLUSTER": "ops-gnn/README.md",
    "MFG2026-SPARSE-SPGEMM": "ops-sparse/README.md",
    "MFG2026-SPARSE-SPMM": "ops-sparse/README.md",
    "MFG2026-RADIUS": "driving-sdk/README.md",
    "MFG2026-FPS": "driving-sdk/README.md",
    "MFG2026-CYLINDER-QUERY": "driving-sdk/README.md",
    "MFG2026-MATH-STACK-BALL-QUERY": "ops-math/README.md",
}


def load() -> dict:
    return json.loads(SPECS.read_text(encoding="utf-8"))


def load_problems() -> dict:
    return json.loads(PROBLEMS.read_text(encoding="utf-8"))


def load_catalog() -> dict:
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def rel_link(target: Path) -> str:
    return os.path.relpath(target, CANJUDGE_MD.parent).replace(os.sep, "/")


def build_specs_md(data: dict) -> str:
    lines = ["# 官方题面与源码匹配逐题表", "", "由 `tools/build_mfg_official.py` 从 `official/specs.json` 生成；完整题面见 `descriptions/`。", ""]
    for i, p in enumerate(data["problems"], 1):
        s = p["source"]
        lines += [
            f"## {i:02d} {p['title']}", "",
            f"- 官方题面：[{p['name']}](descriptions/{p['name']}.md) · [CANN Judge](https://cannjudge.cn/public/gcc_2026_mfg/{p['name']})",
            f"- 公式：{p['formula']}",
            f"- 输入：{'；'.join(p['inputs'])}",
            f"- 输出：{'；'.join(p['outputs'])}",
            f"- 属性：{'；'.join(p['attrs']) if p['attrs'] else '无'}",
            f"- 约束：{'；'.join(p['constraints'])}",
            f"- 容差：{p['tolerance']}",
            f"- 源码级别：`{s['coverage']}`；候选：`{s['candidate'] or '—'}`",
            f"- 源码差异：{'；'.join(s['mismatch'])}",
        ]
        if s.get("building_blocks"):
            lines.append(f"- 补充基础件：{'；'.join(s['building_blocks'])}")
        lines += [f"- 复用判断：{s['decision']}", ""]
    return "\n".join(lines).rstrip() + "\n"


def build_problems_md(data: dict) -> str:
    lines = [
        "# 15 道赛题：官方规格与源码匹配", "",
        "> 官方题面抓取于 2026-11-17：`https://cannjudge.cn/public/gcc_2026_mfg/<problem>`。完整题面在 [`official/descriptions/`](official/descriptions/)，机器可读摘要在 [`official/specs.json`](official/specs.json)。", "",
        "本页不再用“公开源码存在”推断“可直接解决赛题”。每题的 `direct` / `adapt` / `reference` / `none` 见 [official/README.md](official/README.md)。", "",
    ]
    for i, p in enumerate(data["problems"], 1):
        s = p["source"]
        cand = s["candidate"]
        link = ENTRY.get(cand, "")
        url = f"https://cannjudge.cn/public/gcc_2026_mfg/{p['name']}"
        lines += [
            f"## {i:02d} {p['title']}", "",
            f"- **官方题面**：[{p['name']}](official/descriptions/{p['name']}.md) · [CANN Judge]({url})",
            f"- **源码级别**：`{s['coverage']}`；候选：`{cand or '—'}`" + (f" · [条目说明]({link})" if link else ""),
            f"- **公式**：{p['formula']}",
            f"- **输入**：{'；'.join(p['inputs'])}",
            f"- **输出**：{'；'.join(p['outputs'])}",
            f"- **属性**：{'；'.join(p['attrs']) if p['attrs'] else '无'}",
            f"- **约束**：{'；'.join(p['constraints'])}",
            f"- **容差**：{p['tolerance']}",
            f"- **源码差异**：{'；'.join(s['mismatch'])}",
            f"- **复用判断**：{s['decision']}",
        ]
        if s.get("building_blocks"):
            lines.append(f"- **补充基础件**：{'；'.join(s['building_blocks'])}")
        lines.append("")
    lines += [
        "## 结论", "",
        "- **2 题**有语义高度匹配的 A2/910B 实现：`SegmentMinCsr`、`FurthestPointSample`（后者仍待本机逐位验证）；",
        "- **6 题**有 A2/910B 目标源码但需适配：`FourierTransform2D`、`SpmmSum`、`Spspmm`、`Radius`、`BallQuery`、`CylinderQuery`；",
        "- **5 题**只有 950/arch35 或算法 helper：`ComplexAbs`、`ComplexDiv`、`GatherCoo`、`RandomWalk`、`GridCluster`；",
        "- **2 题**未找到公开单 kernel 实现：`ComplexExp`、`FusedPropStep`，但已有 910B 基础件可组合（exp/sin/cos + ComplexV3；fft1_d/fft2_d）。", "",
        "“有 A2 路径”不等于“可直接提交”：例如 FFT 源码只覆盖 32/64/128，官方要求到 8192；Spspmm 源码是 CSR 动态输出，官方是 COO 固定槽位；CylinderQuery 源码 Python 层会排序，官方要求扫描序 first-K。", "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def build_cannjudge_map(data: dict, problems: dict, catalog: dict) -> str:
    by_name = {p["canonical_name"]: p for p in problems["problems"]}
    impls = catalog["implementations"]
    lines = [
        "# 制造专场 × CANN Judge 联合地图",
        "",
        "由 `tools/build_mfg_official.py` 从 `official/specs.json`、`official/problems.json` 和 `index/catalog.json` 生成；不要手改。",
        "",
        "## 赛事",
        "",
        f"- contest_id：`{problems['contest_id']}`",
        f"- 标题：{problems['contest_title']}",
        f"- 题数：{len(data['problems'])}",
        f"- 官方题面：<https://cannjudge.cn/public/gcc_2026_mfg/>",
        f"- 抓取日期：{problems.get('retrieved_at', '2026-11-17')}",
        "",
        "## 15 题联合表",
        "",
        "| # | CANN Judge 题目 | 题面 | 官方分级 | 候选实现（候选级别 / 本机） | 测试点（总/排名） |",
        "|---:|---|---|---|---|---|",
    ]
    for i, p in enumerate(data["problems"], 1):
        name = p["name"]
        meta = by_name.get(name, {})
        candidates = [
            x for x in impls
            if (x.get("official_problem") or {}).get("id") == name
        ]
        candidates.sort(key=lambda x: x["id"])
        cand_text = "<br>".join(
            f"[`{x['id']}`]({rel_link(ROOT / x['entry_path'] / 'README.md')}) "
            f"`{x.get('source_match', '—')}` / `{x.get('local_status', 'not-run')}`"
            for x in candidates
        ) or "—"
        total = meta.get("total_testcase_count", "?")
        ranking = meta.get("ranking_testcase_count", "?")
        lines.append(
            f"| {i:02d} | [{p['title']}](https://cannjudge.cn/public/gcc_2026_mfg/{name})<br>`{name}` | "
            f"[desc](official/descriptions/{name}.md) | `{p['source']['coverage']}` | "
            f"{cand_text} | {total} / {ranking} |"
        )
    lines += [
        "",
        "## 三条链路",
        "",
        "| 层 | 作用 | 入口 |",
        "|---|---|---|",
        "| 知识库 | 源码映射、语义分级、适配蓝图、CPU 参考 | `competition/`、`general/`、`index/catalog.json` |",
        "| 官方题面 | 题目 ID/slug、shape/dtype、测试点数、容差 | `competition/mfg-2026/official/` |",
        "| CANN Judge | 登录、模板、提交、回读、榜单、额度 | `tools/competition/`（便携）或 ACW `bin/cann-*` |",
        "",
        "## 联合工作流",
        "",
        "1. **发现**：`bin/cann-client resolve gcc_2026_mfg`，再 `bin/cann-client problems gcc_2026_mfg`；确认 `problemId`/slug。",
        "2. **对齐**：在本页找到 slug → 官方题面 → 源码级别 → 候选实现/适配蓝图。",
        "3. **开发**：只按官方 shape/dtype/属性特化；不按测试用例编号/hash 分支。",
        "4. **本机验证**：经 `bin/npu-lock` 串行；记录源码/产物 SHA、设备、CANN、shape、误差、时延。",
        "5. **打包**：`bin/cann-package --problem-id <PID> --account <A> --source-dir <src> --out <zip>`；只生成包，不 POST。",
        "6. **dry-run**：`bin/cann-submit --problem-id <PID> --account <A> --source-dir <src> --tag <tag>`；核对模板路径、文件 SHA、payload。",
        "7. **提交/回读**：仅在用户明确授权后 `ACW_ALLOW_SUBMIT=1 bin/cann-submit ... --submit`；拿 SID 后只 GET，保存 detail/ranking。",
        "",
        "便携工具在 `tools/competition/tools/`，ACW 包装入口是 `bin/cann-client`、`bin/cann-package`、`bin/cann-submit`；两者语义相同，按运行环境选择。",
        "",
        "## 当前状态与边界",
        "",
        "- 15 题公开题面、结构化 specs、15 份 `descriptions/` 已落库；隐藏测试数据未获取，`/api/testcases/<id>` 返回 403，未绕过。",
        "- 本页不包含任何 Cookie、Token、账号私有配置；账号文件只在私有目录/`0600`。",
        "- 本库未记录制造专场的平台提交；本轮没有 POST。",
        "- 本机 910B3 证据仍以条目 `local_status` 为准；`pass` 只代表本机测试，不等于平台通过。",
        "- 官方题面/测试点数来自 CANN Judge 公开 API；平台后续更新时重新抓取并重生成。",
        "",
        "## 安全红线",
        "",
        "- 不把 Cookie/账号写入仓库、日志或提交包；",
        "- 不绕过隐藏测试数据、额度或重复提交限制；",
        "- 不把本地 `pass` 写成平台通过；不伪造 SID/榜单；",
        "- 提交默认 dry-run；POST 必须用户显式授权（`ACW_ALLOW_SUBMIT=1` + `--submit`）。",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_or_check(path: Path, text: str, check: bool) -> bool:
    if check:
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            print(f"out-of-date: {path.relative_to(ROOT)}", file=sys.stderr)
            return False
        return True
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = load()
    problems = load_problems()
    catalog = load_catalog()
    ok = write_or_check(SPECS_MD, build_specs_md(data), args.check)
    ok &= write_or_check(PROBLEMS_MD, build_problems_md(data), args.check)
    ok &= write_or_check(CANJUDGE_MD, build_cannjudge_map(data, problems, catalog), args.check)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
