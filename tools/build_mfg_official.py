#!/usr/bin/env python3
"""Generate manufacturing-official quick specs and per-problem source mapping.

Source of truth: competition/mfg-2026/official/specs.json
Do not edit the generated Markdown by hand.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPECS = ROOT / "competition/mfg-2026/official/specs.json"
SPECS_MD = ROOT / "competition/mfg-2026/official/specs.md"
PROBLEMS_MD = ROOT / "competition/mfg-2026/problems.md"

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
    import json
    return json.loads(SPECS.read_text(encoding="utf-8"))


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
    ok = write_or_check(SPECS_MD, build_specs_md(data), args.check)
    ok &= write_or_check(PROBLEMS_MD, build_problems_md(data), args.check)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
