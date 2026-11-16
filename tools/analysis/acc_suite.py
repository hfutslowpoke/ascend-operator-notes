#!/usr/bin/env python3
"""Run sample on selected cases and compare output/y.bin to golden/y.bin."""
import json, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
REF = ROOT.parent / "project" / "kernel.asc"
KERNEL = ROOT / "kernel.asc"
BIN = ROOT / "build-npu" / "sample"


def cases():
    out = []
    for cfg in ("bench_shape_cases.json", "bench_cases.json"):
        p = ROOT / cfg
        if not p.exists():
            continue
        for c in json.loads(p.read_text()):
            if c["name"] in {x["name"] for x in out}:
                continue
            d = ROOT / c["dir"]
            if (d / "golden" / "y.bin").exists() and (d / "input" / "x.bin").exists():
                out.append(c)
    return out


def run(c):
    cmd = [str(BIN), "0", str(c["M"]), str(c["K"]), str(c["N"]), str(c["E"]),
           str(ROOT / c["dir"])]
    p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if p.returncode != 0:
        return None
    return np.fromfile(ROOT / c["dir"] / "output" / "y.bin", dtype=np.float16)


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "cand"
    bad_total = 0
    n_total = 0
    rows = []
    for c in cases():
        y = run(c)
        if y is None:
            rows.append((c["name"], "RUNFAIL", 0, 0, 0))
            continue
        g = np.fromfile(ROOT / c["dir"] / "golden" / "y.bin", dtype=np.float16)
        a64 = y.astype(np.float64)
        g64 = g.astype(np.float64)
        err = np.abs(a64 - g64)
        ok = (err < 0.001) & ((g64 == 0) | (err < 0.001 * np.abs(g64)))
        bad = int((~ok).sum())
        nonfinite = int((~np.isfinite(a64)).sum())
        maxrel = 0.0
        nz = err[g64 != 0]
        if nz.size:
            maxrel = float(np.max(nz / np.abs(g64[g64 != 0])))
        bad_total += bad
        n_total += a64.size
        rows.append((c["name"], bad, nonfinite, float(err.max()), maxrel))
    print(f"== {label} ==")
    for r in rows:
        if r[1] == "RUNFAIL":
            print(f"{r[0]:18s} RUNFAIL")
        elif r[1] or r[2]:
            print(f"{r[0]:18s} bad={r[1]:8d} nonfinite={r[2]:6d} maxabs={r[3]:.6g} maxrel={r[4]:.3g}")
    print(f"TOTAL bad={bad_total}/{n_total} cases={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
