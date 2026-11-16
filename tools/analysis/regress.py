#!/usr/bin/env python3
"""End-to-end regression for a GroupedMatmulGelu kernel candidate.

Builds the reference (project/kernel.asc = r12) and the candidate, runs
`sample` on all bench/randomized/boundary cases, and checks that the
candidate output is bit-identical to the reference. Optionally runs timing.

Usage:
  python3 regress.py --candidate opt/candidate_r47/kernel.asc
  python3 regress.py --candidate opt/candidate_r47/kernel.asc --timing
Exit code 0 iff every case is bit-identical to the reference.
"""
import argparse, fcntl, json, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REF = ROOT.parent / "project" / "kernel.asc"
KERNEL = ROOT / "kernel.asc"
LOCK = "/tmp/cann-npu.lock"
CONFIGS = ["bench_cases.json", "bench_rand_cases.json", "bench_wide_cases.json", "bench_boundary_cases.json", "bench_edge_cases.json", "bench_m256_cases.json", "bench_shape_cases.json"]


def load_cases():
    cases, seen = [], set()
    for cfgs in CONFIGS:
        p = ROOT / cfgs
        if not p.exists():
            continue
        for c in json.loads(p.read_text()):
            if c["name"] in seen:
                continue
            seen.add(c["name"])
            cases.append(c)
    return cases


def run(cmd, cwd=ROOT):
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)


def build(src):
    shutil.copy2(src, KERNEL)
    r = run(["cmake", "--build", "build-npu", "--target", "bench", "sample", "--parallel", "4"])
    if r.returncode != 0:
        raise RuntimeError("build failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])


def gen_cases():
    for cfgs in CONFIGS:
        p = ROOT / cfgs
        if p.exists():
            r = run([sys.executable, "bench_gen.py", "--config", str(p), "--golden"])
            if r.returncode != 0:
                raise RuntimeError("gen failed for " + cfgs + ":\n" + r.stdout + r.stderr)


def sample(case):
    cmd = [str(ROOT / "build-npu" / "sample"), "0", str(case["M"]), str(case["K"]),
           str(case["N"]), str(case["E"]), str(ROOT / case["dir"])]
    lf = open(LOCK, "w")
    fcntl.flock(lf, fcntl.LOCK_EX)
    try:
        p = run(cmd)
    finally:
        fcntl.flock(lf, fcntl.LOCK_UN)
        lf.close()
    if p.returncode != 0:
        raise RuntimeError(f"sample failed for {case['name']}:\n" + p.stdout[-500:] + p.stderr[-500:])


def snap(cases, outdir):
    for c in cases:
        d = ROOT / c["dir"]
        src = d / "output" / "y.bin"
        if not src.exists():
            raise RuntimeError("missing output for " + c["name"])
        shutil.copy2(src, Path(outdir) / (c["name"] + ".bin"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--no-gen", action="store_true")
    ap.add_argument("--timing", action="store_true")
    args = ap.parse_args()
    cand = (ROOT.parent / args.candidate).resolve()
    cases = load_cases()
    print(f"regression: {len(cases)} cases vs r12 ({REF.name})")
    if not args.no_gen:
        gen_cases()
    tmp = tempfile.mkdtemp(prefix="gmm_regress_")
    build(REF)
    for c in cases:
        sample(c)
    snap(cases, tmp)
    build(cand)
    for c in cases:
        sample(c)
    diffs = []
    for c in cases:
        ref = (Path(tmp) / (c["name"] + ".bin")).read_bytes()
        got = (ROOT / c["dir"] / "output" / "y.bin").read_bytes()
        if ref != got:
            diffs.append(c["name"])
    print(f"bit-identical: {len(cases)-len(diffs)}/{len(cases)}")
    if diffs:
        print("DIFF:", ", ".join(diffs))
    if args.timing:
        r = run([sys.executable, "bench_all.py", "--tag", "regress-timing"])
        print(r.stdout[-800:])
    # restore reference kernel
    build(REF)
    return 0 if not diffs else 1


if __name__ == "__main__":
    raise SystemExit(main())
