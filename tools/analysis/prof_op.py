#!/usr/bin/env python3
"""Profile one GroupedMatmulGelu case with msprof op and summarize the CSVs.

Usage:
  python3 prof_op.py --name sh_c3 --M ... --K ... --N ... --E ... --dir cases/official4 [--metrics PipeUtilization,ArithmeticUtilization,MemoryUB,Memory]

Runs the already-built build-npu/bench with 2 warmups and 1 iteration, lets
msprof op replay the kernel, then prints mean/max per-core pipe times.
"""
import argparse, csv, fcntl, glob, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "build-npu" / "bench"
MSPROF = shutil.which("msprof") or "/home/developer/Ascend/cann-9.0.0/bin/msprof"


def parse_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(row, key):
    try:
        return float(row.get(key, "NA"))
    except (TypeError, ValueError):
        return None


def summarize(rows, prefix):
    keys = [
        "time", "cube_ratio", "mte1_ratio", "mte2_ratio", "mte3_ratio",
        "fixpipe_ratio", "scalar_ratio", "vec_ratio",
        "scalar_mte2_stall_time", "scalar_wait_time", "scalar_wait_id5_time",
        "ub_read_bw_vector", "ub_write_bw_vector", "mte2_active_bw",
        "mte3_active_bw", "cube_fops",
    ]
    out = {}
    for k in keys:
        col = prefix + "_" + k
        vals = [num(r, col) for r in rows]
        vals = [v for v in vals if v is not None]
        if vals:
            out[k] = (sum(vals) / len(vals), max(vals), min(vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--E", type=int, required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--metrics", default="PipeUtilization,ArithmeticUtilization,MemoryUB")
    ap.add_argument("--out", default=None)
    ap.add_argument("--binary", type=Path, default=BENCH,
                    help="Private bench or verify_batch binary with the bench entry")
    ap.add_argument("--timeout", type=int, default=60,
                    help="Kill the profiler process group before releasing the NPU lock")
    ap.add_argument("--warmup", type=int, default=2,
                    help="Application warmup launches before replay; default preserves historical profiles")
    args = ap.parse_args()
    if args.timeout <= 0 or args.warmup < 0 or not args.binary.is_file():
        ap.error("Use an existing binary, positive timeout, and nonnegative warmup count")

    out_dir = Path("/tmp/opprof") / args.name
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    app = (
        f"{args.binary.resolve()} 0 {args.M} {args.K} {args.N} {args.E} "
        f"{ROOT / args.dir} {args.warmup} 1 {args.name}"
    )
    cmd = [MSPROF, "op", f"--application={app}", f"--output={out_dir}",
           f"--aic-metrics={args.metrics}"]
    cmd = ["timeout", "--signal=TERM", "--kill-after=5s", f"{args.timeout}s"] + cmd
    lock = open("/tmp/cann-npu.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    sys.stdout.write(proc.stdout[-4000:])
    sys.stderr.write(proc.stderr[-2000:])
    if proc.returncode != 0:
        return proc.returncode

    dirs = sorted(glob.glob(str(out_dir / "OPPROF_*")))
    if not dirs:
        print("no OPPROF dir")
        return 2
    d = dirs[-1]
    basic = parse_csv(d + "/OpBasicInfo.csv")[0]
    print(f"\n== {args.name} M={args.M} K={args.K} N={args.N} E={args.E} ==")
    print(f"task_duration_us={basic['Task Duration(us)']} block_dim={basic['Block Dim']}")
    for fname, prefix in (("PipeUtilization.csv", "aic"), ("PipeUtilization.csv", "aiv")):
        rows = parse_csv(d + "/" + fname)
        if prefix == "aic":
            aic_rows = [r for r in rows if r["sub_block_id"].startswith("cube")]
            s = summarize(aic_rows, "aic")
            print("AIC mean/max/min: " + ", ".join(
                f"{k}={v[0]:.2f}/{v[1]:.2f}/{v[2]:.2f}" for k, v in s.items()))
        else:
            aiv_rows = [r for r in rows if r["sub_block_id"].startswith("vector")]
            s = summarize(aiv_rows, "aiv")
            print("AIV mean/max/min: " + ", ".join(
                f"{k}={v[0]:.2f}/{v[1]:.2f}/{v[2]:.2f}" for k, v in s.items()))
    if os.path.exists(d + "/MemoryUB.csv"):
        rows = parse_csv(d + "/MemoryUB.csv")
        aiv_rows = [r for r in rows if r["sub_block_id"].startswith("vector")]
        s = summarize(aiv_rows, "aiv")
        print("AIV UB: " + ", ".join(
            f"{k}={v[0]:.2f}/{v[1]:.2f}/{v[2]:.2f}" for k, v in s.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
