import argparse
import csv
import fcntl
import glob
import hashlib
import json
import os
import re
import signal
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "build-npu" / "bench"
CONFIG = ROOT / "bench_cases.json"
RESULTS = ROOT / "results"
LOCK_PATH = "/tmp/cann-npu.lock"
MSPROF = shutil.which("msprof") or "/home/developer/Ascend/cann-9.0.0/bin/msprof"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cases(path):
    return json.loads(Path(path).read_text())


def selected_cases(cases, names, classes):
    wanted_names = set(names or [])
    wanted_classes = set(classes or [])
    result = []
    for case in cases:
        if wanted_names and case["name"] not in wanted_names:
            continue
        if wanted_classes and case["class"] not in wanted_classes:
            continue
        result.append(case)
    return result


def lock_npu():
    lock_file = open(LOCK_PATH, "w")
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    return lock_file


def unlock_npu(lock_file):
    fcntl.flock(lock_file, fcntl.LOCK_UN)
    lock_file.close()


def run_generator(cases, config, gen_golden):
    cmd = [
        sys.executable,
        str(ROOT / "bench_gen.py"),
        "--config",
        str(config),
    ]
    for case in cases:
        cmd.extend(["--name", case["name"]])
    if gen_golden:
        cmd.append("--golden")
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "bench_gen failed:\n" + proc.stdout + "\n" + proc.stderr
        )
    print(proc.stdout, end="")


def bench_command(case, iters):
    return [
        str(BENCH),
        "0",
        str(case["M"]),
        str(case["K"]),
        str(case["N"]),
        str(case["E"]),
        str(ROOT / case["dir"]),
        "0",
        str(iters),
        case["name"],
    ]


def parse_json_line(stdout):
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise RuntimeError("bench did not emit a JSON result: " + stdout[-1000:])


def run_event(case, iters):
    lock = lock_npu()
    try:
        proc = subprocess.run(
            bench_command(case, iters),
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=600,
        )
    finally:
        unlock_npu(lock)
    if proc.returncode != 0:
        raise RuntimeError(
            f"bench failed for {case['name']} rc={proc.returncode}\n"
            + proc.stdout[-2000:]
            + "\n"
            + proc.stderr[-2000:]
        )
    return parse_json_line(proc.stdout)


def parse_task_time(csv_path):
    tasks = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("kernel_name", "")
            # r1489 retains an inverse identifier map for these private entries.
            entry = name
            mangled = re.match(r"_Z(\d+)", name)
            if mangled:
                entry = name[mangled.end():mangled.end() + int(mangled[1])]
            if not (re.fullmatch(r"bk\d+(?:_.*)?", entry) or any(token in name for token in
                       ("grouped_matmul_gelu", "gmm_", "main_small_compact", "main_small_half_gelu_rows32", "main_small_manual",
                        "main_small_owner_tile", "main_i5_mask_calibration", "main_r1_micro_rows",
                        "main_r1_", "main_r1_b_consumer", "main_r1_compiled_grid", "main_r1_flat_A2", "main_r1_retired_half", "main_r1_retired_scale_newton", "main_r1_float_owners", "main_r1_vector_chunk", "main_r1_a1_lookahead", "RowHalfUnroll", "RowHalfLoop"))):
                continue
            try:
                task_us = float(row["task_time(us)"])
                start_us = float(row["task_start(us)"])
            except (KeyError, TypeError, ValueError):
                continue
            if task_us > 0:
                tasks.append((start_us, task_us, row))
    tasks.sort(key=lambda item: item[0])
    return tasks


def run_msprof(case, iters=1, timeout=600):
    out_dir = Path("/tmp/gmm_bench_msprof") / (
        case["name"] + "-" + str(os.getpid()) + "-" + str(int(time.time() * 1000))
    )
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_cmd = shlex.join(bench_command(case, iters))
    cmd = [
        MSPROF,
        f"--output={out_dir}",
        f"--application={bench_cmd}",
        "--task-time=on",
        "--ai-core=on",
    ]
    lock = lock_npu()
    try:
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Stop the application as well as the profiler while retaining the
            # NPU lock. Killing only msprof can leave its device process alive.
            os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
            raise RuntimeError(f'msprof timed out after {timeout}s; process group stopped; raw evidence {out_dir}')
        proc = subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)
    finally:
        unlock_npu(lock)
    if proc.returncode != 0:
        raise RuntimeError(
            f"msprof failed for {case['name']} rc={proc.returncode}\n"
            + proc.stdout[-2000:]
            + "\n"
            + proc.stderr[-2000:]
        )
    matches = sorted(glob.glob(str(out_dir / "*" / "mindstudio_profiler_output" / "task_time_*.csv")))
    if not matches:
        raise RuntimeError("msprof produced no task_time csv for " + case["name"])
    tasks = parse_task_time(matches[-1])
    if not tasks:
        raise RuntimeError("msprof produced no kernel task for " + case["name"] + "; raw evidence " + str(out_dir))
    RESULTS.mkdir(exist_ok=True)
    msprof_dir = RESULTS / "msprof"
    msprof_dir.mkdir(exist_ok=True)
    shutil.copy2(matches[-1], msprof_dir / (case["name"] + ".csv"))
    cold_us = tasks[0][1]
    warm_us = statistics.median([item[1] for item in tasks[1:]]) if len(tasks) > 1 else None
    raw_dir = out_dir
    summary = {
        "cold_us": cold_us,
        "warm_us": warm_us,
        "task_count": len(tasks),
        "task_us": [item[1] for item in tasks],
        "csv": str(msprof_dir / (case["name"] + ".csv")),
    }
    shutil.rmtree(raw_dir, ignore_errors=True)
    return summary


def enrich(case, events, msprofs):
    m, k, n, e = (int(case[x]) for x in ("M", "K", "N", "E"))
    macs = m * k * n
    data_bytes = (
        e * k * n
        + m * k
        + 2 * m * n
        + 4 * e * n
        + 8 * e
    )
    valid_msprofs = [item for item in msprofs if item]
    if valid_msprofs:
        cold_us = statistics.median([item["cold_us"] for item in valid_msprofs])
        warm_values = [item["warm_us"] for item in valid_msprofs if item["warm_us"] is not None]
        warm_us = statistics.median(warm_values) if warm_values else None
    else:
        cold_us = statistics.median([item["cold_device_us"] for item in events])
        warm_us = statistics.median(
            [statistics.median(item["device_samples_us"]) for item in events]
        )
    event = events[0]
    msprof = valid_msprofs[0] if valid_msprofs else None
    result = {
        "name": case["name"],
        "class": case["class"],
        "M": m,
        "K": k,
        "N": n,
        "E": e,
        "pattern": case["pattern"],
        "note": case.get("note", ""),
        "macs": macs,
        "input_output_bytes": data_bytes,
        "cold_us": cold_us,
        "warm_us": warm_us,
        "cold_gmacs": macs / cold_us / 1e3 if cold_us else None,
        "warm_gmacs": macs / warm_us / 1e3 if warm_us else None,
        "event": event,
        "msprof": msprof,
        "reps": {
            "count": len(events),
            "event_cold_us": [item["cold_device_us"] for item in events],
            "msprof_cold_us": [item["cold_us"] for item in valid_msprofs],
            "msprof_warm_us": [item["warm_us"] for item in valid_msprofs],
        },
    }
    return result


def fmt(value, digits=2):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def summarize(records):
    groups = {}
    for record in records:
        groups.setdefault(record["class"], []).append(record)
    summary = []
    for class_name in sorted(groups):
        items = groups[class_name]
        cold = [r["cold_us"] for r in items if r["cold_us"] is not None]
        warm = [r["warm_us"] for r in items if r["warm_us"] is not None]
        summary.append(
            {
                "class": class_name,
                "n": len(items),
                "cold_median_us": statistics.median(cold) if cold else None,
                "warm_median_us": statistics.median(warm) if warm else None,
                "cold_min_us": min(cold) if cold else None,
                "cold_max_us": max(cold) if cold else None,
                "warm_min_us": min(warm) if warm else None,
                "warm_max_us": max(warm) if warm else None,
            }
        )
    return summary


def write_outputs(records, summary, tag, metadata):
    RESULTS.mkdir(exist_ok=True)
    json_path = RESULTS / f"baseline-{tag}.json"
    csv_path = RESULTS / f"baseline-{tag}.csv"
    md_path = RESULTS / f"baseline-{tag}.md"
    payload = {
        "metadata": metadata,
        "records": records,
        "classes": summary,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    fields = [
        "class",
        "name",
        "M",
        "K",
        "N",
        "E",
        "pattern",
        "cold_us",
        "warm_us",
        "cold_gmacs",
        "warm_gmacs",
        "input_output_bytes",
        "note",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fields})

    lines = [
        "# GroupedMatmulGelu local shape timing",
        "",
        f"- generated: {metadata['generated_at']}",
        f"- kernel sha256: `{metadata['kernel_sha256']}`",
        f"- device: {metadata['device']} (cores={metadata['cores']})",
        f"- primary timing: msprof Task Duration; cold=first call in fresh process, warm=second call",
        "- group patterns are synthetic; hidden group_list is not observed",
        "",
        "## Per-case",
        "",
        "| class | case | M | K | N | E | pattern | cold us | warm us | cold GMAC/s |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            "| {cls} | {name} | {M} | {K} | {N} | {E} | {pattern} | {cold} | {warm} | {eff} |".format(
                cls=record["class"],
                name=record["name"],
                M=record["M"],
                K=record["K"],
                N=record["N"],
                E=record["E"],
                pattern=record["pattern"],
                cold=fmt(record["cold_us"]),
                warm=fmt(record["warm_us"]),
                eff=fmt(record["cold_gmacs"], 1),
            )
        )
    lines += [
        "",
        "## Per-class",
        "",
        "| class | n | cold median us | warm median us | cold min/max us |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            "| {cls} | {n} | {cold} | {warm} | {cmin}/{cmax} |".format(
                cls=item["class"],
                n=item["n"],
                cold=fmt(item["cold_median_us"]),
                warm=fmt(item["warm_median_us"]),
                cmin=fmt(item["cold_min_us"]),
                cmax=fmt(item["cold_max_us"]),
            )
        )
    lines += [
        "",
        "## Caveats",
        "",
        "- cold is the first isolated call (fresh process, fresh allocations, cold caches); warm is the next call with the same input.",
        "- platform Task Duration and local msprof are not guaranteed to use the same warmup/cache protocol.",
        "- hidden_regular_candidate and hidden_irregular_* are shape hypotheses or partial tuples, not confirmed hidden points.",
        "- R2/R3 pattern variants use synthetic group_list; only the shape is known.",
        "",
    ]
    md_path.write_text("\n".join(lines))
    return json_path, csv_path, md_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--name", action="append", default=[])
    parser.add_argument("--class", dest="classes", action="append", default=[])
    parser.add_argument("--event-only", action="store_true")
    parser.add_argument("--gen-golden", action="store_true")
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not BENCH.exists():
        raise SystemExit(f"bench binary missing: {BENCH}; build it first")

    cases = load_cases(args.config)
    cases = selected_cases(cases, args.name, args.classes)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("no cases selected")

    RESULTS.mkdir(exist_ok=True)
    run_generator(cases, args.config, args.gen_golden)

    tag = args.tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    records = []
    errors = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case['name']} ({case['class']}) ...", flush=True)
        try:
            events = []
            msprofs = []
            for rep in range(args.reps):
                events.append(run_event(case, args.iters))
                if not args.event_only:
                    msprofs.append(run_msprof(case))
            record = enrich(case, events, msprofs)
            records.append(record)
            print(
                "  cold={} us warm={} us pattern={} reps={}".format(
                    fmt(record["cold_us"]),
                    fmt(record["warm_us"]),
                    record["pattern"],
                    args.reps,
                ),
                flush=True,
            )
        except Exception as exc:
            errors.append({"case": case["name"], "error": str(exc)})
            print(f"  ERROR: {exc}", flush=True)

    if not records:
        raise SystemExit("no successful measurements")

    summary = summarize(records)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kernel_sha256": sha256(ROOT / "kernel.asc"),
        "source": str(ROOT / "kernel.asc"),
        "device": 0,
        "cores": records[0]["event"]["cores"],
        "primary_timing": "msprof cold/warm" if not args.event_only else "aclrtEvent cold/warm",
        "errors": errors,
    }
    json_path, csv_path, md_path = write_outputs(records, summary, tag, metadata)
    print("\nPer-class summary:")
    for item in summary:
        print(
            "{cls:24s} n={n:2d} cold={cold:8s} us warm={warm:8s} us".format(
                cls=item["class"],
                n=item["n"],
                cold=fmt(item["cold_median_us"]),
                warm=fmt(item["warm_median_us"]),
            )
        )
    print(f"\nJSON: {json_path}\nCSV:  {csv_path}\nMD:   {md_path}")
    if errors:
        print(f"WARNING: {len(errors)} case(s) failed; see metadata.errors", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
