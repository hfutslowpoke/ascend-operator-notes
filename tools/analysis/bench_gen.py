import argparse
import json
from pathlib import Path

import numpy as np

import reference
from gen_case import make_sizes

ROOT = Path(__file__).resolve().parent


def generate_case(case, force=False, with_golden=False):
    name = case["name"]
    out = ROOT / case["dir"]
    meta_path = out / "meta.json"
    input_paths = [
        out / "input" / "x.bin",
        out / "input" / "weight.bin",
        out / "input" / "scale.bin",
        out / "input" / "group_list.bin",
    ]
    if not case.get("generate", True):
        missing = [str(p) for p in input_paths + [meta_path] if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{name}: existing case files missing: {missing}"
            )
        return "existing"

    if meta_path.exists() and all(p.exists() for p in input_paths) and not force:
        if with_golden and not (out / "golden" / "y.bin").exists():
            pass
        else:
            return "cached"

    m, k, n, e = (int(case[x]) for x in ("M", "K", "N", "E"))
    pattern = case["pattern"]
    seed = int(case.get("seed", 1))
    rng = np.random.default_rng(seed)
    sizes = make_sizes(rng, e, m, pattern)
    ends = np.cumsum(sizes).astype(np.int64)

    x = rng.integers(-128, 128, size=(m, k), dtype=np.int8)
    weight = rng.integers(-128, 128, size=(e, k, n), dtype=np.int8)
    max_scale = min(1.0, 60000.0 / (16384.0 * k)) * 0.9
    scale = rng.uniform(0.0, max_scale, size=(e, n)).astype(np.float32)
    scale[0, ::17] = 0.0

    reference._validate(x, weight, scale, ends)

    for folder in ("input", "output", "golden"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    (out / "input" / "x.bin").write_bytes(x.tobytes())
    (out / "input" / "weight.bin").write_bytes(weight.tobytes())
    (out / "input" / "scale.bin").write_bytes(scale.tobytes())
    (out / "input" / "group_list.bin").write_bytes(ends.tobytes())

    if with_golden:
        golden = reference.impl(x, weight, scale, ends)
        (out / "golden" / "y.bin").write_bytes(golden.tobytes())
    else:
        (out / "golden" / "y.bin").unlink(missing_ok=True)

    meta = {
        "M": m,
        "K": k,
        "N": n,
        "E": e,
        "class": case["class"],
        "pattern": pattern,
        "seed": seed,
        "sizes": sizes,
        "max_scale": max_scale,
        "known": case.get("note", ""),
        "golden": bool(with_golden),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return "generated"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "bench_cases.json")
    parser.add_argument("--name", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--golden", action="store_true")
    parser.add_argument("--all-golden", action="store_true")
    args = parser.parse_args()

    cases = json.loads(args.config.read_text())
    selected = set(args.name)
    for case in cases:
        if selected and case["name"] not in selected:
            continue
        use_golden = args.all_golden or (args.golden and case.get("golden", False))
        status = generate_case(case, force=args.force, with_golden=use_golden)
        print(f"{status:9s} {case['name']:20s} {case['class']}")


if __name__ == "__main__":
    main()
