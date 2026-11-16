import json
import sys
from pathlib import Path

import numpy as np

import reference

ROOT = Path(__file__).resolve().parent


def make_sizes(rng, e, m, pattern):
    if pattern == "sample":
        sizes = [m // e] * e
        sizes[-1] += m - sum(sizes)
    elif pattern == "empty":
        sizes = [0] * e
        idx = [i for i in range(e) if i % 2 == 1]
        for i in idx:
            sizes[i] = 1
        left = m - sum(sizes)
        sizes[idx[-1]] += left
    elif pattern == "last":
        sizes = [0] * (e - 1) + [m]
    elif pattern == "random":
        sizes = rng.integers(0, 5, size=e).tolist()
        diff = m - sum(sizes)
        if diff >= 0:
            sizes[-1] += diff
        else:
            for i in range(e):
                take = min(sizes[i], -diff)
                sizes[i] -= take
                diff += take
                if diff == 0:
                    break
    else:
        raise ValueError(pattern)
    assert len(sizes) == e and min(sizes) >= 0 and sum(sizes) == m
    return sizes


def main():
    name = sys.argv[1]
    cfg = json.loads((ROOT / "cases.json").read_text())[name]
    m, k, n, e = (cfg[x] for x in ("M", "K", "N", "E"))
    rng = np.random.default_rng(cfg["seed"])

    sizes = make_sizes(rng, e, m, cfg["pattern"])
    ends = np.cumsum(sizes).astype(np.int64)

    x = rng.integers(-128, 128, size=(m, k), dtype=np.int8)
    weight = rng.integers(-128, 128, size=(e, k, n), dtype=np.int8)
    max_scale = min(1.0, 60000.0 / (16384.0 * k)) * 0.9
    scale = rng.uniform(0.0, max_scale, size=(e, n)).astype(np.float32)
    scale[0, ::17] = 0.0

    golden = reference.impl(x, weight, scale, ends)

    out = ROOT / "cases" / name
    for folder in ("input", "output", "golden"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    (out / "input" / "x.bin").write_bytes(x.tobytes())
    (out / "input" / "weight.bin").write_bytes(weight.tobytes())
    (out / "input" / "scale.bin").write_bytes(scale.tobytes())
    (out / "input" / "group_list.bin").write_bytes(ends.tobytes())
    (out / "golden" / "y.bin").write_bytes(golden.tobytes())
    (out / "output" / "y.bin").unlink(missing_ok=True)
    (out / "meta.json").write_text(json.dumps(
        {"M": m, "K": k, "N": n, "E": e, "sizes": sizes, "max_scale": max_scale},
        indent=2))
    print(f"generated {name}: M={m} K={k} N={n} E={e} sizes={sizes}")


if __name__ == "__main__":
    main()
