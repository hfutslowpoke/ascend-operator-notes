import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    name = sys.argv[1]
    out = ROOT / "cases" / name
    actual = np.fromfile(out / "output/y.bin", dtype=np.float16)
    golden = np.fromfile(out / "golden/y.bin", dtype=np.float16)
    if actual.size != golden.size:
        print(f"FAIL {name}: size {actual.size} != {golden.size}")
        return 1
    a = actual.astype(np.float64)
    g = golden.astype(np.float64)
    err = np.abs(a - g)
    ok = (err < 0.001) & ((g == 0) | (err < 0.001 * np.abs(g)))
    bad = int((~ok).sum())
    nonfinite = int((~np.isfinite(a)).sum())
    print(f"{'PASS' if bad == 0 and nonfinite == 0 else 'FAIL'} {name}: "
          f"bad={bad}/{a.size} nonfinite={nonfinite} max_abs={err.max():.6g}")
    if bad:
        idx = np.argwhere(~ok)[:10]
        for i in idx:
            print("  first bad idx", i.tolist(), "actual", a[tuple(i)], "golden", g[tuple(i)])
    return 0 if bad == 0 and nonfinite == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
