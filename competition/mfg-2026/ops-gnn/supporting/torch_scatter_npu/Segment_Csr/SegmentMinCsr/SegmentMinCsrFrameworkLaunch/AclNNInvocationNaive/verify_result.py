#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
SegmentMinCsr 轻量化样例精度校验脚本
"""

import logging
import sys
from pathlib import Path

import numpy as np

LOSS = 1e-4
MINIMUM = 1e-9


def verify_out(real_path: Path, golden_path: Path) -> bool:
    real = np.fromfile(real_path, dtype=np.float32)
    gold = np.fromfile(golden_path, dtype=np.float32)
    if real.size != gold.size:
        logging.error("out 大小不一致, real=%d, golden=%d", real.size, gold.size)
        return False
    diff = np.abs(real - gold)
    denom = np.maximum(np.abs(real), np.abs(gold))
    atol_ok = diff <= LOSS
    rtol_ok = diff / (denom + MINIMUM) <= LOSS
    if not np.all(atol_ok | rtol_ok):
        logging.error("out 校验失败, 最大误差=%.6f", diff.max())
        return False
    logging.info("out tensor compare pass")
    return True


def main():
    if len(sys.argv) != 3:
        print("用法: python3 verify_result.py <real_out> <gold_out>")
        sys.exit(1)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    real_out, gold_out = map(Path, sys.argv[1:3])
    ok_out = verify_out(real_out, gold_out)
    sys.exit(0 if ok_out else 1)


if __name__ == "__main__":
    main()
