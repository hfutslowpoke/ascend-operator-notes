#!/usr/bin/env bash
# Serialize actual NPU use across processes/agents.
#
# Scope: wrap only commands that initialize or use the device (run/bench/profile,
# ACL init, aclrtMemcpy, kernel launch, msprof). Do NOT wrap cmake/make/ascendc
# compilation, CPU reference checks, or static analysis; those can run in parallel.
#
# Usage:
#   tools/npu_lock.sh <command> [args...]
#   NPU_LOCK=/path/to/npu.lock tools/npu_lock.sh ./run_bench.sh
set -euo pipefail

lock="${NPU_LOCK:-/tmp/cann-npu.lock}"
mkdir -p "$(dirname "$lock")"
if ! command -v flock >/dev/null 2>&1; then
  echo "[npu_lock] flock not found; refusing to run without a lock" >&2
  exit 2
fi
exec flock "$lock" "$@"
