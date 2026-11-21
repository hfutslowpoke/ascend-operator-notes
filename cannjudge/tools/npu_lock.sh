#!/usr/bin/env bash
# Hold the shared device lock only for commands that use the NPU.
set -euo pipefail
if [ "${ACW_NPU_LOCK_HELD:-0}" = 1 ]; then
  exec "$@"
fi
if [ -n "${NPU_LOCK:-}" ]; then
  lock="$NPU_LOCK"
elif [ -d /workspace/acw/var/run ]; then
  lock=/workspace/acw/var/run/npu.lock
else
  lock="${XDG_RUNTIME_DIR:-/tmp}/cannjudge-$(id -u)/npu.lock"
fi
mkdir -p -m 700 "$(dirname "$lock")"
export ACW_NPU_LOCK_HELD=1
exec flock --no-fork "$lock" "$@"
