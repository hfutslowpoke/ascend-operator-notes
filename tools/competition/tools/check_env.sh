#!/usr/bin/env bash
# Read-only environment check.  Does not run a kernel.
set -u

echo "== system =="
uname -a
echo "python: $(python3 --version 2>&1)"
echo "cmake:  $(command -v cmake || echo MISSING)"
echo "cc:     $(command -v gcc || echo MISSING)"
echo "flock:  $(command -v flock || echo MISSING)"

echo
echo "== CANN =="
for v in CANN_HOME ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_CUSTOM_OPP_PATH LD_LIBRARY_PATH; do
  printf '%-24s %s\n' "$v" "${!v:-<unset>}"
done
for d in /usr/local/Ascend /home/developer/Ascend; do
  [ -d "$d" ] && ls -1 "$d" 2>/dev/null | sed "s#^#  $d/#"
done

echo
echo "== tools =="
for t in bisheng ccec msprof npu-smi ascend-dmi; do
  printf '%-12s %s\n' "$t" "$(command -v "$t" || echo MISSING)"
done

echo
echo "== device =="
if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info 2>&1 | head -60
else
  echo "npu-smi not found"
fi

echo
echo "NOTE: run all NPU work through tools/npu_lock.sh to avoid contention."
