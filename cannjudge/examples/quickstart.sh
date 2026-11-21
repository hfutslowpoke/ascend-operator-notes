#!/usr/bin/env bash
# CannJudge quickstart. Replace placeholders before running.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/cannjudge/cann.env.example"

PROBLEM_ID="${PROBLEM_ID:-<problemId>}"
ACCOUNT="${ACCOUNT:-B}"
SOURCE_DIR="${SOURCE_DIR:-/path/to/project}"
TAG="${TAG:-my-r1}"
CONTEST_NAME="${CONTEST:-gcc_2026_mfg}"

echo "[1/4] accounts"
python3 "$ROOT/cannjudge/tools/cann_client.py" accounts

echo "[2/4] quota window (12:00 -> 12:00 Beijing)"
python3 "$ROOT/cannjudge/tools/cann_quota.py" \
  --contest "$CONTEST_NAME" --account "$ACCOUNT" --details

echo "[3/4] dry-run"
python3 "$ROOT/cannjudge/tools/cann_submit.py" \
  --problem-id "$PROBLEM_ID" \
  --account "$ACCOUNT" \
  --source-dir "$SOURCE_DIR" \
  --tag "$TAG"

echo "[4/4] submit (requires explicit confirmation)"
if [ "${CONFIRM_SUBMIT:-0}" != "1" ]; then
  echo "skip POST; set CONFIRM_SUBMIT=1 and ACW_ALLOW_SUBMIT=1 to submit"
  exit 0
fi
ACW_ALLOW_SUBMIT=1 python3 "$ROOT/cannjudge/tools/cann_submit.py" \
  --problem-id "$PROBLEM_ID" \
  --account "$ACCOUNT" \
  --source-dir "$SOURCE_DIR" \
  --tag "$TAG" --submit
