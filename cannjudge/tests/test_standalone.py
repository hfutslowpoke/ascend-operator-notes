"""Regression checks for the shared lock and archived command paths."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import cann_submit

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "cannjudge/tools"


class StandaloneTests(unittest.TestCase):
    def test_archived_submit_uses_canonical_auth_guard(self):
        command = [
            sys.executable, str(ROOT / "tools/competition/tools/cann_submit.py"),
            "--problem-id", "test", "--account", "test", "--source-dir", ".",
            "--tag", "test", "--submit",
        ]
        result = subprocess.run(command, env=dict(os.environ, ACW_ALLOW_SUBMIT="0", CANNJUDGE_ALLOW_SUBMIT="0"),
                                text=True, capture_output=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ACW_ALLOW_SUBMIT=1", result.stderr)

    def test_recipe_build_does_not_wait_for_device_lock(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            manifest = {"cases": [{"id": "test"}]}
            (root / "cases.json").write_text(json.dumps(manifest))
            (root / "environment.json").write_text(json.dumps({"cann": {"home": name, "set_env": "unused"}}))
            (root / "adapter.json").write_text(json.dumps({
                "evaluation_scope": "local",
                "commands": {"build": ["{python}", "-c", "print('build ok')"]},
            }))
            (root / "binding.json").write_text(json.dumps({
                "cases_ref": "cases.json", "golden_ref": "cases.json",
                "environment_ref": "environment.json", "score_adapter_ref": "adapter.json",
            }))
            lock_file = root / "npu.lock"
            with lock_file.open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                completed = subprocess.run(
                    [sys.executable, str(TOOLS / "cann_eval.py"), "build", "--binding",
                     str(root / "binding.json"), "--run-dir", str(root / "run")],
                    env=dict(os.environ, NPU_LOCK=str(lock_file)), text=True,
                    capture_output=True, check=True, timeout=10,
                )
            self.assertIn("build ok", completed.stdout)

    def test_submit_lock_is_shared_across_ledgers(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "submit.lock"
            with patch.dict(os.environ, ACW_SUBMIT_LOCK=str(path)), cann_submit.submission_lock():
                with path.open("a") as other:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with path.open("a") as other:
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
