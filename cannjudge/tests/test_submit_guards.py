"""Offline submit-guard tests; no network and no credentials."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import cann_submit  # noqa: E402


class DigestTests(unittest.TestCase):
    def test_digest_is_order_independent(self):
        a = {"path": "a.cpp", "content": "A"}
        b = {"path": "b.cpp", "content": "B"}
        self.assertEqual(
            cann_submit.source_digest([a, b]),
            cann_submit.source_digest([b, a]),
        )

    def test_digest_changes_with_content(self):
        a = {"path": "a.cpp", "content": "A"}
        self.assertNotEqual(
            cann_submit.source_digest([a]),
            cann_submit.source_digest([{"path": "a.cpp", "content": "B"}]),
        )


class ReadSourcesTests(unittest.TestCase):
    def test_only_editable_files_are_selected(self):
        template = [
            {"path": "op_kernel/a.cpp", "editable": True},
            {"path": "CMakeLists.txt", "editable": False},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "op_kernel").mkdir()
            (root / "op_kernel/a.cpp").write_text("kernel\n", encoding="utf-8")
            (root / "CMakeLists.txt").write_text("build\n", encoding="utf-8")
            selected = cann_submit.read_sources(template, root, {})
        self.assertEqual([x["path"] for x in selected], ["op_kernel/a.cpp"])
        self.assertEqual(selected[0]["content"], "kernel\n")

    def test_missing_file_fails(self):
        template = [{"path": "op_kernel/missing.cpp", "editable": True}]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                cann_submit.read_sources(template, tmp, {})

    def test_nul_byte_is_rejected(self):
        template = [{"path": "op_kernel/bad.cpp", "editable": True}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "op_kernel").mkdir()
            (root / "op_kernel/bad.cpp").write_bytes(b"a\x00b")
            with self.assertRaises(SystemExit):
                cann_submit.read_sources(template, root, {})

    def test_non_utf8_is_rejected(self):
        template = [{"path": "op_kernel/bad.cpp", "editable": True}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "op_kernel").mkdir()
            (root / "op_kernel/bad.cpp").write_bytes(b"\xff\xfe")
            with self.assertRaises(SystemExit):
                cann_submit.read_sources(template, root, {})

    def test_override_maps_template_path(self):
        template = [{"path": "op_kernel/a.cpp", "editable": True}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local = root / "custom.cpp"
            local.write_text("custom\n", encoding="utf-8")
            selected = cann_submit.read_sources(
                template, root, {"op_kernel/a.cpp": str(local)}
            )
        self.assertEqual(selected[0]["content"], "custom\n")


class CompatTests(unittest.TestCase):
    def test_compat_fields_pick_expected_paths(self):
        selected = [
            {"path": "op_host/run.cpp", "content": "host"},
            {"path": "op_kernel/run.cpp", "content": "kernel"},
            {"path": "op_kernel/tiling_key_run.h", "content": "key"},
            {"path": "op_kernel/tiling_run.h", "content": "tiling"},
        ]
        fields = cann_submit.compat_fields(selected)
        self.assertEqual(
            set(fields), {"host_cpp", "kernel_cpp", "tiling_key_h", "tiling_h"}
        )
        self.assertEqual(fields["host_cpp"], "host")
        self.assertEqual(fields["kernel_cpp"], "kernel")


if __name__ == "__main__":
    unittest.main()
