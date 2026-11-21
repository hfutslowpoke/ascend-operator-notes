"""Offline score/headroom tests; no network and no credentials."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import cann_score  # noqa: E402


class ScoreFormulaTests(unittest.TestCase):
    def test_at_tbest_is_100(self):
        self.assertAlmostEqual(cann_score.case_score(3.5, 3.5), 100.0)

    def test_1_5x_is_50(self):
        self.assertAlmostEqual(cann_score.case_score(1.5, 1.0), 50.0, places=6)

    def test_2_25x_is_about_33(self):
        self.assertAlmostEqual(cann_score.case_score(2.25, 1.0), 33.3333, places=3)

    def test_ratio_for_score_round_trip(self):
        ratio = cann_score.ratio_for_score(60.0)
        self.assertAlmostEqual(cann_score.case_score(ratio, 1.0), 60.0, places=6)

    def test_invalid_values_return_none(self):
        self.assertIsNone(cann_score.case_score(None, 1.0))
        self.assertIsNone(cann_score.case_score(1.0, None))
        self.assertIsNone(cann_score.case_score(0, 1.0))
        self.assertIsNone(cann_score.case_score(1.0, 0))


class AnalyzeTests(unittest.TestCase):
    def test_total_and_headroom(self):
        results = [
            {"testcase_id": "a", "time": 1.5, "testcase_status": "Pass"},
            {"testcase_id": "b", "time": 2.25, "testcase_status": "Pass"},
        ]
        testcases = [{"id": "a", "tbest": 1.0}, {"id": "b", "tbest": 1.0}]
        summary = cann_score.analyze(results, testcases, target=60.0)
        self.assertEqual(summary["valid_cases"], 2)
        self.assertAlmostEqual(summary["total"], 41.66, places=2)
        self.assertAlmostEqual(summary["all_at_1_5x"], 50.0, places=3)
        # A single case cannot close a 60 average when the other stays low.
        self.assertTrue(
            all(c.get("target_state") != "already_enough" for c in summary["cases"])
        )

    def test_missing_tbest_falls_back_to_best_time(self):
        results = [
            {"testcase_id": "a", "time": 2.0, "best_time": 1.0, "testcase_status": "Pass"}
        ]
        summary = cann_score.analyze(results, [])
        self.assertEqual(summary["valid_cases"], 1)
        self.assertEqual(summary["cases"][0]["tbest"], 1.0)

    def test_failed_case_has_no_score(self):
        results = [
            {"testcase_id": "a", "time": None, "testcase_status": "Runtime Error"}
        ]
        summary = cann_score.analyze(results, [{"id": "a", "tbest": 1.0}])
        self.assertEqual(summary["valid_cases"], 0)
        self.assertIsNone(summary["total"])


class CliTests(unittest.TestCase):
    def test_ranking_with_sid(self):
        ranking = {
            "testcases": [{"id": "a", "tbest": 1.0}],
            "rows": [
                {
                    "submission_id": "other",
                    "score": 99.0,
                    "result": [{"testcase_id": "a", "time": 1.0}],
                },
                {
                    "submission_id": "mine",
                    "score": 50.0,
                    "result": [{"testcase_id": "a", "time": 1.5}],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ranking.json"
            path.write_text(json.dumps(ranking), encoding="utf-8")
            args = ["--ranking", str(path), "--sid", "mine", "--json"]
            import subprocess

            output = subprocess.check_output(
                [sys.executable, str(Path(cann_score.__file__)), *args],
                text=True,
            )
        parsed = json.loads(output)
        self.assertEqual(parsed["total"], 50.0)
        self.assertEqual(parsed["valid_cases"], 1)

    def test_ranking_without_sid_is_rejected_when_multiple_rows(self):
        ranking = {
            "testcases": [{"id": "a", "tbest": 1.0}],
            "rows": [
                {"submission_id": "a", "score": 99.0, "result": []},
                {"submission_id": "b", "score": 50.0, "result": []},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ranking.json"
            path.write_text(json.dumps(ranking), encoding="utf-8")
            import subprocess

            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.check_output(
                    [sys.executable, str(Path(cann_score.__file__)), "--ranking", str(path)],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )


if __name__ == "__main__":
    unittest.main()
