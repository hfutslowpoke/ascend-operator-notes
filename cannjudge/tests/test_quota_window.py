"""Pure quota-window tests; no network and no credentials."""

import datetime
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import cann_quota  # noqa: E402

BEIJING = datetime.timezone(datetime.timedelta(hours=8))
UTC = datetime.timezone.utc


class QuotaWindowTests(unittest.TestCase):
    def test_after_noon_starts_today(self):
        now = datetime.datetime(2026, 11, 19, 16, 13, tzinfo=BEIJING)
        start, end = cann_quota.quota_window(now)
        self.assertEqual(start, datetime.datetime(2026, 11, 19, 12, 0, tzinfo=BEIJING))
        self.assertEqual(end, datetime.datetime(2026, 11, 20, 12, 0, tzinfo=BEIJING))

    def test_before_noon_starts_yesterday(self):
        now = datetime.datetime(2026, 11, 19, 11, 59, tzinfo=BEIJING)
        start, end = cann_quota.quota_window(now)
        self.assertEqual(start, datetime.datetime(2026, 11, 18, 12, 0, tzinfo=BEIJING))
        self.assertEqual(end, datetime.datetime(2026, 11, 19, 12, 0, tzinfo=BEIJING))

    def test_utc_input_is_converted(self):
        now = datetime.datetime(2026, 11, 18, 8, 37, tzinfo=UTC)
        start, end = cann_quota.quota_window(now)
        self.assertEqual(start, datetime.datetime(2026, 11, 18, 12, 0, tzinfo=BEIJING))
        self.assertEqual(end, datetime.datetime(2026, 11, 19, 12, 0, tzinfo=BEIJING))

    def test_count_window_uses_half_open_interval(self):
        start = datetime.datetime(2026, 11, 18, 12, 0, tzinfo=BEIJING)
        end = datetime.datetime(2026, 11, 19, 12, 0, tzinfo=BEIJING)
        rows = [
            {"create_time": "2026-11-18T03:59:59Z"},  # 11:59:59 Beijing, before
            {"create_time": "2026-11-18T04:00:00Z"},  # 12:00:00 Beijing, inside
            {"create_time": "2026-11-19T03:59:59Z"},  # 11:59:59 Beijing, inside
            {"create_time": "2026-11-19T04:00:00Z"},  # 12:00:00 Beijing, after
            {"create_time": None},
            {"create_time": "not-a-time"},
        ]
        selected = cann_quota.count_window(rows, start, end)
        self.assertEqual(len(selected), 2)

    def test_parse_time_normalizes_to_beijing(self):
        parsed = cann_quota.parse_time("2026-11-18T08:37:00Z")
        self.assertEqual(parsed, datetime.datetime(2026, 11, 18, 16, 37, tzinfo=BEIJING))

    def test_parse_time_rejects_invalid(self):
        self.assertIsNone(cann_quota.parse_time(""))
        self.assertIsNone(cann_quota.parse_time("not-a-time"))


if __name__ == "__main__":
    unittest.main()
