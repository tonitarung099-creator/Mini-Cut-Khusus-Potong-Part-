import unittest

from minicut_agent.core import parse_time_ms
from minicut_agent.manual_commands import extract_manual_timestamps, looks_like_manual_cut


class ManualCommandTimestampTests(unittest.TestCase):
    def times(self, text: str) -> list[int]:
        return [item.time_ms for item in extract_manual_timestamps(text)]

    def test_hour_more_minutes(self):
        self.assertEqual(self.times("cut 1 jam lebih 2 menit"), [3_720_000])

    def test_hour_minutes_seconds(self):
        self.assertEqual(
            self.times("potong 1 jam 2 menit 3 detik"),
            [3_723_000],
        )

    def test_hour_lewat_minutes_seconds(self):
        self.assertEqual(
            self.times("cut 1 jam lewat 2 menit 30 detik"),
            [3_750_000],
        )

    def test_long_minutes_are_valid(self):
        self.assertEqual(self.times("potong di 62 menit"), [3_720_000])

    def test_hms_stays_single_timestamp(self):
        self.assertEqual(
            self.times("cut 01:02:03.250"),
            [3_723_250],
        )

    def test_cut_intent(self):
        self.assertTrue(looks_like_manual_cut("tolong cut 1 jam lebih 2 menit"))

    def test_remove_cut_is_not_add_cut_intent(self):
        self.assertFalse(looks_like_manual_cut("hapus cut di 1 jam lebih 2 menit"))

    def test_interval_request_is_not_hijacked_as_one_manual_cut(self):
        self.assertFalse(looks_like_manual_cut("potong tiap 1 jam 12 detik"))
        self.assertFalse(looks_like_manual_cut("setiap part 1 jam 12 detik"))
        self.assertFalse(looks_like_manual_cut("potong berkala interval 62 menit"))

    def test_plain_arbitrary_timestamp_still_uses_manual_cut(self):
        self.assertTrue(looks_like_manual_cut("potong 1jam 12dtik"))

    def test_compact_typo_seconds_are_not_lost(self):
        self.assertEqual(self.times("potong 1jam 12dtik"), [3_612_000])

    def test_compact_english_units(self):
        self.assertEqual(self.times("cut 1h12s"), [3_612_000])

    def test_compact_minute_second_aliases(self):
        self.assertEqual(self.times("potong 62mnt 5dtk"), [3_725_000])

    def test_unknown_suffix_does_not_lock_partial_hour(self):
        self.assertEqual(self.times("potong 1jam 12xyz"), [])

    def test_parse_time_ms_accepts_natural_and_compact_units(self):
        self.assertEqual(parse_time_ms("1 jam 12 dtik"), 3_612_000)
        self.assertEqual(parse_time_ms("1h12s"), 3_612_000)
        self.assertEqual(parse_time_ms("62mnt 5dtk"), 3_725_000)


if __name__ == "__main__":
    unittest.main()
