import unittest

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


if __name__ == "__main__":
    unittest.main()
