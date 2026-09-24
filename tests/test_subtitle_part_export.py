import tempfile
import unittest
from pathlib import Path

from minicut_agent.subtitles import SubtitleTrack, write_srt_parts


class SubtitlePartExportTests(unittest.TestCase):
    def _source_srt(self, root: Path) -> Path:
        path = root / "Film.srt"
        path.write_text(
            "1\n"
            "00:00:01,000 --> 00:00:03,000\n"
            "Halo\n"
            "dunia\n\n"
            "2\n"
            "00:00:04,500 --> 00:00:06,500\n"
            "Melewati batas\n\n"
            "3\n"
            "00:00:07,000 --> 00:00:08,000\n"
            "Part dua\n",
            encoding="utf-8",
        )
        return path

    def test_cross_boundary_cue_is_clipped_into_both_parts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            track = SubtitleTrack.load(self._source_srt(root))

            part1 = track.to_srt_range(0, 5_000)
            part2 = track.to_srt_range(5_000, 10_000)

            self.assertIn(
                "00:00:04,500 --> 00:00:05,000\nMelewati batas",
                part1,
            )
            self.assertIn(
                "00:00:00,000 --> 00:00:01,500\nMelewati batas",
                part2,
            )
            self.assertNotIn("00:00:05,000 -->", part2)

    def test_each_part_is_renumbered_and_multiline_text_is_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            track = SubtitleTrack.load(self._source_srt(root))

            part1 = track.to_srt_range(0, 5_000)
            part2 = track.to_srt_range(5_000, 10_000)

            self.assertTrue(part1.startswith("1\n00:00:01,000"))
            self.assertIn("Halo\ndunia", part1)
            self.assertTrue(part2.startswith("1\n00:00:00,000"))
            self.assertIn("\n\n2\n00:00:02,000 --> 00:00:03,000\nPart dua", part2)

    def test_write_srt_parts_creates_matching_part_names(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self._source_srt(root)
            out = root / "out"

            files = write_srt_parts(
                source,
                out,
                "Film",
                [(0, 5_000), (5_000, 10_000)],
            )

            self.assertEqual(
                [path.name for path in files],
                ["Film_Part-01.srt", "Film_Part-02.srt"],
            )
            self.assertTrue(all(path.is_file() for path in files))
            self.assertIn(
                "00:00:00,000 --> 00:00:01,500",
                files[1].read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
