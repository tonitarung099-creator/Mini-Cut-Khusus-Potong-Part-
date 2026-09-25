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

    def test_cue_ending_exactly_at_boundary_stays_only_in_previous_part(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "boundary-end.srt"
            path.write_text(
                "1\n"
                "00:00:04,000 --> 00:00:05,000\n"
                "Selesai sebelum cut\n",
                encoding="utf-8",
            )
            track = SubtitleTrack.load(path)

            part1 = track.to_srt_range(0, 5_000)
            part2 = track.to_srt_range(5_000, 10_000)

            self.assertIn("Selesai sebelum cut", part1)
            self.assertNotIn("Selesai sebelum cut", part2)

    def test_cue_starting_exactly_at_boundary_stays_only_in_next_part(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "boundary-start.srt"
            path.write_text(
                "1\n"
                "00:00:05,000 --> 00:00:06,000\n"
                "Mulai sesudah cut\n",
                encoding="utf-8",
            )
            track = SubtitleTrack.load(path)

            part1 = track.to_srt_range(0, 5_000)
            part2 = track.to_srt_range(5_000, 10_000)

            self.assertNotIn("Mulai sesudah cut", part1)
            self.assertIn(
                "00:00:00,000 --> 00:00:01,000\nMulai sesudah cut",
                part2,
            )

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

    def test_multiline_export_stays_multiline_but_gemini_context_stays_compact(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            track = SubtitleTrack.load(self._source_srt(root))

            exported = track.to_srt_range(0, 5_000)
            context = track.between_text(0, 5_000)

            self.assertIn("Halo\ndunia", exported)
            self.assertIn("Halo dunia", context)
            self.assertNotIn("Halo\ndunia", context)

    def test_parser_accepts_missing_blank_line_between_cues(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "rapat.srt"
            path.write_text(
                "1\n"
                "00:00:01,000 --> 00:00:02,000\n"
                "Baris satu\n"
                "2\n"
                "00:00:03,000 --> 00:00:04,000\n"
                "Baris dua\n",
                encoding="utf-8",
            )

            track = SubtitleTrack.load(path)

            self.assertEqual(len(track.cues), 2)
            self.assertEqual(track.cues[0].text, "Baris satu")
            self.assertEqual(track.cues[1].text, "Baris dua")
            self.assertEqual(track.cues[1].start_ms, 3_000)

    def test_parser_scales_short_fractional_milliseconds(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "fraction.srt"
            path.write_text(
                "1\n"
                "00:00:01,5 --> 00:00:02,25\n"
                "Tes pecahan\n",
                encoding="utf-8",
            )

            track = SubtitleTrack.load(path)

            self.assertEqual(track.cues[0].start_ms, 1_500)
            self.assertEqual(track.cues[0].end_ms, 2_250)

    def test_parser_rejects_invalid_minute_or_second_fields(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bad_minute = root / "bad-minute.srt"
            bad_minute.write_text(
                "1\n00:61:00,000 --> 00:61:01,000\nSalah menit\n",
                encoding="utf-8",
            )
            bad_second = root / "bad-second.srt"
            bad_second.write_text(
                "1\n00:00:70,000 --> 00:00:71,000\nSalah detik\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "di luar rentang"):
                SubtitleTrack.load(bad_minute)
            with self.assertRaisesRegex(ValueError, "di luar rentang"):
                SubtitleTrack.load(bad_second)

    def test_parser_rejects_four_digit_millisecond_field(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad-fraction.srt"
            path.write_text(
                "1\n"
                "00:00:01,1234 --> 00:00:02,000\n"
                "Ambigu\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "tidak berisi cue|Timestamp"):
                SubtitleTrack.load(path)

    def test_parser_preserves_windows_1252_text_without_replacement(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cp1252.srt"
            payload = (
                "1\n"
                "00:00:00,000 --> 00:00:01,000\n"
                "café – bagus\n"
            )
            path.write_bytes(payload.encode("cp1252"))

            track = SubtitleTrack.load(path)

            self.assertEqual(track.cues[0].text, "café – bagus")
            self.assertNotIn("�", track.cues[0].text)

    def test_parser_reads_utf16_bom(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "utf16.srt"
            payload = (
                "1\n"
                "00:00:00,000 --> 00:00:01,000\n"
                "Halo dunia\n"
            )
            path.write_bytes(payload.encode("utf-16"))

            track = SubtitleTrack.load(path)

            self.assertEqual(track.cues[0].text, "Halo dunia")

    def test_parser_rejects_reversed_or_zero_duration_cue(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reversed_path = root / "reversed.srt"
            reversed_path.write_text(
                "1\n"
                "00:00:02,000 --> 00:00:01,000\n"
                "Jangan hilangkan dialog ini\n",
                encoding="utf-8",
            )
            zero_path = root / "zero.srt"
            zero_path.write_text(
                "1\n"
                "00:00:01,000 --> 00:00:01,000\n"
                "Durasi nol\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Durasi cue SRT tidak valid"):
                SubtitleTrack.load(reversed_path)
            with self.assertRaisesRegex(ValueError, "Durasi cue SRT tidak valid"):
                SubtitleTrack.load(zero_path)

    def test_parser_rejects_negative_or_prefixed_timestamp(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            negative = root / "negative.srt"
            negative.write_text(
                "1\n"
                "-00:00:01,000 --> 00:00:02,000\n"
                "Negatif\n",
                encoding="utf-8",
            )
            prefixed = root / "prefixed.srt"
            prefixed.write_text(
                "1\n"
                "abc00:00:01,000 --> 00:00:02,000\n"
                "Prefix\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "tidak berisi cue|Timestamp"):
                SubtitleTrack.load(negative)
            with self.assertRaisesRegex(ValueError, "tidak berisi cue|Timestamp"):
                SubtitleTrack.load(prefixed)

    def test_cue_spanning_multiple_boundaries_is_clipped_to_every_matching_part(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "long.srt"
            path.write_text(
                "1\n"
                "00:00:04,000 --> 00:00:11,000\n"
                "Dialog panjang\n",
                encoding="utf-8",
            )
            track = SubtitleTrack.load(path)

            part1 = track.to_srt_range(0, 5_000)
            part2 = track.to_srt_range(5_000, 10_000)
            part3 = track.to_srt_range(10_000, 15_000)

            self.assertIn(
                "00:00:04,000 --> 00:00:05,000\nDialog panjang",
                part1,
            )
            self.assertIn(
                "00:00:00,000 --> 00:00:05,000\nDialog panjang",
                part2,
            )
            self.assertIn(
                "00:00:00,000 --> 00:00:01,000\nDialog panjang",
                part3,
            )

    def test_write_srt_parts_rejects_gap_between_parts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self._source_srt(root)

            with self.assertRaisesRegex(ValueError, "tidak kontinu.*gap"):
                write_srt_parts(
                    source,
                    root / "out-gap",
                    "Film",
                    [(0, 5_000), (5_001, 10_000)],
                )

    def test_write_srt_parts_rejects_overlap_between_parts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self._source_srt(root)

            with self.assertRaisesRegex(ValueError, "tidak kontinu.*overlap"):
                write_srt_parts(
                    source,
                    root / "out-overlap",
                    "Film",
                    [(0, 5_000), (4_999, 10_000)],
                )

    def test_write_srt_parts_rejects_incomplete_full_duration(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self._source_srt(root)

            with self.assertRaisesRegex(ValueError, "tidak menutup durasi video"):
                write_srt_parts(
                    source,
                    root / "out-short",
                    "Film",
                    [(0, 5_000), (5_000, 9_999)],
                    expected_duration_ms=10_000,
                )

    def test_write_srt_parts_rejects_full_export_not_starting_at_zero(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self._source_srt(root)

            with self.assertRaisesRegex(ValueError, "harus dimulai"):
                write_srt_parts(
                    source,
                    root / "out-offset",
                    "Film",
                    [(1, 5_000), (5_000, 10_000)],
                    expected_duration_ms=10_000,
                )

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
