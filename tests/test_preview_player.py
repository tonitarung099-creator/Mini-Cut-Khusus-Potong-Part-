import unittest
from pathlib import Path

from minicut_agent.preview_player import PreviewPlayer


class _FakePlayer:
    _apply_mpv_pending_seek = PreviewPlayer._apply_mpv_pending_seek

    def __init__(self):
        self._load_generation = 2
        self._source = Path("new.mp4")
        self.using_mpv = True
        self.seeks = []

    def set_position(self, ms):
        self.seeks.append(ms)


class PreviewPlayerRaceTests(unittest.TestCase):
    def test_stale_seek_from_previous_media_is_ignored(self):
        player = _FakePlayer()
        player._apply_mpv_pending_seek(1, Path("old.mp4"), 55_000)
        self.assertEqual(player.seeks, [])

    def test_current_media_delayed_seek_is_applied(self):
        player = _FakePlayer()
        player._apply_mpv_pending_seek(2, Path("new.mp4"), 12_000)
        self.assertEqual(player.seeks, [12_000])


if __name__ == "__main__":
    unittest.main()
