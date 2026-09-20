import os
import tempfile
import unittest
from pathlib import Path

from minicut_agent.core import preview_proxy_path


class PreviewProxyTests(unittest.TestCase):
    def test_proxy_cache_path_is_stable_for_same_source(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "movie.mp4"
            source.write_bytes(b"abc")
            a = preview_proxy_path(source)
            b = preview_proxy_path(source)
            self.assertEqual(a, b)
            self.assertEqual(a.suffix, ".mp4")
            self.assertIn("preview-proxy", str(a.parent))

    def test_proxy_cache_changes_when_source_changes(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "movie.mp4"
            source.write_bytes(b"abc")
            a = preview_proxy_path(source)
            source.write_bytes(b"abcdef")
            os.utime(source, None)
            b = preview_proxy_path(source)
            self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
