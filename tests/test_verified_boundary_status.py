from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from minicut_agent.core import CutPoint, ProjectModel, load_project_file
from minicut_agent.verified_window import (
    VerifiedMiniCutWindow,
    apply_cut_verification_metadata,
    cut_verification_metadata,
)


class VerificationPersistenceTests(unittest.TestCase):
    def test_project_serializer_keeps_camera_verification_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "movie.mp4"
            source.write_bytes(b"video")
            project = root / "movie.minicut.json"

            host = object.__new__(VerifiedMiniCutWindow)
            host._pending_camera_reviews = [{
                "requested_ms": 20_000,
                "needs_review": True,
                "reason": "boundary belum terbukti",
            }]
            host.model = ProjectModel()
            host.model.source = source
            host.model.duration_ms = 60_000
            cut = CutPoint(12_900, 12_804, "307307/24000")
            apply_cut_verification_metadata(cut, {
                "provenance": "local-camera-boundary",
                "source_pts": 307307,
                "source_time_base": "1/24000",
                "raw_scene_boundary_exact": "307307/24000",
                "pts_verified": True,
                "camera_boundary_verified": True,
                "needs_review": False,
                "fallback_to_nearest_frame": False,
                "requires_camera_boundary": True,
                "verification_reason": "exact PTS",
                "verification_status": "verified",
            })
            host.model.cuts = [cut]
            host._install_project_serializer()
            host.model.save(project)

            data, loaded_source = load_project_file(project)
            self.assertEqual(loaded_source, source)
            self.assertGreaterEqual(int(data["version"]), 3)
            self.assertEqual(data["camera_boundary_schema"], 1)
            self.assertEqual(len(data["pending_camera_reviews"]), 1)
            saved = data["cuts"][0]
            self.assertTrue(saved["pts_verified"])
            self.assertTrue(saved["camera_boundary_verified"])
            self.assertFalse(saved["needs_review"])
            self.assertEqual(saved["source_time_base"], "1/24000")
            self.assertEqual(saved["raw_scene_boundary_exact"], "307307/24000")

    def test_metadata_helpers_round_trip_on_cutpoint(self):
        cut = CutPoint(1000, 1000, "1")
        expected = {
            "provenance": "test",
            "pts_verified": True,
            "camera_boundary_verified": True,
            "needs_review": False,
            "fallback_to_nearest_frame": False,
            "requires_camera_boundary": True,
            "verification_reason": "ok",
            "verification_status": "verified",
        }
        apply_cut_verification_metadata(cut, expected)
        actual = cut_verification_metadata(cut)
        for key, value in expected.items():
            self.assertEqual(actual[key], value)


class RequiredBoundaryToolTests(unittest.TestCase):
    class ExportModeStub:
        def __init__(self):
            self.value = "smartcut"

        def currentData(self):
            return self.value

        def findData(self, value):
            return 0 if value == "smartcut" else -1

        def setCurrentIndex(self, _index):
            self.value = "smartcut"

    def _host(self, source: Path):
        host = object.__new__(VerifiedMiniCutWindow)
        host.model = ProjectModel()
        host.model.source = source
        host.model.duration_ms = 60_000
        host.export_mode = self.ExportModeStub()
        host._pending_camera_reviews = []
        host._ensure_timeline_mutation_idle = lambda: None
        host._require_tool_media_ready = lambda: None
        host._refresh = lambda: None
        return host

    def test_missing_camera_change_creates_review_without_installing_cut(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "movie.mp4"
            source.write_bytes(b"video")
            host = self._host(source)
            unresolved = {
                "camera_change_found": False,
                "camera_boundary_verified": False,
                "pts_verified": False,
                "needs_review": True,
                "fallback_to_nearest_frame": False,
                "raw_scene_boundary_ms": None,
                "raw_scene_boundary_exact": None,
                "reason": "tidak ada boundary",
            }
            with patch(
                "minicut_agent.verified_window.find_tool",
                side_effect=lambda name: name,
            ), patch(
                "minicut_agent.verified_window.resolve_camera_boundary",
                return_value=unresolved,
            ) as resolver:
                result = host.tool_add_cut(12_123)

            self.assertTrue(result["ok"])
            self.assertFalse(result["applied"])
            self.assertTrue(result["needs_review"])
            self.assertFalse(result["fallback_to_nearest_frame"])
            self.assertEqual(host.model.cuts, [])
            self.assertEqual(len(host._pending_camera_reviews), 1)
            self.assertTrue(resolver.call_args.kwargs["require_camera_change"])

    def test_smartcut_guard_rejects_unverified_legacy_camera_cut(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "movie.mp4"
            source.write_bytes(b"video")
            host = self._host(source)
            cut = CutPoint(12_000, 12_000, "12")
            apply_cut_verification_metadata(cut, {
                "requires_camera_boundary": True,
                "camera_boundary_verified": False,
                "needs_review": True,
            })
            host.model.cuts = [cut]
            error = host._camera_export_error()
            self.assertIsNotNone(error)
            self.assertIn("belum terverifikasi", error)


if __name__ == "__main__":
    unittest.main()
