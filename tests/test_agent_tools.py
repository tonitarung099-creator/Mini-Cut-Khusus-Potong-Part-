import queue
import mcp_server
import unittest
from unittest.mock import patch

from minicut_agent.agent import AgentPlanner, MUTATING_TOOLS, ToolRegistry
from minicut_agent.bridge import BridgeCall
from minicut_agent.core import CutPoint, ProjectModel
from minicut_agent.ui import MiniCutWindow


class ToolManifestCoverageTests(unittest.TestCase):
    def test_mcp_companion_matches_manifest(self):
        names = {
            spec["name"]
            for spec in ToolRegistry(object()).manifest()["tools"]
        }
        missing = sorted(
            name for name in names
            if not callable(getattr(mcp_server, name, None))
        )
        self.assertEqual(missing, [])

    def test_apply_film_cut_is_treated_as_timeline_mutation(self):
        self.assertIn("apply_film_cut", MUTATING_TOOLS)

    def test_choose_subtitle_is_treated_as_project_mutation(self):
        self.assertIn("choose_subtitle", MUTATING_TOOLS)

    def test_every_manifest_tool_has_ui_executor(self):
        registry = ToolRegistry(object())
        missing = [
            spec["name"]
            for spec in registry.manifest()["tools"]
            if not hasattr(MiniCutWindow, f"tool_{spec['name']}")
        ]
        self.assertEqual(missing, [])

    def test_major_gemini_controls_are_exposed(self):
        names = {x["name"] for x in ToolRegistry(object()).manifest()["tools"]}
        expected = {
            "open_video", "open_project", "choose_subtitle",
            "seek", "play", "pause", "set_playback_rate", "step_frame",
            "add_cut", "remove_cut", "clear_cuts", "divide_equal", "divide_interval",
            "start_film_cut", "cancel_film_cut", "apply_film_cut", "preview_film_cut",
            "set_export_mode", "save_project", "export_all", "undo",
        }
        self.assertTrue(expected.issubset(names))


class _TransactionalHost:
    def __init__(self):
        self.cuts = ["awal"]
        self.dirty = False
        self.calls = []

    def tool_transaction_snapshot(self):
        return {"cuts": list(self.cuts), "dirty": self.dirty}

    def tool_transaction_restore(self, snapshot):
        self.cuts = list(snapshot["cuts"])
        self.dirty = bool(snapshot["dirty"])

    def tool_clear_cuts(self):
        self.calls.append("clear_cuts")
        self.cuts = []
        self.dirty = True
        return {"ok": True}

    def tool_remove_cut(self, index):
        self.calls.append("remove_cut")
        raise RuntimeError("simulasi gagal")

    def tool_save_project(self):
        self.calls.append("save_project")
        return {"ok": False, "cancelled": True}

    def tool_open_video(self):
        self.calls.append("open_video")
        return {"ok": True, "loading": True, "stop_plan": True}


class AgentTransactionTests(unittest.TestCase):
    def test_failed_multistep_plan_rolls_back_timeline(self):
        host = _TransactionalHost()
        planner = AgentPlanner(ToolRegistry(host))
        with self.assertRaises(RuntimeError):
            planner.apply({
                "steps": [
                    {"tool": "clear_cuts", "args": {}},
                    {"tool": "remove_cut", "args": {"index": 0}},
                ]
            })
        self.assertEqual(host.cuts, ["awal"])
        self.assertFalse(host.dirty)
        self.assertEqual(host.calls, ["clear_cuts", "remove_cut"])

    def test_cancelled_tool_stops_following_mutation(self):
        host = _TransactionalHost()
        planner = AgentPlanner(ToolRegistry(host))
        with self.assertRaises(RuntimeError):
            planner.apply({
                "steps": [
                    {"tool": "save_project", "args": {}},
                    {"tool": "clear_cuts", "args": {}},
                ]
            })
        self.assertEqual(host.calls, ["save_project"])
        self.assertEqual(host.cuts, ["awal"])
        self.assertFalse(host.dirty)

    def test_async_boundary_stops_following_steps_without_error(self):
        host = _TransactionalHost()
        planner = AgentPlanner(ToolRegistry(host))
        results = planner.apply({
            "steps": [
                {"tool": "open_video", "args": {}},
                {"tool": "clear_cuts", "args": {}},
            ]
        })
        self.assertEqual(host.calls, ["open_video"])
        self.assertEqual(len(results), 1)
        self.assertEqual(host.cuts, ["awal"])
        self.assertFalse(host.dirty)


    def test_save_project_terminal_step_stops_following_tools(self):
        class Host(_TransactionalHost):
            def tool_save_project(self):
                self.calls.append("save_project")
                return {"ok": True, "path": "movie.minicut.json", "stop_plan": True}

        host = Host()
        planner = AgentPlanner(ToolRegistry(host))
        results = planner.apply({
            "steps": [
                {"tool": "clear_cuts", "args": {}},
                {"tool": "save_project", "args": {}},
                {"tool": "remove_cut", "args": {"index": 0}},
            ]
        })

        self.assertEqual(host.calls, ["clear_cuts", "save_project"])
        self.assertEqual(len(results), 2)
        self.assertEqual(host.cuts, [])
        self.assertTrue(host.dirty)


class _UndoHost:
    _snapshot = MiniCutWindow._snapshot
    _restore_snapshot = MiniCutWindow._restore_snapshot
    _require_tool_project_ready = MiniCutWindow._require_tool_project_ready
    _require_tool_media_ready = MiniCutWindow._require_tool_media_ready
    tool_undo = MiniCutWindow.tool_undo

    def __init__(self):
        self.model = ProjectModel()
        self.model.source = __import__("pathlib").Path("movie.mp4")
        self.model.duration_ms = 120_000
        self.analyze_worker = None
        self.undo_stack = []

    def _refresh(self):
        pass


class UndoStateTests(unittest.TestCase):
    def test_undo_restores_previous_dirty_state(self):
        host = _UndoHost()
        host.model.cuts = [CutPoint(30_000, 30_000)]
        host.model.dirty = False
        before = host._snapshot()

        host.model.cuts.append(CutPoint(60_000, 60_000))
        host.model.dirty = True
        host.undo_stack.append(before)

        result = host.tool_undo()
        self.assertTrue(result["ok"])
        self.assertFalse(result["dirty"])
        self.assertFalse(host.model.dirty)
        self.assertEqual(
            [cut.actual_ms for cut in host.model.cuts],
            [30_000],
        )


    def test_undo_restores_subtitle_and_film_cut_state(self):
        from pathlib import Path

        host = _UndoHost()
        host.srt_path = Path("before.srt")
        host._srt_project_reference = Path("before.srt")
        host._srt_auto_disabled = False
        host._srt_user_disabled = False
        host.film_cut_results = [{
            "target_ms": 10_000,
            "selected_time_ms": 10_040,
            "confidence": 0.9,
        }]
        host.model.dirty = False
        before = host._snapshot()

        host.srt_path = Path("after.srt")
        host._srt_project_reference = Path("after.srt")
        host._srt_auto_disabled = True
        host._srt_user_disabled = True
        host.film_cut_results = []
        host.model.dirty = True
        host.undo_stack.append(before)

        result = host.tool_undo()

        self.assertTrue(result["ok"])
        self.assertEqual(host.srt_path, Path("before.srt"))
        self.assertEqual(
            host._srt_project_reference,
            Path("before.srt"),
        )
        self.assertFalse(host._srt_auto_disabled)
        self.assertFalse(host._srt_user_disabled)
        self.assertEqual(len(host.film_cut_results), 1)
        self.assertEqual(
            host.film_cut_results[0]["selected_time_ms"],
            10_040,
        )
        self.assertFalse(host.model.dirty)


class SavedPlanUndoSafetyTests(unittest.TestCase):
    def test_undo_snapshot_is_dirty_after_plan_that_saved_project(self):
        class Sink:
            def setPlainText(self, _text):
                pass

            def setEnabled(self, _enabled):
                pass

        class Host:
            _snapshot = MiniCutWindow._snapshot
            _apply_plan = MiniCutWindow._apply_plan

            def __init__(self):
                self.model = ProjectModel()
                self.model.source = __import__("pathlib").Path("movie.mp4")
                self.model.duration_ms = 120_000
                self.model.cuts = [CutPoint(30_000, 30_000)]
                self.model.dirty = False
                self.undo_stack = []
                self.pending_plan = {
                    "summary": "ubah lalu simpan",
                    "steps": [],
                }
                self.plan_preview = Sink()
                self.apply_btn = Sink()

            def _log(self, _text):
                pass

            def _refresh(self):
                pass

        host = Host()

        class Planner:
            def apply(self, _plan):
                host.model.cuts = [CutPoint(60_000, 60_000)]
                host.model.dirty = False
                return [
                    {
                        "step": {"tool": "clear_cuts", "args": {}},
                        "result": {"ok": True},
                    },
                    {
                        "step": {"tool": "save_project", "args": {}},
                        "result": {
                            "ok": True,
                            "path": "movie.minicut.json",
                            "stop_plan": True,
                        },
                    },
                ]

        host.planner = Planner()
        host._apply_plan()

        self.assertEqual(len(host.undo_stack), 1)
        self.assertTrue(host.undo_stack[0]["dirty"])
        self.assertEqual(
            [cut.actual_ms for cut in host.undo_stack[0]["cuts"]],
            [30_000],
        )


class SubtitleFilmCutConcurrencyTests(unittest.TestCase):
    def test_choose_subtitle_is_blocked_while_film_cut_runs(self):
        class Host:
            _choose_srt = MiniCutWindow._choose_srt

        host = Host()
        host.film_cut_worker = _RunningWorker()

        with patch(
            "minicut_agent.ui.QMessageBox.information",
        ) as info, patch(
            "minicut_agent.ui.QFileDialog.getOpenFileName",
        ) as picker:
            selected = host._choose_srt()

        self.assertFalse(selected)
        info.assert_called_once()
        picker.assert_not_called()

    def test_clear_subtitle_is_blocked_while_film_cut_runs(self):
        from pathlib import Path

        class Host:
            _clear_srt = MiniCutWindow._clear_srt

        host = Host()
        host.film_cut_worker = _RunningWorker()
        host.srt_path = Path("movie.srt")
        host._srt_project_reference = Path("movie.srt")
        host._srt_auto_disabled = False
        host._srt_user_disabled = False

        with patch(
            "minicut_agent.ui.QMessageBox.information",
        ) as info:
            host._clear_srt()

        info.assert_called_once()
        self.assertEqual(host.srt_path, Path("movie.srt"))
        self.assertEqual(
            host._srt_project_reference,
            Path("movie.srt"),
        )
        self.assertFalse(host._srt_auto_disabled)
        self.assertFalse(host._srt_user_disabled)


class WorkerFinalizationRaceTests(unittest.TestCase):
    def test_choose_subtitle_stays_blocked_until_film_worker_callback_finishes(self):
        class Host:
            _choose_srt = MiniCutWindow._choose_srt

        host = Host()
        host.film_cut_worker = _StoppedWorker()

        with patch(
            "minicut_agent.ui.QMessageBox.information",
        ) as info, patch(
            "minicut_agent.ui.QFileDialog.getOpenFileName",
        ) as picker:
            selected = host._choose_srt()

        self.assertFalse(selected)
        info.assert_called_once()
        picker.assert_not_called()

    def test_export_stays_blocked_until_previous_export_callback_finishes(self):
        import tempfile
        from pathlib import Path

        class Host:
            _require_tool_project_ready = MiniCutWindow._require_tool_project_ready
            _require_tool_media_ready = MiniCutWindow._require_tool_media_ready
            tool_export_all = MiniCutWindow.tool_export_all

        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "movie.mp4"
            source.write_bytes(b"video")

            host = Host()
            host.model = ProjectModel()
            host.model.source = source
            host.model.duration_ms = 1_000
            host.analyze_worker = None
            host.film_cut_worker = None
            host.export_worker = _StoppedWorker()

            with self.assertRaisesRegex(RuntimeError, "memfinalisasi"):
                host.tool_export_all()


class MandatorySubtitleExportTests(unittest.TestCase):
    def test_fast_copy_does_not_require_srt_or_switch_mode(self):
        import tempfile
        from pathlib import Path

        class ModeStub:
            def __init__(self):
                self.value = "fast"

            def currentData(self):
                return self.value

            def findData(self, _value):
                return -1

            def setCurrentIndex(self, _index):
                self.value = "smartcut"

        class Host:
            _require_tool_project_ready = MiniCutWindow._require_tool_project_ready
            _require_tool_media_ready = MiniCutWindow._require_tool_media_ready
            tool_export_all = MiniCutWindow.tool_export_all

            def _choose_srt(self):
                self.srt_picker_called = True
                return False

            def _log(self, _text):
                pass

        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "movie.mp4"
            source.write_bytes(b"video")

            host = Host()
            host.model = ProjectModel()
            host.model.source = source
            host.model.duration_ms = 2_000
            host.model.keyframes = [0, 1_000, 2_000]
            host.model.cuts = [CutPoint(1_000, 1_000, None)]
            host.analyze_worker = None
            host.film_cut_worker = None
            host.export_worker = None
            host.srt_path = None
            host._srt_project_reference = None
            host._srt_auto_disabled = False
            host._srt_user_disabled = False
            host.srt_picker_called = False
            host.export_mode = ModeStub()

            with patch("minicut_agent.ui.find_tool", return_value="ffmpeg"), patch(
                "minicut_agent.ui.QFileDialog.getExistingDirectory",
                return_value="",
            ):
                result = host.tool_export_all()

            self.assertFalse(result["ok"])
            self.assertTrue(result["cancelled"])
            self.assertFalse(host.srt_picker_called)
            self.assertEqual(host.export_mode.value, "fast")

    def test_smartcut_requires_srt_before_output_folder_is_requested(self):
        import tempfile
        from pathlib import Path

        class ModeStub:
            def currentData(self):
                return "smartcut"

            def findData(self, _value):
                return -1

            def setCurrentIndex(self, _index):
                pass

        class Host:
            _require_tool_project_ready = MiniCutWindow._require_tool_project_ready
            _require_tool_media_ready = MiniCutWindow._require_tool_media_ready
            tool_export_all = MiniCutWindow.tool_export_all

            def _choose_srt(self):
                self.srt_picker_called = True
                return False

            def _log(self, _text):
                pass

        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "movie.mp4"
            source.write_bytes(b"video")

            host = Host()
            host.model = ProjectModel()
            host.model.source = source
            host.model.duration_ms = 1_000
            host.analyze_worker = None
            host.film_cut_worker = None
            host.export_worker = None
            host.srt_path = None
            host._srt_project_reference = None
            host._srt_auto_disabled = False
            host._srt_user_disabled = False
            host.srt_picker_called = False
            host.export_mode = ModeStub()

            def fake_find_tool(name):
                if name == "ffmpeg":
                    return "ffmpeg"
                if name in {"MiniCut SmartCut", "smartcut"}:
                    return "smartcut"
                return None

            with patch("minicut_agent.ui.find_tool", side_effect=fake_find_tool), patch(
                "minicut_agent.ui.QFileDialog.getExistingDirectory"
            ) as folder_picker:
                with self.assertRaisesRegex(RuntimeError, "SmartCut wajib menyertakan SRT"):
                    host.tool_export_all()

            self.assertTrue(host.srt_picker_called)
            folder_picker.assert_not_called()

    def test_agent_state_describes_mode_specific_srt_requirement(self):
        import tempfile
        from pathlib import Path

        class Host:
            _state_with_subtitle = MiniCutWindow._state_with_subtitle
            tool_get_state = MiniCutWindow.tool_get_state

        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "movie.mp4"
            source.write_bytes(b"video")

            host = Host()
            host.model = ProjectModel()
            host.model.source = source
            host.model.duration_ms = 1_000
            host.srt_path = None

            state = host.tool_get_state()["state"]["subtitle"]

        self.assertFalse(state["required_for_export"])
        self.assertTrue(state["required_for_current_export"])
        self.assertTrue(state["required_for_smartcut_export"])
        self.assertFalse(state["required_for_fast_export"])
        self.assertTrue(state["required_for_film_cut"])
        self.assertTrue(state["smartcut_outputs_srt"])
        self.assertFalse(state["fast_copy_outputs_srt"])
        self.assertFalse(state["loaded"])


class SubtitleProjectDirtyTests(unittest.TestCase):
    def test_clearing_subtitle_marks_loaded_project_dirty(self):
        class Host:
            _clear_srt = MiniCutWindow._clear_srt

            def _log(self, _text):
                pass

        host = Host()
        host.model = ProjectModel()
        host.model.source = __import__("pathlib").Path("movie.mp4")
        host.model.dirty = False
        host.srt_path = __import__("pathlib").Path("movie.srt")
        host._srt_auto_disabled = False
        host._srt_user_disabled = False
        host.film_cut_results = []

        host._clear_srt()

        self.assertTrue(host.model.dirty)
        self.assertTrue(host._srt_auto_disabled)
        self.assertTrue(host._srt_user_disabled)
        self.assertIsNone(host.srt_path)


class SubtitleMissingReferenceTests(unittest.TestCase):
    def test_saving_project_preserves_missing_subtitle_reference(self):
        import json
        import tempfile
        from pathlib import Path

        class StatusStub:
            def setText(self, _text):
                pass

        class Host:
            _require_tool_project_ready = MiniCutWindow._require_tool_project_ready
            tool_save_project = MiniCutWindow.tool_save_project

            def _refresh(self):
                pass

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "movie.mp4"
            source.write_bytes(b"video")
            project = root / "movie.minicut.json"
            missing_srt = root / "custom-missing.srt"

            host = Host()
            host.model = ProjectModel()
            host.model.source = source
            host.model.duration_ms = 10_000
            host.model.project_path = project
            host.analyze_worker = None
            host.srt_path = None
            host._srt_project_reference = missing_srt
            host._srt_user_disabled = False
            host.status = StatusStub()

            result = host.tool_save_project()

            self.assertTrue(result["ok"])
            data = json.loads(project.read_text(encoding="utf-8"))
            self.assertEqual(
                Path(data["subtitle_absolute"]),
                missing_srt.resolve(),
            )
            self.assertFalse(data["subtitle_auto_disabled"])

class FilmCutApplyResultTests(unittest.TestCase):
    def test_declining_replace_does_not_report_success(self):
        class Host:
            pass

        host = Host()
        host.film_cut_results = [{
            "selected_time_ms": 60_000,
            "confidence": 0.9,
            "needs_review": False,
        }]
        host.model = ProjectModel()
        host.model.duration_ms = 120_000
        host.model.cuts = [CutPoint(30_000, 30_000)]

        with patch(
            "minicut_agent.ui.QMessageBox.question",
            return_value=__import__("minicut_agent.ui", fromlist=["QMessageBox"]).QMessageBox.StandardButton.No,
        ):
            result = MiniCutWindow._apply_film_cut(host)

        self.assertFalse(result["ok"])
        self.assertTrue(result["cancelled"])
        self.assertEqual(
            [cut.actual_ms for cut in host.model.cuts],
            [30_000],
        )


class _ExportModeStub:
    def __init__(self):
        self.value = "fast"

    def findData(self, value):
        return 0 if value == "smartcut" else -1

    def setCurrentIndex(self, _index):
        self.value = "smartcut"


class GeminiExactCutToolTests(unittest.TestCase):
    def test_gemini_add_cut_preserves_exact_timestamp_and_forces_smartcut(self):
        import tempfile
        from pathlib import Path

        class Host:
            _require_tool_project_ready = MiniCutWindow._require_tool_project_ready
            _require_tool_media_ready = MiniCutWindow._require_tool_media_ready
            tool_add_cut = MiniCutWindow.tool_add_cut

            def _refresh(self):
                pass

        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "movie.mp4"
            source.write_bytes(b"video")

            host = Host()
            host.model = ProjectModel()
            host.model.source = source
            host.model.duration_ms = 120_000
            host.analyze_worker = None
            host.export_mode = _ExportModeStub()

            result = host.tool_add_cut(61_237)

        self.assertTrue(result["ok"])
        self.assertTrue(result["exact_timestamp_preserved"])
        self.assertEqual(host.model.cuts[0].requested_ms, 61_237)
        self.assertEqual(host.model.cuts[0].actual_ms, 61_237)
        self.assertEqual(host.export_mode.value, "smartcut")


class _BridgeDrainHost:
    _drain_bridge = MiniCutWindow._drain_bridge
    _snapshot = MiniCutWindow._snapshot
    _state_with_subtitle = MiniCutWindow._state_with_subtitle

    def __init__(self):
        self.model = ProjectModel()
        self.model.duration_ms = 120_000
        self.srt_path = None
        self.bridge_queue = queue.Queue()
        self.bridge_state = {}
        self.undo_stack = []
        self.registry = self

    def state(self):
        return self.model.state()

    def execute(self, tool, args):
        return {"ok": False, "cancelled": True}

    def _refresh(self):
        pass


class BridgeUndoTests(unittest.TestCase):
    def test_failed_bridge_mutation_does_not_create_undo_entry(self):
        host = _BridgeDrainHost()
        call = BridgeCall(tool="apply_film_cut", args={})
        host.bridge_queue.put(call)

        host._drain_bridge()

        self.assertTrue(call.event.is_set())
        self.assertFalse(call.result["ok"])
        self.assertEqual(host.undo_stack, [])
        self.assertTrue(host.bridge_state["subtitle"]["required_for_export"])
        self.assertFalse(host.bridge_state["subtitle"]["loaded"])


class _StoppedWorker:
    def isRunning(self):
        return False


class _RunningWorker:
    def isRunning(self):
        return True


class ToolMediaReadyTests(unittest.TestCase):
    def test_project_state_can_remain_usable_when_media_file_is_missing(self):
        class Host:
            _require_tool_project_ready = MiniCutWindow._require_tool_project_ready

        host = Host()
        host.model = ProjectModel()
        host.model.source = __import__("pathlib").Path(
            "missing-but-project-can-still-be-saved.mp4"
        )
        host.analyze_worker = None

        # Saving/undoing project state must remain possible.
        host._require_tool_project_ready()

    def test_media_tools_reject_while_new_media_is_loading(self):
        class Host:
            _require_tool_project_ready = MiniCutWindow._require_tool_project_ready
            _require_tool_media_ready = MiniCutWindow._require_tool_media_ready

        host = Host()
        host.model = ProjectModel()
        host.model.source = __import__("pathlib").Path("old.mp4")
        host.analyze_worker = _RunningWorker()

        with self.assertRaises(RuntimeError):
            host._require_tool_media_ready()

    def test_media_tools_reject_without_loaded_video(self):
        class Host:
            _require_tool_project_ready = MiniCutWindow._require_tool_project_ready
            _require_tool_media_ready = MiniCutWindow._require_tool_media_ready

        host = Host()
        host.model = ProjectModel()
        host.analyze_worker = None

        with self.assertRaises(ValueError):
            host._require_tool_media_ready()

    def test_media_tools_reject_when_source_file_disappeared(self):
        class Host:
            _require_tool_project_ready = MiniCutWindow._require_tool_project_ready
            _require_tool_media_ready = MiniCutWindow._require_tool_media_ready

        host = Host()
        host.model = ProjectModel()
        host.model.source = __import__("pathlib").Path(
            "definitely-missing-video-for-test.mp4"
        )
        host.analyze_worker = None

        with self.assertRaises(FileNotFoundError):
            host._require_tool_media_ready()


if __name__ == "__main__":
    unittest.main()
