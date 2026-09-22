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


class _UndoHost:
    _snapshot = MiniCutWindow._snapshot
    _restore_snapshot = MiniCutWindow._restore_snapshot
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


class _BridgeDrainHost:
    _drain_bridge = MiniCutWindow._drain_bridge
    _snapshot = MiniCutWindow._snapshot

    def __init__(self):
        self.model = ProjectModel()
        self.model.duration_ms = 120_000
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


class _RunningWorker:
    def isRunning(self):
        return True


class ToolMediaReadyTests(unittest.TestCase):
    def test_media_tools_reject_while_new_media_is_loading(self):
        class Host:
            _require_tool_media_ready = MiniCutWindow._require_tool_media_ready

        host = Host()
        host.model = ProjectModel()
        host.model.source = __import__("pathlib").Path("old.mp4")
        host.analyze_worker = _RunningWorker()

        with self.assertRaises(RuntimeError):
            host._require_tool_media_ready()

    def test_media_tools_reject_without_loaded_video(self):
        class Host:
            _require_tool_media_ready = MiniCutWindow._require_tool_media_ready

        host = Host()
        host.model = ProjectModel()
        host.analyze_worker = None

        with self.assertRaises(ValueError):
            host._require_tool_media_ready()

    def test_media_tools_reject_when_source_file_disappeared(self):
        class Host:
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
