import queue
import unittest

from minicut_agent.bridge import BridgeCall
from minicut_agent.hardened_window import HardenedMiniCutWindow


class _Registry:
    def __init__(self, call=None, cancel_during_execute=False):
        self.call = call
        self.cancel_during_execute = cancel_during_execute
        self.executed = 0

    def execute(self, tool, args):
        self.executed += 1
        if self.cancel_during_execute and self.call is not None:
            self.call.cancelled.set()
        return {"ok": True, "tool": tool}


class _DrainHost:
    _drain_bridge = HardenedMiniCutWindow._drain_bridge

    def __init__(self, call, *, cancel_during_execute=False):
        self.bridge_queue = queue.Queue()
        self.bridge_queue.put(call)
        self.bridge_state = {}
        self.registry = _Registry(call, cancel_during_execute)
        self.undo_stack = []
        self.restored = []
        self.refresh_count = 0
        self.export_worker = None

    def _state_with_subtitle(self):
        return {"state": "ok"}

    def _snapshot(self):
        return {"cuts": [1000]}

    def _restore_snapshot(self, snapshot):
        self.restored.append(snapshot)

    def _refresh(self):
        self.refresh_count += 1

    def tool_cancel_export(self):
        return {"ok": True}


class BridgeTimeoutSafetyTests(unittest.TestCase):
    def test_cancelled_queued_call_is_never_executed(self):
        call = BridgeCall(tool="add_cut", args={"time_ms": 1000})
        call.cancelled.set()
        call.result.update({"ok": False, "cancelled": True})
        host = _DrainHost(call)

        host._drain_bridge()

        self.assertEqual(host.registry.executed, 0)
        self.assertTrue(call.event.is_set())
        self.assertFalse(call.result["ok"])
        self.assertTrue(call.result["cancelled"])

    def test_timeout_during_mutation_restores_snapshot_and_skips_undo(self):
        call = BridgeCall(tool="clear_cuts", args={})
        host = _DrainHost(call, cancel_during_execute=True)

        host._drain_bridge()

        self.assertEqual(host.registry.executed, 1)
        self.assertEqual(host.restored, [{"cuts": [1000]}])
        self.assertEqual(host.undo_stack, [])
        self.assertFalse(call.result["ok"])
        self.assertTrue(call.result["cancelled"])


if __name__ == "__main__":
    unittest.main()
