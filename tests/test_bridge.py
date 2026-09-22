import queue
import unittest

from minicut_agent.bridge import BridgeCall, LocalBridge


class BridgeShutdownTests(unittest.TestCase):
    def test_stop_releases_pending_bridge_calls(self):
        calls = queue.Queue()
        call = BridgeCall(tool="seek", args={"time_ms": 1000})
        calls.put(call)

        bridge = LocalBridge(
            calls,
            state_provider=lambda: {},
            manifest_provider=lambda: {},
            port=0,
        )
        bridge.stop()

        self.assertTrue(call.event.is_set())
        self.assertFalse(call.result["ok"])
        self.assertIn("ditutup", call.result["error"])
        self.assertTrue(bridge.stopping)


if __name__ == "__main__":
    unittest.main()
