import unittest

from minicut_agent.agent import AgentPlanner, MUTATING_TOOLS, ToolRegistry


class ExportModeCoreTransactionTests(unittest.TestCase):
    def test_export_mode_is_core_mutation(self):
        self.assertIn("set_export_mode", MUTATING_TOOLS)

    def test_failed_followup_rolls_export_mode_back(self):
        class Host:
            def __init__(self):
                self.mode = "smartcut"

            def tool_transaction_snapshot(self):
                return {"mode": self.mode}

            def tool_transaction_restore(self, snapshot):
                self.mode = snapshot["mode"]

            def tool_set_export_mode(self, mode):
                self.mode = mode
                return {"ok": True, "mode": mode}

            def tool_remove_cut(self, index):
                raise RuntimeError("simulasi langkah berikutnya gagal")

        host = Host()
        planner = AgentPlanner(ToolRegistry(host))

        with self.assertRaises(RuntimeError):
            planner.apply({
                "steps": [
                    {"tool": "set_export_mode", "args": {"mode": "fast"}},
                    {"tool": "remove_cut", "args": {"index": 0}},
                ]
            })

        self.assertEqual(host.mode, "smartcut")


if __name__ == "__main__":
    unittest.main()
