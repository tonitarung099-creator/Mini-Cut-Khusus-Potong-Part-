import unittest

from minicut_agent.agent import AgentPlanner, ToolRegistry


class AgentLocalPlanIntentSafetyTests(unittest.TestCase):
    def setUp(self):
        self.planner = AgentPlanner(ToolRegistry(object()))

    def test_shortcut_with_timestamp_is_not_a_cut_command(self):
        with self.assertRaises(ValueError):
            self.planner.local_plan("shortcut 00:10:00")

    def test_cuti_with_timestamp_is_not_a_cut_command(self):
        with self.assertRaises(ValueError):
            self.planner.local_plan("cuti 00:10:00")

    def test_real_cut_word_still_creates_add_cut_step(self):
        plan = self.planner.local_plan("cut 00:10:00")
        self.assertEqual(plan["steps"], [
            {"tool": "add_cut", "args": {"time_ms": "00:10:00"}},
        ])


if __name__ == "__main__":
    unittest.main()
