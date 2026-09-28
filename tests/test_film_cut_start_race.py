import unittest

from minicut_agent.camera_cut_window import CameraAwareMiniCutWindow


class FilmCutStartRaceTests(unittest.TestCase):
    def test_scheduled_worker_is_reported_started_before_is_running_flips_true(self):
        class Worker:
            def isRunning(self):
                return False

        class Host:
            tool_start_film_cut = CameraAwareMiniCutWindow.tool_start_film_cut

            def __init__(self):
                self.film_cut_worker = None

            def _require_tool_media_ready(self):
                pass

            def _start_film_cut(self):
                # Simulate QThread.start() having accepted the launch while the
                # worker has not yet been scheduled onto its OS thread.
                self.film_cut_worker = Worker()

        host = Host()
        result = host.tool_start_film_cut()

        self.assertTrue(result["ok"])
        self.assertTrue(result["started"])
        self.assertFalse(result["running"])
        self.assertTrue(result["stop_plan"])


if __name__ == "__main__":
    unittest.main()
