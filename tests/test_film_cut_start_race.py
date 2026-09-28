import unittest

from minicut_agent.camera_cut_window import CameraAwareMiniCutWindow


class _ControlStub:
    def __init__(self):
        self.enabled = True
        self.text = ""

    def setEnabled(self, value):
        self.enabled = bool(value)

    def setText(self, value):
        self.text = str(value)


class _ScheduledWorker:
    def __init__(self):
        self.cancelled = False

    def isRunning(self):
        return False

    def cancel(self):
        self.cancelled = True


class FilmCutStartRaceTests(unittest.TestCase):
    def test_scheduled_worker_is_reported_started_before_is_running_flips_true(self):
        class Host:
            tool_start_film_cut = CameraAwareMiniCutWindow.tool_start_film_cut

            def __init__(self):
                self.film_cut_worker = None

            def _require_tool_media_ready(self):
                pass

            def _start_film_cut(self):
                # Simulate QThread.start() having accepted the launch while the
                # worker has not yet been scheduled onto its OS thread.
                self.film_cut_worker = _ScheduledWorker()

        host = Host()
        result = host.tool_start_film_cut()

        self.assertTrue(result["ok"])
        self.assertTrue(result["started"])
        self.assertFalse(result["running"])
        self.assertTrue(result["stop_plan"])

    def test_film_cut_can_be_cancelled_before_is_running_flips_true(self):
        class Host:
            tool_cancel_film_cut = CameraAwareMiniCutWindow.tool_cancel_film_cut

        host = Host()
        host.film_cut_worker = _ScheduledWorker()
        host.film_cancel_btn = _ControlStub()
        host.film_status_label = _ControlStub()

        result = host.tool_cancel_film_cut()

        self.assertTrue(result["ok"])
        self.assertTrue(result["cancel_requested"])
        self.assertTrue(host.film_cut_worker.cancelled)
        self.assertFalse(host.film_cancel_btn.enabled)

    def test_export_can_be_cancelled_before_is_running_flips_true(self):
        class Host:
            tool_cancel_export = CameraAwareMiniCutWindow.tool_cancel_export

        host = Host()
        host.export_worker = _ScheduledWorker()
        host.status = _ControlStub()

        result = host.tool_cancel_export()

        self.assertTrue(result["ok"])
        self.assertTrue(result["cancel_requested"])
        self.assertTrue(host.export_worker.cancelled)
        self.assertIn("Membatalkan ekspor", host.status.text)


if __name__ == "__main__":
    unittest.main()
