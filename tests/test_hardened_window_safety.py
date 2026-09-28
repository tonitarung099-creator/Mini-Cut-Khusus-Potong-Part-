import unittest

from minicut_agent.hardened_window import HardenedMiniCutWindow


class _Worker:
    def isRunning(self):
        # Reproduce the scheduler window immediately after QThread.start().
        return False


class _Host:
    _WORKER_LABELS = HardenedMiniCutWindow._WORKER_LABELS
    _present_worker_labels = HardenedMiniCutWindow._present_worker_labels
    _project_switch_blockers = HardenedMiniCutWindow._project_switch_blockers
    _timeline_mutation_blockers = HardenedMiniCutWindow._timeline_mutation_blockers
    _gemini_key_mutation_blockers = HardenedMiniCutWindow._gemini_key_mutation_blockers
    _manual_cut_workers_busy = HardenedMiniCutWindow._manual_cut_workers_busy
    _fast_copy_safety_error = HardenedMiniCutWindow._fast_copy_safety_error

    def __init__(self):
        for _label, attr in self._WORKER_LABELS:
            setattr(self, attr, None)


class _Mode:
    def __init__(self, value):
        self.value = value

    def currentData(self):
        return self.value


class _Model:
    def __init__(self, *, cuts=True, non_keyframe=True):
        self.cuts = [object()] if cuts else []
        self._non_keyframe = non_keyframe

    def has_non_keyframe_cuts(self):
        return self._non_keyframe


class HardenedWindowSafetyTests(unittest.TestCase):
    def test_scheduled_chat_worker_is_busy_even_before_is_running_turns_true(self):
        host = _Host()
        host.gemini_chat_worker = _Worker()

        self.assertTrue(host._manual_cut_workers_busy())
        self.assertIn("Gemini Chat", host._project_switch_blockers())
        self.assertIn("Gemini Chat", host._timeline_mutation_blockers())

    def test_scheduled_export_worker_locks_project_switch_and_timeline(self):
        host = _Host()
        host.export_worker = _Worker()

        self.assertIn("Ekspor", host._project_switch_blockers())
        self.assertIn("Ekspor", host._timeline_mutation_blockers())

    def test_gemini_key_edit_remove_are_blocked_until_worker_callback_finishes(self):
        for attr, label in (
            ("gemini_test_worker", "Tes Gemini"),
            ("gemini_batch_worker", "Cek Semua API"),
            ("gemini_chat_worker", "Gemini Chat"),
            ("film_cut_worker", "AI Film Cut"),
        ):
            host = _Host()
            setattr(host, attr, _Worker())
            self.assertIn(label, host._gemini_key_mutation_blockers())

    def test_fast_copy_rejects_frame_accurate_non_keyframe_timeline(self):
        host = _Host()
        host.export_mode = _Mode("fast")
        host.model = _Model(cuts=True, non_keyframe=True)

        message = host._fast_copy_safety_error()
        self.assertIsNotNone(message)
        self.assertIn("SmartCut", message)
        self.assertIn("non-keyframe", message)

    def test_fast_copy_allows_keyframe_only_timeline(self):
        host = _Host()
        host.export_mode = _Mode("fast")
        host.model = _Model(cuts=True, non_keyframe=False)

        self.assertIsNone(host._fast_copy_safety_error())

    def test_smartcut_is_not_blocked_by_fast_copy_guard(self):
        host = _Host()
        host.export_mode = _Mode("smartcut")
        host.model = _Model(cuts=True, non_keyframe=True)

        self.assertIsNone(host._fast_copy_safety_error())


if __name__ == "__main__":
    unittest.main()
