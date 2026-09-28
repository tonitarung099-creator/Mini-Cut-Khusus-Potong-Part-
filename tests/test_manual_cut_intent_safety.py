import unittest

from minicut_agent.manual_commands import looks_like_manual_cut


class ManualCutIntentSafetyTests(unittest.TestCase):
    def test_shortcut_does_not_trigger_cut_intent(self):
        self.assertFalse(looks_like_manual_cut("shortcut 00:10:00"))

    def test_cuti_does_not_trigger_cut_intent(self):
        self.assertFalse(looks_like_manual_cut("cuti jam 10:00"))

    def test_real_cut_words_still_trigger(self):
        self.assertTrue(looks_like_manual_cut("cut 00:10:00"))
        self.assertTrue(looks_like_manual_cut("potong di 00:10:00"))
        self.assertTrue(looks_like_manual_cut("Titik Potong 1 : 00:10:00"))


if __name__ == "__main__":
    unittest.main()
