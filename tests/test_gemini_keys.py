import tempfile
import unittest
from pathlib import Path

from minicut_agent.gemini_keys import GeminiKeyStore, MAX_GEMINI_KEYS, model_limits


class GeminiKeyStoreTests(unittest.TestCase):
    def test_roundtrip_and_active_key(self):
        with tempfile.TemporaryDirectory() as td:
            store = GeminiKeyStore(Path(td) / "keys.json")
            key_id = store.add("Utama", "Project-01", "AIza-test-key-1234567890")
            self.assertEqual(store.count(), 1)
            self.assertEqual(store.active_id(), key_id)
            self.assertEqual(store.get_secret(key_id), "AIza-test-key-1234567890")

            store.record_usage(
                key_id,
                "gemini-3.5-flash-lite",
                requests=7,
                prompt_tokens=23644,
                status="ready",
                checked=True,
            )
            snap = store.snapshot(key_id, "gemini-3.5-flash-lite")
            self.assertEqual(snap["rpd_used"], 7)
            self.assertEqual(snap["rpd_limit"], 500)
            self.assertGreater(snap["rpd_pct"], 1.0)
            self.assertEqual(snap["status"], "ready")

            reloaded = GeminiKeyStore(Path(td) / "keys.json")
            self.assertEqual(reloaded.get_secret(key_id), "AIza-test-key-1234567890")
            self.assertEqual(reloaded.snapshot(key_id, "gemini-3.5-flash-lite")["rpd_used"], 7)

    def test_limit_is_100_keys(self):
        with tempfile.TemporaryDirectory() as td:
            store = GeminiKeyStore(Path(td) / "keys.json")
            store.data["keys"] = [{"id": str(i)} for i in range(MAX_GEMINI_KEYS)]
            with self.assertRaises(ValueError):
                store.add("Overflow", "Project", "AIza-overflow")

    def test_known_model_limits(self):
        self.assertEqual(model_limits("gemini-3.5-flash-lite"), (15, 250000, 500))
        self.assertEqual(model_limits("gemini-2.5-flash-lite"), (10, 250000, 20))


if __name__ == "__main__":
    unittest.main()
