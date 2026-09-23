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
            self.assertEqual(snap["rpd_remaining"], 493)
            self.assertEqual(snap["rpm_remaining"], 8)
            self.assertEqual(snap["tpm_remaining"], 250000 - 23644)
            self.assertGreater(snap["rpd_reset_seconds"], 0)
            self.assertLessEqual(snap["rpd_reset_seconds"], 25 * 60 * 60)
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

    def test_resource_exhausted_is_recorded_as_limit(self):
        with tempfile.TemporaryDirectory() as td:
            store = GeminiKeyStore(Path(td) / "keys.json")
            key_id = store.add("Key", "Project", "AIza-resource-1234567890")
            store.mark_error(
                key_id,
                "gemini-3.5-flash-lite",
                "RESOURCE_EXHAUSTED: request limit reached",
            )
            self.assertEqual(
                store.snapshot(key_id, "gemini-3.5-flash-lite")["status"],
                "limited",
            )

    def test_known_model_limits(self):
        self.assertEqual(model_limits("gemini-3.5-flash-lite"), (15, 250000, 500))
        self.assertEqual(model_limits("gemini-2.5-flash-lite"), (10, 250000, 20))


    def test_failover_prefers_ready_key_and_skips_limited_or_excluded(self):
        with tempfile.TemporaryDirectory() as td:
            store = GeminiKeyStore(Path(td) / "keys.json")
            first = store.add("First", "P1", "AIza-first-1234567890")
            ready = store.add("Ready", "P2", "AIza-ready-1234567890")
            limited = store.add("Limited", "P3", "AIza-limit-1234567890")

            store.record_usage(
                ready,
                "gemini-3.5-flash-lite",
                status="ready",
                checked=True,
            )
            store.mark_error(
                limited,
                "gemini-3.5-flash-lite",
                "Gemini API 429: quota exhausted",
            )

            ids = store.usable_key_ids(
                "gemini-3.5-flash-lite",
                exclude_ids={first},
            )
            self.assertEqual(ids[0], ready)
            self.assertNotIn(first, ids)
            self.assertNotIn(limited, ids)

    def test_failover_skips_locally_exhausted_daily_key(self):
        with tempfile.TemporaryDirectory() as td:
            store = GeminiKeyStore(Path(td) / "keys.json")
            exhausted = store.add("Full", "P1", "AIza-full-1234567890")
            spare = store.add("Spare", "P2", "AIza-spare-1234567890")

            store.record_usage(
                exhausted,
                "gemini-2.5-flash-lite",
                requests=20,
                status="ready",
            )
            store.record_usage(
                spare,
                "gemini-2.5-flash-lite",
                status="ready",
            )

            ids = store.usable_key_ids("gemini-2.5-flash-lite")
            self.assertNotIn(exhausted, ids)
            self.assertIn(spare, ids)

    def test_corrupt_store_is_backed_up_before_reset(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "keys.json"
            broken = b"{not-valid-json"
            path.write_bytes(broken)

            store = GeminiKeyStore(path)

            self.assertEqual(store.count(), 0)
            backups = list(Path(td).glob("keys.corrupt-*.json"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), broken)


if __name__ == "__main__":
    unittest.main()
