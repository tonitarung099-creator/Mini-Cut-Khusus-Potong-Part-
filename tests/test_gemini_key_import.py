import tempfile
import unittest
from pathlib import Path

from minicut_agent.gemini_key_import import (
    add_unnamed_gemini_key,
    import_gemini_api_keys_text,
    parse_gemini_api_keys,
)
from minicut_agent.gemini_keys import GeminiKeyStore


KEY_1 = "AIza" + "A" * 35
KEY_2 = "AIza" + "B" * 35
KEY_3 = "AIza" + "C" * 35


class GeminiKeyImportTests(unittest.TestCase):
    def test_parse_txt_supports_plain_env_and_duplicates(self):
        text = (
            f"{KEY_1}\n"
            f"GEMINI_API_KEY={KEY_2}\n"
            f"{KEY_1}\n"
            "# komentar\n"
            "bukan-api\n"
        )

        keys, duplicates, invalid = parse_gemini_api_keys(text)

        self.assertEqual(keys, [KEY_1, KEY_2])
        self.assertEqual(duplicates, 1)
        self.assertEqual(invalid, 1)

    def test_import_skips_existing_and_duplicate_without_names(self):
        with tempfile.TemporaryDirectory() as td:
            store = GeminiKeyStore(Path(td) / "keys.json")
            first_id = add_unnamed_gemini_key(store, KEY_1)
            self.assertEqual(store.get_secret(first_id), KEY_1)
            self.assertEqual(store.summaries()[0].name, "API 001")

            result = import_gemini_api_keys_text(
                store,
                f"{KEY_1}\n{KEY_2}\n{KEY_2}\n{KEY_3}\n",
            )

            self.assertEqual(result.added, 2)
            self.assertEqual(result.duplicates, 2)
            self.assertEqual(result.invalid, 0)
            self.assertEqual(result.capacity_skipped, 0)
            self.assertEqual(store.count(), 3)
            self.assertEqual(
                [item.name for item in store.summaries()],
                ["API 001", "API 002", "API 003"],
            )

    def test_single_add_rejects_duplicate(self):
        with tempfile.TemporaryDirectory() as td:
            store = GeminiKeyStore(Path(td) / "keys.json")
            add_unnamed_gemini_key(store, KEY_1)

            with self.assertRaisesRegex(ValueError, "sudah tersimpan"):
                add_unnamed_gemini_key(store, KEY_1)


if __name__ == "__main__":
    unittest.main()
