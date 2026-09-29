from __future__ import annotations

import copy
import re
from dataclasses import dataclass

from .gemini_keys import GeminiKeyStore, MAX_GEMINI_KEYS


_GEMINI_KEY_RE = re.compile(r"AIza[0-9A-Za-z_-]{20,}")
_ENV_NAMES = {"GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY"}


@dataclass(frozen=True)
class GeminiKeyImportResult:
    added: int = 0
    duplicates: int = 0
    invalid: int = 0
    capacity_skipped: int = 0
    candidates: int = 0


def _clean_fallback_candidate(line: str) -> str:
    value = line.strip().lstrip("\ufeff")
    if not value or value.startswith("#"):
        return ""

    if "=" in value:
        left, right = value.split("=", 1)
        if left.strip().upper() in _ENV_NAMES:
            value = right.strip()
        else:
            return ""

    value = value.strip().strip("\"'").rstrip(",;").strip()
    if len(value) < 20 or any(ch.isspace() for ch in value):
        return ""
    return value


def parse_gemini_api_keys(text: str) -> tuple[list[str], int, int]:
    """Parse API keys from TXT content while preserving the original order.

    Preferred format is one key per line, but the parser also accepts common
    .env forms and can extract multiple normal Google API keys from one line.
    Empty/comment/prose lines are ignored. Duplicate keys in the same file are
    returned only once and counted separately.
    """
    keys: list[str] = []
    seen: set[str] = set()
    duplicates = 0
    invalid = 0

    for raw_line in str(text or "").splitlines():
        line = raw_line.strip().lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue

        matches = _GEMINI_KEY_RE.findall(line)
        candidates = matches if matches else [_clean_fallback_candidate(line)]
        accepted_on_line = False

        for candidate in candidates:
            key = candidate.strip()
            if not key:
                continue
            accepted_on_line = True
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            keys.append(key)

        if not accepted_on_line:
            invalid += 1

    return keys, duplicates, invalid


def _stored_secrets(store: GeminiKeyStore) -> set[str]:
    secrets: set[str] = set()
    for summary in store.summaries():
        try:
            secret = store.get_secret(summary.id).strip()
        except Exception:
            continue
        if secret:
            secrets.add(secret)
    return secrets


def _next_auto_name(store: GeminiKeyStore) -> str:
    used = {item.name for item in store.summaries()}
    index = 1
    while True:
        candidate = f"API {index:03d}"
        if candidate not in used:
            return candidate
        index += 1


def add_unnamed_gemini_key(store: GeminiKeyStore, api_key: str) -> str:
    """Add one API key without asking the user for a name/project."""
    key = str(api_key or "").strip()
    if not key:
        raise ValueError("API key tidak boleh kosong.")
    if key in _stored_secrets(store):
        raise ValueError("API key ini sudah tersimpan.")
    if store.count() >= MAX_GEMINI_KEYS:
        raise ValueError(f"Maksimal {MAX_GEMINI_KEYS} Gemini API key.")
    return store.add(_next_auto_name(store), "", key)


def import_gemini_api_keys_text(
    store: GeminiKeyStore,
    text: str,
) -> GeminiKeyImportResult:
    """Import many keys transactionally from TXT text.

    Existing keys and duplicates inside the TXT are skipped. If an unexpected
    storage/encryption error occurs, the key store is rolled back to its state
    before the import started.
    """
    candidates, duplicate_in_file, invalid = parse_gemini_api_keys(text)
    existing = _stored_secrets(store)
    before = copy.deepcopy(store.data)

    added = 0
    duplicates = duplicate_in_file
    capacity_skipped = 0

    try:
        for key in candidates:
            if key in existing:
                duplicates += 1
                continue
            if store.count() >= MAX_GEMINI_KEYS:
                capacity_skipped += 1
                continue
            store.add(_next_auto_name(store), "", key)
            existing.add(key)
            added += 1
    except Exception:
        store.data = before
        store.save()
        raise

    return GeminiKeyImportResult(
        added=added,
        duplicates=duplicates,
        invalid=invalid,
        capacity_skipped=capacity_skipped,
        candidates=len(candidates),
    )
