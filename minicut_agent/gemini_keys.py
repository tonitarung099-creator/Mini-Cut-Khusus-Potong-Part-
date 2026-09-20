from __future__ import annotations

import base64
import ctypes
import json
import os
import time
import uuid
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

MAX_GEMINI_KEYS = 100
APP_DIR_NAME = "MiniCut Studio Agent"
PACIFIC = ZoneInfo("America/Los_Angeles")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _pacific_day() -> str:
    return datetime.now(PACIFIC).date().isoformat()


class _DATA_BLOB(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_byte)),
    ]


def _protect_windows(data: bytes) -> bytes:
    if os.name != "nt":
        return b"plain:" + data
    in_buffer = ctypes.create_string_buffer(data)
    in_blob = _DATA_BLOB(
        len(data),
        ctypes.cast(in_buffer, ctypes.POINTER(ctypes.c_byte)),
    )
    out_blob = _DATA_BLOB()
    description = ctypes.c_wchar_p("MiniCut Gemini API Key")
    ok = ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(in_blob),
        description,
        None,
        None,
        None,
        0,
        ctypes.byref(out_blob),
    )
    if not ok:
        raise ctypes.WinError()
    try:
        return ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(out_blob.pbData)


def _unprotect_windows(data: bytes) -> bytes:
    if data.startswith(b"plain:"):
        return data[6:]
    if os.name != "nt":
        raise RuntimeError("API key ini dibuat di Windows dan tidak dapat dibuka di sistem ini.")
    in_buffer = ctypes.create_string_buffer(data)
    in_blob = _DATA_BLOB(
        len(data),
        ctypes.cast(in_buffer, ctypes.POINTER(ctypes.c_byte)),
    )
    out_blob = _DATA_BLOB()
    description = ctypes.c_wchar_p()
    ok = ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(in_blob),
        ctypes.byref(description),
        None,
        None,
        None,
        0,
        ctypes.byref(out_blob),
    )
    if not ok:
        raise ctypes.WinError()
    try:
        return ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(out_blob.pbData)
        if description.value:
            ctypes.windll.kernel32.LocalFree(ctypes.cast(description, ctypes.c_void_p))


def _encrypt_text(text: str) -> str:
    raw = _protect_windows(text.encode("utf-8"))
    return base64.b64encode(raw).decode("ascii")


def _decrypt_text(text: str) -> str:
    raw = base64.b64decode(text.encode("ascii"))
    return _unprotect_windows(raw).decode("utf-8")


def _default_store_path() -> Path:
    base = os.environ.get("APPDATA")
    if base:
        root = Path(base)
    else:
        root = Path.home() / ".minicut"
    folder = root / APP_DIR_NAME
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "gemini_keys.json"


def model_limits(model: str) -> tuple[int, int, int]:
    """Known defaults from the user's current free-tier screen.

    These values are only used to calculate MiniCut-side percentages and
    can differ if Google changes a project's quota.
    """
    name = (model or "").strip().lower()
    if name in {"gemini-3.5-flash-lite", "gemini-3.1-flash-lite"}:
        return 15, 250_000, 500
    if name == "gemini-2.5-flash-lite":
        return 10, 250_000, 20
    return 5, 250_000, 20


@dataclass
class GeminiKeySummary:
    id: str
    name: str
    project: str
    masked_key: str
    is_active: bool


class GeminiKeyStore:
    def __init__(self, path: Path | None = None):
        self.path = path or _default_store_path()
        self.data: dict[str, Any] = {"version": 1, "active_id": None, "keys": []}
        self.load()

    def load(self):
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("keys"), list):
                self.data = raw
        except Exception:
            self.data = {"version": 1, "active_id": None, "keys": []}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def count(self) -> int:
        return len(self.data.get("keys") or [])

    def summaries(self) -> list[GeminiKeySummary]:
        active = self.data.get("active_id")
        result: list[GeminiKeySummary] = []
        for rec in self.data.get("keys") or []:
            masked = str(rec.get("masked") or "")
            if not masked:
                try:
                    key = self.get_secret(rec["id"])
                    masked = self.mask_key(key)
                except Exception:
                    masked = "(tidak dapat dibuka)"
            result.append(GeminiKeySummary(
                id=str(rec.get("id") or ""),
                name=str(rec.get("name") or "Gemini API"),
                project=str(rec.get("project") or ""),
                masked_key=masked,
                is_active=rec.get("id") == active,
            ))
        return result

    @staticmethod
    def mask_key(api_key: str) -> str:
        key = api_key.strip()
        if len(key) <= 8:
            return "•" * len(key)
        return key[:4] + "••••••••" + key[-4:]

    def _record(self, key_id: str) -> dict[str, Any]:
        for rec in self.data.get("keys") or []:
            if rec.get("id") == key_id:
                return rec
        raise KeyError("API key tidak ditemukan.")

    def add(self, name: str, project: str, api_key: str) -> str:
        if self.count() >= MAX_GEMINI_KEYS:
            raise ValueError(f"Maksimal {MAX_GEMINI_KEYS} Gemini API key.")
        api_key = api_key.strip()
        if not api_key:
            raise ValueError("API key tidak boleh kosong.")
        key_id = uuid.uuid4().hex
        rec = {
            "id": key_id,
            "name": (name.strip() or f"Gemini API {self.count() + 1}")[:80],
            "project": project.strip()[:120],
            "secret": _encrypt_text(api_key),
            "masked": self.mask_key(api_key),
            "created_at": _now_iso(),
            "usage": {},
        }
        self.data.setdefault("keys", []).append(rec)
        if not self.data.get("active_id"):
            self.data["active_id"] = key_id
        self.save()
        return key_id

    def update(self, key_id: str, name: str, project: str, api_key: str | None = None):
        rec = self._record(key_id)
        rec["name"] = (name.strip() or rec.get("name") or "Gemini API")[:80]
        rec["project"] = project.strip()[:120]
        if api_key is not None and api_key.strip():
            clean_key = api_key.strip()
            rec["secret"] = _encrypt_text(clean_key)
            rec["masked"] = self.mask_key(clean_key)
        self.save()

    def remove(self, key_id: str):
        items = self.data.get("keys") or []
        self.data["keys"] = [x for x in items if x.get("id") != key_id]
        if self.data.get("active_id") == key_id:
            self.data["active_id"] = self.data["keys"][0]["id"] if self.data["keys"] else None
        self.save()

    def set_active(self, key_id: str):
        self._record(key_id)
        self.data["active_id"] = key_id
        self.save()

    def active_id(self) -> str | None:
        value = self.data.get("active_id")
        return str(value) if value else None

    def active_summary(self) -> GeminiKeySummary | None:
        active = self.active_id()
        if not active:
            return None
        return next((x for x in self.summaries() if x.id == active), None)

    def get_secret(self, key_id: str) -> str:
        rec = self._record(key_id)
        return _decrypt_text(str(rec["secret"]))

    def active_secret(self) -> str:
        key_id = self.active_id()
        if not key_id:
            raise ValueError("Belum ada Gemini API key aktif.")
        return self.get_secret(key_id)

    def _usage(self, key_id: str, model: str) -> dict[str, Any]:
        rec = self._record(key_id)
        usage = rec.setdefault("usage", {})
        model_usage = usage.setdefault(model, {
            "day": _pacific_day(),
            "daily_requests": 0,
            "events": [],
            "status": "unknown",
            "last_error": "",
            "last_checked": "",
        })
        if model_usage.get("day") != _pacific_day():
            model_usage["day"] = _pacific_day()
            model_usage["daily_requests"] = 0
            model_usage["events"] = []
            if model_usage.get("status") == "limited":
                model_usage["status"] = "unknown"
                model_usage["last_error"] = ""
        now = time.time()
        model_usage["events"] = [
            e for e in model_usage.get("events") or []
            if now - float(e.get("ts") or 0) <= 120
        ]
        return model_usage

    def record_usage(
        self,
        key_id: str,
        model: str,
        requests: int = 0,
        prompt_tokens: int = 0,
        status: str | None = None,
        error: str = "",
        checked: bool = False,
    ):
        u = self._usage(key_id, model)
        req = max(0, int(requests))
        tokens = max(0, int(prompt_tokens))
        if req or tokens:
            u["events"].append({"ts": time.time(), "requests": req, "prompt_tokens": tokens})
            u["daily_requests"] = int(u.get("daily_requests") or 0) + req
        if status:
            u["status"] = status
        if error:
            u["last_error"] = error[:1200]
        elif status == "ready":
            u["last_error"] = ""
        if checked:
            u["last_checked"] = _now_iso()
        self.save()

    def mark_error(self, key_id: str, model: str, message: str):
        limited = "429" in message or "rate limit" in message.lower() or "quota" in message.lower()
        self.record_usage(
            key_id,
            model,
            status="limited" if limited else "error",
            error=message,
            checked=True,
        )

    def snapshot(self, key_id: str, model: str) -> dict[str, Any]:
        u = self._usage(key_id, model)
        now = time.time()
        minute = [
            e for e in u.get("events") or []
            if now - float(e.get("ts") or 0) <= 60
        ]
        rpm_used = sum(int(e.get("requests") or 0) for e in minute)
        tpm_used = sum(int(e.get("prompt_tokens") or 0) for e in minute)
        rpd_used = int(u.get("daily_requests") or 0)
        rpm_limit, tpm_limit, rpd_limit = model_limits(model)

        def pct(used: int, limit: int) -> float:
            return round((used / max(1, limit)) * 100, 1)

        return {
            "status": str(u.get("status") or "unknown"),
            "last_error": str(u.get("last_error") or ""),
            "last_checked": str(u.get("last_checked") or ""),
            "rpm_used": rpm_used,
            "rpm_limit": rpm_limit,
            "rpm_pct": pct(rpm_used, rpm_limit),
            "tpm_used": tpm_used,
            "tpm_limit": tpm_limit,
            "tpm_pct": pct(tpm_used, tpm_limit),
            "rpd_used": rpd_used,
            "rpd_limit": rpd_limit,
            "rpd_pct": pct(rpd_used, rpd_limit),
            "day": u.get("day"),
        }
