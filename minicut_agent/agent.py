from __future__ import annotations

import json
import re
import urllib.request
from typing import Any

from .core import parse_time_ms

MUTATING_TOOLS = {"add_cut", "remove_cut", "clear_cuts", "divide_equal", "divide_interval"}

class ToolRegistry:
    def __init__(self, host):
        self.host = host

    def manifest(self) -> dict[str, Any]:
        return {
            "version": 1,
            "tools": [
                {"name": "get_state", "args": {}, "description": "Baca status proyek dan timeline."},
                {"name": "seek", "args": {"time_ms": "int|string"}, "description": "Pindahkan playhead."},
                {"name": "play", "args": {}, "description": "Putar video."},
                {"name": "pause", "args": {}, "description": "Jeda video."},
                {"name": "add_cut", "args": {"time_ms": "int|string"}, "description": "Tambah batas part."},
                {"name": "remove_cut", "args": {"index": "int"}, "description": "Hapus cut berdasarkan index 0-based."},
                {"name": "clear_cuts", "args": {}, "description": "Hapus semua cut."},
                {"name": "divide_equal", "args": {"parts": "int"}, "description": "Bagi film menjadi N part sama panjang."},
                {"name": "divide_interval", "args": {"interval_ms": "int|string"}, "description": "Buat cut berkala."},
                {"name": "save_project", "args": {}, "description": "Simpan proyek ke path aktif atau minta lokasi."},
                {"name": "export_all", "args": {}, "description": "Ekspor semua part."},
                {"name": "undo", "args": {}, "description": "Batalkan perubahan agent terakhir."},
            ],
        }

    def execute(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        args = dict(args or {})
        fn = getattr(self.host, f"tool_{name}", None)
        if fn is None:
            raise ValueError(f"Tool tidak dikenal: {name}")
        result = fn(**args)
        if isinstance(result, dict):
            return result
        return {"ok": True, "result": result}

class AgentPlanner:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def validate(self, plan: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(plan, dict):
            raise ValueError("Rencana harus berupa objek JSON.")
        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("Rencana harus mempunyai steps.")
        allowed = {x["name"] for x in self.registry.manifest()["tools"]} - {"get_state"}
        normalized: list[dict[str, Any]] = []
        for i, step in enumerate(steps, 1):
            if not isinstance(step, dict):
                raise ValueError(f"Step {i} bukan objek.")
            tool = str(step.get("tool") or "")
            if tool not in allowed:
                raise ValueError(f"Step {i}: tool '{tool}' tidak diizinkan.")
            normalized.append({"tool": tool, "args": dict(step.get("args") or {})})
        return {
            "version": 1,
            "summary": str(plan.get("summary") or "Rencana edit"),
            "steps": normalized,
        }

    def local_plan(self, text: str) -> dict[str, Any]:
        raw = text.strip()
        if not raw:
            raise ValueError("Perintah agent masih kosong.")
        if raw.startswith("{"):
            return self.validate(json.loads(raw))
        low = raw.lower()

        m = re.search(r"(?:bagi|pecah).*?(\d+)\s*part", low)
        if m:
            parts = int(m.group(1))
            return self.validate({
                "summary": f"Bagi film menjadi {parts} part",
                "steps": [{"tool": "divide_equal", "args": {"parts": parts}}],
            })

        m = re.search(r"(?:setiap|tiap)\s+(\d+(?:[.,]\d+)?)\s*(detik|menit|jam)", low)
        if m:
            value = float(m.group(1).replace(",", "."))
            factor = {"detik": 1000, "menit": 60000, "jam": 3600000}[m.group(2)]
            return self.validate({
                "summary": f"Bagi film tiap {value:g} {m.group(2)}",
                "steps": [{"tool": "divide_interval", "args": {"interval_ms": int(value * factor)}}],
            })

        if "hapus semua cut" in low or "bersihkan cut" in low:
            return self.validate({"summary": "Hapus semua cut", "steps": [{"tool": "clear_cuts", "args": {}}]})

        if low in {"undo", "urungkan", "batalkan terakhir"}:
            return self.validate({"summary": "Undo perubahan agent", "steps": [{"tool": "undo", "args": {}}]})

        if "simpan" in low and ("proyek" in low or "project" in low):
            return self.validate({"summary": "Simpan proyek", "steps": [{"tool": "save_project", "args": {}}]})

        if "ekspor" in low or "export" in low:
            return self.validate({"summary": "Ekspor semua part", "steps": [{"tool": "export_all", "args": {}}]})

        times = re.findall(r"\b\d{1,2}:\d{2}(?::\d{2}(?:[.,]\d{1,3})?)?\b", raw)
        if ("cut" in low or "potong" in low) and times:
            return self.validate({
                "summary": "Tambahkan cut pada timestamp yang diminta",
                "steps": [{"tool": "add_cut", "args": {"time_ms": t}} for t in times],
            })

        if ("seek" in low or "pergi" in low or "playhead" in low) and times:
            return self.validate({
                "summary": "Pindahkan playhead",
                "steps": [{"tool": "seek", "args": {"time_ms": times[0]}}],
            })

        if low in {"play", "putar", "lanjutkan video"}:
            return self.validate({"summary": "Putar video", "steps": [{"tool": "play", "args": {}}]})
        if low in {"pause", "jeda", "berhenti sebentar"}:
            return self.validate({"summary": "Jeda video", "steps": [{"tool": "pause", "args": {}}]})

        raise ValueError(
            "Perintah lokal belum dikenali. Contoh: 'bagi jadi 8 part', "
            "'tiap 10 menit', 'tambah cut di 00:10:00', atau 'hapus semua cut'."
        )

    def system_prompt(self, state: dict[str, Any]) -> str:
        return (
            "Kamu adalah planner MiniCut Studio. Jangan mengedit langsung. "
            "Keluarkan HANYA satu objek JSON valid tanpa markdown berbentuk "
            '{"version":1,"summary":"...","steps":[{"tool":"...","args":{...}}]}. '
            "Gunakan hanya tool yang tersedia. Jangan mengarang tool. "
            "Jangan menyimpan atau mengekspor kecuali pengguna memintanya eksplisit.\n\n"
            "TOOL MANIFEST:\n" + json.dumps(self.registry.manifest(), ensure_ascii=False) +
            "\n\nCURRENT STATE:\n" + json.dumps(state, ensure_ascii=False)
        )

    def remote_plan(self, endpoint: str, model: str, api_key: str, user_text: str, state: dict[str, Any]) -> dict[str, Any]:
        endpoint = endpoint.strip().rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions" if endpoint.endswith("/v1") else "/v1/chat/completions"
        payload = {
            "model": model.strip() or "local-model",
            "temperature": 0,
            "messages": [
                {"role": "system", "content": self.system_prompt(state)},
                {"role": "user", "content": user_text},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"].strip()
        marker = chr(96) * 3
        if content.startswith(marker):
            content = re.sub("^" + marker + r"(?:json)?\s*|\s*" + marker + "$", "", content, flags=re.S)
        return self.validate(json.loads(content))

    def apply(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        plan = self.validate(plan)
        results = []
        for step in plan["steps"]:
            args = dict(step.get("args") or {})
            if "time_ms" in args:
                args["time_ms"] = parse_time_ms(args["time_ms"])
            if "interval_ms" in args and isinstance(args["interval_ms"], str):
                args["interval_ms"] = parse_time_ms(args["interval_ms"])
            result = self.registry.execute(step["tool"], args)
            results.append({"step": step, "result": result})
        return results
