from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from mcp.server.fastmcp import FastMCP

BRIDGE = "http://127.0.0.1:8765"
mcp = FastMCP("MiniCut Studio")

def request_json(
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 65,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BRIDGE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw or "{}")
        except json.JSONDecodeError:
            raise RuntimeError(
                f"MiniCut bridge HTTP {exc.code}: {raw or exc.reason}"
            ) from exc
        if isinstance(payload, dict):
            return payload
        raise RuntimeError(
            f"MiniCut bridge HTTP {exc.code}: respons tidak valid."
        ) from exc

def run_tool(name: str, **args):
    interactive_tools = {
        "open_video", "open_project", "choose_subtitle",
        "save_project", "export_all",
    }
    timeout = 310 if name in interactive_tools else 65
    result = request_json("/tool", {"tool": name, "args": args}, timeout=timeout)
    if not result.get("ok", False):
        raise RuntimeError(result.get("error") or "MiniCut tool gagal.")
    return result

@mcp.tool()
def get_state() -> dict:
    """Baca status proyek MiniCut, termasuk status SRT wajib untuk ekspor."""
    return request_json("/state")

@mcp.tool()
def open_video() -> dict:
    """Buka dialog pemilihan video di MiniCut."""
    return run_tool("open_video")

@mcp.tool()
def open_project() -> dict:
    """Buka dialog pemilihan proyek MiniCut."""
    return run_tool("open_project")

@mcp.tool()
def choose_subtitle() -> dict:
    """Pilih subtitle SRT untuk AI Film Cut dan ekspor wajib video + SRT."""
    return run_tool("choose_subtitle")

@mcp.tool()
def seek(time_ms: int) -> dict:
    """Pindahkan playhead ke timestamp dalam milidetik."""
    return run_tool("seek", time_ms=time_ms)

@mcp.tool()
def play() -> dict:
    """Putar video."""
    return run_tool("play")

@mcp.tool()
def pause() -> dict:
    """Jeda video."""
    return run_tool("pause")

@mcp.tool()
def set_playback_rate(rate: float) -> dict:
    """Atur kecepatan playback 0.25x sampai 4x."""
    return run_tool("set_playback_rate", rate=rate)

@mcp.tool()
def step_frame(direction: int) -> dict:
    """Maju 1 frame (1) atau mundur 1 frame (-1)."""
    return run_tool("step_frame", direction=direction)

@mcp.tool()
def add_cut(time_ms: int) -> dict:
    """Tambah batas part pada timestamp dalam milidetik."""
    return run_tool("add_cut", time_ms=time_ms)

@mcp.tool()
def remove_cut(index: int) -> dict:
    """Hapus cut dengan index 0-based."""
    return run_tool("remove_cut", index=index)

@mcp.tool()
def clear_cuts() -> dict:
    """Hapus semua cut."""
    return run_tool("clear_cuts")

@mcp.tool()
def divide_equal(parts: int) -> dict:
    """Bagi seluruh film menjadi jumlah part yang diminta."""
    return run_tool("divide_equal", parts=parts)

@mcp.tool()
def divide_interval(interval_ms: int) -> dict:
    """Bagi film berdasarkan interval milidetik."""
    return run_tool("divide_interval", interval_ms=interval_ms)

@mcp.tool()
def start_film_cut() -> dict:
    """Mulai AI Film Cut menggunakan pengaturan saat ini."""
    return run_tool("start_film_cut")

@mcp.tool()
def cancel_film_cut() -> dict:
    """Batalkan AI Film Cut yang sedang berjalan."""
    return run_tool("cancel_film_cut")

@mcp.tool()
def apply_film_cut() -> dict:
    """Terapkan hasil AI Film Cut ke timeline."""
    return run_tool("apply_film_cut")

@mcp.tool()
def preview_film_cut(row: int) -> dict:
    """Preview baris hasil AI Film Cut (1-based)."""
    return run_tool("preview_film_cut", row=row)

@mcp.tool()
def set_export_mode(mode: str) -> dict:
    """Pilih mode ekspor: smartcut atau fast."""
    return run_tool("set_export_mode", mode=mode)

@mcp.tool()
def save_project() -> dict:
    """Simpan proyek MiniCut."""
    return run_tool("save_project")

@mcp.tool()
def export_all() -> dict:
    """Mulai ekspor seluruh part video + SRT. Ekspor tanpa SRT ditolak."""
    return run_tool("export_all")

@mcp.tool()
def cancel_export() -> dict:
    """Batalkan ekspor yang sedang berjalan."""
    return run_tool("cancel_export")

@mcp.tool()
def undo() -> dict:
    """Batalkan perubahan timeline terakhir yang dibuat melalui agent/tool."""
    return run_tool("undo")

if __name__ == "__main__":
    mcp.run()
