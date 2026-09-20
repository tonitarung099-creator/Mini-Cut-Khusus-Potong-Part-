from __future__ import annotations

import json
import urllib.request
from typing import Any

from mcp.server.fastmcp import FastMCP

BRIDGE = "http://127.0.0.1:8765"
mcp = FastMCP("MiniCut Studio")

def request_json(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BRIDGE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=65) as response:
        return json.loads(response.read().decode("utf-8"))

def run_tool(name: str, **args):
    result = request_json("/tool", {"tool": name, "args": args})
    if not result.get("ok", False):
        raise RuntimeError(result.get("error") or "MiniCut tool gagal.")
    return result

@mcp.tool()
def get_state() -> dict:
    """Baca status proyek MiniCut, playhead, cut, dan jumlah part."""
    return request_json("/state")

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
def save_project() -> dict:
    """Simpan proyek MiniCut."""
    return run_tool("save_project")

@mcp.tool()
def export_all() -> dict:
    """Mulai ekspor seluruh part."""
    return run_tool("export_all")

@mcp.tool()
def undo() -> dict:
    """Batalkan perubahan timeline terakhir yang dibuat melalui agent/tool."""
    return run_tool("undo")

if __name__ == "__main__":
    mcp.run()
