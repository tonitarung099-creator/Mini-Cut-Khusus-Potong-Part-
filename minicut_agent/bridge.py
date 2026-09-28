from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable


@dataclass
class BridgeCall:
    tool: str
    args: dict[str, Any]
    event: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] = field(default_factory=dict)
    # A bridge request may time out while it is still queued behind another
    # long-running GUI action. Without an explicit cancellation marker, the GUI
    # would execute that stale request later even though the caller already saw
    # a failure and may have retried it.
    cancelled: threading.Event = field(default_factory=threading.Event)
    started: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)


class LocalBridge:
    def __init__(
        self,
        calls: "queue.Queue[BridgeCall]",
        state_provider: Callable[[], dict[str, Any]],
        manifest_provider: Callable[[], dict[str, Any]],
        host: str = "127.0.0.1",
        port: int = 8765,
    ):
        self.calls = calls
        self.state_provider = state_provider
        self.manifest_provider = manifest_provider
        self.host = host
        self.port = port
        self.httpd: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.stopping = False

    def start(self) -> None:
        if self.httpd:
            return
        self.stopping = False
        outer = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "MiniCutBridge/1.0"

            def _send(self, status: int, payload: dict[str, Any]) -> None:
                raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path == "/health":
                    self._send(200, {"ok": True, "service": "MiniCut Studio Agent"})
                elif self.path == "/state":
                    self._send(200, {"ok": True, "state": outer.state_provider()})
                elif self.path == "/manifest":
                    self._send(200, {"ok": True, "manifest": outer.manifest_provider()})
                else:
                    self._send(404, {"ok": False, "error": "Not found"})

            def do_POST(self):
                if self.path != "/tool":
                    return self._send(404, {"ok": False, "error": "Not found"})
                try:
                    length = int(self.headers.get("Content-Length") or "0")
                    payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                    tool = str(payload.get("tool") or "")
                    args = dict(payload.get("args") or {})
                    if not tool:
                        raise ValueError("tool wajib diisi")
                    if outer.stopping:
                        raise RuntimeError("MiniCut sedang ditutup.")
                    call = BridgeCall(tool=tool, args=args)
                    outer.calls.put(call)
                    if outer.stopping and not call.event.is_set():
                        with call.lock:
                            call.cancelled.set()
                            call.result.update({
                                "ok": False,
                                "cancelled": True,
                                "error": "MiniCut sedang ditutup.",
                            })
                            call.event.set()
                    long_running_tools = {
                        "open_video", "open_project", "choose_subtitle",
                        "save_project", "export_all",
                        # add_cut now runs local FFmpeg scene detection + ffprobe
                        # frame locking. On 4K/HEVC/HDD sources it can exceed the
                        # old 60 s bridge limit; timing out early could report a
                        # failure even though the GUI later applies the cut.
                        "add_cut",
                    }
                    wait_seconds = 300 if tool in long_running_tools else 60
                    if not call.event.wait(timeout=wait_seconds):
                        with call.lock:
                            if not call.event.is_set():
                                call.cancelled.set()
                                call.result.clear()
                                call.result.update({
                                    "ok": False,
                                    "cancelled": True,
                                    "error": (
                                        "MiniCut tidak merespons tool dalam "
                                        f"{wait_seconds} detik; request dibatalkan "
                                        "agar tidak dieksekusi terlambat."
                                    ),
                                })
                                call.event.set()
                        raise TimeoutError(
                            f"MiniCut tidak merespons tool dalam {wait_seconds} detik. "
                            "Request dibatalkan; jangan anggap aksi berhasil."
                        )
                    status = 200 if call.result.get("ok", False) else 400
                    self._send(status, call.result)
                except Exception as exc:
                    self._send(400, {"ok": False, "error": str(exc)})

            def log_message(self, fmt, *args):
                return

        self.httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(
            target=self.httpd.serve_forever,
            name="MiniCutLocalBridge",
            daemon=True,
        )
        self.thread.start()

    def _cancel_pending_calls(self) -> None:
        while True:
            try:
                call = self.calls.get_nowait()
            except queue.Empty:
                break
            with call.lock:
                call.cancelled.set()
                call.result.clear()
                call.result.update({
                    "ok": False,
                    "cancelled": True,
                    "error": "MiniCut sedang ditutup.",
                })
                call.event.set()

    def stop(self) -> None:
        self.stopping = True
        self._cancel_pending_calls()
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None
        self.thread = None
