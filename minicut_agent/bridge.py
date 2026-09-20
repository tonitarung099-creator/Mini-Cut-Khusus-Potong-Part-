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

    def start(self) -> None:
        if self.httpd:
            return
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
                    call = BridgeCall(tool=tool, args=args)
                    outer.calls.put(call)
                    if not call.event.wait(timeout=60):
                        raise TimeoutError("MiniCut tidak merespons tool dalam 60 detik.")
                    status = 200 if call.result.get("ok", False) else 400
                    self._send(status, call.result)
                except Exception as exc:
                    self._send(400, {"ok": False, "error": str(exc)})

            def log_message(self, fmt, *args):
                return

        self.httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="MiniCutLocalBridge", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None
        self.thread = None
