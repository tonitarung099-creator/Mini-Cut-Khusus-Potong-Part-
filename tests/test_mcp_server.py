import io
import json
import unittest
import urllib.error
from unittest.mock import patch

import mcp_server


class McpErrorHandlingTests(unittest.TestCase):
    def test_http_error_keeps_bridge_json_message(self):
        body = json.dumps({
            "ok": False,
            "error": "Belum ada video.",
        }).encode("utf-8")
        error = urllib.error.HTTPError(
            "http://127.0.0.1:8765/tool",
            400,
            "Bad Request",
            hdrs=None,
            fp=io.BytesIO(body),
        )

        with patch("mcp_server.urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "Belum ada video"):
                mcp_server.run_tool("play")

    def test_camera_aware_add_cut_uses_long_timeout(self):
        with patch(
            "mcp_server.request_json",
            return_value={"ok": True},
        ) as request:
            result = mcp_server.run_tool("add_cut", time_ms=61_237)

        self.assertTrue(result["ok"])
        request.assert_called_once_with(
            "/tool",
            {"tool": "add_cut", "args": {"time_ms": 61_237}},
            timeout=310,
        )

    def test_lightweight_tool_keeps_short_timeout(self):
        with patch(
            "mcp_server.request_json",
            return_value={"ok": True},
        ) as request:
            mcp_server.run_tool("play")

        request.assert_called_once_with(
            "/tool",
            {"tool": "play", "args": {}},
            timeout=65,
        )


if __name__ == "__main__":
    unittest.main()
