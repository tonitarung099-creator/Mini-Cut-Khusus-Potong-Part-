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


if __name__ == "__main__":
    unittest.main()
