from __future__ import annotations

import json
import threading

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Seconds the mock model holds every completion before answering (0 = answer at
# once). A browser fixture that needs a GENUINELY running root sets this so the
# dispatched task stays in its first model call for the test's duration; setting
# HOLD_RELEASE lets a held handler answer at once (fixture teardown joins it).
HOLD_SECONDS = 0.0
HOLD_RELEASE = threading.Event()


class MockLLMServer:
    def __init__(self):
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._server.shutdown()
        self._server.server_close()


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 - stdlib callback name
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        if HOLD_SECONDS > 0:
            HOLD_RELEASE.wait(HOLD_SECONDS)
        payload = {
            "id": "mock-chat",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "OK"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args):
        return
