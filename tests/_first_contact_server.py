"""Shared llama.cpp-like HTTP server for first-contact and refusal recovery tests."""
from __future__ import annotations

import io
import json
import math

from tests.system_e2e.harness import ScriptedStubModel
from tests._first_contact import ANSWER

BYTES_PER_TOKEN = 3  # the fake server's tokenizer: a request's JSON bytes / 3


def llama_refusal(n_prompt: int, n_ctx: int) -> dict:
    """llama.cpp's own body (server-context.cpp:4016; stand evidence2/00002, byte for byte apart from numbers)."""
    return {"error": {"code": 400, "message": f"request ({n_prompt} tokens) exceeds the available context size "
                                              f"({n_ctx} tokens), try increasing it",
                      "type": "exceed_context_size_error", "n_prompt_tokens": n_prompt, "n_ctx": n_ctx}}


class FakeLlamaServer(ScriptedStubModel):
    def __init__(self, *, n_ctx: int, publish_window: bool, refuse_with=None):
        super().__init__(final_answer=ANSWER)
        self.n_ctx, self.publish_window = n_ctx, publish_window
        self.refuse_with = refuse_with or (lambda n_prompt: (400, llama_refusal(n_prompt, self.n_ctx)))
        self.posts, self.gets = [], []
        base, outer = self._server.RequestHandlerClass, self

        class Handler(base):
            def _raw(self, status, payload):
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Server", "llama.cpp")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802
                outer.gets.append(self.path)
                meta = {"n_ctx": outer.n_ctx, "n_ctx_train": 262_144} if outer.publish_window else {}
                if self.path.rstrip("/").endswith("/models"):
                    return self._raw(200, {"object": "list", "data": [
                        {"id": "mock-model", "object": "model", "owned_by": "llamacpp", **({"meta": meta} if meta else {})}]})
                return self._raw(404, {"error": {"code": 404, "message": "File Not Found", "type": "not_found_error"}})

            def do_POST(self):  # noqa: N802
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                body = json.loads(raw or b"{}")
                n_prompt = math.ceil(len(json.dumps({"messages": body.get("messages"), "tools": body.get("tools")},
                                                    ensure_ascii=False).encode("utf-8")) / BYTES_PER_TOKEN)
                outer.posts.append({"n_prompt": n_prompt, "tools": len(body.get("tools") or []),
                                    "max_tokens": body.get("max_tokens") or body.get("max_completion_tokens")})
                if n_prompt >= outer.n_ctx:  # llama.cpp refuses at n_tokens >= n_ctx, before any prefill
                    outer.posts[-1]["refused"] = True
                    return self._raw(*outer.refuse_with(n_prompt))
                original = self.rfile
                self.rfile = io.BytesIO(raw)
                try:
                    super().do_POST()
                finally:
                    self.rfile = original

        self._server.RequestHandlerClass = Handler
