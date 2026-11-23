"""Shared native Ollama HTTP fixtures; no sockets or model processes.

The counters and refusal sentence reproduce Ollama 0.40.16. Synthetic
responses exercise the client contract; they do not claim live model quality.
"""


from __future__ import annotations
import asyncio
import copy
import json
import httpx
import pytest


TARGET = {
    "provider": "openai-compatible", "resolved_model": "mock-model",
    "requested_model": "openai-compatible::mock-model", "usage_model": "openai-compatible::mock-model",
    "base_url": "http://127.0.0.1:11434/v1", "api_key": "test-only",
}


MESSAGES = [{"role": "user", "content": "Сколько будет 17 × 23?"}]


TOOLS = [{"type": "function", "function": {"name": "get_value", "description": "Read one test value",
         "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}}]


def completed(**overrides):
    return {"model": "mock-model", "message": {"role": "assistant", "content": "391"},
            "done": True, "done_reason": "stop", "prompt_eval_count": 25, "eval_count": 4, **overrides}


def overflow_reply(body, tokens, window):
    """Native semantics: default drops history, truncate:false refuses it."""
    if body.get("truncate") is False:
        return httpx.Response(400, json={"error": f"request ({tokens} tokens) exceeds the available context size "
                                                   f"({window} tokens), try increasing it"})
    return completed(message={"role": "assistant", "content": "MISSING_START_CANARY"},
                     prompt_eval_count=window // 2)


class OllamaHTTP:
    """A transport-level native server with no listening port."""

    def __init__(self, reply=None, *, window=80_000, status=200, loaded=True):
        self.reply = reply or completed()
        self.window, self.status, self.loaded = window, status, loaded
        self.posts, self.gets, self.shows = [], [], []

    def __call__(self, request):
        path = request.url.path
        if request.method == "GET":
            self.gets.append(path)
            if path == "/api/version":
                return httpx.Response(200, json={"version": "0.40.16"})
            if path == "/api/ps":
                return httpx.Response(200, json={"models": [{"name": "mock-model:latest", "model": "mock-model:latest",
                    "digest": "sha256:test-model", "context_length": self.window}] if self.loaded else []})
            if path == "/v1/models":
                return httpx.Response(200, json={"object": "list", "data": [{"id": "mock-model", "owned_by": "library"}]})
            return httpx.Response(404, json={"error": "not found"})
        if path == "/api/show":
            self.shows.append(json.loads(request.content))
            return httpx.Response(200, json={"model_info": {"qwen35.context_length": 131072}, "capabilities": ["completion", "tools"]})
        assert path == "/api/chat", f"Ollama must use its native API, got {path}"
        body = json.loads(request.content)
        self.posts.append(body)
        self.loaded = True
        self.window = body.get("options", {}).get("num_ctx", self.window)
        reply = self.reply(body) if callable(self.reply) else self.reply
        if isinstance(reply, httpx.Response):
            return reply
        if self.status != 200:
            return httpx.Response(self.status, json=reply)
        if body.get("stream"):
            chunks = reply if isinstance(reply, list) else [reply]
            return httpx.Response(200, content=b"".join(json.dumps(chunk).encode() + b"\n" for chunk in chunks),
                                  headers={"Content-Type": "application/x-ndjson"})
        return httpx.Response(200, json=reply)


@pytest.fixture
def install_http(monkeypatch, tmp_path):
    """Replace both client transports, including clients built by route discovery."""
    from ouroboros import capability_evidence as ce, llm_ollama
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path / "capability-evidence")
    with llm_ollama._identity_lock:
        llm_ollama._rejected.clear()
    sync_init, async_init = httpx.Client.__init__, httpx.AsyncClient.__init__

    def install(handler):
        transport = httpx.MockTransport(handler)

        def sync(self, *args, **kwargs):
            kwargs.update(transport=transport, trust_env=False)
            kwargs.pop("mounts", None)
            sync_init(self, *args, **kwargs)

        def asynchronous(self, *args, **kwargs):
            kwargs.update(transport=transport, trust_env=False)
            kwargs.pop("mounts", None)
            async_init(self, *args, **kwargs)

        monkeypatch.setattr(httpx.Client, "__init__", sync)
        monkeypatch.setattr(httpx.AsyncClient, "__init__", asynchronous)
        return handler

    return install


def chat(*, asynchronous=False, **overrides):
    from ouroboros import llm_ollama

    kwargs = dict(target=dict(TARGET), messages=copy.deepcopy(MESSAGES), tools=None,
                  reasoning_effort="high", max_tokens=2048, tool_choice="auto", no_proxy=True)
    kwargs.update(overrides)
    from ouroboros.llm_routing import configured_window
    kwargs.setdefault("window", configured_window(kwargs["target"]))
    if asynchronous:
        return asyncio.run(llm_ollama.chat_ollama_async(**kwargs))
    return llm_ollama.chat_ollama(**kwargs)


def _chat_completion(content):
    return httpx.Response(200, json={
        "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "mock-model",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}})


class LlamaCppHTTP:
    """A llama.cpp-like compatible server: its own /props, no Ollama paths, an OpenAI chat answer on /v1."""

    def __init__(self):
        self.paths = []

    def __call__(self, request):
        self.paths.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path == "/v1/chat/completions":
            return _chat_completion("from llama.cpp")
        if request.method == "GET" and request.url.path == "/props":
            return httpx.Response(200, json={"default_generation_settings": {"n_ctx": 8192}, "total_slots": 1,
                                             "model_path": "mock-model.gguf", "build_info": "b0000-synthetic"})
        return httpx.Response(404, json={"error": {"code": 404, "message": "File Not Found", "type": "not_found_error"}})


class GatewayHTTP:
    """A compatible gateway that answers chat on /v1 and shows no program: every service read gets ``service``
    (an ``httpx.Response``, an exception class to raise, or a callable of the request)."""

    def __init__(self, service=None, chat=None):
        self.service = service if service is not None else httpx.Response(404, json={"detail": "Not Found"})
        self.chat = chat
        self.paths, self.requests = [], []

    def __call__(self, request):
        self.paths.append((request.method, request.url.path))
        self.requests.append(request)
        if request.method == "POST" and request.url.path.endswith("/chat/completions"):
            return self.chat(request) if self.chat else _chat_completion("from the gateway")
        if isinstance(self.service, type):
            raise self.service("synthetic transport failure", request=request)
        if callable(self.service):
            return self.service(request)
        reply = self.service  # one reply object serves one request: answer each with its own copy
        return httpx.Response(reply.status_code, headers=reply.headers, content=reply.content)

    @property
    def chats(self):
        return [path for method, path in self.paths if method == "POST" and path.endswith("/chat/completions")]


def _compatible_target(monkeypatch, base_url):
    from ouroboros import config, llm_routing

    settings = {"OUROBOROS_MODEL": "openai-compatible::mock-model", "OPENAI_COMPATIBLE_BASE_URL": base_url,
                "OPENAI_COMPATIBLE_API_KEY": "test-only"}
    monkeypatch.setattr(config, "runtime_settings", lambda: settings)
    monkeypatch.setattr(llm_routing, "runtime_setting", settings.get)
    return llm_routing.resolve_target(settings["OUROBOROS_MODEL"], settings=settings)
