"""The owned server counts with its actual formatter and never invokes inference."""
import asyncio
import copy
import os
from types import SimpleNamespace

import httpx
import pytest

from ouroboros.local_model_server import input_fingerprint, measure_chat_input, with_measurement_route


@pytest.fixture
def native_model():
    llama = pytest.importorskip("llama_cpp")
    seen = []

    def formatter(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(prompt="actual rendered template", added_special=True)

    def tokenize(text, *, add_bos, special):
        assert text == b"actual rendered template" and not add_bos and special
        return list(range(60826))

    handler = llama.llama_chat_format.chat_formatter_to_chat_completion_handler(formatter)
    return SimpleNamespace(chat_handler=None, chat_format="actual-configured-format",
        _chat_handlers={"actual-configured-format": handler}, n_ctx=lambda: 81920, tokenize=tokenize,
        create_completion=lambda **kwargs: pytest.fail("Measurement must not generate"), seen=seen)


def test_selected_formatter_tokenizer_and_serving_capacity_are_used(native_model):
    payload = {"model": "local-model", "messages": [{"role": "user", "content": "exact"}],
               "tools": [{"type": "function", "function": {"name": "inspect"}}], "tool_choice": "auto"}
    result = measure_chat_input(native_model, payload)
    assert result["input_tokens"] == 60826 and result["context_window"] == 81920
    assert result["input_is_exact"] and result["output_limit_enforced"]
    assert result["native_input_sha256"] == input_fingerprint(payload)
    assert native_model.seen == [{**{key: None for key in ("functions", "function_call")},
                                 **{key: payload[key] for key in ("messages", "tools", "tool_choice")}}]
    assert "actual rendered template" not in str(result)


def test_measurement_fingerprint_is_after_host_metadata_scrubbing(native_model):
    payload = {"model": "local-model", "messages": [
        {"role": "user", "content": "exact", "_context_capsule": "host-only"},
    ], "tools": []}
    clean = {**payload, "messages": [{"role": "user", "content": "exact"}], "tools": []}
    result = measure_chat_input(native_model, clean)
    assert result["native_input_sha256"] == input_fingerprint(clean)
    assert result["native_input_sha256"] != input_fingerprint(payload)


def test_finalize_measures_the_provider_clean_candidate_not_the_caller_payload(monkeypatch):
    from ouroboros import llm_local

    client = object.__new__(llm_local._LocalLaneMixin)
    payload = {"model": "local-model", "messages": [
        {"role": "user", "content": "exact", "_context_capsule": "host-only"},
    ], "max_tokens": 256}
    clean = {"model": "local-model", "messages": [{"role": "user", "content": "exact"}], "max_tokens": 256}
    seen = []

    monkeypatch.setattr(llm_local, "_finalized_physical_candidate", lambda target, value, surface: clean)
    monkeypatch.setattr("ouroboros.local_model.get_manager", lambda: SimpleNamespace(
        measure_prepared_input=lambda value: seen.append(copy.deepcopy(value)) or {
            "supported": True, "input_is_exact": True, "input_tokens": 8,
            "context_window": 81920, "process_id": 1,
            "native_input_sha256": input_fingerprint(value),
            "output_limit_enforced": True,
            "reasoning_included_in_limit": True,
            "tokenizer_template_provenance": {"source": "fixture"},
        }))
    target = {"provider": "local", "resolved_model": "local-model", "usage_model": "local-model"}
    assert client._finalize_local_candidate(target, payload) == clean
    assert seen == [clean]
    assert target["local_input_measurement"]["native_input_sha256"] == input_fingerprint(clean)


def test_unrecognized_handler_is_not_called_to_guess_its_behavior(native_model):
    native_model.chat_handler = lambda **kwargs: pytest.fail("Opaque handler was invoked")
    result = measure_chat_input(native_model, {"messages": []})
    assert not result["supported"] and result["input_tokens"] is None
    assert not result["output_limit_enforced"] and native_model.seen == []


@pytest.mark.parametrize("launch_id", [None, "fixture-launch"])
def test_owned_model_list_adds_only_its_launch_header(monkeypatch, launch_id):
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    from ouroboros.local_model_server import LOCAL_LAUNCH_ENV, LOCAL_LAUNCH_HEADER, with_launch_identity

    if launch_id is None:
        monkeypatch.delenv(LOCAL_LAUNCH_ENV, raising=False)
    else:
        monkeypatch.setenv(LOCAL_LAUNCH_ENV, launch_id)
    models = {"object": "list", "data": [{"id": "fixture"}]}

    async def catalogue(request):
        return JSONResponse(models, status_code=int(request.query_params.get("status", 200)))

    app = with_launch_identity(Starlette(routes=[
        Route("/v1/models", catalogue, methods=["GET", "POST"]), Route("/other", catalogue),
    ]))

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
            reply = await client.get("/v1/models")
            assert reply.status_code == 200 and reply.json() == models
            assert reply.headers.get(LOCAL_LAUNCH_HEADER) == launch_id
            for other in (await client.post("/v1/models"), await client.get("/other"),
                          await client.get("/v1/models", params={"status": 503})):
                assert LOCAL_LAUNCH_HEADER not in other.headers

    asyncio.run(run())


def test_count_waits_for_generation_without_using_its_interrupt_lock(native_model, monkeypatch):
    FastAPI = pytest.importorskip("fastapi").FastAPI
    server = pytest.importorskip("llama_cpp.server.app")

    async def run():
        inner, outer = asyncio.Lock(), asyncio.Lock()
        monkeypatch.setattr(server, "llama_inner_lock", inner)
        monkeypatch.setattr(server, "llama_outer_lock", outer)
        monkeypatch.setattr(server, "_llama_proxy", lambda _model: native_model)
        app = with_measurement_route(FastAPI())
        app.dependency_overrides[server.authenticate] = lambda: None
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
            await inner.acquire()
            request = asyncio.create_task(client.post("/extras/measure_chat", json={
                "model": "local-model", "messages": [{"role": "user", "content": "exact"}]}))
            await asyncio.sleep(0.02)
            assert not request.done() and not outer.locked() and not native_model.seen
            inner.release()
            result = await request
            assert result.status_code == 200
            assert result.json()["input_tokens"] == 60826 and result.json()["process_id"] == os.getpid()
            assert not inner.locked() and not outer.locked()

    asyncio.run(run())
