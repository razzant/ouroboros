"""The owned server counts with its actual formatter and never invokes inference."""
import asyncio
import copy
import os
import sys
from types import SimpleNamespace

import httpx
import pytest

from ouroboros.local_model_server import (
    input_fingerprint, launched_n_ctx, measure_chat_input, with_measurement_route,
)


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


def test_launched_n_ctx_reads_this_process_launch_argument(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["python", "-m", "ouroboros.local_model_server",
                                      "--model", "m.gguf", "--port", "1",
                                      "--n_gpu_layers", "-1", "--n_ctx", "80000"])
    assert launched_n_ctx() == 80_000
    monkeypatch.setattr(sys, "argv", ["--n_ctx=80000", "--model", "m.gguf"])
    assert launched_n_ctx() == 80_000
    monkeypatch.setattr(sys, "argv", ["python", "-m", "ouroboros.local_model_server"])
    assert launched_n_ctx() == 0


def test_the_count_names_the_effective_window_beside_the_raw_allocation(native_model, monkeypatch):
    """Synthetic fixture numbers: the model's ``n_ctx()`` (81,920) stands for the backend's
    padded allocation of a launched 80,000, as measured releases showed 80,000 -> 80,128."""
    monkeypatch.setattr(sys, "argv", ["python", "-m", "ouroboros.local_model_server",
                                      "--model", "m.gguf", "--port", "1", "--n_ctx", "80000"])
    payload = {"model": "local-model", "messages": [{"role": "user", "content": "exact"}]}
    result = measure_chat_input(native_model, payload)
    assert result["input_tokens"] == 60_826
    assert result["context_window"] == 80_000  # the launched effective limit, both sides name it
    assert result["raw_context_window"] == 81_920  # the same instance's allocation rides beside it
    assert result["tokenizer_template_provenance"]["launched_n_ctx"] == 80_000


def test_an_allocation_smaller_than_requested_caps_the_effective_window(native_model, monkeypatch):
    """Never claim more room than the backend actually allocated."""
    monkeypatch.setattr(sys, "argv", ["--n_ctx", "90000"])
    result = measure_chat_input(native_model, {"messages": []})
    assert result["context_window"] == result["raw_context_window"] == 81_920


def test_without_a_launch_argument_the_raw_allocation_stands_alone(native_model, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["python", "-m", "ouroboros.local_model_server", "--model", "m.gguf"])
    result = measure_chat_input(native_model, {"messages": []})
    assert result["context_window"] == result["raw_context_window"] == 81_920
    assert result["tokenizer_template_provenance"]["launched_n_ctx"] == 0
