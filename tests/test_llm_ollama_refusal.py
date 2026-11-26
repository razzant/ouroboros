"""Captured native errors through the shared refusal reader and real recovery.

The HTTP transport is fake; bodies are captured Ollama 0.40.16 responses.
No server/model is started. The real loop fits views and persists observations.
"""
import hashlib
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from tests._first_contact import ANSWER, _episode, _run
from tests._ollama import TARGET, OllamaHTTP, chat, completed, install_http as install_http

ROOT = Path(__file__).parent / "fixtures/size_refusals"
CASES = [c for c in json.loads((ROOT / "manifest.json").read_text()) if "stand-ollama" in c["file"]]


@pytest.mark.parametrize("case", CASES, ids=lambda c: Path(c["file"]).stem)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_stand_refusal_uses_adapter_type_then_upstream_reader(case, asynchronous, stream, install_http, tmp_path):
    from ouroboros import capability_evidence as ce, size_refusal
    from ouroboros.context_anchor import cited_limit_refusal
    from ouroboros.context_window import bound_target_scope, record_refusal
    from ouroboros.loop_llm_call import classify_llm_exception
    from ouroboros.loop_transport import provider_failure_hint

    raw = (ROOT / case["file"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == case["sha256"]
    # HTTP bytes are unchanged; NDJSON framing alone adds a newline.
    install_http(OllamaHTTP(lambda _: httpx.Response(200 if stream else 400, content=raw + (b"\n" if stream else b""),
        headers={"content-type": "application/x-ndjson" if stream else "application/json"}), window=80_000))
    with pytest.raises(Exception) as caught:
        chat(asynchronous=asynchronous, stream=stream, window=80_000)
    verdict = classify_llm_exception(caught.value)
    usage = {}
    size_refusal.stamp_failure(usage, caught.value, verdict)
    expected = case["native_expected_refusal"]
    sentence = json.loads(raw)["error"]
    assert sentence in provider_failure_hint(usage)
    assert usage["_last_llm_provider_message"] == sentence
    plan = SimpleNamespace(provider="openai-compatible", model=TARGET["requested_model"],
                           route_fp=bound_target_scope(TARGET, TARGET["requested_model"]),
                           window_record=None, window_witness_bound=True)
    before = ce._read(ce.canonical_evidence_root()).get("main_windows", {})
    if expected:
        from ouroboros._usage_response import provider_answer
        from ouroboros.context_budget import cited_declared_limit
        assert cited_declared_limit(provider_answer(caught.value).said, 2048, {80_000: "total"}) is None
        assert verdict.kind == "context_overflow" and verdict.basis == "typed"
        assert verdict.provider_code == "ollama_context_length_exceeded"
        found = size_refusal.from_exception(caught.value)
        assert {key: getattr(found, key) for key in expected} == expected
        # 80,128 is not 80,000: no equality-based fallback is needed or allowed to teach this observation.
        assert usage["_last_llm_provider_answer"]["basis"] == "typed"
        record_refusal(usage, plan)
        windows = ce._read(ce.canonical_evidence_root())["main_windows"]
        assert any(row.get("refused") == min(expected["window"], 80_000) and row["basis"] == "provider_stated"
                   for record in windows.values() for row in record.get("observations", []))
    else:
        assert verdict.kind == ("provider_error" if stream else "bad_request")
        assert size_refusal.from_exception(caught.value) is None
        capture = SimpleNamespace(physical_context=SimpleNamespace(route_fp=plan.route_fp), max_completion_tokens=2048)
        measurement = SimpleNamespace(route_fp=plan.route_fp, reply_allowance_tokens=2048)
        assert not cited_limit_refusal(usage, capture, measurement, plan)
        assert ce._read(ce.canonical_evidence_root()).get("main_windows", {}) == before


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("control", [False, True], ids=["size", "grammar-control"])
def test_real_turn_rebuilds_after_captured_80128_refusal(tmp_path, monkeypatch, install_http, control, stream):
    from ouroboros import capability_evidence as ce, llm_ollama, loop, loop_delivery
    from ouroboros.tool_policy import initial_tool_schemas
    from ouroboros.openai_chat_dispatch import sanitize_function_tools
    native_chat = llm_ollama.chat_ollama
    delivered = []

    def transport(*args, **kwargs):
        kwargs["stream"] = stream
        return native_chat(*args, **kwargs)

    def deliver(content, *args, **kwargs):
        delivered.append(content)
        return loop_delivery._no_tool_final_answer(content, *args, **kwargs)

    monkeypatch.setattr(llm_ollama, "chat_ollama", transport)
    monkeypatch.setattr(loop, "_no_tool_final_answer", deliver)
    case = next(c for c in CASES if ("grammar-control" if control else "native-90k") in c["file"])
    raw = (ROOT / case["file"]).read_bytes()
    observations = []

    def answer(body):
        tokens = math.ceil(len(json.dumps({"messages": body["messages"], "tools": body.get("tools")},
                                        ensure_ascii=False).encode()) / 3)
        observations.append(tokens)
        if len(observations) == 1 or control:
            return httpx.Response(200 if stream else 400, content=raw + (b"\n" if stream else b""),
                                  headers={"content-type": "application/x-ndjson" if stream else "application/json"})
        return completed(message={"role": "assistant", "content": ANSWER, "thinking": "native private plan"}, prompt_eval_count=tokens)

    server = install_http(OllamaHTTP(answer, window=80_000, loaded=False))
    shutil.copytree(Path(__file__).resolve().parents[1] / "docs/development", tmp_path / "repo/docs/development")
    episode = _episode(tmp_path, monkeypatch, SimpleNamespace(base_url=TARGET["base_url"]))
    assert episode.plan.initial_mode == "max"
    text, usage, _ = _run(episode)
    windows = ce._read(ce.canonical_evidence_root()).get("main_windows", {})
    refused = [row.get("refused") for record in windows.values() for row in record.get("observations", [])]
    assert all(post["truncate"] is False and post["shift"] is False for post in server.posts)
    if control:
        assert ANSWER not in str(text)
        assert json.loads(raw)["error"] in str(text)
        assert not any(refused)
        assert episode.tools._ctx.active_context_mode == "max"
    else:
        assert ANSWER in str(text), (text, observations)
        assert delivered == [ANSWER], delivered
        assert "native private plan" not in str(text)
        expected = sanitize_function_tools(initial_tool_schemas(episode.tools, "nano"),
                                           description_normalizer=lambda value: str(value or ""))
        assert server.posts[-1]["tools"] == expected
        assert len(server.posts[0]["tools"]) > len(expected)
        assert all(post["stream"] is stream for post in server.posts)
        assert 80_000 in refused and 80_128 not in refused
        assert episode.tools._ctx.active_context_mode == "nano"
        assert len(server.posts) >= 2 and server.posts[0]["messages"] != server.posts[-1]["messages"]
        assert observations[-1] < 80_000 < observations[0]
        assert max(post["options"].get("num_ctx", 0) for post in server.posts) <= 80_000
        assert episode.task["id"] == "first-contact"
    print("CAPTURED_OLLAMA_TURN", {"control": control, "input_estimates": observations,
                                 "refused": refused, "owner_text": str(text)})


@pytest.mark.parametrize("stream", [False, True], ids=["http-400", "ndjson-200"])
def test_full_runner_overflow_preserves_words_and_uses_loaded_window(install_http, stream):
    from ouroboros.loop_llm_call import classify_llm_exception
    from ouroboros.size_refusal import stamp_failure

    # llm/llama_server.go in Ollama v0.40.16. This refusal contains no token counts.
    sentence = ("the prompt is longer than the context length currently available to the model; "
                "shorten the prompt, adjust the context length in settings, or use a model with a longer context length")
    body = {"error": sentence}
    server = install_http(OllamaHTTP([body] if stream else body, status=200 if stream else 400, window=80000))
    with pytest.raises(Exception) as caught:
        chat(stream=stream, window=0)
    verdict = classify_llm_exception(caught.value)
    usage = {}
    refusal = stamp_failure(usage, caught.value, verdict)
    assert verdict.kind == "context_overflow" and verdict.provider_code == "ollama_context_length_exceeded"
    assert usage["_last_llm_provider_message"] == sentence
    assert (refusal.window, refusal.requested, refusal.token_window) == (0, 0, 80000)
    assert len(server.posts) == 1


@pytest.mark.parametrize("status", [401, 429, 500])
def test_native_words_cannot_retype_another_http_family(status):
    from ouroboros.llm_ollama import _http_error
    from ouroboros.loop_llm_call import classify_llm_exception
    raw = (ROOT / next(c for c in CASES if "native-90k" in c["file"])["file"]).read_bytes()
    response = httpx.Response(status, content=raw, request=httpx.Request("POST", "http://localhost/api/chat"))
    with pytest.raises(httpx.HTTPStatusError) as caught:
        _http_error(response)
    assert classify_llm_exception(caught.value).kind != "context_overflow"
