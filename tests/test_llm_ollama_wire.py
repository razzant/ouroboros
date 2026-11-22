"""Native Ollama contracts for llm_ollama_wire."""
import copy
import json
import httpx
import pytest
from tests._ollama import (
    MESSAGES, TARGET, TOOLS, completed, OllamaHTTP,
    chat, install_http as install_http,
)


@pytest.mark.parametrize("mode,shown", [
    ("off", {"capabilities": ["completion"]}),
    ("legacy", {"capabilities": ["completion", "thinking"]}),
    ("boolean", {"capabilities": ["completion", "thinking"], "thinking": {"values": [False, True], "default": True}}),
    ("named", {"capabilities": ["completion", "thinking"], "thinking": {"values": ["low", "high", "xhigh"], "default": "low"}}),
    ("legacy", {"capabilities": ["completion", "thinking"], "thinking": {"values": [False, True]}}),
])
def test_public_reasoning_effort_uses_the_existing_show_read(install_http, monkeypatch, mode, shown):
    from ouroboros.llm_ollama_wire import OllamaUnsupportedRequest
    from ouroboros.llm import LLMClient
    from tests._ollama import _compatible_target

    _compatible_target(monkeypatch, TARGET["base_url"])
    server, shows = OllamaHTTP(), []
    def handle(request):
        if request.url.path == "/api/show":
            shows.append(json.loads(request.content)["model"])
            return httpx.Response(200, json=shown)
        if request.url.path == "/api/chat" and mode == "named":
            think = json.loads(request.content).get("think")
            if think is not None and think is not False and think not in shown["thinking"]["values"]:
                return httpx.Response(400, json={"error": "unsupported thinking level"})
        return server(request)

    install_http(handle)
    client = LLMClient()
    for requested, legacy in [("", None), ("none", False), ("minimal", "low"), ("low", "low"), ("medium", "medium"),
                              ("high", "high"), ("xhigh", "max"), ("max", "max"), ("ultra", "max"),
                              ("custom-level", None)]:
        kwargs = dict(reasoning_effort=requested, no_proxy=True, max_tokens=64)
        if mode == "legacy" and requested and legacy is None:
            count = len(server.posts)
            with pytest.raises(OllamaUnsupportedRequest):
                client.chat(MESSAGES, TARGET["requested_model"], **kwargs)
            assert len(server.posts) == count
            continue
        if mode == "named" and requested not in ("", "none") and requested not in shown["thinking"]["values"]:
            with pytest.raises(httpx.HTTPStatusError) as refused:
                client.chat(MESSAGES, TARGET["requested_model"], **kwargs)
            assert refused.value.physical_attempt_capture.effort["sent"] == {"think": requested}
            continue
        _, usage = client.chat(MESSAGES, TARGET["requested_model"], **kwargs)
        expected = {"off": {}, "legacy": {"think": legacy}, "boolean": {"think": True if legacy else requested},
                    "named": {"think": requested}}[mode] if requested != "none" else {"think": False}
        if not requested:
            expected = {}
        assert {k: v for k, v in server.posts[-1].items() if k == "think"} == expected
        assert usage["effort"]["requested"] == requested and usage["effort"]["sent"] == expected
        assert usage["effort"]["sent_state"] == ("explicit" if expected else "omitted")
        assert usage["effort"]["reported"] is None
    assert shows == [TARGET["resolved_model"]]


@pytest.mark.parametrize("cached", [None, 0, 30_026])
def test_full_prompt_count_is_not_reduced_by_cached_tokens(install_http, tmp_path, cached):
    from ouroboros import usage_accounting as ua
    from ouroboros.event_usage import event_token_counts
    from tests._usage_store_testing import attempt_rows_in_start_order
    counters = {"prompt_eval_cached_count": cached} if cached is not None else {}
    install_http(OllamaHTTP(completed(prompt_eval_count=30_030, **counters)))
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="cache", root_task_id="cache")):
        ua.stash_task_cache_split("cache", TARGET["usage_model"], 133, provider=TARGET["provider"], ttl_seconds=300)
        _, usage = chat()
        assert ua.last_task_cache_split("cache", TARGET["usage_model"], provider=TARGET["provider"]) == cached
    [row] = attempt_rows_in_start_order(tmp_path)
    assert row["cached_tokens"] == event_token_counts(usage)["cached_tokens"] == cached
    assert usage["prompt_tokens"] == 30_030
    assert usage["cached_tokens"] == cached
    assert usage["completion_tokens"] == 4


@pytest.mark.parametrize("stream", [False, True])
def test_generation_boundary_is_length_not_successful_stop(install_http, stream):
    install_http(OllamaHTTP(completed(done_reason="length", prompt_eval_count=79_818, eval_count=310)))
    _, usage = chat(stream=stream, max_tokens=512)
    assert usage["response_finish_reason"] == "length"
    assert usage["completion_tokens"] == 310


def test_stream_preserves_two_calls_and_keeps_thinking_out_of_content(install_http):
    calls = [{"function": {"name": "get_value", "arguments": {"name": name}}, "index": index}
             for index, name in enumerate(("alpha", "beta"))]
    chunks = [{"message": {"role": "assistant", "thinking": "Read both "}, "done": False},
              {"message": {"role": "assistant", "thinking": "values.", "tool_calls": [calls[0]]}, "done": False},
              {"message": {"role": "assistant", "content": "Checking.", "tool_calls": [calls[1]]}, "done": False},
              completed(message={"role": "assistant", "content": ""}, prompt_eval_count=314, eval_count=50)]
    server = install_http(OllamaHTTP(chunks))
    message, _ = chat(stream=True, tools=TOOLS)
    assert message["content"] == "Checking."
    assert message["reasoning"] == "Read both values."
    assert len(message["tool_calls"]) == 2
    ids = [call["id"] for call in message["tool_calls"]]
    assert len(set(ids)) == 2 and all(ids)
    assert [json.loads(call["function"]["arguments"]) for call in message["tool_calls"]] == [
        {"name": "alpha"}, {"name": "beta"}]
    server.reply = completed(message={"role": "assistant", "content": "200"}, prompt_eval_count=413)
    history = MESSAGES + [message] + [{"role": "tool", "tool_call_id": ids[index], "content": value}
                                    for index, value in enumerate(("120", "80"))]
    original = copy.deepcopy(history)
    chat(messages=history, tools=TOOLS)
    assert history == original, "native projection must not mutate canonical history"
    wire = server.posts[-1]["messages"]
    assert [item["tool_name"] for item in wire if item["role"] == "tool"] == ["get_value", "get_value"]
    assert [call["function"]["arguments"] for call in wire[1]["tool_calls"]] == [{"name": "alpha"}, {"name": "beta"}]


def test_replay_original_ollama_0_40_16_canary(install_http):
    from hashlib import sha256
    from pathlib import Path

    wire = (Path(__file__).parent / "fixtures/llm_wire/ollama/ollama-0.40.16-synthetic-canary.ndjson").read_bytes()
    assert len(wire) == 2273 and sha256(wire).hexdigest() == "015b9e95717067ebcbef2e75f437bdfb87426e496978b481d5bfa3cc44e2df47"
    install_http(OllamaHTTP(lambda _: httpx.Response(200, content=wire, headers={"Content-Type": "application/x-ndjson"})))
    message, usage = chat(stream=True)
    assert message["content"] == "ЛАЗУРНЫЙ_КАПИБАРА_917"
    assert usage["response_finish_reason"] == "stop"
    assert (usage["prompt_tokens"], usage["cached_tokens"], usage["completion_tokens"]) == (44_498, 0, 16)


def test_compressible_input_keeps_the_guarded_answer_and_raw_usage(install_http):
    history = [{"role": "user", "content": "START" + "=" * 32768 + "END"}]
    server = install_http(OllamaHTTP(completed(prompt_eval_count=560)))
    message, usage = chat(messages=history)
    assert message["content"] == "391" and usage["prompt_tokens"] == 560
    [post] = server.posts
    assert post["messages"] == history and post["truncate"] is post["shift"] is False


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("tail", [[], [{"error": "failed after output"}]], ids=["eof", "error-after-output"])
def test_stream_without_terminal_receipt_is_not_a_complete_answer(install_http, monkeypatch, asynchronous, tail):
    from ouroboros import observability
    from ouroboros.llm_stream import IncompleteProviderStream
    from ouroboros.loop_llm_call import classify_llm_exception

    frame = {"message": {"role": "assistant", "content": "Partial"}, "done": False}
    frames = [frame, *tail]
    server = install_http(OllamaHTTP(frames))
    saved = []
    write = observability.write_blob
    def capture(root, value, **options):
        if isinstance(value, dict) and "partial_assembly" in value:
            saved.append(value)
        return write(root, value, **options)
    monkeypatch.setattr(observability, "write_blob", capture)
    with pytest.raises(IncompleteProviderStream) as caught:
        chat(asynchronous=asynchronous, stream=True)
    error = caught.value
    assert classify_llm_exception(error).kind == "provider_outcome_unknown"
    retained = saved[0] if saved else error.stream_evidence
    import base64
    assert base64.b64decode(retained["wire_base64"]) == b"".join(json.dumps(item).encode() + b"\n" for item in frames)
    assert retained["partial_assembly"]["content"] == "Partial" and error.stream_receipt["model_output_received"]
    assert error.retain_partial_reply
    assert len(server.posts) == 1


def test_unsupported_message_block_refuses_before_sending(install_http):
    server = install_http(OllamaHTTP())
    with pytest.raises(ValueError):
        chat(messages=[{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "AA=="}}]}])
    assert server.posts == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("fault", ["reason_missing", "reason_invalid", "call_missing_name", "call_bad_object", "cache_invalid"])
def test_terminal_integrity_rejection_keeps_valid_usage(install_http, tmp_path, stream, fault):
    from ouroboros import usage_accounting as ua
    from ouroboros.llm_stream import RejectedProviderStream
    from ouroboros.loop_llm_call import classify_llm_exception
    from tests._usage_store_testing import attempt_rows_in_start_order

    response = completed(prompt_eval_cached_count=10)
    if fault == "reason_missing":
        response.pop("done_reason")
    elif fault == "reason_invalid":
        response["done_reason"] = []
    elif fault == "cache_invalid":
        response["prompt_eval_cached_count"] = 26
    else:
        response["message"]["tool_calls"] = [None if fault == "call_bad_object" else {"function": {"arguments": {}}}]
    server = install_http(OllamaHTTP(response))
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="reject", root_task_id="reject")), \
            pytest.raises(RejectedProviderStream) as caught:
        chat(stream=stream)
    verdict = classify_llm_exception(caught.value)
    assert verdict.kind == "provider_error" and not verdict.retry_same_request
    assert caught.value.stream_receipt["complete"]
    [row] = attempt_rows_in_start_order(tmp_path)
    assert len(server.posts) == 1 and row["state"] == "settled"
    assert (row["prompt_tokens"], row["completion_tokens"], row["cached_tokens"]) == (
        25, 4, None if fault == "cache_invalid" else 10)


def test_inline_image_reaches_the_guarded_native_request_as_pixels(install_http):
    server = install_http(OllamaHTTP(completed(prompt_eval_count=500)))
    encoded = "YWJj" * 30_000  # synthetic inline media payload; no remote URL or file read
    message, _ = chat(messages=[{"role": "user", "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}},
    ]}])
    assert message["content"] == "391"
    assert server.posts[-1]["messages"] == [
        {"role": "user", "content": "Describe this image."}, {"role": "user", "content": "", "images": [encoded]},
    ]


def test_main_admits_native_pixels_and_attributes_refusal_to_the_sent_attachment(install_http, monkeypatch, tmp_path):
    import base64
    import hashlib
    import io
    from PIL import Image
    from ouroboros import capability_evidence as ce, usage_accounting as ua, vision_routing
    from ouroboros.llm import LLMClient
    from ouroboros.loop_llm_call import _send_main_candidate
    from ouroboros.loop_model_call import _presend_context_fit
    from tests._ollama import _compatible_target

    target = _compatible_target(monkeypatch, TARGET["base_url"])
    png = io.BytesIO()
    Image.new("RGB", (512, 512), "green").save(png, format="PNG", compress_level=0)
    encoded = base64.b64encode(png.getvalue()).decode("ascii")
    arg = {"images": [{"type": "image", "content": "q" * 20_000}]}
    history = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
         "function": {"name": "lookup", "arguments": json.dumps(arg)}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "OK"},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": ""}}]},
    ]
    model = target["capability_route"]["model"]
    physical = ua.PhysicalAttemptContext("owner_low", "low", "cold_estimate", "route", "round", None,
                                         80_000, False, False, view_kind="focus", response_reserve_tokens=8192)
    server = install_http(OllamaHTTP(completed(prompt_eval_count=6500)))
    density = []
    monkeypatch.setattr(ce, "record_token_density", lambda *args, **kw: density.append(kw))
    admitted, captures = {}, []
    client = LLMClient()
    def send():
        kwargs = dict(messages=history, model=model, max_tokens=64, no_proxy=True)
        with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="pixels", root_task_id="pixels")), \
                _presend_context_fit(admitted=admitted) as admit:
            return _send_main_candidate(client, kwargs, model=model, use_local=False, deadline_ts=None,
                                        physical_context=physical, candidate_predicate=None, physical_fit_predicate=admit)

    for index, mime in enumerate(("image/png", "image/x-png")):
        url = f"data:{mime};base64,{encoded}"
        history[-1]["content"][0]["image_url"]["url"] = url
        original = copy.deepcopy(history)
        if index == 0:
            assert send()[0]["content"] == "391"
            capture = ua.last_physical_attempt_capture()
            estimate = admitted[capture.candidate_raw_sha256]
            # The actual argument is ordinary text (~5000 tokens); one bounded image adds ~1100.
            assert 6000 < estimate < 7000
            assert density[-1]["prompt_chars"] == estimate * 4
        else:
            server.status, server.reply = 400, {"error": "image input is unsupported"}
            with pytest.raises(httpx.HTTPStatusError) as caught:
                send()
            capture = caught.value.physical_attempt_capture
            refusal = vision_routing.image_refusal({"_last_llm_error_kind": "bad_request"}, capture)
            assert refusal["digests"] == {hashlib.sha256(url.encode()).hexdigest()}
        captures.append(capture)
        assert history == original
        assert server.posts[-1]["messages"][-1]["images"] == [encoded]
        assert server.posts[-1]["messages"][0]["tool_calls"][0]["function"]["arguments"] == arg
    assert captures[0].candidate_raw_sha256 == captures[1].candidate_raw_sha256
    assert vision_routing.candidate_images(captures[0]) != vision_routing.candidate_images(captures[1])


@pytest.mark.parametrize("asynchronous", [False, True])
def test_caption_order_and_tool_identity_survive_mixed_history(install_http, asynchronous):
    server = install_http(OllamaHTTP())
    call = {"id": "call-7", "type": "function", "function": {"name": "read_pair", "arguments": "{}"}}
    history = [{"role": "assistant", "tool_calls": [call]}, {"role": "tool", "tool_call_id": "call-7", "content": [
        {"type": "text", "text": "Before:"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,QQ=="}},
        {"type": "text", "text": "After:"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,Qg=="}},
    ]}]
    original = copy.deepcopy(history)
    chat(asynchronous=asynchronous, messages=history)
    assert history == original
    identity = {"role": "tool", "tool_name": "read_pair", "tool_call_id": "call-7"}
    assert server.posts[-1]["messages"][1:] == [
        {**identity, "content": "Before:"}, {**identity, "content": "", "images": ["QQ=="]},
        {**identity, "content": "After:"}, {**identity, "content": "", "images": ["Qg=="]},
    ]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("prompt,output,reason", [(80_042, 1, "length"), (79_818, 310, "stop")])
def test_padding_keeps_valid_answer_and_counts_without_learning_capacity(install_http, asynchronous, stream, prompt, output, reason):
    from ouroboros import capability_evidence as ce
    server = install_http(OllamaHTTP(completed(prompt_eval_count=prompt, eval_count=output, done_reason=reason)))
    message, usage = chat(asynchronous=asynchronous, stream=stream, window=80_000)
    assert message["content"] == "391" and len(server.posts) == 1
    assert usage["prompt_tokens"] == prompt and usage["completion_tokens"] == output
    assert usage["ollama_context"]["exceeded_by"] == prompt + output - 80_000
    assert usage["response_finish_reason"] == reason
    assert usage["capacity_input_tokens"] == min(prompt, 80_000)
    data = ce._read(ce.canonical_evidence_root())
    assert all(row["window_tokens"] <= 80_000 for row in data.get("probes", {}).values())
    [scope] = data["main_windows"].values()
    assert [(r["source"], r["tokens"]) for r in scope["reads"]] == [(ce.SOURCE_SERVER_METADATA, 80_000)]
    assert not scope.get("observations")


@pytest.mark.parametrize("requested,loaded,accepted", [(120000, 80000, 80000), (0, 0, 0), (80000, 0, 80000)])
def test_native_acceptance_uses_only_known_buffers(install_http, requested, loaded, accepted):
    server = OllamaHTTP(completed(prompt_eval_count=80042, prompt_eval_cached_count=40021))
    def handle(request):
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": TARGET["resolved_model"], "context_length": loaded}]})
        return server(request)

    install_http(handle)
    message, usage = chat(window=requested)
    assert message["content"] == "391"
    assert (usage["prompt_tokens"], usage["cached_tokens"], usage["capacity_input_tokens"]) == (80042, 40021, accepted)


@pytest.mark.parametrize("stream", [False, True])
def test_cancel_during_optional_ps_retains_original_paid_response(install_http, tmp_path, stream):
    import asyncio
    from ouroboros import usage_accounting as ua
    from ouroboros.llm_ollama import chat_ollama_async
    from ouroboros.observability import read_call_payload
    from tests._usage_store_testing import attempt_rows_in_start_order

    server = OllamaHTTP(completed(prompt_eval_cached_count=7))

    async def exercise():
        enriching = asyncio.Event()

        class PendingMetadata(httpx.AsyncByteStream):
            async def __aiter__(self):
                enriching.set()
                await asyncio.Future()  # only the caller's cancellation can end the read
                yield b""

        def handle(request):
            if request.url.path == "/api/ps" and server.posts:
                return httpx.Response(200, stream=PendingMetadata())
            return server(request)

        install_http(handle)
        caught_in_receiver = []

        async def receiver():
            try:
                await chat_ollama_async(dict(TARGET), MESSAGES, None, "high", 2048, "auto", stream=stream)
            except asyncio.CancelledError as error:
                caught_in_receiver.append(error)
                raise

        running = asyncio.create_task(receiver())
        try:
            await asyncio.wait_for(enriching.wait(), 5)
        finally:
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
        assert running.cancelled()
        [error] = caught_in_receiver  # An outer Task await may synthesize a new exception.
        return error

    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="cancel", root_task_id="cancel")):
        cancelled = asyncio.run(exercise())
    [row] = attempt_rows_in_start_order(tmp_path)
    assert len(server.posts) == 1 and row["state"] == "settled"
    assert cancelled.physical_attempt_capture.attempt_id == row["attempt_id"]
    assert cancelled.response["choices"][0]["message"]["content"] == "391"
    manifest, payload, ref = read_call_payload(
        tmp_path, task_id="cancel", call_id=f'physical_{row["attempt_id"]}_response')
    assert cancelled.response_manifest_ref == ref["manifest_ref"]
    assert manifest["control_reason"] == "caller_cancelled"
    assert payload["response"] == cancelled.response
    assert payload["physical_attempt_capture"]["attempt_id"] == row["attempt_id"]
    assert payload["physical_attempt_capture"]["state"] == "settled"
    assert (row["prompt_tokens"], row["cached_tokens"], row["completion_tokens"]) == (25, 7, 4)


def test_replayed_thinking_is_preserved_when_the_template_counts_visible_text_only(install_http):
    content = "a short visible question " * 40
    thinking = "a long private chain of earlier reasoning " * 1500
    history = [{"role": "user", "content": content},
               {"role": "assistant", "content": "ok", "reasoning": thinking},
               {"role": "user", "content": "and now?"}]
    # The template drops replayed thinking: the server counts the visible text only.
    server = install_http(OllamaHTTP(completed(prompt_eval_count=260)))
    message, _ = chat(messages=copy.deepcopy(history))
    assert message["content"] == "391"
    assert server.posts[-1]["messages"][1]["thinking"] == thinking, "thinking is still sent"


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream-before-eof", "ndjson-during-close"])
def test_cancel_over_a_complete_terminal_reply_keeps_the_one_paid_answer(install_http, tmp_path, stream):
    """A Pause between the complete terminal reply and the transport's EOF (nonstream)
    or during the async response close (NDJSON) settles exactly that paid answer:
    usage on the attempt, the response on the cancellation, one POST ever, and the
    caller's cancellation still propagates promptly."""
    import asyncio
    from ouroboros import usage_accounting as ua
    from ouroboros.llm_ollama import chat_ollama_async
    from ouroboros.observability import read_call_payload
    from tests._usage_store_testing import attempt_rows_in_start_order

    server = OllamaHTTP()
    body = json.dumps(completed(prompt_eval_cached_count=7)).encode()
    posts = []

    async def exercise():
        terminal = asyncio.Event()

        class HeldEOF(httpx.AsyncByteStream):
            """The complete terminal bytes arrive; EOF (nonstream) or the response
            close (NDJSON) never completes: only the caller's cancellation ends it."""
            async def __aiter__(self):
                yield body + (b"\n" if stream else b"")
                if not stream:
                    terminal.set()
                    await asyncio.Future()
                yield b""

            async def aclose(self):
                if stream:
                    terminal.set()
                    await asyncio.Future()

        def handle(request):
            if request.url.path == "/api/chat":
                posts.append(json.loads(request.content))
                return httpx.Response(200, stream=HeldEOF(),
                                      headers={"Content-Type": "application/x-ndjson"})
            return server(request)

        install_http(handle)
        caught_in_receiver = []

        async def receiver():
            try:
                await chat_ollama_async(dict(TARGET), MESSAGES, None, "high", 2048, "auto", stream=stream)
            except asyncio.CancelledError as error:
                caught_in_receiver.append(error)
                raise

        running = asyncio.create_task(receiver())
        try:
            await asyncio.wait_for(terminal.wait(), 5)
        finally:
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
        assert running.cancelled()
        [error] = caught_in_receiver  # An outer Task await may synthesize a new exception.
        return error

    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="cancel", root_task_id="cancel")):
        cancelled = asyncio.run(exercise())
    [row] = attempt_rows_in_start_order(tmp_path)
    assert len(posts) == 1 and row["state"] == "settled"
    assert cancelled.physical_attempt_capture.attempt_id == row["attempt_id"]
    assert cancelled.response["choices"][0]["message"]["content"] == "391"
    manifest, payload, ref = read_call_payload(
        tmp_path, task_id="cancel", call_id=f'physical_{row["attempt_id"]}_response')
    assert cancelled.response_manifest_ref == ref["manifest_ref"]
    assert manifest["control_reason"] == "caller_cancelled"
    assert payload["response"] == cancelled.response
    assert (row["prompt_tokens"], row["cached_tokens"], row["completion_tokens"]) == (25, 7, 4)
    # The wire receipt proves the complete terminal reply; nothing is ever re-sent to recover it.
    receipt = cancelled.response["_stream_receipt"]
    assert receipt["complete"] is True
    assert row["cost_evidence"]["stream_receipt"] == receipt["manifest_ref"]


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream-prefix", "ndjson-prefix"])
def test_cancel_over_a_partial_prefix_stays_uncertain(install_http, tmp_path, stream, monkeypatch):
    """A genuinely partial byte prefix is never dressed as the paid answer: the
    attempt stays under unknown-outcome custody with its exact bytes retained,
    the cancellation carries no response, and nothing is re-sent."""
    import asyncio
    import base64
    from ouroboros import observability, usage_accounting as ua
    from ouroboros.llm_ollama import chat_ollama_async
    from tests._usage_store_testing import attempt_rows_in_start_order

    server = OllamaHTTP()
    if stream:
        prefix = (json.dumps({"message": {"role": "assistant", "content": "39"}, "done": False}).encode()
                  + b"\n" + b'{"message": {"content": "1"')
    else:
        prefix = b'{"message": {"role": "assistant", "content": "391"}, "done": tr'
    posts, saved = [], []
    write = observability.write_blob
    def capture(root, value, **options):
        if isinstance(value, dict) and "partial_assembly" in value:
            saved.append(value)
        return write(root, value, **options)

    async def exercise():
        delivered = asyncio.Event()

        class HeldPrefix(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield prefix
                delivered.set()
                await asyncio.Future()  # only the caller's cancellation can end the read
                yield b""

        def handle(request):
            if request.url.path == "/api/chat":
                posts.append(json.loads(request.content))
                return httpx.Response(200, stream=HeldPrefix(),
                                      headers={"Content-Type": "application/x-ndjson"})
            return server(request)

        install_http(handle)
        caught_in_receiver = []

        async def receiver():
            try:
                await chat_ollama_async(dict(TARGET), MESSAGES, None, "high", 2048, "auto", stream=stream)
            except asyncio.CancelledError as error:
                caught_in_receiver.append(error)
                raise

        running = asyncio.create_task(receiver())
        try:
            await asyncio.wait_for(delivered.wait(), 5)
        finally:
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
        [error] = caught_in_receiver
        return error

    monkeypatch.setattr(observability, "write_blob", capture)
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="cancel", root_task_id="cancel")):
        cancelled = asyncio.run(exercise())
    [row] = attempt_rows_in_start_order(tmp_path)
    assert len(posts) == 1 and row["state"] == "unresolved"
    assert cancelled.physical_attempt_capture.attempt_id == row["attempt_id"]
    assert not hasattr(cancelled, "response")  # no paid answer was invented
    retained = saved[0]
    assert retained["complete"] is False and retained["model_output_received"]
    assert base64.b64decode(retained["wire_base64"]) == prefix


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("arguments", ['{"broken":', '{"x": NaN,'])
def test_malformed_replayed_arguments_are_terminal_before_dispatch(install_http, asynchronous, arguments):
    from ouroboros.llm_ollama_wire import OllamaUnsupportedRequest
    from ouroboros.loop_llm_call import classify_llm_exception
    server = install_http(OllamaHTTP())
    messages = [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "call1", "type": "function", "function": {"name": "get_value", "arguments": arguments}}]}]
    with pytest.raises(OllamaUnsupportedRequest) as raised:
        chat(asynchronous=asynchronous, messages=messages)
    assert not server.posts
    assert classify_llm_exception(raised.value).retry_same_request is False


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("frame", [None, [], "invalid", 3])
def test_non_object_response_is_terminal_integrity_failure(install_http, asynchronous, stream, frame):
    from ouroboros.llm_ollama_wire import OllamaIntegrityError
    from ouroboros.loop_llm_call import classify_llm_exception
    server = install_http(OllamaHTTP(reply=lambda body: [frame] if body.get("stream") else httpx.Response(200, content=json.dumps(frame))))
    with pytest.raises(OllamaIntegrityError) as raised:
        chat(asynchronous=asynchronous, stream=stream)
    assert len(server.posts) == 1
    assert classify_llm_exception(raised.value).retry_same_request is False

@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("frame", ["unfinished", "done-with-newline", "done-without-newline"])
@pytest.mark.parametrize("stream", [False, True])
def test_reset_preserves_the_original_native_attempt(install_http, tmp_path, asynchronous, frame, stream):
    import base64
    from ouroboros import usage_accounting as ua
    from ouroboros.llm_stream import IncompleteProviderStream
    from ouroboros.observability import read_blob_ref
    from tests._usage_store_testing import attempt_rows_in_start_order

    terminal = frame != "unfinished"
    wire = json.dumps(completed()).encode() if terminal else b'{"message":{"content":"\xd0'
    if frame == "done-with-newline":
        wire += b"\n"
    tail_reads = []
    class Stream(httpx.SyncByteStream, httpx.AsyncByteStream):
        def __iter__(self):
            yield wire
            tail_reads.append(True)
            raise httpx.ReadError("reset")
        async def __aiter__(self):
            for chunk in self:
                yield chunk

    server = install_http(OllamaHTTP(lambda body: httpx.Response(200, stream=Stream())))
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="wire", root_task_id="wire")):
        if terminal:
            message, usage = chat(stream=stream, asynchronous=asynchronous)
            assert message["content"] == "391" and usage["cost"] is None
            receipt = usage["stream_receipt"]
        else:
            with pytest.raises(IncompleteProviderStream) as caught:
                chat(stream=stream, asynchronous=asynchronous)
            from ouroboros.loop_llm_call import classify_llm_exception
            verdict = classify_llm_exception(caught.value)
            assert verdict.kind == "provider_outcome_unknown" and not verdict.retry_same_request
            assert caught.value.retain_partial_reply and caught.value.stream_receipt["model_output_received"]
            receipt = caught.value.stream_receipt
    assert tail_reads == ([] if stream and frame == "done-with-newline" else [True])
    [row] = attempt_rows_in_start_order(tmp_path)
    assert len(server.posts) == 1 and receipt["attempt_id"] == row["attempt_id"]
    assert receipt["complete"] is terminal
    if terminal:
        assert row["cost_evidence"]["stream_receipt"] == receipt["manifest_ref"]
        assert row["prompt_tokens"] == 25 and row["completion_tokens"] == 4
        assert row.get("cost_usd") in (None, 0)
    manifest = json.loads((tmp_path / receipt["manifest_ref"]["path"]).read_text())
    projection = read_blob_ref(tmp_path, manifest["full_payload_ref"])
    evidence = read_blob_ref(tmp_path, projection["private_wire_ref"])
    assert base64.b64decode(evidence["wire_base64"]) == wire
