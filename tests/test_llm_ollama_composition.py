"""Native Ollama shares the chat reasoning reader and Nano schema bytes."""
import pytest

from tests._ollama import TARGET, MESSAGES, OllamaHTTP, chat, completed, install_http as install_http


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("content,thinking,answer,reasoning,parsed", [
    ("<think>plan</think>\n391", "", "391", "plan", True),
    ("plan\n</think>\n391", "", "391", "plan", True),
    ("391", "native plan", "391", "native plan", False),
    ("<think>quoted</think>", "native plan", "<think>quoted</think>", "native plan", False),
    ("Here is code:\n```\n</think>\n```", "", "Here is code:\n```\n</think>\n```", "", True),
])
def test_native_reasoning_uses_the_shared_reader(
        monkeypatch, install_http, asynchronous, stream, content, thinking, answer, reasoning, parsed):
    from ouroboros import llm_local
    calls, reader = [], llm_local._split_leading_reasoning

    def read(text):
        calls.append(text)
        return reader(text)

    monkeypatch.setattr(llm_local, "_split_leading_reasoning", read)
    message = {"role": "assistant", "content": content, "thinking": thinking}
    reply = completed(message=message)
    if stream:
        # Both carriers span physical chunks; parsing happens after assembly.
        reply = [{"message": {"content": content[:5], "thinking": thinking[:3]}, "done": False},
                 completed(message={"content": content[5:], "thinking": thinking[3:]})]
    install_http(OllamaHTTP(reply))
    result, _ = chat(asynchronous=asynchronous, stream=stream)
    assert result["content"] == answer
    assert result.get("reasoning", "") == reasoning
    assert calls == ([content] if parsed else [])


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_native_tool_call_history_keeps_bytes_and_projects_display(monkeypatch, install_http, asynchronous, stream):
    from ouroboros import llm_local
    from ouroboros.llm_messages import DISPLAY_KEY

    calls, reader = [], llm_local._split_leading_reasoning
    def read(text):
        calls.append(text)
        return reader(text)
    monkeypatch.setattr(llm_local, "_split_leading_reasoning", read)
    raw = "plan\n</think>\nReading the file."
    message = {"role": "assistant", "content": raw,
               "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "test.txt"}}}]}
    reply = completed(message=message)
    if stream:
        reply = [{"message": {"content": raw[:5]}, "done": False},
                 completed(message={**message, "content": raw[5:]})]
    install_http(OllamaHTTP(reply))
    result, _ = chat(asynchronous=asynchronous, stream=stream)
    assert result["content"] == raw and result["tool_calls"]
    assert result[DISPLAY_KEY] == {"text": "Reading the file.", "reasoning": "plan"}
    assert calls == [raw]


def test_nano_schemas_are_identical_on_both_wire_builders(tmp_path):
    from ouroboros.llm import LLMClient
    from ouroboros.llm_ollama import _prepare
    from ouroboros.tool_policy import initial_tool_schemas, NANO_RESIDENT_SCHEMA_NAMES
    from ouroboros.tools.registry import ToolRegistry

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    schemas = initial_tool_schemas(registry, "nano")
    assert "finish_task" in {s["function"]["name"] for s in schemas}
    assert {s["function"]["name"] for s in schemas} <= NANO_RESIDENT_SCHEMA_NAMES
    compatible = LLMClient(api_key="x")._build_remote_kwargs(
        dict(TARGET), MESSAGES, "high", 2048, "auto", None, schemas, skip_capability_fetch=True)
    native, _ = _prepare(dict(TARGET), MESSAGES, schemas, "high", 2048, "auto", None, False, None, 80_000)
    assert native["tools"] == compatible["tools"]


@pytest.mark.parametrize("as_json", [False, True])
def test_replayed_tool_arguments_match_ledger_and_wire(install_http, tmp_path, as_json):
    import hashlib
    import json
    from ouroboros import usage_accounting as ua
    from ouroboros.llm_attempt import _canonical_candidate_bytes
    from tests._usage_store_testing import attempt_rows_in_start_order

    server = install_http(OllamaHTTP())
    args = {"_context_capsule": {"rows": [{"_context_capsule": "owner value", "cache_control": "user value"}]}}
    call = {"id": "one", "type": "function", "function": {"name": "run", "_display": "host",
                                                          "arguments": json.dumps(args) if as_json else args}}
    history = [*MESSAGES, {"role": "assistant", "tool_calls": [call]},
               {"role": "tool", "tool_call_id": "one", "content": "ok"}]
    scope = ua.UsageScope(drive_root=tmp_path, task_id="sample", source="review_substrate.extraction")
    with ua.usage_scope(scope):
        chat(messages=history)
    (sent,) = server.posts
    assert sent["messages"][1]["tool_calls"][0]["function"] == {"name": "run", "arguments": args}
    assert call["function"] == {"name": "run", "_display": "host", "arguments": json.dumps(args) if as_json else args}
    (attempt,) = attempt_rows_in_start_order(tmp_path)
    assert attempt["source"] == scope.source
    assert attempt["candidate_raw_sha256"] == hashlib.sha256(_canonical_candidate_bytes(sent)).hexdigest()


@pytest.mark.parametrize("resume", ["same", "legacy", "changed_endpoint", "pending_role"])
def test_cold_resume_and_admitted_forced_send_keep_the_scoped_ollama_load(install_http, tmp_path, monkeypatch, resume):
    import copy
    import json
    from hashlib import sha256
    from ouroboros import config, context_fit as cf, owner_wait, task_pacing, usage_accounting as ua
    from ouroboros import loop_forced_finalization as forced
    from ouroboros.llm import LLMClient
    from ouroboros.llm_attempt import _canonical_candidate_bytes
    from ouroboros.loop_round_limits import _RoundLimitContext
    from datetime import datetime, timezone
    from ouroboros import send_clock
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.registry import ToolRegistry
    from tests._ollama import _compatible_target

    when = datetime(2030, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    monkeypatch.setattr(send_clock, "_now", lambda: when)
    model = _compatible_target(monkeypatch, TARGET["base_url"])["capability_route"]["model"]
    settings = config.runtime_settings()
    # Exercise two roles sharing one physical Ollama runner.
    settings.update(OUROBOROS_MODEL_CONTEXT_WINDOWS={"main": 60_000, "light": 80_000}, OUROBOROS_CONTEXT_MODE="max")
    settings.update(OUROBOROS_MODEL_LIGHT=model, USE_LOCAL_LIGHT=False)
    monkeypatch.setenv("OUROBOROS_MODEL", model)
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", model)
    monkeypatch.setenv("USE_LOCAL_MODEL", "false")
    monkeypatch.setenv("USE_LOCAL_LIGHT", "false")
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "capability-evidence")
    server = OllamaHTTP()
    clock_at_post = []

    def handle(request):
        if request.method == "POST" and request.url.path == "/api/chat":
            wire = json.loads(request.content)
            note, split = send_clock.split_clock_note(wire)
            clock_at_post.append((note, split))
        return server(request)

    install_http(handle)
    task = {"model": model, "use_local_model": False, "model_role": "main"}
    core = cf.ContextCore("SYSTEM", "BIBLE", "", "", "Identity", "Facts", '"Finish"', False)

    def plan(resolver=cf.resolve_context_fit_route):
        return cf.build_context_fit_plan(None, core, task, preferred_mode="max", tool_schemas=[],
                                         reasoning_effort="high", route_resolver=resolver)

    old = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)._ctx
    old.task_id, old.task_metadata, old.task_attempt = "resume", {}, 1
    old.active_model, old.active_use_local, old.active_effort = model, False, "high"
    old.active_context_mode, old.context_fit_plan = "max", plan()
    old.primary_route = {"model": model, "use_local": False, "role": "main"}
    assert old.context_fit_plan.requested_load_window == 80_000
    assert old.context_fit_plan.window_record.total.tokens == 60_000
    messages = old.context_fit_plan.messages_for("max") + [{"role": "system", "content": "Runtime notice"}]
    state = json.loads(json.dumps(owner_wait.continuation_state(old, messages, {}, {}, 2, [], set())))
    assert state["load_window_binding"] == {"scope": old.context_fit_plan.window_record.route_fp,
                                           "role": "main", "load": 80_000, "cap": 60_000}
    if resume == "legacy":
        state.pop("load_window_binding")
    if resume == "changed_endpoint":
        settings["OPENAI_COMPATIBLE_BASE_URL"] = "http://127.0.0.1:11436/v1"
    if resume == "pending_role":
        state["route"]["active_role_override"] = "light"
    settings["OUROBOROS_MODEL_CONTEXT_WINDOWS"] = {"main": 110_000, "light": 120_000}

    write_task_result(tmp_path, old.task_id, "running", root_task_id=old.task_id, _attempt=1)
    tools = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    restored = tools._ctx
    restored.task_id, restored.task_attempt, restored.task_metadata = old.task_id, 1, {}
    restored.active_model, restored.active_use_local = model, False
    resolver, _ = owner_wait.context_start_route(restored, task, state, "high")
    restored.context_fit_plan = plan(resolver)
    expected_load = 80_000 if resume in {"same", "pending_role"} else 120_000
    expected_cap = 60_000 if expected_load == 80_000 else 110_000
    messages = []
    owner_wait.restore_continuation_state(tools, state, messages, {}, {}, set())
    startup = restored.context_fit_plan
    fitted, mode = owner_wait.rebind_restored_route(tools, state, messages)
    if resume != "pending_role":
        assert all(p.requested_load_window == expected_load and p.window_record.total.tokens == expected_cap
                   for p in (startup, fitted))
    else:
        # A later ordinary-role choice may have a 120k preview; forced wrap-up is still the saved actor.
        assert fitted.model_role == "light" and fitted.requested_load_window == 120_000
        restored.active_role_override = None
        restored._restored_context_model_role = "main"
    ctx = _RoundLimitContext(messages=messages, llm=LLMClient(), active_model=model, active_effort="high", max_retries=1,
        drive_logs=tmp_path / "logs", task_id=old.task_id, round_idx=2, event_queue=None, accumulated_usage={},
        task_type="task", active_use_local=False, max_rounds=2, drive_root=tmp_path, tools=tools, tool_schemas=[])
    restored.active_context_mode = mode
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id=old.task_id, root_task_id=old.task_id)):
        forced._bind_forced_route(ctx)
        assert restored.context_fit_plan.requested_load_window == expected_load
        assert restored.context_fit_plan.window_record.total.tokens == expected_cap
        original = copy.deepcopy(messages)
        request, prepared = task_pacing.prepared_wrapup_candidate(ctx, messages, allow_server_web_search=False)
        assert forced._call_forced_model_once(ctx, initial_messages=prepared, admitted_request=request) == "391"
        capture = ua.last_physical_attempt_capture()
    [post] = server.posts
    assert post["options"]["num_ctx"] == expected_load
    assert {"role": "user", "content": "[SYSTEM NOTICE]\nRuntime notice"} in post["messages"]
    assert post["messages"][-1] == {"role": "user", "content": "[Host clock at this request: "
                                  "2030-01-02T03:04:05Z UTC; sender's time zone unknown]"}
    clock_free = {**post, "messages": post["messages"][:-1]}
    assert clock_at_post == [(post["messages"][-1]["content"], clock_free)]
    assert capture.candidate_raw_sha256 == sha256(_canonical_candidate_bytes(post)).hexdigest()
    assert request.candidate_clock_free_sha256 == sha256(_canonical_candidate_bytes(clock_free)).hexdigest()
    assert {"role": "system", "content": "Runtime notice"} in original
