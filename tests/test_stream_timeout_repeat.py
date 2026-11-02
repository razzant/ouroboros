"""#1886: an explicit provider timeout that ends a configured API child's stream after model output.

Owner decision 21.10.2026 (answer B): one automatic repeat on the same configured model; if that
repeat does not answer, the child ends without another call and its parent decides. Every check here
feeds scripted SSE bytes to the real stream parser inside the real ``LLMClient`` send and its physical
attempt ledger; the round, loop and parent-reader checks run the real retry owner, child loop terminal
and ``get_task_result`` / ``wait_task`` / ``wait_tasks`` consumers. No provider, engine or socket is used.
"""
from __future__ import annotations

import json
import queue
import socket
import time
from types import SimpleNamespace

import pytest

import ouroboros.loop_llm_call as call_mod
from ouroboros import usage_accounting as ua
from ouroboros.llm import LLMClient
from ouroboros.llm_stream import ProviderStreamError, consume_stream
from ouroboros.loop import run_llm_loop
from ouroboros.loop_llm_call import TIMEOUT_REPEAT_KEY, call_llm_with_retry, provider_no_call_source
from ouroboros.tools.registry import ToolRegistry
from tests._usage_store_testing import ledger_rows
from tests._shared import saved_settings

MODEL = "vendor/fixture-model"
CHILD, PARENT = "child-1886", "parent-1886"
DONE = b"data: [DONE]\n\n"
KEEPALIVE = b": OPENROUTER PROCESSING\n\n"


def chunk(delta=None, *, finish=None, usage=None, gen="gen-1"):
    body = {"id": gen}
    if delta is not None or finish is not None:
        body["choices"] = [{"index": 0, "delta": delta or {}, "finish_reason": finish}]
    if usage is not None:
        body["usage"] = usage
    return ("data: " + json.dumps(body) + "\n\n").encode()


def error_frame(code=504, error_type="timeout", gen="gen-1"):
    """OpenRouter's terminal SSE error frame (#1886 recorded it with code 504, error_type timeout)."""
    error = {"message": "Upstream idle timeout exceeded", **({"code": code} if code is not None else {})}
    if error_type:
        error["metadata"] = {"error_type": error_type}
    return ("data: " + json.dumps({"id": gen, "provider": "Upstream", "choices": [], "error": error}) + "\n\n").encode()


def call_start(arguments=""):
    return chunk({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                  "function": {"name": "write_file", "arguments": arguments}}]})


# Typed model output before the frame: each arms the policy.
OUTPUT = {
    "content": [chunk({"role": "assistant", "content": "Partial finding"})],
    "reasoning": [chunk({"role": "assistant", "reasoning": "Comparing the two sources"})],
    "reasoning_text": [chunk({"reasoning_details": [{"type": "reasoning.text", "text": "step one", "index": 0}]})],
    "reasoning_summary": [chunk({"reasoning_details": [{"type": "reasoning.summary", "summary": "plan", "index": 0}]})],
    "encrypted_reasoning": [chunk({"reasoning_details": [
        {"type": "reasoning.encrypted", "id": "rs_1", "format": "fmt", "data": "QUJDREVG", "index": 0}]})],
    "tool_arguments": [call_start(), chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"path": "rep'}}]})],
}
# Containers, identities and keep-alives before the frame: none is model output.
NO_OUTPUT = {
    "nothing": [],
    "role_only": [chunk({"role": "assistant"})],
    "id_only": [chunk()],
    "usage_only": [chunk(usage={"prompt_tokens": 40, "completion_tokens": 0})],
    "keepalives": [KEEPALIVE, KEEPALIVE, KEEPALIVE],
    "empty_content": [chunk({"role": "assistant", "content": ""})],
    "blank_content": [chunk({"content": " \n"})],
    "name_only_tool_start": [call_start()],
    "encrypted_id_only": [chunk({"reasoning_details": [
        {"type": "reasoning.encrypted", "id": "rs_1", "format": "fmt", "index": 0}]})],
}


class Wire:
    """One scripted HTTP stream: the bytes the provider sent, then optionally a socket failure."""

    def __init__(self, *chunks, read_error=None, on_read=None):
        self.chunks, self.read_error, self.headers, self.closed = list(chunks), read_error, {}, 0
        self.on_read = on_read

    def iter_bytes(self):
        if self.on_read is not None:
            self.on_read()
        yield from self.chunks
        if self.read_error is not None:
            raise self.read_error

    def close(self):
        self.closed += 1


def ok_wire(text="Recovered answer.", gen="gen-ok"):
    return Wire(chunk({"role": "assistant", "content": text}, gen=gen),
                chunk({}, finish="stop", usage={"prompt_tokens": 12, "completion_tokens": 3, "cost": 0.01}, gen=gen), DONE)


def timeout_wire(*prefix, code=504, error_type="timeout", gen="gen-1"):
    """One generation's chunks (all carrying ``gen``) ending in the provider's error frame."""
    frames = [frame.replace(b'"id": "gen-1"', f'"id": "{gen}"'.encode()) for frame in (prefix or OUTPUT["reasoning"])]
    return Wire(*frames, error_frame(code, error_type, gen=gen))


class WireLLM(LLMClient):
    """The real client; only the OpenRouter endpoint object is scripted. Each ``create`` is one physical send."""

    def __init__(self, *wires):
        super().__init__(api_key="fixture-key")
        self.wires, self.sent = list(wires), []
        self._endpoint = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self._create)))

    def default_model(self):
        return MODEL

    def _get_remote_client(self, target):
        return self._endpoint

    def _make_no_proxy_client(self, target, timeout=None):
        return self._endpoint, None

    def _create(self, **kwargs):
        self.sent.append(kwargs)
        wire = self.wires.pop(0) if self.wires else ok_wire()
        if isinstance(wire, BaseException):
            raise wire
        return wire


def refused_connect():
    """The OpenAI SDK's shape for a connect that failed before any request byte left the host."""
    import httpx

    try:
        raise RuntimeError("Connection error.") from httpx.ConnectError("connection refused")
    except RuntimeError as exc:
        return exc


@pytest.fixture
def wire_env(tmp_path, monkeypatch):
    """Isolated money and evidence roots, offline pricing and a socket trap."""
    import ouroboros.pricing as pricing
    import ouroboros.request_wire_contract as wire_contract

    dialed = []

    def refuse(*args, **_kwargs):
        dialed.append(args)
        raise OSError("network is not available to this test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(wire_contract, "canonical_wire_evidence_root", lambda: tmp_path)
    monkeypatch.setattr(pricing, "estimate_cost_optional", lambda *a, **k: None)
    monkeypatch.setattr(pricing, "_fetch_live_rows", lambda *a, **k: {})
    monkeypatch.setattr(ua, "_reservation_cost", lambda request: 1.0)
    monkeypatch.setattr(LLMClient, "_get_supported_parameters", lambda *a, **k: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    saved_settings(monkeypatch, tmp_path, OUROBOROS_MODEL_FALLBACKS="")  # a saved blank: no chain
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    (tmp_path / "logs").mkdir()
    # The child bills as its own root here; the parent link is the stored task row's (``deliver_to_parent``).
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id=CHILD, root_task_id=CHILD)):
        yield tmp_path
    assert dialed == []


@pytest.fixture
def sleeps(monkeypatch):
    slept = []
    monkeypatch.setattr(call_mod, "_sleep_within_deadline", lambda sec, _deadline, **_kw: (slept.append(sec), True)[1])
    return slept


def attempts(root):
    return [row for row in ledger_rows(root) if row.get("kind") == "attempt"]


def events(root, kind):
    """``root`` is the drive whose ``logs/events.jsonl`` the call wrote (the child's own drive in loop tests)."""
    path = root / "logs" / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return [row for row in rows if row.get("type") == kind]


def round_call(llm, root, usage, *, child=True, round_idx=1, deadline_ts=None):
    return call_llm_with_retry(llm, [{"role": "user", "content": "research"}], MODEL, None, "low", 3, root / "logs",
                               CHILD, round_idx, None, usage, "task", deadline_ts=deadline_ts, configured_api_child=child)


# ---------------------------------------------------------------- the decoder's typed facts


def _stream_error(*frames):
    with pytest.raises(ProviderStreamError) as caught:
        consume_stream(Wire(*frames))
    return caught.value


@pytest.mark.parametrize("error,expected", [
    ({"code": 504, "message": "Upstream idle timeout exceeded"}, True),
    ({"code": 408}, True),
    ({"code": 502, "metadata": {"error_type": "timeout"}}, True),  # the provider's own typed field
    ({"code": 502, "message": "Upstream idle timeout exceeded"}, False),  # prose is never read
    ({"code": 503, "message": "timeout"}, False),
    ({"code": 429, "message": "timed out"}, False),
    ({"code": "504"}, False),  # not an HTTP-shaped code
    ({"type": "timeout_error", "message": "request ended"}, True),
    ({"type": "api_error", "message": "timeout"}, False),
])
def test_explicit_timeout_reads_only_typed_frame_fields(error, expected):
    assert ProviderStreamError({"error": error}).explicit_timeout is expected


@pytest.mark.parametrize("name", sorted(OUTPUT))
def test_model_output_counts_text_reasoning_encrypted_data_and_tool_arguments(wire_env, name):
    failure = _stream_error(*OUTPUT[name], error_frame())
    assert failure.stream_receipt["model_output_received"] is True and failure.explicit_timeout is True


@pytest.mark.parametrize("name", sorted(NO_OUTPUT))
def test_roles_ids_usage_keepalives_and_empty_starts_are_not_model_output(wire_env, name):
    failure = _stream_error(*NO_OUTPUT[name], error_frame())
    assert failure.stream_receipt["model_output_received"] is False


@pytest.mark.parametrize("events_,expected", [
    ([{"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}], False),
    ([{"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "t", "name": "n", "input": {}}}], False),
    ([{"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
      {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}], True),
    ([{"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "t", "name": "n", "input": {}}},
      {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"a\""}}], True),
])
def test_native_accumulator_applies_the_same_output_rule(events_, expected):
    from ouroboros.llm_stream import AnthropicAccumulator

    accumulator = AnthropicAccumulator()
    for event in [{"type": "message_start", "message": {"id": "m", "content": []}}, *events_]:
        accumulator.accept(event["type"], json.dumps(event))
    assert accumulator.output_received() is expected


def test_durable_error_row_carries_the_typed_facts(wire_env, sleeps):
    llm, usage = WireLLM(timeout_wire(*OUTPUT["content"], gen="gen-first"), ok_wire()), {}
    assert round_call(llm, wire_env, usage)[0]["content"] == "Recovered answer."
    (row,) = events(wire_env, "llm_api_error")
    assert (row["error_kind"], row["status_code"], row["retry_same_request"]) == ("provider_transient", 504, True)
    assert row["explicit_timeout"] is True and row["model_output_received"] is True and row["generation_id"] == "gen-first"
    assert row["physical_attempt_id"] == attempts(wire_env)[0]["attempt_id"]
    assert "Partial finding" not in json.dumps(row)  # identifiers and types only, never the partial text


# ---------------------------------------------------------------- the round's one repeat


def test_first_timeout_then_success_sends_exactly_twice_and_keeps_the_failed_attempt(wire_env, sleeps):
    llm, usage = WireLLM(timeout_wire(), ok_wire()), {}
    msg, _cost = round_call(llm, wire_env, usage)
    assert msg["content"] == "Recovered answer." and len(llm.sent) == 2 and sleeps == [4.0]
    rows = attempts(wire_env)
    assert len({row["attempt_id"] for row in rows}) == 2 and rows[0]["physical_failure"]
    assert rows[-1]["state"] == "settled" and rows[-1]["cost_usd"] == 0.01
    assert TIMEOUT_REPEAT_KEY not in usage and usage["llm_call_refs"][0]["explicit_timeout"] is True
    # The same request on the same model, route and session affinity: nothing about the send changed.
    assert {k: v for k, v in llm.sent[0].items() if k != "timeout"} == {k: v for k, v in llm.sent[1].items() if k != "timeout"}


def test_a_code_less_typed_timeout_frame_earns_the_same_one_repeat(wire_env, sleeps):
    """No HTTP-shaped code (a ``stream_rejected`` provider error) but the provider's typed ``error_type``."""
    llm, usage = WireLLM(timeout_wire(code=None), timeout_wire(code=None, gen="gen-2"), ok_wire()), {}
    assert round_call(llm, wire_env, usage)[0] is None and len(llm.sent) == 2 and len(attempts(wire_env)) == 2
    assert [row["error_kind"] for row in events(wire_env, "llm_api_error")] == ["provider_error", "provider_error"]
    assert provider_no_call_source(usage, False) == ("timeout_repeat_spent_no_resend", True)
    llm, usage = WireLLM(timeout_wire(code=None), ok_wire()), {}  # Main keeps a stream rejection's no-repeat policy
    assert round_call(llm, wire_env, usage, child=False)[0] is None and len(llm.sent) == 1
    assert TIMEOUT_REPEAT_KEY not in usage and provider_no_call_source(usage, False) == ("", False)


def test_a_timeout_on_the_last_ordinary_attempt_still_hands_over_without_a_call(wire_env, sleeps):
    llm, usage = WireLLM(*[Wire(error_frame(502, None)) for _ in range(5)], timeout_wire(), ok_wire()), {}
    assert round_call(llm, wire_env, usage)[0] is None and len(llm.sent) == 6
    assert usage[TIMEOUT_REPEAT_KEY]["budget_spent"] is True
    assert usage[call_mod.RETRY_ATTEMPTS_SPENT_KEY] is True  # six real attempts: an honest series
    assert provider_no_call_source(usage, True)[0] == "timeout_repeat_spent_no_resend"


@pytest.mark.parametrize("name", sorted(OUTPUT))
def test_timeout_after_any_model_output_then_timeout_sends_twice_and_hands_over(wire_env, sleeps, name):
    llm, usage = WireLLM(timeout_wire(*OUTPUT[name]), timeout_wire(*OUTPUT[name], gen="gen-2"), ok_wire()), {}
    assert round_call(llm, wire_env, usage) == (None, 0.0)
    assert len(llm.sent) == 2 and len(attempts(wire_env)) == 2
    assert provider_no_call_source(usage, False) == ("timeout_repeat_spent_no_resend", True)
    assert provider_no_call_source(usage, True)[0] == "timeout_repeat_spent_no_resend"  # no deadline grace call
    assert usage.get(call_mod.RETRY_ATTEMPTS_SPENT_KEY) is not True  # two attempts claim no spent series


@pytest.mark.parametrize("second", ["pre_output_502", "role_only_504", "unknown_eof", "bad_request"])
def test_a_failed_repeat_keeps_its_own_class_and_never_sends_a_third_time(wire_env, sleeps, second):
    repeat = {
        "pre_output_502": Wire(error_frame(502, None)),
        "role_only_504": Wire(*NO_OUTPUT["role_only"], error_frame(504, None)),
        "unknown_eof": Wire(*OUTPUT["content"]),  # the wire ends without terminal framing
        "bad_request": Wire(*OUTPUT["content"], error_frame(400, None)),
    }[second]
    llm, usage = WireLLM(timeout_wire(), repeat, ok_wire(), ok_wire()), {}
    assert round_call(llm, wire_env, usage)[0] is None and len(llm.sent) == 2
    kind = usage["_last_llm_error_kind"]
    assert kind == {"pre_output_502": "provider_transient", "role_only_504": "provider_transient",
                    "unknown_eof": "provider_outcome_unknown", "bad_request": "bad_request"}[second]
    source = provider_no_call_source(usage, False)[0]
    assert source == ("provider_outcome_unknown_no_resend" if second == "unknown_eof" else "timeout_repeat_spent_no_resend")


@pytest.mark.parametrize("name", ["nothing", "role_only", "id_only", "usage_only", "name_only_tool_start"])
def test_a_timeout_before_model_output_keeps_the_ordinary_transient_budget(wire_env, sleeps, name):
    """Six real sends each; every other pre-output shape is pinned by the decoder test above."""
    llm, usage = WireLLM(*[Wire(*NO_OUTPUT[name], error_frame()) for _ in range(6)]), {}
    assert round_call(llm, wire_env, usage)[0] is None
    assert len(llm.sent) == 6 and TIMEOUT_REPEAT_KEY not in usage
    assert provider_no_call_source(usage, False)[0] == "retry_wall_exhausted_no_repay"


@pytest.mark.parametrize("case", ["untyped_503_after_output", "pre_output_502", "main_route", "unknown_eof_first"])
def test_other_failures_and_other_actors_keep_their_existing_policy(wire_env, sleeps, case):
    if case == "untyped_503_after_output":  # a 5xx without a typed timeout stays an ordinary transient
        wires, child, sends = [Wire(*OUTPUT["content"], error_frame(503, None)) for _ in range(6)], True, 6
    elif case == "pre_output_502":
        wires, child, sends = [Wire(error_frame(502, None)) for _ in range(6)], True, 6
    elif case == "main_route":  # Main (and every non-API actor) keeps its six same-model attempts
        wires, child, sends = [timeout_wire() for _ in range(6)], False, 6
    else:  # an initial unknown outcome is never repeated as the same request
        wires, child, sends = [Wire(*OUTPUT["content"])], True, 1
    llm, usage = WireLLM(*wires), {}
    assert round_call(llm, wire_env, usage, child=child)[0] is None
    assert len(llm.sent) == sends and TIMEOUT_REPEAT_KEY not in usage


def test_the_deadline_can_refuse_the_repeat_and_still_hands_over(wire_env):
    """The real backoff gate: a deadline that cannot absorb 4 s plus the retry floor sends nothing more."""
    llm, usage = WireLLM(timeout_wire(), ok_wire()), {}
    assert round_call(llm, wire_env, usage, deadline_ts=time.time() + 6)[0] is None
    assert len(llm.sent) == 1 and usage[TIMEOUT_REPEAT_KEY]["first"]["explicit_timeout"] is True
    assert provider_no_call_source(usage, True)[0] == "timeout_repeat_spent_no_resend"
    assert [row["error_kind"] for row in events(wire_env, "llm_retry_deadline_exhausted")] == ["provider_transient"]


def test_a_budget_refusal_of_the_repeat_propagates_to_the_budget_rail(wire_env, sleeps, monkeypatch):
    """An unresolved bound is exposure, not spend (#1487): another task's known charge fills the wallet."""
    monkeypatch.setenv("TOTAL_BUDGET", "1.5")

    def known_spend_lands():
        ua.record_subscription_session("other-session", drive_root=wire_env, route="subscription",
                                       task_id="other", root_task_id="other", spend_usd=1.5)

    llm = WireLLM(Wire(*OUTPUT["reasoning"], error_frame(), on_read=known_spend_lands), ok_wire())
    with pytest.raises(ua.BudgetExceeded):
        round_call(llm, wire_env, {})
    assert len(llm.sent) == 1 and [row["task_id"] for row in attempts(wire_env)].count(CHILD) == 1


def test_only_a_usable_response_resets_the_allowance_for_a_later_round(wire_env, sleeps):
    llm, usage = WireLLM(timeout_wire(), ok_wire("first"), timeout_wire(gen="gen-3"), ok_wire("second")), {}
    assert round_call(llm, wire_env, usage, round_idx=1)[0]["content"] == "first"
    assert round_call(llm, wire_env, usage, round_idx=2)[0]["content"] == "second"
    assert len(llm.sent) == 4 and TIMEOUT_REPEAT_KEY not in usage
    stale = {TIMEOUT_REPEAT_KEY: {"round_id": "other-execution:round:9", "first": {}}}
    llm, usage = WireLLM(timeout_wire(gen="gen-5"), ok_wire("third")), {**stale, "execution_id": "e"}
    assert round_call(llm, wire_env, usage, round_idx=3)[0]["content"] == "third" and len(llm.sent) == 2


# ---------------------------------------------------------------- the child loop and its parent


def child_drive(root):
    """The child's own execution drive in the host layout (``task_custody.own_child_drives``)."""
    drive = root / "task_drives" / CHILD
    (drive / "logs").mkdir(parents=True, exist_ok=True)
    return drive


def child_loop(root, llm, *, route_kind="api_model", exact=True, prior=None):
    drive = child_drive(root)
    registry = ToolRegistry(repo_dir=root, drive_root=drive)
    registry._ctx.exact_model_route = exact
    registry._ctx._configured_subagent_route_kind = route_kind  # set by subagent_bootstrap for a configured row
    registry._ctx.task_model_override = MODEL
    notes = []
    messages = [{"role": "user", "content": "Research the question and report."}, *(prior or [])]
    result = run_llm_loop(messages=messages, tools=registry, llm=llm, drive_logs=drive / "logs",
                          emit_progress=lambda text, **_kw: notes.append(text), incoming_messages=queue.Queue(),
                          task_type="task", task_id=CHILD, drive_root=drive)
    return (*result, notes)


def parent_tools(root):
    from ouroboros.task_results import write_task_result

    write_task_result(root, PARENT, "running", root_task_id=PARENT)  # the lineage query_data checks
    tools = ToolRegistry(root / "parent-repo", root / "parent-execution")
    tools._ctx.task_id = PARENT
    tools._ctx.task_metadata = {"budget_drive_root": str(root), "root_task_id": PARENT}
    return tools


def parent_reads_error_rows(root):
    """The notice's own locator, through the parent's real ``query_data`` tool."""
    result = parent_tools(root).execute_result("query_data", {
        "op": "jsonl", "kind": "events_log", "child_task_id": CHILD,
        "filter": {"field": "type", "op": "eq", "value": "llm_api_error"}})
    assert result.status == "ok", result.text
    return json.loads(result.text)["data"]


def deliver_to_parent(root, monkeypatch, text, usage, trace):
    """The real child terminal: ``emit_task_results`` stores the row the parent's readers project."""
    from ouroboros import agent_task_pipeline as pipeline

    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *_a, **_kw: None)
    task = {"id": CHILD, "type": "task", "chat_id": 1, "text": "Research the question and report.",
            "parent_task_id": PARENT, "root_task_id": PARENT, "delegation_role": "subagent"}
    pipeline.emit_task_results(SimpleNamespace(drive_root=root, repo_dir=root), None, None, [], task, text, usage,
                               trace, start_time=0.0, drive_logs=root / "logs")


def parent_views(root):
    from ouroboros.tools.control_task_results import _get_task_result, _wait_for_task
    from tests.test_child_result_disposition import _parent_ctx

    parent = _parent_ctx(root, PARENT)
    batch = parent_tools(root).execute_result("wait_tasks", {"task_ids": [CHILD], "timeout_sec": 0, "mode": "all_terminal"})
    assert batch.status == "ok"
    return _get_task_result(parent, CHILD), _wait_for_task(parent, CHILD, timeout_sec=0), json.loads(batch.text)["tasks"][CHILD]


def test_child_hands_its_parent_the_decision_with_saved_work_and_no_tool_or_summary_call(wire_env, sleeps, monkeypatch):
    """Timeout after partial tool arguments, then the repeat times out: two sends, no tool, no forced final."""
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.tools.join_ledger import _child_result_sha256

    (task_artifact_dir_path(wire_env, CHILD, create=True) / "notes.md").write_text("earlier deliverable\n")
    prior = [{"role": "assistant", "content": "Draft finding: source A disagrees with B."},
             {"role": "user", "content": "Continue."}]
    llm = WireLLM(timeout_wire(*OUTPUT["tool_arguments"], gen="gen-a"), timeout_wire(*OUTPUT["tool_arguments"], gen="gen-b"))
    text, usage, trace, _notes = child_loop(wire_env, llm, prior=prior)

    assert len(llm.sent) == 2 and trace["tool_calls"] == [] and not (wire_env / "rep").exists()
    assert trace["forced_finalization"]["source"] == "timeout_repeat_spent_no_resend"
    assert text == "Draft finding: source A disagrees with B."  # saved work, never the partial stream
    assert (usage["execution_status"], usage["reason_code"]) == ("infra_failed", "provider_unavailable")
    ids = [row["attempt_id"] for row in attempts(wire_env)]
    notice = usage["terminal_host_notice"]
    assert len(ids) == 2 and [row["state"] for row in attempts(wire_env)] == ["unresolved", "unresolved"]  # cost kept unknown
    assert "explicit timeout (status 504)" in notice and "It was repeated once" in notice
    assert "failed as provider_transient" in notice and "not adopted as an answer" in notice
    assert all(attempt_id in notice for attempt_id in ids) and "gen-a" in notice and "gen-b" in notice
    assert f'ids=["{CHILD}"]' in notice and "The parent task decides what follows." in notice
    assert "write_file" not in notice and '"path"' not in notice  # never the partial tool call

    deliver_to_parent(wire_env, monkeypatch, text, usage, trace)
    full, waited, compact = parent_views(wire_env)
    for view in (full, waited):
        assert "Draft finding: source A disagrees with B." in view and view.endswith("[Host status]\n" + notice)
        assert '"reason_code": "provider_unavailable"' in view and "notes.md" in view
    assert compact["terminal_host_notice"] == notice and compact["status"] == "failed"
    rows = parent_reads_error_rows(wire_env)  # the notice's locator opens the attempts' own typed rows
    assert [row["physical_attempt_id"] for row in rows] == ids and [row["generation_id"] for row in rows] == ["gen-a", "gen-b"]
    assert all(row["explicit_timeout"] and row["model_output_received"] for row in rows)
    stored = load_effective_task_result(wire_env, CHILD)
    assert compact["child_result_sha256"] == _child_result_sha256(stored)
    # The failure facts are part of the result identity; late money is not.
    assert _child_result_sha256({**stored, "terminal_host_notice": ""}) != _child_result_sha256(stored)
    assert _child_result_sha256({**stored, "accounted_upper_bound_usd": 9.0, "cost_final": True}) == _child_result_sha256(stored)


def test_new_child_without_saved_work_fails_with_host_words_and_the_typed_notice(wire_env, sleeps, monkeypatch):
    llm = WireLLM(timeout_wire(*OUTPUT["content"]), Wire(*OUTPUT["content"]))  # timeout, then an unknown repeat
    text, usage, trace, _notes = child_loop(wire_env, llm)
    assert len(llm.sent) == 2 and trace["forced_finalization"]["source"] == "provider_outcome_unknown_no_resend"
    assert "Partial finding" not in text and usage["terminal_origin"] == "host_notice"
    notice = usage["terminal_host_notice"]
    assert "the repeat has no confirmed provider outcome" in notice and "failed as" not in notice
    assert events(child_drive(wire_env), "network_wait") == []  # no wait episode, no new generation after it
    deliver_to_parent(wire_env, monkeypatch, text, usage, trace)
    full, _waited, compact = parent_views(wire_env)
    assert compact["status"] == "failed" and compact["terminal_host_notice"] == notice and notice in full


@pytest.mark.parametrize("repeat,said", [
    (refused_connect(), "was not sent (transport_unavailable)"),  # released before dispatch: never reads as sent
    (timeout_wire(code=None, gen="gen-2"), "It was repeated once"),  # a code-less typed timeout, end to end
])
def test_the_notice_reads_the_repeats_physical_receipt_and_no_summary_is_bought(wire_env, sleeps, repeat, said):
    first = timeout_wire(code=None) if isinstance(repeat, Wire) else timeout_wire()
    llm = WireLLM(first, repeat, ok_wire("a third generation"))
    text, usage, trace, _notes = child_loop(wire_env, llm)
    assert len(llm.sent) == 2 and "a third generation" not in text and trace["tool_calls"] == []
    assert trace["forced_finalization"]["source"] == "timeout_repeat_spent_no_resend"
    notice = usage["terminal_host_notice"]
    assert said in notice and ("typed as a timeout" in notice) is isinstance(repeat, Wire)
    states = [row["state"] for row in attempts(wire_env)]
    assert states[0] == "unresolved" and states[-1] == ("unresolved" if isinstance(repeat, Wire) else "released")


def test_child_whose_repeat_answers_continues_normally(wire_env, sleeps):
    text, usage, trace, _notes = child_loop(wire_env, WireLLM(timeout_wire(), ok_wire("Final report.")))
    assert text == "Final report." and "terminal_host_notice" not in usage and TIMEOUT_REPEAT_KEY not in usage
    assert usage.get("reason_code") is None and "forced_finalization" not in trace


@pytest.mark.parametrize("route_kind,exact", [("agent_session", True), ("", False)], ids=["session_nanny", "main"])
def test_nanny_and_main_rounds_keep_the_six_attempt_burst(wire_env, sleeps, route_kind, exact):
    llm = WireLLM(*[timeout_wire() for _ in range(6)])
    _text, usage, trace, _notes = child_loop(wire_env, llm, route_kind=route_kind, exact=exact)
    assert len(llm.sent) == 6 and TIMEOUT_REPEAT_KEY not in usage and "terminal_host_notice" not in usage
    assert trace["forced_finalization"]["source"] == "retry_wall_exhausted_no_repay"
