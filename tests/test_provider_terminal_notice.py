"""Outage facts reach real delivery projections without rewriting model sources."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import agent_task_pipeline as pipeline, cancel_intents, loop, loop_llm_call, loop_transport
from ouroboros.gateway import host_service
from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, write_owner_message
from ouroboros.presence_runner import PresenceTurnError, PresenceTurnGate, presence_result_from_stored, run_presence_turn
from ouroboros.task_finalization import send_provider_death_notice
from ouroboros.task_results import load_task_result
from ouroboros.tools.registry import ToolRegistry
from supervisor.owner_stop import owner_stop_control_id
from supervisor.terminal_delivery import build_completed_result_event, pending_deliveries
from tests.test_delivery_forced_finalization import _forced_test_context
from tests.test_loop_transport_wait import _loop_kwargs, _read_network_wait_events
from tests.test_presence_runner import _admission, _event


RAW = "Exact model source: λ\n\nA useful intermediate result."


def _terminal(tmp_path, *, current, task_id="parent1"):
    usage = {"_last_llm_error_kind": "provider_outcome_unknown",
             loop_llm_call.TRANSPORT_DEATHS_KEY: {"round_id": "round", "count": 1,
                                                "error_kind": "provider_outcome_unknown"}}
    _loop, registry, ctx, trace = _forced_test_context(tmp_path, usage=usage)
    ctx.task_id = registry._ctx.task_id = task_id
    registry._ctx.task_metadata["root_task_id"] = task_id
    if current:
        loop._replace_delivery_candidate(registry, ctx, trace, RAW, control="replace")
    else:
        ctx.messages.append({"role": "assistant", "content": RAW})
    text, usage, trace = loop._handle_provider_unavailable(ctx, error_kind="provider_outcome_unknown",
        wait_cause="transport_unavailable", waited_sec=125.0)
    assert text == RAW
    assert "spent 2.1 min in the provider wait" in usage["terminal_provider_notice"]
    assert "no confirmed provider outcome" in usage["terminal_provider_notice"]
    assert "waited and redialed" not in usage["terminal_provider_notice"]
    return text, usage, trace


@pytest.mark.parametrize("mode", ["managed", "direct"])
@pytest.mark.parametrize("current", [False, True])
def test_pipeline_delivery_and_rebuild_keep_raw_bytes_and_known_wait_custody(tmp_path, monkeypatch, mode, current):
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *_a, **_k: None)
    text, usage, trace = _terminal(tmp_path, current=current)
    task = {"id": "parent1", "type": "task", "chat_id": 7, "text": "finish the task"}
    if mode != "managed":
        task["_is_direct_chat"] = True
    pending = []
    pipeline.emit_task_results(SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path), None, None,
        pending, task, text, usage, trace, start_time=0.0, drive_logs=tmp_path / "logs")
    sent = next(row for row in pending if row["type"] == "send_message")
    notice = usage["terminal_provider_notice"]
    stored = load_task_result(tmp_path, "parent1")
    assert stored["result"] == RAW and stored["terminal_provider_notice"] == notice
    assert stored["status"] == "failed"  # same provider-outage category
    if current:
        assert sent["text"] == RAW
    else:
        assert RAW not in sent["text"] and notice in sent["text"]
        assert Path(stored["terminal_salvage_path"]).read_text(encoding="utf-8") == RAW
    replay = build_completed_result_event(tmp_path, task, "parent1", stored)
    assert replay["text"] == sent["text"] and replay["delivery_id"] == sent["delivery_id"]
    assert pending_deliveries(tmp_path)[0]["text"] == sent["text"]
    incidents = []
    notified = send_provider_death_notice(SimpleNamespace(send_with_budget=lambda *a, **k: incidents.append((a, k))),
                                          7, "parent1", stored)
    assert notified is current
    if current:
        assert notice in incidents[0][0][1]
        assert "re-run the task" not in incidents[0][0][1]
        assert "Retry when connectivity returns" not in incidents[0][0][1]


@pytest.mark.parametrize("outcome", ["message", "deferred", "silent", "tool_delivered"])
@pytest.mark.parametrize("current", [False, True])
def test_actual_presence_and_cached_read_keep_authored_speech_and_owner_notice_separate(tmp_path, monkeypatch, outcome, current):
    """An unresolved dispatched attempt is a typed empty 409 on the first call and on replay.

    The forced rail still words the durable row ``provider_unavailable`` and keeps the authored
    draft as its result; the pipeline stamps the loop's own no-resend predicate on that row and
    the Host guard reads the marker back. Neither RAW nor the owner notice becomes speech,
    whatever outcome the envelope claims, and the admitted child keeps its custody.
    """
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *_a, **_k: None)
    notices = []
    monkeypatch.setattr("ouroboros.presence_runner._write_unresolved_notice",
                        lambda _root, task_id: notices.append(task_id))
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    created, invoked = [], []
    class Agent:
        def handle_task(self, task):
            invoked.append(task["id"])
            text, usage, trace = _terminal(data, current=current, task_id=task["id"])
            task["_skip_post_task_synthesis"] = True
            ctx = SimpleNamespace(_presence_completion={"outcome": outcome, "message": "Typed Presence reply"},
                                  _swarm_handoff_attempt={"status": "scheduled", "task_id": "next-task"})
            pending = []
            pipeline.emit_task_results(SimpleNamespace(drive_root=data, repo_dir=repo), None, None,
                pending, task, text, usage, trace, start_time=0.0, drive_logs=data / "logs", ctx=ctx)
            return pending
    def factory(**kwargs):
        created.append(kwargs)
        return Agent()
    args = dict(admission=_admission(), event=_event(), repo_dir=repo, drive_root=data,
                agent_factory=factory, gate=PresenceTurnGate(2))
    with pytest.raises(PresenceTurnError) as first:
        run_presence_turn(**args)
    with pytest.raises(PresenceTurnError) as replay:
        run_presence_turn(**args)
    task_id = first.value.turn_ref
    assert invoked == [task_id] and len(created) == 1  # replay never regenerates
    stored = load_task_result(data, task_id)
    notice = stored["terminal_provider_notice"]
    assert stored["result"] == RAW and "no terminal provider outcome" in notice
    marker = stored["metadata"]["presence_unknown_outcome"]
    assert marker["source"] == "provider_outcome_unknown_no_resend"
    assert marker["error_kind"] == "provider_outcome_unknown"
    assert "presence_retry_proof" not in stored["metadata"]  # unknown is not not_started
    assert stored["metadata"]["presence_work_ref"] == "next-task"
    assert stored["metadata"]["presence_result_text"] == (RAW if current else "")  # the envelope, not speech
    for err in (first.value, replay.value):
        assert err.code == "presence_attempt_outcome_unknown"
        assert (err.turn_ref, err.work_ref) == (task_id, "next-task")  # admitted child custody preserved
        response = host_service._presence_exception(err, 409)
        body = json.loads(response.body)
        assert response.status_code == 409 and body["code"] == "presence_attempt_outcome_unknown"
        assert body["disposition"] == "retry" and not body.get("text") and "outcome" not in body
        assert body["error"] == "presence_attempt_outcome_unknown: source_event_id"
        assert (body["turn_ref"], body["work_ref"]) == (task_id, "next-task")
        assert all(RAW not in str(value) and notice not in str(value) for value in body.values())
    # each refusal consults the owner-notice writer (mocked here; production dedups on
    # presence_recovery_owner_notified, so the owner hears it once)
    assert notices == [task_id] * 2
    view = presence_result_from_stored(stored, task_id)
    assert (view.outcome, view.text, view.work_ref) == ("silent", "", "next-task")


@pytest.mark.parametrize("reason", [REASON_OWNER_REQUESTED_FINALIZATION, "deadline", "budget ceiling reached"])
def test_real_mailbox_wait_exit_carries_the_control_cause_without_another_call(tmp_path, monkeypatch, reason):
    calls = []
    def fail(_llm, _messages, _model, _tools, _effort, _retries, _logs, _tid, _round, _queue, usage, *_a, **_k):
        calls.append(_tid)
        mid = "control"
        if reason == REASON_OWNER_REQUESTED_FINALIZATION:
            intent = cancel_intents.request_cancel(tmp_path, "t-wait", requested_stop_policy=cancel_intents.STOP_POLICY_FINALIZE)
            mid = owner_stop_control_id(intent)
        write_owner_message(tmp_path, reason, "t-wait", msg_id=mid, kind=KIND_FINALIZE_NOW)
        usage["_last_llm_error_kind"] = "transport_unavailable"
        return None, 0.0
    monkeypatch.setattr(loop, "call_llm_with_retry", fail)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    text, usage, trace = loop.run_llm_loop(**_loop_kwargs(tmp_path, ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path), []))
    assert len(calls) == 1
    assert usage["execution_status"] == "infra_failed" and usage["reason_code"] == "provider_unavailable"
    assert trace["forced_finalization"]["source"] == "transport_unavailable_no_resend"
    assert _read_network_wait_events(tmp_path)[-1]["detail"] == "finalize_now"
    if reason == REASON_OWNER_REQUESTED_FINALIZATION:
        assert "owner requested Wrap up" in text and "own limits ran out" not in text
        assert "no new summary request was sent" in usage["terminal_provider_notice"].lower()
    else:
        assert "owner requested Wrap up" not in text and "provider outage" in text


def test_deadline_text_does_not_hide_an_existing_unknown_attempt():
    usage = {"_last_llm_error_kind": "provider_outcome_unknown"}
    text = loop_transport.provider_terminal_fallback_text(usage, is_context_overflow=False,
        is_transport_wait=False, waited_sec=0.0, interactive=False, is_deadline_exhausted=True)
    assert "owner deadline" in text and "no terminal provider outcome" in text
    assert "no retry or paid fallback was sent" in text


def test_provider_terminal_text_claims_only_recorded_recovery_facts():
    ordinary = loop_transport.provider_terminal_fallback_text(
        {"_last_llm_error_kind": "provider_transient", "_last_llm_error": "HTTP 503"},
        is_context_overflow=False, is_transport_wait=False, waited_sec=0.0,
        interactive=False, is_deadline_exhausted=False,
    )
    assert "provider returned no usable response" in ordinary
    assert "same-model reroute" not in ordinary

    unknown = loop_transport.provider_terminal_fallback_text(
        {"_last_llm_error_kind": "provider_outcome_unknown"},
        is_context_overflow=False, is_transport_wait=False, waited_sec=0.0,
        interactive=False, is_deadline_exhausted=False,
    )
    assert unknown.count("dispatched request has no terminal provider outcome") == 1
    assert "same-model reroute" not in unknown


@pytest.mark.parametrize("honored", ["confirmed", "unknown"])
def test_applied_options_without_mismatch_emit_no_owner_line(honored):
    progress = []
    usage = {"_options": {"options_honored": honored}}

    loop_transport.emit_model_effort_mismatch(
        usage, task_id="task-7",
        emit_progress=lambda text, *, incident=None: progress.append((text, incident)),
    )

    assert progress == []


def test_owner_line_speaks_only_for_a_changed_reasoning_effort():
    """A mismatch on another submitted option is durable, never an effort claim."""
    progress = []
    route = {"credentialProfileId": "acct-a", "model": "codex=model"}
    usage = {"_model_route": dict(route), "_options": {
        "options_honored": "mismatch", "route": dict(route),
        "requested_options": {"reasoningEffort": "high", "cacheKey": "execution-a"},
        "applied_options": {"reasoningEffort": "high", "cacheKey": "engine-b"}}}

    def emit(text, *, incident=None):
        progress.append(text)

    loop_transport.emit_model_effort_mismatch(usage, task_id="task-7", emit_progress=emit)
    assert progress == [] and usage["_options"]["options_honored"] == "mismatch"

    # The silent round spent no dedupe slot: a real effort change still speaks.
    usage["_options"]["applied_options"] = {"reasoningEffort": "low", "cacheKey": "engine-b"}
    loop_transport.emit_model_effort_mismatch(usage, task_id="task-7", emit_progress=emit)
    assert progress == ["⚠️ Claudexor served at low effort while high was requested"
                        " (Claudexor account acct-a)."]


def _mismatch_round_context(tmp_path, monkeypatch, *, emit_progress, applied_values):
    """A Main round whose subscription answer reports a lowered effort."""
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = loop._RoundModelCallContext(
        llm=None, messages=[], tools=registry, context_fit_plan=None,
        active_model="claudexor::codex=model", tool_schemas=[], active_effort="high",
        max_retries=1, drive_logs=tmp_path / "logs", task_id="task-7", round_idx=1,
        event_queue=None, accumulated_usage={}, task_type="task", active_use_local=False,
        active_context_mode="max", drive_root=tmp_path, model_role="main",
        emit_progress=emit_progress,
    )

    route = {"credentialProfileId": "account-a", "model": "codex=model"}

    def call(_llm, _messages, _model, _tools, _effort, _retries, _logs, _tid,
             _round, _queue, usage, *_args, **_kwargs):
        usage["_options"] = {
            "requested_options": {"reasoningEffort": "high"},
            "applied_options": {"reasoningEffort": next(applied_values)},
            "options_honored": "mismatch",
            "route": dict(route),
        }
        usage["_model_route"] = dict(route)
        return {"role": "assistant", "content": "done"}, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", call)
    monkeypatch.setattr(loop, "_server_web_allowed_by_task", lambda _ctx: False)
    return ctx


def test_effort_mismatch_emits_one_typed_owner_line_per_task_and_model(tmp_path, monkeypatch):
    progress = []
    ctx = _mismatch_round_context(
        tmp_path, monkeypatch, applied_values=iter(("medium", "low")),
        emit_progress=lambda text, *, incident=None: progress.append((text, incident)),
    )
    loop._dispatch_round_model(ctx, None, attempt_cap=None)
    loop._dispatch_round_model(ctx, None, attempt_cap=None)

    assert len(progress) == 1
    text, incident = progress[0]
    assert "served at medium effort while high was requested" in text
    assert "Claudexor account account-a" in text
    assert incident == {
        "task_incident": "model_effort_mismatch",
        "toast_once": "task-7:model_effort_mismatch:codex=model",
    }


def test_failed_round_route_never_borrows_the_previous_applied_options(tmp_path, monkeypatch):
    """Only the route that reported applied options can be named in its line."""
    from ouroboros.llm_claudexor import ClaudexorModelError

    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    usage, progress = {}, []
    route_a = {"model": "MODEL-A", "credentialProfileId": "acct-a", "source": "codex"}
    route_b = {"model": "MODEL-B", "credentialProfileId": "acct-b", "source": "claude"}

    def served(route, requested, applied):
        return {"role": "assistant", "content": "done"}, {"claudexor": {
            "route": dict(route), "options_honored": "mismatch",
            "requested_options": {"reasoningEffort": requested},
            "applied_options": {"reasoningEffort": applied}}}

    def failed(route):
        raise ClaudexorModelError({"code": "model_operation_failed", "message": "engine down",
                                   "context": {"httpStatus": 503}}, route=route)

    rounds = [lambda: served(route_a, "xhigh", "low"), lambda: failed(route_b),
              lambda: served(route_b, "high", "medium")]
    lines_after = []

    for index, step in enumerate(rounds):
        monkeypatch.setattr(loop_llm_call, "_send_main_candidate",
                            lambda *_args, _step=step, **_kwargs: _step())
        loop_llm_call.call_llm_with_retry(
            SimpleNamespace(), [], "MODEL", [], "xhigh", 1, logs, "task-1", index, None, usage,
            "task", attempt_cap=1, initial_messages=[])
        loop_transport.emit_model_effort_mismatch(
            usage, task_id="task-1",
            emit_progress=lambda text, *, incident=None: progress.append((text, incident)))
        lines_after.append(len(progress))
        if index == 1:  # the failed round names its own route and carries no applied options
            assert usage["_model_route"] == route_b and usage["_options"]["route"] == route_a

    assert lines_after == [1, 1, 2]  # the failed round adds nothing; MODEL-B speaks for itself
    assert [incident["toast_once"] for _text, incident in progress] == [
        "task-1:model_effort_mismatch:MODEL-A", "task-1:model_effort_mismatch:MODEL-B"]
    assert progress[1][0] == ("⚠️ Claudexor served at medium effort while high was requested"
                              " (Claudexor account acct-b).")


def test_mismatch_round_never_calls_the_one_argument_tool_context_emitter(tmp_path, monkeypatch):
    """The frozen ToolContext seam takes one argument and stays out of this notice."""
    seen = []
    ctx = _mismatch_round_context(tmp_path, monkeypatch, emit_progress=None,
                                  applied_values=iter(("medium",)))
    ctx.tools._ctx.emit_progress_fn = seen.append  # rejects incident=, exactly like the ABI default

    loop._dispatch_round_model(ctx, None, attempt_cap=None)

    assert seen == [] and ctx.accumulated_usage["_options"]["options_honored"] == "mismatch"


def test_body_error_diagnostic_is_masked_before_terminal_publication(tmp_path, monkeypatch):
    from ouroboros.utils import sanitize_tool_result_for_log
    from tests.test_transport_death_retry import _ScriptedLLM, _death, _primary_call

    secret = "synthetic_notice_canary_" + "x" * 32
    body = "Request failed for api_key=" + secret
    assert secret not in sanitize_tool_result_for_log(body)
    usage = {}
    llm = _ScriptedLLM(_death, ({"content": "", "tool_calls": []}, {
        "cost": 0.0, "provider_error": {"kind": "provider_error", "code": "401", "message": body},
    }))
    monkeypatch.setattr(loop_llm_call, "_sleep_within_deadline", lambda *_a, **_k: True)
    message, _ = _primary_call(llm, tmp_path / "logs", usage)
    assert message is None and llm.calls == 2
    _, registry, ctx, trace = _forced_test_context(tmp_path, usage=usage)
    loop._replace_delivery_candidate(registry, ctx, trace, RAW, control="replace")
    text, usage, trace = loop._handle_provider_unavailable(ctx, error_kind=usage["_last_llm_error_kind"])
    notices = []
    assert send_provider_death_notice(SimpleNamespace(send_with_budget=lambda *a, **k: notices.append(a[1])), 7, "parent1", usage)
    assert text == RAW
    assert secret not in usage["_last_llm_error"] + usage["terminal_provider_notice"] + notices[0]
    assert "***" in notices[0]
