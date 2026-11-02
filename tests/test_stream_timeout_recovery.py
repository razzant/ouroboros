"""#1886 repair negatives at sends, forced rails, cold re-entry, captions and retained parent readers."""
import json
import queue
import time
from types import SimpleNamespace

import pytest

from ouroboros import loop, working_checkpoint as wc
from ouroboros.loop_llm_call import TIMEOUT_REPEAT_KEY, call_llm_with_retry
from ouroboros.task_results import write_task_result
from tests.test_stream_timeout_repeat import (
    CHILD, MODEL, PARENT, Wire, WireLLM, child_drive, child_loop, chunk,
    deliver_to_parent, ok_wire, parent_tools, parent_views, sleeps as sleeps, timeout_wire, wire_env as wire_env,
)
from tests.test_working_checkpoint import _registry


class WorkerLost(BaseException):
    pass


def crash(*_args, **_kwargs):
    raise WorkerLost()


def tool_wire(content=""):
    """One usable round asking for a read-only tool."""
    return Wire(chunk({"role": "assistant", "content": content, "tool_calls": [{"index": 0, "id": "read-tools",
        "type": "function", "function": {"name": "list_available_tools", "arguments": "{}"}}]}),
        chunk({}, finish="tool_calls"), b"data: [DONE]\n\n")


def http_error(message):
    error = RuntimeError(message)
    error.status_code = 400
    return error


@pytest.mark.parametrize("failure", [
    "Invalid signature in thinking block", "Unsupported parameter prompt_cache_key",
    "Unsupported parameter temperature",
])
def test_repeat_rejection_never_repairs_into_a_third_send(wire_env, sleeps, monkeypatch, failure):
    llm = WireLLM(timeout_wire(), http_error(failure), ok_wire("forbidden third"))
    # Seed real affinity and recoverable wire fields on the physical candidate.
    build = llm._create_chat_completion_with_retries
    def seeded(create, kwargs, target):
        kwargs.setdefault("extra_body", {})["session_id"] = "original-session"
        kwargs.update(temperature=0.5, prompt_cache_key="original-key")
        return build(create, kwargs, target)
    monkeypatch.setattr(llm, "_create_chat_completion_with_retries", seeded)
    messages = [{"role": "assistant", "content": "prior", "reasoning_details": [
        {"type": "reasoning.text", "text": "prior reasoning", "index": 0}]},
        {"role": "user", "content": "continue"}]
    usage = {}
    msg, _ = call_llm_with_retry(llm, messages, MODEL, None, "low", 3, wire_env / "logs",
                               CHILD, 1, None, usage, "task", configured_api_child=True)
    print({"sends": len(llm.sent), "accepted": msg is not None})
    assert len(llm.sent) == 2 and msg is None
    assert [s["extra_body"]["session_id"] for s in llm.sent] == ["original-session"] * 2
    assert failure in usage["_last_llm_failure"]["error"]


@pytest.mark.parametrize("code,message", [(504, "upstream timeout"), (400, "Invalid encrypted reasoning content")])
def test_returned_body_failure_does_not_buy_a_reroute(wire_env, sleeps, monkeypatch, code, message):
    body = {"error": {"code": code, "message": message}, "choices": [], "model": MODEL}
    llm = WireLLM(timeout_wire(), SimpleNamespace(model_dump=lambda: body), ok_wire("forbidden third"))
    build = llm._create_chat_completion_with_retries
    def json_repeat(create, kwargs, target):
        if llm.sent:
            kwargs.pop("stream", None)  # exercise the returned-JSON consumer inside the same send owner
        return build(create, kwargs, target)
    monkeypatch.setattr(llm, "_create_chat_completion_with_retries", json_repeat)
    usage = {}
    answer, _ = call_llm_with_retry(llm, [{"role": "user", "content": "research"}], MODEL, None,
        "low", 3, wire_env / "logs", CHILD, 1, None, usage, "task", configured_api_child=True)
    assert len(llm.sent) == 2 and answer is None
    assert message in usage["_last_llm_failure"]["error"]


@pytest.mark.parametrize("progress", [True, False])
def test_native_anthropic_typed_timeout_uses_actual_send_consumer(wire_env, sleeps, monkeypatch, progress):
    import requests
    def frame(kind, **data):
        return ("event: " + kind + "\ndata: " + json.dumps({"type": kind, **data}) + "\n\n").encode()
    def native_wire():
        pieces = [frame("message_start", message={"id": "native-gen", "model": MODEL, "role": "assistant", "content": []})]
        if progress:
            pieces += [frame("content_block_start", index=0, content_block={"type": "text", "text": ""}),
                       frame("content_block_delta", index=0, delta={"type": "text_delta", "text": "Partial"})]
        pieces += [frame("error", error={"type": "timeout_error", "message": "request ended"})]
        wire = Wire(*pieces)
        wire.status_code = 200
        wire.iter_content = lambda **_kw: wire.iter_bytes()
        return wire
    llm, sent = WireLLM(), []
    target = dict(provider="anthropic", resolved_model=MODEL, usage_model=MODEL,
                  base_url="https://native.invalid/v1", api_key="fixture-key")
    monkeypatch.setattr(llm, "_resolve_remote_target", lambda *_a, **_k: dict(target))
    def post(*_a, **kw):
        sent.append(kw["json"])
        return native_wire()
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests.Session, "post", post)
    usage = {}
    answer, _ = call_llm_with_retry(llm, [{"role": "user", "content": "research"}], MODEL, None,
        "low", 3, wire_env / "logs", CHILD, 1, None, usage, "task", configured_api_child=True)
    assert len(sent) == (2 if progress else 1) and answer is None
    assert (TIMEOUT_REPEAT_KEY in usage) is progress


def test_accepted_success_checkpoint_allows_the_next_author_round(wire_env, sleeps, monkeypatch):
    llm = WireLLM(timeout_wire(), tool_wire(), timeout_wire(gen="next"), ok_wire("Next round answer."))
    save = wc.save_round
    def crash_after_accepted_batch(ctx, boundary):
        result = save(ctx, boundary)
        if boundary == "post_batch":
            raise WorkerLost()
        return result
    with monkeypatch.context() as interruption:
        interruption.setattr(wc, "save_round", crash_after_accepted_batch)
        with pytest.raises(WorkerLost):
            run_child(wire_env, monkeypatch, llm)
    assert len(llm.sent) == 2
    _data, saved = wc.read_working(wire_env, CHILD, 1)
    assert TIMEOUT_REPEAT_KEY not in saved["usage"]
    task = {"id": CHILD}
    assert wc.attach_recovery(wire_env, task, source_task_id=CHILD, from_attempt=1, cause="worker_crash")
    text, usage, trace = run_child(wire_env, monkeypatch, llm, handoff=task["_working_recovery"])
    assert text == "Next round answer." and len(llm.sent) == 4 and TIMEOUT_REPEAT_KEY not in usage
    assert len(trace["tool_calls"]) == 1


def run_child(root, monkeypatch, llm, task_id=CHILD, handoff=None, messages=None):
    tools = _registry(root, monkeypatch, task_id, 2 if handoff else 1)
    tools._ctx.exact_model_route = True
    tools._ctx._configured_subagent_route_kind = "api_model"
    tools._ctx.task_model_override = MODEL
    tools._ctx.working_recovery = handoff
    write_task_result(root, task_id, "running")
    return loop.run_llm_loop(messages=messages or [{"role": "user", "content": "Research"}], tools=tools,
        llm=llm, drive_logs=root / "logs", emit_progress=lambda *_a, **_k: None,
        incoming_messages=queue.Queue(), task_type="task", task_id=task_id, drive_root=root)


def run_controlled_child(root, monkeypatch, llm):
    from ouroboros import owner_wait
    from ouroboros.model_wait import task_model_wait_scope
    tools = _registry(root, monkeypatch, CHILD, 1)
    ctx = tools._ctx
    ctx.exact_model_route, ctx._configured_subagent_route_kind, ctx.task_model_override = True, "api_model", MODEL
    ctx.owner_wait_callback, ctx.pending_events = owner_wait.direct_owner_wait, []
    write_task_result(root, CHILD, "running", root_task_id=CHILD)
    with task_model_wait_scope(task={"id": CHILD, "_attempt": 1, "budget_drive_root": str(root)},
                              drive_root=root, event_queue=None, worker_slot_held=False) as waiter:
        ctx.model_wait_context, waiter.tool_context = waiter, ctx
        return loop.run_llm_loop(messages=[{"role": "user", "content": "Research"}], tools=tools,
            llm=llm, drive_logs=root / "logs", emit_progress=lambda *_a, **_k: None,
            incoming_messages=queue.Queue(), task_type="task", task_id=CHILD, drive_root=root)


def test_warm_pause_resumes_same_round_without_reopening_repeat(wire_env, sleeps, monkeypatch):
    from ouroboros import owner_pause, owner_wait, review_pause
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.model_wait import ModelWaitInterrupted
    from ouroboros.task_results import load_task_result
    parked = []
    def pause():
        owner_pause.install_fence(wire_env, CHILD, request_id="repeat-pause")
        raise ModelWaitInterrupted("owner_pause")
    repeat = timeout_wire(gen="second")
    repeat.on_read = pause
    llm = WireLLM(timeout_wire(), repeat, ok_wire("forbidden third"))
    # Synthetic detached-review census selects warm parking; all pause/restore consumers are real.
    monkeypatch.setattr(review_pause, "live_detached_operations", lambda _tid: [{"owner_id": "fixture-review", "surface": "test"}])
    def resume(_seconds):
        row = load_task_result(wire_env, CHILD)["owner_wait"]
        assert row["state"] == "waiting" and row["reason"] == "owner_pause" and len(llm.sent) == 2
        state = json.loads(read_actor_source_bytes(wire_env, CHILD, row["source_ref"]))
        assert state["usage"][TIMEOUT_REPEAT_KEY]["state"] == "dispatched"
        parked.append(row)
        owner_pause.release_fence(wire_env, CHILD, reason="owner_resume")
    monkeypatch.setattr(owner_wait, "time", SimpleNamespace(time=time.time, monotonic=time.monotonic, sleep=resume))
    text, usage, trace = run_controlled_child(wire_env, monkeypatch, llm)
    assert len(parked) == 1 and len(llm.sent) == 2 and "forbidden third" not in text
    assert load_task_result(wire_env, CHILD)["owner_wait"]["state"] == "resumed"
    assert "no confirmed provider outcome" in usage["terminal_host_notice"]
    assert trace["forced_finalization"]["source"] == "provider_outcome_unknown_no_resend"


def test_wrap_up_during_timeout_backoff_sends_nothing_and_retains_notice(wire_env, monkeypatch):
    from ouroboros import cancel_intents
    from ouroboros.loop_llm_call import time as call_time
    from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
    from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, write_owner_message
    from supervisor.owner_stop import owner_stop_control_id
    llm = WireLLM(timeout_wire(), ok_wire("forbidden repeat"))
    def wrap_up(_seconds):
        intent = cancel_intents.request_cancel(wire_env, CHILD, requested_stop_policy=cancel_intents.STOP_POLICY_FINALIZE)
        write_owner_message(wire_env, REASON_OWNER_REQUESTED_FINALIZATION, CHILD,
                            msg_id=owner_stop_control_id(intent), kind=KIND_FINALIZE_NOW)
    monkeypatch.setattr("ouroboros.loop_llm_call.time", SimpleNamespace(
        time=call_time.time, monotonic=call_time.monotonic, perf_counter=call_time.perf_counter, sleep=wrap_up))
    text, usage, trace = run_controlled_child(wire_env, monkeypatch, llm)
    assert len(llm.sent) == 1 and "forbidden repeat" not in text
    assert usage["reason_code"] == REASON_OWNER_REQUESTED_FINALIZATION
    assert trace["forced_finalization"]["source"] == "model_wait_control"
    assert TIMEOUT_REPEAT_KEY in usage and "was not sent" in usage["terminal_host_notice"]


@pytest.mark.parametrize("new_id", [False, True], ids=["same_id", "successor_id"])
@pytest.mark.parametrize("lost_at", ["granted", "dispatch_intent", "dispatched", "spent"])
def test_real_checkpoint_recovery_does_not_renew_repeat(wire_env, sleeps, monkeypatch, new_id, lost_at):
    def crash(*_a, **_k):
        raise WorkerLost()
    repeat = timeout_wire(gen="second")
    if lost_at == "dispatched":
        repeat.on_read = crash
    llm = WireLLM(timeout_wire(), repeat, ok_wire("forbidden third"))
    with monkeypatch.context() as interruption:
        if lost_at == "granted":
            interruption.setattr("ouroboros.loop_llm_call._sleep_within_deadline", crash)
        elif lost_at == "dispatch_intent":
            save = wc.save_model_call
            def crash_before_wire():
                save()
                _raw, state = wc.read_working(wire_env, CHILD, 1)
                if state["usage"][TIMEOUT_REPEAT_KEY]["state"] == "dispatched":
                    crash()
            interruption.setattr(wc, "save_model_call", crash_before_wire)
        elif lost_at == "spent":
            interruption.setattr(loop, "_timeout_repeat_handover", crash)
        from ouroboros.usage_accounting import PhysicalAttemptPreparationFailed
        expected = PhysicalAttemptPreparationFailed if lost_at == "dispatch_intent" else WorkerLost
        with pytest.raises(expected):
            run_child(wire_env, monkeypatch, llm)
    before = len(llm.sent)
    target = CHILD + "-successor" if new_id else CHILD
    task = {"id": target}
    assert wc.attach_recovery(wire_env, task, source_task_id=CHILD, from_attempt=1, cause="worker_crash")
    text, usage, _trace = run_child(wire_env, monkeypatch, llm, target, task["_working_recovery"])
    print({"lost_at": lost_at, "new_id": new_id, "before": before, "after": len(llm.sent)})
    assert before == (1 if lost_at in {"granted", "dispatch_intent"} else 2)
    assert len(llm.sent) == (1 if lost_at == "dispatch_intent" else 2) and "forbidden third" not in text
    assert TIMEOUT_REPEAT_KEY in usage and usage["execution_status"] == "infra_failed"
    if lost_at in {"dispatch_intent", "dispatched"}:
        assert "no confirmed provider outcome" in usage["terminal_host_notice"]


@pytest.mark.parametrize("empty", [False, True], ids=["exception", "empty_response"])
def test_parent_reads_failure_facts_after_actual_drive_settlement(wire_env, sleeps, monkeypatch, empty):
    from ouroboros.task_custody import settle_child_drive
    from ouroboros.headless import retry_child_task_refs
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.tools.join_ledger import _child_result_sha256

    repeat = Wire(chunk({}, finish="stop"), b"data: [DONE]\n\n") if empty else timeout_wire(gen="second")
    llm = WireLLM(timeout_wire(), repeat, ok_wire("forbidden third"))
    prior = [{"role": "assistant", "content": "Saved finding."}, {"role": "user", "content": "Continue."}]
    text, usage, trace, _ = child_loop(wire_env, llm, prior=prior)
    deliver_to_parent(wire_env, monkeypatch, text, usage, trace)
    parent_tools(wire_env)
    result_before = load_effective_task_result(wire_env, CHILD)
    drive = child_drive(wire_env)
    # GC arms the existing maintenance duty; promotion must finish before removal.
    pending = settle_child_drive(wire_env, CHILD, drive, live=lambda _: False)
    assert pending["reason"] == "call_inventory_pending", pending
    retry_child_task_refs(wire_env, drive, CHILD)
    settled = settle_child_drive(wire_env, CHILD, drive, live=lambda _: False)
    assert settled["status"] == "removed", settled
    assert not drive.exists()
    result = parent_tools(wire_env).execute_result("query_data", {
        "op": "json", "kind": "task_result", "ids": [CHILD], "select": "trace_refs.timeout_repeat"})
    assert result.status == "ok", result.text
    facts = json.loads(result.text)["data"]
    print({"sends": len(llm.sent), "retained_failure": facts})
    assert len(llm.sent) == 2 and facts["first"]["explicit_timeout"]
    assert facts["repeat"]["failure_code"] == ("llm_empty_response" if empty else "provider_transient")
    assert facts["repeat"]["physical_attempt_id"] != facts["first"]["physical_attempt_id"]
    # The referenced exact source also survives: raw exception fields or the actual empty response.
    from ouroboros.observability import read_call_manifest_ref, read_blob_ref
    original = facts["repeat"].get("manifest_ref") or facts["repeat"]["response_ref"]
    manifest = read_call_manifest_ref(wire_env, original, task_id=CHILD)
    payload = read_blob_ref(wire_env, manifest["full_payload_ref"])
    assert payload and (payload["message"]["content"] is None if empty else payload["explicit_timeout"])
    if empty:
        assert facts["repeat"]["attempt_custody_state"] == "settled"
        assert payload["usage"]["cost"] is None and payload["usage"]["cost_final"] is False
    full, waited, compact = parent_views(wire_env)
    assert "Saved finding." in full and "Saved finding." in waited
    assert "trace_refs.timeout_repeat" in compact["terminal_host_notice"]
    assert compact["child_result_sha256"] == _child_result_sha256(result_before)


# ---------------------------------------------------------------- review repairs: every rail, owner and projection


@pytest.mark.parametrize("route_kind,exact,sends", [("api_model", True, 3), ("", False, 7)], ids=["configured_child", "main"])
def test_round_limit_forced_call_keeps_the_actors_timeout_policy(wire_env, sleeps, monkeypatch, route_kind, exact, sends):
    """The real round-limit rail's summary call is still the actor's call: a configured API child sends
    its one repeat and nothing after it; Main keeps its six-attempt burst."""
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "1")
    forced = [timeout_wire(gen=f"forced-{n}") for n in range(1, 7)]
    llm = WireLLM(tool_wire("Checking the tools."), *forced, ok_wire("forbidden extra summary"))
    text, usage, trace, _notes = child_loop(wire_env, llm, route_kind=route_kind, exact=exact)
    assert len(llm.sent) == sends and "forbidden extra summary" not in text and len(trace["tool_calls"]) == 1
    assert all("[ROUND_LIMIT]" in json.dumps(sent["messages"]) for sent in llm.sent[1:])
    assert usage["reason_code"] == "round_limit"
    if not exact:
        assert TIMEOUT_REPEAT_KEY not in usage and "Stream timeout" not in usage.get("terminal_host_notice", "")
        return
    notice = usage["terminal_host_notice"]
    assert trace["timeout_repeat"]["repeat"]["generation_id"] == "forced-2"
    assert "It was repeated once" in notice and "forced-1" in notice and "forced-2" in notice
    assert trace["forced_finalization"]["source"] == "timeout_repeat_spent_no_resend"
    assert text == "Checking the tools."  # the saved work, never the partial summary stream


def recover_near_deadline(root, monkeypatch, llm):
    """The real cold recovery, its deadline now inside the finalization reserve."""
    import datetime as dt
    from ouroboros.tools.registry import ToolRegistry

    deadline = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
    original = ToolRegistry.__init__

    def with_deadline(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self._ctx.task_metadata = {**(getattr(self._ctx, "task_metadata", {}) or {}), "deadline_at": deadline}

    task = {"id": CHILD}
    assert wc.attach_recovery(root, task, source_task_id=CHILD, from_attempt=1, cause="worker_crash")
    with monkeypatch.context() as patched:
        patched.setattr(ToolRegistry, "__init__", with_deadline)
        return run_child(root, monkeypatch, llm, CHILD, task["_working_recovery"])


@pytest.mark.parametrize("lost_at", ["granted", "dispatched"])
def test_restored_repeat_buys_no_deadline_summary_and_keeps_its_facts(wire_env, sleeps, monkeypatch, lost_at):
    """The deadline rail runs before the round's own guard: a restored grant is not spent on a [DEADLINE]
    summary, a restored dispatch stays UNKNOWN, and either terminal carries the notice and trace."""
    repeat = timeout_wire(gen="second")
    llm = WireLLM(timeout_wire(), repeat, ok_wire("forbidden summary"))
    with monkeypatch.context() as interruption:
        if lost_at == "granted":
            interruption.setattr("ouroboros.loop_llm_call._sleep_within_deadline", crash)
        else:
            repeat.on_read = crash
        with pytest.raises(WorkerLost):
            run_child(wire_env, monkeypatch, llm)
    before = len(llm.sent)
    text, usage, trace = recover_near_deadline(wire_env, monkeypatch, llm)
    assert len(llm.sent) == before == (1 if lost_at == "granted" else 2) and "forbidden summary" not in text
    assert not any("[DEADLINE]" in json.dumps(sent["messages"]) for sent in llm.sent)
    notice = usage["terminal_host_notice"]
    assert usage["reason_code"] == "deadline_local" and trace["timeout_repeat"]["state"] == lost_at
    if lost_at == "granted":
        assert "was not sent (the task deadline left no room)" in notice
        assert trace["forced_finalization"]["source"] == "timeout_repeat_spent_no_resend"
    else:
        assert "no confirmed provider outcome (UNKNOWN)" in notice
        assert trace["forced_finalization"]["source"] == "provider_outcome_unknown_no_resend"


@pytest.mark.parametrize("linked", [True, False], ids=["reaper_links", "owner_claim_alone"])
def test_new_id_recovery_keeps_the_predecessors_failure_refs_readable(wire_env, sleeps, monkeypatch, linked):
    """A stored pre-upgrade successor publishes the predecessor's failure rows; promotion verifies their
    sources under that owner only along its reciprocal retry links (same-ID settlement is pinned above)."""
    from ouroboros import agent_task_pipeline as pipeline, observability as obs
    from ouroboros.task_results import load_task_result

    llm = WireLLM(timeout_wire(), timeout_wire(gen="second"), ok_wire("forbidden third"))
    with monkeypatch.context() as interrupted:  # both sends and the spent checkpoint exist, no terminal yet
        interrupted.setattr(loop, "_timeout_repeat_handover", crash)
        with pytest.raises(WorkerLost):
            run_child(wire_env, monkeypatch, llm)
    successor = CHILD + "-successor"
    task = {"id": successor, "_attempt": 2, "type": "task", "chat_id": 1, "text": "Research",
            "parent_task_id": PARENT, "root_task_id": PARENT, "delegation_role": "subagent",
            "original_task_id": CHILD, "timeout_retry_from": CHILD}
    if linked:  # the rows a pre-upgrade timeout retry admission wrote
        write_task_result(wire_env, successor, "scheduled", original_task_id=CHILD, timeout_retry_from=CHILD)
        write_task_result(wire_env, CHILD, "interrupted", retry_task_id=successor, superseded_by=successor)
    assert wc.attach_recovery(wire_env, task, source_task_id=CHILD, from_attempt=1, cause="idle_timeout")
    text, usage, trace = run_child(wire_env, monkeypatch, llm, successor, task["_working_recovery"])
    rows = trace["timeout_repeat"]["first"], trace["timeout_repeat"]["repeat"]
    assert len(llm.sent) == 2 and [row["task_id"] for row in rows] == [CHILD, CHILD]
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *_a, **_k: None)
    write_task_result(wire_env, PARENT, "running", root_task_id=PARENT)
    pipeline.emit_task_results(SimpleNamespace(drive_root=wire_env, repo_dir=wire_env), None, None, [], task, text,
                               usage, trace, start_time=0.0, drive_logs=wire_env / "logs")
    promoted, state = obs.promote_child_task_refs(wire_env, wire_env, successor, load_task_result(wire_env, successor))
    for row in ("first", "repeat"):
        ref = promoted["trace_refs"]["timeout_repeat"][row]["manifest_ref"]
        if linked:
            assert ref.get("path") and obs.read_call_manifest_ref(wire_env, ref, task_id=CHILD)["task_id"] == CHILD
        else:  # a recorded owner alone never verifies another task's source
            assert ref.get("availability") == "unavailable"
    assert bool(state["unavailable_refs"]) is not linked


def caption_reply(text, gen):
    from ouroboros.llm_stream import AssembledResponse

    return AssembledResponse({"id": gen, "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                              "finish_reason": "stop"}], "usage": {"prompt_tokens": 12, "completion_tokens": 1, "cost": 0.01}})


CAPTION_MODEL = "vendor/fixture-caption"


def image_task(*names):
    return [{"role": "user", "content": [{"type": "text", "text": "Compare the images."}, *[
        {"type": "image_url", "image_url": {"url": f"https://images.invalid/{name}.png"}} for name in names]]}]


@pytest.fixture
def caption_mode(monkeypatch):
    import ouroboros.vision_routing as vision

    monkeypatch.setattr(vision, "get_image_input_mode", lambda: "caption")
    monkeypatch.setattr(vision, "prepare_caption_image", lambda block, *_a, **_k: (  # the memo's real pixel key
        CAPTION_MODEL, block["image_url"]["url"], vision._url_digest(block["image_url"]["url"]), "", ""))


def test_timeout_repeat_buys_no_caption_in_process_or_after_a_resume(wire_env, sleeps, monkeypatch, caption_mode):
    """Caption mode: the in-process repeat resends the projection it timed out on; a resumed grant projects
    from the memo or a marker without a caption call; once the obligation closes, the next round captions."""
    messages = image_task("a", "b")

    def call(llm, usage, round_idx=1):
        return call_llm_with_retry(llm, messages, MODEL, None, "low", 3, wire_env / "logs", CHILD, round_idx, None,
                                   usage, "task", configured_api_child=True)

    llm = WireLLM(caption_reply("A diagram.", "cap-a"), caption_reply("", "cap-b"),  # b's caption comes back empty
                  timeout_wire(gen="s-1"), timeout_wire(gen="s-2"), caption_reply("B photo.", "cap-b2"))
    assert call(llm, {})[0] is None
    assert [sent["model"] for sent in llm.sent] == [CAPTION_MODEL, CAPTION_MODEL, MODEL, MODEL]
    assert llm.sent[2]["messages"] == llm.sent[3]["messages"]

    usage = {}
    llm = WireLLM(caption_reply("A diagram.", "cap-a"), caption_reply("", "cap-b"), timeout_wire(gen="s-1"),
                  ok_wire("Seen both."), caption_reply("B photo.", "cap-b2"), ok_wire("Next round."))
    with monkeypatch.context() as interruption:  # lost in the backoff; the warm or same-ID resume keeps this usage
        interruption.setattr("ouroboros.loop_llm_call._sleep_within_deadline", crash)
        with pytest.raises(WorkerLost):
            call(llm, usage)
    assert usage[TIMEOUT_REPEAT_KEY]["state"] == "granted" and len(llm.sent) == 3
    assert call(llm, usage)[0]["content"] == "Seen both."
    resumed = json.dumps(llm.sent[3]["messages"])
    assert llm.sent[3]["model"] == MODEL and "[image caption: A diagram.]" in resumed
    assert "[image caption unavailable: no new caption is generated for a timeout repeat]" in resumed
    assert call(llm, usage, round_idx=2)[0]["content"] == "Next round."
    assert [sent["model"] for sent in llm.sent[4:]] == [CAPTION_MODEL, MODEL]


def test_new_id_resume_of_a_granted_repeat_reuses_the_predecessors_captions(wire_env, sleeps, monkeypatch, caption_mode):
    """A new execution resuming the grant: the captions its request had come with the allowance; none is bought."""
    llm = WireLLM(caption_reply("A diagram.", "cap-a"), timeout_wire(gen="s-1"), ok_wire("Seen it."))
    with monkeypatch.context() as interruption:
        interruption.setattr("ouroboros.loop_llm_call._sleep_within_deadline", crash)
        with pytest.raises(WorkerLost):
            run_child(wire_env, monkeypatch, llm, messages=image_task("a"))
    task = {"id": CHILD + "-successor"}
    assert wc.attach_recovery(wire_env, task, source_task_id=CHILD, from_attempt=1, cause="worker_crash")
    text, _usage, _trace = run_child(wire_env, monkeypatch, llm, task["id"], task["_working_recovery"])
    assert text == "Seen it." and [sent["model"] for sent in llm.sent] == [CAPTION_MODEL, MODEL, MODEL]
    assert "[image caption: A diagram.]" in json.dumps(llm.sent[2]["messages"])


@pytest.mark.parametrize("lost_at", ["granted", "dispatched"])
def test_forced_timeout_allowance_survives_worker_loss(wire_env, sleeps, monkeypatch, lost_at):
    """The timeout starts in the real finalizer, not in the ordinary round before it."""
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "1")
    repeat = timeout_wire(gen="forced-repeat")
    llm = WireLLM(tool_wire("Saved finding."), timeout_wire(gen="forced-first"),
                  repeat, ok_wire("FORBIDDEN THIRD FORCED GENERATION"))
    with monkeypatch.context() as interrupted:
        if lost_at == "granted":
            interrupted.setattr("ouroboros.loop_llm_call._sleep_within_deadline", crash)
        else:
            repeat.on_read = crash
        with pytest.raises(WorkerLost):
            run_child(wire_env, monkeypatch, llm)
    before = len(llm.sent)
    _raw, checkpoint = wc.read_working(wire_env, CHILD, 1)
    assert checkpoint["usage"][TIMEOUT_REPEAT_KEY]["state"] == lost_at
    task = {"id": CHILD}
    assert wc.attach_recovery(wire_env, task, source_task_id=CHILD,
                              from_attempt=1, cause="worker_crash")
    text, usage, trace = run_child(wire_env, monkeypatch, llm, handoff=task["_working_recovery"])
    assert len(llm.sent) == before == (2 if lost_at == "granted" else 3)
    assert "FORBIDDEN THIRD" not in text
    assert trace["timeout_repeat"]["state"] == lost_at
    assert trace["timeout_repeat"]["first"]["manifest_ref"]
    notice = usage["terminal_host_notice"]
    assert ("was not sent" if lost_at == "granted" else "no confirmed provider outcome") in notice


def test_initial_missing_caption_route_is_not_a_timeout_repeat(wire_env, sleeps, monkeypatch):
    """An ordinary initial missing route keeps its own explanation and buys no caption."""
    import ouroboros.vision_routing as vision

    monkeypatch.setattr(vision, "get_image_input_mode", lambda: "caption")
    monkeypatch.setattr(vision, "prepare_caption_image", lambda *_a, **_k: ("", "", "", "", ""))
    llm = WireLLM(ok_wire("No caption route."))
    usage = {}
    result = call_llm_with_retry(llm, image_task("a"), MODEL, None, "low", 3, wire_env / "logs",
                                 CHILD, 1, None, usage, "task", configured_api_child=False)
    assert result[0]["content"] == "No caption route." and len(llm.sent) == 1
    projected = json.dumps(llm.sent[0]["messages"])
    assert "no caption route is available" in projected and "timeout repeat" not in projected
    assert TIMEOUT_REPEAT_KEY not in usage
