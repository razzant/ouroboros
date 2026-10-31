"""Executor questions interrupt model-selected sleep; exposure is not an answer.

Use the real interaction producer, fitted-result ACK, warm park, cold grant and
continuation consumers. A question's existence alone never grants execution.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests.test_model_sleep import _ctx, _result
from tests._budget_pause_exact_helpers import _loop_ctx

pytestmark = pytest.mark.serial  # cold consumers bind queue module state


def _run(root, run_id="run-one", task_id="sleeper", source="delegate_start"):
    from ouroboros import delegate_custody as custody, delegate_custody_current as current

    assert custody.emit(root, custody.STARTED, {
        "run_id": run_id, "invocation_id": f"invoke-{run_id}", "task_id": task_id,
        "root_task_id": task_id, "route": "route", "source": source, "access": "readonly"})
    return current.run(root, run_id)


def _question(root, run, question_id="q1", seq=2):
    from ouroboros.delegate_collection import observe_run
    from ouroboros.delegate_custody_current import run as current_run

    assert observe_run(root, run, {"summary": {"runId": run.run_id, "state": "running"}, "lastSeq": seq,
        "pendingInteractions": [{"interactionId": question_id,
                                 "questions": [{"id": "choice", "question": "Which path?"}]}]})
    return current_run(root, run.run_id)


def _source(root, question):
    from ouroboros.artifacts import read_actor_source_bytes

    return json.loads(read_actor_source_bytes(root, "sleeper", question["interaction_source"]))


def _deliver(ctx, text, monkeypatch, *, partial=False, error=False, tool="await_messages", meta=None):
    from ouroboros import loop_tool_execution as loop

    monkeypatch.setattr(loop, "_deliver_tool_results", lambda *_a, **_k: [
        {"result": text, "result_partial": partial}])
    messages = []
    loop.process_tool_results([{
        "fn_name": tool, "tool_call_id": "wait-call", "result": text,
        "is_error": error, "args_for_log": {}, "tool_args": {},
        "result_meta": {"tool_result_meta": meta or {"sleep_question_notice": True}}}],
        messages, {"tool_calls": []}, lambda *_a, **_k: None, tools=SimpleNamespace(_ctx=ctx))
    return messages


def test_question_before_arm_returns_source_but_only_complete_delivery_marks_exposure(tmp_path, monkeypatch):
    from ouroboros.model_sleep import wake_reason
    from ouroboros.tools.control_task_results import _await_messages

    _result(tmp_path, "sleeper")
    _result(tmp_path, "other-branch")
    ctx = _ctx(tmp_path)
    run = _question(tmp_path, _run(tmp_path))
    raw = _await_messages(ctx, mode="warm", senders=["other-branch"])
    payload = json.loads(raw)
    assert payload["reason"] == "ready" and payload["woke_by"] == "run_question:run-one"
    assert not getattr(ctx, "_model_sleep", None) and not getattr(ctx, "_sleep_interaction_seen", {})
    assert _source(tmp_path, payload["questions"][0])["pending_interactions"][0]["interaction_id"] == "q1"
    chosen = {"senders": ["other-branch"], "any_mail": False}
    assert wake_reason(ctx, chosen) == "run_question:run-one"
    _deliver(ctx, raw, monkeypatch, partial=True)
    assert wake_reason(ctx, chosen) == "run_question:run-one"
    _deliver(ctx, raw, monkeypatch, error=True)
    assert wake_reason(ctx, chosen) == "run_question:run-one"
    messages = _deliver(ctx, raw, monkeypatch)
    assert messages[0]["role"] == "tool" and ctx._sleep_interaction_seen == {"run-one": ["q1"]}
    assert wake_reason(ctx, chosen) == ""
    assert json.loads(_await_messages(ctx, mode="warm", senders=["other-branch"]))["reason"] == "sleep_armed"
    assert run.pending_interaction_ids == ["q1"], "notification never answers"
    _question(tmp_path, run, "q2", 3)
    assert wake_reason(ctx, chosen) == "run_question:run-one"


def test_warm_question_landing_after_arm_returns_same_author_with_exact_source(tmp_path):
    from ouroboros import owner_wait
    from ouroboros.model_sleep import wake_reason
    from ouroboros.tools.control_task_results import _await_messages

    _result(tmp_path, "sleeper")
    _result(tmp_path, "other")
    ctx = _ctx(tmp_path)
    run = _run(tmp_path)
    assert json.loads(_await_messages(ctx, mode="warm", senders=["other"]))["reason"] == "sleep_armed"
    _question(tmp_path, run)
    ctx.owner_wait_callback = owner_wait.direct_owner_wait
    messages = [{"role": "user", "content": "intent"}]
    owner_wait.wait_after_tools(ctx, messages, {}, {}, 1, [], set())
    assert ctx.task_id == "sleeper" and ctx._model_sleep is None
    question = json.JSONDecoder().raw_decode(messages[-1]["content"].split("observation: ", 1)[1])[0][0]
    assert _source(tmp_path, question)["run_id"] == "run-one"
    assert wake_reason(ctx, {"any_mail": False}) == ""
    assert ctx._sleep_interaction_seen == {"run-one": ["q1"]}


def test_pooled_warm_question_uses_existing_park_resume_protocol(tmp_path):
    from ouroboros import owner_wait

    _result(tmp_path, "sleeper")
    ctx = _ctx(tmp_path)
    _question(tmp_path, _run(tmp_path))
    checkpoint = {"wait_id": "wait", "reason": "sleep", "sleep": {"any_mail": False}}
    events = []
    class Commands:
        calls = 0
        def get(self, **_kw):
            self.calls += 1
            assert self.calls <= 2
            if self.calls == 2:
                assert any(e.get("phase") == "resume" for e in events)
            return {"type": "owner_wait", "task_id": "sleeper", "task_attempt": 1,
                    "wait_id": "wait", "phase": "parked" if self.calls == 1 else "resume_granted"}
    assert owner_wait.worker_owner_wait(0, Commands(), SimpleNamespace(put=events.append), ctx,
                                        checkpoint) == "run_question:run-one"
    assert [e["phase"] for e in events] == ["park", "resume"]


@pytest.mark.parametrize("source,owner", [("review_substrate", "sleeper"), ("delegate_start", "foreign")])
def test_foreign_and_review_question_custody_does_not_belong_to_sleeper(tmp_path, source, owner):
    from ouroboros.model_sleep import wake_reason

    _result(tmp_path, "sleeper")
    run = _question(tmp_path, _run(tmp_path, task_id=owner, source=source))
    assert run.review_owned if source == "review_substrate" else run.task_id == "foreign"
    assert wake_reason(_ctx(tmp_path), {"any_mail": False}) == ""


def test_native_escalation_not_peer_text_interrupts_filtered_sleep_without_ack(tmp_path, monkeypatch):
    from ouroboros import owner_wait, task_status
    from ouroboros.model_sleep import wake_reason
    from ouroboros.owner_mailbox import write_task_message
    from ouroboros.tools.core_artifacts import _escalate

    _result(tmp_path, "sleeper")
    ctx = _ctx(tmp_path)
    chosen = {"senders": ["other"], "any_mail": False}
    assert write_task_message(tmp_path, "is this a question?", "sleeper", source_task_id="child",
                              provenance="peer_task")
    assert wake_reason(ctx, chosen) == "", "question-looking prose is not authority"
    monkeypatch.setattr(task_status, "load_effective_task_result", lambda *_a: {
        "status": "running", "task_id": "sleeper", "drive_root": str(tmp_path)})
    child = _ctx(tmp_path, "child")
    child.task_metadata = {"parent_task_id": "sleeper", "root_task_id": "sleeper",
                           "delegation_role": "subagent"}
    assert "escalated to parent" in _escalate(child, "Which path?", assumption="retain the result")
    assert wake_reason(ctx, chosen) == "mail:child"
    entries = owner_wait._wait_entries(ctx)
    escalation = next(e for e in entries if e.get("provenance") == "descendant_task")
    assert "Which path?" in escalation["text"] and not ctx._loop_mailbox_seen_ids
    ctx._loop_mailbox_seen_ids.add(escalation["msg_id"])
    assert wake_reason(ctx, chosen) == "", "exposure is owned by the ordinary mailbox drain"


@pytest.mark.parametrize("direct", [False, True])
def test_cold_unselected_question_keeps_pause_veto_then_returns_and_carries_exposure(tmp_path, monkeypatch, direct):
    from tests.test_cold_external_custody import _setup, _park
    from ouroboros import budget_pause as bp, delegate_custody_current as current, owner_pause, owner_wait
    from ouroboros.model_sleep import wake_reason
    from supervisor.sleep_wake import wake_ready_sleepers

    env = _setup(tmp_path, monkeypatch, direct=direct)
    _park(env, tmp_path, monkeypatch, direct=direct, selected={"wake_at": "2099-01-01T00:00:00+00:00"})
    _question(tmp_path, current.run(tmp_path, "run-one"))
    owner_pause.install_fence(tmp_path, "sleeper", request_id="pause")
    out = wake_ready_sleepers(env.q)
    assert out[0]["ready"]["reason"] == "run_question:run-one" and out[0]["veto"] == "owner_pause"
    assert not bp.budget_pause_row(tmp_path, "sleeper").get("grant")
    owner_pause.release_fence(tmp_path, "sleeper", reason="test")
    out = wake_ready_sleepers(env.q)
    assert out[0]["ok"]
    fresh, _ = _loop_ctx(tmp_path, "sleeper", direct=direct)
    fresh.budget_pause_resume = env.workers.PENDING[0]["_budget_pause_resume"]
    saved = bp.load_budget_pause(fresh)
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a: (None, "max"))
    messages, usage = [], {}
    bp.resume_paused_loop(SimpleNamespace(_ctx=fresh), saved, messages, {}, usage, set(), budget_remaining_usd=5)
    assert any("interaction_source" in m["content"] for m in messages)
    assert not any("could not be re-read" in m["content"] for m in messages), "the census shows it pending"
    assert bp.budget_pause_row(tmp_path, "sleeper")["grant"]["consumed_at"]
    assert wake_reason(fresh, {"any_mail": False}) == ""
    retained = owner_wait.continuation_state(fresh, messages, {}, usage, 4, [], set())
    restored, _ = _loop_ctx(tmp_path, "sleeper")
    owner_wait.restore_continuation_state(SimpleNamespace(_ctx=restored), retained, [], {}, {}, set())
    assert wake_reason(restored, {"any_mail": False}) == ""
    _question(tmp_path, current.run(tmp_path, "run-one"), "q2", 3)
    assert wake_reason(restored, {"any_mail": False}) == "run_question:run-one"


def test_native_saved_sleep_delivers_question_on_its_existing_return_path(tmp_path, monkeypatch):
    from ouroboros import owner_wait

    _result(tmp_path, "sleeper")
    ctx = _ctx(tmp_path)
    _question(tmp_path, _run(tmp_path))
    state = owner_wait.continuation_state(ctx, [{"role": "user", "content": "intent"}], {}, {}, 3, [], set())
    ctx.owner_wait_resume = {"reason": "sleep", "sleep": {"mode": "warm"}}
    ctx.owner_wait_callback = lambda *_a: "run_question:run-one"
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a: (None, "max"))
    messages = []
    owner_wait.resume_native_loop(SimpleNamespace(_ctx=ctx), state, messages, {}, {}, set())
    assert "interaction_source" in messages[-1]["content"]
    assert "requesting input" in messages[-2]["content"], "the wake reason precedes its sources"
    assert ctx._sleep_interaction_seen == {"run-one": ["q1"]}


@pytest.mark.parametrize("outcome,named", [("timeout", "the wake time you chose"), ("mail:peer", "mail from peer"),
                                           ("run:run-x", "delegated run run-x reaching an engine terminal")])
@pytest.mark.parametrize("started", [None, 30.0])
def test_native_saved_sleep_names_its_wake_without_inventing_elapsed_time(tmp_path, monkeypatch, outcome,
                                                                         named, started):
    import re
    import time
    from ouroboros import owner_wait

    ctx = _ctx(tmp_path)
    state = owner_wait.continuation_state(ctx, [{"role": "user", "content": "intent"}], {}, {}, 3, [], set())
    ctx.owner_wait_resume = {"reason": "sleep", "sleep": {"mode": "warm"},
                             **({"sleep_started_at": time.time() - started} if started else {})}
    ctx.owner_wait_callback = lambda *_a: outcome
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a: (None, "max"))
    messages = []
    owner_wait.resume_native_loop(SimpleNamespace(_ctx=ctx), state, messages, {}, {}, set())
    notice = messages[-1]["content"]
    assert named in notice, messages
    assert bool(re.search(r"You slept 3[01]s \(warm\)", notice)) == bool(started), notice
    assert started or "You slept (warm)" in notice, "no recorded start: no invented duration"


def test_delegate_wait_a_b_ack_union_and_legacy_ack_survive_restored_cognition(tmp_path):
    from ouroboros import delegate_supervision as sup, owner_wait, sleep_custody
    from ouroboros.model_sleep import wake_reason
    ctx, _ = _loop_ctx(tmp_path, "sleeper")
    for run_id in ("run-a", "run-b"):
        _question(tmp_path, _run(tmp_path, run_id))
        payload = {"run_id": run_id, "status": "waiting_on_user", "supervision_wake_id": run_id,
                   "pending_interactions": [{"interaction_id": "q1", "questions": [{"id": "choice"}]}]}
        sup._save_state(ctx, {"run_id": run_id, "pending_wake": {"wake_id": run_id,
            "interaction_ids": ["q1"], "payload": payload}})
        assert not sup.acknowledge_pending_wake(ctx, json.dumps(payload)[:20])
        assert sup.acknowledge_pending_wake(ctx, json.dumps(payload))
    assert ctx._sleep_interaction_seen == {"run-a": ["q1"], "run-b": ["q1"]}
    assert wake_reason(ctx, {"any_mail": False}) == ""
    state = owner_wait.continuation_state(ctx, [], {}, {}, 4, [], set())
    restored, _ = _loop_ctx(tmp_path, "sleeper")
    owner_wait.restore_continuation_state(SimpleNamespace(_ctx=restored), state, [], {}, {}, set())
    assert wake_reason(restored, {"any_mail": False}) == ""
    legacy, _ = _loop_ctx(tmp_path, "sleeper")
    assert sleep_custody.seen_interactions(legacy) == {}
    assert wake_reason(legacy, {"any_mail": False}).startswith("run_question:"), "legacy exposure is unknown"


def test_unavailable_question_source_does_not_suppress_later_ready_question(tmp_path, monkeypatch):
    from ouroboros import delegate_custody_current as current, model_sleep

    _question(tmp_path, _run(tmp_path))
    ctx = _ctx(tmp_path)
    original = current.run
    with monkeypatch.context() as fault:
        fault.setattr(model_sleep, "_owned_runs",
                      lambda *_a, **_k: (_ for _ in ()).throw(OSError("read unavailable")))
        messages = []
        model_sleep.append_question_notice(ctx, "run_question:run-one", messages)
        assert "could not be re-read" in messages[-1]["content"] and not getattr(ctx, "_sleep_interaction_seen", {})
    assert original(tmp_path, "run-one").pending_interaction_ids == ["q1"]
    assert model_sleep.wake_reason(ctx, {"any_mail": False}) == "run_question:run-one"


def test_only_polls_and_implicit_question_reads_skip_store_contention_waits(tmp_path, monkeypatch):
    from ouroboros import delegate_custody_current as current, model_sleep, owner_wait, store
    from ouroboros.tools.control_task_results import _await_messages
    from supervisor.sleep_wake import _readiness

    _result(tmp_path, "sleeper")
    _run(tmp_path)
    reads, real = [], current.runs

    def observed(root, where="", params=()):
        reads.append((where, store._WAIT_BUDGET.get()))
        return real(root, where, params)

    monkeypatch.setattr(current, "runs", observed)
    ctx = _ctx(tmp_path)
    assert json.loads(_await_messages(ctx, mode="warm", runs=["run-one"]))["reason"] == "sleep_armed"
    assert reads == [("task_id=?", None), ("task_id=? AND state=?", 0), ("task_id=?", None)], \
        "arming a selected run keeps the ordinary contention wait; only the implicit read is a probe"
    reads.clear()
    assert owner_wait._sleep_wake(ctx, {"sleep": ctx._model_sleep}) == ""
    assert _readiness(SimpleNamespace(DRIVE_ROOT=tmp_path), {"id": "sleeper"}, {"sleep": ctx._model_sleep}) == ""
    assert [budget for _where, budget in reads] == [0] * 4, "a poll retries next beat instead of stalling"

    def busy(root, where="", params=()):
        reads.append((where, store._WAIT_BUDGET.get()))
        raise store.StoreBusy("contended", reason="contention")

    monkeypatch.setattr(current, "runs", busy)
    reads.clear()
    out = json.loads(_await_messages(_ctx(tmp_path), mode="warm", wake_at="2099-01-01T00:00:00+00:00"))
    assert out["reason"] == "sleep_armed" and reads == [("task_id=? AND state=?", 0)]
    assert model_sleep.wake_reason(ctx, {"wake_at": "2000-01-01T00:00:00+00:00"}) == "timeout"
    # An explicitly selected unreadable run still preserves its original error.
    with pytest.raises(store.StoreBusy):
        model_sleep.wake_reason(ctx, {"runs": ["run-one"]})


def test_implicit_question_poll_reads_open_rows_but_selected_settled_runs_still_wake(tmp_path, monkeypatch):
    from ouroboros import delegate_custody as custody, delegate_custody_current as current, model_sleep, owner_wait
    from ouroboros.tools.control_task_results import _await_messages

    _result(tmp_path, "sleeper")
    for index in range(3):
        _run(tmp_path, f"done-{index}")
        assert custody.emit(tmp_path, custody.SETTLED, {"run_id": f"done-{index}", "task_id": "sleeper",
                                                         "state": "succeeded"})
    _question(tmp_path, _run(tmp_path, "open"))
    decoded, original = [], current._decode
    monkeypatch.setattr(current, "_decode", lambda raw: decoded.append(raw.get("run_id")) or original(raw))
    ctx = _ctx(tmp_path)
    assert model_sleep.wake_reason(ctx, {"any_mail": False}) == "run_question:open"
    assert decoded == ["open"], "implicit attention never decodes closed history"
    assert owner_wait._sleep_wake(ctx, {"sleep": {"runs": ["done-0"]}}) == "run_question:open"
    raw = _await_messages(ctx, mode="warm", runs=["done-1"])
    assert json.loads(raw)["woke_by"] == "run_question:open"
    assert [question["run_id"] for question in json.loads(raw)["questions"]] == ["open"]
    _deliver(ctx, raw, monkeypatch)
    assert owner_wait._sleep_wake(ctx, {"sleep": {"runs": ["done-0"]}}) == "run:done-0"
    ready = json.loads(_await_messages(ctx, mode="warm", runs=["done-1"]))
    assert ready["reason"] == "ready" and ready["woke_by"] == "run:done-1", "a settled selection answers at once"
    assert not ready.get("questions")


def test_cold_resume_retry_rolls_back_question_exposure_with_its_deleted_disclosure(tmp_path, monkeypatch):
    from tests.test_cold_external_custody import _setup, _park
    from ouroboros import budget_pause as bp, delegate_custody_current as current, owner_wait
    from ouroboros import working_checkpoint as wc
    from ouroboros.owner_mailbox import write_owner_message
    from supervisor.sleep_wake import wake_ready_sleepers

    env = _setup(tmp_path, monkeypatch)
    _park(env, tmp_path, monkeypatch, selected={"wake_at": "2099-01-01T00:00:00+00:00"})
    assert write_owner_message(tmp_path, "come back", "sleeper", msg_id="owner-wake")
    assert wake_ready_sleepers(env.q)[0]["ok"]
    _question(tmp_path, current.run(tmp_path, "run-one"))  # asked after the grant's census
    fresh, _ = _loop_ctx(tmp_path, "sleeper")
    fresh.budget_pause_resume = env.workers.PENDING[0]["_budget_pause_resume"]
    saved = bp.load_budget_pause(fresh)
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a: (None, "max"))
    monkeypatch.setattr(bp, "_publish_hold", lambda *_a: None)
    real, attempts = wc.save_resumed, []

    def flaky(*args, **kwargs):
        attempts.append(dict(fresh._sleep_interaction_seen))
        if len(attempts) == 1:
            raise OSError("disk full")
        return real(*args, **kwargs)

    monkeypatch.setattr(wc, "save_resumed", flaky)
    messages = []
    bp.resume_paused_loop(SimpleNamespace(_ctx=fresh), saved, messages, {}, {}, set(), budget_remaining_usd=5)
    assert attempts == [{"run-one": ["q1"]}] * 2
    assert sum("Delegated question observation" in m["content"] for m in messages) == 1, \
        "the retried disclosure shows the question its deleted predecessor had shown"
    retained = wc.read_working(tmp_path, "sleeper", 1)[1]
    assert retained["messages"] == messages and retained["sleep_interaction_seen"] == {"run-one": ["q1"]}



@pytest.mark.parametrize("woke_by", ["task:ready:completed", "mail:peer", "timeout"])
def test_concurrent_ready_condition_delivers_all_unseen_questions(tmp_path, monkeypatch, woke_by):
    from ouroboros import model_sleep, owner_wait
    from ouroboros.owner_mailbox import write_task_message
    from ouroboros.tools.control_task_results import _await_messages

    _result(tmp_path, "sleeper")
    _result(tmp_path, "ready", "completed")
    _result(tmp_path, "peer")
    ctx = _ctx(tmp_path)
    for rid in ("run-a", "run-b"):
        _question(tmp_path, _run(tmp_path, rid))
    if woke_by.startswith("task:"):
        args = {"tasks": ["ready"]}
    elif woke_by.startswith("mail:"):
        assert write_task_message(tmp_path, "ordinary", "sleeper", source_task_id="peer", provenance="peer_task")
        args = {"senders": ["peer"]}
    else:
        # Real return can have selected timeout recorded before the questions land.
        args = {"wake_at": "2000-01-01T00:00:00+00:00"}
    if woke_by != "timeout":
        raw = _await_messages(ctx, mode="warm", **args)
        # D4 keeps responsible-child questions ahead of selected terminal facts.
        assert json.loads(raw)["woke_by"] == ("run_question:run-a" if woke_by.startswith("task:") else woke_by)
        assert {q["run_id"] for q in json.loads(raw)["questions"]} == {"run-a", "run-b"}
        _deliver(ctx, raw, monkeypatch)
        if woke_by.startswith("task:"):
            assert model_sleep.wake_reason(ctx, args) == woke_by
    else:
        messages = []
        model_sleep.append_question_notice(ctx, woke_by, messages)
        assert "run-a" in messages[0]["content"] and "run-b" in messages[0]["content"]
    assert ctx._sleep_interaction_seen == {"run-a": ["q1"], "run-b": ["q1"]}
    ctx._loop_mailbox_seen_ids.update(e["msg_id"] for e in owner_wait._wait_entries(ctx))
    assert model_sleep.wake_reason(ctx, {"any_mail": False}) == ""


@pytest.mark.parametrize("late", [False, True])
@pytest.mark.parametrize("fresh", ["readable", "unreadable"])
def test_cold_notice_marks_all_sources_it_actually_shows(tmp_path, monkeypatch, fresh, late):
    from ouroboros import budget_pause, model_sleep

    ctx = _ctx(tmp_path)
    external = []
    for rid in ("run-a", "run-b"):
        run = _question(tmp_path, _run(tmp_path, rid))
        external.append({"run_id": rid, "state": "running", "pending_interaction_ids": ["q1"],
                         "interaction_source": run.interaction_source})
    if late:
        _question(tmp_path, _run(tmp_path, "run-late"))  # asked after the grant's census
    row = {"reason": "sleep", "sleep": {"mode": "cold"}, "external_custody": {"state": "accepted"},
           "external_runs": {"runs": external}, "sleep_ready": {"reason": "run_question:run-a"}}
    messages = []
    with monkeypatch.context() as fault:
        if fresh == "unreadable":
            fault.setattr(model_sleep, "_owned_runs", lambda *_a, **_k: (_ for _ in ()).throw(OSError("busy")))
        budget_pause._resume_disclosure(ctx, row, {}, {"consumed_at": 100}, {}, messages)
    text = "\n".join(m["content"] for m in messages)
    assert "run-a" in text and "run-b" in text
    assert "could not be re-read" not in text, "the census itself delivered the waking question"
    shown_late = late and fresh == "readable"
    assert ("run-late" in text) is shown_late
    assert ctx._sleep_interaction_seen == {"run-a": ["q1"], "run-b": ["q1"],
                                           **({"run-late": ["q1"]} if shown_late else {})}
    lost = late and not shown_late
    assert model_sleep.wake_reason(ctx, {"any_mail": False}) == ("run_question:run-late" if lost else "")


@pytest.mark.parametrize("view", ["complete_refusal", "spilled_refusal", "partial"])
def test_loop_acks_each_delivered_wake_but_exposes_only_questions_its_text_carries(tmp_path, monkeypatch, view):
    """Wake ACK keys on the wake a delivered result names, error or not; question
    exposure keys only on interaction ids present in that delivered text."""
    from ouroboros import delegate_supervision as sup, sleep_custody, tool_capabilities
    from ouroboros.tool_result_delivery import render_result_view

    ctx = _ctx(tmp_path)
    payload = {"status": "refused", "ok": False, "host_code": "TOOL_REPORTED_FAILURE", "run_id": "run-a",
               "reason": "daemon_busy", "detail": "x" * 1500, "supervision_wake_id": "wake",
               "pending_interactions": [{"interaction_id": "q1", "questions": [{"id": "choice"}]}]}
    sup._save_state(ctx, {"run_id": "run-a", "pending_wake": {"wake_id": "wake", "interaction_ids": ["q1"],
                                                              "payload": payload}})
    if view == "spilled_refusal":
        monkeypatch.setattr(tool_capabilities, "tool_result_limit", lambda _name: 900)
    result = sup._render_wake_payload(ctx, payload)
    assert result.status == "error" and result.meta == {"supervision_wake_id": "wake"}
    text = render_result_view(result.text, head_chars=80, tail_chars=80)[0] if view == "partial" else result.text
    _deliver(ctx, text, monkeypatch, partial=view == "partial", error=True, tool="delegate_wait", meta=result.meta)
    assert sup._load_state(ctx, "")["pending_wake"], "delivery alone cannot ACK unsaved content"
    from ouroboros.working_checkpoint import flush_content_acks

    flush_content_acks(ctx)  # The incorporated save boundary; _deliver only appends.
    pending = sup._load_state(ctx, "").get("pending_wake")
    if view == "partial":
        assert pending and sleep_custody.seen_interactions(ctx) == {}, "a partial view proves no delivery"
    else:
        assert not pending, "a delivered refusal wake is acknowledged, so it never replays"
        assert sleep_custody.seen_interactions(ctx) == ({"run-a": ["q1"]} if view == "complete_refusal" else {})


def test_supervision_id_without_delivered_question_cannot_mark_sleep_exposure(tmp_path):
    from ouroboros import delegate_supervision as sup, sleep_custody

    ctx = _ctx(tmp_path)
    payload = {"run_id": "run-a", "status": "waiting_on_user", "supervision_wake_id": "wake",
               "pending_interactions": [{"interaction_id": "q1"}]}
    sup._save_state(ctx, {"run_id": "run-a", "pending_wake": {"wake_id": "wake",
        "interaction_ids": ["q1"], "payload": payload}})
    assert sup.acknowledge_pending_wake(ctx, {"supervision_wake_id": "wake"})
    assert sleep_custody.seen_interactions(ctx) == {}
    assert sup._load_state(ctx, "")["interaction_acknowledged_ids"] == ["q1"], "old ACK is not sleep exposure"


def test_restart_retained_warm_sleep_keeps_delivered_question_exposure(tmp_path):
    from ouroboros import model_sleep, owner_wait
    from ouroboros.artifacts import read_actor_source_bytes
    from supervisor.restart_retention import retain_sleep_checkpoint
    from supervisor.sleep_wake import _readiness

    _result(tmp_path, "sleeper")
    _result(tmp_path, "other")
    ctx = _ctx(tmp_path)
    run = _question(tmp_path, _run(tmp_path))
    messages = [{"role": "user", "content": "original intent"}]
    model_sleep.append_question_notice(ctx, "run_question:run-one", messages)
    assert ctx._sleep_interaction_seen == {"run-one": ["q1"]}
    chosen = model_sleep.selectors(ctx, senders=["other"])
    assert model_sleep.request_sleep(ctx, chosen, "warm")["reason"] == "sleep_armed"
    checkpoint = owner_wait.checkpoint_owner_wait(ctx, messages, {}, {}, 1, [], set())
    owner_wait.set_owner_wait(tmp_path, "sleeper", {**checkpoint, "state": "waiting"})
    retained = retain_sleep_checkpoint(tmp_path, "sleeper", 1)
    task = {"id": "sleeper", "_attempt": 1, "drive_root": str(tmp_path)}
    queue = SimpleNamespace(DRIVE_ROOT=tmp_path)
    assert _readiness(queue, task, retained) == "", "an already-shown question cannot mint readiness"
    source = json.loads(read_actor_source_bytes(tmp_path, "sleeper", retained["source_ref"]))
    assert source["messages"] == messages and source["sleep_interaction_seen"] == {"run-one": ["q1"]}
    assert retained["auto_resume"] is False and not retained.get("grant")
    _question(tmp_path, run, "q2", 3)
    assert _readiness(queue, task, retained) == "run_question:run-one", "a new question still wakes"


@pytest.mark.parametrize("control", ["owner_pause", "finalize_now", "hurry"])
def test_control_returns_do_not_read_or_ack_opportunistic_questions(tmp_path, monkeypatch, control):
    from ouroboros import model_sleep
    from tests.test_model_sleep import _mail

    _result(tmp_path, "sleeper")
    ctx = _ctx(tmp_path)
    _question(tmp_path, _run(tmp_path))
    _mail(tmp_path, "sleeper", control, kind=control)
    monkeypatch.setattr(model_sleep, "_question_runs", lambda *_a: pytest.fail("control must not read questions"))
    ready = model_sleep.request_sleep(ctx, {"any_mail": False}, "warm")
    assert ready["woke_by"] == "control:" + control and "questions" not in ready
    messages = []
    model_sleep.append_question_notice(ctx, ready["woke_by"], messages)
    assert not messages and not getattr(ctx, "_sleep_interaction_seen", {})
