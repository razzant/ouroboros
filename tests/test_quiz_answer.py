"""#Q-2b: the quiz answer half — escalate verb, decision ingress, projection
lifecycle, mailbox delivery, structural expiry, replay merge.

Owner decisions pinned here: 27=A (assumption mandatory — validated upstream),
30=A (structural expiry only, no host TTL), 31 (one verb; root → owner card,
subagent → parent mailbox frame; forged owner provenance impossible).
"""

from __future__ import annotations

import types

import pytest

from ouroboros.owner_quiz import (
    STATE_ANSWERED,
    STATE_EXPIRED_TERMINAL,
    STATE_OPEN,
    quiz_states,
    reconcile_terminal,
    record_answered,
    record_asked,
)
from tests._store_shared import read_result, result_signature


def _plant(tmp_path, task_id, **fields):
    """Merge FIELDS into the stored row exactly as a concurrent writer would."""
    from ouroboros.task_store import patch_document

    return patch_document(tmp_path, task_id, lambda row: {**row, **fields})


# ---------------------------------------------------------------- projection

def test_projection_lifecycle_first_answer_wins(tmp_path):
    block = record_asked(tmp_path, "t1", quiz_id="q1", question="Which?",
                         options=["A", "B"], stake="s", assumption="assume A")
    assert block["state"] == STATE_OPEN

    out = record_answered(tmp_path, "t1", quiz_id="q1", option_index=1,
                          request_id="r1", comment="go B")
    assert out["ok"] is True and out["state"] == STATE_ANSWERED
    assert out["block"]["answered_index"] == 1
    assert out["block"]["comment"] == "go B"

    # Same request_id replays the confirmation without a second write.
    dup = record_answered(tmp_path, "t1", quiz_id="q1", option_index=0,
                          request_id="r1")
    assert dup["ok"] is True and dup["duplicate"] is True
    assert quiz_states(tmp_path, "t1")["q1"]["answered_index"] == 1

    # A DIFFERENT id after the first answer is a truthful refusal.
    late = record_answered(tmp_path, "t1", quiz_id="q1", option_index=0,
                           request_id="r2")
    assert late["ok"] is False and late["error"] == "quiz_closed"
    assert late["state"] == STATE_ANSWERED


def test_projection_admits_the_late_answer_only_under_the_explicit_flag(tmp_path):
    """В17a=A at the projection layer: an expired block is answerable when the
    caller says the author is gone, and the acceptance is auditable. Without the
    flag the expiry still refuses, and an ANSWERED block always does."""
    record_asked(tmp_path, "t1", quiz_id="q1", question="Which?", options=["A", "B"],
                 assumption="assume A", chat_id=7, max_wait_minutes=20)
    block = quiz_states(tmp_path, "t1")["q1"]
    assert block["chat_id"] == 7 and block["max_wait_minutes"] == 20
    assert reconcile_terminal(tmp_path, "t1") == ["q1"]

    refused = record_answered(tmp_path, "t1", quiz_id="q1", option_index=0, request_id="r0")
    assert refused["ok"] is False and refused["error"] == "quiz_closed"
    assert refused["state"] == STATE_EXPIRED_TERMINAL

    late = record_answered(tmp_path, "t1", quiz_id="q1", option_index=1,
                           request_id="r1", comment="B, please", allow_expired=True)
    assert late["ok"] is True and late["state"] == STATE_ANSWERED
    assert late["block"]["answered_after_terminal"] is True
    assert late["block"]["answered_index"] == 1 and late["block"]["comment"] == "B, please"

    # First-wins survives: another id cannot overwrite the recorded answer.
    second = record_answered(tmp_path, "t1", quiz_id="q1", option_index=0,
                             request_id="r2", allow_expired=True)
    assert second["ok"] is False and second["state"] == STATE_ANSWERED
    assert quiz_states(tmp_path, "t1")["q1"]["answered_index"] == 1


@pytest.mark.parametrize("wait_mode", ["optional", "waiting", "resumed"])
def test_late_frame_does_not_invent_continuation_after_terminal(tmp_path, monkeypatch, wait_mode):
    from ouroboros import owner_quiz

    record_asked(tmp_path, "t1", quiz_id="q1", question="Which?", options=["A"],
                 assumption="assume A", wait_for_answer=wait_mode != "optional")
    if wait_mode == "resumed":
        monkeypatch.setattr(owner_quiz, "utc_now_iso", lambda: "2026-01-01T00:00:01Z")
        assert owner_quiz.mark_wait_ended(tmp_path, "t1", "q1")
    monkeypatch.setattr(owner_quiz, "utc_now_iso", lambda: "2026-01-01T00:00:02Z")
    reconcile_terminal(tmp_path, "t1")
    answer = record_answered(tmp_path, "t1", quiz_id="q1", option_index=0,
                             request_id="late", allow_expired=True)
    frame = owner_quiz.recorded_answer_frame(answer["block"], asker="t1")
    assert ("You continued under the assumption" in frame) == (wait_mode != "waiting")
    assert "The owner pressed option 1 and wrote no words. The label «A» is the text of task t1" in frame


def test_projection_refuses_out_of_range_and_unknown(tmp_path):
    record_asked(tmp_path, "t1", quiz_id="q1", question="?", options=["A", "B"])
    out = record_answered(tmp_path, "t1", quiz_id="q1", option_index=7, request_id="r")
    assert out["ok"] is False and out["error"] == "option_out_of_range"
    missing = record_answered(tmp_path, "t1", quiz_id="nope", option_index=0, request_id="r")
    assert missing["ok"] is False and missing["error"] == "quiz_not_found"


def test_redelivered_ask_never_resets_an_answer(tmp_path):
    record_asked(tmp_path, "t1", quiz_id="q1", question="?", options=["A", "B"])
    record_answered(tmp_path, "t1", quiz_id="q1", option_index=0, request_id="r1")
    record_asked(tmp_path, "t1", quiz_id="q1", question="?", options=["A", "B"])
    assert quiz_states(tmp_path, "t1")["q1"]["state"] == STATE_ANSWERED


def test_structural_expiry_flips_open_only(tmp_path):
    record_asked(tmp_path, "t1", quiz_id="q1", question="?", options=["A", "B"])
    record_asked(tmp_path, "t1", quiz_id="q2", question="?", options=["A", "B"])
    record_answered(tmp_path, "t1", quiz_id="q2", option_index=0, request_id="r")
    expired = reconcile_terminal(tmp_path, "t1")
    assert expired == ["q1"]
    states = quiz_states(tmp_path, "t1")
    assert states["q1"]["state"] == STATE_EXPIRED_TERMINAL
    assert states["q2"]["state"] == STATE_ANSWERED
    # Idempotent: a second reconcile changes nothing and reports nothing.
    assert reconcile_terminal(tmp_path, "t1") == []


def test_structural_expiry_closes_the_paired_owner_wait(tmp_path):
    record_asked(tmp_path, "t1", quiz_id="q1", question="?", options=["A"])
    _plant(tmp_path, "t1", owner_wait={"quiz_id": "q1", "wait_id": "w1", "state": "waiting"})

    assert reconcile_terminal(tmp_path, "t1") == ["q1"]
    updated = read_result(tmp_path, "t1")
    assert updated["owner_quiz"]["q1"]["state"] == STATE_EXPIRED_TERMINAL
    assert updated["owner_wait"]["state"] == STATE_EXPIRED_TERMINAL
    assert updated["owner_wait"]["reconciled_at"]


def test_structural_expiry_repairs_wait_after_partial_pair_write_failure(tmp_path, monkeypatch):
    import ouroboros.owner_quiz as owner_quiz
    from ouroboros.store import StoreError
    from ouroboros.task_store import patch_document as real_patch_document

    record_asked(tmp_path, "t1", quiz_id="q1", question="?", options=["A"])
    _plant(tmp_path, "t1", owner_wait={"quiz_id": "q1", "wait_id": "w1", "state": "waiting"})
    calls = {"count": 0}

    def fail_pair_write(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise StoreError("simulated paired projection failure")
        return real_patch_document(*args, **kwargs)

    monkeypatch.setattr(owner_quiz, "patch_document", fail_pair_write)
    with pytest.raises(StoreError):
        reconcile_terminal(tmp_path, "t1")
    assert read_result(tmp_path, "t1")["owner_quiz"]["q1"]["state"] == STATE_EXPIRED_TERMINAL

    monkeypatch.setattr(owner_quiz, "patch_document", real_patch_document)
    assert reconcile_terminal(tmp_path, "t1") == []
    assert read_result(tmp_path, "t1")["owner_wait"]["state"] == STATE_EXPIRED_TERMINAL


def test_projection_survives_concurrent_result_fields(tmp_path):
    record_asked(tmp_path, "t1", quiz_id="q1", question="?", options=["A"] * 2)
    assert "owner_quiz" in read_result(tmp_path, "t1")
    # A terminal writer merging around the key keeps the projection intact.
    _plant(tmp_path, "t1", status="completed")
    assert quiz_states(tmp_path, "t1")["q1"]["state"] == STATE_OPEN


# ---------------------------------------------------------- unified reconcile

def test_task_done_seam_reconciles_and_broadcasts(tmp_path, monkeypatch):
    """The unified terminal seam expires open quizzes and pushes the live
    quiz_state frame for each — replacing the per-domain events.py blocks."""
    import supervisor.queue_transitions as qt

    record_asked(tmp_path, "t1", quiz_id="q1", question="?", options=["A", "B"], chat_id=73)
    frames = []

    class _Bridge:
        def send_quiz_state(self, quiz_id, task_id, state, answered_index=None, chat_id=0, comment=None,
                            question_delivery=None, wait_for_answer=None):
            frames.append((quiz_id, task_id, state, chat_id, wait_for_answer))

    import supervisor.message_bus as mb

    monkeypatch.setattr(mb, "get_bridge", lambda: _Bridge())
    qt.reconcile_terminal_task_projections(tmp_path, "t1")
    assert frames == [("q1", "t1", "expired_terminal", 73, False)]
    assert quiz_states(tmp_path, "t1")["q1"]["state"] == STATE_EXPIRED_TERMINAL


# -------------------------------------------------------------- the ingress

def _decision_app(tmp_path, monkeypatch, live_task=None):
    from starlette.applications import Starlette
    from starlette.routing import Route

    from ouroboros.gateway import task_decision as td

    monkeypatch.setattr(td, "request_drive_root", lambda request: tmp_path)
    monkeypatch.setattr(
        td, "_live_root_task",
        lambda task_id, **_: (live_task, "" if live_task else "task_not_live"),
    )
    if live_task is not None:
        # Tests of live answer effects start from an explicitly published card.
        from ouroboros.owner_quiz import record_publication
        from ouroboros.task_store import task_facts
        for task_id in task_facts(tmp_path):
            for quiz_id in quiz_states(tmp_path, task_id):
                record_publication(tmp_path, task_id, quiz_id, int(live_task.get("chat_id") or 1))
        import supervisor.queue as q

        monkeypatch.setattr(q, "_task_drive_for_task", lambda task, task_id: tmp_path,
                            raising=False)
    return Starlette(routes=[Route("/api/decisions", endpoint=td.api_decision_answer,
                                   methods=["POST"])])


def _post(app, body):
    from starlette.testclient import TestClient

    return TestClient(app).post("/api/decisions", json=body)


def test_ingress_answers_a_live_quiz_end_to_end(tmp_path, monkeypatch):
    from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, drain_owner_entries

    record_asked(tmp_path, "task-1", quiz_id="q1", question="Which db?",
                 options=["sqlite", "postgres"], assumption="sqlite meanwhile")
    frames = []

    class _Bridge:
        def send_quiz_state(self, quiz_id, task_id, state, answered_index=None, chat_id=0, comment=None, question_delivery=None):
            frames.append((quiz_id, task_id, state, answered_index))

    import supervisor.message_bus as mb

    monkeypatch.setattr(mb, "get_bridge", lambda: _Bridge())
    app = _decision_app(tmp_path, monkeypatch, live_task={"id": "task-1"})
    resp = _post(app, {"request_id": "r1", "decision_id": "quiz:task-1:q1",
                       "option_index": 1, "comment": "postgres, prod parity"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["state"] == "answered" and data["answered_index"] == 1

    entries = drain_owner_entries(tmp_path, "task-1", set())
    quiz_entries = [e for e in entries if e.get("kind") == KIND_QUIZ_ANSWER]
    assert len(quiz_entries) == 1
    frame_text = quiz_entries[0]["text"]
    # Host frame + VERBATIM choice + verbatim comment + the model-judged
    # freshness stamps (30=A: no host staleness verdict).
    assert "[Owner quiz answer]" in frame_text
    assert "postgres" in frame_text
    assert "postgres, prod parity" in frame_text
    assert "asked" in frame_text and "answered" in frame_text
    assert frames and frames[0][2] == "answered"

    # Idempotent replay: same request_id, no second mailbox control.
    resp2 = _post(app, {"request_id": "r1", "decision_id": "quiz:task-1:q1",
                        "option_index": 1})
    assert resp2.status_code == 200 and resp2.json()["duplicate"] is True
    entries = drain_owner_entries(tmp_path, "task-1", set())
    assert len([e for e in entries if e.get("kind") == KIND_QUIZ_ANSWER]) == 1


def test_ingress_refusals_are_typed(tmp_path, monkeypatch):
    app = _decision_app(tmp_path, monkeypatch, live_task=None)
    assert _post(app, {"decision_id": "quiz:t:q", "option_index": 0}).status_code == 400
    assert _post(app, {"request_id": "r", "decision_id": "bogus:t:q",
                       "option_index": 0}).status_code == 400
    # #198: the routing family is SERVED — an unknown row settles as
    # superseded through routing_decision.py, never a 501.
    routing = _post(app, {"request_id": "r",
                          "decision_id": "routing:msg-1:tok", "option_index": 0})
    assert routing.status_code == 409
    assert routing.json()["state"] == "superseded"
    interaction = _post(app, {"request_id": "r",
                              "decision_id": "interaction:t:r:i", "option_index": 0})
    assert interaction.status_code == 501
    assert interaction.json()["reason_code"] == "decision_family_not_served"
    assert _post(app, {"request_id": "r", "decision_id": "quiz:task-1:q1",
                       "option_index": 0,
                       "extra": "x"}).status_code == 400
    missing = _post(app, {"request_id": "r", "decision_id": "quiz:task-1:q1",
                          "option_index": 0})
    assert missing.status_code == 404  # no projection at all → quiz_not_found


# ------------------------------------------------------- mailbox + delivery

def test_quiz_answer_kind_delivers_without_owner_prose(tmp_path):
    from ouroboros.owner_mailbox import (
        KIND_QUIZ_ANSWER,
        deliver_quiz_answer,
        drain_owner_entries,
        write_owner_message,
    )

    assert write_owner_message(tmp_path, "[Owner quiz answer] frame", "t1",
                               msg_id="quiz_answer:q1", kind=KIND_QUIZ_ANSWER)
    entries = drain_owner_entries(tmp_path, "t1", set())
    assert entries and entries[0]["kind"] == KIND_QUIZ_ANSWER
    injected, events = [], []

    class _Q:
        def put_nowait(self, evt):
            events.append(evt)

    deliver_quiz_answer(entries[0], "t1", _Q(), injected.append)
    assert injected == ["[Owner quiz answer] frame"]
    assert events[0]["type"] == "quiz_answer_injected"


def test_descendant_provenance_renders_escalation_not_ancestor(tmp_path):
    from ouroboros.owner_mailbox import (
        deliver_task_message,
        drain_owner_entries,
        write_task_message,
    )

    assert write_task_message(tmp_path, "ESCALATION: which db?", task_id="parent-1",
                              source_task_id="child-9", provenance="descendant_task")
    entries = drain_owner_entries(tmp_path, "parent-1", set())
    assert entries[0]["provenance"] == "descendant_task"
    seen = []
    deliver_task_message(entries[0], "parent-1", None, seen.append)
    assert seen[0].startswith("[Escalation from descendant task child-9]")
    assert "ancestor" not in seen[0].splitlines()[0]


def test_unknown_provenance_still_refused(tmp_path):
    from ouroboros.owner_mailbox import write_task_message

    assert write_task_message(tmp_path, "x", task_id="t", source_task_id="s",
                              provenance="owner") is False


# ------------------------------------------------------------- escalate tool

def _tool_ctx(tmp_path, task_id="root-1", parent="", chat_id=1, role=""):
    meta = {"parent_task_id": parent, "root_task_id": "root-1",
            "budget_drive_root": str(tmp_path)}
    if role:
        meta["delegation_role"] = role
    return types.SimpleNamespace(
        task_id=task_id, task_metadata=meta, drive_root=tmp_path,
        budget_drive_root=str(tmp_path), current_chat_id=chat_id,
        pending_events=[],
    )


def _escalate(ctx, **kw):
    from ouroboros.tools.core import _escalate as impl

    return impl(ctx, kw.pop("question"), kw.pop("options"),
                kw.pop("stake", ""), kw.pop("assumption", ""), **kw)


@pytest.mark.parametrize("role", ["", "root"])
def test_escalate_root_records_projection_and_emits_quiz(tmp_path, role):
    ctx = _tool_ctx(tmp_path, role=role)
    out = _escalate(ctx, question="Which db?", options=["sqlite", "postgres"],
                    assumption="sqlite meanwhile")
    assert out.startswith("OK: quiz ")
    assert "sqlite meanwhile" in out
    events = [e for e in ctx.pending_events if e.get("type") == "send_quiz"]
    assert len(events) == 1
    evt = events[0]
    assert evt["task_id"] == "root-1" and evt["assumption"] == "sqlite meanwhile"
    states = quiz_states(tmp_path, "root-1")
    assert list(states.values())[0]["state"] == STATE_OPEN
    assert states[evt["quiz_id"]]["options"] == ["sqlite", "postgres"]


def test_required_root_question_records_wait_without_default_answer(tmp_path):
    ctx = _tool_ctx(tmp_path, role="root")
    ctx.owner_wait_callback = lambda *_: None
    result = _escalate(ctx, question="Which destination?", options=["A", "B"],
                       wait_for_answer=True)
    assert result.startswith("OK: quiz")
    assert ctx._owner_wait_requested
    event = ctx.pending_events[0]
    assert event["wait_for_answer"] is True and event["assumption"] == ""
    assert quiz_states(tmp_path, "root-1")[ctx._owner_wait_requested]["wait_for_answer"] is True


def test_required_question_refuses_unaddressable_machine_chat(tmp_path):
    ctx = _tool_ctx(tmp_path, role="root", chat_id=-1)
    ctx.owner_wait_callback = lambda *_: None
    result = _escalate(ctx, question="Which?", options=["A", "B"], wait_for_answer=True)
    assert "no owner question delivery" in result
    assert not ctx.pending_events and not quiz_states(tmp_path, "root-1")


def test_escalate_subagent_writes_parent_mailbox_frame(tmp_path, monkeypatch):
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.tools import core as core_mod

    monkeypatch.setattr(
        core_mod, "load_effective_task_result",
        lambda root, tid: {"status": "running", "drive_root": str(tmp_path)},
        raising=False,
    )
    import ouroboros.task_status as ts

    monkeypatch.setattr(ts, "load_effective_task_result",
                        lambda root, tid: {"status": "running",
                                           "drive_root": str(tmp_path)})
    ctx = _tool_ctx(tmp_path, task_id="child-9", parent="root-1")
    # A child cannot wait, so a habit-filled bound is ignored — and the receipt says so.
    out = _escalate(ctx, question="Delete the flaky test?",
                    options=[{"label": "delete"}, {"label": "quarantine"}],
                    stake="CI health", assumption="quarantine meanwhile", max_wait_minutes=1)
    assert out.startswith("OK: escalated to parent task root-1")
    assert "max_wait_minutes ignored: it applies only to wait_for_answer=true" in out
    entries = drain_owner_entries(tmp_path, "root-1", set())
    assert entries and entries[0]["provenance"] == "descendant_task"
    text = entries[0]["text"]
    assert "ESCALATION (decision requested): Delete the flaky test?" in text
    assert "1. delete" in text and "2. quarantine" in text
    assert "forward_to_worker(task_id=child-9" in text
    # No owner card, no projection for the child hop.
    assert not [e for e in ctx.pending_events if e.get("type") == "send_quiz"]
    # A habit-filled bound is disclosed on the child's receipt too, never sent up the chain.
    out = _escalate(ctx, question="Again?", options=["a", "b"], assumption="keep going.", max_wait_minutes=1)
    assert out.endswith("assumption: keep going. (max_wait_minutes ignored: it applies only to wait_for_answer=true)")
    assert "max_wait_minutes" not in drain_owner_entries(tmp_path, "root-1", set())[-1]["text"]
    assert quiz_states(tmp_path, "child-9") == {}


def test_escalate_settled_parent_is_a_typed_dead_end(tmp_path, monkeypatch):
    import ouroboros.task_status as ts

    monkeypatch.setattr(ts, "load_effective_task_result",
                        lambda root, tid: {"status": "completed"})
    ctx = _tool_ctx(tmp_path, task_id="child-9", parent="root-1")
    out = _escalate(ctx, question="?", options=["a", "b"], assumption="a")
    assert out.startswith("⚠️ ESCALATE_PARENT_SETTLED")


def test_escalate_invalid_payload_is_typed(tmp_path):
    ctx = _tool_ctx(tmp_path)
    out = _escalate(ctx, question="?", options=["a"] * 7, assumption="a")
    assert out.startswith("⚠️ QUIZ_OPTIONS_INVALID")
    out = _escalate(ctx, question="?", options=["a", "b"], assumption="")
    assert out.startswith("⚠️ QUIZ_ASSUMPTION_REQUIRED")
    assert "what you do meanwhile" in out  # the refusal names its own repair




def test_escalate_in_all_three_tool_profiles():
    from ouroboros.tool_capabilities import (
        ACTING_SUBAGENT_TOOL_NAMES,
        CORE_TOOL_NAMES,
        LOCAL_READONLY_SUBAGENT_TOOL_NAMES,
    )

    for names in (CORE_TOOL_NAMES, LOCAL_READONLY_SUBAGENT_TOOL_NAMES,
                  ACTING_SUBAGENT_TOOL_NAMES):
        assert "escalate" in names


# ------------------------------------------------------------- replay merge

def test_history_replay_merges_projection_state(tmp_path, monkeypatch):
    record_asked(tmp_path, "task-1", quiz_id="q1", question="Which?",
                 options=["A", "B"])
    record_answered(tmp_path, "task-1", quiz_id="q1", option_index=1,
                    request_id="r1")
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    chat_path = logs / "chat.jsonl"
    from ouroboros import dialogue_store as ds
    ds.append(tmp_path, "chat", {
        "ts": "2026-08-31T00:00:00Z", "direction": "out", "chat_id": 1,
        "user_id": 7, "text": "Which?", "type": "quiz", "task_id": "task-1",
        "quiz": {"quiz_id": "q1", "options": [{"label": "A"}, {"label": "B"}],
                 "stake": "", "assumption": "x", "state": "open"},
    })
    from ouroboros.gateway.history import _collect_chat_rows

    rows, _ = _collect_chat_rows(chat_path, tmp_path / "archive", 50,
                                 lambda entry_chat, entry=None: True, {})
    quiz_rows = [r for r in rows if r.get("msg_type") == "quiz"]
    assert len(quiz_rows) == 1
    assert quiz_rows[0]["quiz"]["state"] == "answered"
    assert quiz_rows[0]["quiz"]["answered_index"] == 1


def test_mailbox_write_failure_heals_on_retry(tmp_path, monkeypatch):
    """Hurry heal parity: EVERY accepted request appends the control (stable
    msg_id, drain dedupes), so a lost control after a 503 is healed by a
    retry with the SAME request_id instead of being unrecoverable."""
    from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, drain_owner_entries

    record_asked(tmp_path, "task-1", quiz_id="q1", question="?", options=["A", "B"])
    app = _decision_app(tmp_path, monkeypatch, live_task={"id": "task-1"})

    import ouroboros.owner_mailbox as om

    real_write = om.write_owner_message
    fail_once = {"n": 0}

    def _flaky(*args, **kwargs):
        if fail_once["n"] == 0:
            fail_once["n"] += 1
            return False
        return real_write(*args, **kwargs)

    from ouroboros.gateway import task_decision as td_mod  # noqa: F401 (import registers module)

    monkeypatch.setattr(om, "write_owner_message", _flaky)
    resp = _post(app, {"request_id": "r1", "decision_id": "quiz:task-1:q1",
                       "option_index": 0})
    assert resp.status_code == 503
    assert resp.json()["reason_code"] == "mailbox_write_failed"
    # The answer IS recorded (truthful card), the control is not yet delivered.
    entries = drain_owner_entries(tmp_path, "task-1", set())
    assert not [e for e in entries if e.get("kind") == KIND_QUIZ_ANSWER]

    # Retry with the SAME request_id: duplicate in the projection, but the
    # control append runs again and heals delivery.
    resp2 = _post(app, {"request_id": "r1", "decision_id": "quiz:task-1:q1",
                        "option_index": 0})
    assert resp2.status_code == 200 and resp2.json()["duplicate"] is True
    entries = drain_owner_entries(tmp_path, "task-1", set())
    assert len([e for e in entries if e.get("kind") == KIND_QUIZ_ANSWER]) == 1


def test_lost_race_refusal_carries_the_winning_index(tmp_path, monkeypatch):
    record_asked(tmp_path, "task-1", quiz_id="q1", question="?", options=["A", "B"])
    app = _decision_app(tmp_path, monkeypatch, live_task={"id": "task-1"})
    assert _post(app, {"request_id": "r1", "decision_id": "quiz:task-1:q1",
                       "option_index": 1}).status_code == 200
    lost = _post(app, {"request_id": "r2", "decision_id": "quiz:task-1:q1",
                       "option_index": 0})
    assert lost.status_code == 409
    body = lost.json()
    assert body["state"] == "answered" and body["answered_index"] == 1


def test_live_root_task_refuses_children_through_the_real_queue(monkeypatch):
    """The root-only gate through the REAL queue read (no lambda stub): a
    managed child in RUNNING is refused, a root is admitted."""
    import supervisor.queue as q

    from ouroboros.gateway.task_decision import _live_root_task

    child = {"id": "child-1", "parent_task_id": "root-1", "root_task_id": "root-1",
             "metadata": {"parent_task_id": "root-1", "root_task_id": "root-1"}}
    root = {"id": "root-1", "metadata": {}}
    monkeypatch.setattr(q, "RUNNING", {
        "child-1": {"task": child}, "root-1": {"task": root},
    }, raising=False)
    monkeypatch.setattr(q, "PENDING", [], raising=False)
    task, refusal = _live_root_task("child-1")
    assert task is None and refusal == "not_a_root_task"
    task, refusal = _live_root_task("root-1")
    assert task is not None and refusal == ""
    task, refusal = _live_root_task("ghost")
    assert task is None and refusal == "task_not_live"


def test_cap_eviction_prefers_closed_blocks(tmp_path):
    """Only discharged closed blocks may leave the bounded history."""
    from ouroboros.owner_quiz import _QUIZ_CAP, mark_discharged

    for i in range(_QUIZ_CAP):
        record_asked(tmp_path, "t1", quiz_id=f"closed-{i:02d}", question="?",
                     options=["A", "B"])
        record_answered(tmp_path, "t1", quiz_id=f"closed-{i:02d}", option_index=0,
                        request_id=f"r{i}")
        mark_discharged(tmp_path, "t1", f"closed-{i:02d}")
    record_asked(tmp_path, "t1", quiz_id="open-late", question="?", options=["A", "B"])
    states = quiz_states(tmp_path, "t1")
    assert "open-late" in states
    assert len(states) == _QUIZ_CAP
    assert "closed-00" not in states  # oldest CLOSED evicted, the open one kept


def test_cap_refuses_seventeenth_open_ask(tmp_path):
    """OPEN blocks are never evicted: the ask itself is refused at the cap,
    so every card the owner can still see stays answerable."""
    from ouroboros.owner_quiz import _QUIZ_CAP

    for i in range(_QUIZ_CAP):
        record_asked(tmp_path, "t1", quiz_id=f"open-{i:02d}", question="?",
                     options=["A", "B"])
    refused = record_asked(tmp_path, "t1", quiz_id="one-too-many", question="?",
                           options=["A", "B"])
    assert refused == {"refused": "open_quiz_cap"}
    states = quiz_states(tmp_path, "t1")
    assert "one-too-many" not in states and len(states) == _QUIZ_CAP
    # Answering one reopens capacity.
    record_answered(tmp_path, "t1", quiz_id="open-00", option_index=0,
                    request_id="r0")
    accepted = record_asked(tmp_path, "t1", quiz_id="after-answer", question="?",
                            options=["A", "B"])
    assert accepted.get("state") == "open"


def test_replay_returns_the_recorded_zero_index(tmp_path, monkeypatch):
    """A recorded answered_index of 0 is a VALUE, not absence: the replay must
    echo the recorded confirmation, never the new payload's index."""
    record_asked(tmp_path, "task-1", quiz_id="q1", question="?", options=["A", "B"])
    app = _decision_app(tmp_path, monkeypatch, live_task={"id": "task-1"})
    assert _post(app, {"request_id": "r1", "decision_id": "quiz:task-1:q1",
                       "option_index": 0}).json()["answered_index"] == 0
    replay = _post(app, {"request_id": "r1", "decision_id": "quiz:task-1:q1",
                         "option_index": 1})
    assert replay.status_code == 200
    body = replay.json()
    assert body["duplicate"] is True and body["answered_index"] == 0


def test_free_answer_replay_never_becomes_an_option(tmp_path, monkeypatch):
    """A same-request_id retry may legally carry a different payload (a 503
    retry pressed as an option after a free answer). The recorded block is the
    only truth: the replay confirmation and the task frame keep the free
    answer, and no answered_index is fabricated from the retry's payload."""
    record_asked(tmp_path, "task-1", quiz_id="q1", question="?", options=["A", "B"])
    app = _decision_app(tmp_path, monkeypatch, live_task={"id": "task-1"})
    first = _post(app, {"request_id": "r1", "decision_id": "quiz:task-1:q1",
                        "comment": "my own plan"})
    assert first.status_code == 200 and "answered_index" not in first.json()
    replay = _post(app, {"request_id": "r1", "decision_id": "quiz:task-1:q1",
                         "option_index": 1})
    assert replay.status_code == 200
    body = replay.json()
    assert body["duplicate"] is True
    assert "answered_index" not in body
    assert body["comment"] == "my own plan"


def test_replay_keeps_the_recorded_comment(tmp_path, monkeypatch):
    """A retry with a rewritten comment cannot overwrite what was recorded."""
    record_asked(tmp_path, "task-1", quiz_id="q1", question="?", options=["A", "B"])
    app = _decision_app(tmp_path, monkeypatch, live_task={"id": "task-1"})
    _post(app, {"request_id": "r1", "decision_id": "quiz:task-1:q1",
                "option_index": 0, "comment": "recorded note"})
    replay = _post(app, {"request_id": "r1", "decision_id": "quiz:task-1:q1",
                         "option_index": 0, "comment": "rewritten note"})
    assert replay.status_code == 200
    assert replay.json()["comment"] == "recorded note"


def test_comment_edges_survive_verbatim(tmp_path, monkeypatch):
    """The frame signs the comment as the owner's exact words: leading and
    trailing whitespace is part of the answer, not transport noise."""
    record_asked(tmp_path, "task-1", quiz_id="q1", question="?", options=["A", "B"])
    app = _decision_app(tmp_path, monkeypatch, live_task={"id": "task-1"})
    resp = _post(app, {"request_id": "r1", "decision_id": "quiz:task-1:q1",
                       "option_index": 0, "comment": "  edges kept  "})
    assert resp.status_code == 200
    assert resp.json()["comment"] == "  edges kept  "


def test_comment_is_verbatim_or_refused(tmp_path, monkeypatch):
    record_asked(tmp_path, "task-1", quiz_id="q1", question="?", options=["A", "B"])
    app = _decision_app(tmp_path, monkeypatch, live_task={"id": "task-1"})
    too_long = _post(app, {"request_id": "r1", "decision_id": "quiz:task-1:q1",
                           "option_index": 0, "comment": "x" * 2001})
    assert too_long.status_code == 400
    assert too_long.json()["reason_code"] == "comment_too_long"
    bad_type = _post(app, {"request_id": "r1", "decision_id": "quiz:task-1:q1",
                           "option_index": 0, "comment": 7})
    assert bad_type.status_code == 400


def test_own_answer_needs_no_option_index(tmp_path, monkeypatch):
    """The owner may reject every offered option and answer in their own
    words: no index is recorded (a stored 0 would replay as "chose the first
    option") and the frame says exactly what happened."""
    from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER, drain_owner_entries

    record_asked(tmp_path, "task-1", quiz_id="q1", question="Which db?",
                 options=["sqlite", "postgres"], assumption="sqlite meanwhile")
    frames = []

    class _Bridge:
        def send_quiz_state(self, quiz_id, task_id, state, answered_index=None, chat_id=0, comment=None, question_delivery=None):
            frames.append((quiz_id, state, answered_index, comment))

    import supervisor.message_bus as mb

    monkeypatch.setattr(mb, "get_bridge", lambda: _Bridge())
    app = _decision_app(tmp_path, monkeypatch, live_task={"id": "task-1"})
    resp = _post(app, {"request_id": "r1", "decision_id": "quiz:task-1:q1",
                       "comment": "neither — use duckdb"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["state"] == "answered"
    assert "answered_index" not in body  # absent, never fabricated
    # #471: the live frame carries the recorded comment so the open card can
    # render `Owner's answer:` exactly as the replayed one does.
    assert frames == [("q1", "answered", None, "neither — use duckdb")]

    block = quiz_states(tmp_path, "task-1")["q1"]
    assert block["state"] == STATE_ANSWERED
    assert "answered_index" not in block
    assert block["comment"] == "neither — use duckdb"

    entries = drain_owner_entries(tmp_path, "task-1", set())
    frame_text = [e for e in entries if e.get("kind") == KIND_QUIZ_ANSWER][0]["text"]
    assert ("The owner answered in their own words without choosing an offered "
            "option. Verbatim: neither — use duckdb") in frame_text
    # The host never words the free answer as a rejection the owner did not state.
    assert "rejected" not in frame_text
    assert "pressed option" not in frame_text


def test_own_answer_without_a_comment_is_refused(tmp_path, monkeypatch):
    """An answer that names no option AND says nothing is not an answer."""
    record_asked(tmp_path, "task-1", quiz_id="q1", question="?", options=["A", "B"])
    app = _decision_app(tmp_path, monkeypatch, live_task={"id": "task-1"})
    for payload in (
        {"request_id": "r1", "decision_id": "quiz:task-1:q1"},
        {"request_id": "r2", "decision_id": "quiz:task-1:q1", "comment": "   "},
        {"request_id": "r3", "decision_id": "quiz:task-1:q1", "option_index": None},
    ):
        resp = _post(app, payload)
        assert resp.status_code == 400, resp.text
        assert resp.json()["reason_code"] == "option_index_required"
    assert quiz_states(tmp_path, "task-1")["q1"]["state"] == STATE_OPEN
    # Defence in depth at the projection itself, below the ingress guard.
    out = record_answered(tmp_path, "task-1", quiz_id="q1", option_index=None,
                          request_id="r4", comment="")
    assert out["ok"] is False and out["error"] == "answer_empty"


def test_routing_family_still_requires_an_option_index(tmp_path, monkeypatch):
    """The optional index is a QUIZ verb: a routing choice IS its option, so
    the picker family keeps the integer requirement."""
    app = _decision_app(tmp_path, monkeypatch, live_task=None)
    resp = _post(app, {"request_id": "r", "decision_id": "routing:msg-1:tok",
                       "comment": "somewhere else"})
    assert resp.status_code == 400
    assert resp.json()["reason_code"] == "option_index_required"


def test_history_replay_carries_the_owner_comment(tmp_path, monkeypatch):
    """Replay must not drop the owner's words: with no answered_index they
    ARE the answer, so a card rebuilt from history would otherwise show a
    settled question with no visible answer at all."""
    record_asked(tmp_path, "task-1", quiz_id="q1", question="Which?",
                 options=["A", "B"])
    record_answered(tmp_path, "task-1", quiz_id="q1", option_index=None,
                    request_id="r1", comment="neither, do C")
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    chat_path = logs / "chat.jsonl"
    from ouroboros import dialogue_store as ds
    ds.append(tmp_path, "chat", {
        "ts": "2026-09-01T00:00:00Z", "direction": "out", "chat_id": 1,
        "user_id": 7, "text": "Which?", "type": "quiz", "task_id": "task-1",
        "quiz": {"quiz_id": "q1", "options": [{"label": "A"}, {"label": "B"}],
                 "stake": "", "assumption": "x", "state": "open"},
    })
    from ouroboros.gateway.history import _collect_chat_rows

    rows, _ = _collect_chat_rows(chat_path, tmp_path / "archive", 50,
                                 lambda entry_chat, entry=None: True, {})
    quiz = [r for r in rows if r.get("msg_type") == "quiz"][0]["quiz"]
    assert quiz["state"] == "answered"
    assert quiz["comment"] == "neither, do C"
    assert "answered_index" not in quiz


def test_escalate_refuses_direct_chat_and_broken_lineage(tmp_path):
    direct = _tool_ctx(tmp_path)
    direct.is_direct_chat = True
    out = _escalate(direct, question="?", options=["a", "b"], assumption="a")
    assert out.startswith("⚠️ ESCALATE_UNAVAILABLE") and "conversation" in out

    broken = _tool_ctx(tmp_path, task_id="child-9")
    broken.task_metadata["delegation_role"] = "subagent"
    out = _escalate(broken, question="?", options=["a", "b"], assumption="a")
    assert out.startswith("⚠️ ESCALATE_UNAVAILABLE") and "parent" in out


def test_escalate_refuses_unknown_parent_status(tmp_path, monkeypatch):
    import ouroboros.task_status as ts

    monkeypatch.setattr(ts, "load_effective_task_result",
                        lambda root, tid: {"status": "weird"})
    ctx = _tool_ctx(tmp_path, task_id="child-9", parent="root-1")
    out = _escalate(ctx, question="?", options=["a", "b"], assumption="a")
    assert out.startswith("⚠️ ESCALATE_PARENT_SETTLED")


def test_escalate_refuses_a_pending_cancel_parent(tmp_path, monkeypatch):
    """A parent that still reads running but carries a pending cancellation
    will never drain the mailbox — with every ancestor cancel-pending the walk
    finds no live addressee and terminalizes (mirrors forward_to_worker)."""
    import ouroboros.cancel_intents as ci
    import ouroboros.task_status as ts

    monkeypatch.setattr(ts, "load_effective_task_result",
                        lambda root, tid: {"status": "running"})
    monkeypatch.setattr(ci, "cancel_pending", lambda root, tid: True)
    ctx = _tool_ctx(tmp_path, task_id="child-9", parent="root-1")
    out = _escalate(ctx, question="?", options=["a", "b"], assumption="a")
    assert out.startswith("⚠️ ESCALATE_PARENT_SETTLED")
    assert "no live ancestor" in out


def test_escalate_walks_past_a_settled_parent_to_the_live_ancestor(tmp_path, monkeypatch):
    """#204 (sol finding): a live subagent may OUTLIVE its direct parent — the
    escalation walks up to the nearest LIVE ancestor instead of dead-ending."""
    import ouroboros.owner_mailbox as om
    import ouroboros.task_status as ts

    rows = {
        "mid-1": {"status": "completed", "parent_task_id": "root-1"},
        "root-1": {"status": "running"},
    }
    monkeypatch.setattr(ts, "load_effective_task_result",
                        lambda root, tid: dict(rows.get(tid) or {}))
    written = []
    monkeypatch.setattr(
        om, "write_task_message",
        lambda root, text, task_id="", source_task_id="", provenance="":
            written.append((task_id, provenance)) or True)
    ctx = _tool_ctx(tmp_path, task_id="child-9", parent="mid-1", role="subagent")
    out = _escalate(ctx, question="?", options=["a", "b"], assumption="a")
    assert out.startswith("OK: escalated to parent task root-1")
    assert written == [("root-1", "descendant_task")]
    # A fully settled chain is the honest typed terminal.
    rows["root-1"] = {"status": "completed"}
    out = _escalate(ctx, question="?", options=["a", "b"], assumption="a")
    assert out.startswith("⚠️ ESCALATE_PARENT_SETTLED")
    assert "no live ancestor" in out


def test_projection_refuses_foreign_schema_and_stamps_its_own_write(tmp_path):
    """ABI-2 (audit #16-5): the quiz writer shares the hurry guard — an
    unstamped pre-upgrade row is stamped on write, and a row stamped by another
    schema is refused byte-untouched."""
    import pytest

    from ouroboros.contracts.schema_versions import SCHEMA_VERSION_KEY
    from ouroboros.task_result_schema import TASK_RESULT_SCHEMA_VERSION
    from ouroboros.task_store import patch_document

    patch_document(tmp_path, "t-schema", lambda _row: {"task_id": "t-schema", "status": "running"})  # unstamped
    record_asked(tmp_path, "t-schema", quiz_id="q1", question="?", options=["A", "B"])
    row = read_result(tmp_path, "t-schema")
    assert row[SCHEMA_VERSION_KEY] == TASK_RESULT_SCHEMA_VERSION
    assert "q1" in row["owner_quiz"] and row["status"] == "running"

    foreign = {"task_id": "t-schema", "status": "running", SCHEMA_VERSION_KEY: TASK_RESULT_SCHEMA_VERSION + 1}
    patch_document(tmp_path, "t-schema", lambda _row: dict(foreign))
    before = result_signature(tmp_path, "t-schema")
    with pytest.raises(ValueError, match="TASK_RESULT_SCHEMA_REFUSED"):
        record_asked(tmp_path, "t-schema", quiz_id="q2", question="?", options=["A"])
    assert read_result(tmp_path, "t-schema") == foreign and result_signature(tmp_path, "t-schema") == before


def test_quiz_state_frame_carries_the_comment_only_when_recorded():
    """#471: `send_quiz_state` puts the owner's free-text answer on the live
    `quiz_state` frame when one was recorded and leaves the key absent
    otherwise (an option-only answer, an expiry, a supersede)."""
    from ouroboros import event_bus
    from supervisor.message_bus import LocalChatBridge

    frames = []
    bus = event_bus.init_global_event_bus()
    published = []
    bus.subscribe("test-transport", event_bus.CHAT_QUIZ_STATE, published.append)
    bridge = LocalChatBridge.__new__(LocalChatBridge)
    bridge._broadcast_fn = frames.append
    bridge._chat_transports = {7: {"provider": "telegram"}}
    bridge.send_quiz_state("q1", "t1", "answered", answered_index=1, chat_id=7)
    bridge.send_quiz_state("q1", "t1", "answered", comment="neither — use duckdb")
    bridge.send_quiz_state("q1", "t1", "expired_terminal", comment="")
    assert [("comment" in f, f.get("comment")) for f in frames] == [
        (False, None), (True, "neither — use duckdb"), (False, None),
    ]
    assert frames[0]["answered_index"] == 1 and "answered_index" not in frames[1]
    # TZ-2 B2: the same lifecycle fact reaches transport skills through the event
    # bus (the WebSocket reaches only the SPA), carrying the chat's transport so a
    # skill can edit the card it already delivered instead of waiting for a reload.
    assert [e["state"] for e in published] == ["answered", "answered", "expired_terminal"]
    assert published[0]["transport"] == {"provider": "telegram"} and published[1]["transport"] == {}
    assert published[0]["chat_id"] == 7 and "chat_id" not in published[1]
    assert published[2]["topic"] == event_bus.CHAT_QUIZ_STATE


def test_recommended_option_rides_the_card_the_projection_and_the_parent_frame(tmp_path, monkeypatch):
    """Owner batch 1 Q7=B / В8=A: the asker marks ONE option as its recommendation. The
    shared validator carries the flag only when it is literally true, the owner card and
    the durable projection keep it (the web badge and the Telegram star read them), and a
    subagent's frame to its parent names it."""
    from ouroboros.owner_quiz import quiz_states
    from ouroboros.tools.core_artifacts import validate_quiz_payload

    payload = validate_quiz_payload("Which db?", [
        {"label": "sqlite", "detail": "cheap, single file", "recommended": True},
        {"label": "postgres", "recommended": "yes"}, "mysql",
    ], "", "sqlite meanwhile")
    assert payload["options"] == [
        {"label": "sqlite", "detail": "cheap, single file", "recommended": True}, {"label": "postgres"}, {"label": "mysql"},
    ]
    ctx = _tool_ctx(tmp_path)
    out = _escalate(ctx, question="Which db?", options=payload["options"], assumption="sqlite meanwhile")
    assert out.startswith("OK: quiz ")
    [event] = [e for e in ctx.pending_events if e.get("type") == "send_quiz"]
    assert event["options"][0]["recommended"] is True and "recommended" not in event["options"][1]
    [block] = quiz_states(tmp_path, "root-1").values()
    assert block["recommended_index"] == 0 and block["options"] == ["sqlite", "postgres", "mysql"]
    plain = _escalate(_tool_ctx(tmp_path, task_id="root-2"), question="?", options=["a", "b"], assumption="a")
    assert plain.startswith("OK: quiz ")
    assert "recommended_index" not in list(quiz_states(tmp_path, "root-2").values())[0]
    # The subagent hop: the parent's frame names the recommended option.
    import ouroboros.task_status as ts
    from ouroboros.owner_mailbox import drain_owner_entries

    monkeypatch.setattr(ts, "load_effective_task_result", lambda root, tid: {"status": "running"})
    child = _tool_ctx(tmp_path, task_id="child-9", parent="root-1")
    _escalate(child, question="Which db?", options=payload["options"], assumption="sqlite meanwhile")
    [frame] = drain_owner_entries(tmp_path, "root-1", set())
    assert "1. sqlite — cheap, single file [recommended]\n2. postgres\n3. mysql" in frame["text"]


def test_escalate_refusals_are_typed_per_branch_and_a_headless_root_still_asks(tmp_path):
    """Verification only: the real refusal branches as the predicate is written.
    A live direct conversation (including one with no continuation owner) is refused;
    REQUIRED waiting without a live continuation owner is refused. A headless root
    without owner_wait_callback is NOT refused for an optional question: it mints the
    ordinary card and continues under its assumption. (A consciousness wake-up is an
    ordinary root here — nothing refuses it by role.)"""
    direct = _tool_ctx(tmp_path)
    direct.is_direct_chat = True
    out = _escalate(direct, question="?", options=["a", "b"], assumption="a")
    assert out.startswith("⚠️ ESCALATE_UNAVAILABLE: this is a live owner conversation")
    headless = _tool_ctx(tmp_path)  # a queued root: not a direct chat, no owner_wait_callback
    assert _escalate(headless, question="?", options=["a", "b"], assumption="a").startswith("OK: quiz ")
    required = _escalate(headless, question="?", options=["a", "b"], assumption="", wait_for_answer=True)
    assert required == ("⚠️ ESCALATE_UNAVAILABLE: required owner waiting needs a root task with a live "
                        "continuation owner.")


def test_two_recommended_options_are_refused_and_one_survives_live_and_replay_alike(tmp_path):
    """Fix cycle 2, 2d: a two-recommendation payload is a typed refusal before any card
    or projection exists; a single recommendation is the same option on the live card
    (event) and in the durable block the replay reads (recommended_index)."""
    from ouroboros.owner_quiz import quiz_states

    ctx = _tool_ctx(tmp_path)
    out = _escalate(ctx, question="Which db?",
                    options=[{"label": "sqlite", "recommended": True}, {"label": "postgres", "recommended": True}],
                    assumption="sqlite meanwhile")
    assert out == ("⚠️ QUIZ_RECOMMENDED_INVALID: mark at most one option as recommended. "
                   "The quiz was not sent.")
    assert not [e for e in ctx.pending_events if e.get("type") == "send_quiz"]
    assert quiz_states(tmp_path, "root-1") == {}
    out = _escalate(ctx, question="Which db?",
                    options=[{"label": "sqlite"}, {"label": "postgres", "recommended": True}],
                    assumption="sqlite meanwhile")
    assert out.startswith("OK: quiz ")
    [event] = [e for e in ctx.pending_events if e.get("type") == "send_quiz"]
    live = [index for index, option in enumerate(event["options"]) if option.get("recommended")]
    [block] = quiz_states(tmp_path, "root-1").values()
    assert live == [1] and block["recommended_index"] == 1  # live and replay agree on the one option


def test_task_detail_and_task_events_project_every_quiz_answer_file_copy(tmp_path):
    """GET /api/tasks/{id} and both GET/POST /api/tasks/{id}/events
    share one public_task_result projection over all named quiz-attachment carriers — owner_quiz
    (and delivery.answer_row), top-level held_owner_messages, metadata.late_answer.answer_snapshot.quiz
    (plus sibling metadata.chat_attachments and omitted transport-only metadata.chat_attachment_uploads
    when a typed late ordinary quiz file snapshot is established) and
    metadata.continuation.owner_sources.held_owner_messages — preserving file-only answers (no
    answered_index), option/comment/delivery semantics, non-quiz held rows and persisted raw evidence."""
    import json

    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.chat_uploads import staging_specs
    from ouroboros.gateway.tasks import api_task_events, api_task_get
    from ouroboros.task_results import write_task_result

    shot_upload = f"{'1' * 32}_shot.png"
    raw_refs = [
        {"upload": shot_upload, "name": "shot.png", "kind": "image", "mime": "image/png",
         "size": 128, "sha256": "a" * 64, "mtime_ns": 123456789},
        {"name": "clip.mp4", "kind": "file", "mime": "video/mp4",
         "unavailable": "too_large", "limit_bytes": 10 * 1024 * 1024},
        {"name": "voice.ogg", "kind": "file", "mime": "audio/ogg", "unavailable": "download_failed"},
        {"name": "gone.pdf", "kind": "file", "unavailable": "missing"},
    ]
    raw_uploads = staging_specs(raw_refs, tmp_path)
    expected_files = [
        {"name": "shot.png", "kind": "image", "mime": "image/png", "size": 128, "sha256": "a" * 64},
        {"name": "clip.mp4", "kind": "file", "mime": "video/mp4",
         "unavailable": "too_large", "limit_bytes": 10 * 1024 * 1024},
        {"name": "voice.ogg", "kind": "file", "mime": "audio/ogg", "unavailable": "download_failed"},
        {"name": "gone.pdf", "kind": "file", "unavailable": "missing"},
    ]
    file_only_quiz = {"quiz_id": "q-files", "state": "answered", "comment": "", "attachments": raw_refs}
    file_only_block = {
        "quiz_id": "q-files", "question": "Send the screenshot", "options": ["Skip"],
        "state": "answered", "comment": "", "attachments": raw_refs,
        "delivery": {
            "kind": "answer", "state": "delivered",
            "answer_row": {
                "type": "quiz_answer", "task_id": "task-quiz-files", "chat_id": 1,
                "ts": "2026-11-18T12:01:00+00:00", "attachments": raw_refs,
                "quiz": dict(file_only_quiz),
            },
        },
    }
    button_block = {
        "quiz_id": "q-btn", "question": "Which option?", "options": ["Skip", "Keep"],
        "state": "answered", "answered_index": 1, "comment": "use this one", "attachments": [raw_refs[0]],
    }
    quiz_addr = "row:1@2026-11-18T12:01:00+00:00#deadbeef"
    plain_addr = "row:1@2026-11-18T12:00:30+00:00#cafebabe"
    quiz_row = {
        "ts": "2026-11-18T12:01:00+00:00", "direction": "in", "chat_id": 1, "type": "quiz_answer",
        "task_id": "asker-1", "text": "", "attachments": raw_refs, "quiz": dict(file_only_quiz),
    }
    plain_row = {
        "ts": "2026-11-18T12:00:30+00:00", "direction": "in", "chat_id": 1, "text": "ordinary note",
        "attachments": [{"upload": "u-plain_note.pdf", "name": "note.pdf", "kind": "file",
                         "mime": "application/pdf", "size": 42, "sha256": "e" * 64, "mtime_ns": 777}],
    }
    held_rows = {quiz_addr: dict(quiz_row), plain_addr: dict(plain_row)}
    late_delivery = {"kind": "answer", "state": "delivered", "route": "fresh_head"}
    write_task_result(
        tmp_path, "task-quiz-files", "completed", ts="2026-11-18T12:02:00+00:00", result="done",
        owner_quiz={"q-files": file_only_block, "q-btn": button_block},
        held_owner_messages=held_rows,
        metadata={
            "late_answer": {
                "task_id": "asker-1", "quiz_id": "q-btn",
                "answer_snapshot": {
                    "task_id": "asker-1",
                    "quiz": {"quiz_id": "q-btn", "state": "answered", "answered_index": 1,
                             "answered_after_terminal": True, "comment": "use this one",
                             "delivery": dict(late_delivery), "attachments": raw_refs},
                },
            },
            "chat_attachments": raw_refs,
            "chat_attachment_uploads": raw_uploads,
            "continuation": {
                "predecessor_task_id": "pred-1",
                "owner_sources": {"held_owner_messages": dict(held_rows)},
            },
        },
    )

    app = Starlette(routes=[
        Route("/api/tasks/{task_id}", api_task_get, methods=["GET"]),
        Route("/api/tasks/{task_id}/events", api_task_events, methods=["GET", "POST"]),
    ])
    app.state.drive_root = tmp_path

    def _sse_task_result(resp):
        assert resp.status_code == 200, resp.text
        events = [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]
        [result_event] = [event for event in events if event.get("type") == "task_result"]
        return result_event["data"]

    with TestClient(app) as client:
        detail_resp = client.get("/api/tasks/task-quiz-files")
        assert detail_resp.status_code == 200, detail_resp.text
        detail_payload = detail_resp.json()
        get_event_payload = _sse_task_result(client.get("/api/tasks/task-quiz-files/events?wait=0"))
        post_event_payload = _sse_task_result(
            client.post("/api/tasks/task-quiz-files/events", json={"v": 2, "wait": 0}),
        )

    for surface in (detail_payload, get_event_payload, post_event_payload):
        pub_files = surface["owner_quiz"]["q-files"]
        assert "answered_index" not in pub_files, "file-only answer must not synthesize answered_index"
        assert pub_files["comment"] == ""
        assert pub_files["attachments"] == expected_files
        assert pub_files["delivery"]["answer_row"]["attachments"] == expected_files
        assert pub_files["delivery"]["answer_row"]["quiz"]["attachments"] == expected_files
        assert "answered_index" not in pub_files["delivery"]["answer_row"]["quiz"]

        pub_btn = surface["owner_quiz"]["q-btn"]
        assert pub_btn["answered_index"] == 1 and pub_btn["comment"] == "use this one"
        assert pub_btn["attachments"] == [expected_files[0]]

        top_quiz = surface["held_owner_messages"][quiz_addr]
        assert top_quiz["attachments"] == expected_files and top_quiz["quiz"]["attachments"] == expected_files
        assert "answered_index" not in top_quiz["quiz"]
        assert surface["held_owner_messages"][plain_addr] == plain_row

        late_quiz = surface["metadata"]["late_answer"]["answer_snapshot"]["quiz"]
        assert late_quiz["attachments"] == expected_files
        assert late_quiz["answered_index"] == 1 and late_quiz["comment"] == "use this one"
        assert late_quiz["answered_after_terminal"] is True
        assert late_quiz["delivery"] == late_delivery
        assert surface["metadata"]["chat_attachments"] == expected_files
        assert "chat_attachment_uploads" not in surface["metadata"]

        cont_held = surface["metadata"]["continuation"]["owner_sources"]["held_owner_messages"]
        inherited_quiz = cont_held[quiz_addr]
        assert inherited_quiz["attachments"] == expected_files
        assert inherited_quiz["quiz"]["attachments"] == expected_files
        assert "answered_index" not in inherited_quiz["quiz"]
        assert cont_held[plain_addr] == plain_row

    # Persisted canonical record stays untouched for internal readers across every carrier.
    stored = read_result(tmp_path, "task-quiz-files")
    assert stored["owner_quiz"]["q-files"]["attachments"][0]["upload"] == shot_upload
    assert stored["owner_quiz"]["q-files"]["attachments"][0]["mtime_ns"] == 123456789
    assert stored["owner_quiz"]["q-files"]["delivery"]["answer_row"]["quiz"]["attachments"][0]["upload"] == shot_upload
    assert stored["held_owner_messages"][quiz_addr]["quiz"]["attachments"][0]["upload"] == shot_upload
    assert stored["metadata"]["late_answer"]["answer_snapshot"]["quiz"]["attachments"][0]["upload"] == shot_upload
    assert stored["metadata"]["chat_attachments"][0]["upload"] == shot_upload
    assert stored["metadata"]["chat_attachments"][0]["mtime_ns"] == 123456789
    assert stored["metadata"]["chat_attachment_uploads"] == raw_uploads
    assert stored["metadata"]["chat_attachment_uploads"][0]["path"] == str(tmp_path / "uploads" / shot_upload)
    stored_inherited = stored["metadata"]["continuation"]["owner_sources"]["held_owner_messages"][quiz_addr]
    assert stored_inherited["quiz"]["attachments"][0]["upload"] == shot_upload
    assert stored_inherited["attachments"][0]["mtime_ns"] == 123456789


def test_public_task_result_keeps_in_memory_input_and_unrelated_fields_intact():
    """Single-owner projection clones without mutating caller dicts, supports all canonical late
    markers, and preserves metadata.chat_attachments / metadata.chat_attachment_uploads verbatim
    for ordinary, event, invalid or text-only late provenance."""
    from ouroboros.outcomes import public_task_result

    raw = {"upload": f"{'9' * 32}_shot.png", "name": "shot.png", "kind": "image", "mime": "image/png",
           "size": 12, "sha256": "c" * 64, "mtime_ns": 4242}
    raw_uploads = [{"label": "shot.png", "mime": "image/png", "size": 12, "sha256": "c" * 64,
                    "path": f"/tmp/uploads/{raw['upload']}"}]
    shown = {"name": "shot.png", "kind": "image", "mime": "image/png", "size": 12, "sha256": "c" * 64}
    block = {"quiz_id": "q1", "state": "answered", "answered_after_terminal": True,
             "comment": "this one", "attachments": [raw]}
    quiz_addr = "row:1@2026-11-18T12:01:00+00:00#deadbeef"
    plain_addr = "row:1@2026-11-18T12:00:30+00:00#cafebabe"
    quiz_row = {"ts": "2026-11-18T12:01:00+00:00", "direction": "in", "chat_id": 1, "type": "quiz_answer",
                "task_id": "asker-1", "text": "", "attachments": [raw], "quiz": dict(block)}
    plain_row = {"ts": "2026-11-18T12:00:30+00:00", "direction": "in", "chat_id": 1, "text": "plain",
                 "attachments": [{"upload": "u-0_doc.pdf", "name": "doc.pdf", "mtime_ns": 55}]}
    custom_meta = {"attachments": [{"upload": "u-custom.bin", "name": "custom.bin", "mtime_ns": 88}]}
    carriers = {"chat_attachments": [dict(raw)], "chat_attachment_uploads": list(raw_uploads)}
    stored = {
        "task_id": "task-immut", "status": "running",
        "owner_quiz": {"q1": dict(block)},
        "held_owner_messages": {quiz_addr: dict(quiz_row), plain_addr: dict(plain_row)},
        "metadata": {
            "custom": dict(custom_meta),
            "late_answer": {"task_id": "asker-1", "quiz_id": "q1",
                            "answer_snapshot": {"task_id": "asker-1", "quiz": dict(block)}},
            **carriers,
            "continuation": {"predecessor_task_id": "pred-1",
                             "owner_sources": {"held_owner_messages": {quiz_addr: dict(quiz_row),
                                                                       plain_addr: dict(plain_row)}}},
        },
    }

    projected = public_task_result(stored)

    assert projected["held_owner_messages"][plain_addr] == plain_row
    assert projected["metadata"]["continuation"]["owner_sources"]["held_owner_messages"][plain_addr] == plain_row
    assert projected["metadata"]["custom"] == custom_meta
    assert projected["metadata"]["chat_attachments"] == [shown]
    assert "chat_attachment_uploads" not in projected["metadata"]
    # Original in-memory input dict is completely unmutated across all carriers.
    assert stored["owner_quiz"]["q1"]["attachments"][0]["upload"] == raw["upload"]
    assert stored["owner_quiz"]["q1"]["attachments"][0]["mtime_ns"] == 4242
    assert stored["held_owner_messages"][quiz_addr]["quiz"]["attachments"][0]["upload"] == raw["upload"]
    assert stored["metadata"]["late_answer"]["answer_snapshot"]["quiz"]["attachments"][0]["upload"] == raw["upload"]
    assert stored["metadata"]["chat_attachments"][0]["upload"] == raw["upload"]
    assert stored["metadata"]["chat_attachment_uploads"] == raw_uploads
    assert stored["metadata"]["continuation"]["owner_sources"]["held_owner_messages"][quiz_addr]["quiz"]["attachments"][0]["mtime_ns"] == 4242

    # All three canonical late-block markers establish typed late quiz file provenance.
    for marker in (
        {"answered_after_terminal": True},
        {"delivery_after_terminal": True},
        {"delivery": {"route": "fresh_head"}},
    ):
        late_out = public_task_result({
            "task_id": "task-late", "status": "completed",
            "metadata": {
                "late_answer": {"task_id": "asker-1", "quiz_id": "q1",
                                "answer_snapshot": {"task_id": "asker-1",
                                                    "quiz": {"quiz_id": "q1", "state": "answered",
                                                             "attachments": [dict(raw)], **marker}}},
                **carriers,
            },
        })
        assert late_out["metadata"]["chat_attachments"] == [shown]
        assert "chat_attachment_uploads" not in late_out["metadata"]

    # Ordinary task, event row, invalid late provenance, secret request, and text-only late
    # provenance all preserve metadata.chat_attachments and metadata.chat_attachment_uploads.
    event_row = {"type": "message", "task_id": "task-ev", "ts": "2026-11-18T12:00:00+00:00",
                 "metadata": dict(carriers)}
    assert public_task_result(event_row, include_outcome_axes=False) == event_row

    missing_flag_out = public_task_result({
        "task_id": "task-permissive", "status": "completed",
        "metadata": {
            "late_answer": {"task_id": "asker-1", "quiz_id": "q1",
                            "answer_snapshot": {"task_id": "asker-1",
                                                "quiz": {"quiz_id": "q1", "state": "answered",
                                                         "attachments": [dict(raw)]}}},
            **carriers,
        },
    })
    assert missing_flag_out["metadata"]["late_answer"]["answer_snapshot"]["quiz"]["attachments"] == [shown]
    assert missing_flag_out["metadata"]["chat_attachments"] == [raw]
    assert missing_flag_out["metadata"]["chat_attachment_uploads"] == raw_uploads

    for meta in (
        dict(carriers),  # ordinary non-late task
        {**carriers, "late_answer": {"task_id": "asker-1", "quiz_id": "q1",
                                     "answer_snapshot": {"task_id": "other-asker",
                                                         "quiz": dict(block)}}},  # mismatched task_id
        {**carriers, "late_answer": {"task_id": "asker-1", "quiz_id": "q1",
                                     "answer_snapshot": {"task_id": "asker-1",
                                                         "quiz": {"quiz_id": "q1", "state": "answered",
                                                                  "delivery": "bad",
                                                                  "attachments": [dict(raw)]}}}},  # invalid delivery
        {**carriers, "late_answer": {"task_id": "asker-1", "quiz_id": "q1",
                                     "answer_snapshot": {"task_id": "asker-1",
                                                         "quiz": {**block, "secret_request": {"pluginName": "p"}}}}},
        {**carriers, "late_answer": {"task_id": "asker-1", "quiz_id": "q1",
                                     "answer_snapshot": {"task_id": "asker-1",
                                                         "quiz": {"quiz_id": "q1", "state": "answered",
                                                                  "answered_after_terminal": True,
                                                                  "comment": "text only"}}}},
    ):
        neg = public_task_result({"task_id": "task-neg", "status": "completed", "metadata": meta})
        assert neg["metadata"]["chat_attachments"] == [raw]
        assert neg["metadata"]["chat_attachment_uploads"] == raw_uploads


def _quiz_projection_fixture():
    """Shared synthetic fixtures for ordinary late-row, chat-stream and mailbox quiz-file projections."""
    import copy

    note_upload = f"{'2' * 32}_note.png"
    raw_refs = [
        {"upload": note_upload, "name": "note.png", "kind": "image", "mime": "image/png",
         "size": 64, "sha256": "b" * 64, "mtime_ns": 987654321},
        {"name": "clip.mp4", "kind": "file", "mime": "video/mp4",
         "unavailable": "too_large", "limit_bytes": 10 * 1024 * 1024},
    ]
    expected_files = [
        {"name": "note.png", "kind": "image", "mime": "image/png", "size": 64, "sha256": "b" * 64},
        {"name": "clip.mp4", "kind": "file", "mime": "video/mp4",
         "unavailable": "too_large", "limit_bytes": 10 * 1024 * 1024},
    ]
    answer_row = {
        "type": "quiz_answer", "task_id": "asker-1", "chat_id": 1, "attachments": copy.deepcopy(raw_refs),
        "quiz": {"quiz_id": "q-files", "state": "answered", "attachments": copy.deepcopy(raw_refs)},
    }
    delivery = {"kind": "answer", "state": "delivered", "route": "fresh_head",
                "answer_row": copy.deepcopy(answer_row)}
    file_only_block = {
        "quiz_id": "q-files", "question": "Send the screenshot", "options": ["Skip"], "state": "answered",
        "comment": "", "attachments": copy.deepcopy(raw_refs), "delivery": copy.deepcopy(delivery),
    }
    button_block = {
        "quiz_id": "q-btn", "question": "Which option?", "options": ["Skip", "Keep"], "state": "answered",
        "answered_index": 1, "comment": "use this one", "answered_after_terminal": True,
        "attachments": [copy.deepcopy(raw_refs[0])],
    }
    secret_block = {
        "quiz_id": "q-secret", "state": "answered", "secret_request": {"pluginName": "p"},
        "secret_outcome": {"event": "saved", "pluginName": "p", "fields": ["token"]},
    }
    late_row = {
        "ts": "2026-11-18T12:01:30+00:00", "direction": "in", "chat_id": 1, "source": "web",
        "client_message_id": "quiz_late_answer:asker-1:q-late", "text": "use this one",
        "attachments": copy.deepcopy(raw_refs),
        "late_answer": {
            "task_id": "asker-1", "quiz_id": "q-late",
            "answer_snapshot": {"task_id": "asker-1",
                                "quiz": {**copy.deepcopy(button_block), "quiz_id": "q-late",
                                         "attachments": copy.deepcopy(raw_refs),
                                         "delivery": copy.deepcopy(delivery)}},
        },
    }
    file_only_late_row = {
        "ts": "2026-11-18T12:01:45+00:00", "direction": "in", "chat_id": 1, "source": "web",
        "client_message_id": "quiz_late_answer:asker-1:q-file-only",
        "text": "[Attached file: note.png]", "text_placeholder": True, "attachments": copy.deepcopy(raw_refs),
        "late_answer": {
            "task_id": "asker-1", "quiz_id": "q-file-only",
            "answer_snapshot": {"task_id": "asker-1",
                                "quiz": {**copy.deepcopy(file_only_block), "quiz_id": "q-file-only"}},
        },
    }
    plain_row = {
        "ts": "2026-11-18T12:00:30+00:00", "direction": "in", "chat_id": 1, "text": "ordinary note",
        "attachments": [{"upload": f"{'3' * 32}_note.pdf", "name": "note.pdf", "kind": "file",
                         "mime": "application/pdf", "size": 42, "sha256": "e" * 64, "mtime_ns": 777}],
    }
    return {
        "note_upload": note_upload, "raw_refs": raw_refs, "expected_files": expected_files,
        "file_only_block": file_only_block, "button_block": button_block, "secret_block": secret_block,
        "late_row": late_row, "file_only_late_row": file_only_late_row, "plain_row": plain_row,
    }


def _read_public_task_surfaces(tmp_path, task_id):
    """Read GET /api/tasks/{id} plus GET and POST /api/tasks/{id}/events SSE streams."""
    import json
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient
    from ouroboros.gateway.tasks import api_task_events, api_task_get

    app = Starlette(routes=[
        Route("/api/tasks/{task_id}", api_task_get, methods=["GET"]),
        Route("/api/tasks/{task_id}/events", api_task_events, methods=["GET", "POST"]),
    ])
    app.state.drive_root = tmp_path

    def _sse_events(resp):
        assert resp.status_code == 200, resp.text
        return [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]

    with TestClient(app) as client:
        detail = client.get(f"/api/tasks/{task_id}")
        assert detail.status_code == 200, detail.text
        return (
            detail.json(),
            _sse_events(client.get(f"/api/tasks/{task_id}/events?wait=0")),
            _sse_events(client.post(f"/api/tasks/{task_id}/events", json={"v": 2, "wait": 0})),
        )


def test_held_late_chat_event_and_mailbox_quiz_copies_project_in_public_readers(tmp_path):
    """Real detail and GET/POST SSE readers project ordinary quiz-answer files across held inbound
    late-answer rows (top-level and continuation ``held_owner_messages``), task-scoped ``chat``
    ``quiz_answer`` events and serialized ``owner_mailbox.rows`` / ``unread_mailbox.rows`` while
    keeping persisted rows, string wire types, ordinary chat/mail entries and protected forms intact."""
    import copy
    import json
    from ouroboros import dialogue_store as ds
    from ouroboros.owner_mailbox import (
        KIND_OWNER_TEXT, KIND_QUIZ_ANSWER, acknowledge_task_messages, write_owner_message,
    )
    from ouroboros.task_results import write_task_result
    from supervisor.message_ingress import log_chat

    fx, task_id = _quiz_projection_fixture(), "task-held-late"
    late_addr = "row:1@2026-11-18T12:01:30+00:00#1111aaaa"
    file_only_addr = "row:1@2026-11-18T12:01:45+00:00#2222bbbb"
    plain_addr = "row:1@2026-11-18T12:00:30+00:00#cafebabe"
    held_rows = {late_addr: copy.deepcopy(fx["late_row"]),
                 file_only_addr: copy.deepcopy(fx["file_only_late_row"]),
                 plain_addr: copy.deepcopy(fx["plain_row"])}
    # Form 1: real chat-stream quiz_answer, secret quiz_answer and ordinary chat rows for this task.
    quiz_cid, secret_cid, plain_cid = f"quiz_answer:{task_id}:q-files", f"quiz_answer:{task_id}:q-secret", "m-ordinary-chat"
    assert log_chat("system", 1, 0, "[Owner quiz answer] file-only", ts="2026-11-18T12:01:05+00:00",
                    source="owner_quiz_answer", task_id=task_id, client_message_id=quiz_cid,
                    record_type="quiz_answer", quiz=copy.deepcopy(fx["file_only_block"]),
                    drive_root=tmp_path, require_write=True) is not None
    assert log_chat("system", 1, 0, "[Secret request outcome]", ts="2026-11-18T12:01:10+00:00",
                    source="owner_quiz_answer", task_id=task_id, client_message_id=secret_cid,
                    record_type="quiz_answer", quiz=copy.deepcopy(fx["secret_block"]),
                    drive_root=tmp_path, require_write=True) is not None
    assert log_chat("in", 1, 1, "ordinary task msg", ts="2026-11-18T12:00:40+00:00", source="web",
                    task_id=task_id, client_message_id=plain_cid,
                    message_meta={"attachments": copy.deepcopy(fx["plain_row"]["attachments"])},
                    drive_root=tmp_path, require_write=True) is not None
    # Form 2: real mailbox rows captured on terminal transition into owner_mailbox and unread_mailbox.
    mail_files_id, mail_btn_id = f"quiz_answer:{task_id}:q-files", f"quiz_answer:{task_id}:q-btn"
    mail_plain_id, mail_secret_id = "m-plain-letter", f"quiz_answer:{task_id}:q-secret"
    for mid, text, qid, block in (
        (mail_files_id, "[Owner quiz answer] file-only", "q-files", fx["file_only_block"]),
        (mail_btn_id, "[Owner quiz answer] button", "q-btn", fx["button_block"]),
        (mail_secret_id, "[Secret request outcome]", "q-secret", fx["secret_block"]),
    ):
        assert write_owner_message(
            tmp_path, text, task_id, msg_id=mid, kind=KIND_QUIZ_ANSWER,
            quiz_delivery={"task_id": task_id, "quiz_id": qid, "canonical_root": str(tmp_path),
                           "answer_snapshot": copy.deepcopy(block)},
        )
    assert write_owner_message(tmp_path, "ordinary owner letter", task_id, msg_id=mail_plain_id, kind=KIND_OWNER_TEXT)
    assert acknowledge_task_messages(tmp_path, task_id, [mail_btn_id], wake_id="test")
    write_task_result(
        tmp_path, task_id, "completed", ts="2026-11-18T12:02:00+00:00", result="done",
        held_owner_messages=copy.deepcopy(held_rows),
        metadata={"continuation": {"predecessor_task_id": "pred-late",
                                   "owner_sources": {"held_owner_messages": copy.deepcopy(held_rows)}}},
    )
    detail_payload, get_events, post_events = _read_public_task_surfaces(tmp_path, task_id)
    [get_result] = [e["data"] for e in get_events if e.get("type") == "task_result"]
    [post_result] = [e["data"] for e in post_events if e.get("type") == "task_result"]
    stored, expected_files = read_result(tmp_path, task_id), fx["expected_files"]
    stored_owner = {json.loads(line)["msg_id"]: line for line in stored["owner_mailbox"]["rows"]}
    stored_unread = {json.loads(line)["msg_id"]: line for line in stored["unread_mailbox"]["rows"]}

    for surface in (detail_payload, get_result, post_result):
        for held_map in (surface["held_owner_messages"],
                         surface["metadata"]["continuation"]["owner_sources"]["held_owner_messages"]):
            pub_late, pub_file_only = held_map[late_addr], held_map[file_only_addr]
            assert "quiz" not in pub_late and pub_late["text"] == "use this one"
            assert pub_late["attachments"] == expected_files
            assert "upload" not in pub_late["attachments"][0] and "mtime_ns" not in pub_late["attachments"][0]
            late_quiz = pub_late["late_answer"]["answer_snapshot"]["quiz"]
            assert late_quiz["attachments"] == expected_files
            assert late_quiz["delivery"]["answer_row"]["attachments"] == expected_files
            assert late_quiz["delivery"]["answer_row"]["quiz"]["attachments"] == expected_files
            assert pub_file_only["text_placeholder"] is True and pub_file_only["attachments"] == expected_files
            file_only_quiz = pub_file_only["late_answer"]["answer_snapshot"]["quiz"]
            assert file_only_quiz["attachments"] == expected_files and "answered_index" not in file_only_quiz
            assert held_map[plain_addr] == fx["plain_row"]
        for cap_key, raw_by_id, expected_ids in (
            ("owner_mailbox", stored_owner, (mail_files_id, mail_btn_id)),
            ("unread_mailbox", stored_unread, (mail_files_id,)),
        ):
            pub_rows = surface[cap_key]["rows"]
            assert all(isinstance(line, str) for line in pub_rows)
            pub_by_id = {json.loads(line)["msg_id"]: line for line in pub_rows}
            assert pub_by_id[mail_plain_id] == raw_by_id[mail_plain_id]
            assert pub_by_id[mail_secret_id] == raw_by_id[mail_secret_id]
            for mid in expected_ids:
                snap = json.loads(pub_by_id[mid])["quiz_delivery"]["answer_snapshot"]
                assert snap["attachments"] == (expected_files if mid == mail_files_id else [expected_files[0]])
                assert "upload" not in snap["attachments"][0] and "mtime_ns" not in snap["attachments"][0]
                if mid == mail_files_id:
                    assert "answered_index" not in snap
                    assert snap["delivery"]["answer_row"]["attachments"] == expected_files
                    assert snap["delivery"]["answer_row"]["quiz"]["attachments"] == expected_files
    for stream_events in (get_events, post_events):
        chat_by_cid = {e["data"].get("client_message_id"): e for e in stream_events if e.get("source") == "chat"}
        pub_quiz_ev = chat_by_cid[quiz_cid]["data"]
        assert pub_quiz_ev["type"] == "quiz_answer" and pub_quiz_ev["text"] == "[Owner quiz answer] file-only"
        assert pub_quiz_ev["quiz"]["attachments"] == expected_files and "upload" not in pub_quiz_ev["quiz"]["attachments"][0]
        assert pub_quiz_ev["quiz"]["delivery"]["answer_row"]["attachments"] == expected_files
        assert pub_quiz_ev["quiz"]["delivery"]["answer_row"]["quiz"]["attachments"] == expected_files
        assert chat_by_cid[secret_cid]["data"]["quiz"] == fx["secret_block"]
        assert chat_by_cid[plain_cid]["data"]["attachments"] == fx["plain_row"]["attachments"]
    # Persisted canonical storage stays untouched across held rows, chat rows and mailbox captures.
    for stored_map in (stored["held_owner_messages"],
                       stored["metadata"]["continuation"]["owner_sources"]["held_owner_messages"]):
        assert stored_map[late_addr]["attachments"][0]["upload"] == fx["note_upload"]
        assert stored_map[late_addr]["attachments"][0]["mtime_ns"] == 987654321
        assert stored_map[plain_addr] == fx["plain_row"]
    for raw_line in (stored_owner[mail_files_id], stored_owner[mail_btn_id], stored_unread[mail_files_id]):
        raw_snap = json.loads(raw_line)["quiz_delivery"]["answer_snapshot"]
        assert raw_snap["attachments"][0]["upload"] == fx["note_upload"] and raw_snap["attachments"][0]["mtime_ns"] == 987654321
    stored_chat = {row.row.get("client_message_id"): row.row for row in ds.iter_rows(tmp_path, "chat")}
    assert stored_chat[quiz_cid]["quiz"]["attachments"][0]["upload"] == fx["note_upload"]
    assert stored_chat[quiz_cid]["quiz"]["attachments"][0]["mtime_ns"] == 987654321


def test_public_quiz_projection_keeps_inputs_immutable_and_boundaries_exact():
    """``public_owner_row`` and ``public_task_result`` clone without mutating inputs, support all
    three canonical late markers, and keep ordinary, legacy, invalid, protected-form, text-only
    and malformed chat/mailbox records untouched."""
    import copy
    import json
    from ouroboros.chat_uploads import public_owner_row
    from ouroboros.outcomes import public_task_result

    fx = _quiz_projection_fixture()
    raw_refs, expected_files, late_row, plain_row = fx["raw_refs"], fx["expected_files"], fx["late_row"], fx["plain_row"]

    def _late(quiz, snap_task="asker-1"):
        return {"task_id": "asker-1", "quiz_id": "q-late", "answer_snapshot": {"task_id": snap_task, "quiz": quiz}}

    def _mail_line(mid, kind, text, **extra):
        return json.dumps({"msg_id": mid, "kind": kind, "text": text, **extra}, ensure_ascii=False)

    row_before = copy.deepcopy(late_row)
    pub_direct = public_owner_row(late_row)
    assert pub_direct["attachments"] == expected_files
    assert pub_direct["late_answer"]["answer_snapshot"]["quiz"]["attachments"] == expected_files
    assert late_row == row_before

    plain_line = _mail_line("m1", "owner_text", "hello", attachments=raw_refs)
    text_only_quiz_line = _mail_line("m2", "quiz_answer", "ok", quiz_delivery={
        "task_id": "t1", "quiz_id": "q-txt", "answer_snapshot": {"quiz_id": "q-txt", "state": "answered", "comment": "yes"}})
    secret_quiz_line = _mail_line("m3", "quiz_answer", "secret", quiz_delivery={
        "task_id": "t1", "quiz_id": "q-sec", "answer_snapshot": {**fx["secret_block"], "attachments": raw_refs}})
    file_quiz_line = _mail_line("m4", "quiz_answer", "with file", quiz_delivery={
        "task_id": "t1", "quiz_id": "q-files", "answer_snapshot": copy.deepcopy(fx["file_only_block"])})
    in_mem = {
        "task_id": "task-mem", "status": "completed",
        "held_owner_messages": {"r1": copy.deepcopy(late_row), "r2": copy.deepcopy(plain_row)},
        "owner_mailbox": {"schema": 1, "rows": [plain_line, text_only_quiz_line, secret_quiz_line,
                                                file_quiz_line, "not-json", "[]", 42]},
        "unread_mailbox": {"schema": 1, "rows": [file_quiz_line]},
        "metadata": {"continuation": {"predecessor_task_id": "pred-late",
                                      "owner_sources": {"held_owner_messages": {"r1": copy.deepcopy(late_row)}}}},
    }
    in_mem_before = copy.deepcopy(in_mem)
    projected = public_task_result(in_mem)
    assert projected["held_owner_messages"]["r1"]["attachments"] == expected_files
    assert projected["held_owner_messages"]["r2"] == plain_row
    assert projected["metadata"]["continuation"]["owner_sources"]["held_owner_messages"]["r1"]["attachments"] == expected_files
    out_rows = projected["owner_mailbox"]["rows"]
    assert out_rows[:3] == [plain_line, text_only_quiz_line, secret_quiz_line] and out_rows[4:] == ["not-json", "[]", 42]
    assert isinstance(out_rows[3], str)
    assert json.loads(out_rows[3])["quiz_delivery"]["answer_snapshot"]["attachments"] == expected_files
    assert json.loads(projected["unread_mailbox"]["rows"][0])["quiz_delivery"]["answer_snapshot"]["attachments"] == expected_files
    assert in_mem == in_mem_before

    chat_ev = {"type": "quiz_answer", "task_id": "t1", "source": "owner_quiz_answer",
               "quiz": copy.deepcopy(fx["file_only_block"])}
    chat_ev_before = copy.deepcopy(chat_ev)
    assert public_task_result(chat_ev, include_outcome_axes=False)["quiz"]["attachments"] == expected_files
    assert chat_ev == chat_ev_before
    secret_ev = {"type": "quiz_answer", "task_id": "t1",
                 "quiz": {**fx["secret_block"], "attachments": copy.deepcopy(raw_refs)}}
    assert public_task_result(secret_ev, include_outcome_axes=False) == secret_ev
    ordinary_ev = {"type": "message", "task_id": "t1", "attachments": copy.deepcopy(raw_refs)}
    assert public_task_result(ordinary_ev, include_outcome_axes=False) == ordinary_ev

    for marker in ({"answered_after_terminal": True}, {"delivery_after_terminal": True},
                   {"delivery": {"route": "fresh_head"}}):
        marked_out = public_owner_row({
            **plain_row, "attachments": copy.deepcopy(raw_refs),
            "late_answer": _late({"quiz_id": "q-late", "state": "answered",
                                  "attachments": copy.deepcopy(raw_refs), **marker}),
        })
        assert marked_out["attachments"] == expected_files
        assert marked_out["late_answer"]["answer_snapshot"]["quiz"]["attachments"] == expected_files

    assert public_owner_row(plain_row) == plain_row
    legacy_late = {**plain_row, "attachments": copy.deepcopy(raw_refs),
                   "late_answer": {"task_id": "asker-1", "quiz_id": "q-late"}}
    assert public_owner_row(legacy_late) == legacy_late
    missing_out = public_owner_row({
        **plain_row, "attachments": copy.deepcopy(raw_refs),
        "late_answer": _late({"quiz_id": "q-late", "state": "answered", "attachments": copy.deepcopy(raw_refs)}),
    })
    assert missing_out["late_answer"]["answer_snapshot"]["quiz"]["attachments"] == expected_files
    assert missing_out["attachments"] == raw_refs

    for bad_late in (
        _late({"quiz_id": "q-late", "state": "answered", "answered_after_terminal": True,
               "attachments": copy.deepcopy(raw_refs)}, "other-asker"),
        _late({"quiz_id": "q-late", "state": "answered", "delivery": "bad",
               "attachments": copy.deepcopy(raw_refs)}),
        _late({"quiz_id": "q-late", "state": "answered", "answered_after_terminal": True,
               "secret_request": {"pluginName": "p"}, "attachments": copy.deepcopy(raw_refs)}),
        _late({"quiz_id": "q-late", "state": "answered", "answered_after_terminal": True, "comment": "text only"}),
    ):
        neg_out = public_owner_row({**plain_row, "attachments": copy.deepcopy(raw_refs), "late_answer": bad_late})
        assert neg_out["attachments"] == raw_refs
        if "secret_request" in bad_late["answer_snapshot"]["quiz"]:
            assert neg_out["late_answer"] == bad_late


def test_a_rejoin_reports_an_unreadable_coverage_as_unconfirmed(tmp_path, monkeypatch):
    """The coverage read behind an indexed miss (``dialogue_import.coverage``) needs the store as well: an
    unavailable or still-importing store is the SAME typed 503 ``quiz_answer_unconfirmed`` the bridge already reads
    as unknown — never an uncaught store failure, and never "this is not the answer". A readable coverage still
    decides the miss the same way it did."""
    from ouroboros import dialogue_import
    from ouroboros.gateway import task_decision as td
    from ouroboros.store import StoreError

    body = {"request_id": "tg:9:m:42:7", "decision_id": "quiz:task-1:q1", "rejoin": True}
    read_streams = []

    def _unreadable(root, stream, seq=None):
        read_streams.append(stream)
        raise StoreError("the canonical store is unavailable")

    monkeypatch.setattr(dialogue_import, "coverage", _unreadable)
    status, payload = td.rejoin_decision(tmp_path, body)
    assert read_streams == ["chat"], "the miss read the canonical chat stream's coverage"
    assert status == 503 and payload["reason_code"] == "quiz_answer_unconfirmed"

    monkeypatch.setattr(dialogue_import, "coverage",
                        lambda root, stream, seq=None: {"imported": True, "pending": 0, "mismatch": 0, "gaps": 0})
    status, payload = td.rejoin_decision(tmp_path, body)
    assert status == 404 and payload["reason_code"] == "request_not_recorded"
