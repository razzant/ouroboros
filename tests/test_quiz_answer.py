"""#Q-2b: the quiz answer half — escalate verb, decision ingress, projection
lifecycle, mailbox delivery, structural expiry, replay merge.

Owner decisions pinned here: 27=A (assumption mandatory — validated upstream),
30=A (structural expiry only, no host TTL), 31 (one verb; root → owner card,
subagent → parent mailbox frame; forged owner provenance impossible).
"""

from __future__ import annotations

import json
import types


from ouroboros.owner_quiz import (
    STATE_ANSWERED,
    STATE_EXPIRED_TERMINAL,
    STATE_OPEN,
    quiz_states,
    reconcile_terminal,
    record_answered,
    record_asked,
)


def _result_path(tmp_path, task_id):
    from ouroboros.task_results import task_result_path

    return task_result_path(tmp_path, task_id)


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


def test_projection_survives_concurrent_result_fields(tmp_path):
    record_asked(tmp_path, "t1", quiz_id="q1", question="?", options=["A"] * 2)
    path = _result_path(tmp_path, "t1")
    data = json.loads(path.read_text())
    assert "owner_quiz" in data
    # A terminal writer merging around the key keeps the projection intact.
    data["status"] = "completed"
    path.write_text(json.dumps(data))
    assert quiz_states(tmp_path, "t1")["q1"]["state"] == STATE_OPEN


# ---------------------------------------------------------- unified reconcile

def test_task_done_seam_reconciles_and_broadcasts(tmp_path, monkeypatch):
    """The unified terminal seam expires open quizzes and pushes the live
    quiz_state frame for each — replacing the per-domain events.py blocks."""
    import supervisor.queue_transitions as qt

    record_asked(tmp_path, "t1", quiz_id="q1", question="?", options=["A", "B"])
    frames = []

    class _Bridge:
        def send_quiz_state(self, quiz_id, task_id, state, answered_index=None, chat_id=0):
            frames.append((quiz_id, task_id, state))

    import supervisor.message_bus as mb

    monkeypatch.setattr(mb, "get_bridge", lambda: _Bridge())
    qt.reconcile_terminal_task_projections(tmp_path, "t1")
    assert frames == [("q1", "t1", "expired_terminal")]
    assert quiz_states(tmp_path, "t1")["q1"]["state"] == STATE_EXPIRED_TERMINAL


# -------------------------------------------------------------- the ingress

def _decision_app(tmp_path, monkeypatch, live_task=None):
    from starlette.applications import Starlette
    from starlette.routing import Route

    from ouroboros.gateway import task_decision as td

    monkeypatch.setattr(td, "request_drive_root", lambda request: tmp_path)
    monkeypatch.setattr(
        td, "_live_root_task",
        lambda task_id: (live_task, "" if live_task else "task_not_live"),
    )
    if live_task is not None:
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
        def send_quiz_state(self, quiz_id, task_id, state, answered_index=None, chat_id=0):
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


def test_ingress_late_answer_is_an_honest_409(tmp_path, monkeypatch):
    record_asked(tmp_path, "task-1", quiz_id="q1", question="?", options=["A", "B"])
    reconcile_terminal(tmp_path, "task-1")  # the task settled
    app = _decision_app(tmp_path, monkeypatch, live_task=None)
    resp = _post(app, {"request_id": "r1", "decision_id": "quiz:task-1:q1",
                       "option_index": 0})
    assert resp.status_code == 409
    assert resp.json()["state"] == "expired_terminal"


def test_ingress_heals_an_unreconciled_quiz_of_a_dead_task(tmp_path, monkeypatch):
    """Crash window: the author died before the task-done seam expired its
    open quiz. A late answer must NOT be recorded into a mailbox nobody
    drains — the ingress reconciles first and answers the honest 409."""
    record_asked(tmp_path, "task-1", quiz_id="q1", question="?", options=["A", "B"])
    app = _decision_app(tmp_path, monkeypatch, live_task=None)
    resp = _post(app, {"request_id": "r1", "decision_id": "quiz:task-1:q1",
                       "option_index": 0})
    assert resp.status_code == 409
    assert resp.json()["state"] == "expired_terminal"
    assert quiz_states(tmp_path, "task-1")["q1"]["state"] == STATE_EXPIRED_TERMINAL


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
                kw.pop("stake", ""), kw.pop("assumption", ""))


def test_escalate_root_records_projection_and_emits_quiz(tmp_path):
    ctx = _tool_ctx(tmp_path)
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
    out = _escalate(ctx, question="Delete the flaky test?",
                    options=[{"label": "delete"}, {"label": "quarantine"}],
                    stake="CI health", assumption="quarantine meanwhile")
    assert out.startswith("OK: escalated to parent task root-1")
    entries = drain_owner_entries(tmp_path, "root-1", set())
    assert entries and entries[0]["provenance"] == "descendant_task"
    text = entries[0]["text"]
    assert "ESCALATION (decision requested): Delete the flaky test?" in text
    assert "1. delete" in text and "2. quarantine" in text
    assert "forward_to_worker(task_id=child-9" in text
    # No owner card, no projection for the child hop.
    assert not [e for e in ctx.pending_events if e.get("type") == "send_quiz"]
    assert quiz_states(tmp_path, "child-9") == {}


def test_escalate_settled_parent_is_a_typed_dead_end(tmp_path, monkeypatch):
    import ouroboros.task_status as ts

    monkeypatch.setattr(ts, "load_effective_task_result",
                        lambda root, tid: {"status": "completed"})
    ctx = _tool_ctx(tmp_path, task_id="child-9", parent="root-1")
    out = _escalate(ctx, question="?", options=["a", "b"], assumption="a")
    assert out.startswith("⚠️ ESCALATE_PARENT_SETTLED")


def test_escalate_background_refused(tmp_path):
    ctx = _tool_ctx(tmp_path, task_id="bg-consciousness", role="background")
    out = _escalate(ctx, question="?", options=["a", "b"], assumption="a")
    assert out.startswith("⚠️ ESCALATE_UNAVAILABLE")


def test_escalate_invalid_payload_is_typed(tmp_path):
    ctx = _tool_ctx(tmp_path)
    out = _escalate(ctx, question="?", options=["only-one"], assumption="a")
    assert out.startswith("⚠️ QUIZ_OPTIONS_INVALID")
    out = _escalate(ctx, question="?", options=["a", "b"], assumption="")
    assert out.startswith("⚠️ QUIZ_ASSUMPTION_REQUIRED")


def test_escalate_absent_from_ephemeral_allowlist():
    """A decision turn cannot escalate — the structural default-deny refusal
    comes free, exactly like forward_to_worker."""
    from ouroboros.tools.registry import _EPHEMERAL_ALLOWED_TOOLS

    assert "escalate" not in _EPHEMERAL_ALLOWED_TOOLS


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
    chat_path.write_text(json.dumps({
        "ts": "2026-08-31T00:00:00Z", "direction": "out", "chat_id": 1,
        "user_id": 7, "text": "Which?", "type": "quiz", "task_id": "task-1",
        "quiz": {"quiz_id": "q1", "options": [{"label": "A"}, {"label": "B"}],
                 "stake": "", "assumption": "x", "state": "open"},
    }) + "\n")
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
    """An evicted OPEN block would resurrect as an answerable card on replay
    (the chat row froze state=open) whose click 404s — closed blocks go first."""
    from ouroboros.owner_quiz import _QUIZ_CAP

    for i in range(_QUIZ_CAP):
        record_asked(tmp_path, "t1", quiz_id=f"closed-{i:02d}", question="?",
                     options=["A", "B"])
        record_answered(tmp_path, "t1", quiz_id=f"closed-{i:02d}", option_index=0,
                        request_id=f"r{i}")
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
        def send_quiz_state(self, quiz_id, task_id, state, answered_index=None, chat_id=0):
            frames.append((quiz_id, state, answered_index))

    import supervisor.message_bus as mb

    monkeypatch.setattr(mb, "get_bridge", lambda: _Bridge())
    app = _decision_app(tmp_path, monkeypatch, live_task={"id": "task-1"})
    resp = _post(app, {"request_id": "r1", "decision_id": "quiz:task-1:q1",
                       "comment": "neither — use duckdb"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["state"] == "answered"
    assert "answered_index" not in body  # absent, never fabricated
    assert frames == [("q1", "answered", None)]

    block = quiz_states(tmp_path, "task-1")["q1"]
    assert block["state"] == STATE_ANSWERED
    assert "answered_index" not in block
    assert block["comment"] == "neither — use duckdb"

    entries = drain_owner_entries(tmp_path, "task-1", set())
    frame_text = [e for e in entries if e.get("kind") == KIND_QUIZ_ANSWER][0]["text"]
    assert ("The owner rejected all offered options and answered verbatim: "
            "neither — use duckdb") in frame_text
    assert "chose option" not in frame_text


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
    chat_path.write_text(json.dumps({
        "ts": "2026-09-01T00:00:00Z", "direction": "out", "chat_id": 1,
        "user_id": 7, "text": "Which?", "type": "quiz", "task_id": "task-1",
        "quiz": {"quiz_id": "q1", "options": [{"label": "A"}, {"label": "B"}],
                 "stake": "", "assumption": "x", "state": "open"},
    }) + "\n")
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
