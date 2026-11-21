"""§19.7.4 mandatory no-chat and durability proofs for the owner hurry control (S3).

Every proof runs against the PRODUCTION seams — ``gateway.tasks.api_task_hurry``
(re-export of ``gateway/task_hurry.py``), ``loop._drain_incoming_messages``,
``loop._no_tool_final_answer``, ``loop._enforce_plan_gate``, the dedicated
locked projection writer, and the one shared retry-reset invoked by the
``supervisor/workers.py`` crash-requeue.

The HQ1 contract under test: hurry is a typed task-local control (owner-mailbox
``kind=hurry``) that NEVER creates a chat message, owner directive, outbox row,
or bubble; its one effect is a host rail (the task-local advisory force-plan
projection); P3/commit/safety never consult it; retry attempts start with no
executable latch. Hurry no longer skips a review of the task's result and has no
improvement passes to zero: both left with the task-acceptance machine.
"""

from __future__ import annotations


import json
import pathlib
import queue as _q
import threading
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import owner_hurry as oh
from ouroboros.gateway.tasks import api_task_hurry
from ouroboros.owner_mailbox import (
    KIND_HURRY,
    _mailbox_path,
    drain_owner_entries,
    write_owner_message,
)
from ouroboros.task_results import load_task_result, write_task_result


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _isolate_queue(monkeypatch, tmp_path, *, pending=(), running=None):
    from supervisor import queue as q

    monkeypatch.setattr(q, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(q, "PENDING", [dict(t) for t in pending])
    monkeypatch.setattr(q, "RUNNING", dict(running or {}))
    monkeypatch.setattr(q, "ACCEPTANCE_FENCES", {}, raising=False)
    monkeypatch.setattr(q, "persist_queue_snapshot", lambda reason="": None)
    return q


def _client(tmp_path):
    app = Starlette(routes=[
        Route("/api/tasks/{task_id}/hurry", api_task_hurry, methods=["POST"]),
    ])
    app.state.drive_root = tmp_path
    return TestClient(app)


def _hurry_rows(drive_root, task_id):
    path = _mailbox_path(pathlib.Path(drive_root), task_id)
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("kind") == KIND_HURRY
    ]


# ---------------------------------------------------------------------------
# §19.7.4 bullet 1 — production endpoint: drives, projection, refusals,
# idempotency, terminal TOCTOU
# ---------------------------------------------------------------------------


def test_refusal_matrix_managed_child_cancel_sealed_and_not_live(tmp_path, monkeypatch):
    child = {
        "id": "child-1", "chat_id": 0, "root_task_id": "root-x",
        "parent_task_id": "root-x", "delegation_role": "child",
    }
    cancelled = {"id": "stop-1", "chat_id": 0, "root_task_id": "stop-1"}
    sealed = {"id": "seal-1", "chat_id": 0, "root_task_id": "seal-1"}
    q = _isolate_queue(monkeypatch, tmp_path, pending=[child, cancelled, sealed])
    from ouroboros.cancel_intents import request_cancel

    request_cancel(tmp_path, "stop-1")
    q.ACCEPTANCE_FENCES["seal-1"] = {"status": "sealed"}
    with _client(tmp_path) as client:
        child_resp = client.post("/api/tasks/child-1/hurry", json={"request_id": "r"})
        cancel_resp = client.post("/api/tasks/stop-1/hurry", json={"request_id": "r"})
        sealed_resp = client.post("/api/tasks/seal-1/hurry", json={"request_id": "r"})
        ghost_resp = client.post("/api/tasks/ghost/hurry", json={"request_id": "r"})
        no_id_resp = client.post("/api/tasks/seal-1/hurry", json={})
        text_resp = client.post(
            "/api/tasks/seal-1/hurry", json={"request_id": "r", "text": "smuggled"},
        )
    assert (child_resp.status_code, child_resp.json()["reason_code"]) == (409, "not_a_root_task")
    # A pending stop WINS and owns the terminal reason: hurry is refused.
    assert (cancel_resp.status_code, cancel_resp.json()["reason_code"]) == (409, "cancel_pending")
    assert (sealed_resp.status_code, sealed_resp.json()["reason_code"]) == (409, "acceptance_fence_sealed")
    assert (ghost_resp.status_code, ghost_resp.json()["reason_code"]) == (404, "task_not_live")
    assert (no_id_resp.status_code, no_id_resp.json()["reason_code"]) == (400, "request_id_required")
    # Text-free by contract: a smuggled field is refused, never silently dropped.
    assert (text_resp.status_code, text_resp.json()["reason_code"]) == (400, "unexpected_fields")
    # None of the refused paths wrote a mailbox row or a projection.
    for tid in ("child-1", "stop-1", "seal-1"):
        assert _hurry_rows(tmp_path, tid) == []
        assert "owner_hurry" not in (load_task_result(tmp_path, tid) or {})


def test_same_request_id_is_idempotent_and_different_id_collapses_to_one_latch(tmp_path, monkeypatch):
    task = {"id": "root-3", "chat_id": 0, "root_task_id": "root-3", "_attempt": 1}
    _isolate_queue(monkeypatch, tmp_path, running={"root-3": {"task": task, "attempt": 1}})
    with _client(tmp_path) as client:
        first = client.post("/api/tasks/root-3/hurry", json={"request_id": "req-a"})
        retry = client.post("/api/tasks/root-3/hurry", json={"request_id": "req-a"})
        other = client.post("/api/tasks/root-3/hurry", json={"request_id": "req-b"})
    assert first.json()["duplicate"] is False
    # Same request_id: the existing acknowledgement, and the deliberate
    # re-append is INVISIBLE to the drain (same msg_id dedupe).
    assert retry.json()["duplicate"] is True
    entries = drain_owner_entries(tmp_path, task_id="root-3")
    assert len([e for e in entries if e["kind"] == KIND_HURRY]) == 1
    # Different request_id while a block exists: duplicate acknowledgement, NO
    # second mailbox CONTROL — the healing append reuses the LATCH's msg_id
    # (hurry:req-a), so the drain still dedupes to one control while a lost
    # first append (mailbox write failure + reload) heals instead of a false
    # "already accepted" ack. Projection stays owned by the first request.
    assert other.json()["duplicate"] is True
    assert len(_hurry_rows(tmp_path, "root-3")) == 3      # req-a twice + req-b's heal
    assert all(r["msg_id"] == "hurry:req-a" for r in _hurry_rows(tmp_path, "root-3"))
    assert len([e for e in drain_owner_entries(tmp_path, task_id="root-3")
                if e["kind"] == KIND_HURRY]) == 1
    assert load_task_result(tmp_path, "root-3")["owner_hurry"]["request_id"] == "req-a"


def test_terminal_toctou_reconciles_requested_to_not_applied(tmp_path):
    oh.record_requested(tmp_path, "t-toctou", request_id="r1", attempt=1)
    assert oh.reconcile_terminal(tmp_path, "t-toctou") is True
    block = load_task_result(tmp_path, "t-toctou")["owner_hurry"]
    assert block["state"] == oh.STATE_NOT_APPLIED_BEFORE_TERMINAL
    assert block["reconciled_at"]
    # An APPLIED block is a real acceleration: reconciliation must not touch it.
    oh.record_requested(tmp_path, "t-applied", request_id="r2", attempt=1)
    oh.record_applied(tmp_path, "t-applied", attempt=1)
    assert oh.reconcile_terminal(tmp_path, "t-applied") is False
    assert load_task_result(tmp_path, "t-applied")["owner_hurry"]["state"] == oh.STATE_APPLIED


# ---------------------------------------------------------------------------
# §19.7.4 bullet 2 — production drain: structural routing, no owner-directive,
# ts/id survival, restart re-drain
# ---------------------------------------------------------------------------


def _drain_ctx(tmp_path, task_id="t-drain", attempt=2):
    return SimpleNamespace(
        task_id=task_id,
        budget_drive_root=str(tmp_path),
        drive_root=str(tmp_path),
        task_attempt=attempt,
    )


def test_drain_routes_hurry_structurally_and_never_as_owner_prose(tmp_path, monkeypatch):
    import ouroboros.loop as loop

    write_owner_message(tmp_path, "owner_hurry", "t-drain", msg_id="hurry:rq", kind=KIND_HURRY)
    write_owner_message(tmp_path, "keep going please", "t-drain", msg_id="m-text")
    oh.record_requested(tmp_path, "t-drain", request_id="rq", attempt=2)

    marked = []
    real_marked = loop._owner_marked_content
    monkeypatch.setattr(
        loop, "_owner_marked_content",
        lambda content: marked.append(content) or real_marked(content),
    )
    recorded = []
    real_record = loop._record_owner_directive
    monkeypatch.setattr(
        loop, "_record_owner_directive",
        lambda ctx, **kw: recorded.append(kw) or real_record(ctx, **kw),
    )

    ctx = _drain_ctx(tmp_path)
    messages: list = []
    events: list = []
    controls = loop._drain_incoming_messages(
        messages, _q.Queue(), tmp_path, "t-drain",
        SimpleNamespace(put_nowait=events.append), set(), owner_ctx=ctx,
    )
    # The typed control is returned, never injected.
    assert controls == {"hurry": "hurry:rq"}
    joined = json.dumps(messages, ensure_ascii=False)
    assert "keep going please" in joined
    assert "owner_hurry" not in joined
    # _record_owner_directive/_owner_marked_content saw ONLY the text entry.
    assert [kw["content"] for kw in recorded] == ["keep going please"]
    assert marked == ["keep going please"]
    assert [d["content"] for d in ctx._owner_directives] == ["keep going please"]
    # owner_message_injected fired only for the dialogue text.
    injected = [e for e in events if e.get("type") == "owner_message_injected"]
    assert len(injected) == 1 and injected[0]["text"] == "keep going please"
    # The latch is armed with the mailbox id and REQUEST time preserved.
    latch = oh.latched(ctx)
    assert latch["msg_id"] == "hurry:rq"
    assert latch["requested_at"]                       # drained ts survives
    # The running work reads one typed row of the owner's act, never recorded as a directive.
    from ouroboros.context_budget import OWNER_WORDS_KIND
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
    rows = [m for m in messages if m.get("content") == oh.HURRY_WORDS]
    assert len(rows) == 1 and rows[0][HOST_CONTEXT_KIND_KEY] == OWNER_WORDS_KIND
    assert oh.HURRY_WORDS not in json.dumps(ctx._owner_directives, ensure_ascii=False)
    # The worker-side projection went to APPLIED with its one named effect.
    block = load_task_result(tmp_path, "t-drain")["owner_hurry"]
    assert block["state"] == oh.STATE_APPLIED
    assert block["effects"] == {"words": "finish_sooner"}
    # The applied event rides the log_event envelope (task-detail observability
    # only), never a send_message/chat frame.
    hurry_events = [
        e["data"] for e in events
        if e.get("type") == "log_event" and e.get("data", {}).get("type") == "owner_hurry"
    ]
    assert len(hurry_events) == 1 and hurry_events[0]["is_progress"] is False
    assert not [e for e in events if e.get("type") == "send_message"]


def test_restart_re_drain_restores_exactly_one_latch(tmp_path):
    import ouroboros.loop as loop

    write_owner_message(tmp_path, "owner_hurry", "t-re", msg_id="hurry:r1", kind=KIND_HURRY)
    oh.record_requested(tmp_path, "t-re", request_id="r1", attempt=1)
    ctx = _drain_ctx(tmp_path, task_id="t-re", attempt=1)
    loop._drain_incoming_messages([], _q.Queue(), tmp_path, "t-re", None, set(), owner_ctx=ctx)
    first_latch = oh.latched(ctx)
    assert first_latch is not None
    # Restart: a fresh seen-set re-reads the same durable control (the mailbox
    # is NOT consumed on read); the latch is re-armed identically, exactly one.
    restarted = _drain_ctx(tmp_path, task_id="t-re", attempt=1)
    loop._drain_incoming_messages([], _q.Queue(), tmp_path, "t-re", None, set(), owner_ctx=restarted)
    assert oh.latched(restarted) == first_latch
    # Same-process double drain (second hurry entry): still ONE latch, and no second row of words.
    write_owner_message(tmp_path, "owner_hurry", "t-re", msg_id="hurry:r2", kind=KIND_HURRY)
    again: list = []
    loop._drain_incoming_messages(again, _q.Queue(), tmp_path, "t-re", None, set(), owner_ctx=restarted)
    assert oh.latched(restarted) == first_latch        # first arm wins
    assert oh.HURRY_WORDS not in json.dumps(again, ensure_ascii=False)
    # Without a hurry entry the drain gives no such row (the control of the words).
    quiet = _drain_ctx(tmp_path, task_id="t-quiet", attempt=1)
    write_owner_message(tmp_path, "just words", "t-quiet", msg_id="m-quiet")
    plain: list = []
    loop._drain_incoming_messages(plain, _q.Queue(), tmp_path, "t-quiet", None, set(), owner_ctx=quiet)
    assert "just words" in json.dumps(plain) and oh.HURRY_WORDS not in json.dumps(plain, ensure_ascii=False)


# ---------------------------------------------------------------------------
# §19.7.4 bullet 3 — production _no_tool_final_answer: hurry is not a follow-up
# ---------------------------------------------------------------------------
# §19.7.4 bullets 4+5 — the acceptance skip and the improvement-pass overlay left
# with the task-acceptance machine; the stored reason keeps its vocabulary
# ---------------------------------------------------------------------------
# §19.7.4 bullet 6 — production _enforce_plan_gate force-plan projection
# ---------------------------------------------------------------------------


def _plan_record(kind: str) -> dict:
    """A plan record by its gate state: ``absent`` (no plan at all), ``unsent`` (an ask that
    reached nobody), ``held`` (an owner-marked seat's open objection), ``awaiting`` (an ask
    whose answers are still on their way)."""
    from ouroboros import plan_record as pr

    record = pr.empty_record()
    if kind == "absent":
        return record
    record["revisions"].append({"n": 1, "ts": "t", "goal": "g", "plan": "p", "done_when": [], "affected_paths": [], "body": False,
                                "content_sha256": "0" * 64})
    seat = {"seat_id": "triad_a", "model": "m/a", "route": "api_chat", "owner_marked": True, "dispatched": True, "status": "pending"}
    if kind == "unsent":
        seat.update(dispatched=False, status="not_dispatched", not_dispatched_reason="health_skip")
    elif kind == "held":
        seat.update(status="answered", reply={"prose": "", "objections": [{"id": "o1", "text": "no"}], "withdrawn": []})
    record["asks"].append({"ask_id": "a1", "ts": "t", "revision": 1, "message": "", "participants": [seat]})
    return record


# ---------------------------------------------------------------------------
# §19.7.4 bullet 7 — dedicated locked projection writer vs terminal writers
# ---------------------------------------------------------------------------


def test_projection_survives_concurrent_terminal_writes_and_omission_is_not_erasure(tmp_path):
    oh.record_requested(tmp_path, "t-dur", request_id="r1", attempt=1)
    # A terminal writer that OMITS owner_hurry merges around it (never erases).
    write_task_result(tmp_path, "t-dur", "completed", result="done", reason_code="ok")
    stored = load_task_result(tmp_path, "t-dur")
    assert stored["status"] == "completed"
    assert stored["owner_hurry"]["request_id"] == "r1"
    # The hurry writer mutates ONLY its two keys around the terminal result.
    oh.record_applied(tmp_path, "t-dur", attempt=1, effects={"plan_review": "x"})
    stored = load_task_result(tmp_path, "t-dur")
    assert stored["status"] == "completed" and stored["result"] == "done"
    assert stored["owner_hurry"]["state"] == oh.STATE_APPLIED
    # History rollover: a NEW attempt's request archives the current block.
    oh.record_requested(tmp_path, "t-dur", request_id="r2", attempt=2)
    stored = load_task_result(tmp_path, "t-dur")
    assert stored["owner_hurry"]["attempt_key"] == 2
    history = stored["owner_hurry_history"]
    assert history[-1]["request_id"] == "r1"
    assert history[-1]["archived_reason"] == "superseded_by_new_attempt"
    # Genuinely concurrent writers: N threads of both kinds never lose the block.
    def _terminal():
        write_task_result(tmp_path, "t-dur", "completed", result="done")

    def _hurry():
        oh.record_applied(tmp_path, "t-dur", attempt=2, effects={"plan_review": "y"})

    threads = [threading.Thread(target=t) for t in (_terminal, _hurry) * 8]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stored = load_task_result(tmp_path, "t-dur")
    assert stored["owner_hurry"]["attempt_key"] == 2
    assert stored["status"] == "completed"


# ---------------------------------------------------------------------------
# §19.7.4 bullet 8 — retry reset: crash-requeue exercised end to end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("crash_boundary", ["after_drain", "after_tool_call", "after_tool_execution"])
def test_fresh_attempt_replays_exact_owner_text_until_terminal(
    tmp_path, monkeypatch, crash_boundary,
):
    import ouroboros.loop as loop

    exact = "model must see these exact bytes  \n"
    task_id = f"owner-{crash_boundary}"
    write_owner_message(tmp_path, exact, task_id, msg_id="owner-model")
    first_ctx = _drain_ctx(tmp_path, task_id=task_id, attempt=1)
    first_messages = []
    loop._drain_incoming_messages(
        first_messages, _q.Queue(), tmp_path, task_id, None, set(), owner_ctx=first_ctx,
    )
    if crash_boundary in {"after_tool_call", "after_tool_execution"}:
        tool_call = {
            "id": "tc-1", "type": "function",
            "function": {"name": "probe", "arguments": "{}"},
        }
        monkeypatch.setattr(
            loop,
            "call_llm_with_retry",
            lambda *_args, **_kwargs: (
                {"role": "assistant", "tool_calls": [tool_call]}, 0.01,
            ),
        )
        monkeypatch.setattr(loop, "_server_web_allowed_by_task", lambda _ctx: False)
        loop._dispatch_round_model(
            SimpleNamespace(
                llm=object(), messages=first_messages, active_model="test-model",
                tool_schemas=[], active_effort="high", max_retries=0,
                drive_logs=tmp_path / "logs", task_id=task_id, round_idx=1,
                event_queue=None, accumulated_usage={}, task_type="task",
                active_use_local=False, tools=SimpleNamespace(_ctx=first_ctx),
                drive_root=tmp_path, context_fit_plan=None, active_context_mode="max",
                physical_send_claim=None,
            ),
            None,
            attempt_cap=1,
        )
    if crash_boundary == "after_tool_execution":
        from ouroboros.loop_tool_execution import StatefulToolExecutor, handle_tool_calls

        class _ProbeTools:
            CODE_TOOLS = set()

            def __init__(self):
                self.calls = []
                self._ctx = SimpleNamespace(
                    task_metadata={}, _request_wire_custom_receipts=(), drive_root=tmp_path,
                )

            def get_timeout(self, _name):
                return 10

            def execute(self, name, args):
                self.calls.append((name, args))
                return "durable tool effect"

            def execute_result(self, name, args):
                # The loop reads the typed dispatch seam (D02); adapt the text
                # exactly the way the real registry adapts a legacy handler.
                from ouroboros.tools.tool_result import LegacyTextResultAdapter

                return LegacyTextResultAdapter.from_text(name, self.execute(name, args))

        probe_tools = _ProbeTools()
        executor = StatefulToolExecutor()
        first_messages.append({"role": "assistant", "tool_calls": [tool_call]})
        try:
            assert handle_tool_calls(
                [tool_call], probe_tools, tmp_path / "logs", task_id, executor,
                first_messages, {"tool_calls": []}, lambda _text: None,
            ) == 0
        finally:
            executor.shutdown(wait=True)
        assert probe_tools.calls == [("probe", {})]
        assert first_messages[-1]["role"] == "tool"

    oh.retry_reset(tmp_path, tmp_path, task_id, reason="worker_crash_requeue")
    successor_ctx = _drain_ctx(tmp_path, task_id=task_id, attempt=2)
    successor_messages = []
    loop._drain_incoming_messages(
        successor_messages, _q.Queue(), tmp_path, task_id, None, set(), owner_ctx=successor_ctx,
    )
    observed = []
    monkeypatch.setattr(
        loop,
        "call_llm_with_retry",
        lambda _llm, messages, *_args, **_kwargs: observed.append(messages) or (
            {"role": "assistant", "content": "ok"}, 0.01,
        ),
    )
    monkeypatch.setattr(loop, "_server_web_allowed_by_task", lambda _ctx: False)
    loop._dispatch_round_model(
        SimpleNamespace(
            llm=object(), messages=successor_messages, active_model="test-model",
            tool_schemas=[], active_effort="high", max_retries=0,
            drive_logs=tmp_path / "logs", task_id=task_id, round_idx=1,
            event_queue=None, accumulated_usage={}, task_type="task",
            active_use_local=False, tools=SimpleNamespace(_ctx=successor_ctx),
            drive_root=tmp_path, context_fit_plan=None, active_context_mode="max",
            physical_send_claim=None,
        ),
        None,
        attempt_cap=1,
    )
    wire = json.dumps(observed[0], ensure_ascii=False)
    assert exact.rstrip("\n") in wire
    assert wire.count("model must see these exact bytes") == 1


# ---------------------------------------------------------------------------
# §19.7.4 bullets 9+10 — no-chat counters and P3/safety non-consultation
# ---------------------------------------------------------------------------
