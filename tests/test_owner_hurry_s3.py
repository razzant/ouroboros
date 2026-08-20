"""§19.7.4 mandatory no-chat and durability proofs for the owner hurry control (S3).

Every proof runs against the PRODUCTION seams — ``gateway.tasks.api_task_hurry``
(re-export of ``gateway/task_hurry.py``), ``loop._drain_incoming_messages``,
``loop._no_tool_final_answer``, ``loop._run_task_acceptance_review_once``,
``task_pacing.improvement_pass_allowed`` + the rails line,
``loop._enforce_swarm_actions``, the dedicated locked projection writer, and the
one shared retry-reset invoked by the ``supervisor/workers.py`` crash-requeue.

The HQ1 contract under test: hurry is a typed task-local control (owner-mailbox
``kind=hurry``) that NEVER creates a chat message, owner directive, outbox row,
or bubble; its effects are host rails only (acceptance skip, zero remaining
improvement passes, task-local advisory force-plan projection); P3/commit/
safety never consult it; retry attempts start with no executable latch.
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


def _chat_log_snapshot(drive_root):
    path = pathlib.Path(drive_root) / "logs" / "chat.jsonl"
    return path.read_text(encoding="utf-8") if path.exists() else None


# ---------------------------------------------------------------------------
# §19.7.4 bullet 1 — production endpoint: drives, projection, refusals,
# idempotency, terminal TOCTOU
# ---------------------------------------------------------------------------


def test_direct_task_mailbox_lands_on_canonical_drive_with_projection(tmp_path, monkeypatch):
    task = {"id": "root-1", "chat_id": 0, "root_task_id": "root-1", "_attempt": 1}
    _isolate_queue(monkeypatch, tmp_path, running={"root-1": {"task": task, "attempt": 1}})
    before_chat = _chat_log_snapshot(tmp_path)
    with _client(tmp_path) as client:
        resp = client.post("/api/tasks/root-1/hurry", json={"request_id": "req-1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "ok": True, "task_id": "root-1", "request_id": "req-1",
        "state": "requested", "attempt_key": 1, "duplicate": False,
    }
    # Physical mailbox: DIRECT task -> canonical drive (same rule as steer_task).
    rows = _hurry_rows(tmp_path, "root-1")
    assert len(rows) == 1
    assert rows[0]["msg_id"] == "hurry:req-1"
    assert rows[0]["text"] == "owner_hurry"          # parser-required reason, not prose
    # Canonical projection: requested block keyed by the real attempt.
    block = load_task_result(tmp_path, "root-1")["owner_hurry"]
    assert block["attempt_key"] == 1
    assert block["request_id"] == "req-1"
    assert block["state"] == oh.STATE_REQUESTED
    assert block["reason"] == oh.REASON_OWNER_HURRY
    # NO-CHAT: the ingress produced no chat row anywhere.
    assert _chat_log_snapshot(tmp_path) == before_chat
    # The one durable non-chat event row is is_progress=False.
    events = [
        json.loads(line)
        for line in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    hurry_events = [e for e in events if e.get("type") == "owner_hurry"]
    assert len(hurry_events) == 1
    assert hurry_events[0]["phase"] == "requested"
    assert hurry_events[0]["is_progress"] is False
    assert hurry_events[0]["requested_by"] == "owner"


def test_forked_task_mailbox_goes_to_the_child_drive(tmp_path, monkeypatch):
    child_drive = tmp_path / "forks" / "root-2"
    child_drive.mkdir(parents=True)
    task = {
        "id": "root-2", "chat_id": 0, "root_task_id": "root-2",
        "child_drive_root": str(child_drive), "_attempt": 3,
    }
    _isolate_queue(monkeypatch, tmp_path, running={"root-2": {"task": task, "attempt": 3}})
    with _client(tmp_path) as client:
        resp = client.post("/api/tasks/root-2/hurry", json={"request_id": "req-f"})
    assert resp.status_code == 200
    assert resp.json()["attempt_key"] == 3
    # Mailbox on the CHILD drive, projection on the CANONICAL drive.
    assert len(_hurry_rows(child_drive, "root-2")) == 1
    assert not _mailbox_path(tmp_path, "root-2").exists()
    assert load_task_result(tmp_path, "root-2")["owner_hurry"]["attempt_key"] == 3


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
    # The worker-side projection went to APPLIED with the three named effects.
    block = load_task_result(tmp_path, "t-drain")["owner_hurry"]
    assert block["state"] == oh.STATE_APPLIED
    assert block["effects"] == {
        "improvement_passes": "zeroed_for_attempt",
        "task_acceptance": "skip_next_panel",
        "plan_review": "task_local_advisory",
    }
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
    # Same-process double drain (second hurry entry): still ONE latch.
    write_owner_message(tmp_path, "owner_hurry", "t-re", msg_id="hurry:r2", kind=KIND_HURRY)
    loop._drain_incoming_messages([], _q.Queue(), tmp_path, "t-re", None, set(), owner_ctx=restarted)
    assert oh.latched(restarted) == first_latch        # first arm wins


# ---------------------------------------------------------------------------
# §19.7.4 bullet 3 — production _no_tool_final_answer: hurry is not a follow-up
# ---------------------------------------------------------------------------


def test_post_pass_hurry_drain_never_supersedes_acceptance(tmp_path, monkeypatch):
    from tests.test_delivery_forced_finalization import _forced_test_context

    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    from ouroboros import loop_delivery

    monkeypatch.setattr(loop_delivery, "_compute_subagent_handoff", lambda *_a, **_k: None)
    monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", lambda *_a, **_k: False)
    monkeypatch.setattr(loop, "_run_task_acceptance_review_once", lambda **_k: False)
    superseded = []
    monkeypatch.setattr(
        loop, "_supersede_task_acceptance_for_owner_followup",
        lambda *a, **k: superseded.append((a, k)),
    )
    # Acceptance already terminal; the post-answer admission drain finds a
    # freshly delivered hurry control.
    registry._ctx._task_acceptance_reviewed = True
    registry._ctx.owner_message_admission_lock = threading.Lock()
    registry._ctx.owner_message_admission_agent = SimpleNamespace(
        _accepting_owner_messages=True, _busy=True, _current_task_id="parent1",
    )
    registry._ctx.budget_drive_root = str(tmp_path)
    registry._ctx.task_attempt = 1
    write_owner_message(tmp_path, "owner_hurry", "parent1", msg_id="hurry:late", kind=KIND_HURRY)
    oh.record_requested(tmp_path, "parent1", request_id="late", attempt=1)

    result = loop._no_tool_final_answer(
        "Final answer.", ctx, trace, registry, _q.Queue(), set(), lambda _m: None,
    )
    # The final answer went through: hurry never re-opened acceptance.
    assert result is not None
    assert superseded == []
    assert getattr(registry._ctx, "_owner_directives", []) == []
    assert oh.latched(registry._ctx) is not None       # ...but the latch IS armed


# ---------------------------------------------------------------------------
# §19.7.4 bullet 4 — production acceptance seam: typed skip, zero reviewer calls
# ---------------------------------------------------------------------------


def _acceptance_ctx(tmp_path, *, latched):
    ctx = SimpleNamespace(
        _task_acceptance_reviewed=False,
        is_direct_chat=False,
        drive_root=str(tmp_path),
        budget_drive_root=str(tmp_path),
        task_metadata={},
        task_contract={},
        task_attempt=1,
        task_id="t-acc",
    )
    if latched:
        ctx._owner_hurry_latch = {"msg_id": "hurry:x", "requested_at": "", "reason": oh.REASON_OWNER_HURRY}
    return ctx


def test_acceptance_panel_skips_with_typed_reason_and_zero_reviewer_calls(tmp_path, monkeypatch):
    import ouroboros.loop as loop_mod
    import ouroboros.review_substrate as rs

    monkeypatch.setattr(loop_mod, "get_task_review_mode", lambda: "required")
    reviewer_calls = []
    monkeypatch.setattr(
        rs, "reviewer_slots",
        lambda **kw: reviewer_calls.append(kw) or [object()],
    )
    oh.record_requested(tmp_path, "t-acc", request_id="rq", attempt=1)
    ctx = _acceptance_ctx(tmp_path, latched=True)
    trace = {"tool_calls": [{"tool": "write_file", "args": {"path": "x.py"}}]}
    assert loop_mod._run_task_acceptance_review_once(
        tools=SimpleNamespace(_ctx=ctx), content="done", task_id="t-acc", task_type="task",
        llm_trace=trace, drive_root=None,
        messages=[{"role": "user", "content": "goal"}], emit_progress=lambda _m: None,
    ) is False
    # Exact typed vocabulary (§19.7.2 item 8).
    assert trace["review_decision"] == {
        "eligibility": "eligible", "trigger": "owner_hurry", "skipped": "owner_hurry",
    }
    decision = trace["acceptance_decision"]
    assert decision["status"] == "finalized_unaccepted"
    assert decision["reason"] == oh.REASON_OWNER_HURRY == "owner_hurry"
    assert decision["source"] == "owner_hurry"
    # ZERO reviewer calls; the host panel is complete for this attempt.
    assert reviewer_calls == []
    assert ctx._task_acceptance_reviewed is True
    # The named effect landed in the durable projection.
    block = load_task_result(tmp_path, "t-acc")["owner_hurry"]
    assert block["effects"]["task_acceptance"] == "skipped_owner_hurry"
    # owner_hurry stays OUT of the forced-rail bypass vocabulary.
    from ouroboros.outcomes import ACCEPTANCE_BYPASS_REASON_BY_RAIL, BEST_EFFORT_REASON_CODES

    assert "owner_hurry" not in ACCEPTANCE_BYPASS_REASON_BY_RAIL.values()
    assert "owner_hurry" not in BEST_EFFORT_REASON_CODES


def test_unlatched_task_reaches_the_normal_acceptance_machinery(tmp_path, monkeypatch):
    """Control group: without the latch, the same context proceeds past the
    hurry seam (here: to the acceptance fence/pacing machinery, observed via
    the pacing profile resolve)."""
    import ouroboros.loop as loop_mod
    from ouroboros import task_pacing

    monkeypatch.setattr(loop_mod, "get_task_review_mode", lambda: "required")
    resolved = []
    real = task_pacing.resolve_budget_profile
    monkeypatch.setattr(
        task_pacing, "resolve_budget_profile",
        lambda c: resolved.append(True) or real(c),
    )
    monkeypatch.setattr(loop_mod, "get_review_enforcement", lambda: "advisory")
    ctx = _acceptance_ctx(tmp_path, latched=False)
    trace = {"tool_calls": [{"tool": "write_file", "args": {"path": "x.py"}}]}
    loop_mod._run_task_acceptance_review_once(
        tools=SimpleNamespace(_ctx=ctx), content="done", task_id="t-acc", task_type="task",
        llm_trace=trace, drive_root=None,
        messages=[{"role": "user", "content": "goal"}], emit_progress=lambda _m: None,
    )
    assert trace["review_decision"].get("skipped") != "owner_hurry"
    assert resolved, "the unlatched path must reach the real pacing machinery"


# ---------------------------------------------------------------------------
# §19.7.4 bullet 5 — real improvement_pass_allowed under Required+Blocking +
# the rails display consume the same effective zero cap
# ---------------------------------------------------------------------------


def test_required_blocking_unbounded_loop_collapses_to_zero_under_hurry(tmp_path, monkeypatch):
    from ouroboros import task_pacing

    ctx = _acceptance_ctx(tmp_path, latched=True)
    base_profile = task_pacing.resolve_budget_profile(ctx)
    assert base_profile.get("max_improvement_passes") is None
    # WITHOUT hurry, Required+Blocking has NO local count cap while the shared
    # review-cycle cap is unlimited (D10/D20: the shared cap otherwise binds).
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "unlimited")
    assert task_pacing.effective_max_improvement_passes(
        base_profile, has_deadline=False, required_blocking=True,
    ) is None
    effective = oh.effective_budget_profile(ctx, base_profile)
    assert effective["max_improvement_passes"] == 0
    # The IMMUTABLE input profile object was never mutated.
    assert base_profile.get("max_improvement_passes") is None
    snapshot = task_pacing.build_budget_snapshot(ctx, profile=effective)
    allowed, reason = task_pacing.improvement_pass_allowed(
        snapshot, 0, effective, required_blocking=True,
    )
    assert allowed is False
    assert reason == "improvement_passes_exhausted"
    # The rails/display consumer observes the SAME effective zero cap.
    from ouroboros import task_pacing as tp_mod

    rails = tp_mod.acceptance_rails_line(
        snapshot, effective, 0, None, required_blocking=True,
    )
    assert "review passes: 0/0" in rails
    assert "no local count cap" not in rails
    # An unlatched ctx passes the profile through UNCHANGED (identity).
    unlatched = _acceptance_ctx(tmp_path, latched=False)
    assert oh.effective_budget_profile(unlatched, base_profile) is base_profile


# ---------------------------------------------------------------------------
# §19.7.4 bullet 6 — production _enforce_swarm_actions force-plan projection
# ---------------------------------------------------------------------------


def _plan_state(kind: str) -> dict:
    fingerprint = "a" * 64
    state = {
        "schema_version": 1, "current_attempt": {},
        "latest_review_fingerprint": "", "waves": [],
    }
    if kind == "absent":
        return state
    if kind == "pending":
        state["waves"] = [{"request_fingerprint": fingerprint, "phase": "collected"}]
        return state
    state["current_attempt"] = {
        "fingerprint": fingerprint,
        "status": "unavailable" if kind == "unavailable" else "open",
        "reason": "reviewer unavailable" if kind == "unavailable" else "",
    }
    if kind == "open":
        state["latest_review_fingerprint"] = fingerprint
        state["waves"] = [{
            "request_fingerprint": fingerprint, "phase": "reviewed",
            "review_evidence_status": "integrated",
            "review": {"aggregate_signal": "REVIEW_REQUIRED", "closed": False},
        }]
    return state


@pytest.mark.parametrize(
    ("kind", "hurry_allows"),
    [
        ("open", True), ("unavailable", True),        # locally advisory/proceed
        ("absent", False), ("pending", False),        # remain hold
    ],
)
def test_enforce_swarm_actions_task_local_advisory_matrix(tmp_path, monkeypatch, kind, hurry_allows):
    import ouroboros.loop as loop_mod
    import ouroboros.task_results as tr

    monkeypatch.setattr(loop_mod, "get_review_enforcement", lambda: "blocking")
    monkeypatch.setattr(tr, "load_plan_review_state", lambda _root, _tid: _plan_state(kind))

    def _swarm_held(ctx):
        trace = {"reasoning_notes": []}
        tools = SimpleNamespace(_ctx=ctx)
        held = loop_mod._enforce_swarm_actions("answer", [], tools, trace, lambda _m: None)
        return held, trace

    latched_ctx = _acceptance_ctx(tmp_path, latched=True)
    latched_ctx.task_metadata = {"force_plan": True}
    latched_ctx.is_ephemeral_turn = False
    held, trace = _swarm_held(latched_ctx)
    assert held is (not hurry_allows)
    decision = trace["force_plan_decision"]
    if hurry_allows:
        # Attribution rides the decision for the task detail.
        assert decision["owner_hurry_local_advisory"] is True
        assert decision["configured_enforcement"] == "blocking"
    # WITHOUT the latch the same blocking install always holds these states.
    unlatched_ctx = _acceptance_ctx(tmp_path, latched=False)
    unlatched_ctx.task_metadata = {"force_plan": True}
    unlatched_ctx.is_ephemeral_turn = False
    held_unlatched, _t = _swarm_held(unlatched_ctx)
    assert held_unlatched is True
    # Durable review state was never touched (the monkeypatched loader is the
    # only read; nothing wrote a plan-review file).
    assert not (tmp_path / "task_results").exists() or "plan_review" not in json.dumps(
        load_task_result(tmp_path, "t-acc") or {},
    )


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
    oh.record_applied(tmp_path, "t-dur", attempt=1, effects={"task_acceptance": "x"})
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


def test_writer_never_routes_through_write_task_result():
    """The projection writer must not carry a status argument that could
    regress/drop the write — structural pin on the module source."""
    import inspect

    source = inspect.getsource(oh)
    # Docstrings NAME the forbidden writer; the code must never CALL or import it.
    assert "write_task_result(" not in source
    assert "import write_task_result" not in source
    assert "update_json_locked" in source


# ---------------------------------------------------------------------------
# §19.7.4 bullet 8 — retry reset: crash-requeue exercised end to end
# ---------------------------------------------------------------------------


def test_crash_requeue_runs_the_shared_retry_reset(tmp_path, monkeypatch):
    """The §19.7.2 item 11 mandated path: supervisor/workers.py crash-requeue
    (_attempt+1 same-id front requeue) must strip the executable control AND
    archive the projection before the retry attempt is admitted."""
    from tests.test_terminal_durability_v664 import _crashed_worker, _install_supervisor

    _queue, _state, workers, events = _install_supervisor(tmp_path, monkeypatch)
    task = {"id": "crash-1", "type": "task", "chat_id": 0, "_attempt": 1}
    # The dead attempt had an executable hurry: mailbox control + applied block.
    write_owner_message(tmp_path, "owner_hurry", "crash-1", msg_id="hurry:r1", kind=KIND_HURRY)
    oh.record_requested(tmp_path, "crash-1", request_id="r1", attempt=1)
    oh.record_applied(tmp_path, "crash-1", attempt=1)
    _crashed_worker(monkeypatch, workers, task)
    monkeypatch.setattr(workers, "load_state", lambda: {})

    workers.ensure_workers_healthy()

    # Same-id front requeue with the bumped attempt.
    assert [(t["id"], t["_attempt"]) for t in workers.PENDING] == [("crash-1", 2)]
    # NO executable latch survives: the mailbox (the control carrier) is gone.
    assert not _mailbox_path(tmp_path, "crash-1").exists()
    # ...and the current block is archived into history, not silently dropped.
    stored = load_task_result(tmp_path, "crash-1")
    assert "owner_hurry" not in stored
    history = stored["owner_hurry_history"]
    assert history[-1]["request_id"] == "r1"
    assert history[-1]["archived_reason"] == "worker_crash_requeue"


def test_retry_reset_is_shared_by_reaper_and_evolution_retry(tmp_path):
    """The reaper timeout path calls the SAME helper (source pin), and the
    helper itself is idempotent + fail-soft for any same-id requeue producer."""
    import inspect

    from supervisor import task_reaper

    assert "retry_reset" in inspect.getsource(task_reaper.reap_timed_out_task)
    # Functional: mailbox removed, block archived, and a second call is a no-op.
    write_owner_message(tmp_path, "owner_hurry", "evo-1", msg_id="hurry:r1", kind=KIND_HURRY)
    oh.record_requested(tmp_path, "evo-1", request_id="r1", attempt=1)
    oh.retry_reset(tmp_path, tmp_path, "evo-1", reason="evolution_retry")
    assert not _mailbox_path(tmp_path, "evo-1").exists()
    stored = load_task_result(tmp_path, "evo-1")
    assert "owner_hurry" not in stored
    assert stored["owner_hurry_history"][-1]["archived_reason"] == "evolution_retry"
    oh.retry_reset(tmp_path, tmp_path, "evo-1", reason="evolution_retry")   # idempotent
    assert len(load_task_result(tmp_path, "evo-1")["owner_hurry_history"]) == 1


# ---------------------------------------------------------------------------
# §19.7.4 bullets 9+10 — no-chat counters and P3/safety non-consultation
# ---------------------------------------------------------------------------


def test_full_hurry_lifecycle_writes_zero_chat_rows_and_zero_bus_calls(tmp_path, monkeypatch):
    import ouroboros.loop as loop

    bus_calls = []
    import supervisor.message_bus as mb

    monkeypatch.setattr(
        mb, "log_chat", lambda *a, **k: bus_calls.append(("log_chat", a, k)),
    )
    task = {"id": "nochat-1", "chat_id": 7, "root_task_id": "nochat-1", "_attempt": 1}
    _isolate_queue(monkeypatch, tmp_path, running={"nochat-1": {"task": task, "attempt": 1}})
    with _client(tmp_path) as client:
        client.post("/api/tasks/nochat-1/hurry", json={"request_id": "rq"})
    events: list = []
    ctx = _drain_ctx(tmp_path, task_id="nochat-1", attempt=1)
    loop._drain_incoming_messages(
        [], _q.Queue(), tmp_path, "nochat-1",
        SimpleNamespace(put_nowait=events.append), set(), owner_ctx=ctx,
    )
    oh.reconcile_terminal(tmp_path, "nochat-1")
    # chat.jsonl was never created; the message bus was never called.
    assert _chat_log_snapshot(tmp_path) is None
    assert bus_calls == []
    # Every emitted live event is a log_event envelope carrying the non-chat
    # owner_hurry family, is_progress=False — never send_message/chat frames.
    assert events, "the applied event must exist"
    assert {e.get("type") for e in events} == {"log_event"}
    payloads = [e["data"] for e in events]
    assert {p.get("type") for p in payloads} == {"owner_hurry"}
    assert all(p["is_progress"] is False for p in payloads)


def test_p3_commit_and_safety_surfaces_never_consult_hurry(monkeypatch, tmp_path):
    """§19.7.2 item 10: no hurry predicate/import enters commit review, advisory
    pre-review, triad/scope, deterministic gates, safety, or the tool boundary;
    and arming the latch leaves settings/env byte-identical."""
    import os

    repo = pathlib.Path(oh.__file__).resolve().parent.parent
    guarded = [
        repo / "ouroboros" / "safety.py",
        repo / "ouroboros" / "tools" / "registry.py",
        repo / "ouroboros" / "tools" / "git.py",
        repo / "ouroboros" / "tools" / "parallel_review.py",
        repo / "ouroboros" / "tools" / "claude_advisory_review.py",
        repo / "prompts" / "SAFETY.md",
    ]
    for path in guarded:
        assert path.exists(), path
        text = path.read_text(encoding="utf-8")
        assert "owner_hurry" not in text, f"{path} must not consult hurry"
        assert "KIND_HURRY" not in text, f"{path} must not consult hurry"
    env_before = dict(os.environ)
    from ouroboros.config import get_review_enforcement, get_task_review_mode

    enforcement_before = get_review_enforcement()
    mode_before = get_task_review_mode()
    ctx = _drain_ctx(tmp_path, task_id="t-env", attempt=1)
    oh.record_requested(tmp_path, "t-env", request_id="r", attempt=1)
    oh.apply_latch(ctx, {"msg_id": "hurry:r", "ts": "2026-08-15T00:00:00Z"})
    assert oh.acceptance_skip_decision(ctx) is not None
    assert dict(os.environ) == env_before
    assert get_review_enforcement() == enforcement_before
    assert get_task_review_mode() == mode_before
