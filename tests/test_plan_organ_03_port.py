"""Port of Ouroboros contribution 03 "An already performed review reaches its author" onto the
planning organ (task d4ce874483ff407a, contribution tree HEAD aaa0bdbc2): already-paid answers
stay available after an interruption, an attempt change and the author's completion; they are
never re-paid; an old approval never passes as approval of a new plan; reading is never a
disposition; UNKNOWN is never cleaned. The organ's entries that collect — the free READ, the
gate pre-collection and the late-result hook — go through the shared recorded-actor collector,
over the real ``{op}_prompt`` / ``{op}_response`` CAS the substrate retains (the fake substrate
here writes the same records the real coordinator writes).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from ouroboros import owner_hurry as oh
from ouroboros import plan_record, review_operation
from ouroboros.observability import persist_call
from ouroboros.review_dispatch import review_operation_binding, review_reconciliation_identity
from ouroboros.review_records import ReviewActorRecord
from ouroboros.tools import plan_review as pr
from tests._review_session_route_shared import (  # noqa: F401 - pytest fixtures (the offline delegated route)
    _owned_gateway_uses_each_test_transport,
    fake_route,
)
from tests.test_plan_task_tool import CLEAN, _call, _record, _reply, _slots, _write, harness as _harness

harness = _harness  # noqa: F811 - pytest fixture re-export


def _persist_prompt(drive, request, slot, operation):
    """What the real coordinator retains before any send: the exact request and slot under the
    operation's binding (``recorded_review_assignment`` proves a later collection against it)."""
    request.reconciliation_identity = review_reconciliation_identity(request, [slot], root_task_id=request.task_id)
    persist_call(drive, task_id=request.task_id, call_id=f"{operation}_prompt", call_type="plan_review_prompt",
                 payload={"request": asdict(request), "slot": asdict(slot)},
                 manifest={"review_operation_binding": review_operation_binding(request, slot, operation)})


def _complete_producer(drive, request, slot, operation, text):
    """The producer's complete outcome, retained after the worker that owned it is gone."""
    binding = review_operation_binding(request, slot, operation)
    actor = ReviewActorRecord(slot_id=slot.slot_id, model=slot.model, status="ok", raw_text=text, operation_id=operation,
                              recovery_binding=binding, usage={"cost": 0.01})
    persist_call(drive, task_id=request.task_id, call_id=f"{operation}_response", call_type="plan_review_response",
                 payload={"producer_outcome": asdict(actor), "message": {"content": text}, "usage": {"cost": 0.01}},
                 manifest={"producer_complete": True, "review_operation_binding": binding})


def _pending_row(slot, operation, **extra):
    return {"slot_id": slot.slot_id, "model": slot.model, "status": "error", "raw_text": "", "error": "", "usage": {},
            "prompt_ref": {}, "response_ref": {}, "operation_id": operation, "operation_state": "pending_dispatch",
            "late_result_pending": True, "awaiting_since": "2026-10-15T00:00:00+00:00", **extra}


def _settled_row(slot, operation, text):
    return {"slot_id": slot.slot_id, "model": slot.model, "status": "ok", "raw_text": text, "error": "", "usage": {"cost": 0.01},
            "prompt_ref": {}, "response_ref": {}, "operation_id": operation, "operation_state": "settled"}


class _Barrier:
    """A substrate that records every send, retains each seat's prompt CAS and leaves the seats
    it is told to hold pending at the barrier (the real custody's released rows)."""

    def __init__(self, drive, *, pending, settled=(), row_extra=None):
        self.drive, self.pending, self.settled, self.row_extra = drive, set(pending), dict(settled), dict(row_extra or {})
        self.calls, self.sent = [], {}

    def __call__(self, request, *, slots, drive_root, llm, usage_ctx=None):
        self.calls.append([s.slot_id for s in slots])
        actors = []
        for slot in slots:
            operation = f"op-{slot.slot_id}-{len(self.calls)}"
            _persist_prompt(drive_root, request, slot, operation)
            self.sent[slot.slot_id] = (request, slot, operation)
            if slot.slot_id in self.pending:
                actors.append(_pending_row(slot, operation, **self.row_extra.get(slot.slot_id, {})))
            else:
                actors.append(_settled_row(slot, operation, self.settled.get(slot.slot_id, CLEAN)))
        return SimpleNamespace(actors=actors)


def _install(harness, barrier):
    import ouroboros.review_substrate as rs

    harness.monkeypatch.setattr(rs, "run_review_request", barrier)
    return barrier


def _rows(harness, task_id="task-1"):
    return {p["seat_id"]: p for p in _record(harness, task_id)["asks"][0]["participants"]}


# ---------------------------------------------------------------- D6: the organ collects through the shared collector

def test_cold_collection_reads_a_delegated_seat_attach_only_and_names_a_dead_api_seat_unknown(harness):
    """03 D6 (probe P3 inverted): after a process loss a paid delegated seat whose answer
    exists in the engine reaches the author attach-only; a dead API seat is unresolved —
    neither a failure nor a reply, never re-sent; nothing is dispatched a second time."""
    harness.state["slots"] = _slots(("s1", "m/a", "session"), ("s2", "m/b"), ("s3", "m/c"))
    barrier = _install(harness, _Barrier(harness.drive, pending={"s1", "s2"}, row_extra={"s1": {"pending_invocation_id": "inv-s1"}}))
    observed = []

    def observe(root, request, slot, row, operation_id, *, controller_known=True, usage_ctx=None):
        observed.append((operation_id, controller_known))
        return ReviewActorRecord(slot_id=slot.slot_id, model=slot.model, status="ok", operation_id=operation_id,
                                 operation_state="late_settled",
                                 raw_text=_reply("The delegated run finished.", objections=[("o1", "late objection")])), review_operation.COLLECTED

    harness.monkeypatch.setattr(review_operation, "_observe_delegated_review", observe)
    ctx = harness.make_ctx()
    _write(ctx, ask="council")
    assert {s: p["status"] for s, p in _rows(harness).items()} == {"s1": "pending", "s2": "pending", "s3": "answered"}
    cold = harness.make_ctx()  # a new process: no in-memory custody, a new attempt
    cold.task_attempt = 2
    text, _result = _call(cold)  # the free READ collects
    rows = _rows(harness)
    assert rows["s1"]["status"] == "answered" and rows["s1"]["reply"]["objections"] == [{"id": "o1", "text": "late objection"}]
    assert observed == [("op-s1-1", True)], "the delegated run was read once, attach-only"
    assert rows["s2"]["status"] == "pending" and rows["s2"]["unresolved"] is True and "never re-sent" in rows["s2"]["error"]
    assert rows["s2"]["failure_code"] == "review_custody_lost" and rows["s3"]["status"] == "answered"
    assert barrier.calls == [["s1", "s2", "s3"]], "one physical dispatch, ever"
    assert pr._decision(cold, _record(harness))["holding"] == ["s1:o1"], "the late objection holds like any other"
    assert "s2 (m/b): pending — outcome unknown" in text and "a new ask to it may pay twice" in text
    # Reading again changes nothing: the unknown seat is never cleaned and never re-sent.
    _call(cold)
    assert _rows(harness)["s2"]["unresolved"] is True and barrier.calls == [["s1", "s2", "s3"]]


def test_a_seat_a_live_controller_still_owns_is_deferred_not_read_as_dead(harness):
    """The other direction of D6: the recorded controller of another live process keeps the
    seat pending (not unresolved); the gate reads the same through its pre-collection."""
    harness.state["slots"] = _slots(("s1", "m/a"), ("s2", "m/b"))
    owner = {"pid": 424242, "birth": "2026-10-15T00:00:00", "custody_session": "other-process"}
    barrier = _install(harness, _Barrier(harness.drive, pending={"s1"}, row_extra={"s1": {"usage": {"review_controller": owner}}}))
    harness.monkeypatch.setattr(review_operation, "controller_state", lambda identity: "alive" if identity == owner else "dead")
    ctx = harness.make_ctx(force_plan=True)
    _write(ctx, ask="council")
    assert _rows(harness)["s1"]["review_controller"] == owner, "the owning controller rides the participant row"
    cold = harness.make_ctx(force_plan=True)
    cold.task_attempt = 2
    _call(cold)
    decision = oh.plan_gate_decision(cold, {}, enforcement="blocking")  # the gate pre-collects the same way
    rows = _rows(harness)
    assert rows["s1"]["status"] == "pending" and not rows["s1"].get("unresolved") and rows["s2"]["status"] == "answered"
    assert decision["status"] == "awaiting" and decision["allow"] is True and decision["unknown_outcome"] == []
    assert barrier.calls == [["s1", "s2"]]
    # Once that controller is gone the same seat is read as unknown — and only then.
    harness.monkeypatch.setattr(review_operation, "controller_state", lambda identity: "dead")
    _call(cold)
    row = _rows(harness)["s1"]
    assert row["status"] == "pending" and row["unresolved"] is True and barrier.calls == [["s1", "s2"]]
    assert oh.plan_gate_decision(cold, {}, enforcement="blocking")["unknown_outcome"] == ["a1"]


# ---------------------------------------------------------------- D7 (option A): a v2 wave still owed at upgrade reaches the author as history

def test_a_v2_wave_still_owed_at_upgrade_reaches_the_author_as_history_without_writing_state(harness):
    """03 D7, the coordinator's option A: the earlier schema's owed wave is read through the same
    collector by the free READ and the gate pre-collection. A delegated seat still running in the
    engine stays owed (disclosed, holds nothing); once its run has finished its paid answer shows as
    read-only history ("not approval of the current revision") and the disclosed pending ask settles
    in memory; an API seat of the dead old process is unknown — named, never cleaned, never re-sent.
    The v2 state is never written; no objection or claim is created; nothing is sent."""
    import json

    from ouroboros.review_custody import _frozen_actor
    from ouroboros.review_records import ReviewRequest, ReviewRouteKind, ReviewSlot
    from tests._store_shared import read_result, seed_result
    from tests.test_plan_gate import _v2

    session = ReviewSlot(slot_id="old_a", model="cursor=grok", route=ReviewRouteKind.AGENT_SESSION, session_target="cursor=grok",
                         max_tokens=65536, default_temperature=0.2)
    api = ReviewSlot(slot_id="old_b", model="m/old", route=ReviewRouteKind.API_CHAT, max_tokens=65536, default_temperature=0.2)
    request = ReviewRequest(surface="plan_review", goal="the earlier plan review", task_id="task-1",
                            retry_key="plan:task-1:cycle-1", call_type="plan_review")
    rows = []
    for slot, operation, extra in ((session, "op-old-a", {"pending_invocation_id": "inv-old-a"}), (api, "op-old-b", {})):
        _persist_prompt(harness.drive, request, slot, operation)
        rows.append({"slot_id": slot.slot_id, "model": slot.model, "route": str(slot.route.value), "ok": False, "error": "",
                     "operation_id": operation, "operation_state": "pending_dispatch", "late_result_pending": True,
                     "recovery_binding": review_operation_binding(request, slot, operation), **extra})
    state = _v2(closed=False, blocking=())
    state["waves"][0].update(retry_key="plan:task-1:cycle-1", custody_pending=True, actors=rows)
    seed_result(harness.drive, "task-1", {"task_id": "task-1", "status": "running", "plan_review_state": state})
    before = json.dumps(read_result(harness.drive, "task-1")["plan_review_state"], sort_keys=True)
    finished = {"text": ""}

    def observe(root, request, slot, row, operation_id, *, controller_known=True, usage_ctx=None):  # attach-only
        if not finished["text"]:
            return _frozen_actor(row, slot), review_operation.DEFERRED  # still running
        return ReviewActorRecord(slot_id=slot.slot_id, model=slot.model, status="ok", raw_text=finished["text"],
                                 operation_id=operation_id, operation_state="late_settled"), review_operation.COLLECTED

    harness.monkeypatch.setattr(review_operation, "_observe_delegated_review", observe)
    sub = harness.install({})
    ctx = harness.make_ctx()
    _write(ctx)  # revision 1 on the organ; the earlier wave is still owed
    text, _result = _call(ctx)
    owed = oh.plan_gate_decision(ctx, {}, enforcement="blocking")
    assert "Legacy contributions" not in text and "legacy wave ffffffffffff: owed: old_a; unknown: old_b" in text
    assert owed["pending_asks"] == ["legacy wave " + "f" * 12] and owed["status"] == "awaiting" and owed["allow"] is True
    finished["text"] = "[]\nNO_FINDINGS: the earlier plan read fine."
    text, result = _call(ctx)
    assert "## Legacy contributions (history; not approval of the current revision)" in text
    assert "### old_a (cursor=grok) — answered the earlier plan review; read back from its retained source" in text
    assert "NO_FINDINGS: the earlier plan read fine." in text and "legacy wave ffffffffffff: unknown: old_b" in text
    assert result.meta["plan_open_objections"] == 0
    decision = oh.plan_gate_decision(ctx, {}, enforcement="blocking")
    assert decision["pending_asks"] == [] and decision["open_objections"] == [] and decision["status"] == "clear"
    assert json.dumps(read_result(harness.drive, "task-1")["plan_review_state"], sort_keys=True) == before, "the v2 state is never written"
    assert sub.calls == [], "nothing was sent"
    legacy, _error = plan_record.legacy_facts(harness.drive, "task-1")
    assert legacy["done_when"] == [] and legacy["open_objections"] == [], "history creates no claim and no objection"


# ---------------------------------------------------------------- the organ leg of 03 P2: a refusal before the paid stamp never reached a reviewer

def _fail(message):
    def raise_error(*args, **kwargs):
        raise OSError(message)
    return raise_error


@pytest.mark.serial
@pytest.mark.parametrize("stage", ["gateway", "checking", "budget"])
def test_a_refusal_raised_before_the_paid_stamp_is_never_a_reached_reviewer_on_the_organ(tmp_path, monkeypatch, fake_route, stage):  # noqa: F811
    """03 P2's organ leg, after M1–M4: the gateway, checking and budget refusals used to
    settle as a performed-but-failed seat (live) and recover as UNKNOWN (cold), so a Swarm
    force_plan counted them as a reached reviewer. Both readings are now the positive
    not_dispatched; the organ's facts withdraw the dispatch flag; the record reads the ask as
    attempted — disclosed, never held — and never as reached. (The session-source stage left
    with task acceptance's retrieving source preparation.)"""
    from dataclasses import replace

    from ouroboros import budget_pause, review_checking, review_execution, review_substrate, subagents
    from ouroboros.tools.plan_ask import reply_facts_from_row
    from ouroboros.tools.plan_review_runtime import _plan_row_from_actor
    from tests._review_session_route_shared import FakeLLM, _agent_request, _agent_slot

    request = _agent_request(task_id="presend", task_attempt=1, retry_key="one-review", session_root=str(tmp_path),
                             session_task="Weigh the plan.")
    slot = _agent_slot(session_access="readonly")
    if stage == "gateway":
        monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway", _fail("gateway unavailable"))
    elif stage == "checking":
        slot = replace(slot, session_access="workspace_write")
        monkeypatch.setattr(review_checking, "session_checking", _fail("checking unavailable"))
        monkeypatch.setattr(subagents, "route_health", lambda *a, **k: ("", ""))
    else:
        monkeypatch.setattr(budget_pause, "dispatch_fenced", lambda *_: True)
    stamps = []
    monkeypatch.setattr(review_execution, "invoke_review_paid_stamp", lambda stamp: stamps.append("paid"))
    ctx = SimpleNamespace(task_id="presend", root_task_id="presend", task_attempt=1, task_metadata={}, drive_root=tmp_path,
                          budget_drive_root=tmp_path)
    result = review_substrate.run_review_request(request, slots=[slot], drive_root=tmp_path, llm=FakeLLM(), usage_ctx=ctx)
    row = asdict(result.actors[0]) if not isinstance(result.actors[0], dict) else result.actors[0]
    assert stamps == [] and row["operation_state"] == "not_dispatched", "nothing was paid; the refusal is positive"
    original, original_slot = review_operation.recorded_review_assignment(
        tmp_path, task_id="presend", surface=request.surface, row=row, retry_key="one-review")
    recovered = review_operation.recover_review_producer(tmp_path, original, original_slot, row)
    assert recovered.operation_state == "not_dispatched" and sum(len(g.start_requests) for g in fake_route.instances) == 0
    live, cold = reply_facts_from_row(_plan_row_from_actor(row, slot)), reply_facts_from_row(_plan_row_from_actor(asdict(recovered), original_slot))
    for facts in (live, cold):
        assert facts["status"] == "not_dispatched" and facts["dispatched"] is False and facts["not_dispatched_reason"]
    seat = {"seat_id": slot.slot_id, "model": slot.model, "route": "agent_session", "owner_marked": True, "dispatched": True,
            "status": "pending", **cold}
    record = {**plan_record.empty_record(), "revisions": [{"n": 1, "ts": "t", "goal": "g", "plan": "p", "done_when": [],
                                                           "affected_paths": [], "body": False, "content_sha256": "0" * 64}],
              "asks": [{"ask_id": "a1", "ts": "t", "revision": 1, "message": "", "participants": [seat]}]}
    assert plan_record.asks_dispatched(record) == 0 and plan_record.unroutable_asks(record) == 1
    gate = plan_record.gate_projection(record, enforcement="blocking", cyber_pro=False, hurry=False)
    assert gate["status"] == "clear" and gate["allow"] is True
    # The other direction: a seat that WAS dispatched and merely pending keeps the ask reached.
    reached = {**record, "asks": [{**record["asks"][0], "participants": [{**seat, "status": "pending", "dispatched": True}]}]}
    assert plan_record.asks_dispatched(reached) == 1 and plan_record.unroutable_asks(reached) == 0


# ---------------------------------------------------------------- the rebinding matrix: collection reads the send-time identity

CHANGES = ["none", "prepare_before", "path", "attempt", "both", "deleted_workspace"]
ENTRIES = ["read", "identical", "gate"]


def _git_baseline(repo):
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_AUTHOR_NAME": "Fixture",
           "GIT_AUTHOR_EMAIL": "fixture@invalid", "GIT_COMMITTER_NAME": "Fixture", "GIT_COMMITTER_EMAIL": "fixture@invalid"}
    for args in (["init", "-q"], ["add", "."], ["commit", "-qm", "fixture baseline"]):
        subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True)


def _cold(harness, ctx):
    """A new ToolContext (another process: no in-memory custody) carrying today's roots and attempt."""
    cold = harness.make_ctx()
    for attr in ("repo_dir", "system_repo_dir", "serving_repo_dir", "task_attempt"):
        if getattr(ctx, attr, None) is not None:
            setattr(cold, attr, getattr(ctx, attr))
    return cold


@pytest.mark.parametrize(("change", "entry"), [(change, entry) for change in CHANGES for entry in ENTRIES
                                               if (change, entry) != ("deleted_workspace", "identical")])
def test_collection_reads_the_send_time_identity_whatever_changed_since(harness, change, entry):
    """03's rebinding matrix on the organ: the author's roots moved (a body candidate prepared
    before or after the ask), the attempt changed, both, or the workspace is gone — the free READ,
    an identical write followed by the READ, and the gate pre-collection all settle the paid
    answers from the send-time record (operation and binding unchanged) and never send again."""
    from ouroboros import body_candidate, subagent_worktrees

    harness.state["slots"] = _slots(("s1", "m/a"), ("s2", "m/b"), ("s3", "m/c"))
    harness.monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    harness.monkeypatch.setattr(subagent_worktrees, "get_subagent_worktree_root", lambda: str(harness.system.parent / "candidates"))
    _git_baseline(harness.system)
    barrier = _install(harness, _Barrier(harness.drive, pending={"s1", "s2", "s3"}))
    ctx = harness.make_ctx()
    if change == "prepare_before":
        body_candidate.prepare(ctx)
    _write(ctx, ask="council")
    assert barrier.calls == [["s1", "s2", "s3"]] and all(p["status"] == "pending" for p in _rows(harness).values())
    send_time = {seat: review_operation_binding(*barrier.sent[seat]) for seat in ("s1", "s2", "s3")}
    for seat in ("s1", "s2", "s3"):  # the producers complete after the author moved on
        _complete_producer(harness.drive, *barrier.sent[seat], _reply(f"{seat} answered after the author moved on."))
    if change in {"path", "both"}:
        body_candidate.prepare(ctx)
    if change in {"attempt", "both"}:
        ctx.task_attempt = 2
    if change == "deleted_workspace":
        shutil.rmtree(harness.workspace)  # only this disposable fixture's input
    cold = _cold(harness, ctx)
    if entry == "identical":
        text, result = _write(cold)  # identical content: nothing recorded, nothing sent
        assert "Identical content" in text and result.meta["plan_effect"] == "write" and barrier.calls == [["s1", "s2", "s3"]]
        text, _result = _call(cold)
    elif entry == "gate":
        decision = oh.plan_gate_decision(cold, {}, enforcement="blocking")
        assert decision["status"] == "clear" and decision["pending_asks"] == [] and decision["unknown_outcome"] == []
    else:
        text, _result = _call(cold)
    if entry != "gate":
        assert "s2 answered after the author moved on." in text
    rows = _rows(harness)
    assert {seat: row["status"] for seat, row in rows.items()} == {"s1": "answered", "s2": "answered", "s3": "answered"}
    assert {seat: row["operation_id"] for seat, row in rows.items()} == {seat: barrier.sent[seat][2] for seat in rows}
    assert {seat: row["recovery_binding"] for seat, row in rows.items()} == send_time, "the send-time binding, never today's"
    assert barrier.calls == [["s1", "s2", "s3"]], "collection never sends"
    record = _record(harness)
    assert len(record["asks"]) == 1 and len(record["revisions"]) == 1


def test_a_missing_original_assignment_keeps_that_seat_unknown_and_the_other_answers(harness):
    from ouroboros.observability import call_manifest_path

    harness.state["slots"] = _slots(("s1", "m/a"), ("s2", "m/b"), ("s3", "m/c"))
    barrier = _install(harness, _Barrier(harness.drive, pending={"s1", "s2", "s3"}))
    ctx = harness.make_ctx()
    _write(ctx, ask="council")
    for seat in ("s1", "s2", "s3"):
        _complete_producer(harness.drive, *barrier.sent[seat], CLEAN)
    call_manifest_path(harness.drive, "task-1", barrier.sent["s1"][2] + "_prompt").unlink()
    cold = harness.make_ctx()
    cold.task_attempt = 2
    text, _result = _call(cold)
    rows = _rows(harness)
    assert rows["s1"]["status"] == "pending" and rows["s1"]["unresolved"] is True
    assert "Original review assignment unavailable" in rows["s1"]["error"] and rows["s1"]["operation_id"] == barrier.sent["s1"][2]
    assert rows["s2"]["status"] == "answered" and rows["s3"]["status"] == "answered" and barrier.calls == [["s1", "s2", "s3"]]
    assert "s1 (m/a): pending — outcome unknown" in text
    # Every original assignment missing: nothing settles, nothing re-binds, nothing is re-sent.
    other = harness.make_ctx(task_id="task-2", force_plan=True)
    _write(other, ask="council")
    for seat in ("s1", "s2", "s3"):
        _complete_producer(harness.drive, *barrier.sent[seat], CLEAN)
        call_manifest_path(harness.drive, "task-2", barrier.sent[seat][2] + "_prompt").unlink()
    cold = harness.make_ctx(task_id="task-2", force_plan=True)
    cold.task_attempt = 2
    decision = oh.plan_gate_decision(cold, {}, enforcement="blocking")
    rows = _rows(harness, "task-2")
    assert all(row["status"] == "pending" and row["unresolved"] is True for row in rows.values())
    assert {seat: row["operation_id"] for seat, row in rows.items()} == {seat: barrier.sent[seat][2] for seat in rows}
    assert decision["status"] == "awaiting" and decision["allow"] is True and decision["unknown_outcome"] == ["a1"]
    assert decision["asks_dispatched"] > 0, "the ask was handed off; an unknown outcome is never 'nobody reached'"
    assert barrier.calls == [["s1", "s2", "s3"], ["s1", "s2", "s3"]]


# ---------------------------------------------------------------- the last settlement's frame counts every seat custody settled

def _drive_released_wave(tmp_path, monkeypatch, *, delay_s2=0.0, fail_s2=""):
    """Three seats released at the barrier through the REAL custody (the event-route helpers):
    s1 and s2 settle first, s3 last. ``delay_s2`` holds s2's record write so s3's settlement
    runs ahead of it; ``fail_s2="once"`` makes s2's OWN hook write (its worker thread) raise a
    store error once, ``"always"`` makes every s2 answered write fail. Returns (frames, statuses, progress)."""
    import threading
    import time

    from ouroboros.review_substrate import ReviewSlot
    from ouroboros.store import StoreError
    from tests.test_plan_review_event_route import _TASK, _custody_kwargs, _held_worker, _mailbox_entries, _plant_ask

    slots = [ReviewSlot(slot_id=f"s{index}", model=f"m/{index}", timeout_sec=30.0) for index in (1, 2, 3)]
    _plant_ask(tmp_path, slots)
    release = {slot.slot_id: threading.Event() for slot in slots}
    entered = {slot.slot_id: threading.Event() for slot in slots}
    _calls, run_slot = _held_worker(slots, {}, release, entered)
    progress: list = []
    ctx = SimpleNamespace(drive_root=tmp_path, emit_progress_fn=progress.append)
    request, kwargs = _custody_kwargs(tmp_path, surface="plan_review", retry_key=f"plan_ask:{_TASK}:a1", slots=slots,
                                      run_slot=run_slot, ctx=ctx, plan_ask_id="a1")
    request.drain_deadline = time.monotonic()
    real_update, left = plan_record.update_participant, {"once": fail_s2 == "once"}

    def interfered_update(root, task_id, *, ask_id, seat_id, facts):
        if seat_id == "s2" and facts.get("status") == "answered":
            if delay_s2:
                time.sleep(delay_s2)
            own_hook = threading.current_thread().name.endswith("-s2")  # custody names a worker thread after its seat
            if fail_s2 == "always" or (left["once"] and own_hook):
                left["once"] = False
                raise StoreError("store busy (fixture)")
        return real_update(root, task_id, ask_id=ask_id, seat_id=seat_id, facts=facts)

    monkeypatch.setattr(plan_record, "update_participant", interfered_update)
    import ouroboros.review_custody as custody

    deadline = time.time() + 15
    try:
        first = custody.run_custodied_review_slots(**kwargs)
        assert {actor.operation_state for actor in first} == {"pending_dispatch"}
        assert all(entered[slot.slot_id].wait(10) for slot in slots)
        release["s1"].set()
        release["s2"].set()
        while not all(any(f"reviewer m/{n} answered" in line for line in progress) for n in ("1", "2")) and time.time() < deadline:
            time.sleep(0.01)
        release["s3"].set()  # custody's progress line precedes the record write: s3 now races s2's hook
        while not _mailbox_entries(tmp_path, _TASK) and time.time() < deadline:
            time.sleep(0.01)
        time.sleep(max(delay_s2, 0.05) + 0.3)  # let the delayed hook land before reading the record
    finally:
        for event in release.values():
            event.set()
    frames = [frame["text"] for frame in _mailbox_entries(tmp_path, _TASK)]
    statuses = {p["seat_id"]: (p["status"], bool(p.get("unresolved")))
                for p in plan_record.load_plan_record(tmp_path, _TASK)["asks"][0]["participants"]}
    return frames, statuses, progress


def test_the_frame_counts_a_sibling_whose_record_write_is_still_in_flight(tmp_path, monkeypatch):
    """The three red battery runs: custody emits "reviewer m/2 answered" BEFORE s2's hook writes the
    record, so s3's settlement — the one that composes the frame — read s2 as still owed while
    custody had it settled. The frame-composing hook must absorb, from custody's own copies, every
    seat the record has not taken yet; the frame is then true and the mailbox wakes once."""
    frames, statuses, _progress = _drive_released_wave(tmp_path, monkeypatch, delay_s2=0.6)
    assert frames == [frames[0]] and frames[0].startswith("Plan ask a1: every participant settled — 3 answered"), frames
    assert statuses == {"s1": ("answered", False), "s2": ("answered", False), "s3": ("answered", False)}


def test_a_record_write_lost_to_a_store_failure_is_absorbed_or_named_never_still_owed(tmp_path, monkeypatch):
    """A transient store failure on s2's write: the frame-composing hook absorbs s2 from custody
    ($0, never a send) and the frame counts it. A failure that persists is NAMED — settled in
    custody, not yet in the plan record, plan_task() collects it — never "still owed"; the owner
    sees the failed write on the progress line; nothing is silently lost."""
    frames, statuses, progress = _drive_released_wave(tmp_path, monkeypatch, fail_s2="once")
    assert frames == [frames[0]] and frames[0].startswith("Plan ask a1: every participant settled — 3 answered"), frames
    assert statuses["s2"] == ("answered", False)
    assert any("s2" in line and "plan record could not take it" in line for line in progress), progress
    frames, statuses, progress = _drive_released_wave(tmp_path / "persistent", monkeypatch, fail_s2="always")
    assert len(frames) == 1 and "still owed" not in frames[0], frames
    assert "2 answered, 1 settled in custody but not yet in the plan record (s2" in frames[0] and "plan_task() collects" in frames[0]
    assert statuses["s2"] == ("pending", False) and statuses["s1"] == ("answered", False) and statuses["s3"] == ("answered", False)
