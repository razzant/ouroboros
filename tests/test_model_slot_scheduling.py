"""Scheduling a request: the axes it carries and the envelope that states them.

Split verbatim out of ``tests/test_model_slot_role_model.py`` by theme. This module owns the
executor as a third axis independent of lane and surface, the effort that is derived at
dispatch rather than owned by the owner, the deadline that narrows but never extends, and the
scheduling intent that survives a queue snapshot and a restart.
"""

from __future__ import annotations



import ouroboros.config as config

from tests._model_slot_role_shared import (
    _enqueue_through_supervisor,
    _scheduling_ctx,
)
from tests._model_slot_role_shared import _owned_gateway_uses_each_test_transport as __owned_gateway_uses_each_test_transport

# Fixtures are requested by name as test parameters, so they are re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
_owned_gateway_uses_each_test_transport = __owned_gateway_uses_each_test_transport


def test_executor_is_a_third_axis_independent_of_lane_and_surface(tmp_path):
    """WHO runs a child is its own axis. It is a closed enum of intents — never a harness
    name — so that adding a harness never touches this contract."""
    from ouroboros.subagents import SUBAGENT_EXECUTORS
    from ouroboros.tools.control import _schedule_task

    assert SUBAGENT_EXECUTORS == ("auto", "harness", "native")

    for executor in SUBAGENT_EXECUTORS:
        ctx = _scheduling_ctx(tmp_path / executor)
        out = _schedule_task(ctx, objective="o", expected_output="e", executor=executor)
        assert "TOOL_ARG_ERROR" not in out, executor
        assert ctx.event_queue.get_nowait()["requested_executor"] == executor

    ctx = _scheduling_ctx(tmp_path / "omitted")
    _schedule_task(ctx, objective="o", expected_output="e")
    assert ctx.event_queue.get_nowait()["requested_executor"] == "auto"

    ctx = _scheduling_ctx(tmp_path / "bad")
    out = _schedule_task(ctx, objective="o", expected_output="e", executor="codex")
    assert "TOOL_ARG_ERROR" in out and "executor must be one of" in out
    assert ctx.event_queue.empty()

def test_effort_is_not_an_owner_facing_axis(tmp_path):
    """There are THREE owner-facing axes and effort is not one of them (v6.87.28).

    A parent declares the WORK: write_surface (what the child may do), model_lane
    (how good the answer must be), executor (where it runs). A public `effort` broke
    that twice — it was a second knob for the question `model_lane` already answers,
    so `model_lane=light` with `effort=max` pinned the cheapest model to the
    strongest reasoning with no rule to reconcile them; and a harness route carries
    its own effort, so a parent asking `low` against a route pinned to `xhigh` had no
    rule for who wins. The refusal names the withdrawal instead of calling a
    parameter that was real for four releases 'unsupported'."""
    from ouroboros.tools.control import _schedule_task, schedule_subagent_properties

    assert "effort" not in schedule_subagent_properties()

    ctx = _scheduling_ctx(tmp_path / "named")
    out = _schedule_task(ctx, objective="o", expected_output="e", effort="xhigh")
    assert "TOOL_ARG_ERROR" in out and "effort was withdrawn" in out
    assert "model_lane" in out
    assert ctx.event_queue.empty()

    # The combination that had no answer is refused at the door, not ranked.
    ctx = _scheduling_ctx(tmp_path / "conflict")
    out = _schedule_task(ctx, objective="o", expected_output="e",
                         model_lane="light", effort="max")
    assert "TOOL_ARG_ERROR" in out
    assert ctx.event_queue.empty()

    # Scheduling states intent; nothing about effort is recorded there at all.
    ctx = _scheduling_ctx(tmp_path / "omitted")
    assert "TOOL_ARG_ERROR" not in _schedule_task(ctx, objective="o", expected_output="e")
    assert "reasoning_effort" not in ctx.event_queue.get_nowait()

def test_effort_is_derived_from_the_owner_setting_at_dispatch(tmp_path, monkeypatch):
    """Removing the knob did not remove the capability: the owner still controls
    effort through `config.resolve_effort(task_type)`, exactly as they did whenever
    the parameter was omitted — which was the normal case."""
    from ouroboros.agent import resolve_dispatch_axes

    monkeypatch.setenv("OUROBOROS_EFFORT_TASK", "xhigh")
    task = {"id": "c1", "type": "task", "delegation_role": "subagent"}
    dispatch = resolve_dispatch_axes(task)
    assert dispatch.effort == "xhigh"
    assert task["reasoning_effort"] == "xhigh"
    assert task["capability_delta"]["derived_effort"] == "xhigh"

    monkeypatch.setenv("OUROBOROS_EFFORT_TASK", "low")
    assert resolve_dispatch_axes({"id": "c2", "type": "task",
                                  "delegation_role": "subagent"}).effort == "low"

def test_a_stored_legacy_effort_is_ignored_with_the_reason_stated(tmp_path):
    """`effort` was model-visible and sits on durable records written before it was
    withdrawn. Loading one must not crash, must not obey it, and must not drop it in
    silence — a value that quietly stops meaning anything is the same class of defect
    as a reduction nobody announces."""
    from ouroboros.agent import capability_delta_prompt_block, resolve_dispatch_axes
    from ouroboros.subagents import LEGACY_SUBAGENT_FIELDS

    assert "reasoning_effort" in LEGACY_SUBAGENT_FIELDS

    task = {"id": "c1", "type": "task", "delegation_role": "subagent",
            "reasoning_effort": "max"}
    dispatch = resolve_dispatch_axes(task)
    # Not obeyed: the derived effort wins, whatever the record said.
    assert dispatch.effort == config.resolve_effort("task") != "max"
    assert task["reasoning_effort"] == dispatch.effort
    # ...and not dropped in silence.
    note = task["capability_delta"]["legacy_note"]
    assert "reasoning_effort='max'" in note and "derived" in note
    assert "Ignored on your record" in capability_delta_prompt_block(dispatch)
    # An ignored field is not a REDUCTION — nothing was taken away.
    assert task["capability_delta"]["reduced"] is False

    # A stray `effort` inside a stored task contract is dropped by the contract
    # builder rather than raising: contracts outlive the schema that wrote them.
    from ouroboros.contracts.task_contract import build_task_contract

    contract = build_task_contract({"id": "c1", "task_contract": {"effort": "max"}})
    assert "effort" not in contract

def test_the_request_reaches_the_worker_and_only_the_request(tmp_path, monkeypatch):
    """The parent's INTENT must reach the task the WORKER is handed — and nothing else.

    This asserts on the task the supervisor actually enqueues, not on the event and not on
    a re-implementation of the agent's fallback. An earlier version of this test built the
    payload itself from the event, which meant it supplied the very keys under test and
    could not fail — and a version before THAT re-implemented the agent's three lines in
    the test body. Both passed while the supervisor was silently dropping the keys on the
    floor. The loss is destructive, not merely inert: the worker writes its own view back
    over the durable record, so a drop here also erases the evidence of what was asked.

    The second half is the v6.87.28 invariant: what the child GETS is not on this task,
    because it has not been resolved. A schedule-time answer about live availability is
    an answer about a moment that has passed by the time the child starts."""
    task = _enqueue_through_supervisor(
        tmp_path, monkeypatch, parent_lane="heavy", executor="harness")
    assert task["requested_executor"] == "harness"
    assert task["requested_model_lane"] == "auto"
    assert task["parent_model_lane"] == "heavy"
    assert task["metadata"]["requested_executor"] == "harness"
    assert task["metadata"]["parent_model_lane"] == "heavy"
    for derived in ("effective_model_lane", "model", "use_local_model",
                    "reasoning_effort", "effective_executor", "capability_delta"):
        assert derived not in task, derived
        assert derived not in task["metadata"], derived

def test_availability_is_a_dispatch_fact_not_a_schedule_fact(tmp_path, monkeypatch):
    """The reason there is exactly one resolution and it runs at dispatch.

    A child scheduled while no harness route exists can wait out the whole outage in
    the queue. Resolving at schedule time froze the answer onto the record forever;
    resolving again at dispatch produced a SECOND record that disagreed with the first
    about the same child. With the D28 correction the down state is a typed BLOCK, so
    freezing it at schedule time would have refused a child whose route came back
    while it sat in the queue."""
    from ouroboros.agent import resolve_dispatch_axes
    from ouroboros.gateways import claudexor as gw

    task = _enqueue_through_supervisor(tmp_path, monkeypatch, executor="harness")

    # No route configured while the child sits in the queue: a typed BLOCK (D28).
    monkeypatch.delenv("OUROBOROS_SUBAGENT_HARNESS", raising=False)
    down = resolve_dispatch_axes(dict(task))
    assert (down.executor, down.route) == ("blocked", "")
    assert down.blocked is True and down.delta.reduced is True

    # The route comes back while the child is still queued. Health is a live probe
    # (p34's rule table), so the daemon is faked at the gateway seam the probe's
    # lazy import reads — the same seam test_delegated_subagent_transport fakes.
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "route-a")

    class _Healthy:
        engine_version = "9.9.9"

        def handshake(self, **_kw):
            return {}

        def agent_capabilities(self):
            return {"harnesses": [{"id": "route-a", "enabled": True, "status": "ok",
                                   "accessProfilesSupported": ["readonly", "workspace_write"]}]}

        def quota_snapshots(self):
            return []

        def close(self):
            pass

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Healthy())
    up = resolve_dispatch_axes(dict(task))
    assert (up.executor, up.route) == ("harness", "route-a")
    assert up.delta.reduced is False

def test_deadline_at_narrows_but_never_extends(tmp_path):
    """`deadline_at` is public as of v6.87.7, and narrowing-only: a child may be bound
    tighter than its parent, never looser."""
    from ouroboros.tools.control import _INTERNAL_SCHEDULE_OPTIONS, _schedule_task

    assert _INTERNAL_SCHEDULE_OPTIONS == frozenset()

    # Relative to now, not hardcoded: `deadline_at` must be a FUTURE instant, so fixed
    # calendar dates in this test would silently turn into rejections as time passes.
    from datetime import timedelta

    from ouroboros.deadline_utils import utc_now

    def stamp(hours):
        return (utc_now() + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")

    parent, tighter, looser = stamp(12), stamp(9), stamp(23)

    ctx = _scheduling_ctx(tmp_path / "tighter", parent_deadline=parent)
    _schedule_task(ctx, objective="o", expected_output="e", deadline_at=tighter)
    evt = ctx.event_queue.get_nowait()
    assert evt["task_contract"]["deadline_at"] == tighter

    ctx = _scheduling_ctx(tmp_path / "looser", parent_deadline=parent)
    _schedule_task(ctx, objective="o", expected_output="e", deadline_at=looser)
    evt = ctx.event_queue.get_nowait()
    assert evt["task_contract"]["deadline_at"] == parent

    # A model-authored deadline is validated, because both failures are otherwise silent.
    ctx = _scheduling_ctx(tmp_path / "garbage")
    out = _schedule_task(ctx, objective="o", expected_output="e", deadline_at="in 2 hours")
    assert "TOOL_ARG_ERROR" in out and "ISO-8601" in out
    assert ctx.event_queue.empty()

    ctx = _scheduling_ctx(tmp_path / "past")
    out = _schedule_task(ctx, objective="o", expected_output="e", deadline_at=stamp(-1))
    assert "TOOL_ARG_ERROR" in out and "already in the past" in out
    assert ctx.event_queue.empty()

def test_the_envelope_states_the_request_until_dispatch_fills_it_in(tmp_path, monkeypatch):
    """The envelope is the subagent's public description, and until the child is
    dispatched the honest description has an intent and NO answer. `effective_lane`
    used to default to `light`, so a queued child's envelope named a lane, a slot and
    a strength that no resolution had produced — a claim, not a record."""
    from ouroboros.agent import resolve_dispatch_axes
    from ouroboros.tools.control import _schedule_task

    ctx = _scheduling_ctx(tmp_path / "asked")
    _schedule_task(ctx, objective="o", expected_output="e", executor="harness")
    envelope = ctx.event_queue.get_nowait()["subagent_envelope"]
    assert envelope["executor"] == "harness"          # the request
    assert envelope["effective_lane"] == ""           # nothing resolved yet
    assert envelope["reasoning_effort"] == ""
    assert envelope["effective_executor"] == ""
    assert envelope["capability_delta"] == {}

    task = _enqueue_through_supervisor(tmp_path / "ran", monkeypatch, executor="harness")
    resolve_dispatch_axes(task)
    filled = task["subagent_envelope"]
    assert filled["effective_lane"] == "main"
    assert filled["model"] and filled["reasoning_effort"]
    # The pin no route can honor is a typed block, never a silent re-route to paid
    # native execution (D28).
    assert filled["effective_executor"] == "blocked"
    assert filled["tool_profile"] == "local_readonly_subagent"
    assert filled["capability_delta"]["reduced"] is True

def test_the_scheduling_intent_survives_a_queue_snapshot(tmp_path, monkeypatch):
    """A pending child that waits through a restart must come back holding what its
    parent asked for, INCLUDING the parent's own lane: an omitted lane inherits it and
    only the parent knew it, so a resumed child without it would resolve `auto`
    against the lane of record and silently come back weaker. The intent lives at the
    task TOP LEVEL because that is where the resolution reads it — restoring only the
    copies nested in `metadata` would leave a resumed child resolving from nothing."""
    import supervisor.queue as q

    task = _enqueue_through_supervisor(
        tmp_path, monkeypatch, parent_lane="heavy", executor="harness")

    import json as _json

    captured = {}
    monkeypatch.setattr(q, "atomic_write_text",
                        lambda path, text: captured.update(_json.loads(text)))
    monkeypatch.setattr(q, "PENDING", [task], raising=False)
    monkeypatch.setattr(q, "RUNNING", {}, raising=False)
    assert q.persist_queue_snapshot(reason="test") is True

    rows = captured.get("pending") or []
    assert rows, captured
    restored = rows[0]["task"]
    assert restored["requested_executor"] == "harness"
    assert restored["parent_model_lane"] == "heavy"

def test_a_dispatched_childs_delta_survives_a_restart(tmp_path, monkeypatch):
    """The other half: once a child HAS been dispatched, its resolution must not be
    re-derived by a replay. A RUNNING row that came back through a snapshot without
    the delta would leave the child believing its pin had been honored.

    This pins the SERIALIZATION half only (the snapshot's field list) by injecting
    an already-resolved task into RUNNING; how the resolution REACHES the
    supervisor's RUNNING copy across the process boundary is pinned by
    test_the_workers_resolution_crosses_the_process_boundary_to_the_snapshot —
    without that merge, this test alone passed while real snapshots stayed
    unresolved (XG-2R.1)."""
    import supervisor.queue as q

    from ouroboros.agent import resolve_dispatch_axes

    task = _enqueue_through_supervisor(tmp_path, monkeypatch, executor="harness")
    resolve_dispatch_axes(task)

    import json as _json

    captured = {}
    monkeypatch.setattr(q, "atomic_write_text",
                        lambda path, text: captured.update(_json.loads(text)))
    monkeypatch.setattr(q, "PENDING", [], raising=False)
    monkeypatch.setattr(q, "RUNNING", {"c1": {"task": task, "worker_id": 0,
                                              "started_at": 0.0, "attempt": 1}}, raising=False)
    assert q.persist_queue_snapshot(reason="test") is True

    restored = (captured.get("running") or [])[0]["task"]
    assert restored["effective_executor"] == "blocked"
    assert restored["capability_delta"]["reduced"] is True
    assert restored["reasoning_effort"] == config.resolve_effort("task")

def test_the_workers_resolution_crosses_the_process_boundary_to_the_snapshot(tmp_path, monkeypatch):
    """XG-2R.1 (three reviewers converged): `resolve_dispatch_axes` stamps the WORKER
    process's clone of the task, `assign_tasks` holds its own `dict(task)` in RUNNING,
    and `persist_queue_snapshot` serializes the supervisor's copy — so without a
    worker->supervisor merge the real snapshot carried the UNRESOLVED intent and a
    restart lost the resolved axes and `capability_delta`.

    This test crosses the REAL seam instead of hand-injecting a resolved task:
    the worker's copy is a serialized clone (as pickling across the process
    boundary makes it), the resolution travels ONLY as the JSON-serializable
    `task_dispatch_resolved` event through the REAL registered handler
    (`dispatch_event`), the handler itself takes the snapshot, and the restored
    row must carry the resolved axes + delta."""
    import json as _json
    import queue as queue_mod

    import supervisor.queue as q
    from supervisor import events as ev_module
    from ouroboros.agent import emit_dispatch_resolution, resolve_dispatch_axes
    from ouroboros.subagents import SUBAGENT_RESOLUTION_FIELDS

    supervisor_task = _enqueue_through_supervisor(tmp_path, monkeypatch, executor="harness")
    # The supervisor's RUNNING copy at assignment — intent only, exactly as
    # assign_tasks stores it BEFORE the worker resolves anything.
    running = {"c1": {"task": dict(supervisor_task), "worker_id": 0,
                      "started_at": 0.0, "attempt": 1}}

    # The worker receives a SERIALIZED CLONE: its mutations cannot alias into the
    # supervisor's dict.
    worker_task = _json.loads(_json.dumps(supervisor_task))
    worker_task["id"] = "c1"
    out_q = queue_mod.Queue()
    dispatch = resolve_dispatch_axes(worker_task)
    # The merge set is pinned to the one writer: record_fields() + the envelope.
    assert set(SUBAGENT_RESOLUTION_FIELDS) == set(dispatch.record_fields()) | {"subagent_envelope"}
    emit_dispatch_resolution(out_q, worker_task, dispatch)

    # The masked defect, stated: the supervisor's copy is still unresolved.
    assert "effective_executor" not in running["c1"]["task"]

    captured = {}
    monkeypatch.setattr(q, "atomic_write_text",
                        lambda path, text: captured.update(_json.loads(text)))
    monkeypatch.setattr(q, "PENDING", [], raising=False)
    monkeypatch.setattr(q, "RUNNING", running, raising=False)

    class Ctx:
        RUNNING = running

        @staticmethod
        def persist_queue_snapshot(reason=""):
            return q.persist_queue_snapshot(reason=reason)

    # Only the event crosses — a JSON round-trip proves nothing shared rides along.
    evt = _json.loads(_json.dumps(out_q.get_nowait()))
    assert evt["type"] == "task_dispatch_resolved"
    ev_module.dispatch_event(evt, Ctx())

    # Restart: what a restore reads back is the snapshot the HANDLER persisted.
    rows = captured.get("running") or []
    assert rows and rows[0]["id"] == "c1", captured.get("reason")
    restored = rows[0]["task"]
    assert restored["effective_executor"] == "blocked"
    assert restored["capability_delta"]["reduced"] is True
    assert restored["effective_model_lane"] == "main"
    assert restored["model"]
    assert restored["reasoning_effort"] == config.resolve_effort("task")
    assert restored["subagent_envelope"]["effective_executor"] == "blocked"
    # Intent was merged INTO, not replaced: the request the parent stated survives.
    assert restored["requested_executor"] == "harness"

def test_a_prior_resolutions_residue_is_not_a_legacy_request(tmp_path):
    """Consequence of the resolution surviving the snapshot (XG-2R.1, fable's
    self_consistency half): a crash-requeued child's record now carries
    `reasoning_effort` BECAUSE record_fields() wrote it. Re-dispatching that record
    must not disclose a false 'reasoning_effort=... ignored' legacy note to the
    child prompt and parent readback — LEGACY_SUBAGENT_FIELDS names fields from
    RETIRED SCHEMAS, and a record carrying its own capability_delta proves the
    value is the resolver's residue. A genuinely legacy record (no delta) keeps
    the note."""
    from ouroboros.agent import resolve_dispatch_axes

    task = {"id": "c1", "type": "task", "delegation_role": "subagent"}
    first = resolve_dispatch_axes(task)
    assert first.delta.legacy_note == ""
    assert task["reasoning_effort"]  # the residue the snapshot now preserves

    # The requeue replay: same record, resolution already on it.
    replay = resolve_dispatch_axes(dict(task))
    assert replay.legacy_ignored == {}
    assert replay.delta.legacy_note == ""

    # The genuine legacy case is unchanged: stored effort, no prior resolution.
    legacy = resolve_dispatch_axes({"id": "c2", "type": "task",
                                    "delegation_role": "subagent",
                                    "reasoning_effort": "max"})
    assert "reasoning_effort" in legacy.delta.legacy_note
