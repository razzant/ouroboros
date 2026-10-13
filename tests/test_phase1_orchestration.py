"""Phase 1 golden coverage: honest runtime digest, activity-based timeout model,
no-blind-retry of orchestrators, and the task-tree coordination ledger."""

import json
import time
from types import SimpleNamespace

import pytest


def _patch_queue(queue_module, workers_module, monkeypatch, tmp_path, workers):
    monkeypatch.setattr(queue_module, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue_module, "PENDING", [])
    monkeypatch.setattr(queue_module, "RUNNING", {})
    monkeypatch.setattr(queue_module, "FINALIZATION_GRACE_SEC", 0)
    monkeypatch.setattr(queue_module, "QUEUE_MAX_RETRIES", 1)
    monkeypatch.setattr(queue_module, "load_state", lambda: {})
    monkeypatch.setattr(queue_module, "append_jsonl", lambda *a, **k: None)
    monkeypatch.setattr(queue_module, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr(queue_module, "_ensure_reaper_started", lambda: None)
    monkeypatch.setattr(queue_module, "_reap_queue", queue_module._stdqueue.Queue())
    monkeypatch.setattr(workers_module, "WORKERS", workers)
    monkeypatch.setattr(workers_module, "respawn_worker", lambda worker_id: None)


class _FakeProc:
    pid = 0

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


def test_progressing_subtree_keeps_idle_parent_alive(tmp_path, monkeypatch):
    """The core fix: a parent that is idle ITSELF but whose subtree is progressing must
    NOT be killed (no flat wall-clock guillotine on a productively-waiting orchestrator)."""
    from supervisor import queue as q
    from supervisor import workers as w

    workers = {1: SimpleNamespace(busy_task_id="root1", proc=_FakeProc(), reaping=False),
               2: SimpleNamespace(busy_task_id="child1", proc=_FakeProc(), reaping=False)}
    _patch_queue(q, w, monkeypatch, tmp_path, workers)
    now = time.time()
    q.RUNNING["root1"] = {
        "task": {"id": "root1", "type": "task"},
        "started_at": now - 5000, "last_heartbeat_at": now - 5000,  # idle itself
        "worker_id": 1, "attempt": 1,
    }
    q.RUNNING["child1"] = {
        "task": {"id": "child1", "type": "task", "parent_task_id": "root1", "delegation_role": "subagent"},
        "started_at": now - 60, "last_progress_at": now - 2,  # fresh real progress
        "last_heartbeat_at": now - 2, "worker_id": 2, "attempt": 1,
    }

    q.enforce_task_timeouts()

    assert "root1" in q.RUNNING, "idle parent with a progressing subtree must survive"
    assert "child1" in q.RUNNING
    assert q.PENDING == []


def test_idle_parent_with_pending_descendant_survives(tmp_path, monkeypatch):
    """A parent that is idle ITSELF but has a QUEUED (PENDING, worker-saturated) descendant
    must NOT be killed — killing it would orphan the queued subtree before it ever ran."""
    from supervisor import queue as q
    from supervisor import workers as w

    workers = {1: SimpleNamespace(busy_task_id="root1", proc=_FakeProc(), reaping=False)}
    _patch_queue(q, w, monkeypatch, tmp_path, workers)
    monkeypatch.setattr(q, "get_task_idle_timeout_sec", lambda: 1)
    monkeypatch.setattr(q, "get_per_call_timeout_ceiling_sec", lambda: 1)
    monkeypatch.setattr(q, "PENDING", [{"id": "child1", "parent_task_id": "root1", "root_task_id": "root1"}])
    now = time.time()
    q.RUNNING["root1"] = {
        "task": {"id": "root1", "type": "task"},
        "started_at": now - 5000, "last_heartbeat_at": now - 5000,  # idle itself
        "worker_id": 1, "attempt": 1,
    }

    q.enforce_task_timeouts()

    assert "root1" in q.RUNNING, "idle parent with a queued descendant must survive"
    assert q.PENDING and q.PENDING[0]["id"] == "child1"


def test_explicit_deadline_is_hard_even_while_progressing(tmp_path, monkeypatch):
    """Owner decision (Option A): an explicit deadline_at is HARD — honored promptly even
    while the task is actively progressing. Only the removed BLANKET wall-clock was
    activity-gated; a deliberate/caller deadline is not."""
    from supervisor import queue as q
    from supervisor import workers as w
    from ouroboros.task_results import STATUS_FAILED, load_task_result

    workers = {3: SimpleNamespace(busy_task_id="dl1", proc=_FakeProc(), reaping=False)}
    _patch_queue(q, w, monkeypatch, tmp_path, workers)
    monkeypatch.setattr(q, "load_state", lambda: {"owner_chat_id": 7})
    incidents = []
    monkeypatch.setattr(
        "supervisor.task_reaper.send_with_budget",
        lambda *args, **kwargs: incidents.append((args, kwargs)),
    )
    now = time.time()
    q.RUNNING["dl1"] = {
        "task": {"id": "dl1", "type": "task", "deadline_at": "2000-01-01T00:00:00Z"},
        "started_at": now - 30, "last_progress_at": now - 1,  # actively progressing
        "worker_id": 3, "attempt": 1,
    }

    q.enforce_task_timeouts()
    while not q._reap_queue.empty():
        q._reap_timed_out_task(q._reap_queue.get_nowait())

    assert "dl1" not in q.RUNNING, "a past-deadline task must be stopped even while progressing"
    assert q.PENDING == []  # deadline => no retry
    res = load_task_result(tmp_path, "dl1")
    assert res["status"] == STATUS_FAILED
    assert res["reason_code"] == "deadline"
    assert len(incidents) == 1
    args, kwargs = incidents[0]
    assert args[0] == 7
    assert kwargs["is_progress"] is True
    assert kwargs["task_id"] == "dl1"
    assert kwargs["progress_meta"]["task_incident"] == "task_reaper_stopped"
    assert kwargs["progress_meta"]["toast_once"].startswith("dl1:deadline:")


def test_has_live_descendant_detects_orchestrator(tmp_path, monkeypatch):
    from supervisor import queue as q
    monkeypatch.setattr(q, "RUNNING", {
        "root": {"task": {"id": "root"}},
        "c": {"task": {"id": "c", "parent_task_id": "root"}},
        "gc": {"task": {"id": "gc", "parent_task_id": "c"}},
        "other": {"task": {"id": "other"}},
    })
    monkeypatch.setattr(q, "PENDING", [])
    assert q._has_live_descendant("root") is True   # via c and gc
    assert q._has_live_descendant("c") is True       # via gc
    assert q._has_live_descendant("gc") is False     # leaf
    assert q._has_live_descendant("other") is False

    # A parent whose only child is still QUEUED (PENDING, not yet assigned) is still an
    # orchestrator and must not be blind-retried.
    monkeypatch.setattr(q, "RUNNING", {"p": {"task": {"id": "p"}}})
    monkeypatch.setattr(q, "PENDING", [{"id": "pc", "parent_task_id": "p", "root_task_id": "p"}])
    assert q._has_live_descendant("p") is True
    assert q._has_live_descendant("nope") is False


def test_descendant_detection_survives_missing_intermediate_parent(tmp_path, monkeypatch):
    """A grandchild whose intermediate parent already left RUNNING is still a descendant of
    the root (via root_task_id), so the root orchestrator is recognised (not blind-retried)
    and a progressing grandchild keeps it alive."""
    from supervisor import queue as q
    now = time.time()
    monkeypatch.setattr(q, "RUNNING", {
        "root": {"task": {"id": "root"}},
        # intermediate parent 'c' is GONE from RUNNING; grandchild remains (root_task_id=root)
        "gc": {"task": {"id": "gc", "parent_task_id": "c", "root_task_id": "root"},
               "last_progress_at": now - 1, "started_at": now - 30},
    })
    assert q._has_live_descendant("root") is True
    assert q._subtree_progressing("root", now, 100.0) is True
    # a stale grandchild does NOT keep the root alive
    q.RUNNING["gc"]["last_progress_at"] = now - 10_000
    assert q._subtree_progressing("root", now, 100.0) is False


def test_capability_and_queue_digest_in_runtime_context(tmp_path, monkeypatch):
    """The honesty SSOT: the live capability gate is surfaced into context each turn."""
    import ouroboros.context as ctx_mod

    monkeypatch.setattr(ctx_mod, "get_git_info", lambda *a, **k: ("ouroboros", "abc1234"), raising=False)
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "true")
    env = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path,
                          drive_path=lambda p: tmp_path / p)
    section = ctx_mod.build_runtime_section(env, {"id": "t1", "type": "task"})
    assert '"allow_mutative_subagents": true' in section
    assert "MASTER gate" in section  # the honest note teaching forward reasoning

    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "false")
    section_off = ctx_mod.build_runtime_section(env, {"id": "t1", "type": "task"})
    assert '"allow_mutative_subagents": false' in section_off


def test_tree_tools_in_core_and_initial_schemas():
    """tree_note/tree_read must be in the round-one envelope so a NORMAL parent can publish
    the shared frame before fan-out (no enable_tools detour)."""
    import pathlib
    import tempfile

    from ouroboros.tool_capabilities import CORE_TOOL_NAMES
    from ouroboros.tool_policy import initial_tool_schemas
    from ouroboros.tools.registry import ToolRegistry

    assert "tree_note" in CORE_TOOL_NAMES and "tree_read" in CORE_TOOL_NAMES
    with tempfile.TemporaryDirectory() as d:
        reg = ToolRegistry(repo_dir=pathlib.Path(d), drive_root=pathlib.Path(d))
        names = {s["function"]["name"] for s in initial_tool_schemas(reg)}
        assert "tree_note" in names and "tree_read" in names


def test_tree_tools_available_to_subagents_but_writes_stay_blocked():
    """Both-paths isolation contract: a subagent (read-only AND acting) CAN coordinate via
    the task-tree ledger (tree_note/tree_read), while the repo/data/runtime write-escalation
    surface stays blocked. The capability sets are the SSOT the registry filter reads."""
    from ouroboros.tool_capabilities import (
        ACTING_SUBAGENT_TOOL_NAMES,
        LOCAL_READONLY_SUBAGENT_TOOL_NAMES,
    )

    for name in ("tree_note", "tree_read"):
        assert name in LOCAL_READONLY_SUBAGENT_TOOL_NAMES
        assert name in ACTING_SUBAGENT_TOOL_NAMES
    # a read-only subagent that can tree_note must STILL NOT escalate to real writes
    for blocked in ("write_file", "edit_text", "run_command", "commit_reviewed", "update_identity"):
        assert blocked not in LOCAL_READONLY_SUBAGENT_TOOL_NAMES


def test_tree_ledger_scope_and_attention(monkeypatch, tmp_path):
    # Point the ledger at tmp_path WITHOUT a global config reload (which would leave
    # ouroboros.config.DATA_DIR stuck on this tmp_path and pollute later tests): the ledger
    # reads its module-level DATA_DIR, so monkeypatch THAT (auto-restored after the test).
    import ouroboros.task_tree_ledger as L
    monkeypatch.setattr(L, "DATA_DIR", str(tmp_path))

    assert L.tree_ledger_append("rootA", "contract", "API: f()->g", task_id="rootA", role="lead").startswith("OK")
    assert L.tree_ledger_append("rootA", "question", "lib X or Y?", task_id="c1", role="scout").startswith("OK")
    # blocker/question imply parent attention
    att = L.tree_ledger_attention_after("rootA", "")
    assert len(att) == 1 and att[0]["kind"] == "question"
    # bad kind rejected; empty text rejected
    assert "TOOL_ARG_ERROR" in L.tree_ledger_append("rootA", "bogus", "x")
    assert "TOOL_ARG_ERROR" in L.tree_ledger_append("rootA", "note", "")
    # scope isolation: a different tree has its own ledger
    assert L.tree_ledger_rows("rootB") == []
    # interface_contract is a beacon that ALSO flags parent attention (the shared seam must change)
    assert L.tree_ledger_append("rootA", "interface_contract", "f() now returns h, not g",
                                task_id="c1", role="scout").startswith("OK")
    att2 = L.tree_ledger_attention_after("rootA", "")
    assert any(a["kind"] == "interface_contract" for a in att2), \
        "an interface_contract beacon must surface as a parent early-return"
    # strict root_id validation: a malformed scope is rejected on write; reads soft-fail to []
    assert "TOOL_ARG_ERROR" in L.tree_ledger_append("bad/scope", "note", "x")
    assert "TOOL_ARG_ERROR" in L.tree_ledger_append("", "note", "x")
    assert L.tree_ledger_rows("bad/scope") == []
    digest = L.tree_ledger_tail_digest("rootA")
    assert "contract" in digest and "needs_parent_attention" in digest


def test_review_requested_preserves_typed_concerns_and_exact_hash(monkeypatch, tmp_path):
    import ouroboros.task_tree_ledger as L
    from ouroboros.tools.control import _wait_attention_poll

    monkeypatch.setattr(L, "DATA_DIR", str(tmp_path))
    payload = {"evidence_ref": "artifact:research-notes", "evidence_sha256": "a" * 64}
    first = L.tree_ledger_append(
        "rootA", "review_requested", "Challenge the source coverage.",
        task_id="childA", role="researcher", payload=payload,
    )
    assert first.startswith("OK:")
    second = L.tree_ledger_append(
        "rootA", "review_requested", "Same bytes, different security concern.",
        task_id="childB", role="critic", payload={**payload, "evidence_ref": "artifact:copy"},
    )
    assert second.startswith("OK:")
    rows = [row for row in L.tree_ledger_rows("rootA") if row["kind"] == "review_requested"]
    assert len(rows) == 2
    assert all(row["needs_parent_attention"] is True for row in rows)
    assert rows[0]["payload"] == payload
    # Recovery/context projections retain the exact 64-character binding, not
    # merely a prefix that cannot identify the evidence after wake acknowledgement.
    assert "a" * 64 in L.tree_ledger_page("rootA", data_root=tmp_path)
    assert "a" * 64 in L.tree_ledger_tail_digest("rootA", data_root=tmp_path)

    # Ordinary sliced waits are branch-local: waiting for childA must not surface
    # childB's request merely because both share the same tree ledger.
    ctx = SimpleNamespace(task_id="parentA", drive_root=tmp_path, task_metadata={"root_task_id": "rootA"})
    early = _wait_attention_poll(ctx, "", ["childA"])({}, {})
    assert early is not None
    assert [row["task_id"] for row in early["beacons"]] == ["childA"]
    assert "TOOL_ARG_ERROR" in L.tree_ledger_append(
        "rootA", "review_requested", "missing hash", task_id="childA",
        payload={"evidence_ref": "artifact:x"},
    )


def test_wait_attention_cursor_delivers_preexisting_same_timestamp_rows_fifo(
    monkeypatch, tmp_path,
):
    import ouroboros.task_tree_ledger as L
    from ouroboros.tools.control import _wait_attention_poll

    monkeypatch.setattr(L, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(L, "utc_now_iso", lambda: "2026-08-24T12:00:00Z")
    for index in range(7):
        assert L.tree_ledger_append(
            "rootA", "question", f"question-{index}", task_id="childA",
        ).startswith("OK:")

    # All rows predate entry into the wait.  The response stays bounded, but
    # advances only across rows it actually delivered, including a timestamp
    # shared by more rows than the five-item response bound.
    ctx = SimpleNamespace(task_id="parentA", drive_root=tmp_path, task_metadata={"root_task_id": "rootA"})
    first = _wait_attention_poll(ctx, "", ["childA"])({}, {})
    assert [row["text"] for row in first["beacons"]] == [
        f"question-{index}" for index in range(5)
    ]
    assert first["beacons_remaining"] == 2

    second = _wait_attention_poll(ctx, "", ["childA"])({}, {})
    assert [row["text"] for row in second["beacons"]] == ["question-5", "question-6"]
    assert second["beacons_remaining"] == 0
    assert _wait_attention_poll(ctx, "", ["childA"])({}, {}) is None


def test_delegation_constraint_payload_and_override(monkeypatch, tmp_path):
    import ouroboros.task_tree_ledger as L

    monkeypatch.setattr(L, "DATA_DIR", str(tmp_path))
    result = L.tree_ledger_append(
        "rootA",
        "delegation_constraint",
        "stop fanning out until git evidence is gathered",
        task_id="scout1",
        role="scout",
        payload={"directive": "halt_fanout", "scope": {"role": "release-band"}, "rationale": "readonly child cannot run git"},
    )
    assert result.startswith("OK")
    open_rows = L.open_delegation_constraints("rootA")
    assert len(open_rows) == 1
    cid = open_rows[0]["payload"]["constraint_id"]

    L.tree_ledger_append(
        "rootA",
        "decision",
        "override after parent gathered git evidence",
        task_id="parent",
        role="lead",
        payload={"constraint_id": cid, "decision": "overridden", "reason": "parent owns raw git data"},
        allow_constraint_override=True,
    )
    assert L.open_delegation_constraints("rootA") == []
    L.tree_ledger_append(
        "rootA",
        "delegation_constraint",
        "same id raised later should be open again",
        task_id="scout1",
        role="scout",
        payload={"constraint_id": cid, "directive": "halt_fanout", "scope": {}, "rationale": "new evidence"},
    )
    assert len(L.open_delegation_constraints("rootA")) == 1
    assert "TOOL_ARG_ERROR" in L.tree_ledger_append(
        "rootA", "delegation_constraint", "bad", payload={"directive": "unknown"}
    )
    assert "TOOL_ARG_ERROR" in L.tree_ledger_append(
        "rootA",
        "decision",
        "forged override",
        payload={"constraint_id": "forged", "decision": "overridden", "reason": "no"},
    )


def test_effective_delegation_budget_honors_require_lane_and_scope():
    from ouroboros.tools.control_delegation import effective_delegation_budget

    row = {
        "payload": {
            "constraint_id": "c1",
            "directive": "require_lane",
            "scope": {"role": "critic", "lane": "heavy"},
            "rationale": "needs stronger coding lane",
        }
    }
    ignored = effective_delegation_budget({}, unresolved_constraints=[row], role="researcher", requested_lane="light")
    assert ignored.ok is True

    # The INTENDED lane decides, not the literal request (v6.87.28: this gate runs at
    # ADMISSION, before the child is dispatched, so no effective lane exists yet —
    # `subagents.intended_lane` is what a request MEANS, and its one owner).
    requested_does_not_count = effective_delegation_budget(
        {},
        unresolved_constraints=[row],
        role="critic",
        requested_lane="auto",
        intended_lane="light",
    )
    assert requested_does_not_count.ok is False

    blocked = effective_delegation_budget({}, unresolved_constraints=[row], role="critic", requested_lane="light", intended_lane="light")
    assert blocked.ok is False
    assert blocked.reason_code == "delegation_constraint_require_lane"

    allowed = effective_delegation_budget({}, unresolved_constraints=[row], role="critic", requested_lane="heavy", intended_lane="heavy")
    assert allowed.ok is True
    # F9 (sol #1): a PASSING admission under an applicable require_lane carries
    # the verified lane out, so the scheduler can stamp it onto the child record
    # and the dispatch-time policy default cannot override it.
    assert allowed.required_lane == "heavy"
    # A non-applicable constraint (role mismatch) contributes no required lane.
    assert ignored.required_lane == ""


def test_reaper_finalizes_stuck_artifact_on_self_finalized_result(tmp_path, monkeypatch):
    """Round-10 crit#2: a worker that self-finalized a workspace child but died before the
    parent ran artifact finalization leaves artifact_status stuck at 'finalizing'. The reaper
    terminalized the task (it is no longer in RUNNING), so the normal task_done finalize path
    finds nothing — the reaper must complete it. It must rescue ONLY a stuck non-terminal
    artifact state (re-finalizing a terminal result could regress it to FAILED) and skip
    readonly subagents (no durable artifacts), via the shared task_is_readonly_subagent gate."""
    from supervisor import queue as q
    from supervisor import workers as w
    from ouroboros import headless
    from ouroboros.task_results import write_task_result

    workers = {4: SimpleNamespace(busy_task_id=None, proc=_FakeProc(), reaping=True)}
    _patch_queue(q, w, monkeypatch, tmp_path, workers)
    monkeypatch.setattr(q, "_kept_service_pids", lambda: set(), raising=False)
    # A prior server lifespan can legitimately close the process-global bus.
    # This fixture owns its publication collector, not a new supervisor bus.
    monkeypatch.setattr(w, "_EVENT_Q_SHUTDOWN", True)
    with pytest.raises(RuntimeError, match="supervisor event bus is shutting down"):
        w.get_event_q()
    emitted = []
    monkeypatch.setattr(w, "get_event_q", lambda: SimpleNamespace(put=emitted.append))

    calls = []
    def finalize(root, task):
        calls.append(str(task["id"]))
        write_task_result(root, task["id"], "completed", artifact_status="ready",
                          artifact_finalized_at="fixture")
        return []
    monkeypatch.setattr(headless, "finalize_task_artifacts", finalize)

    def _run(task, artifact_status):
        # Pre-write the worker's own terminal result so the reaper's post-kill re-check honors
        # it (self_status set) instead of clobbering it — the branch crit#2 lives in.
        write_task_result(tmp_path, str(task["id"]), "completed", artifact_status=artifact_status)
        task["workspace_root"] = str(tmp_path / "workspace")
        q._reap_timed_out_task({"worker_id": 4, "proc": workers[4].proc, "worker": workers[4],
                                "drive_root": str(tmp_path), "task_id": task["id"],
                                "task": task, "task_type": "task",
                                "terminal_reason": "idle_timeout", "attempt": 1})

    # 1. stuck 'finalizing' → reaper completes finalization
    _run({"id": "wt1", "type": "task"}, "finalizing")
    assert calls == ["wt1"], "reaper must finalize a self-finalized result stuck at 'finalizing'"

    # 2. already-terminal artifact_status → NOT re-finalized (no regression)
    calls.clear()
    _run({"id": "wt2", "type": "task"}, "ready")
    assert calls == [], "an already-terminal artifact result must not be re-finalized"

    # 3. readonly subagent → skipped even when stuck (no durable owner-facing artifacts)
    calls.clear()
    _run({"id": "wt3", "type": "task", "delegation_role": "subagent",
          "task_constraint": {"mode": "local_readonly_subagent"}}, "finalizing")
    assert calls == [], "a readonly subagent has no durable artifacts to finalize"
    terminals = [event for event in emitted if event.get("type") == "task_done"]
    assert [event["task_id"] for event in terminals] == ["wt1", "wt2", "wt3"]
    assert all(event["status"] == "completed" and event["chat_id"] == 0 for event in terminals)
    assert all(event["_files_prepared_attempt"] == 1 and event["worker_id"] == 4 for event in terminals)


def test_task_is_readonly_subagent_gate():
    """The single SSOT gate the task_done path and the reaper both read (a re-derivation
    drift of this rule is what stranded the reaper's artifact finalization)."""
    from ouroboros.headless import task_is_readonly_subagent

    assert task_is_readonly_subagent(
        {"delegation_role": "subagent", "task_constraint": {"mode": "local_readonly_subagent"}}) is True
    # constraint nested under metadata is honored too
    assert task_is_readonly_subagent(
        {"delegation_role": "subagent", "metadata": {"task_constraint": {"mode": "local_readonly_subagent"}}}) is True
    # acting subagents and plain tasks are NOT readonly → they DO finalize artifacts
    assert task_is_readonly_subagent(
        {"delegation_role": "subagent", "task_constraint": {"mode": "acting_subagent"}}) is False
    assert task_is_readonly_subagent({"id": "root", "type": "task"}) is False
    assert task_is_readonly_subagent(None) is False


def test_reaper_fails_closed_when_worker_not_confirmed_dead(tmp_path, monkeypatch):
    """Variant-A STRICT fail-closed: if the worker is NOT provably dead after kill/join, the reaper
    does NOTHING downstream — no terminal write, no task_done, no retry, no respawn — and HOLDS the
    slot reaping=True so a still-live orphan can never be reused, raced, or have its result clobbered.
    The task stays RUNNING (custody-reaped on the next generation). Guards the codex/cumulative blocker."""
    from supervisor import queue as q
    from supervisor import workers as w
    from ouroboros import platform_layer
    from ouroboros.task_results import STATUS_RUNNING, load_task_result, write_task_result

    class _AliveProc:
        pid = 4242

        def is_alive(self):
            return True  # never confirms dead, even after kill attempts

        def join(self, timeout=None):
            return None

    slot = SimpleNamespace(busy_task_id=None, proc=_AliveProc(), reaping=True)
    _patch_queue(q, w, monkeypatch, tmp_path, {5: slot})
    monkeypatch.setattr(q, "_kept_service_pids", lambda: set(), raising=False)
    monkeypatch.setattr(platform_layer, "kill_pid_tree", lambda *a, **k: None)  # kill is a no-op
    respawns: list = []
    monkeypatch.setattr(w, "respawn_worker", lambda wid: respawns.append(wid))
    emitted: list = []
    monkeypatch.setattr(w, "get_event_q",
                        lambda: SimpleNamespace(put=lambda evt: emitted.append(evt)), raising=False)
    incidents = []
    monkeypatch.setattr(
        "supervisor.task_reaper.send_with_budget",
        lambda *args, **kwargs: incidents.append((args, kwargs)),
    )

    # The task is RUNNING on disk before the reaper runs; the strict stop must NOT terminalize it.
    write_task_result(tmp_path, "wedged1", STATUS_RUNNING, result="in progress")

    q._reap_timed_out_task({
        "worker_id": 5, "proc": slot.proc, "worker": slot, "drive_root": str(tmp_path), "task_id": "wedged1",
        "task": {"id": "wedged1", "type": "task", "chat_id": 7}, "task_type": "task",
        "terminal_reason": "idle_timeout", "attempt": 1,
        "owner_chat_id": 7,
        "will_retry": True, "retry_task_id": "wedged1",
    })

    assert q.PENDING == [], "no colliding retry may be enqueued for an unconfirmed-dead worker"
    assert respawns == [], "the slot must NOT be respawned while the orphan may be alive"
    assert slot.reaping is True, "the slot stays reaping (unavailable) so the live orphan is never reused"
    assert not any(e.get("type") == "task_done" for e in emitted), \
        "no task_done may be emitted while the worker may be alive"
    res = load_task_result(tmp_path, "wedged1")
    assert res and res.get("status") == STATUS_RUNNING, \
        "the task must remain RUNNING (custody-reaped next generation), never terminalized"
    sup_log = tmp_path / "logs" / "supervisor.jsonl"
    assert sup_log.exists(), "a wedged-worker escalation event must be recorded"
    events = [json.loads(line) for line in sup_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(e.get("type") == "task_reaper_wedged" and e.get("task_id") == "wedged1" for e in events), \
        "the strict stop must emit a task_reaper_wedged escalation event"
    assert len(incidents) == 1
    args, kwargs = incidents[0]
    assert args[0] == 7
    assert kwargs["is_progress"] is True
    assert kwargs["task_id"] == "wedged1"
    assert kwargs["progress_meta"] == {
        "task_incident": "task_reaper_wedged",
        "toast_once": "wedged1:task_reaper_wedged:5:idle_timeout",
    }
