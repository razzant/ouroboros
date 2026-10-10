"""Continuation of a settled delegated run: ``delegate_start(continue_from=<run_id>)``.

Four floor invariants gate it (own task line, settled, no pending-ambiguous
apply, authority does not widen); every other fact — any settled cause, an
unread result, another actor, a changed or partial work order — is ADVICE in
the continuing child's prompt and a fact in the parent's started payload. The
engine continues the stopped work (``continueFrom``); a writing continuation
runs in the predecessor's undisposed snapshot and supersedes its capture, so
one cumulative patch is integrated once (closes #1489).
"""

from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import pytest

from ouroboros import delegate_continuation as continuation, delegate_custody as custody
from ouroboros.delegate_registration_policy import (
    CAP_BASIS_DEADLINE_DERIVED,
    CAP_BASIS_LIFETIME_DERIVED,
    CAP_BASIS_OPERATION_WINDOW,
    CAP_BASIS_REQUESTED,
    CAP_BASIS_REQUESTED_CLAMPED_DEADLINE,
    CAP_BASIS_REQUESTED_CLAMPED_LIFETIME,
    CAP_BASIS_REQUESTED_CLAMPED_SCHEMA,
    FINITE_LEAF_CAP_BASES,
)
from tests._delegated_transport_shared import (  # noqa: F401 -- autouse transport fixture
    _delegating_ctx,
    _nanny_ctx,
    _owned_gateway_uses_each_test_transport,
)

ROUTE = "some-route"
ACTOR = "actor-1"
CFG = "cfg-fingerprint-1"
WORK_ORDER = "work-order-fingerprint-1"
LIMIT = "subscription_window_exhausted"


def _seed(tmp_path, run_id, *, task_id="t-a", root_task_id=None, state="failed", reason=LIMIT, settled=True,
          actor=ACTOR, route=ROUTE, access="readonly", mode="ask", isolation="", snapshot_id="",
          target_root="", execution_root="", continuation_of="", config_fingerprint=CFG,
          work_order_fingerprint=WORK_ORDER, output="consumed", source="", **extra):
    """Durable rows for one prior run; the memo is cleared so lookups REPLAY them."""
    entry = custody.RunCustody(
        run_id=run_id, task_id=task_id, route_id=route, model="m", selected_subagent_id=actor,
        snapshot_id=snapshot_id, target_root=target_root, execution_root=execution_root,
        baseline_sha="base-1" if snapshot_id else "", root_task_id=task_id if root_task_id is None else root_task_id,
        continuation_of=continuation_of, config_fingerprint=config_fingerprint,
        work_order_fingerprint=work_order_fingerprint, source=source, **extra,
    )
    assert custody.record_started(tmp_path, entry, shape={
        "access": access, "mode": mode, "isolation": isolation, "delegated": bool(isolation),
        "root": "/r", "max_seconds": 90, "max_seconds_basis": CAP_BASIS_REQUESTED})
    if settled:
        row = {"run_id": run_id, "task_id": task_id, "route": route, "state": state}
        if reason is not None:
            row["outcome_reason"] = reason
        assert custody.emit(tmp_path, custody.SETTLED, row)
    if output in ("staged", "consumed"):
        assert custody.emit(tmp_path, custody.OUTPUT_SPILLED, {
            "run_id": run_id, "task_id": task_id, "artifact": f"delegated_runs/{run_id}.json",
            "sha256": f"sha-{run_id}", "bytes": 3, "staged": True, "full_content": True})
    if output == "consumed":
        assert custody.emit(tmp_path, custody.OUTPUT_CONSUMED, {
            "run_id": run_id, "task_id": task_id, "sha256": f"sha-{run_id}"})
    custody._CUSTODY.clear()
    return entry


def _gate(tmp_path, run_id, *, task_id="t-a", actor=ACTOR, route=ROUTE, access="readonly", mode="ask",
          isolation="", target_root="", config_fingerprint=CFG, canonical=""):
    ctx = SimpleNamespace(task_id=task_id)
    return continuation.bind_continuation(
        ctx, tmp_path, run_id, actor={"selected_subagent_id": actor, "config_fingerprint": config_fingerprint},
        route=SimpleNamespace(route_id=route),
        authority=SimpleNamespace(access=access, mode=mode, isolation=isolation),
        target_root=target_root, canonical_work_order_fingerprint=canonical)


def _write_result(tmp_path, task_id, **row):
    directory = tmp_path / "task_results"
    directory.mkdir(exist_ok=True)
    (directory / f"{task_id}.json").write_text(
        json.dumps({"_schema_version": 1, "task_id": task_id, "status": "cancelled", **row}), encoding="utf-8")


def _continued_by(predecessor, successor, *, tamper=False):
    from ouroboros.owner_continue import binding_sha

    binding = {"verb": "continue", "predecessor_task_id": predecessor,
               "successor_task_id": "someone-else" if tamper else successor, "action_nonce": "nonce-1234"}
    return {"successor_task_id": successor, "action_nonce": "nonce-1234", "binding": binding,
            "binding_sha256": binding_sha(binding), "state": "admitted"}


# --------------------------------------------------------------------------- custody facts

def test_settlement_rows_replay_the_typed_cause_and_the_continuation_lineage(tmp_path):
    _seed(tmp_path, "run-a", continuation_of="run-0")
    replayed = custody.replay(tmp_path)["run-a"]
    assert replayed.settled and replayed.terminal_state == "failed"
    assert replayed.terminal_reason == LIMIT and replayed.continuation_of == "run-0"
    _seed(tmp_path, "run-legacy", reason=None)
    assert custody.replay(tmp_path)["run-legacy"].terminal_reason == ""


def test_settle_run_records_the_engines_typed_outcome_reason(tmp_path, monkeypatch):
    import ouroboros.usage_accounting as accounting

    monkeypatch.setattr(accounting, "record_subscription_session", lambda *_a, **_k: None)
    custody._CUSTODY.pop("run-s", None)
    row = custody.RunCustody(run_id="run-s", task_id="t-a", route_id=ROUTE, model="m")
    assert custody.record_started(tmp_path, row)
    assert custody.settle_run(tmp_path, None, row, {"summary": {
        "state": "cancelled", "spendUsd": 0, "spendEstimated": False,
        "outcomeFacts": {"lifecycle": "cancelled", "reason": "wall_clock_exceeded"},
    }})["settled"]
    settled = [r for r in custody.custody_rows(tmp_path) if r.get("type") == custody.SETTLED and r.get("run_id") == "run-s"]
    assert settled[-1]["outcome_reason"] == "wall_clock_exceeded" and settled[-1]["state"] == "cancelled"
    custody._CUSTODY.clear()
    assert custody.replay(tmp_path)["run-s"].terminal_reason == "wall_clock_exceeded"


# --------------------------------------------------------------------------- the four floors

def test_any_settled_cause_is_admitted_and_copied_into_the_facts(tmp_path):
    """Limit, crash, host restart, cancel, wall clock, a question, no reason at all (#1489)."""
    for state, reason in (("failed", LIMIT), ("failed", "credential_pool_exhausted"),
                          ("interrupted", "crash_interrupted"), ("cancelled", "host_cancelled"),
                          ("cancelled", "owner_task_gone"), ("cancelled", "user_cancelled"),
                          ("cancelled", "wall_clock_exceeded"), ("succeeded", "input_required"),
                          ("failed", None)):
        rid = f"run-{state}-{reason}"
        _seed(tmp_path, rid, state=state, reason=reason)
        facts, code, _detail = _gate(tmp_path, rid)
        assert code == "", (rid, code)
        assert facts["cause"] == (reason or state) and facts["prior_terminal_state"] == state
        assert facts["task_line"] == "own" and facts["prior_patch_disposition"] == "not_applicable"
        assert "state_transfer" not in facts


def test_floor_own_task_line(tmp_path):
    assert _gate(tmp_path, "run-none")[1] == continuation.REFUSAL_SOURCE_UNKNOWN
    _seed(tmp_path, "run-theirs", task_id="t-other")
    assert _gate(tmp_path, "run-theirs")[1] == continuation.REFUSAL_SOURCE_NOT_OWNED
    # A review panel's run is never a task's delegation line, even on the same task id.
    _seed(tmp_path, "run-review", source="review_substrate_triad")
    facts, code, detail = _gate(tmp_path, "run-review")
    assert code == continuation.REFUSAL_SOURCE_NOT_OWNED and "review panel" in detail


def test_floor_settled_and_single_successor(tmp_path):
    _seed(tmp_path, "run-live", settled=False)
    facts, code, detail = _gate(tmp_path, "run-live")
    assert code == continuation.REFUSAL_SOURCE_NOT_TERMINAL and "second writer" in detail
    # Closed as absent: settled custody without a terminal is not a run that can continue.
    _seed(tmp_path, "run-absent", settled=False)
    assert custody.emit(tmp_path, custody.CLOSED_ABSENT, {"run_id": "run-absent", "task_id": "t-a"})
    custody._CUSTODY.clear()
    assert _gate(tmp_path, "run-absent")[1] == continuation.REFUSAL_SOURCE_NOT_TERMINAL
    # A run another run already continues in its snapshot names that head.
    _seed(tmp_path, "run-pred", access="workspace_write", mode="agent", isolation="live",
          snapshot_id="snap-1", target_root="/t", execution_root="/snap")
    _seed(tmp_path, "run-succ", settled=False, access="workspace_write", mode="agent", isolation="live",
          snapshot_id="snap-1", target_root="/t", execution_root="/snap", continuation_of="run-pred",
          capture_id="inv-succ", snapshot_task_id="t-a")
    facts, code, detail = _gate(tmp_path, "run-pred", access="workspace_write", mode="agent", isolation="live",
                                target_root="/t")
    assert code == continuation.REFUSAL_SUPERSEDED and "run-succ" in detail


def test_floor_no_pending_ambiguous_apply(tmp_path):
    _seed(tmp_path, "run-snap", snapshot_id="snap-1", access="workspace_write", mode="agent", isolation="live",
          target_root="/target", execution_root="/snap")
    kwargs = dict(access="workspace_write", mode="agent", isolation="live", target_root="/target")
    facts, code, _d = _gate(tmp_path, "run-snap", **kwargs)
    assert code == "" and facts["prior_patch_disposition"] == "undisposed"   # same tree: no disposition needed
    assert custody.emit(tmp_path, custody.PATCH_APPLY_STARTED, {"run_id": "run-snap", "task_id": "t-a",
                                                                "snapshot_id": "snap-1", "apply_idempotency_key": "k"})
    custody._CUSTODY.clear()
    assert _gate(tmp_path, "run-snap", **kwargs)[1] == continuation.REFUSAL_APPLY_AMBIGUOUS
    assert custody.emit(tmp_path, custody.PATCH_DISPOSED, {"run_id": "run-snap", "task_id": "t-a",
                                                           "snapshot_id": "snap-1", "disposition": "rejected"})
    custody._CUSTODY.clear()
    facts, code, _d = _gate(tmp_path, "run-snap", **kwargs)
    assert code == "" and facts["prior_patch_disposition"] == "rejected"


def test_floor_authority_does_not_widen(tmp_path):
    _seed(tmp_path, "run-read")
    widened = _gate(tmp_path, "run-read", access="workspace_write", mode="agent", isolation="live", target_root="/t")
    assert widened[1] == continuation.REFUSAL_AUTHORITY_WIDENED
    _seed(tmp_path, "run-write", access="workspace_write", mode="agent", isolation="live", target_root="/t")
    assert _gate(tmp_path, "run-write", access="workspace_write", mode="agent", isolation="live",
                 target_root="/elsewhere")[1] == continuation.REFUSAL_TARGET_MISMATCH
    # Lowering is not widening: a read-only continuation of a writer is admitted.
    facts, code, _d = _gate(tmp_path, "run-write")
    assert code == "" and facts["prior_access"] == "workspace_write"
    # An unrecorded prior shape cannot be proven narrower: it refuses.
    _seed(tmp_path, "run-noshape", access="", mode="")
    assert _gate(tmp_path, "run-noshape")[1] == continuation.REFUSAL_AUTHORITY_WIDENED


def test_former_refusals_are_advice_not_gates(tmp_path):
    """Unread result, another executor, a changed or partial work order: admitted, disclosed."""
    _seed(tmp_path, "run-unread", output="staged")
    facts, code, _d = _gate(tmp_path, "run-unread")
    assert code == "" and facts["advice"] == ["result_unread"]
    _seed(tmp_path, "run-actor", actor="actor-a")
    facts, code, _d = _gate(tmp_path, "run-actor", actor="actor-b")
    assert code == "" and "executor_changed" in facts["advice"] and facts["prior_actor"] == "actor-a"
    _seed(tmp_path, "run-route", route="other-route")
    assert "executor_changed" in _gate(tmp_path, "run-route")[0]["advice"]
    _seed(tmp_path, "run-wo")
    assert _gate(tmp_path, "run-wo", canonical="a-different-brief")[0]["advice"] == ["work_order_changed"]
    _seed(tmp_path, "run-partial", work_order_coverage="partial",
          work_order_source_request={"complete_chars": 100, "coverage": "partial"})
    assert "work_order_partial" in _gate(tmp_path, "run-partial")[0]["advice"]
    # The cap basis no longer matters: any cap's expiry continues.
    for basis in (CAP_BASIS_DEADLINE_DERIVED, CAP_BASIS_OPERATION_WINDOW):
        assert basis not in FINITE_LEAF_CAP_BASES
    clean = _gate(tmp_path, "run-wo", canonical=WORK_ORDER)[0]
    assert clean["advice"] == []


def test_owner_continue_root_continues_its_predecessors_runs_through_the_recorded_binding(tmp_path):
    _seed(tmp_path, "run-root", task_id="t-pred")
    _seed(tmp_path, "run-child", task_id="t-child", root_task_id="t-pred")
    _write_result(tmp_path, "t-pred", continued_by=_continued_by("t-pred", "t-cont"))
    for rid in ("run-root", "run-child"):
        facts, code, _d = _gate(tmp_path, rid, task_id="t-cont")
        assert code == "" and facts["task_line"] == "owner_continue" and facts["prior_owner_task_id"]
    # Same room or folder grants nothing: another root is not the recorded successor.
    assert _gate(tmp_path, "run-root", task_id="t-stranger")[1] == continuation.REFUSAL_SOURCE_NOT_OWNED
    # A chain of Continues is followed claim by claim.
    _write_result(tmp_path, "t-cont", continued_by=_continued_by("t-cont", "t-cont2"))
    assert _gate(tmp_path, "run-root", task_id="t-cont2")[0]["task_line"] == "owner_continue"
    # A binding whose hash or successor does not verify is no binding.
    _write_result(tmp_path, "t-pred", continued_by=_continued_by("t-pred", "t-cont", tamper=True))
    assert _gate(tmp_path, "run-root", task_id="t-cont")[1] == continuation.REFUSAL_SOURCE_NOT_OWNED


def test_confirmed_retry_successor_holds_the_task_line(tmp_path, monkeypatch):
    import ouroboros.delegate_shared as shared

    _seed(tmp_path, "run-old", task_id="t-a")
    monkeypatch.setattr(shared, "retry_result_status",
                        lambda ctx, drive, rid, state=None: (custody.OWNED, state[rid], "t-a")
                        if ctx.task_id == "t-retry" else (custody.FOREIGN, state[rid], ""))
    facts, code, _d = _gate(tmp_path, "run-old", task_id="t-retry")
    assert code == "" and facts["task_line"] == "retry_successor"
    assert _gate(tmp_path, "run-old", task_id="t-unrelated")[1] == continuation.REFUSAL_SOURCE_NOT_OWNED


# --------------------------------------------------------------------------- the child's prompt

def test_prompt_states_tree_facts_and_advice_and_no_session_claim():
    base = {"continuation_of": "run-x", "prior_terminal_state": "failed", "prior_terminal_reason": LIMIT}
    same = continuation.continuation_prompt({**base, "workspace": "same_snapshot"}, "finish the tests")
    assert "CONTINUATION OF RUN run-x" in same and LIMIT in same and "SAME private snapshot" in same
    assert same.endswith("finish the tests")
    assert "NOTHING of its session state" not in same and "session" not in same.split("\n\n")[0].lower()
    applied = continuation.continuation_prompt({**base, "workspace": "fresh",
                                                "prior_patch_disposition": "applied"}, "")
    assert "APPLIED" in applied and "do not redo or re-apply" in applied
    rejected = continuation.continuation_prompt({**base, "prior_patch_disposition": "rejected"}, "")
    assert "REJECTED" in rejected
    direct = continuation.continuation_prompt({**base, "prior_access": "workspace_write",
                                               "prior_patch_disposition": "not_applicable"}, "")
    assert "DIRECTLY" in direct and "read-only" not in direct
    advised = continuation.continuation_prompt({**base, "advice": ["result_unread", "executor_changed"]}, "")
    assert "never read" in advised and "different actor" in advised


# --------------------------------------------------------------------------- delegate_start wiring

def _start(tmp_path, monkeypatch, *, acting, run_id, prompt="finish the remaining work", start_kwargs=None,
           capabilities=("continueFrom",), engine_version="3.22.0", start_error=None, task_id=None, access=None):
    """Run _delegate_start against a stub engine; returns (last request, payload, ctx, calls)."""
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    calls = []

    class _Stub:
        def handshake(self, **_kw): return {}
        def agent_capabilities(self, **_kw):
            return {"runControlKeys": list(capabilities), "harnesses": [{
                "id": "some-route", "enabled": True, "status": "ok",
                "accessProfilesSupported": ["readonly", "workspace_write"]}]}
        def quota_snapshots(self): return []
        def find_project_id(self, root): return "prj-existing"
        def register_project(self, root): return "prj-new"
        def remove_project(self, project_id): return {}
        def start_run(self, request, *, idempotency_key=""):
            calls.append(request)
            if start_error is not None:
                raise start_error
            return {"runId": run_id, "runDir": f"/tmp/{run_id}"}
        def close(self): pass

    _Stub.engine_version = engine_version
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(tmp_path / "snap_root"))
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    delegate._CUSTODY.clear()
    ctx = _delegating_ctx(tmp_path, acting=acting, task_id=task_id or f"t-nanny-{'write' if acting else 'read'}")
    if access is None:
        result = delegate._delegate_start(ctx, prompt, **(start_kwargs or {}))
    else:
        from ouroboros import subagent_runtime, subagents
        from tests._delegated_transport_shared import _transport_snapshot

        monkeypatch.setattr(delegate, "prepare_delegate_start_actor", subagent_runtime.prepare_delegate_start_actor)
        result = subagent_runtime.exact_start(ctx, prompt, {
            "snapshot": _transport_snapshot(subagents.get_subagent_harness()),
            "access": access, **(start_kwargs or {})})
    payload = json.loads(result.text)
    delegate._CUSTODY.clear()
    return (calls[-1] if calls else None), payload, ctx, calls


def _settle(tmp_path, run_id, task_id, *, state="failed", reason=LIMIT):
    assert custody.emit(tmp_path, custody.SETTLED, {"run_id": run_id, "task_id": task_id, "route": ROUTE,
                                                    "state": state, "outcome_reason": reason})
    custody._CUSTODY.clear()


def test_continue_from_puts_the_engine_key_in_the_body_and_the_facts_in_the_prompt(tmp_path, monkeypatch):
    _first, _payload, _ctx, _calls = _start(tmp_path, monkeypatch, acting=False, run_id="run-prev",
                                            prompt="summarize the repository")
    _settle(tmp_path, "run-prev", "t-nanny-read")
    request, payload, _ctx, _calls = _start(tmp_path, monkeypatch, acting=False, run_id="run-next",
                                            prompt="also list the tests",
                                            start_kwargs={"continue_from": "run-prev"})
    assert request["continueFrom"] == "run-prev" and "continueCarrier" not in request
    assert request["prompt"].startswith("HOST FACTS FOR THIS CONTINUATION OF RUN run-prev")
    assert request["prompt"].endswith("also list the tests") and LIMIT in request["prompt"]
    # Facts never ride the instructions a resumed vendor session may keep from its first request.
    assert "HOST FACTS" not in request["instructions"] and "CONTINUATION" not in request["instructions"]
    assert payload["status"] == "started" and payload["continuation"]["continuation_of"] == "run-prev"
    assert payload["continuation"]["cause"] == LIMIT and payload["continuation"]["workspace"] == "fresh"
    assert custody.replay(tmp_path)["run-next"].continuation_of == "run-prev"
    # The carrier preference is the mind's: packet forces a re-brief.
    request, _p, _c, _calls = _start(tmp_path, monkeypatch, acting=False, run_id="run-third", prompt="",
                                     start_kwargs={"continue_from": "run-next", "continue_carrier": "packet"})
    assert request is None  # run-next is not settled yet: a second writer is refused
    _settle(tmp_path, "run-next", "t-nanny-read")
    request, payload, _c, _calls = _start(tmp_path, monkeypatch, acting=False, run_id="run-third", prompt="",
                                          start_kwargs={"continue_from": "run-next", "continue_carrier": "packet"})
    assert request["continueCarrier"] == "packet" and payload["continuation"]["carrier_preference"] == "packet"
    # An empty caller text is allowed for a continuation: the host facts are the prompt.
    assert request["prompt"].startswith("HOST FACTS") and not request["prompt"].endswith("\n")


def test_an_engine_without_continue_from_refuses_typed_and_starts_nothing(tmp_path, monkeypatch):
    _start(tmp_path, monkeypatch, acting=False, run_id="run-prev")
    _settle(tmp_path, "run-prev", "t-nanny-read")
    request, payload, _ctx, calls = _start(tmp_path, monkeypatch, acting=False, run_id="run-next",
                                           capabilities=(), start_kwargs={"continue_from": "run-prev"})
    assert request is None and calls == []
    assert payload["reason"] == continuation.REFUSAL_ENGINE_UNSUPPORTED and payload["definitely_unrun"] is True
    assert "plain new run" in payload["detail"]


def test_argument_refusals_precede_the_daemon(tmp_path):
    from ouroboros.delegate_shared import delegate_payload
    from ouroboros.tools import delegate

    ctx = _nanny_ctx(tmp_path)
    clash = delegate_payload(delegate._delegate_start(ctx, "finish it", continue_from="run-x", retry_of="tok"))
    assert clash["reason"] == "continuation_selector_conflict" and clash["definitely_unrun"] is True
    payload_run = delegate_payload(delegate._delegate_start(
        ctx, "finish it", continue_from="run-x", root="skill_payload", bucket="external", skill_name="s"))
    assert payload_run["reason"] == "continuation_resource_conflict"
    assert delegate_payload(delegate._delegate_start(ctx, "   "))["reason"] == "empty_prompt"
    for kwargs in ({"continue_carrier": "packet"}, {"continue_from": "run-x", "continue_carrier": "native"}):
        assert delegate_payload(delegate._delegate_start(ctx, "go", **kwargs))["reason"] == "continuation_carrier_invalid"


def _snapshot_count():
    from ouroboros import subagent_worktrees

    return sum(1 for row in subagent_worktrees._load_registry(None, strict=True, op="test")
               if row.get("kind") == subagent_worktrees._KIND_DELEGATED_EXEC)


def test_readonly_continuation_starts_while_the_writers_patch_awaits_disposition(tmp_path, monkeypatch):
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive

    first, _p, ctx, _calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-writer")
    snapshot_root = pathlib.Path(first["execution"]["workspaceRoot"])
    (snapshot_root / "waiting.txt").write_text("pending disposition\n", encoding="utf-8")
    _settle(tmp_path, "run-writer", ctx.task_id)
    capture = capture_terminal_patch_for_drive(tmp_path, custody.replay(tmp_path)["run-writer"])
    assert capture["status"] == "ready_with_changes"
    patch = pathlib.Path(capture["patch_artifact"])
    original = patch.read_bytes()
    count = _snapshot_count()

    request, payload, _ctx, calls = _start(
        tmp_path, monkeypatch, acting=True, run_id="run-reader", access="readonly",
        start_kwargs={"continue_from": "run-writer"})
    assert payload["status"] == "started", payload
    assert len(calls) == 1 and request["access"] == "readonly" and request["mode"] == "ask"
    assert request["continueFrom"] == "run-writer" and "execution" not in request
    assert "wait for your supervisor's decision" in request["prompt"]
    pred, succ = (custody.replay(tmp_path)[rid] for rid in ("run-writer", "run-reader"))
    assert succ.continuation_of == pred.run_id and not succ.capture_id and not succ.snapshot_id
    assert pred.patch_captured and not pred.patch_disposed and not pred.superseded_by
    assert [row.run_id for row in custody.undisposed_patches(tmp_path)] == [pred.run_id]
    assert _snapshot_count() == count and patch.read_bytes() == original
    assert (snapshot_root / "waiting.txt").read_text(encoding="utf-8") == "pending disposition\n"


def test_a_writing_continuation_runs_in_the_predecessors_snapshot_and_supersedes_its_capture(tmp_path, monkeypatch):
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive
    from ouroboros.tools.subagent_integration_delegated import _integrate_delegated_patch

    before = _snapshot_count()
    first, _p, _ctx, _calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-first")
    snapshot_root = first["execution"]["workspaceRoot"]
    assert _snapshot_count() == before + 1
    _settle(tmp_path, "run-first", "t-nanny-write")
    pred = custody.replay(tmp_path)["run-first"]
    pathlib.Path(snapshot_root, "first.txt").write_text("from the first run\n", encoding="utf-8")

    request, payload, ctx, _calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-next",
                                           start_kwargs={"continue_from": "run-first"})
    # Same tree: the same execution root and baseline, no second snapshot provisioned.
    assert request["continueFrom"] == "run-first" and request["execution"]["workspaceRoot"] == snapshot_root
    assert _snapshot_count() == before + 1 and payload["continuation"]["workspace"] == "same_snapshot"
    assert payload["snapshot"] == {"reused_from_run": "run-first"} and payload["baseline_id"] == pred.baseline_sha
    assert "SAME private snapshot" in request["prompt"]
    state = custody.replay(tmp_path)
    succ, pred = state["run-next"], state["run-first"]
    assert succ.snapshot_id == pred.snapshot_id and succ.capture_id == succ.invocation_id != pred.snapshot_id
    assert succ.snapshot_task_id == "t-nanny-write" and succ.continuation_of == "run-first"
    # The hand-over: the predecessor's capture is superseded, its obligation moved to the successor.
    assert pred.patch_disposed == custody.SUPERSEDED and pred.superseded_by == "run-next"
    assert "run-first" not in [r.run_id for r in custody.undisposed_patches(tmp_path)]
    assert pred.snapshot_id in custody.open_snapshot_ids(tmp_path)
    # Each capture keeps its own identity; ONE disposition lock serves the snapshot.
    assert custody.capture_key(pred) == pred.snapshot_id and custody.capture_key(succ) == succ.invocation_id
    assert custody.disposition_lock_path(tmp_path, pred) == custody.disposition_lock_path(tmp_path, succ)
    # A superseded predecessor can neither be applied nor rejected, and its terminal says where the work went.
    from ouroboros.subagents import DelegatedRunShape
    from ouroboros.tools.delegate_terminal_evidence import _delivered_terminal_payload

    shape = DelegatedRunShape(access="workspace_write", mode="agent", isolation="live", delegated=True)
    assert _delivered_terminal_payload(ctx, "run-first", {"summary": {"state": "failed"}}, shape,
                                       pred)["superseded_by"] == "run-next"
    refusal = _integrate_delegated_patch(ctx, run_id="run-first", decision="reject")
    assert "INTEGRATE_DELEGATED_SUPERSEDED" in refusal and "run-next" in refusal
    assert pathlib.Path(snapshot_root).exists()
    # One cumulative patch, captured under the successor's own identity.
    pathlib.Path(snapshot_root, "second.txt").write_text("from the continuation\n", encoding="utf-8")
    _settle(tmp_path, "run-next", "t-nanny-write")
    block = capture_terminal_patch_for_drive(tmp_path, custody.replay(tmp_path)["run-next"])
    assert block["status"] == "ready_with_changes", block
    assert f"delegated_runs/{succ.invocation_id}/" in block["manifest_read"]["path"]
    patch = pathlib.Path(block["patch_artifact"]).read_text(encoding="utf-8") if block["patch_artifact"] else ""
    manifest = json.loads(pathlib.Path(block["manifest_artifact"]).read_text(encoding="utf-8"))
    touched = set(manifest.get("tracked_changed") or []) | set(manifest.get("untracked_included") or [])
    assert {"first.txt", "second.txt"} <= touched or ("first.txt" in patch and "second.txt" in patch)


@pytest.mark.parametrize("decision", ["apply", "reject"])
def test_pending_continuation_recovery_keeps_lineage_and_refuses_predecessor_disposition(
        tmp_path, monkeypatch, decision):
    from ouroboros.delegate_custody_reconcile import _recover_pending_invocation
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive
    from ouroboros.tools.subagent_integration_delegated import _integrate_delegated_patch

    first, _p, ctx, _calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-first")
    snapshot_root = pathlib.Path(first["execution"]["workspaceRoot"])
    (snapshot_root / "first.txt").write_text("retained work\n", encoding="utf-8")
    _settle(tmp_path, "run-first", ctx.task_id)
    assert capture_terminal_patch_for_drive(tmp_path, custody.replay(tmp_path)["run-first"])[
        "status"] == "ready_with_changes"
    lost = ClaudexorUnavailable("daemon_unreachable", "connection reset", status_code=503)
    request, unknown, _ctx, _calls = _start(
        tmp_path, monkeypatch, acting=True, run_id="run-next", start_error=lost,
        start_kwargs={"continue_from": "run-first"})
    invocation = unknown["pending_invocation_id"]
    pending, = custody.pending_invocations(tmp_path)
    lineage = {"continuation_of": "run-first", "capture_id": invocation, "snapshot_task_id": ctx.task_id}
    assert {key: pending[key] for key in lineage} == lineage

    class RecoveryGateway:
        def start_run(self, body, *, idempotency_key):
            assert body == request and idempotency_key == invocation
            return {"runId": "run-next"}

        def get_run(self, run_id):
            assert run_id == "run-next"
            return {"summary": {"state": "running"}}

    custody._CUSTODY.clear()  # Recovery has only the durable pending invocation.
    recovered = _recover_pending_invocation(tmp_path, RecoveryGateway(), pending)
    assert recovered["action"] == "left_live"
    assert custody.pending_invocations(tmp_path) == []
    refusal = _integrate_delegated_patch(ctx, run_id="run-first", decision=decision)
    assert "INTEGRATE_DELEGATED_SUPERSEDED" in refusal and "run-next" in refusal
    assert (snapshot_root / "first.txt").read_text(encoding="utf-8") == "retained work\n"
    started, = [row for row in custody.custody_rows(tmp_path)
                if row.get("type") == custody.STARTED and row.get("run_id") == "run-next"]
    assert {key: started[key] for key in lineage} == lineage
    custody._CUSTODY.clear()
    state = custody.replay(tmp_path)
    pred, succ = state["run-first"], state["run-next"]
    assert pred.patch_disposed == custody.SUPERSEDED and pred.superseded_by == succ.run_id
    assert not succ.settled and succ.execution_root == str(snapshot_root)
    assert custody.capture_key(pred) != custody.capture_key(succ) == invocation
    assert custody.disposition_lock_path(tmp_path, pred) == custody.disposition_lock_path(tmp_path, succ)
    assert succ.snapshot_id in custody.open_snapshot_ids(tmp_path)


def test_a_definite_engine_refusal_leaves_the_predecessor_and_its_snapshot_intact(tmp_path, monkeypatch):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    first, _p, _ctx, _calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-first")
    _settle(tmp_path, "run-first", "t-nanny-write")
    before = _snapshot_count()
    refusal = ClaudexorUnavailable("continuation_superseded", "already continued", status_code=409)
    request, payload, _ctx, calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-next",
                                           start_error=refusal, start_kwargs={"continue_from": "run-first"})
    assert len(calls) == 1 and payload["reason"] == "continuation_superseded"
    pred = custody.replay(tmp_path)["run-first"]
    assert pred.patch_disposed == "" and not pred.superseded_by  # never bound: the hand-over never happened
    from ouroboros.subagent_worktrees import find_execution_snapshot

    assert _snapshot_count() == before and find_execution_snapshot(pred.snapshot_id) is not None
    assert pathlib.Path(first["execution"]["workspaceRoot"]).exists()


def test_a_pending_hand_over_blocks_dispositions_and_a_second_claim(tmp_path):
    from ouroboros.delegate_start_claims import claimed_start_request

    _seed(tmp_path, "run-pred", access="workspace_write", mode="agent", isolation="live", snapshot_id="snap-1",
          target_root="/t", execution_root="/snap")
    row = {"invocation_id": "inv-1", "task_id": "t-a", "continuation_of": "run-pred", "capture_id": "inv-1",
           "snapshot_id": "snap-1", "request": {"prompt": "x"}}
    assert custody.emit(tmp_path, custody.START_REQUESTED, row)
    custody._CUSTODY.clear()
    pred = custody.replay(tmp_path)["run-pred"]
    assert "CONTINUATION_PENDING" in continuation.disposition_refusal(tmp_path, pred)
    claim, refusal = claimed_start_request(
        tmp_path, claim_target="", payload_busy=lambda *_a: "", task_id="t-a", invocation_id="inv-2",
        continuation_of="run-pred", capture_id="inv-2", snapshot_id="snap-1")
    assert claim is False and refusal["reason"] == continuation.REFUSAL_SUPERSEDED
    assert refusal["pending_invocation_id"] == "inv-1"
    # After a disposition the snapshot is released: a hand-over claim refuses, typed.
    assert custody.emit(tmp_path, custody.START_FAILED, {"run_id": "", "invocation_id": "inv-1", "definite": True})
    assert custody.emit(tmp_path, custody.PATCH_DISPOSED, {"run_id": "run-pred", "task_id": "t-a",
                                                           "snapshot_id": "snap-1", "disposition": "rejected"})
    custody._CUSTODY.clear()
    claim, refusal = claimed_start_request(
        tmp_path, claim_target="", payload_busy=lambda *_a: "", task_id="t-a", invocation_id="inv-3",
        continuation_of="run-pred", capture_id="inv-3", snapshot_id="snap-1")
    assert claim is False and refusal["reason"] == continuation.REFUSAL_SNAPSHOT_RELEASED


def test_a_configured_session_continues_with_its_note_never_the_canonical_work_order(tmp_path, monkeypatch):
    """The old session (or the engine's evidence packet) already holds the work order."""
    import ouroboros.tools.delegate as delegate

    _start(tmp_path, monkeypatch, acting=False, run_id="run-prev")
    _settle(tmp_path, "run-prev", "t-nanny-read")
    bound = delegate.prepare_delegate_start_actor

    def compiled(ctx, drive, **kwargs):
        actor, refusal = bound(ctx, drive, **kwargs)
        return ({**actor, "compiled_work_order": True} if actor else actor), refusal

    monkeypatch.setattr(delegate, "prepare_delegate_start_actor", compiled)
    request, payload, _ctx, _calls = _start(
        tmp_path, monkeypatch, acting=False, run_id="run-next", prompt="THE CANONICAL WORK ORDER",
        start_kwargs={"continue_from": "run-prev", "_coordination_context": "the coordination note"})
    assert payload["status"] == "started"
    assert request["prompt"].endswith("the coordination note") and "CANONICAL" not in request["prompt"]
    assert "the coordination note" not in request["instructions"]


@pytest.mark.parametrize("prompt", ["add tests", ""])
def test_a_continuation_retried_after_an_unknown_outcome_replays_its_lineage(tmp_path, monkeypatch, prompt):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    _start(tmp_path, monkeypatch, acting=True, run_id="run-first")
    _settle(tmp_path, "run-first", "t-nanny-write")
    lost = ClaudexorUnavailable("daemon_unreachable", "connection reset", status_code=503)
    original, unknown, _ctx, _calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-next", prompt=prompt,
                                       start_error=lost, start_kwargs={"continue_from": "run-first"})
    token = unknown["pending_invocation_id"]
    assert token and custody.replay(tmp_path)["run-first"].patch_disposed == ""  # not bound yet: not superseded
    pred = custody.replay(tmp_path)["run-first"]
    assert "CONTINUATION_PENDING" in continuation.disposition_refusal(tmp_path, pred)
    # A different text is not a retry of that invocation.
    _r, other, _ctx, calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-x", prompt="something else",
                                    start_kwargs={"retry_of": token})
    assert other["reason"] == "retry_prompt_mismatch" and calls == []
    # The retry names the caller's own text; the recorded body (host facts included) is replayed.
    request, payload, _ctx, _calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-next", prompt=prompt,
                                            start_kwargs={"retry_of": token})
    assert payload["status"] == "started" and request["continueFrom"] == "run-first"
    assert request == original
    assert request["prompt"].startswith("HOST FACTS") and request["prompt"].endswith(prompt)
    state = custody.replay(tmp_path)
    assert state["run-next"].capture_id == token and state["run-next"].continuation_of == "run-first"
    assert state["run-first"].superseded_by == "run-next"


def test_empty_retry_of_an_ordinary_start_still_requires_the_recorded_prompt(tmp_path, monkeypatch):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    lost = ClaudexorUnavailable("daemon_unreachable", "connection reset", status_code=503)
    _r, unknown, _ctx, _calls = _start(
        tmp_path, monkeypatch, acting=False, run_id="run-first", prompt="original work", start_error=lost)
    invocation = unknown["pending_invocation_id"]
    request, refusal, _ctx, calls = _start(
        tmp_path, monkeypatch, acting=False, run_id="run-first", prompt="",
        start_kwargs={"retry_of": invocation})
    assert refusal["reason"] == "retry_prompt_mismatch" and request is None and calls == []
    assert [row["invocation_id"] for row in custody.pending_invocations(tmp_path)] == [invocation]


def test_an_unavailable_snapshot_lock_is_a_typed_claim_refusal(tmp_path, monkeypatch):
    import ouroboros.platform_layer as platform_layer
    from ouroboros.delegate_start_claims import claimed_start_request

    _seed(tmp_path, "run-pred", access="workspace_write", mode="agent", isolation="live", snapshot_id="snap-1",
          target_root="/t", execution_root="/snap")

    def broken(*_a, **_k):
        raise OSError("lock directory unwritable")

    monkeypatch.setattr(platform_layer, "acquire_exclusive_file_lock", broken)
    claim, refusal = claimed_start_request(
        tmp_path, claim_target="", payload_busy=lambda *_a: "", task_id="t-a", invocation_id="inv-1",
        continuation_of="run-pred", capture_id="inv-1", snapshot_id="snap-1")
    assert claim is False and refusal["reason"] == "continuation_handover_busy"
    assert "lock directory unwritable" in refusal["detail"]
    assert continuation.ENGINE_CONTINUE_KEY == "continueFrom"


def test_a_retried_continuation_replays_its_lineage(tmp_path):
    row = {"invocation_id": "inv-r", "task_id": "t-a", "continuation_of": "run-pred", "capture_id": "inv-r",
           "snapshot_task_id": "t-a", "snapshot_id": "snap-1", "request": {"prompt": "x"}}
    assert custody.emit(tmp_path, custody.START_REQUESTED, row)
    record = custody.invocation_record(tmp_path, "inv-r")
    replayed = continuation.replayed_custody(record)
    assert replayed.custody == {"continuation_of": "run-pred", "capture_id": "inv-r", "snapshot_task_id": "t-a"}
    assert replayed.snapshot is None and replayed.request == {} and replayed.prompt == ""


# --------------------------------------------------------------------------- terminal facts

def test_the_terminal_payload_carries_resumable_and_one_line_per_continued_try():
    from ouroboros.subagents import DelegatedRunShape
    from ouroboros.tools.delegate import _terminal_payload

    resumable = {"cause": "pool_exhausted", "resetsAt": "2026-10-06T21:00:00Z", "limitWindow": "five_hour",
                 "limitEvidence": "window", "carriers": ["native", "packet"], "limitCode": "credential_pool_exhausted",
                 "session": None, "workspace": {"kind": "in_place", "root": "/w"}}
    receipts = [
        {"tryIndex": 2, "attemptId": "a01", "carrier": "native_moved", "cause": "vendor_limit",
         "from": {"runId": "run-1", "attemptId": "a01", "profileId": "proton0"}, "to": {"profileId": "proton00"},
         "workspace": "same_root", "memory": "full", "instructions": "as_sent", "reingestedTokens": 41000,
         "observedModel": "claude-fable-5-1", "modelMismatch": False, "identityCheck": "matched_before_effects",
         "inputDelivery": "confirmed"},
        {"tryIndex": 1, "attemptId": "a01", "carrier": "packet", "cause": "transport",
         "from": {"runId": "run-0", "attemptId": "a01", "profileId": None}, "to": {"profileId": None},
         "workspace": "different_root", "memory": "partial", "instructions": "vendor_snapshot",
         "observedModel": None, "modelMismatch": True, "identityCheck": "mismatch_after_possible_effects",
         "inputDelivery": "uncertain"},
    ]
    shape = DelegatedRunShape(access="readonly", mode="ask", isolation="", delegated=False)
    payload = _terminal_payload("run-1", {"summary": {"state": "failed", "resumable": resumable,
                                                      "continuity": receipts}}, shape)
    assert payload["resumable"] == resumable
    native, packet = payload["continuity"]
    assert native.startswith("try 2 (a01): native_moved after vendor_limit;")
    assert "profile proton0 -> proton00" in native and "memory full" in native
    assert "model claude-fable-5-1" in native and "re-read 41000 tokens" in native and "MISMATCH" not in native
    assert "of run run-0" in packet and "model unattested" in packet and "MODEL MISMATCH" in packet
    for flag in ("different root", "identity mismatch_after_possible_effects",
                 "last input may not have been delivered", "instructions: vendor snapshot"):
        assert flag in packet
    # An older engine reports neither; the payload then carries neither.
    older = _terminal_payload("run-1", {"summary": {"state": "failed"}}, shape)
    assert "resumable" not in older and "continuity" not in older


# --------------------------------------------------------------------------- unchanged neighbours

@pytest.mark.parametrize("chapter,marker,codes", [
    ("development/06-rules-by-change-class.md", "- `continue_from`", (
        continuation.REFUSAL_SNAPSHOT_MISSING, continuation.REFUSAL_SNAPSHOT_RELEASED,
        "continuation_handover_busy")),
    ("architecture/06-agent-core.md", "**Continuation**", (continuation.REFUSAL_SNAPSHOT_MISSING,)),
])
def test_continuation_docs_name_the_snapshot_custody_refusals(chapter, marker, codes):
    text = (pathlib.Path(__file__).resolve().parents[1] / "docs" / chapter).read_text(encoding="utf-8")
    paragraph = text[text.index(marker):].split("\n\n", 1)[0]
    assert all(f"`{code}`" in paragraph for code in codes)
    assert "refuses ONLY on the four floors" not in paragraph


def test_requested_cap_is_clamped_by_the_deadline_and_the_remaining_lifetime(tmp_path, monkeypatch):
    """The cap basis is still RECORDED beside the number (a fact for the parent),
    even though no continuation depends on it any more."""
    import time
    from datetime import datetime, timedelta, timezone

    from ouroboros import config
    from ouroboros.tools.delegate import bounded_max_seconds

    def _ctx(*, deadline_in=None, started_ago=100.0):
        meta = {}
        if deadline_in is not None:
            meta["deadline_at"] = (datetime.now(timezone.utc) + timedelta(seconds=deadline_in)).isoformat()
        return SimpleNamespace(task_id="t-a", task_metadata=meta, task_started_at=time.time() - started_ago,
                               _budget_paused_sec=0.0, budget_pause_resume=None)

    monkeypatch.setattr(config, "get_task_abs_ceiling_sec", lambda: None)
    plain = bounded_max_seconds(_ctx(), 120)
    assert (plain.seconds, plain.basis) == (120, CAP_BASIS_REQUESTED)
    assert bounded_max_seconds(_ctx(), None).basis == CAP_BASIS_OPERATION_WINDOW
    monkeypatch.setattr(config, "get_task_abs_ceiling_sec", lambda: 300.0)
    clamped = bounded_max_seconds(_ctx(), 1000)
    assert clamped.basis == CAP_BASIS_REQUESTED_CLAMPED_LIFETIME and 150 <= clamped.seconds <= 200
    near = bounded_max_seconds(_ctx(deadline_in=50), 1000)
    assert near.basis == CAP_BASIS_REQUESTED_CLAMPED_DEADLINE and 40 <= near.seconds <= 50
    assert bounded_max_seconds(_ctx(), None).basis == CAP_BASIS_LIFETIME_DERIVED
    monkeypatch.setattr(config, "get_task_abs_ceiling_sec", lambda: 10.0)
    spent = bounded_max_seconds(_ctx(started_ago=100.0), 60)
    assert spent.refusal_code == "task_lifetime_exhausted" and spent.seconds == 0
    assert FINITE_LEAF_CAP_BASES == frozenset({CAP_BASIS_REQUESTED, CAP_BASIS_REQUESTED_CLAMPED_SCHEMA})


def test_no_resume_causes_are_untouched_by_the_continuation_seam():
    """Worker-crash recovery stays exact and cause-specific: a continuation is a NEW
    start the mind chooses, never an automatic resurrection."""
    from ouroboros.delegate_recovery import NO_RESUME_CAUSES

    assert NO_RESUME_CAUSES == (
        "owner_restart", "panic", "external_signal", "worker_signal",
        "deadline", "timeout", "explicit_cancellation", "abrupt_whole_app_loss",
    )


@pytest.mark.parametrize("cause", ["vendor_limit", "pool_exhausted"])
def test_receipt_lines_tolerate_partial_rows(cause):
    line = continuation._receipt_line("run-1", {"carrier": "native", "cause": cause})
    assert line.startswith("try None (?): native after " + cause) and "model unattested" in line
