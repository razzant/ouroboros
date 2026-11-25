"""Continuation of a finished delegated run: ``delegate_start(continue_from=<run_id>)``.

Four floor invariants gate it (own task line, observed terminal, no pending-ambiguous
apply, authority does not widen); every other fact — any terminal cause, an
unread result, another actor, a changed or partial work order — is ADVICE in
the continuing child's prompt and a fact in the parent's started payload. The
engine continues the stopped work (``continueFrom``); a writing continuation
runs in the predecessor's undisposed snapshot and supersedes its capture, so
one cumulative patch is integrated once (closes #1489). A predecessor whose
checkout was released after its saved capture gets it back from its result
commit at the hand-over (``tests/test_delegated_released_copies.py``).
"""

from __future__ import annotations

import json
import pathlib
import subprocess
from types import SimpleNamespace

import pytest

from ouroboros import delegate_continuation as continuation, delegate_custody as custody
from ouroboros.delegate_registration_policy import (
    CAP_BASIS_DEADLINE_DERIVED,
    CAP_BASIS_LIFETIME_DERIVED,
    CAP_BASIS_REQUESTED,
    CAP_BASIS_REQUESTED_CLAMPED_DEADLINE,
    CAP_BASIS_REQUESTED_CLAMPED_LIFETIME,
    CAP_BASIS_UNBOUNDED,
)
from tests._delegated_transport_shared import (  # noqa: F401 -- autouse transport fixture
    _delegating_ctx,
    _nanny_ctx,
    _owned_gateway_uses_each_test_transport,
    _select_transport_access,
)
from tests._store_shared import seed_result

ROUTE = "some-route"
ACTOR = "actor-1"
CFG = "cfg-fingerprint-1"
WORK_ORDER = "work-order-fingerprint-1"
LIMIT = "subscription_window_exhausted"


def _seed(tmp_path, run_id, *, task_id="t-a", root_task_id=None, state="failed", reason=LIMIT, settled=True,
          actor=ACTOR, route=ROUTE, access="readonly", mode="ask", isolation="", snapshot_id="",
          target_root="", execution_root="", continuation_of="", config_fingerprint=CFG,
          work_order_fingerprint=WORK_ORDER, output="consumed", source="", basis=CAP_BASIS_REQUESTED, **extra):
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
        "root": "/r", "max_seconds": 90, "max_seconds_basis": basis})
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
    seed_result(tmp_path, task_id, {"_schema_version": 1, "task_id": task_id, "status": "cancelled", **row})


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
    _facts, code, detail = _gate(tmp_path, "run-snap", **kwargs)
    assert code == continuation.REFUSAL_APPLY_AMBIGUOUS
    # An ordinary Git run is first offered the call that records a provable apply without changing a file.
    assert detail.index("decision='apply'") < detail.index("APPLY_AMBIGUOUS") < detail.index("acknowledge_ambiguous=true")
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
    # The cap basis no longer matters: any cap's expiry continues, whatever its recorded
    # basis (the retired FINITE_LEAF_CAP_BASES gate had no reader left), an older row's too.
    for basis in (CAP_BASIS_DEADLINE_DERIVED, CAP_BASIS_UNBOUNDED, "operation_window"):
        _seed(tmp_path, f"run-{basis}", basis=basis)
        assert _gate(tmp_path, f"run-{basis}", canonical=WORK_ORDER)[0]["advice"] == []
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
    # An empty writer result closed itself: the successor hears that nothing changed, never "DIRECTLY".
    empty = continuation.continuation_prompt({**base, "prior_access": "workspace_write", "workspace": "fresh",
                                              "prior_patch_disposition": custody.NO_CHANGES}, "")
    assert "changed no files" in empty and "DIRECTLY" not in empty
    # A checkout restored from the saved result says what is not in it.
    restored = continuation.continuation_prompt({**base, "workspace": "same_snapshot", "checkout": "recreated"}, "")
    assert "SAME private snapshot" in restored and "restored from the saved result" in restored
    assert ".review-evidence/ files are not here" in restored and "restored" not in same


# --------------------------------------------------------------------------- delegate_start wiring

def _start(tmp_path, monkeypatch, *, acting, run_id, prompt="finish the remaining work", start_kwargs=None,
           capabilities=("continueFrom",), engine_version="3.22.0", start_error=None, task_id=None, access=None,
           harness_row=None, prepare=None):
    """Run _delegate_start against a stub engine; returns (last request, payload, ctx, calls).
    ``harness_row`` adds fields to the route's catalog row (a negotiated access-control grant);
    ``prepare`` shapes the starter's context before the start (what it holds of my human's words)."""
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    calls = []

    class _Stub:
        def handshake(self, **_kw): return {}
        def agent_capabilities(self, **_kw):
            return {"runControlKeys": list(capabilities), "harnesses": [{
                "id": "some-route", "enabled": True, "status": "ok",
                "accessProfilesSupported": ["readonly", "workspace_write"], **(harness_row or {})}]}
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
    if prepare is not None:
        prepare(ctx)
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
    # A READ-ONLY reading child continues in a fresh workspace: technical access alone
    # decides the run shape now (#1609), so the read-only premise is stated explicitly.
    _select_transport_access(monkeypatch, "readonly")
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


@pytest.mark.parametrize("carrier", ["auto", "packet"])
def test_continuation_delivers_a_new_selection_only_in_the_current_prompt(tmp_path, monkeypatch, carrier):
    from hashlib import sha256

    from ouroboros.knowledge import resolve_knowledge_address

    selection = [{"topic": "handed-note", "scope": "global"}]
    address = resolve_knowledge_address(tmp_path, "handed-note", "global")
    address.path.parent.mkdir(parents=True)
    old, new = "# Old synthetic note\n\n## Details\nOLD_BODY\n", "# Current synthetic note\n\n## Details\nNEW_BODY\n"
    address.path.write_bytes(old.encode("utf-8"))

    def start(run, **options):
        return _start(tmp_path, monkeypatch, acting=False, access="readonly", run_id=run,
                      prepare=lambda ctx: ctx.task_metadata.update(knowledge_notes=selection), start_kwargs=options)[:2]

    first, initial = start("first-note-run", knowledge_notes=selection)
    assert initial["status"] == "started" and first["instructions"].count(old) == 1
    address.path.write_bytes(new.encode("utf-8"))
    _settle(tmp_path, "first-note-run", "t-nanny-read")
    omitted, result = start("second-note-run", continue_from="first-note-run", continue_carrier=carrier)
    assert result["status"] == "started" and "named_notes" not in result
    assert "## Named knowledge notes" not in omitted["prompt"] + omitted["instructions"]

    _settle(tmp_path, "second-note-run", "t-nanny-read")
    selected, result = start("third-note-run", continue_from="second-note-run", continue_carrier=carrier,
                             knowledge_notes=selection)
    assert selected["continueCarrier"] == carrier and selected["prompt"].count(new) == 1
    assert old not in selected["prompt"] and "## Named knowledge notes" not in selected["instructions"]
    assert result["named_notes"] == {"sent": [{**selection[0], "revision": sha256(new.encode()).hexdigest(),
                                                "chars": len(new)}]}


def test_an_engine_without_continue_from_refuses_typed_and_starts_nothing(tmp_path, monkeypatch):
    _select_transport_access(monkeypatch, "readonly")   # a read-only child (#1609)
    _start(tmp_path, monkeypatch, acting=False, run_id="run-prev")
    _settle(tmp_path, "run-prev", "t-nanny-read")
    request, payload, _ctx, calls = _start(tmp_path, monkeypatch, acting=False, run_id="run-next",
                                           capabilities=(), start_kwargs={"continue_from": "run-prev"})
    assert request is None and calls == []
    assert payload["reason"] == continuation.REFUSAL_ENGINE_UNSUPPORTED and payload["definitely_unrun"] is True
    assert "plain new run" in payload["detail"]


@pytest.mark.parametrize("health,asks", [(("", "2099-01-01T00:00:00Z"), True), (("route_disabled", ""), False)])
def test_a_start_refused_on_a_spent_route_asks_the_subscription_census_for_a_read(tmp_path, monkeypatch, health, asks):
    """Route health proves the subscription spent: the typed, definitely unrun refusal asks the census for a read; an
    unrelated blocker refuses the same way and asks nothing."""
    from ouroboros import claudexor_daemon, config, subagents
    from ouroboros import subscription_census as census

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(claudexor_daemon, "owned_daemon_provisioned", lambda: True)
    monkeypatch.setattr(subagents, "route_health", lambda *_a, **_k: health)
    _select_transport_access(monkeypatch, "readonly")
    request, payload, _ctx, calls = _start(tmp_path, monkeypatch, acting=False, run_id="run-spent")
    assert request is None and calls == [] and payload["definitely_unrun"] is True
    assert (census.request_mtime() is not None) is asks


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

    return sum(1 for row in subagent_worktrees._records(op="test")
               if row.get("kind") == subagent_worktrees._KIND_DELEGATED_EXEC)


@pytest.mark.parametrize("auxiliary_supported", [False, True])
def test_readonly_continuation_starts_while_the_writers_patch_awaits_disposition(
        tmp_path, monkeypatch, auxiliary_supported):
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive
    from ouroboros import test_environment

    scratch_parent = tmp_path / "temp"
    scratch_parent.mkdir()
    monkeypatch.setattr(test_environment, "session_scratch_parent", lambda: scratch_parent)
    row = {"writeMechanism": "fs_sandbox", "accessControl": {
        "auxiliaryWritableRoots": auxiliary_supported}}
    first, _p, ctx, _calls = _start(
        tmp_path, monkeypatch, acting=True, run_id="run-writer", harness_row=row)
    assert bool(first["execution"].get("auxiliaryWritableRoots")) is auxiliary_supported
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
        start_kwargs={"continue_from": "run-writer"}, harness_row=row)
    assert payload["status"] == "started", payload
    assert len(calls) == 1 and request["access"] == "readonly" and request["mode"] == "ask"
    assert request["continueFrom"] == "run-writer"
    # A missing execution object inherits the writer's live geometry and cache grant.
    # Lowering must instead read the current source, keeping the unaccepted patch separate.
    assert request["execution"] == {
        "isolation": "envelope", "delegated": False,
        **({"auxiliaryWritableRoots": []} if auxiliary_supported else {}),
    }
    assert request["scope"]["root"] == str(pathlib.Path(ctx.workspace_root).resolve())
    assert request["scope"]["root"] != str(snapshot_root)
    assert "workspaceRoot" not in request["execution"]
    assert "wait for your supervisor's decision" in request["prompt"]
    pred, succ = (custody.replay(tmp_path)[rid] for rid in ("run-writer", "run-reader"))
    assert succ.continuation_of == pred.run_id and not succ.capture_id and not succ.snapshot_id
    assert pred.patch_captured and not pred.patch_disposed and not pred.superseded_by
    assert [row.run_id for row in custody.undisposed_patches(tmp_path)] == [pred.run_id]
    assert _snapshot_count() == count and patch.read_bytes() == original
    # The writer's checkout went with its saved capture; its result stays as commit C.
    from ouroboros.subagent_worktrees import find_execution_snapshot

    record = find_execution_snapshot(pred.snapshot_id)
    assert not snapshot_root.exists() and record["result_sha"] == capture["result_commit"]
    shown = subprocess.run(["git", "show", f"{record['result_sha']}:waiting.txt"], cwd=ctx.workspace_root,
                           check=True, capture_output=True, text=True).stdout
    assert shown == "pending disposition\n"


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
        def operations(self):
            return [{"method": "POST", "path": "/v2/runs/lookup"}]
        def lookup_run(self, body, *, idempotency_key):
            assert body == request and idempotency_key == invocation
            return {"status": "found", "storeId": "fixture-store", "handle": {
                "jobId": "accepted-next", "runId": "run-next", "state": "running"}}
        def start_run(self, *_a, **_kw):
            pytest.fail("an unknown continuation must be read, not automatically re-POSTed")

        def get_run(self, run_id):
            assert run_id == "run-next"
            return {"summary": {"state": "running"}}

    custody._CUSTODY.clear()  # Recovery has only the durable pending invocation.
    recovered = _recover_pending_invocation(tmp_path, RecoveryGateway(), pending, may_read=True)
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
        tmp_path, task_id="t-a", invocation_id="inv-2",
        continuation_of="run-pred", capture_id="inv-2", snapshot_id="snap-1")
    assert claim is False and refusal["reason"] == continuation.REFUSAL_SUPERSEDED
    assert refusal["pending_invocation_id"] == "inv-1"
    # After a disposition the snapshot is released: a hand-over claim refuses, typed.
    assert custody.emit(tmp_path, custody.START_FAILED, {"run_id": "", "invocation_id": "inv-1", "definite": True})
    assert custody.emit(tmp_path, custody.PATCH_DISPOSED, {"run_id": "run-pred", "task_id": "t-a",
                                                           "snapshot_id": "snap-1", "disposition": "rejected"})
    custody._CUSTODY.clear()
    claim, refusal = claimed_start_request(
        tmp_path, task_id="t-a", invocation_id="inv-3",
        continuation_of="run-pred", capture_id="inv-3", snapshot_id="snap-1")
    assert claim is False and refusal["reason"] == continuation.REFUSAL_SNAPSHOT_RELEASED


def test_a_configured_session_continues_with_its_note_never_the_canonical_work_order(tmp_path, monkeypatch):
    """The old session (or the engine's evidence packet) already holds the work order."""
    import ouroboros.tools.delegate as delegate

    _select_transport_access(monkeypatch, "readonly")   # a read-only child (#1609)
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


def test_a_continuation_carries_the_rows_no_start_of_its_line_held_under_its_own_heading(tmp_path, monkeypatch):
    """The continuation tail, both directions. A first start's references carry the rows its body does not hold
    (the start tail); a continuation carries exactly the rows absent from the START bodies of its ``continue_from``
    line, never the canonical order (the witness above stays), a relayed row dated before the run's start included,
    under its own heading after the binding and before the caller's note, a row's attachment named; a second
    continuation carries only the row that arrived after the first; a row a frame delivered live rides the next
    continuation once, under ``packet`` too, and not the one after; a continuation carries no start-tail row; an
    unreadable START body carries every row."""
    from ouroboros import delegate_message_store as frames, owner_words as ow
    from ouroboros.subagent_work_order import CONTINUATION_TAIL_FACT, START_TAIL_FACT

    heading = ow._HEADINGS["session"]
    file = {"upload": "u1", "name": "quota.png", "mime": "image/png", "kind": "image", "size": 1234, "sha256": "ab" * 32}
    rows = [ow._row("Build it", source="initial_user", carrier="ctx", task_id="t-root", ts="2026-10-04T05:00:00+00:00",
                    ref="chat 1 / msg-a"),
            ow._row("Before the run started", source="owner_mailbox", carrier="held", task_id="t-root",
                    ts="2026-10-04T04:00:00+00:00", ref="chat 1 / msg-b"),
            ow._row("See the file", source="owner_mailbox", carrier="held", task_id="t-root",
                    ts="2026-10-04T06:00:00+00:00", ref="chat 1 / msg-c", attachments=[file]),
            ow._row("Delivered live", source="owner_mailbox", carrier="held", task_id="t-root",
                    ts="2026-10-04T06:30:00+00:00", ref="chat 1 / msg-d"),
            ow._row("After the first continuation", source="owner_mailbox", carrier="held", task_id="t-root",
                    ts="2026-10-04T07:00:00+00:00", ref="chat 1 / msg-e")]

    def holding(*held):
        def prepare(ctx):
            ctx.task_metadata[ow.FIELD] = [rows[index] for index in held]
        return prepare

    def section(text, fact):
        assert text.count(heading) == 1 and text.count(fact) == 1, text
        return text.split(heading, 1)[1].split(fact, 1)[0]

    _select_transport_access(monkeypatch, "readonly")
    first, payload, _ctx, _calls = _start(tmp_path, monkeypatch, acting=False, run_id="run-prev", prepare=holding(0))
    assert payload["status"] == "started"
    tail = section(first["instructions"], START_TAIL_FACT)  # the start tail: the body held no stamp of row a
    assert "\nBuild it\n" in tail and "msg-a]" in tail and heading not in first["prompt"]
    _settle(tmp_path, "run-prev", "t-nanny-read")
    # A frame delivered live into run-prev: its row is not in any START body, so it rides the next continuation.
    entry = custody.replay(tmp_path)["run-prev"]
    live = frames.prepare(_ctx, entry, "starter", "Delivered live", "", None,
                          owner_words={"address": "row:1@2026-10-04T06:30:00+00:00#abc", "chat_id": 1,
                                       "ts": "2026-10-04T06:30:00+00:00"})
    frames.record_outcome(_ctx, "run-prev", live["frame"]["message_id"], "delivered", "")
    assert frames.placed_words(_ctx, "run-prev")[0]["outcome"] == "delivered"
    request, payload, _ctx, _calls = _start(
        tmp_path, monkeypatch, acting=False, run_id="run-next", prompt="the coordination note",
        start_kwargs={"continue_from": "run-prev", "continue_carrier": "packet"}, prepare=holding(0, 1, 2, 3))
    assert payload["status"] == "started" and request["continueFrom"] == "run-prev"
    prompt = request["prompt"]
    tail = section(prompt, CONTINUATION_TAIL_FACT)
    assert "Build it" not in tail  # held by run-prev's START body
    assert "\nBefore the run started\n" in tail and "msg-b]" in tail  # dated before the run, absent from it: rides
    assert "\nSee the file\n[Host-retained attachments: quota.png (image/png, 1234 bytes)]\n" in tail
    assert "\nDelivered live\n" in tail  # the live frame's row rides once: the host cannot know what the engine resumes
    binding, _words = prompt.split("\n\n" + heading, 1)
    assert "THIS RUN'S ACCESS AND WORKSPACE" in binding and prompt.endswith(CONTINUATION_TAIL_FACT + "\n\nthe coordination note")
    assert START_TAIL_FACT not in request["instructions"] and heading not in request["instructions"]  # no start tail
    _settle(tmp_path, "run-next", "t-nanny-read")
    # The second continuation carries only the row that arrived after the first; the delivered row does not ride again.
    request, payload, _ctx, _calls = _start(
        tmp_path, monkeypatch, acting=False, run_id="run-third", prompt="go on",
        start_kwargs={"continue_from": "run-next"}, prepare=holding(0, 1, 2, 3, 4))
    assert payload["status"] == "started"
    tail = section(request["prompt"], CONTINUATION_TAIL_FACT)
    assert "\nAfter the first continuation\n" in tail and "msg-e]" in tail
    for held in ("Build it", "Before the run started", "See the file", "Delivered live"):
        assert held not in tail
    assert START_TAIL_FACT not in request["instructions"]
    _settle(tmp_path, "run-third", "t-nanny-read")
    # The other side: every row held by the line's START bodies, no tail at all; an unreadable body, every row.
    request, payload, _ctx, _calls = _start(
        tmp_path, monkeypatch, acting=False, run_id="run-fourth", prompt="and on",
        start_kwargs={"continue_from": "run-third"}, prepare=holding(0, 1, 2, 3, 4))
    assert payload["status"] == "started" and heading not in request["prompt"] and request["prompt"].endswith("and on")
    assert continuation.line_start_bodies(tmp_path, "run-third") is not None
    assert len(continuation.line_start_bodies(tmp_path, "run-third")) == 3  # run-third, run-next, run-prev
    import ouroboros.delegate_invocation as invocation

    monkeypatch.setattr(invocation, "invocation_record", lambda *_a, **_k: (_ for _ in ()).throw(OSError("locked")))
    assert continuation.line_start_bodies(tmp_path, "run-third") is None
    holding(0, 1, 2, 3, 4)(_ctx)
    every = continuation.continuation_words(_ctx, tmp_path, "run-third")
    for row in rows:
        assert f"\n{row['text']}\n" in every + "\n"
    assert every.endswith(CONTINUATION_TAIL_FACT) and every.startswith(heading)


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


def test_empty_retry_of_an_ordinary_start_replays_the_recorded_request(tmp_path, monkeypatch):
    """#1614: the retained invocation IS the request, so '' reuses it; whitespace is text."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    lost = ClaudexorUnavailable("daemon_unreachable", "connection reset", status_code=503)
    _select_transport_access(monkeypatch, "readonly")   # a read-only child (#1609)
    original, unknown, _ctx, _calls = _start(
        tmp_path, monkeypatch, acting=False, run_id="run-first", prompt="original work", start_error=lost)
    invocation = unknown["pending_invocation_id"]
    request, refusal, _ctx, calls = _start(
        tmp_path, monkeypatch, acting=False, run_id="run-first", prompt=" ",
        start_kwargs={"retry_of": invocation})
    assert refusal["reason"] == "retry_prompt_mismatch" and request is None and calls == []
    request, payload, _ctx, calls = _start(
        tmp_path, monkeypatch, acting=False, run_id="run-first", prompt="",
        start_kwargs={"retry_of": invocation})
    assert payload["status"] == "started" and payload["invocation_id"] == invocation
    assert calls == [original] and request["prompt"] == "original work"
    assert custody.pending_invocations(tmp_path) == []


def test_an_unavailable_snapshot_lock_is_a_typed_claim_refusal(tmp_path, monkeypatch):
    import ouroboros.platform_layer as platform_layer
    from ouroboros.delegate_start_claims import claimed_start_request

    _seed(tmp_path, "run-pred", access="workspace_write", mode="agent", isolation="live", snapshot_id="snap-1",
          target_root="/t", execution_root="/snap")

    def broken(*_a, **_k):
        raise OSError("lock directory unwritable")

    monkeypatch.setattr(platform_layer, "acquire_exclusive_file_lock", broken)
    claim, refusal = claimed_start_request(
        tmp_path, task_id="t-a", invocation_id="inv-1",
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
    unbounded = bounded_max_seconds(_ctx(), None)
    assert (unbounded.seconds, unbounded.basis) == (0, CAP_BASIS_UNBOUNDED)
    monkeypatch.setattr(config, "get_task_abs_ceiling_sec", lambda: 300.0)
    clamped = bounded_max_seconds(_ctx(), 1000)
    assert clamped.basis == CAP_BASIS_REQUESTED_CLAMPED_LIFETIME and 150 <= clamped.seconds <= 200
    near = bounded_max_seconds(_ctx(deadline_in=50), 1000)
    assert near.basis == CAP_BASIS_REQUESTED_CLAMPED_DEADLINE and 40 <= near.seconds <= 50
    assert bounded_max_seconds(_ctx(), None).basis == CAP_BASIS_LIFETIME_DERIVED
    monkeypatch.setattr(config, "get_task_abs_ceiling_sec", lambda: 10.0)
    spent = bounded_max_seconds(_ctx(started_ago=100.0), 60)
    assert spent.refusal_code == "task_lifetime_exhausted" and spent.seconds == 0


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
    line = continuation._receipt_line(frozenset({"run-1"}), {"carrier": "native", "cause": cause})
    assert line.startswith("try None (?): native after " + cause) and "model unattested" in line


# ---------------------------------------------------------------------------
# Pre-admission continuation_superseded durable settlement (Part A)
# ---------------------------------------------------------------------------


_PROVEN_ENGINE_VERSION = "6.3.0"
_PROVEN_ENGINE_SHA = "0ff3df717c2e807fdbd3a233487e35d81993fdea"


def _continuation_superseded_problem(pred_run_id: str, head_run_id: str = "run-other-head", **extra):
    context = {"runId": pred_run_id, "head": head_run_id, **extra.pop("context_extra", {})}
    return {"code": "continuation_superseded", "message": f"{pred_run_id} already continued by {head_run_id}",
            "retryable": False, "context": context, **extra}


def _make_continuation_engine(
    tmp_path, *, store="store-a", lookup=True, seed_projects=True,
    engine_version=_PROVEN_ENGINE_VERSION, engine_sha=_PROVEN_ENGINE_SHA,
):
    import httpx
    from ouroboros.config import CLAUDEXOR_PROTOCOL_MAJOR
    from tests._delegated_transport_shared import Engine

    class _ContinuationEngine(Engine):
        def __init__(self):
            super().__init__(tmp_path, store=store, lookup=lookup)
            self.projects = [{"id": "prj-existing", "root": self.root}] if seed_projects else []
            self.deleted_projects, self.thread_turns, self.lookup_gaps = [], [], set()
            self.thread_refusal = self.on_before_create_response = self.create_raw_response = self.on_lookup = None
            self.thread_head = self.thread_root = ""

        def _lookup(self, request, key, path):
            if self.on_lookup is not None:
                self.on_lookup(key, path)
            if key in self.lookup_gaps:
                thread = path.split("/")[3] if path.startswith("/v2/threads/") else ""
                self.lookups.append((thread or "run", key, json.loads(request.read())))
                return httpx.Response(200, json={"status": "not_found", "storeId": self.store})
            return super()._lookup(request, key, path)

        def __call__(self, request):
            path, method = request.url.path, request.method
            if path == "/v2/handshake":
                self.paths.append((method, path))
                eng = {"version": engine_version, **({"sha": engine_sha} if engine_sha is not None else {})}
                return httpx.Response(200, json={"protocolMajor": CLAUDEXOR_PROTOCOL_MAJOR, "compatible": True,
                                                 "engine": eng, **self._named(request)})
            if path == "/v2/agent-capabilities":
                self.paths.append((method, path))
                return httpx.Response(200, json={"modes": ["ask", "agent", "plan"], "runControlKeys": ["continueFrom"],
                                                 "harnesses": [{
                                                     "id": "some-route", "enabled": True, "status": self.health,
                                                     "accessProfilesSupported": ["readonly", "workspace_write"]}]})
            if path == "/v2/projects" and method == "GET":
                self.paths.append((method, path))
                return httpx.Response(200, json={"projects": list(self.projects)})
            if path == "/v2/projects" and method == "POST":
                self.paths.append((method, path))
                root = str(json.loads(request.read())["root"])
                existing = next((row for row in self.projects if row["root"] == root), None)
                if existing is not None:
                    return httpx.Response(200, json={"id": existing["id"], "created": False, "root": root,
                                                     **self._named(request)})
                pid = f"prj-created-{len(self.projects) + 1}"
                self.projects.append({"id": pid, "root": root})
                return httpx.Response(200, json={"id": pid, "created": True, "root": root, **self._named(request)})
            if (path.startswith("/v2/projects/") and method == "DELETE") or path.endswith("/release"):
                self.paths.append((method, path))
                self.deleted_projects.append((method, path))
                return httpx.Response(200, json={"state": "released"})
            if path == "/v2/threads" and method == "POST":
                self.paths.append((method, path))
                self.thread_root = str((json.loads(request.read()).get("scope") or {}).get("root") or "")
                return httpx.Response(200, json={"id": "thr-plan-1", **self._named(request)})
            if path == "/v2/threads/thr-plan-1" and method == "GET":
                self.paths.append((method, path))
                return httpx.Response(200, json={"thread": {"id": "thr-plan-1", "headRunId": self.thread_head,
                                                            "repoRoot": self.thread_root}})
            if path.startswith("/v2/threads/") and path.endswith("/turns") and method == "POST":
                self.paths.append((method, path))
                key, sent = request.headers.get("Idempotency-Key"), json.loads(request.read())
                self.thread_turns.append((key, sent))
                if self.thread_refusal is not None:
                    return httpx.Response(self.thread_refusal[0], json=self.thread_refusal[1])
                job = self._held(key) or self.accept(key, sent, run=True)
                self.thread_head = str(self.commands[job]["run"] or "")
                return httpx.Response(200, json={"jobId": job, "runId": self.commands[job]["run"], "turnId": "turn-1",
                                                 **self._named(request)})
            if path == "/v2/runs" and method == "POST":
                self.paths.append((method, path))
                self.store, self.replace_at_post = self.replace_at_post or self.store, ""
                key, sent = request.headers.get("Idempotency-Key"), json.loads(request.read())
                self.creates.append((key, self.store, sent))
                if self.on_before_create_response is not None:
                    hook, self.on_before_create_response = self.on_before_create_response, None
                    hook(key, sent)
                if self.create_raw_response is not None:
                    return self.create_raw_response
                if self.create_refusal is not None:
                    status, body = (self.create_refusal(key, sent) if callable(self.create_refusal)
                                    else self.create_refusal)
                    return httpx.Response(status, json=body)
                job = self._held(key) or self.accept(key, sent, run=not self.queue_runs)
                if self.on_accept is not None:
                    hook, self.on_accept = self.on_accept, None
                    hook(job)
                if self.lose:
                    self.lose -= 1
                    raise httpx.ConnectError("the reply was lost after the POST was accepted")
                return httpx.Response(200, json={"jobId": job, "runId": self.commands[job]["run"], "runDir": "/tmp/run",
                                                 **self._named(request)})
            return super().__call__(request)

    return _ContinuationEngine()


def _bind_engine(tmp_path, monkeypatch, engine, *, acting=False, task_id=None):
    from ouroboros.gateways import claudexor as gw

    _select_transport_access(monkeypatch, None if acting else "readonly")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(tmp_path / "snap_root"))
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *_a, **_k: engine.gateway())
    custody._CUSTODY.clear()
    return _delegating_ctx(tmp_path, acting=acting, task_id=task_id or f"t-nanny-{'write' if acting else 'read'}")


def _seed_continued_engine(tmp_path, monkeypatch, name="", *, acting=False, task_id=None, mode=None, **engine_kw):
    from ouroboros.tools import delegate

    case_dir = tmp_path / name if name else tmp_path
    case_dir.mkdir(parents=True, exist_ok=True)
    engine = _make_continuation_engine(case_dir, **engine_kw)
    ctx = _bind_engine(case_dir, monkeypatch, engine, acting=acting, task_id=task_id)
    start_kw = {"mode": mode} if mode is not None else {}
    pred_id = json.loads(delegate._delegate_start(ctx, "seed", max_seconds=120, **start_kw).text)["run_id"]
    _settle(case_dir, pred_id, ctx.task_id)
    return case_dir, engine, ctx, pred_id


def test_pre_admission_continuation_superseded_settles_readonly_continuation_durably(tmp_path, monkeypatch):
    from ouroboros import (
        delegate_custody_current as current,
        delegate_invocation as inv,
        delegate_pending as pending,
        delegate_recovery as recovery,
    )
    from ouroboros.gateways.claudexor_observation import canonical_request_digest
    from ouroboros.subagent_bootstrap import _startup_refusal_definite
    from ouroboros.tools import delegate

    case_dir, engine, ctx, pred_id = _seed_continued_engine(tmp_path, monkeypatch)
    in_guard, body_reads_in_guard, guard_calls = [False], [], []
    real_body, real_guard = pending.request_body, continuation.can_settle_pre_admission_rejection

    def spy_body(*a, **k):
        if in_guard[0]:
            body_reads_in_guard.append(a)
        return real_body(*a, **k)

    def spy_guard(*a, **k):
        in_guard[0] = True
        try:
            ok = real_guard(*a, **k)
            guard_calls.append(ok)
            return ok
        finally:
            in_guard[0] = False

    monkeypatch.setattr(pending, "request_body", spy_body)
    monkeypatch.setattr(continuation, "can_settle_pre_admission_rejection", spy_guard)
    engine.create_refusal = (409, _continuation_superseded_problem(pred_id, "run-other-head"))
    refusal = json.loads(delegate._delegate_start(
        ctx, "continue inspection", max_seconds=120, continue_from=pred_id).text)
    iid, _store, sent_body = engine.creates[-1]

    assert guard_calls == [True] and body_reads_in_guard == []
    assert refusal["ok"] is False and refusal["status"] == "refused"
    assert refusal["reason"] == continuation.REFUSAL_SUPERSEDED
    assert refusal["start_rejected_before_admission"] is True
    assert refusal["continuation_of"] == pred_id and refusal["superseded_by_head"] == "run-other-head"
    assert refusal["project_retired"] is False
    assert "pending_invocation_id" not in refusal and "retry_hint" not in refusal
    assert "definitely_unrun" not in refusal and _startup_refusal_definite(refusal) is False

    rows = custody.custody_rows(case_dir)
    assert [r for r in rows if r.get("type") == custody.START_UNSENT and r.get("invocation_id") == iid] == []
    failed_rows = [r for r in rows if r.get("type") == custody.START_FAILED and r.get("invocation_id") == iid]
    assert len(failed_rows) == 1
    row = failed_rows[0]
    assert row["definite"] is True and row["pre_admission_rejected"] is True and row["start_sent"] is True
    assert row["reason"] == continuation.REFUSAL_SUPERSEDED and row["status_code"] == 409
    assert row["continuation_of"] == pred_id and row["superseded_by_head"] == "run-other-head"
    assert row["daemon_store_id"] == "store-a" and row["request_digest"] == canonical_request_digest(sent_body)
    assert row["engine_version"] == _PROVEN_ENGINE_VERSION and row["engine_build_sha"] == _PROVEN_ENGINE_SHA

    custody._CUSTODY.clear()
    assert custody.invocation_record(case_dir, iid)["state"] == "failed_definite"
    assert inv.invocation_record(case_dir, iid, rows=rows)["state"] == "failed_definite"
    assert custody.pending_invocations(case_dir) == []
    assert pending.pending_invocations(case_dir, rows=rows) == []
    assert recovery.unsettled_start_ids(case_dir, ctx.task_id)["pending_invocation_ids"] == []
    assert recovery.unsettled_start_ids(case_dir, ctx.task_id, rows=rows)["pending_invocation_ids"] == []
    folded = current.fold(rows)
    assert f"invocation:{iid}" not in folded["open"] and iid in folded["refused"]

    replayed = custody.replay(case_dir)
    assert "run-other-head" not in replayed
    assert replayed[pred_id].patch_disposed == "" and replayed[pred_id].superseded_by == ""

    creates_before_retry = len(engine.creates)
    retry = json.loads(delegate._delegate_start(ctx, "", retry_of=iid).text)
    assert retry["ok"] is False and retry["reason"] == "invocation_definitely_refused"
    assert len(engine.creates) == creates_before_retry


def test_pre_admission_continuation_superseded_keeps_snapshots_and_owes_owned_registration(tmp_path, monkeypatch):
    from ouroboros.delegate_custody_current import owed_registrations
    from ouroboros.subagent_worktrees import find_execution_snapshot
    from ouroboros.tools import delegate

    # 1. Non-persistent newly created registration on readonly continuation: no eager DELETE, OWED duty kept.
    ro_dir = tmp_path / "ro"
    ro_dir.mkdir()
    ro_engine = _make_continuation_engine(ro_dir, seed_projects=False)
    ro_ctx = _bind_engine(ro_dir, monkeypatch, ro_engine, acting=False, task_id="t-ro")
    _seed(ro_dir, "run-ro-pred", task_id="t-ro", root_task_id="t-root")
    ro_engine.create_refusal = (409, _continuation_superseded_problem("run-ro-pred", "run-ro-head"))
    ro_refusal = json.loads(delegate._delegate_start(
        ro_ctx, "continue readonly", max_seconds=120, continue_from="run-ro-pred").text)
    assert ro_refusal["start_rejected_before_admission"] is True and ro_refusal["project_retired"] is False
    assert ro_refusal["project_id"].startswith("prj-created-")
    assert ro_refusal["project_retention_reason"] == "continuation_head_may_share_registration"
    assert ro_engine.deleted_projects == []
    assert any(duty.get("project_id") == ro_refusal["project_id"] for duty in owed_registrations(ro_dir))
    assert custody.pending_invocations(ro_dir) == []

    # 2. Same-snapshot mutating continuation: clears CONTINUATION_PENDING and leaves predecessor snapshot intact.
    before_snaps = _snapshot_count()
    mut_dir, mut_engine, mut_ctx, pred_id = _seed_continued_engine(
        tmp_path, monkeypatch, "mut", acting=True, task_id="t-mut", seed_projects=False)
    pred = custody.replay(mut_dir)[pred_id]
    assert _snapshot_count() == before_snaps + 1

    mut_engine.create_refusal = (409, _continuation_superseded_problem(pred_id, "run-mut-head"))
    same_snap = json.loads(delegate._delegate_start(
        mut_ctx, "continue in snapshot", max_seconds=120, continue_from=pred_id).text)
    assert same_snap["start_rejected_before_admission"] is True and "pending_invocation_id" not in same_snap
    assert mut_engine.deleted_projects == []
    assert _snapshot_count() == before_snaps + 1 and find_execution_snapshot(pred.snapshot_id) is not None
    assert pathlib.Path(pred.execution_root).exists()
    custody._CUSTODY.clear()
    pred_after = custody.replay(mut_dir)[pred_id]
    assert pred_after.patch_disposed == "" and pred_after.superseded_by == ""
    assert "CONTINUATION_PENDING" not in continuation.disposition_refusal(mut_dir, pred_after)

    # 3. Fresh-snapshot mutating continuation (predecessor already disposed): fresh snapshot stays intact too.
    assert custody.emit(mut_dir, custody.PATCH_DISPOSED, {
        "run_id": pred_id, "task_id": mut_ctx.task_id, "snapshot_id": pred.snapshot_id, "disposition": "rejected"})
    custody._CUSTODY.clear()
    fresh_refusal = json.loads(delegate._delegate_start(
        mut_ctx, "continue in fresh snapshot", max_seconds=120, continue_from=pred_id).text)
    fresh_rec = custody.invocation_record(mut_dir, mut_engine.creates[-1][0])
    assert fresh_refusal["start_rejected_before_admission"] is True and fresh_rec["state"] == "failed_definite"
    assert fresh_rec["snapshot_id"] and fresh_rec["snapshot_id"] != pred.snapshot_id
    assert _snapshot_count() == before_snaps + 2 and find_execution_snapshot(fresh_rec["snapshot_id"]) is not None
    assert pathlib.Path(fresh_rec["execution_root"]).exists() and mut_engine.deleted_projects == []


def test_pre_admission_continuation_superseded_admitted_failed_job_stays_on_accepted_recovery(tmp_path, monkeypatch):
    from ouroboros.tools import delegate

    case_dir, engine, ctx, pred_id = _seed_continued_engine(tmp_path, monkeypatch)

    def admit_and_refuse(key, sent):
        problem = _continuation_superseded_problem(pred_id, "run-other-head", jobId="job-admitted", state="failed")
        engine.accept(key, sent, run=False, state="failed", problem=problem)
        return 409, problem

    engine.create_refusal = admit_and_refuse
    result = json.loads(delegate._delegate_start(ctx, "continue", max_seconds=120, continue_from=pred_id).text)
    iid = engine.creates[-1][0]
    assert result["status"] == "started" and result["run_id"]
    assert custody.invocation_record(case_dir, iid)["state"] == "started"
    assert custody.pending_invocations(case_dir) == []


@pytest.mark.parametrize("mode", [
    "unbound_possession", "lookup_refused", "lookup_unavailable", "lookup_unsupported",
    "in_place_store_replaced",
])
def test_pre_admission_continuation_superseded_stays_unknown_on_possession_lookup_or_store_gap(
        tmp_path, monkeypatch, mode):
    from ouroboros import delegate_accepted
    from ouroboros.tools import delegate

    case_dir, engine, ctx, pred_id = _seed_continued_engine(
        tmp_path, monkeypatch, mode, lookup=mode != "lookup_unsupported")
    if mode == "unbound_possession":
        engine.create_refusal = lambda key, sent: (
            engine.accept(key, sent, run=True, state="failed"),
            (409, _continuation_superseded_problem(pred_id, "run-other-head")),
        )[1]
        monkeypatch.setattr(delegate_accepted, "bind_started",
                            lambda *_a, **_k: delegate_accepted.Bound(None, False, False))
    else:
        engine.create_refusal = (409, _continuation_superseded_problem(pred_id, "run-other-head"))
        if mode == "lookup_refused":
            engine.answer = (409, _continuation_superseded_problem(pred_id, "run-other-head"))
        elif mode == "lookup_unavailable":
            engine.answer = (503, {"code": "daemon_busy", "message": "busy", "retryable": True})
        elif mode == "in_place_store_replaced":
            engine.replace_at_post = "store-b"

    payload = json.loads(delegate._delegate_start(ctx, "continue", max_seconds=120, continue_from=pred_id).text)
    iid = engine.creates[-1][0]
    assert payload["ok"] is False and payload["pending_invocation_id"] == iid
    assert "start_rejected_before_admission" not in payload
    assert custody.invocation_record(case_dir, iid)["state"] == "pending"
    assert [row["invocation_id"] for row in custody.pending_invocations(case_dir)] == [iid]
    assert [r for r in custody.custody_rows(case_dir)
            if r.get("type") == custody.START_FAILED and r.get("invocation_id") == iid
            and r.get("definite") is True] == []


def test_pre_admission_continuation_superseded_stays_unknown_on_empty_store_rotated_descriptor_retry_thread_or_lookup_gap(
        tmp_path, monkeypatch):
    from ouroboros import delegate_accepted
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.tools import delegate

    # 1. Empty handshake store identity -> stays unknown.
    empty_dir, eng_empty, ctx_empty, pred_empty = _seed_continued_engine(tmp_path, monkeypatch, "empty_store", store="")
    eng_empty.create_refusal = (409, _continuation_superseded_problem(pred_empty, "run-head-e"))
    res_empty = json.loads(delegate._delegate_start(ctx_empty, "continue", max_seconds=120, continue_from=pred_empty).text)
    assert res_empty["pending_invocation_id"] and custody.invocation_record(
        empty_dir, res_empty["pending_invocation_id"])["state"] == "pending"

    # 2. Descriptor rotated during POST -> stays unknown.
    rot_dir, eng_rot, ctx_rot, pred_rot = _seed_continued_engine(tmp_path, monkeypatch, "rotated")
    eng_rot.on_before_create_response = lambda _k, _s: pathlib.Path(
        eng_rot.endpoint.descriptor_path).write_text('{"rotated": true}', encoding="utf-8")
    eng_rot.create_refusal = (409, _continuation_superseded_problem(pred_rot, "run-head-r"))
    res_rot = json.loads(delegate._delegate_start(ctx_rot, "continue", max_seconds=120, continue_from=pred_rot).text)
    assert res_rot["pending_invocation_id"] and custody.invocation_record(
        rot_dir, res_rot["pending_invocation_id"])["state"] == "pending"

    # 3. Replay (retry_of, recovering=True) that actually re-POSTs and gets 409 -> stays unknown.
    retry_dir, eng_retry, ctx_retry, pred_retry = _seed_continued_engine(tmp_path, monkeypatch, "retry")
    eng_retry.lose = 1
    seed_iid = json.loads(delegate._delegate_start(
        ctx_retry, "continue", max_seconds=120, continue_from=pred_retry).text)["pending_invocation_id"]
    orig_row = next(r for r in custody.custody_rows(retry_dir)
                    if r.get("type") == custody.START_REQUESTED and r.get("invocation_id") == seed_iid)
    orig_body = custody.invocation_record(retry_dir, seed_iid)["request"]
    retry_iid = "inv-retry-409"
    retry_fields = {k: v for k, v in orig_row.items()
                    if k not in ("ts", "type", "event_id", "custody_ordinal", "request", "request_ref", "invocation_id")}
    assert custody.record_start_requested(
        retry_dir, **{**retry_fields, "invocation_id": retry_iid, "daemon_store_id": "store-a",
                      "handshake_store_id": "store-a", "store_identity_unproven": False, "request": orig_body})
    creates_before = len(eng_retry.creates)
    eng_retry.create_refusal = (409, _continuation_superseded_problem(pred_retry, "run-head-retry"))
    replayed_res = json.loads(delegate._delegate_start(ctx_retry, "", retry_of=retry_iid).text)
    assert len(eng_retry.creates) == creates_before + 1
    assert replayed_res["ok"] is False and replayed_res["pending_invocation_id"] == retry_iid
    assert "start_rejected_before_admission" not in replayed_res
    assert custody.invocation_record(retry_dir, retry_iid)["state"] == "pending"

    # 4. Thread/plan continuation (POST /v2/threads/:id/turns) receiving 409 -> stays unknown.
    plan_dir, eng_plan, ctx_plan, plan_pred = _seed_continued_engine(tmp_path, monkeypatch, "plan", mode="plan")
    eng_plan.thread_refusal = (409, _continuation_superseded_problem(plan_pred, "run-head-plan"))
    plan_res = json.loads(delegate._delegate_start(
        ctx_plan, "continue plan", max_seconds=120, continue_from=plan_pred, mode="plan").text)
    assert len(eng_plan.thread_turns) == 2 and plan_res["ok"] is False and plan_res["pending_invocation_id"]
    assert eng_plan.thread_turns[-1][1]["mode"] == "plan"
    assert "start_rejected_before_admission" not in plan_res
    assert custody.invocation_record(plan_dir, plan_res["pending_invocation_id"])["state"] == "pending"

    # 5. Historical pending continuation + same-store lookup gap (earlier accepted command remains
    #    intact in engine.commands, lookup explicitly returns not_found): never settles without live
    #    proof, never binds the unlooked-up run, and retry_of sends no second POST.
    gap_dir, eng_gap, ctx_gap, pred_gap = _seed_continued_engine(tmp_path, monkeypatch, "lookup_gap")
    eng_gap.lose = 1
    lost_iid = json.loads(delegate._delegate_start(
        ctx_gap, "continue", max_seconds=120, continue_from=pred_gap).text)["pending_invocation_id"]
    accepted_before = {job: dict(row) for job, row in eng_gap.commands.items() if row["key"] == lost_iid}
    assert len(accepted_before) == 1
    accepted_run_id = next(iter(accepted_before.values()))["run"]
    eng_gap.lookup_gaps.add(lost_iid)
    gw_gap = eng_gap.gateway()
    try:
        untagged_409 = ClaudexorUnavailable("continuation_superseded", "historical", status_code=409)
        held, read = delegate_accepted.accepted_after_refusal(gap_dir, gw_gap, lost_iid, untagged_409)
        assert held is None and read["lookup"] == "not_found" and read["store_id"] == "store-a"
        assert continuation.settle_pre_admission_continuation_rejection(
            gap_dir, task_id=ctx_gap.task_id, root_task_id="t-root", invocation_id=lost_iid,
            gateway=gw_gap, exc=untagged_409, accepted_read=read,
            request_body=custody.invocation_record(gap_dir, lost_iid)["request"],
        ) is None
    finally:
        gw_gap.close()
    creates_before_gap_retry = len(eng_gap.creates)
    gap_retry = json.loads(delegate._delegate_start(ctx_gap, "", retry_of=lost_iid).text)
    assert gap_retry["ok"] is False and gap_retry["pending_invocation_id"] == lost_iid
    assert len(eng_gap.creates) == creates_before_gap_retry
    assert custody.invocation_record(gap_dir, lost_iid)["state"] == "pending"
    assert accepted_run_id not in custody.replay(gap_dir)
    assert {job: dict(row) for job, row in eng_gap.commands.items() if row["key"] == lost_iid} == accepted_before


def test_pre_admission_continuation_reducer_guard_refuses_races_risk_choice_drift_or_lock_failure(
        tmp_path, monkeypatch):
    from ouroboros import (
        delegate_custody_current as current,
        delegate_invocation as inv,
        delegate_pending as pending,
        store,
    )
    from ouroboros.gateways.claudexor_observation import canonical_request_digest
    from ouroboros.observability import write_blob
    from ouroboros.task_results import load_task_result
    from ouroboros.tools import delegate

    # 1. Competing STARTED arrives AFTER own lookup returned not_found AND AFTER off-lock
    #    binding prep (right before BEGIN IMMEDIATE): reducer guard fires, refuses settlement,
    #    keeps bound run, predecessor, and root task result intact.
    race_dir, eng_race, ctx_race, pred_race = _seed_continued_engine(tmp_path, monkeypatch, "race_started")
    seed_result(race_dir, "t-root", {"task_id": "t-root", "root_task_id": "t-root", "status": "running", "note": "keep"})
    eng_race.create_refusal = (409, _continuation_superseded_problem(pred_race, "run-other-head"))
    race_guard_calls = []
    real_prepare = continuation.prepare_pre_admission_rejection_binding
    real_guard = continuation.can_settle_pre_admission_rejection

    def inject_started_after_prepare(drive_root, event):
        prepared = real_prepare(drive_root, event)
        if prepared is not None:
            assert custody.record_started(race_dir, custody.RunCustody(
                run_id="run-won-after-lookup", task_id=ctx_race.task_id, root_task_id="t-root",
                route_id=ROUTE, model="m", invocation_id=event["invocation_id"],
                continuation_of=pred_race, daemon_store_id="store-a"))
        return prepared

    def spy_race_guard(txn, event, *, verified=None):
        ok = real_guard(txn, event, verified=verified)
        race_guard_calls.append((event["invocation_id"], ok))
        return ok

    monkeypatch.setattr(continuation, "prepare_pre_admission_rejection_binding", inject_started_after_prepare)
    monkeypatch.setattr(continuation, "can_settle_pre_admission_rejection", spy_race_guard)
    res_race = json.loads(delegate._delegate_start(ctx_race, "continue", max_seconds=120, continue_from=pred_race).text)
    monkeypatch.setattr(continuation, "prepare_pre_admission_rejection_binding", real_prepare)
    monkeypatch.setattr(continuation, "can_settle_pre_admission_rejection", real_guard)
    race_iid = eng_race.creates[-1][0]
    assert len(eng_race.lookups) == 1
    assert race_guard_calls == [(race_iid, False)]
    assert res_race["ok"] is False and "start_rejected_before_admission" not in res_race
    assert [r for r in custody.custody_rows(race_dir)
            if r.get("type") == custody.START_FAILED and r.get("invocation_id") == race_iid
            and r.get("definite") is True] == []
    with store.open_store(race_dir, write=False) as txn:
        assert txn.get_row("custody_refused", race_iid) is None
    assert custody.invocation_record(race_dir, race_iid)["state"] == "started"
    assert custody.invocation_record(race_dir, race_iid)["run_id"] == "run-won-after-lookup"
    assert pred_race in custody.replay(race_dir) and "run-won-after-lookup" in custody.replay(race_dir)
    assert load_task_result(race_dir, "t-root")["note"] == "keep"

    # 2. Concurrent START_RISK_ACCEPTED (Resume risk choice) lands during POST -> refuses settlement.
    risk_dir, eng_risk, ctx_risk, pred_risk = _seed_continued_engine(tmp_path, monkeypatch, "race_risk")

    def accept_risk_before_409(key, _sent):
        seed_result(risk_dir, "t-root", {
            "task_id": "t-root", "root_task_id": "t-root", "status": "paused",
            "owner_pause": {"fence_id": "f-1"}, "budget_pause": {"pause_id": "p-1", "state": "paused"}})
        seed_result(risk_dir, ctx_risk.task_id, {
            "task_id": ctx_risk.task_id, "root_task_id": "t-root", "status": "paused",
            "budget_pause": {"pause_id": "p-1", "state": "paused"}})
        choice_id = f"delegate_start:t-root/{ctx_risk.task_id}/{key}"
        assert custody.emit(risk_dir, custody.START_RISK_ACCEPTED, {
            "root_task_id": "t-root", "task_id": ctx_risk.task_id, "invocation_id": key,
            "choice_id": choice_id, "resuming_task_id": ctx_risk.task_id,
            "decision": {"choice_id": choice_id, "fence_id": "f-1", "pause_id": "p-1",
                         "root_task_id": "t-root", "task_id": ctx_risk.task_id, "invocation_id": key,
                         "resuming_task_id": ctx_risk.task_id, "by": {"actor": "owner"}, "reason": "resume"}})

    eng_risk.on_before_create_response = accept_risk_before_409
    eng_risk.create_refusal = (409, _continuation_superseded_problem(pred_risk, "run-other-head"))
    res_risk = json.loads(delegate._delegate_start(ctx_risk, "continue", max_seconds=120, continue_from=pred_risk).text)
    risk_iid = eng_risk.creates[-1][0]
    assert len(eng_risk.creates) == 2 and res_risk["pending_invocation_id"] == risk_iid
    risk_rec = custody.invocation_record(risk_dir, risk_iid)
    assert risk_rec["state"] == "pending" and risk_rec.get("resume_decisions")

    # 3. Replay parity across fold, invocation_record(rows=...), and pending_invocations(rows=...).
    synth_rows = [
        {"type": custody.START_REQUESTED, "event_id": "e1", "invocation_id": "inv-p", "task_id": "t-a",
         "root_task_id": "t-a", "continuation_of": "run-pred", "request": {"prompt": "x", "continueFrom": "run-pred"}},
        {"type": custody.START_RISK_ACCEPTED, "event_id": "e2", "invocation_id": "inv-p", "task_id": "t-a",
         "root_task_id": "t-a", "resuming_task_id": "t-a", "decision": {"choice_id": "c1"}},
        {"type": custody.START_FAILED, "event_id": "e3", "invocation_id": "inv-p", "task_id": "t-a",
         "root_task_id": "t-a", "definite": True, "pre_admission_rejected": True,
         "start_sent": True, "reason": continuation.REFUSAL_SUPERSEDED},
    ]
    assert inv.invocation_record(tmp_path, "inv-p", rows=synth_rows)["state"] == "pending"
    assert [r["invocation_id"] for r in pending.pending_invocations(tmp_path, rows=synth_rows)] == ["inv-p"]
    synth_fold = current.fold(synth_rows)
    assert "invocation:inv-p" in synth_fold["open"] and "inv-p" not in synth_fold["refused"]

    # 4. Off-lock binding prep, producer allowlist, unreadable CAS blob, review row, and source identity drift.
    guard_dir = tmp_path / "guard"
    guard_dir.mkdir()
    body = {"prompt": "continue", "continueFrom": "run-g-pred", "access": "readonly", "mode": "ask"}
    assert custody.record_start_requested(
        guard_dir, invocation_id="inv-guard", task_id="t-a", root_task_id="t-a", idempotency_key="k-1",
        continuation_of="run-g-pred", handshake_store_id="store-a", daemon_store_id="",
        store_identity_unproven=True, request=body)
    good_event = {
        "type": custody.START_FAILED, "run_id": "", "task_id": "t-a", "root_task_id": "t-a",
        "invocation_id": "inv-guard", "idempotency_key": "k-1", "wire_idempotency_key": "inv-guard",
        "reason": continuation.REFUSAL_SUPERSEDED, "status_code": 409, "definite": True,
        "pre_admission_rejected": True, "start_sent": True, "continuation_of": "run-g-pred",
        "superseded_by_head": "run-g-head", "daemon_store_id": "store-a",
        "request_digest": canonical_request_digest(body),
        "engine_version": _PROVEN_ENGINE_VERSION, "engine_build_sha": _PROVEN_ENGINE_SHA,
    }
    prepared = continuation.prepare_pre_admission_rejection_binding(guard_dir, good_event)
    assert prepared is not None
    assert continuation.prepare_pre_admission_rejection_binding(guard_dir, {**good_event, "request_digest": "0" * 64}) is None
    assert continuation.prepare_pre_admission_rejection_binding(guard_dir, {**good_event, "engine_version": "3.2.0"}) is None
    with store.open_store(guard_dir, write=False) as txn:
        assert continuation.can_settle_pre_admission_rejection(txn, good_event) is False  # verified binding required
        assert continuation.can_settle_pre_admission_rejection(txn, {**good_event, "daemon_store_id": "store-b"}, verified=prepared) is False
        assert continuation.can_settle_pre_admission_rejection(txn, {**good_event, "superseded_by_head": "inv-guard"}, verified=prepared) is False
        assert continuation.can_settle_pre_admission_rejection(txn, {**good_event, "engine_build_sha": "0" * 40}, verified=prepared) is False
        assert continuation.can_settle_pre_admission_rejection(txn, good_event, verified=prepared) is True

    # Source identity drift between off-lock prepare and commit closes nothing.
    drifted_ref = write_blob(guard_dir, {**body, "prompt": "drifted"}, kind="json")

    def drift_before_commit(drive_root, event):
        binding = real_prepare(drive_root, event)
        assert binding is not None
        with store.open_store(guard_dir, write=True) as wtxn:
            open_row = wtxn.get_row("custody_open", "invocation:inv-guard")
            inv_row = wtxn.get_row(inv.INVOCATIONS, "inv-guard")
            open_row["request"]["request_ref"] = dict(drifted_ref)
            inv_row["request"]["request_ref"] = dict(drifted_ref)
            wtxn.put_row("custody_open", "invocation:inv-guard", open_row)
            wtxn.put_row(inv.INVOCATIONS, "inv-guard", inv_row)
        return binding

    monkeypatch.setattr(continuation, "prepare_pre_admission_rejection_binding", drift_before_commit)
    assert custody.emit(guard_dir, custody.START_FAILED, good_event) is False
    monkeypatch.setattr(continuation, "prepare_pre_admission_rejection_binding", real_prepare)
    assert custody.invocation_record(guard_dir, "inv-guard")["state"] == "pending"
    with store.open_store(guard_dir, write=False) as txn:
        assert txn.get_row("custody_open", "invocation:inv-guard") is not None
        assert txn.get_row("custody_refused", "inv-guard") is None

    # Tampered/missing CAS blob and review-owned row both refuse settlement.
    assert custody.emit(guard_dir, custody.START_REQUESTED, {
        "invocation_id": "inv-bad-blob", "task_id": "t-a", "root_task_id": "t-a", "idempotency_key": "k-bad",
        "continuation_of": "run-g-pred", "handshake_store_id": "store-a", "daemon_store_id": "",
        "store_identity_unproven": True,
        "request_ref": {"sha256": "f" * 64, "kind": "json", "encoding": "gzip", "size": 12},
    })
    bad_blob_event = {**good_event, "invocation_id": "inv-bad-blob", "wire_idempotency_key": "inv-bad-blob",
                      "idempotency_key": "k-bad"}
    assert continuation.prepare_pre_admission_rejection_binding(guard_dir, bad_blob_event) is None
    assert custody.emit(guard_dir, custody.START_FAILED, bad_blob_event) is False
    assert custody.record_start_requested(
        guard_dir, invocation_id="inv-rev", task_id="t-a", root_task_id="t-a", idempotency_key="k-rev",
        continuation_of="run-g-pred", handshake_store_id="store-a", daemon_store_id="",
        store_identity_unproven=True, source="review_substrate_triad", request=body)
    rev_event = {**good_event, "invocation_id": "inv-rev", "wire_idempotency_key": "inv-rev", "idempotency_key": "k-rev"}
    rev_prepared = continuation.prepare_pre_admission_rejection_binding(guard_dir, rev_event)
    assert rev_prepared is not None
    with store.open_store(guard_dir, write=False) as txn:
        assert continuation.can_settle_pre_admission_rejection(txn, rev_event, verified=rev_prepared) is False

    # 5. Lock unavailable falls back cleanly to pending/unknown.
    lock_dir, eng_lock, ctx_lock, pred_lock = _seed_continued_engine(tmp_path, monkeypatch, "lock_busy")
    eng_lock.create_refusal = (409, _continuation_superseded_problem(pred_lock, "run-lock-head"))
    real_lock = custody.actor_decision_lock
    busy_claims = []

    def fail_lock_on_second(_root, _task_id, **_kw):
        if len(eng_lock.creates) >= 2:
            busy_claims.append(_task_id)
            raise custody.ActorDecisionLockUnavailable({"reason": "contention"}, 20.0)
        return real_lock(_root, _task_id, **_kw)

    monkeypatch.setattr(custody, "actor_decision_lock", fail_lock_on_second)
    res_lock = json.loads(delegate._delegate_start(ctx_lock, "continue", max_seconds=120, continue_from=pred_lock).text)
    lock_iid = eng_lock.creates[-1][0]
    assert ctx_lock.task_id in busy_claims and res_lock["pending_invocation_id"] == lock_iid
    assert "start_rejected_before_admission" not in res_lock
    monkeypatch.setattr(custody, "actor_decision_lock", real_lock)
    assert custody.invocation_record(lock_dir, lock_iid)["state"] == "pending"


def _seed_queued_job_predecessor(tmp_path, monkeypatch, name, *, bind_canonical=True, store="store-a", lookup=True):
    from ouroboros import delegate_custody_current as current
    from ouroboros.tools import delegate

    case_dir = tmp_path / name
    case_dir.mkdir(parents=True, exist_ok=True)
    engine = _make_continuation_engine(case_dir, store=store, lookup=lookup)
    engine.queue_runs, engine.lose = True, 1
    ctx = _bind_engine(case_dir, monkeypatch, engine, acting=False, task_id=f"t-{name}")
    lost = json.loads(delegate._delegate_start(ctx, "seed queued", max_seconds=120).text)
    pred_iid = lost["pending_invocation_id"]
    custody._CUSTODY.clear()
    recovered = json.loads(delegate._delegate_start(ctx, "", retry_of=pred_iid).text)
    assert recovered["status"] == "started" and recovered["run_id"] == "job-1"
    job_id = recovered["run_id"]
    canonical_id = engine.bind_run(job_id, state="succeeded") if bind_canonical else ""
    _settle(case_dir, job_id, ctx.task_id, state="succeeded")
    assert current.run(case_dir, job_id).daemon_store_id == store
    if canonical_id:
        assert current.run(case_dir, canonical_id) is None
    return case_dir, engine, ctx, pred_iid, job_id, canonical_id


def test_pre_admission_continuation_superseded_proves_queued_job_to_canonical_run_alias_in_same_store(
        tmp_path, monkeypatch):
    from ouroboros import (
        delegate_custody_current as current,
        delegate_invocation as inv,
        delegate_pending as pending,
    )
    from ouroboros.gateways.claudexor_observation import canonical_request_digest
    from ouroboros.subagent_bootstrap import _startup_refusal_definite
    from ouroboros.tools import delegate

    case_dir, engine, ctx, pred_iid, job_id, canonical_id = _seed_queued_job_predecessor(
        tmp_path, monkeypatch, "job_alias_ok")
    assert canonical_id == "run-1"
    in_guard, body_reads_in_guard, guard_calls = [False], [], []
    real_body, real_guard = pending.request_body, continuation.can_settle_pre_admission_rejection

    def spy_body(*a, **k):
        if in_guard[0]:
            body_reads_in_guard.append(a)
        return real_body(*a, **k)

    def spy_guard(*a, **k):
        in_guard[0] = True
        try:
            ok = real_guard(*a, **k)
            guard_calls.append(ok)
            return ok
        finally:
            in_guard[0] = False

    monkeypatch.setattr(pending, "request_body", spy_body)
    monkeypatch.setattr(continuation, "can_settle_pre_admission_rejection", spy_guard)
    engine.create_refusal = (409, _continuation_superseded_problem(canonical_id, "run-other-head"))
    paths_before = len(engine.paths)
    refusal = json.loads(delegate._delegate_start(
        ctx, "continue queued predecessor", max_seconds=120, continue_from=job_id).text)
    cont_iid, _store, sent_body = engine.creates[-1]

    assert sent_body["continueFrom"] == job_id
    assert guard_calls == [True] and body_reads_in_guard == []
    assert not any(m == "GET" and p.startswith("/v2/runs/") for m, p in engine.paths[paths_before:])
    assert [key for _kind, key, _sent in engine.lookups[-2:]] == [cont_iid, pred_iid]
    assert refusal["ok"] is False and refusal["status"] == "refused"
    assert refusal["reason"] == continuation.REFUSAL_SUPERSEDED
    assert refusal["start_rejected_before_admission"] is True
    assert refusal["continuation_of"] == job_id and refusal["predecessor_canonical_run_id"] == canonical_id
    assert refusal["superseded_by_head"] == "run-other-head"
    assert "pending_invocation_id" not in refusal and "definitely_unrun" not in refusal
    assert _startup_refusal_definite(refusal) is False

    rows = custody.custody_rows(case_dir)
    failed_rows = [r for r in rows if r.get("type") == custody.START_FAILED and r.get("invocation_id") == cont_iid]
    assert len(failed_rows) == 1
    row = failed_rows[0]
    assert row["definite"] is True and row["pre_admission_rejected"] is True and row["start_sent"] is True
    assert row["continuation_of"] == job_id and row["predecessor_canonical_run_id"] == canonical_id
    assert row["superseded_by_head"] == "run-other-head" and row["daemon_store_id"] == "store-a"
    assert row["request_digest"] == canonical_request_digest(sent_body)
    custody._CUSTODY.clear()
    assert custody.invocation_record(case_dir, cont_iid)["state"] == "failed_definite"
    assert inv.invocation_record(case_dir, cont_iid, rows=rows)["state"] == "failed_definite"
    assert custody.pending_invocations(case_dir) == []
    assert f"invocation:{cont_iid}" not in current.fold(rows)["open"] and cont_iid in current.fold(rows)["refused"]
    creates_before_retry = len(engine.creates)
    retry = json.loads(delegate._delegate_start(ctx, "", retry_of=cont_iid).text)
    assert retry["ok"] is False and retry["reason"] == "invocation_definitely_refused"
    assert len(engine.creates) == creates_before_retry


def test_pre_admission_continuation_superseded_refuses_unproven_job_alias_cases(tmp_path, monkeypatch):
    from ouroboros import delegate_accepted, store
    from ouroboros.gateways.claudexor_observation import canonical_request_digest
    from ouroboros.tools import delegate

    def assert_stays_pending(case_dir, engine, refusal):
        iid = engine.creates[-1][0]
        assert refusal["ok"] is False and refusal["pending_invocation_id"] == iid
        assert "start_rejected_before_admission" not in refusal
        assert custody.invocation_record(case_dir, iid)["state"] == "pending"
        assert [r for r in custody.custody_rows(case_dir)
                if r.get("type") == custody.START_FAILED and r.get("invocation_id") == iid
                and r.get("definite") is True] == []

    # 1. Unrelated context.runId on a direct run-id predecessor stays unknown.
    unrel_dir, eng_unrel, ctx_unrel, pred_unrel = _seed_continued_engine(tmp_path, monkeypatch, "alias_unrelated")
    eng_unrel.create_refusal = (409, _continuation_superseded_problem("run-unrelated", "run-head-u"))
    res_unrel = json.loads(delegate._delegate_start(
        ctx_unrel, "continue", max_seconds=120, continue_from=pred_unrel).text)
    assert_stays_pending(unrel_dir, eng_unrel, res_unrel)

    # 2. Queued job still without a bound canonical runId (similar ID shape job-1 vs run-1) stays unknown.
    unbound_dir, eng_unbound, ctx_unbound, _piid, job_unbound, _ = _seed_queued_job_predecessor(
        tmp_path, monkeypatch, "alias_still_null", bind_canonical=False)
    eng_unbound.create_refusal = (409, _continuation_superseded_problem("run-1", "run-head-n"))
    res_unbound = json.loads(delegate._delegate_start(
        ctx_unbound, "continue", max_seconds=120, continue_from=job_unbound).text)
    assert_stays_pending(unbound_dir, eng_unbound, res_unbound)

    # 3. Queued job whose canonical runId is run-1, but 409 reports a different run-2, stays unknown.
    wrong_dir, eng_wrong, ctx_wrong, _piid, job_wrong, _canon = _seed_queued_job_predecessor(
        tmp_path, monkeypatch, "alias_wrong_canon")
    eng_wrong.create_refusal = (409, _continuation_superseded_problem("run-2", "run-head-w"))
    res_wrong = json.loads(delegate._delegate_start(
        ctx_wrong, "continue", max_seconds=120, continue_from=job_wrong).text)
    assert_stays_pending(wrong_dir, eng_wrong, res_wrong)

    # 4. Missing durable predecessor invocation record (hand-seeded row without START_REQUESTED) stays unknown.
    noinv_dir = tmp_path / "alias_no_inv"
    noinv_dir.mkdir()
    eng_noinv = _make_continuation_engine(noinv_dir)
    ctx_noinv = _bind_engine(noinv_dir, monkeypatch, eng_noinv, acting=False, task_id="t-noinv")
    _seed(noinv_dir, "job-seeded", task_id="t-noinv", root_task_id="t-root", daemon_store_id="store-a")
    eng_noinv.create_refusal = (409, _continuation_superseded_problem("run-seeded", "run-head-s"))
    res_noinv = json.loads(delegate._delegate_start(
        ctx_noinv, "continue", max_seconds=120, continue_from="job-seeded").text)
    assert_stays_pending(noinv_dir, eng_noinv, res_noinv)

    # 5. Predecessor lookup returns not_found while GET /v2/runs/job-1 would return 200 without storeId:
    #    never falls back to GET /v2/runs/:id 200 as alias proof.
    get200_dir, eng_get200, ctx_get200, piid_get200, job_get200, canon_get200 = _seed_queued_job_predecessor(
        tmp_path, monkeypatch, "alias_no_get200")
    eng_get200.lookup_gaps.add(piid_get200)
    eng_get200.create_refusal = (409, _continuation_superseded_problem(canon_get200, "run-head-g"))
    paths_before = len(eng_get200.paths)
    res_get200 = json.loads(delegate._delegate_start(
        ctx_get200, "continue", max_seconds=120, continue_from=job_get200).text)
    assert_stays_pending(get200_dir, eng_get200, res_get200)
    assert not any(m == "GET" and p.startswith("/v2/runs/") for m, p in eng_get200.paths[paths_before:])

    # 6. Descriptor rotates during the predecessor alias lookup -> post-lookup generation re-check rejects.
    rot_dir, eng_rot, ctx_rot, piid_rot, job_rot, canon_rot = _seed_queued_job_predecessor(
        tmp_path, monkeypatch, "alias_rotated_lookup")
    eng_rot.on_lookup = lambda key, _path: (
        pathlib.Path(eng_rot.endpoint.descriptor_path).write_text('{"rotated": true}', encoding="utf-8")
        if key == piid_rot else None
    )
    eng_rot.create_refusal = (409, _continuation_superseded_problem(canon_rot, "run-head-rot"))
    res_rot = json.loads(delegate._delegate_start(
        ctx_rot, "continue", max_seconds=120, continue_from=job_rot).text)
    assert_stays_pending(rot_dir, eng_rot, res_rot)

    # 7. Store mismatch or unbound store on predecessor custody never proves an alias.
    other_dir, eng_other, _ctx_other, _piid_other, job_other, canon_other = _seed_queued_job_predecessor(
        tmp_path, monkeypatch, "alias_other_store", store="store-b")
    store_a_dir = other_dir / "store_a_gw"
    store_a_dir.mkdir()
    eng_store_a = _make_continuation_engine(store_a_dir, store="store-a")
    gw_other, gw_store_a = eng_other.gateway(), eng_store_a.gateway()
    try:
        assert delegate_accepted.prove_job_run_alias(
            other_dir, gw_other, job_other, canon_other, store_id="store-a") is False
        assert delegate_accepted.prove_job_run_alias(
            other_dir, gw_store_a, job_other, canon_other, store_id="store-a") is False
        _seed(other_dir, "job-unbound-store", task_id="t-unbound", invocation_id="inv-unbound-store", daemon_store_id="")
        assert delegate_accepted.prove_job_run_alias(
            other_dir, gw_store_a, "job-unbound-store", "run-unbound-store", store_id="store-a") is False
    finally:
        gw_other.close()
        gw_store_a.close()

    # 8. Reducer guard rejects forged/missing predecessor custody or malformed predecessor_canonical_run_id.
    g_dir = tmp_path / "alias_guard"
    g_dir.mkdir()
    body = {"prompt": "continue", "continueFrom": "job-g", "access": "readonly", "mode": "ask"}
    assert custody.record_start_requested(
        g_dir, invocation_id="inv-ag", task_id="t-a", root_task_id="t-a", idempotency_key="k-ag",
        continuation_of="job-g", handshake_store_id="store-a", daemon_store_id="",
        store_identity_unproven=True, request=body)
    alias_event = {
        "type": custody.START_FAILED, "run_id": "", "task_id": "t-a", "root_task_id": "t-a",
        "invocation_id": "inv-ag", "idempotency_key": "k-ag", "wire_idempotency_key": "inv-ag",
        "reason": continuation.REFUSAL_SUPERSEDED, "status_code": 409, "definite": True,
        "pre_admission_rejected": True, "start_sent": True, "continuation_of": "job-g",
        "predecessor_canonical_run_id": "run-g", "superseded_by_head": "run-g-head",
        "daemon_store_id": "store-a", "request_digest": canonical_request_digest(body),
        "engine_version": _PROVEN_ENGINE_VERSION, "engine_build_sha": _PROVEN_ENGINE_SHA,
    }
    prepared = continuation.prepare_pre_admission_rejection_binding(g_dir, alias_event)
    assert prepared is not None
    assert continuation.prepare_pre_admission_rejection_binding(
        g_dir, {**alias_event, "predecessor_canonical_run_id": "job-g"}) is None
    assert continuation.prepare_pre_admission_rejection_binding(
        g_dir, {**alias_event, "predecessor_canonical_run_id": "run-g-head"}) is None
    assert continuation.prepare_pre_admission_rejection_binding(
        g_dir, {**alias_event, "predecessor_canonical_run_id": ""}) is None
    with store.open_store(g_dir, write=False) as txn:
        assert continuation.can_settle_pre_admission_rejection(txn, alias_event, verified=prepared) is False  # no pred row
    _seed(g_dir, "job-g", task_id="t-a", invocation_id="inv-pred-g", daemon_store_id="store-b")
    with store.open_store(g_dir, write=False) as txn:
        assert continuation.can_settle_pre_admission_rejection(txn, alias_event, verified=prepared) is False  # wrong store
    _seed(g_dir, "job-g-ok", task_id="t-a", invocation_id="inv-pred-ok", daemon_store_id="store-a")
    body_ok = {**body, "continueFrom": "job-g-ok"}
    assert custody.record_start_requested(
        g_dir, invocation_id="inv-ag-ok", task_id="t-a", root_task_id="t-a", idempotency_key="k-ag-ok",
        continuation_of="job-g-ok", handshake_store_id="store-a", daemon_store_id="",
        store_identity_unproven=True, request=body_ok)
    ok_event = {**alias_event, "invocation_id": "inv-ag-ok", "idempotency_key": "k-ag-ok",
                "wire_idempotency_key": "inv-ag-ok", "continuation_of": "job-g-ok",
                "request_digest": canonical_request_digest(body_ok)}
    ok_prepared = continuation.prepare_pre_admission_rejection_binding(g_dir, ok_event)
    assert ok_prepared is not None
    with store.open_store(g_dir, write=False) as txn:
        assert continuation.can_settle_pre_admission_rejection(txn, ok_event, verified=ok_prepared) is True
        assert continuation.can_settle_pre_admission_rejection(
            txn, {**ok_event, "predecessor_canonical_run_id": "run-drifted"}, verified=ok_prepared) is False
