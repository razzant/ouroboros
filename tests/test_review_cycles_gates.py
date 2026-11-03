"""Max-Review-Cycles semantics on the commit gate and skill review (Δ5/Δ6;
owner decisions Q12/Q16/Q17/Q22/Q23).

Contract under test:
* blocked commit attempts are TYPED (verdict vs infra) at record time;
* the paid fact is recorded at dispatch on the attempt ledger and survives the
  terminal merge; per-root-task counts are derived from it (P7);
* the free-cycle gate runs BEFORE advisory freshness: blocking → free typed
  refusal (managed resolvers get MERGE_HEAD repaired), advisory → free replay
  marker (typed event, commit proceeds, no paid dispatch);
* skill review: panel-contract fingerprint, per-key paid-cycle counting,
  $0 replay of an identical substantive verdict, typed exhaustion refusal,
  and the Max-Review-Cycles facts on terminal history rows.
"""

from __future__ import annotations

import json
import pathlib
import types

import pytest


REPO = pathlib.Path(__file__).resolve().parents[1]
KEY = "OUROBOROS_REVIEW_MAX_CYCLES"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(KEY, raising=False)
    monkeypatch.delenv("OUROBOROS_REVIEW_ENFORCEMENT", raising=False)
    yield


# ---------------------------------------------------------------------------
# Block-row classification at record time (Δ5.1)


def test_classify_review_block_verdict_vs_infra():
    from ouroboros.tools.commit_gate import classify_review_block

    # Triad critical findings = verdict.
    assert classify_review_block(
        triad_blocked=True, triad_block_reason="critical_findings",
        scope_blocked=False,
    ) == "verdict"
    # Triad fit/quorum/transport = infra.
    for reason in ("fixed_overflow", "review_quorum", "infra_failure", "parse_failure"):
        assert classify_review_block(
            triad_blocked=True, triad_block_reason=reason, scope_blocked=False,
        ) == "infra"
    # Scope: a RESPONDED actor row that blocked on critical findings = verdict …
    responded = {"raw_results": [
        {"status": "responded", "critical_findings": [{"item": "x"}]},
    ]}
    assert classify_review_block(
        triad_blocked=False, triad_block_reason="", scope_blocked=True,
        scope_raw_result=responded,
    ) == "verdict"
    # … while sub-floor / overflow / error scope blocks are infra.
    for status in ("sub_floor", "fixed_overflow", "error", "empty_response", "parse_failure"):
        infra_rows = {"raw_results": [{"status": status, "critical_findings": []}]}
        assert classify_review_block(
            triad_blocked=False, triad_block_reason="", scope_blocked=True,
            scope_raw_result=infra_rows,
        ) == "infra"
    # A verdict on EITHER side wins even when the other side infra-blocked.
    assert classify_review_block(
        triad_blocked=True, triad_block_reason="review_quorum",
        scope_blocked=True, scope_raw_result=responded,
    ) == "verdict"


def test_finalize_blocked_review_records_block_class(tmp_path, monkeypatch):
    """git.py's blocked finalizer stamps the typed class on the durable row."""
    import ouroboros.tools.git as git_mod
    from ouroboros.review_state import load_state, make_repo_key

    monkeypatch.setattr(git_mod, "run_cmd", lambda *a, **k: "")
    ctx = types.SimpleNamespace(
        repo_dir=tmp_path, drive_root=tmp_path, task_id="t1", task_metadata={},
        _current_review_tool_name="commit_reviewed",
        _last_triad_models=[], _last_scope_model="",
        _last_triad_raw_results=[], _last_scope_raw_result={},
        _review_degraded_reasons=[],
    )
    git_mod._finalize_blocked_review(
        ctx, "msg", 0.0,
        combined_msg="blocked", block_reason="critical_findings",
        combined_findings=[{"item": "x", "reason": "r", "severity": "critical"}],
        pre_fingerprint={"fingerprint": "fp"}, post_fingerprint={"fingerprint": "fp"},
        block_class="verdict",
    )
    state = load_state(pathlib.Path(tmp_path))
    rows = state.filter_attempts(repo_key=make_repo_key(pathlib.Path(tmp_path)))
    assert rows and rows[-1].block_class == "verdict"
    assert rows[-1].root_task_id == "t1"  # ctx.task_id is its own root


# ---------------------------------------------------------------------------
# Paid-at-dispatch on the ledger (Δ5 paid + merge semantics, Q23 root scoping)


def test_paid_fact_survives_terminal_merge_and_roundtrip(tmp_path):
    from ouroboros.review_state import (
        CommitAttemptRecord,
        load_state,
        make_repo_key,
        update_state,
        _utc_now,
    )

    repo_key = make_repo_key(pathlib.Path(tmp_path))

    def _record(state, **kw):
        base = dict(
            ts=_utc_now(), commit_message="m", repo_key=repo_key,
            tool_name="commit_reviewed", task_id="t1", attempt=1,
        )
        base.update(kw)
        state.record_attempt(CommitAttemptRecord(**base))

    # Dispatch-time row: paid recorded on the in-flight marker.
    update_state(pathlib.Path(tmp_path), lambda s: _record(
        s, status="reviewing", phase="review", paid=True,
        rebuttal_sha256="r" * 64, review_contract_fingerprint="c" * 64,
        root_task_id="root-1", pre_review_fingerprint="fp-1",
    ))
    # Terminal update on the SAME attempt does not pass paid — the merge must
    # keep it (a terminal row must never launder the spend).
    update_state(pathlib.Path(tmp_path), lambda s: _record(
        s, status="blocked", phase="blocking_review",
        block_reason="critical_findings", block_class="verdict",
        pre_review_fingerprint="fp-1",
    ))
    state = load_state(pathlib.Path(tmp_path))
    row = state.filter_attempts(repo_key=repo_key)[-1]
    assert row.status == "blocked" and row.paid is True
    assert row.block_class == "verdict"
    assert row.rebuttal_sha256 == "r" * 64
    assert row.review_contract_fingerprint == "c" * 64
    assert row.root_task_id == "root-1"
    # And all of it survives the JSON roundtrip (save happened in update_state).
    reloaded = load_state(pathlib.Path(tmp_path)).filter_attempts(repo_key=repo_key)[-1]
    assert (reloaded.paid, reloaded.block_class, reloaded.root_task_id) == (True, "verdict", "root-1")


def test_record_commit_attempt_reviewing_lands_paid_on_the_ledger(tmp_path):
    """Behavior test for the dispatch seam (wording-1): driving the REAL
    recorder with the reviewing-row kwargs must land paid/rebuttal/contract/
    root facts on the durable ledger, and the ceiling counter must see them."""
    from ouroboros.review_state import load_state, make_repo_key
    from ouroboros.tools.commit_gate import (
        _record_commit_attempt,
        count_paid_review_cycles,
    )

    ctx = types.SimpleNamespace(
        repo_dir=tmp_path, drive_root=tmp_path, task_id="t-1",
        task_metadata={"root_task_id": "root-9"},
        _current_review_tool_name="commit_reviewed",
    )
    _record_commit_attempt(
        ctx, "msg", "reviewing",
        duration_sec=0.0, phase="review",
        pre_review_fingerprint="fp-1", fingerprint_status="pending",
        paid=True, rebuttal_sha256="r" * 64, review_contract_fingerprint="c" * 64,
    )
    row = load_state(pathlib.Path(tmp_path)).filter_attempts(
        repo_key=make_repo_key(pathlib.Path(tmp_path)))[-1]
    assert row.status == "reviewing" and row.paid is True
    assert row.rebuttal_sha256 == "r" * 64
    assert row.review_contract_fingerprint == "c" * 64
    assert row.root_task_id == "root-9"
    assert count_paid_review_cycles(ctx, root_task_id="root-9") == 1


def test_stage_cycle_free_gate_runs_before_preflight_gate():
    """Order pin: the free Max-Review-Cycles gate precedes the preflight/tests
    gate, which precedes the paid dispatch."""
    import inspect

    from ouroboros.tools import git_review_cycle

    source = inspect.getsource(git_review_cycle._reviewed_stage_cycle)  # the cycle body under the panel
    assert source.index("_free_cycle_gate(") < source.index("_preflight_and_tests_gate(")
    assert source.index("_preflight_and_tests_gate(") < source.index("_run_parallel_review(")


def test_resolve_root_task_id_ignores_the_followup_chain():
    """machine-3 decision: a follow-up task is a FRESH root for the ceiling —
    origin_root_task_id (the follow-up chain marker) is deliberately not
    honored; the identical-fingerprint refusal stays the anti-laundering
    backstop."""
    from ouroboros.tools.commit_gate import resolve_root_task_id

    followup = types.SimpleNamespace(
        task_metadata={"origin_root_task_id": "old-root", "origin_task_id": "old-task"},
        task_id="f-1",
    )
    assert resolve_root_task_id(followup) == "f-1"
    subtask = types.SimpleNamespace(
        task_metadata={"root_task_id": "tree-root", "origin_root_task_id": "old-root"},
        task_id="child-1",
    )
    assert resolve_root_task_id(subtask) == "tree-root"


def test_repair_managed_merge_head_writes_merge_head(tmp_path, monkeypatch):
    """Behavior test (wording-1): the refusal path's repair helper re-writes
    the REAL .git/MERGE_HEAD via reestablish_merge_head with the tx target."""
    import subprocess

    import ouroboros.tools.git as git_mod
    import supervisor.git_ops as git_ops
    import supervisor.update_merge as update_merge

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    target = "a" * 40
    monkeypatch.setattr(
        update_merge, "managed_assisted_tx_for",
        lambda task_id, metadata: ({"target_sha": target}, ""),
    )
    ctx = types.SimpleNamespace(task_id="t-1", task_metadata={})
    git_mod._repair_managed_merge_head(ctx)
    assert (repo / ".git" / "MERGE_HEAD").read_text(encoding="utf-8") == target + "\n"


# ---------------------------------------------------------------------------
# The free-cycle gate (Δ5.2/Δ5.4): blocking refusals, advisory replay


def _gate_ctx(tmp_path, task_id="root-1"):
    (pathlib.Path(tmp_path) / "logs").mkdir(parents=True, exist_ok=True)
    return types.SimpleNamespace(
        repo_dir=tmp_path, drive_root=pathlib.Path(tmp_path), task_id=task_id,
        task_metadata={}, event_queue=None,
        drive_logs=lambda: pathlib.Path(tmp_path) / "logs",
        _current_review_tool_name="commit_reviewed",
    )


def _seed_verdict_block(tmp_path, fingerprint, contract_fp, **kw):
    from ouroboros.review_state import CommitAttemptRecord, make_repo_key, update_state, _utc_now

    repo_key = make_repo_key(pathlib.Path(tmp_path))
    row = dict(
        ts=_utc_now(), commit_message="m", status="blocked",
        block_reason="critical_findings", block_class="verdict",
        repo_key=repo_key, tool_name="commit_reviewed", task_id="root-1",
        attempt=1, phase="blocking_review", pre_review_fingerprint=fingerprint,
        review_contract_fingerprint=contract_fp,
        critical_findings=[{"item": "bug_y", "reason": "broken", "severity": "critical"}],
    )
    row.update(kw)
    update_state(pathlib.Path(tmp_path), lambda s: s.attempts.append(CommitAttemptRecord(**row)))


def test_free_cycle_gate_blocking_identical_refusal_repairs_managed_merge_head(
    tmp_path, monkeypatch,
):
    import ouroboros.tools.git as git_mod

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setattr(git_mod, "commit_review_contract_fingerprint", lambda: "cf-1")
    monkeypatch.setattr(git_mod, "run_cmd", lambda *a, **k: "")
    repaired = []
    monkeypatch.setattr(git_mod, "_authorized_managed_update_resolver", lambda ctx: True)
    monkeypatch.setattr(git_mod, "_repair_managed_merge_head", lambda ctx: repaired.append(True))

    ctx = _gate_ctx(tmp_path)
    _seed_verdict_block(tmp_path, "fp-1", "cf-1")
    outcome = git_mod._free_cycle_gate(
        ctx, "msg", 0.0, pre_fingerprint={"fingerprint": "fp-1"}, review_rebuttal="",
    )
    assert outcome is not None and outcome["status"] == "blocked"
    assert outcome["block_reason"] == "identical_diff_refused"
    assert "IDENTICAL_DIFF_REFUSED" in outcome["message"] and "bug_y" in outcome["message"]
    assert repaired == [True]  # MERGE_HEAD kept repaired on the free refusal path
    # The refusal itself is durably recorded and never resets the streak.
    from ouroboros.review_state import load_state, make_repo_key

    rows = load_state(pathlib.Path(tmp_path)).filter_attempts(
        repo_key=make_repo_key(pathlib.Path(tmp_path)))
    assert rows[-1].block_reason == "identical_diff_refused"
    again = git_mod._free_cycle_gate(
        ctx, "msg", 0.0, pre_fingerprint={"fingerprint": "fp-1"}, review_rebuttal="",
    )
    assert again is not None and again["block_reason"] == "identical_diff_refused"
    # A fresh diff passes the gate.
    assert git_mod._free_cycle_gate(
        ctx, "msg", 0.0, pre_fingerprint={"fingerprint": "fp-2"}, review_rebuttal="",
    ) is None


def test_free_cycle_gate_blocking_ceiling_emits_typed_event(tmp_path, monkeypatch):
    import ouroboros.tools.git as git_mod
    from ouroboros.review_state import CommitAttemptRecord, make_repo_key, update_state, _utc_now

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setenv(KEY, "1")
    monkeypatch.setattr(git_mod, "commit_review_contract_fingerprint", lambda: "cf-1")
    monkeypatch.setattr(git_mod, "run_cmd", lambda *a, **k: "")
    monkeypatch.setattr(git_mod, "_authorized_managed_update_resolver", lambda ctx: False)

    ctx = _gate_ctx(tmp_path)
    repo_key = make_repo_key(pathlib.Path(tmp_path))
    update_state(pathlib.Path(tmp_path), lambda s: s.attempts.append(CommitAttemptRecord(
        ts=_utc_now(), commit_message="m", status="succeeded", repo_key=repo_key,
        tool_name="commit_reviewed", task_id="root-1", attempt=1, phase="commit",
        paid=True, root_task_id="root-1", pre_review_fingerprint="fp-old",
    )))
    outcome = git_mod._free_cycle_gate(
        ctx, "msg", 0.0, pre_fingerprint={"fingerprint": "fp-new"}, review_rebuttal="",
    )
    assert outcome is not None and outcome["block_reason"] == "review_cycles_exhausted"
    assert "REVIEW_CYCLES_EXHAUSTED" in outcome["message"]
    events = [json.loads(line) for line in
              (pathlib.Path(tmp_path) / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    exhausted = [e for e in events if e["type"] == "review_cycles_exhausted"]
    assert exhausted and exhausted[0]["surface"] == "commit_gate"
    assert exhausted[0]["cycles_paid"] == 1 and exhausted[0]["cap"] == 1
    assert exhausted[0]["enforcement"] == "blocking"


def test_free_cycle_gate_advisory_returns_replay_marker_not_a_block(tmp_path, monkeypatch):
    """A.4/A.2 advisory contract: exhaustion/identical-refusal do NOT hard-block
    — the gate returns a replay marker (typed free-replay event), never a
    blocked outcome, and dispatches nothing."""
    import ouroboros.tools.git as git_mod

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    monkeypatch.setattr(git_mod, "commit_review_contract_fingerprint", lambda: "cf-1")
    monkeypatch.setattr(
        git_mod, "run_cmd",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no git calls on the advisory replay path")),
    )
    ctx = _gate_ctx(tmp_path)
    _seed_verdict_block(tmp_path, "fp-1", "cf-1")
    outcome = git_mod._free_cycle_gate(
        ctx, "msg", 0.0, pre_fingerprint={"fingerprint": "fp-1"}, review_rebuttal="",
    )
    assert outcome is not None and "status" not in outcome
    assert outcome["replay_reason"] == "identical_diff_refused"
    assert "IDENTICAL_DIFF_REFUSED" in outcome["advisory_replay"]
    events = [json.loads(line) for line in
              (pathlib.Path(tmp_path) / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    replays = [e for e in events if e["type"] == "commit_review_free_replay"]
    assert replays and replays[0]["reason"] == "identical_diff_refused"
    assert replays[0]["enforcement"] == "advisory"


def test_free_cycle_gate_infra_streak_never_gates_advisory_or_blocking(tmp_path, monkeypatch):
    """Infra-blocks retry freely: they neither refuse under blocking nor
    trigger the advisory replay."""
    import ouroboros.tools.git as git_mod

    monkeypatch.setattr(git_mod, "commit_review_contract_fingerprint", lambda: "cf-1")
    monkeypatch.setattr(git_mod, "run_cmd", lambda *a, **k: "")
    ctx = _gate_ctx(tmp_path)
    _seed_verdict_block(
        tmp_path, "fp-1", "cf-1",
        block_reason="review_quorum", block_class="infra", critical_findings=[],
    )
    for enforcement in ("blocking", "advisory"):
        monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
        assert git_mod._free_cycle_gate(
            ctx, "msg", 0.0, pre_fingerprint={"fingerprint": "fp-1"}, review_rebuttal="",
        ) is None


def test_commit_contract_fingerprint_tracks_resolved_review_efforts(monkeypatch):
    """Synthesis F4 (commit side): the commit fingerprint's pool rows carry their
    RESOLVED efforts, so a pool row's effort change lapses refusal/replay. PR-3:
    the pool reads no global review effort — a row with no effort of its own
    reviews at the pool default — so only the ROW's value moves the fingerprint.
    (The scope-review effort leg was the lane's; the fold that read it is gone, I3-B2.)"""
    import json

    from ouroboros.tools.commit_gate import commit_review_contract_fingerprint

    def _catalog(effort):
        return json.dumps({"enabled": True, "items": [{
            "subagent_id": "critic", "recommended_use": "Reviewer.",
            "route": {"kind": "api_model", "target_id": "openai/gpt-5.6-terra"},
            "effort": effort, "review_eligible": True,
        }]})

    monkeypatch.setenv("OUROBOROS_SUBAGENTS", _catalog("high"))
    monkeypatch.setenv("OUROBOROS_EFFORT_REVIEW", "high")
    base = commit_review_contract_fingerprint()
    assert base and len(base) == 64
    monkeypatch.setenv("OUROBOROS_EFFORT_REVIEW", "low")
    assert base == commit_review_contract_fingerprint()  # the lane-era global effort is not a pool fact
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", _catalog("low"))
    assert base != commit_review_contract_fingerprint()
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", _catalog("high"))
    assert base == commit_review_contract_fingerprint()
    monkeypatch.setenv("OUROBOROS_EFFORT_SCOPE_REVIEW", "low")
    assert base == commit_review_contract_fingerprint()  # the retired scope-lane effort is not a pool fact either
