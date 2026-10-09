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


# ---------------------------------------------------------------------------
# Skill review (Δ6, Q17/Q23)


def test_skill_review_contract_fingerprint_tracks_roster_items_and_profile(monkeypatch):
    from ouroboros.skill_review_cycles import skill_review_contract_fingerprint

    monkeypatch.setenv("OUROBOROS_EFFORT_REVIEW", "high")
    base = skill_review_contract_fingerprint(["m1", "m2"], required_items=("a", "b"))
    assert base and len(base) == 64
    assert base == skill_review_contract_fingerprint(["m1", "m2"], required_items=("a", "b"))
    assert base != skill_review_contract_fingerprint(["m1"], required_items=("a", "b"))
    assert base != skill_review_contract_fingerprint(["m1", "m2"], required_items=("a",))
    # skill-4: the resolved review profile is contract identity — official_hub
    # aggregates blockers differently, so a profile change lapses replay.
    assert base != skill_review_contract_fingerprint(
        ["m1", "m2"], required_items=("a", "b"), review_profile="official_hub")
    # The legacy identity's effort is the pool default; the lane-era global
    # OUROBOROS_EFFORT_REVIEW is retired, and an exported one is inert here too
    # (a structured panel carries each row's own effort in its row identity).
    monkeypatch.setenv("OUROBOROS_EFFORT_REVIEW", "low")
    assert base == skill_review_contract_fingerprint(["m1", "m2"], required_items=("a", "b"))


def test_skill_review_contract_fingerprint_preserves_legacy_and_tracks_rows(monkeypatch):
    import ouroboros.skill_review_cycles as cycles
    from ouroboros.skill_review_cycles import skill_review_contract_fingerprint

    monkeypatch.setenv("OUROBOROS_EFFORT_REVIEW", "high")
    legacy = skill_review_contract_fingerprint(["m1", "m2"], required_items=("a",))
    # The author-finality contract is part of the skill-review prompt contract;
    # its deliberate wording change invalidates the old fingerprint. Decision 3A
    # then removed the advisory-evidence input from the prompt builder (no
    # advisory critic feeds the skill reviewer), which moved it once more.
    assert legacy == "2401d2c34e3805f5ba6dcd933a2abf4d44c3af88dc5d140bddc24018cfebfca7"
    legacy_delivery = {
        "legacy_skill_fingerprint": True,
        "models": ["m1", "m2"], "routes": ["api_chat", "api_chat"],
        "efforts": ["high", "high"], "session_targets": ["", ""],
        "session_profiles": ["", ""], "slot_ids": ["slot_1", "slot_2"],
    }
    assert legacy == skill_review_contract_fingerprint(
        ["m1", "m2"], required_items=("a",), delivery=legacy_delivery,
    )

    structured = dict(legacy_delivery, legacy_skill_fingerprint=False)
    structured_fp = skill_review_contract_fingerprint(
        ["m1", "m2"], required_items=("a",), delivery=structured,
    )
    assert structured_fp != legacy
    with monkeypatch.context() as contract_patch:
        contract_patch.setattr(cycles, "_skill_prompt_contract_hash", lambda: "api-contract")
        contract_patch.setattr(
            "ouroboros.skill_review_passes.skill_review_session_contract_hash",
            lambda: "session-contract-a",
        )
        api_only_fp = skill_review_contract_fingerprint(
            ["m1", "m2"], required_items=("a",), delivery=structured,
        )
        contract_patch.setattr(
            "ouroboros.skill_review_passes.skill_review_session_contract_hash",
            lambda: "session-contract-b",
        )
        assert api_only_fp == skill_review_contract_fingerprint(
            ["m1", "m2"], required_items=("a",), delivery=structured,
        )
        session_delivery = dict(structured, routes=["agent_session", "api_chat"])
        session_b = skill_review_contract_fingerprint(
            ["m1", "m2"], required_items=("a",), delivery=session_delivery,
        )
        contract_patch.setattr(
            "ouroboros.skill_review_passes.skill_review_session_contract_hash",
            lambda: "session-contract-c",
        )
        assert session_b != skill_review_contract_fingerprint(
            ["m1", "m2"], required_items=("a",), delivery=session_delivery,
        )
    reordered = {
        key: ([value[1], value[0]] if isinstance(value, list) else value)
        for key, value in structured.items()
    }
    assert structured_fp == skill_review_contract_fingerprint(
        ["m2", "m1"], required_items=("a",), delivery=reordered,
    )
    changes = (
        {"models": ["different-model", "m2"]},
        {"routes": ["agent_session", "api_chat"]},
        {"efforts": ["xhigh", "high"]},
        {"session_targets": ["codex=different-model", ""]},
        {"session_profiles": ["profile-a", ""]},
        {"slot_ids": ["renamed-slot", "slot_2"]},
    )
    for delta in changes:
        changed = dict(structured, **delta)
        assert structured_fp != skill_review_contract_fingerprint(
            ["m1", "m2"], required_items=("a",), delivery=changed,
        ), delta


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


def _write_history(drive_root, skill, rows):
    from ouroboros.skill_review_history import review_history_path

    path = review_history_path(pathlib.Path(drive_root), skill)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_count_paid_skill_review_cycles_task_and_manual_keys(tmp_path):
    """Q23 + machine-5 + machine-1: a task-driven group counts every PAID
    (dispatched) cycle across EVERY skill the root task reviewed — including
    dispatched-then-degraded waves; the manual lane is scoped per
    content_hash (a revised snapshot restarts its count); unpaid/legacy rows
    never count."""
    from ouroboros.skill_review_cycles import count_paid_skill_review_cycles

    _write_history(tmp_path, "skill_a", [
        {"ts": "t1", "status": "blockers", "content_hash": "h1", "paid": True,
         "group_id": "task:root-9:skill_a", "root_task_id": "root-9", "job_id": "j1"},
        {"ts": "t2", "status": "interrupted", "content_hash": "h1", "paid": True,
         "group_id": "task:root-9:skill_a", "root_task_id": "root-9", "job_id": "j2"},
        {"ts": "t3", "status": "clean", "content_hash": "h2",  # legacy: no paid fact
         "group_id": "task:root-9:skill_a", "root_task_id": "root-9", "job_id": "j3"},
    ])
    _write_history(tmp_path, "skill_b", [
        {"ts": "t4", "status": "warnings", "content_hash": "h3", "paid": True,
         "group_id": "task:root-9:skill_b", "root_task_id": "root-9", "job_id": "j4"},
        {"ts": "t5", "status": "clean", "content_hash": "h4", "paid": True,
         "group_id": "manual:skill_b", "job_id": "j5"},
        {"ts": "t6", "status": "pending", "content_hash": "h4", "paid": True,  # quorum-failed wave: money
         "group_id": "manual:skill_b", "job_id": "j6"},
    ])
    # Task key = money across the tree: blockers(a) + interrupted-dispatched(a)
    # + warnings(b) = 3; legacy (no paid) and the manual lane excluded.
    assert count_paid_skill_review_cycles(pathlib.Path(tmp_path), "skill_a", "task:root-9:skill_a") == 3
    assert count_paid_skill_review_cycles(pathlib.Path(tmp_path), "skill_b", "task:root-9:skill_b") == 3
    # Manual key: per content snapshot — h4 spent 2 (clean + quorum-failed wave),
    # a revised snapshot h5 starts fresh at 0.
    assert count_paid_skill_review_cycles(
        pathlib.Path(tmp_path), "skill_b", "manual:skill_b", content_hash="h4") == 2
    assert count_paid_skill_review_cycles(
        pathlib.Path(tmp_path), "skill_b", "manual:skill_b", content_hash="h5") == 0


def test_skill_free_replay_row_rules(tmp_path):
    """Q17-A: replay needs the identical (group, content-hash, contract) triple
    with a SUBSTANTIVE verdict; infra terminals are skipped; a legacy row
    without a fingerprint never replays; a NEW rebuttal buys a paid rerun and
    a repeated one replays free."""
    from ouroboros.skill_review_cycles import find_free_replay_row

    group = "manual:demo"
    _write_history(tmp_path, "demo", [
        {"ts": "t1", "status": "warnings", "content_hash": "h1", "paid": True,
         "group_id": group, "review_contract_fingerprint": "cf-1",
         "rebuttal_sha256": "reb-1", "job_id": "j1",
         "fail_findings": [{"item": "bug_z", "severity": "warning", "reason_excerpt": "meh"}]},
        {"ts": "t2", "status": "timeout", "content_hash": "h1", "paid": True,
         "group_id": group, "review_contract_fingerprint": "cf-1", "job_id": "j2"},
    ])
    row = find_free_replay_row(
        pathlib.Path(tmp_path), "demo", group_id=group, content_hash="h1",
        contract_fingerprint="cf-1",
    )
    assert row is not None and row["status"] == "warnings"  # timeout skipped, verdict quoted
    # Different content hash / contract / group: paid run due.
    assert find_free_replay_row(pathlib.Path(tmp_path), "demo", group_id=group,
                                content_hash="h2", contract_fingerprint="cf-1") is None
    assert find_free_replay_row(pathlib.Path(tmp_path), "demo", group_id=group,
                                content_hash="h1", contract_fingerprint="cf-2") is None
    assert find_free_replay_row(pathlib.Path(tmp_path), "demo", group_id="manual:other",
                                content_hash="h1", contract_fingerprint="cf-1") is None
    # Rebuttal by content: new hash = paid rerun; the spent hash replays free.
    assert find_free_replay_row(pathlib.Path(tmp_path), "demo", group_id=group,
                                content_hash="h1", contract_fingerprint="cf-1",
                                rebuttal_sha256="reb-NEW") is None
    assert find_free_replay_row(pathlib.Path(tmp_path), "demo", group_id=group,
                                content_hash="h1", contract_fingerprint="cf-1",
                                rebuttal_sha256="reb-1") is not None
    # machine-6/skill-3: a rebuttal recorded only on an INFRA terminal (its
    # paid rerun quorum-failed/timed out) was never answered by a substantive
    # verdict — it stays fresh and buys the paid rerun it is owed.
    _write_history(tmp_path, "demo", [
        {"ts": "t3", "status": "timeout", "content_hash": "h1", "paid": True,
         "group_id": group, "review_contract_fingerprint": "cf-1",
         "rebuttal_sha256": "reb-infra", "job_id": "j3"},
    ])
    assert find_free_replay_row(pathlib.Path(tmp_path), "demo", group_id=group,
                                content_hash="h1", contract_fingerprint="cf-1",
                                rebuttal_sha256="reb-infra") is None
    # A legacy substantive row WITHOUT a fingerprint never replays.
    _write_history(tmp_path, "legacy", [
        {"ts": "t1", "status": "clean", "content_hash": "h1", "paid": True,
         "group_id": "manual:legacy", "job_id": "j1"},
    ])
    assert find_free_replay_row(pathlib.Path(tmp_path), "legacy", group_id="manual:legacy",
                                content_hash="h1", contract_fingerprint="cf-1") is None


def test_skill_free_replay_outcome_quotes_verdict_and_is_unpaid(tmp_path):
    from ouroboros.skill_review_cycles import free_replay_outcome

    group = "manual:demo"
    _write_history(tmp_path, "demo", [
        {"ts": "2026-08-20T00:00:00Z", "status": "blockers", "content_hash": "h1",
         "paid": True, "group_id": group, "review_contract_fingerprint": "cf-1",
         "job_id": "j1",
         "fail_findings": [{"item": "no_repo_mutation", "severity": "critical",
                            "reason_excerpt": "writes into repo/"}]},
    ])
    review_state = types.SimpleNamespace(
        content_hash="h1", status="blockers",
        findings=[{"item": "no_repo_mutation", "verdict": "FAIL", "severity": "critical",
                   "reason": "writes into repo/"}],
        reviewer_models=["m1", "m2"],
    )
    skill = types.SimpleNamespace(name="demo", review=review_state)
    outcome = free_replay_outcome(
        skill, drive_root=pathlib.Path(tmp_path), group_id=group, content_hash="h1",
        contract_fingerprint="cf-1",
    )
    assert outcome is not None
    assert outcome.status == "blockers" and outcome.paid is False
    assert outcome.replayed_from_ts == "2026-08-20T00:00:00Z"
    assert outcome.findings == review_state.findings  # full findings reused from state
    assert "FREE REPLAY" in outcome.convergence_hint
    assert "no_repo_mutation" in outcome.convergence_hint
    # No recorded verdict → no replay.
    assert free_replay_outcome(
        skill, drive_root=pathlib.Path(tmp_path), group_id=group, content_hash="h2",
        contract_fingerprint="cf-1",
    ) is None
    # skill-2: when the mutable persisted state (review.json) has DIVERGED from
    # the history row, the replay falls through to a PAID rerun that
    # re-persists — never a $0 replay whose effects the grant/enable gates
    # cannot see.
    diverged = types.SimpleNamespace(
        name="demo",
        review=types.SimpleNamespace(
            content_hash="OTHER", status="clean", findings=[], reviewer_models=[]),
    )
    assert free_replay_outcome(
        diverged, drive_root=pathlib.Path(tmp_path), group_id=group, content_hash="h1",
        contract_fingerprint="cf-1",
    ) is None
    stateless = types.SimpleNamespace(name="demo", review=None)
    assert free_replay_outcome(
        stateless, drive_root=pathlib.Path(tmp_path), group_id=group, content_hash="h1",
        contract_fingerprint="cf-1",
    ) is None


def test_skill_review_cycles_refusal_emits_typed_event(tmp_path, monkeypatch):
    from ouroboros.skill_review_cycles import skill_review_cycles_refusal

    monkeypatch.setenv(KEY, "1")
    group = "task:root-7:demo"
    _write_history(tmp_path, "demo", [
        {"ts": "t1", "status": "clean", "content_hash": "h1", "paid": True,
         "group_id": group, "root_task_id": "root-7", "job_id": "j1"},
    ])
    ctx = types.SimpleNamespace(task_id="t-77", event_queue=None)
    outcome = skill_review_cycles_refusal(
        ctx, "demo", drive_root=pathlib.Path(tmp_path), group_id=group,
        models=["m1"], content_hash="h2", contract_fingerprint="cf-1",
    )
    assert outcome is not None and outcome.status == "pending"
    assert "REVIEW_CYCLES_EXHAUSTED" in outcome.error
    assert outcome.paid is False
    events = [json.loads(line) for line in
              (pathlib.Path(tmp_path) / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    exhausted = [e for e in events if e["type"] == "review_cycles_exhausted"]
    assert exhausted and exhausted[0]["surface"] == "skill_review"
    assert exhausted[0]["root_task_id"] == "root-7" and exhausted[0]["cap"] == 1
    # Under the cap (another key) nothing is refused.
    assert skill_review_cycles_refusal(
        ctx, "demo", drive_root=pathlib.Path(tmp_path), group_id="manual:demo",
        models=["m1"], content_hash="h2", contract_fingerprint="cf-1",
    ) is None
    # Unlimited: never a ceiling.
    monkeypatch.setenv(KEY, "unlimited")
    assert skill_review_cycles_refusal(
        ctx, "demo", drive_root=pathlib.Path(tmp_path), group_id=group,
        models=["m1"], content_hash="h2", contract_fingerprint="cf-1",
    ) is None


@pytest.mark.parametrize("enforcement,mode,author_finish", [
    ("advisory", "advanced", True), ("blocking", "advanced", False),
    ("advisory", "cyber_pro", True), ("blocking", "cyber_pro", True),
])
def test_skill_review_cycles_refusal_names_the_author_finish_where_it_is_legal(
    tmp_path, monkeypatch, enforcement, mode, author_finish,
):
    from ouroboros.skill_review_cycles import skill_review_cycles_refusal

    monkeypatch.setenv(KEY, "1")
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: mode)
    group = "task:root-7:demo"
    _write_history(tmp_path, "demo", [
        {"ts": "t1", "status": "blockers", "content_hash": "h1", "paid": True,
         "group_id": group, "root_task_id": "root-7", "job_id": "j1"},
    ])
    outcome = skill_review_cycles_refusal(
        types.SimpleNamespace(task_id="t-77", event_queue=None), "demo",
        drive_root=tmp_path, group_id=group, models=["m1"],
        content_hash="h2", contract_fingerprint="cf-1",
    )
    assert outcome.status == "pending" and outcome.paid is False
    assert "REVIEW_CYCLES_EXHAUSTED" in outcome.error
    assert "finalize and disclose" in outcome.error
    assert "ask the owner" in outcome.error and "fresh task" in outcome.error
    assert ("author_disposition" in outcome.error) == author_finish
    assert ("author_rationale" in outcome.error) == author_finish
    if author_finish:
        assert "Ordinary Advisory requires prior reviewer feedback" in outcome.error
        assert "passing current preflight" in outcome.error


def test_terminal_history_payload_carries_cycle_facts():
    from ouroboros.skill_review_runner import _terminal_history_payload

    result = types.SimpleNamespace(
        findings=[], content_hash="h1", raw_actor_records=[],
        single_reviewer_no_diversity=False,
        paid=True, review_contract_fingerprint="cf-1",
        rebuttal_sha256="reb-1", replayed_from_ts="",
    )
    payload = _terminal_history_payload(
        {"job_id": "j1", "group_id": "manual:demo"},
        status="clean", terminal_reason="clean", result=result, ts="t1",
    )
    assert payload["paid"] is True
    assert payload["review_contract_fingerprint"] == "cf-1"
    assert payload["rebuttal_sha256"] == "reb-1"
    assert "replayed_from_ts" not in payload  # empty facts are not written
    # A replayed outcome discloses its provenance and stays unpaid.
    replayed = types.SimpleNamespace(
        findings=[], content_hash="h1", raw_actor_records=[],
        single_reviewer_no_diversity=False,
        paid=False, review_contract_fingerprint="cf-1",
        rebuttal_sha256="", replayed_from_ts="t0",
    )
    payload = _terminal_history_payload(
        {"job_id": "j2", "group_id": "manual:demo"},
        status="clean", terminal_reason="clean", result=replayed, ts="t2",
    )
    assert "paid" not in payload and payload["replayed_from_ts"] == "t0"


def _fake_manifest():
    return types.SimpleNamespace(
        name="demo", description="d", version="1", type="tool", runtime="python",
        timeout_sec=30, permissions=[], conflicts=[], env_from_settings=[],
        requires=[], scripts=[], scheduled_tasks=[], entry="main.py",
    )


def _wire_review_skill(
    monkeypatch, tmp_path, *, content_hash, review_state, passes, delivery=None,
):
    """Wire review_skill's heavy collaborators to a temp drive so the REAL
    pipeline (cycles gate → budget → panel seam) executes end to end."""
    import ouroboros.skill_review_passes as passes_mod
    from ouroboros import skill_review
    from ouroboros.review_execution import ReviewRouteKind

    drive = pathlib.Path(tmp_path)
    skill = types.SimpleNamespace(
        name="demo", manifest=_fake_manifest(), skill_dir=drive / "skill",
        load_error="", review=review_state, source="",
    )
    binding = types.SimpleNamespace(state_drive_root=drive)
    monkeypatch.setattr(skill_review, "build_resolved_resource_binding",
                        lambda *a, **k: binding)
    monkeypatch.setattr(skill_review, "load_bound_skill", lambda b: skill)
    monkeypatch.setattr(skill_review, "compute_content_hash", lambda *a, **k: content_hash)
    monkeypatch.setattr(skill_review, "_run_deterministic_preflight",
                        lambda *a, **k: None)
    monkeypatch.setattr(skill_review, "reviewer_slot_config_error", lambda: "")
    monkeypatch.setattr(skill_review, "_build_skill_file_packs", lambda *a, **k: ["pack"])
    monkeypatch.setattr(skill_review, "_official_hub_review_profile", lambda s: "")
    monkeypatch.setattr(skill_review, "_review_wave_budget_block",
                        lambda *a, **k: None)
    default_delivery = {
        "models": ["m1", "m2"],
        "routes": [ReviewRouteKind.API_CHAT, ReviewRouteKind.API_CHAT],
        "efforts": ["", ""], "session_targets": ["", ""],
        "session_profiles": ["", ""], "slot_ids": ["slot_1", "slot_2"],
        "legacy_skill_fingerprint": True,
    }
    monkeypatch.setattr(
        skill_review, "commit_triad_delivery", lambda: delivery or default_delivery,
    )
    monkeypatch.setattr(passes_mod, "run_skill_review_passes", passes)
    return skill


def _live_skill_contract_fp():
    from ouroboros import skill_review
    from ouroboros.skill_review_cycles import skill_review_contract_fingerprint

    return skill_review_contract_fingerprint(
        ["m1", "m2"], required_items=skill_review._SKILL_REVIEW_ITEMS, review_profile="",
    )


def test_review_skill_replays_refuses_and_pays_functionally(tmp_path, monkeypatch):
    """Behavior test (wording-1): the REAL review_skill (a) free-replays an
    identical snapshot without touching the panel seam, (b) refuses at the
    exhausted ceiling without dispatching, (c) stamps the paid facts on a
    dispatched outcome."""
    from ouroboros import skill_review

    def _must_not_dispatch(*a, **k):
        raise AssertionError("run_skill_review_passes must not be called")

    contract_fp = _live_skill_contract_fp()
    state = types.SimpleNamespace(
        content_hash="h1", status="warnings",
        findings=[{"item": "bug_hunting", "verdict": "FAIL", "severity": "advisory",
                   "reason": "meh"}],
        reviewer_models=["m1", "m2"],
    )
    # (a) $0 replay: seeded substantive row under the LIVE contract + matching state.
    _write_history(tmp_path, "demo", [
        {"ts": "t1", "status": "warnings", "content_hash": "h1", "paid": True,
         "group_id": "manual:demo", "review_contract_fingerprint": contract_fp,
         "job_id": "j1"},
    ])
    _wire_review_skill(monkeypatch, tmp_path, content_hash="h1",
                       review_state=state, passes=_must_not_dispatch)
    ctx = types.SimpleNamespace(task_id="", task_metadata={}, event_queue=None)
    outcome = skill_review.review_skill(ctx, "demo", persist=False)
    assert outcome.status == "warnings" and outcome.paid is False
    assert outcome.replayed_from_ts == "t1"
    # (b) ceiling: cap=1 with one paid row already spent on this snapshot,
    # and the persisted state diverged so no $0 replay can rescue it —
    # the typed refusal fires before any dispatch.
    monkeypatch.setenv(KEY, "1")
    _wire_review_skill(monkeypatch, tmp_path, content_hash="h1",
                       review_state=types.SimpleNamespace(
                           content_hash="OTHER", status="clean", findings=[],
                           reviewer_models=[]),
                       passes=_must_not_dispatch)
    refused = skill_review.review_skill(ctx, "demo", persist=False)
    assert refused.status == "pending"
    assert "REVIEW_CYCLES_EXHAUSTED" in refused.error and refused.paid is False
    # (c) paid dispatch: fresh snapshot, the wave DISPATCHES (the fake invokes
    # the write-ahead stamp exactly like the shared transport entry does) and
    # then fails — the outcome still carries the paid facts (money was spent).
    from ouroboros.review_dispatch import stamp_review_paid_on_dispatch

    def _dispatch_then_fail(ctx_arg, *a, **k):
        stamp_review_paid_on_dispatch(ctx_arg)
        return ("prompt", {}, "", "provider exploded")

    monkeypatch.setenv(KEY, "5")
    _wire_review_skill(monkeypatch, tmp_path, content_hash="h2",
                       review_state=None, passes=_dispatch_then_fail)
    paid_outcome = skill_review.review_skill(ctx, "demo", persist=False)
    assert paid_outcome.status == "pending" and paid_outcome.paid is True
    assert paid_outcome.review_contract_fingerprint == contract_fp
    assert paid_outcome.wave_id  # the write-ahead dispatch marker names the wave
    assert "infrastructure failure" in paid_outcome.error
    # (c2) a post-install wave whose seam never fires stays unpaid, while its
    # identity remains available to join a worker that dispatches after return.
    _wire_review_skill(monkeypatch, tmp_path, content_hash="h3",
                       review_state=None,
                       passes=lambda *a, **k: ("prompt", {}, "", "packs never assembled"))
    unpaid_outcome = skill_review.review_skill(ctx, "demo", persist=False)
    assert unpaid_outcome.status == "pending" and unpaid_outcome.paid is False
    assert unpaid_outcome.wave_id


def test_review_skill_uses_configured_rows_and_prices_only_api(tmp_path, monkeypatch):
    from ouroboros import skill_review
    from ouroboros.review_execution import ReviewRouteKind

    delivery = {
        "models": ["codex=gpt-5.6-sol", "openai/gpt-5.6-terra"],
        "routes": [ReviewRouteKind.AGENT_SESSION, ReviewRouteKind.API_CHAT],
        "efforts": ["high", "high"],
        "session_targets": ["codex=gpt-5.6-sol", ""],
        "session_profiles": ["profile-a", ""],
        "slot_ids": ["session-slot", "api-slot"],
        "legacy_skill_fingerprint": False,
    }
    captured = {}

    def passes(*args, **kwargs):
        captured["passes"] = kwargs
        return "prompt", {}, "", "stop after delivery capture"

    _wire_review_skill(
        monkeypatch, tmp_path, content_hash="h-route", review_state=None,
        passes=passes, delivery=delivery,
    )

    def budget(_ctx, _skill, _packs, models):
        captured["budget_models"] = list(models)
        return None

    monkeypatch.setattr(skill_review, "_review_wave_budget_block", budget)
    outcome = skill_review.review_skill(
        types.SimpleNamespace(task_id="", task_metadata={}, event_queue=None),
        "demo", persist=False,
    )

    assert outcome.status == "pending"
    assert captured["budget_models"] == ["openai/gpt-5.6-terra"]
    assert captured["passes"]["models"] == delivery["models"]
    assert captured["passes"]["row_plan"] is delivery
    assert captured["passes"]["session_root"] == str(skill_review._REPO_ROOT)


def test_review_skill_pipeline_order_pin():
    """Order pin (kept thin beside the behavior test): the cycles gate runs
    before budget admission, before the panel seam; post-dispatch outcomes go
    through the paid-facts stamp."""
    import inspect

    from ouroboros import skill_review

    gate_source = inspect.getsource(skill_review._skill_cycles_gate)
    assert gate_source.index("free_replay_outcome(") < gate_source.index("skill_review_cycles_refusal(")
    source = inspect.getsource(skill_review.review_skill)
    assert source.index("_skill_cycles_gate(") < source.index("_review_wave_budget_block(")
    assert source.index("_review_wave_budget_block(") < source.index("run_skill_review_passes(")
    assert source.count("_paid_facts(") >= 5  # dispatch marks every post-panel outcome paid
