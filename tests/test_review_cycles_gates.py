"""Max-Review-Cycles semantics on the commit gate and skill review (Δ5/Δ6;
owner decisions Q12/Q16/Q17/Q22/Q23).

Contract under test:
* blocked commit attempts are TYPED (verdict vs infra) at record time;
* the paid fact is recorded at dispatch on the attempt ledger and survives the
  terminal merge; per-root-task counts are derived from it (P7);
* the free-cycle gate runs BEFORE advisory freshness: a spent allowance is a
  free replay marker (typed event, the outcome returns, no paid dispatch); a
  verdict never refuses an identical resubmission;
* skill review: panel-contract fingerprint, per-key paid-cycle counting,
  $0 replay of an identical substantive verdict, typed exhaustion refusal,
  and the Max-Review-Cycles facts on terminal history rows.
"""

from __future__ import annotations

import json
import pathlib
import types

import pytest

from tests.test_git_review_preflight_gate import candidate  # noqa: F401


REPO = pathlib.Path(__file__).resolve().parents[1]
KEY = "OUROBOROS_REVIEW_MAX_CYCLES"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(KEY, raising=False)
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
    assert rows[-1].phase == "blocking_review", "a reviewer's refusal keeps its phase"


def test_an_owner_stop_is_named_by_its_cause_never_as_a_review_refusal(tmp_path, monkeypatch):
    """The owner's stop of a commit review: its cause in words, its own phase and continuation source, the review's
    findings kept, and no light-model call spent after the stop; a reviewer's refusal still synthesizes."""
    import ouroboros.cancel_intents as intents
    import ouroboros.tools.git as git_mod
    from ouroboros import review_custody
    from ouroboros.review_state import load_state, make_repo_key
    from ouroboros.task_continuation import load_review_continuation

    ctx = types.SimpleNamespace(drive_root=tmp_path, task_id="t1")
    assert review_custody.review_stop_cause(ctx) == "" and not review_custody.review_retry_cancelled(ctx)
    monkeypatch.setattr(intents, "cancel_pending", lambda root, task, **k: task == "t1")
    assert review_custody.review_stop_cause(ctx) == "stopped" and review_custody.review_retry_cancelled(ctx)
    monkeypatch.setattr(intents, "cancel_pending", lambda *a, **k: (_ for _ in ()).throw(OSError("unreadable")))
    assert review_custody.review_stop_cause(ctx) == "stop_unreadable" and review_custody.review_retry_cancelled(ctx)
    assert set(review_custody.REVIEW_STOP_WORDS) == {"stopped", "owner_deadline", "stop_unreadable"}
    assert "owner" not in review_custody.REVIEW_STOP_WORDS["stopped"], "a stop is not always the owner's"

    monkeypatch.setattr(git_mod, "run_cmd", lambda *a, **k: "")
    calls = []
    monkeypatch.setattr("ouroboros.tools.review_synthesis.synthesize_to_canonical_issues",
                        lambda findings, **k: calls.append("synthesis") or list(findings))
    monkeypatch.setattr("ouroboros.review_state.compute_obligation_semantic_redirects",
                        lambda *a, **k: calls.append("dedup") or {})
    findings = [{"item": "x", "reason": "r", "severity": "critical", "verdict": "FAIL"}]
    for task, reason in (("t-stop", "owner_stopped"), ("t-refused", "critical_findings")):
        work = types.SimpleNamespace(
            repo_dir=tmp_path, drive_root=tmp_path, task_id=task, task_metadata={},
            _current_review_tool_name="commit_reviewed", _last_triad_models=[], _last_scope_model="",
            _last_triad_raw_results=[], _last_scope_raw_result={}, _review_degraded_reasons=[])
        git_mod._finalize_blocked_review(
            work, "msg", 0.0, combined_msg=f"{reason} words", block_reason=reason, combined_findings=findings,
            pre_fingerprint={"fingerprint": "fp"}, post_fingerprint={"fingerprint": "fp"}, block_class="verdict",
            phase="owner_control" if reason == "owner_stopped" else "blocking_review")
        if reason == "owner_stopped":
            assert calls == [], "no model is spent after the owner's stop"
    rows = {row.task_id: row for row in load_state(pathlib.Path(tmp_path)).filter_attempts(
        repo_key=make_repo_key(pathlib.Path(tmp_path)))}
    assert rows["t-stop"].phase == "owner_control" and rows["t-stop"].block_reason == "owner_stopped"
    assert rows["t-stop"].critical_findings, "the review's findings stay on the stopped attempt"
    assert rows["t-refused"].phase == "blocking_review" and calls == ["synthesis", "dedup"]
    assert load_review_continuation(pathlib.Path(tmp_path), "t-stop").source == "owner_stopped"
    assert load_review_continuation(pathlib.Path(tmp_path), "t-refused").source == "blocked_review"


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
    honored."""
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


# ---------------------------------------------------------------------------
# The free-cycle gate (Δ5.2/Δ5.4): the spent allowance replays free; a verdict refuses nothing


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


def test_free_cycle_gate_spent_ceiling_replays_with_a_typed_event(tmp_path, monkeypatch):
    """A spent allowance never refuses: the gate returns the replay marker (the outcome reaches the
    author, nothing is unstaged or recorded as a block) and emits the typed exhaustion event, with
    the recorded authority ``advisory`` even where the retired key still says ``blocking``."""
    import ouroboros.tools.git as git_mod
    from ouroboros.review_state import CommitAttemptRecord, load_state, make_repo_key, update_state, _utc_now

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")  # retired: read by nothing
    monkeypatch.setenv(KEY, "1")
    monkeypatch.setattr(git_mod, "commit_review_contract_fingerprint", lambda: "cf-1")
    monkeypatch.setattr(
        git_mod, "run_cmd",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no git calls on the replay path")),
    )

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
    assert outcome is not None and "status" not in outcome and "block_reason" not in outcome
    assert outcome["replay_reason"] == "review_cycles_exhausted"
    assert "REVIEW_CYCLES_EXHAUSTED" in outcome["advisory_replay"]
    assert "Decide whether what was reviewed is enough" in outcome["advisory_replay"]
    assert "escalate to the owner" not in outcome["advisory_replay"]
    events = [json.loads(line) for line in
              (pathlib.Path(tmp_path) / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    exhausted = [e for e in events if e["type"] == "review_cycles_exhausted"]
    assert exhausted and exhausted[0]["surface"] == "commit_gate"
    assert exhausted[0]["cycles_paid"] == 1 and exhausted[0]["cap"] == 1
    assert exhausted[0]["enforcement"] == "advisory"
    replays = [e for e in events if e["type"] == "commit_review_free_replay"]
    assert replays and replays[0]["reason"] == "review_cycles_exhausted" and replays[0]["enforcement"] == "advisory"
    assert [row.status for row in load_state(pathlib.Path(tmp_path)).attempts] == ["succeeded"]
    # Below the ceiling the same gate lets the paid review run.
    monkeypatch.setenv(KEY, "2")
    assert git_mod._free_cycle_gate(
        ctx, "msg", 0.0, pre_fingerprint={"fingerprint": "fp-new"}, review_rebuttal="",
    ) is None


def test_a_verdict_never_refuses_an_identical_resubmission(tmp_path, monkeypatch):
    """The free identical-diff refusal left with Blocking: a recorded verdict or infra block of the
    same staged bytes refuses nothing (whatever the retired key says), and a spent allowance still
    replays (the control that the gate is live)."""
    import ouroboros.tools.git as git_mod

    monkeypatch.setattr(git_mod, "commit_review_contract_fingerprint", lambda: "cf-1")
    monkeypatch.setattr(git_mod, "run_cmd", lambda *a, **k: "")
    ctx = _gate_ctx(tmp_path)
    _seed_verdict_block(tmp_path, "fp-1", "cf-1")
    _seed_verdict_block(tmp_path, "fp-1", "cf-1", attempt=2, block_reason="review_quorum", block_class="infra",
                        critical_findings=[])
    for enforcement in ("blocking", "advisory"):
        monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
        assert git_mod._free_cycle_gate(
            ctx, "msg", 0.0, pre_fingerprint={"fingerprint": "fp-1"}, review_rebuttal="",
        ) is None
    _seed_verdict_block(tmp_path, "fp-0", "cf-1", attempt=3, status="succeeded", block_reason="", block_class="",
                        phase="commit", paid=True, root_task_id="root-1", critical_findings=[])
    monkeypatch.setenv(KEY, "1")
    replay = git_mod._free_cycle_gate(ctx, "msg", 0.0, pre_fingerprint={"fingerprint": "fp-1"}, review_rebuttal="")
    assert replay is not None and replay["replay_reason"] == "review_cycles_exhausted"


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


# ---------------------------------------------------------------------------
# Home of the authority after Blocking left (STEP4 7(iv) item 2): no outcome refuses a commit


@pytest.mark.parametrize("outcome", ["critical", "quorum", "not_performed", "owner_stop"])
def test_no_review_outcome_refuses_a_commit_and_an_owner_stop_still_ends_it(candidate, monkeypatch, tmp_path, outcome):  # noqa: F811
    """With the retired setting still saying Blocking in the environment and the settings document, a FAIL, a
    quorum failure and a NOT_PERFORMED all return before Git as ``reviewed``: nothing is committed, the author's
    bytes stay in the working tree, no ``phase=blocking_review`` row is written, and the author's explicit
    continuation on the returned reference lands them. The control: an owner Stop of the work still ends the call
    in ``owner_control`` without a commit."""
    from ouroboros import config
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.mutation_attribution import capture_mutation_baseline
    from ouroboros.review_dispatch import invoke_review_paid_stamp
    from ouroboros.review_ledger import CouplingOutcome
    from ouroboros.review_state import load_state
    from ouroboros.task_results import write_task_result
    from ouroboros.tools import git

    ctx = candidate
    document = tmp_path / "settings.json"
    document.write_text(json.dumps({"OUROBOROS_REVIEW_ENFORCEMENT": "blocking"}), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_PATH", document)
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")  # retired: read by nothing
    ctx.branch_dev = git.run_cmd(["git", "branch", "--show-current"], cwd=ctx.repo_dir).strip()
    git.run_cmd(["git", "reset", "--hard", "HEAD"], cwd=ctx.repo_dir)
    (ctx.repo_dir / "VERSION").write_text("1.0.0\n", encoding="utf-8")
    git.run_cmd(["git", "add", "VERSION"], cwd=ctx.repo_dir)
    git.run_cmd(["git", "commit", "-m", "fixture version"], cwd=ctx.repo_dir)
    write_task_result(ctx.drive_root, ctx.task_id, "running")
    capture_mutation_baseline(ctx.drive_root, ctx.task_id, [{"surface_type": "system_repo", "host_root": str(ctx.repo_dir)}])
    (ctx.repo_dir / "change.py").write_text("value = 2\n", encoding="utf-8")
    monkeypatch.setattr(git, "_run_review_preflight_tests", lambda *_a, **_kw: None)
    monkeypatch.setattr(git, "_post_commit_result", lambda *_a, **_kw: None)
    monkeypatch.setattr(git, "_auto_push", lambda *_a, **_kw: "")
    finding = {"item": "budget", "severity": "critical", "verdict": "FAIL", "reason": "wrong amount"}

    def reviewer(_ctx, message, **_kw):
        invoke_review_paid_stamp(ctx._review_paid_stamp)
        ctx._last_review_critical_findings = [finding] if outcome == "critical" else []
        ctx._last_triad_raw_results = [{"slot_id": "critic", "status": "responded", "parsed": [finding] if outcome == "critical" else [],
                                        "raw_text": "checked"}]
        if outcome == "quorum":
            ctx._last_triad_raw_results = [{"slot_id": "critic", "status": "error", "error": "unavailable"}]
            ctx._last_review_block_reason = "review_quorum"
            return "⚠️ REVIEW_BLOCKED: quorum", CouplingOutcome(), "review_quorum", []
        if outcome == "not_performed":
            return ("⚠️ REVIEW_BLOCKED: review NOT_PERFORMED", CouplingOutcome(blocked=True, status="error"),
                    "coupling_not_performed", [])
        if outcome == "owner_stop":
            request_cancel(ctx.drive_root, ctx.task_id, reason="owner_stopped", source="test")
        return None, CouplingOutcome(verdict="PASS", status="responded"), "", []

    monkeypatch.setattr(git, "_run_parallel_review", reviewer)
    head = git.run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir)
    first = git._repo_commit_push(ctx, "Fix amount", skip_advisory_review=True)
    assert git.run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir) == head, first
    assert (ctx.repo_dir / "change.py").read_text(encoding="utf-8") == "value = 2\n", "the author's bytes stay"
    rows = load_state(ctx.drive_root).attempts
    if outcome == "owner_stop":
        assert "REVIEW_STOPPED" in first and "this commit was not made" in first, first
        assert (rows[-1].phase, rows[-1].block_reason) == ("owner_control", "owner_stopped")
        return
    assert first.startswith("Review outcome returned before commit"), first
    assert all(row.phase != "blocking_review" for row in rows), [(row.phase, row.status) for row in rows]
    reference = json.loads(first.split("\n", 1)[1])["review_reference"]
    landed = git._repo_commit_push(ctx, "Fix amount", review_reference=reference, author_disposition={
        "disposition": "accepted", "rationale": "I read the returned outcome and accept these bytes."})
    assert git.run_cmd(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir) != head, landed
    assert git.run_cmd(["git", "show", "HEAD:change.py"], cwd=ctx.repo_dir).strip() == "value = 2"
