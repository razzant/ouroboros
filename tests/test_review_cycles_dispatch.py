"""Max-Review-Cycles fix round: dispatch-time paid accounting and the
authority-preserving attempt-ledger eviction.

Contract under test (accepted panel fixes F1-F2, fable P3-2/P3-3):
* F1 — trimming the commit-attempt ledger never evicts paid rows or the
  verdict anchors of the identical-diff refusal streak: a capped root cannot
  loop free refusals until the ceiling and the quoted verdict forget
  themselves;
* F2 — the commit gate's paid fact lands WRITE-AHEAD at the first PHYSICAL
  reviewer dispatch (either side): an attempt where both packs refuse at
  assembly spends $0 and consumes no ceiling cycle, while one dispatched side
  makes the cycle paid;
* P3-3 — under advisory enforcement each free-replay reason drives the full
  stage cycle to a passing commit with its own honest progress note and a
  disclosure that lands in the commit result formatting.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import types

import pytest


KEY = "OUROBOROS_REVIEW_MAX_CYCLES"


# ---------------------------------------------------------------------------
# F1 — authority-preserving attempt-ledger eviction


def test_free_refusal_flood_cannot_evict_paid_rows_or_the_verdict_anchor(
    tmp_path, monkeypatch,
):
    """60+ free refusals after the cap is reached change NOTHING: the paid
    count stays, and the identical-diff refusal still quotes the original
    verdict — only the refusal noise is trimmed."""
    import ouroboros.tools.git as git_mod
    from ouroboros.review_state import (
        CommitAttemptRecord,
        load_state,
        make_repo_key,
        update_state,
        _utc_now,
    )
    from ouroboros.tools.commit_gate import count_paid_review_cycles

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setenv(KEY, "2")
    monkeypatch.setattr(git_mod, "commit_review_contract_fingerprint", lambda: "cf-1")
    monkeypatch.setattr(git_mod, "run_cmd", lambda *a, **k: "")
    monkeypatch.setattr(git_mod, "_authorized_managed_update_resolver", lambda ctx: False)
    (pathlib.Path(tmp_path) / "logs").mkdir(parents=True, exist_ok=True)
    ctx = types.SimpleNamespace(
        repo_dir=tmp_path, drive_root=pathlib.Path(tmp_path), task_id="root-1",
        task_metadata={}, event_queue=None,
        drive_logs=lambda: pathlib.Path(tmp_path) / "logs",
        _current_review_tool_name="commit_reviewed",
    )
    repo_key = make_repo_key(pathlib.Path(tmp_path))

    def _seed(state):
        for attempt, (status, phase, paid) in enumerate(
            [("succeeded", "commit", True), ("blocked", "blocking_review", True)], start=1,
        ):
            state.attempts.append(CommitAttemptRecord(
                ts=_utc_now(), commit_message="m", status=status, phase=phase,
                block_reason="critical_findings" if status == "blocked" else "",
                block_class="verdict" if status == "blocked" else "",
                critical_findings=(
                    [{"item": "bug_original", "reason": "the anchor", "severity": "critical"}]
                    if status == "blocked" else []
                ),
                repo_key=repo_key, tool_name="commit_reviewed", task_id="root-1",
                attempt=attempt, paid=paid, root_task_id="root-1",
                pre_review_fingerprint="fp-1",
                review_contract_fingerprint="cf-1",
            ))

    update_state(pathlib.Path(tmp_path), _seed)
    assert count_paid_review_cycles(ctx, root_task_id="root-1") == 2

    for _ in range(60):
        outcome = git_mod._free_cycle_gate(
            ctx, "msg", 0.0, pre_fingerprint={"fingerprint": "fp-1"}, review_rebuttal="",
        )
        assert outcome is not None and outcome["status"] == "blocked"
        assert outcome["block_reason"] == "identical_diff_refused"
        assert "bug_original" in outcome["message"]  # the anchor is still quoted

    assert count_paid_review_cycles(ctx, root_task_id="root-1") == 2
    state = load_state(pathlib.Path(tmp_path))
    rows = state.filter_attempts(repo_key=repo_key)
    # The authority rows survived the flood; the noise portion stayed capped.
    assert any(r.paid and r.status == "succeeded" for r in rows)
    assert any(r.block_class == "verdict" and r.pre_review_fingerprint == "fp-1" for r in rows)
    assert len([r for r in rows if not r.paid and r.block_class != "verdict"]) <= 50


def test_ledger_growth_stays_bounded_while_accounting_authority_survives(tmp_path):
    """F1 follow-up (strip-not-evict): flooding far past the 50-row window with
    HEAVY paid rows plus refusal noise must keep the serialized ledger bounded
    — over-window preserved rows lose their raw payloads (raw_stripped=True) —
    while every accounting fact and the refusal-quote behavior survive."""
    from ouroboros.review_state import (
        CommitAttemptRecord,
        load_state,
        make_repo_key,
        update_state,
        _utc_now,
    )
    from ouroboros.tools.commit_gate import (
        check_identical_verdict_refusal,
        count_paid_review_cycles,
    )

    repo_key = make_repo_key(pathlib.Path(tmp_path))
    heavy_raw = "HEAVYRAWX" * 600  # ~5.4KB per payload field

    def _flood(state):
        for i in range(1, 61):
            # A paid verdict-blocked wave with FULL forensic payloads …
            state.record_attempt(CommitAttemptRecord(
                ts=_utc_now(), commit_message="m" * 400, status="blocked",
                block_reason="critical_findings", block_class="verdict",
                block_details="details " + heavy_raw,
                repo_key=repo_key, tool_name="commit_reviewed", task_id="root-1",
                attempt=2 * i - 1, phase="blocking_review", paid=True,
                root_task_id="root-1", pre_review_fingerprint="fp-1",
                review_contract_fingerprint="cf-1",
                critical_findings=[{"item": "bug_anchor", "reason": "still broken",
                                    "severity": "critical"}],
                triad_raw_results=[{"model_id": "m1", "raw_text": heavy_raw}],
                scope_raw_result={"status": "responded", "raw_text": heavy_raw,
                                  "raw_results": [{"status": "responded",
                                                   "critical_findings": [{"item": "x"}]}]},
            ))
            # … followed by a free-refusal noise row.
            state.record_attempt(CommitAttemptRecord(
                ts=_utc_now(), commit_message="m", status="blocked",
                block_reason="identical_diff_refused", block_details="refused",
                repo_key=repo_key, tool_name="commit_reviewed", task_id="root-1",
                attempt=2 * i, phase="preflight", pre_review_fingerprint="fp-1",
                review_contract_fingerprint="cf-1",
            ))

    update_state(pathlib.Path(tmp_path), _flood)
    ctx = types.SimpleNamespace(
        repo_dir=tmp_path, drive_root=pathlib.Path(tmp_path), task_id="root-1",
        task_metadata={}, _current_review_tool_name="commit_reviewed",
    )
    # Every paid dispatch is still counted; the refusal still quotes the verdict.
    assert count_paid_review_cycles(ctx, root_task_id="root-1") == 60
    refusal = check_identical_verdict_refusal(ctx, "fp-1", contract_fingerprint="cf-1")
    assert "IDENTICAL_DIFF_REFUSED" in refusal and "bug_anchor" in refusal
    state = load_state(pathlib.Path(tmp_path))
    rows = state.filter_attempts(repo_key=repo_key)
    over_window, in_window = rows[:-50], rows[-50:]
    assert len(rows) > 50 and over_window  # preservation forced past the cap
    for row in over_window:
        assert row.raw_stripped is True
        assert row.triad_raw_results == [] and row.scope_raw_result == {}
        assert len(row.block_details) <= 700 and len(row.commit_message) <= 400
        # The accounting facts are intact on every compacted row.
        assert row.paid is True and row.block_class == "verdict"
        assert row.root_task_id == "root-1"
        assert row.review_contract_fingerprint == "cf-1"
        assert row.critical_findings and row.critical_findings[0]["item"] == "bug_anchor"
    # The serialized ledger carries full raw payloads ONLY inside the window:
    # every heavy-sentinel occurrence is accounted for by in-window rows, and
    # each compacted over-window row serializes to a small bounded record —
    # the immortal portion grows ~O(preserved rows x small record), never by
    # full reviewer raw output per reviewed commit.
    import dataclasses

    raw = (pathlib.Path(tmp_path) / "state" / "advisory_review.json").read_text(encoding="utf-8")
    heavy_in_window = sum(
        1 for row in in_window if row.triad_raw_results or row.scope_raw_result
    )
    assert heavy_in_window <= 50
    in_window_sentinels = sum(
        json.dumps(dataclasses.asdict(row)).count("HEAVYRAWX" * 600) for row in in_window
    )
    assert raw.count("HEAVYRAWX" * 600) == in_window_sentinels > 0
    for row in over_window:
        assert len(json.dumps(dataclasses.asdict(row))) < 4_000


def test_history_compaction_never_strips_active_roster_or_invocation_tokens(tmp_path):
    from ouroboros.review_state import (
        CommitAttemptRecord,
        load_state,
        make_repo_key,
        update_state,
        _utc_now,
    )

    repo_key = make_repo_key(pathlib.Path(tmp_path))
    triad = [{
        "slot_id": "slot_api", "operation_id": "op-api",
        "operation_state": "in_flight", "late_result_pending": True,
        "raw_text": "ACTIVE_TRIAD_ROSTER",
    }]
    scope = {"raw_results": [{
        "slot_id": "scope_session", "operation_id": "op-session",
        "operation_state": "in_flight", "late_result_pending": True,
        "pending_invocation_id": "invocation-preserved",
        "raw_text": "ACTIVE_SCOPE_ROSTER",
    }]}

    def _seed(state):
        # Deliberately inconsistent top-level terminal status exercises the
        # row-level custody authority: compaction must follow the live roster,
        # not only the lifecycle projection.
        state.record_attempt(CommitAttemptRecord(
            ts=_utc_now(), commit_message="active", status="failed",
            repo_key=repo_key, tool_name="commit_reviewed", task_id="active-task",
            attempt=1, paid=True, raw_stripped=False,
            triad_raw_results=triad, scope_raw_result=scope,
        ))
        for index in range(2, 64):
            state.record_attempt(CommitAttemptRecord(
                ts=_utc_now(), commit_message=f"terminal-{index}", status="succeeded",
                repo_key=repo_key, tool_name="commit_reviewed",
                task_id=f"terminal-{index}", attempt=index, paid=True,
                triad_raw_results=[{"raw_text": "HEAVY" * 100}],
            ))

    update_state(pathlib.Path(tmp_path), _seed)

    state = load_state(pathlib.Path(tmp_path))
    active = next(row for row in state.attempts if row.task_id == "active-task")
    assert active.raw_stripped is False
    assert active.triad_raw_results == triad
    assert active.scope_raw_result == scope
    assert active in state.get_active_attempts(repo_key=repo_key)


# ---------------------------------------------------------------------------
# F2 / P3-3 — the write-ahead paid stamp on the real stage cycle


def _stage_cycle_harness(tmp_path, monkeypatch, *, fingerprint):
    """A REAL git repo with a stageable change plus the heavy collaborators
    (advisory gate, binding, fingerprint) pinned, so _run_reviewed_stage_cycle
    runs its true order: free gate -> advisory gate -> dispatch."""
    import ouroboros.tools.git as git_mod

    repo = pathlib.Path(tmp_path) / "repo"
    repo.mkdir()
    for cmd in (
        ["git", "init"],
        ["git", "config", "user.email", "t@t"],
        ["git", "config", "user.name", "t"],
    ):
        subprocess.run(cmd, cwd=repo, check=True, capture_output=True)
    (repo / "f.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True, capture_output=True)
    (repo / "f.txt").write_text("changed\n", encoding="utf-8")

    drive = pathlib.Path(tmp_path) / "drive"
    (drive / "logs").mkdir(parents=True, exist_ok=True)
    progress: list = []
    ctx = types.SimpleNamespace(
        repo_dir=repo, drive_root=drive, task_id="root-1", task_metadata={},
        event_queue=None, branch_dev="dev",
        drive_logs=lambda: drive / "logs",
        emit_progress_fn=progress.append,
        _current_review_tool_name="commit_reviewed",
        _review_advisory=[],
        _last_triad_models=[], _last_scope_model="",
        _last_triad_raw_results=[], _last_scope_raw_result={},
        _review_degraded_reasons=[],
    )
    monkeypatch.setattr(git_mod, "commit_review_contract_fingerprint", lambda: "cf-1")
    monkeypatch.setattr(
        git_mod, "_fingerprint_staged_diff",
        lambda repo_dir: {"ok": True, "fingerprint": fingerprint},
    )
    monkeypatch.setattr(git_mod, "_advisory_and_tests_gate", lambda *a, **k: None)
    monkeypatch.setattr(git_mod, "_review_binding_precondition_error", lambda *a, **k: "")
    return git_mod, ctx, progress


def _overflow_wave(dispatch):
    """A wave whose BOTH sides land infra-blocked; ``dispatch`` controls
    whether the (real) transport seam was reached before the refusal."""
    from ouroboros.review_dispatch import stamp_review_paid_on_dispatch
    from ouroboros.tools.scope_review import ScopeReviewResult

    def _wave(ctx, commit_message, **kwargs):
        if dispatch:
            stamp_review_paid_on_dispatch(ctx)  # simulate the route-executor seam
        ctx._last_review_block_reason = "fixed_overflow"
        ctx._last_review_critical_findings = []
        scope = ScopeReviewResult(
            blocked=True,
            block_message="⚠️ SCOPE_REVIEW_BLOCKED: pack did not assemble.",
            status="fixed_overflow",
        )
        return "⚠️ REVIEW_BLOCKED: prompt cannot fit.", scope, "fixed_overflow", []

    return _wave


def test_all_assembly_refused_attempt_stays_unpaid(tmp_path, monkeypatch):
    """F2(a): BOTH packs refusing at assembly ($0 spent) must not consume a
    ceiling cycle — with the default cap this used to exhaust a root for free."""
    from ouroboros.tools.commit_gate import count_paid_review_cycles

    git_mod, ctx, _progress = _stage_cycle_harness(tmp_path, monkeypatch, fingerprint="fp-a")
    monkeypatch.setattr(git_mod, "_run_parallel_review", _overflow_wave(dispatch=False))
    outcome = git_mod._run_reviewed_stage_cycle(ctx, "msg", 0.0)
    assert outcome["status"] == "blocked" and outcome["block_reason"] == "fixed_overflow"
    assert count_paid_review_cycles(ctx, root_task_id="root-1") == 0
    from ouroboros.review_state import load_state, make_repo_key

    rows = load_state(ctx.drive_root).filter_attempts(
        repo_key=make_repo_key(pathlib.Path(ctx.repo_dir)))
    assert rows and rows[-1].paid is False and rows[-1].block_class == "infra"


def test_one_side_dispatched_attempt_counts_as_paid(tmp_path, monkeypatch):
    """F2(b): parallel dispatch means one side can spend while the other
    overflows at assembly — any side dispatching makes the cycle paid."""
    from ouroboros.tools.commit_gate import count_paid_review_cycles

    git_mod, ctx, _progress = _stage_cycle_harness(tmp_path, monkeypatch, fingerprint="fp-b")
    monkeypatch.setattr(git_mod, "_run_parallel_review", _overflow_wave(dispatch=True))
    outcome = git_mod._run_reviewed_stage_cycle(ctx, "msg", 0.0)
    assert outcome["status"] == "blocked"
    assert count_paid_review_cycles(ctx, root_task_id="root-1") == 1
    assert ctx._review_paid_stamp is None  # the seam never leaks past the wave
    from ouroboros.review_state import load_state, make_repo_key

    rows = load_state(ctx.drive_root).filter_attempts(
        repo_key=make_repo_key(pathlib.Path(ctx.repo_dir)), task_id="root-1",
    )
    assert len(rows) == 1
    assert rows[0].attempt == 1 and rows[0].paid is True
    assert rows[0].review_retry_key


@pytest.mark.parametrize(
    "seed,expected_reason,expected_note_part",
    [
        ("verdict", "identical_diff_refused", "reusing the recorded"),
        ("ceiling", "review_cycles_exhausted", "no review outcome"),
    ],
)
def test_advisory_replay_reasons_drive_the_stage_cycle_to_a_disclosed_pass(
    tmp_path, monkeypatch, seed, expected_reason, expected_note_part,
):
    """fable P3-3 (end to end): under advisory each free-replay reason lets the
    REAL stage cycle pass without any dispatch, with its own honest progress
    note, and the disclosure lands in the commit result formatting."""
    from ouroboros.review_state import CommitAttemptRecord, make_repo_key, update_state, _utc_now

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    git_mod, ctx, progress = _stage_cycle_harness(tmp_path, monkeypatch, fingerprint="fp-r")
    monkeypatch.setattr(
        git_mod, "_run_parallel_review",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("free replay must not dispatch")),
    )
    repo_key = make_repo_key(pathlib.Path(ctx.repo_dir))
    if seed == "verdict":
        update_state(ctx.drive_root, lambda s: s.attempts.append(CommitAttemptRecord(
            ts=_utc_now(), commit_message="m", status="blocked",
            block_reason="critical_findings", block_class="verdict",
            repo_key=repo_key, tool_name="commit_reviewed", task_id="root-1",
            attempt=1, phase="blocking_review", pre_review_fingerprint="fp-r",
            review_contract_fingerprint="cf-1",
            critical_findings=[{"item": "bug_q", "reason": "r", "severity": "critical"}],
        )))
    else:
        monkeypatch.setenv(KEY, "1")
        update_state(ctx.drive_root, lambda s: s.attempts.append(CommitAttemptRecord(
            ts=_utc_now(), commit_message="m", status="succeeded", repo_key=repo_key,
            tool_name="commit_reviewed", task_id="root-1", attempt=1, phase="commit",
            paid=True, root_task_id="root-1", pre_review_fingerprint="fp-old",
        )))

    outcome = git_mod._run_reviewed_stage_cycle(ctx, "msg", 0.0)
    assert outcome["status"] == "passed"
    notes = [n for n in progress if "Max Review Cycles" in n]
    assert notes and expected_note_part in notes[0]
    # The loud disclosure reached the advisory channel AND the commit result.
    assert any(expected_reason in w for w in ctx._review_advisory)
    result = git_mod._format_commit_result(ctx, "msg", "", "")
    assert "no new triad+scope review was bought" in result
    assert expected_reason in result
    events = [json.loads(line) for line in
              (ctx.drive_root / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    replays = [e for e in events if e["type"] == "commit_review_free_replay"]
    assert replays and replays[-1]["reason"] == expected_reason


def test_managed_advisory_ceiling_replay_survives_stale_subject_trees(
    tmp_path, monkeypatch,
):
    """Synthesis wave W1 (cross-lane interference): the MANAGED resolver's
    advisory free replay skips run_parallel_review — previously the ONLY reset
    point of ``ctx._last_review_subject_trees`` — so subject-tree residue from
    a previous PAID attempt was compared against the CURRENT binding tree and
    blocked every retry with a typed review_subject_binding_mismatch (ceiling
    still exhausted -> replay again -> same stale set: the managed update
    dead-ended). The stage cycle now resets the set at every attempt start."""
    from ouroboros.review_state import CommitAttemptRecord, make_repo_key, update_state, _utc_now

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    monkeypatch.setenv(KEY, "1")
    git_mod, ctx, progress = _stage_cycle_harness(tmp_path, monkeypatch, fingerprint="fp-m")
    monkeypatch.setattr(
        git_mod, "_fingerprint_staged_diff",
        lambda repo_dir: {
            "ok": True, "fingerprint": "fp-m",
            "binding": {"tree_sha": "tree-NEW"},
        },
    )
    monkeypatch.setattr(git_mod, "_authorized_managed_update_resolver", lambda c: True)
    monkeypatch.setattr(
        git_mod, "_run_parallel_review",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("free replay must not dispatch")),
    )
    repo_key = make_repo_key(pathlib.Path(ctx.repo_dir))
    update_state(ctx.drive_root, lambda s: s.attempts.append(CommitAttemptRecord(
        ts=_utc_now(), commit_message="m", status="blocked", repo_key=repo_key,
        tool_name="commit_reviewed", task_id="root-1", attempt=1,
        phase="blocking_review", block_reason="quorum_failure", block_class="infra",
        paid=True, root_task_id="root-1", pre_review_fingerprint="fp-old",
    )))
    # Residue of the previous PAID attempt in this same resolver task/ctx.
    ctx._last_review_subject_trees = {"tree-OLD"}

    outcome = git_mod._run_reviewed_stage_cycle(ctx, "resolve", 0.0)

    assert outcome["status"] == "passed", outcome
    assert any("paid-cycle ceiling exhausted" in n for n in progress)


def test_managed_in_attempt_subject_mismatch_still_blocks(tmp_path, monkeypatch):
    """W1 control: the per-attempt reset must not weaken the assertion — a
    subject tree recorded DURING the attempt that diverges from the binding
    tree still blocks with the typed mismatch (and only in-attempt trees are
    asserted: pre-attempt residue never reaches the message)."""
    git_mod, ctx, _progress = _stage_cycle_harness(tmp_path, monkeypatch, fingerprint="fp-c")
    monkeypatch.setattr(
        git_mod, "_fingerprint_staged_diff",
        lambda repo_dir: {
            "ok": True, "fingerprint": "fp-c",
            "binding": {"tree_sha": "tree-REAL"},
        },
    )
    monkeypatch.setattr(git_mod, "_authorized_managed_update_resolver", lambda c: True)
    ctx._last_review_subject_trees = {"tree-STALE-RESIDUE"}  # must be irrelevant

    def _wave(inner_ctx, *a, **kw):
        inner_ctx._last_review_subject_trees.add("tree-WRONG")
        return None, None, "", []

    monkeypatch.setattr(git_mod, "_run_parallel_review", _wave)
    monkeypatch.setattr(
        git_mod, "_aggregate_review_verdict", lambda *a, **kw: (False, "", "", [], []),
    )

    outcome = git_mod._run_reviewed_stage_cycle(ctx, "resolve", 0.0)

    assert outcome["status"] == "blocked"
    assert outcome["block_reason"] == "review_subject_binding_mismatch"
    assert "tree-WRONG" in outcome["message"]
    assert "tree-STALE-RESIDUE" not in outcome["message"]


# ---------------------------------------------------------------------------
# F3 — the skill dispatch marker: four wave outcomes, one paid unit each


# ---------------------------------------------------------------------------
# F4a — spent-rebuttal memory is scoped to the CURRENT panel contract


# ---------------------------------------------------------------------------
# C4 — append-only per-wave dispatch markers (concurrency + legacy migration)


def test_fail_closed_paid_stamp_replays_one_failure_to_every_dispatcher():
    from ouroboros.review_dispatch import ReviewPaidStamp, invoke_review_paid_stamp

    writes = []

    def refuse():
        writes.append("attempted")
        raise RuntimeError("wallet unavailable")

    stamp = ReviewPaidStamp(refuse, fail_closed=True)
    with pytest.raises(RuntimeError, match="wallet unavailable"):
        invoke_review_paid_stamp(stamp)
    with pytest.raises(RuntimeError, match="wallet unavailable"):
        invoke_review_paid_stamp(stamp)

    assert writes == ["attempted"]
    assert stamp.fired is True


def test_strict_api_stamp_veto_releases_sync_and_async_attempts(tmp_path):
    import asyncio
    from ouroboros import usage_accounting as ua
    from ouroboros.review_dispatch import ReviewPaidStamp, bind_api_review_paid_stamp

    def request(root):
        return ua.AttemptRequest(model="local", provider="local", reservation_usd=1.0,
                                 drive_root=root, task_id="t", root_task_id="t")

    def refuse(root):
        rows = (root / ua.LEDGER_REL).read_text().splitlines()
        assert json.loads(rows[-1])["state"] == "reserved"
        raise RuntimeError("wallet unavailable")

    sync_root, sent = tmp_path / "sync", []
    with bind_api_review_paid_stamp(ReviewPaidStamp(lambda: refuse(sync_root), fail_closed=True)):
        with pytest.raises(ua.PhysicalAttemptPreparationFailed):
            ua.execute_physical_attempt(request(sync_root), lambda: sent.append("sync"))
    assert sent == []
    assert ua.usage_projection(sync_root)["attempt_counts"] == {"released": 1}

    async_root = tmp_path / "async"
    async def run():
        with bind_api_review_paid_stamp(ReviewPaidStamp(lambda: refuse(async_root), fail_closed=True)):
            return await ua.execute_physical_attempt_async(
                request(async_root), lambda: (_ for _ in ()).throw(AssertionError("sent")))
    with pytest.raises(ua.PhysicalAttemptPreparationFailed):
        asyncio.run(run())
    assert ua.usage_projection(async_root)["attempt_counts"] == {"released": 1}
