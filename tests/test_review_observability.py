"""Regression tests for review-pipeline observability / epistemic-integrity.

Verifies that:
1. parse_failure is distinguishable from PASS in CommitAttemptRecord
2. Every triad actor record carries model_id and full raw_text (no provenance loss)
3. Partial participation (1 of 3 models ERROR) is recorded as DEGRADED, not PASS
4. scope_raw_result is populated with raw_text and model_id after a scope review
5. ScopeReviewResult.status = "budget_exceeded" when budget gate fires
6. ScopeReviewResult.status = "parse_failure" on unparseable scope output
7. triad_raw_results / scope_raw_result survive save/load roundtrip in review_state.py
8. _collect_review_findings returns 4-tuple (not 3-tuple)
9. ctx._last_triad_raw_results is reset at start of each _run_unified_review attempt
10. parse_failure actors do NOT count toward quorum (same as transport errors)
11. scope_raw_result includes parsed_items field for shape parity with triad actors
"""
from __future__ import annotations

import json
import pathlib
from unittest.mock import MagicMock, patch



# ── helpers ───────────────────────────────────────────────────────────────────

def _make_ctx(tmp_path: pathlib.Path) -> MagicMock:
    ctx = MagicMock()
    ctx.repo_dir = str(tmp_path)
    # Pin drive_root to a real temp path. Left unset, ``ctx.drive_root`` is an
    # auto-MagicMock and any production ``ctx.drive_root / "state" / ...`` write
    # materialises a literal ``MagicMock/mock.drive_root.__truediv__()...`` dir
    # in the repo CWD (then committed by a careless ``git add -A``).
    ctx.drive_root = str(tmp_path)
    ctx.task_id = "test-task"
    ctx._review_history = []
    ctx._review_iteration_count = 0
    ctx._last_review_block_reason = ""
    ctx._last_review_critical_findings = []
    ctx._last_review_advisory_findings = []
    ctx._last_triad_raw_results = []
    ctx._last_triad_models = []
    ctx._review_advisory = []
    ctx._review_degraded_reasons = []
    ctx.drive_logs = MagicMock(return_value=tmp_path)
    return ctx


def _model_result(model: str, text: str, verdict: str = "UNKNOWN",
                  tokens_in: int = 10, tokens_out: int = 5,
                  cost: float = 0.001) -> dict:
    return {
        "model": model,
        "text": text,
        "verdict": verdict,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_estimate": cost,
    }


# ── Test 1: parse_failure ≠ PASS ─────────────────────────────────────────────

def test_parse_failure_distinct_from_pass(tmp_path):
    """parse_failure must produce status='parse_failure' in actor record, not 'responded'."""
    from ouroboros.tools.review import _collect_review_findings

    ctx = _make_ctx(tmp_path)
    bad_json = "This is definitely not JSON at all."
    results = [_model_result("model-a", bad_json)]
    critical_fails, advisory_warns, errored_models, triad_raw = _collect_review_findings(ctx, results)

    assert len(triad_raw) == 1
    assert triad_raw[0]["model_id"] == "model-a"
    assert triad_raw[0]["status"] == "parse_failure", (
        f"Expected parse_failure, got {triad_raw[0]['status']!r}"
    )
    # raw_text must be preserved in full — not truncated
    assert triad_raw[0]["raw_text"] == bad_json
    assert triad_raw[0]["parsed_items"] == []


def test_pass_verdict_distinct_from_parse_failure(tmp_path):
    """A fully PASS review must produce status='responded', not 'parse_failure'."""
    from ouroboros.tools.review import _collect_review_findings

    ctx = _make_ctx(tmp_path)
    pass_items = json.dumps([
        {"item": "bible_compliance", "verdict": "PASS", "severity": "critical", "reason": "OK"}
    ])
    results = [_model_result("model-b", pass_items)]
    critical_fails, advisory_warns, errored_models, triad_raw = _collect_review_findings(ctx, results)

    assert len(triad_raw) == 1
    assert triad_raw[0]["status"] == "responded"
    assert triad_raw[0]["parsed_items"] != []
    assert len(critical_fails) == 0


# ── Test 2: provenance — model_id and raw_text always present ────────────────

def test_triad_actor_records_carry_model_id_and_raw_text(tmp_path):
    """Every actor record must have non-empty model_id and raw_text."""
    from ouroboros.tools.review import _collect_review_findings

    ctx = _make_ctx(tmp_path)
    items_json = json.dumps([
        {"item": "code_quality", "verdict": "FAIL", "severity": "critical", "reason": "bug"}
    ])
    results = [
        _model_result("gpt-5.5", items_json),
        _model_result("gemini-pro", items_json),
        _model_result("claude-opus", items_json),
    ]
    _, _, _, triad_raw = _collect_review_findings(ctx, results)

    assert len(triad_raw) == 3
    for record in triad_raw:
        assert record["model_id"], f"model_id empty in record: {record}"
        assert record["raw_text"], f"raw_text empty in record: {record}"
        assert isinstance(record["tokens_in"], int)
        assert isinstance(record["cost_usd"], float)


def test_error_model_provenance_preserved(tmp_path):
    """ERROR responses must still have model_id and status='error'."""
    from ouroboros.tools.review import _collect_review_findings

    ctx = _make_ctx(tmp_path)
    results = [_model_result("bad-model", "Error: API timeout", verdict="ERROR")]
    _, _, errored, triad_raw = _collect_review_findings(ctx, results)

    assert "bad-model" in errored
    assert len(triad_raw) == 1
    assert triad_raw[0]["model_id"] == "bad-model"
    assert triad_raw[0]["status"] == "error"
    assert triad_raw[0]["parsed_items"] == []


# ── Test 3: degraded participation recorded ───────────────────────────────────

def test_partial_participation_sets_degraded_reasons(tmp_path):
    """1 of 3 models ERROR while quorum met → ctx._review_degraded_reasons populated."""
    from ouroboros.tools.review import _collect_review_findings

    ctx = _make_ctx(tmp_path)
    ok_items = json.dumps([
        {"item": "bible_compliance", "verdict": "PASS", "severity": "critical", "reason": "OK"}
    ])
    results = [
        _model_result("model-1", ok_items),
        _model_result("model-2", ok_items),
        _model_result("model-3", "Error: timeout", verdict="ERROR"),
    ]
    _, _, errored, triad_raw = _collect_review_findings(ctx, results)

    statuses = [r["status"] for r in triad_raw]
    assert "error" in statuses, "Should have an error actor record"
    assert "responded" in statuses, "Should have responded actor records"

    # Degraded reasons should be populated on ctx
    degraded = getattr(ctx, "_review_degraded_reasons", [])
    assert len(degraded) >= 1
    assert any("DEGRADED" in r for r in degraded)


def test_all_three_models_ok_no_degraded(tmp_path):
    """When all 3 models respond successfully, no degraded reasons."""
    from ouroboros.tools.review import _collect_review_findings

    ctx = _make_ctx(tmp_path)
    ok_items = json.dumps([
        {"item": "bible_compliance", "verdict": "PASS", "severity": "critical", "reason": "OK"}
    ])
    results = [_model_result(f"m{i}", ok_items) for i in range(3)]
    _, _, _, triad_raw = _collect_review_findings(ctx, results)

    degraded = getattr(ctx, "_review_degraded_reasons", [])
    assert len(degraded) == 0
    assert all(r["status"] == "responded" for r in triad_raw)


# ── Test 4: scope_raw_result populated ────────────────────────────────────────

# test_scope_review_result_has_raw_text_and_model_id removed in v5.15.x —
# the test only constructed a ScopeReviewResult dataclass and asserted the
# field values it just passed in. No production code is exercised. The
# dataclass shape is enforced by the type system + downstream tests that
# actually consume ScopeReviewResult instances from real review runs.






# ── Test 6: parse_failure on scope output ─────────────────────────────────────

def test_coupling_seat_parse_failure_status():
    """Unparseable output of a seat asked the coupling question must produce
    status='parse_failure' with raw_text preserved — never a PASS of Part 2."""
    from ouroboros.triad_review import parse_seat_answers

    bad_raw = "Sorry, I cannot review this at this time."
    parsed = parse_seat_answers({"results": [{"model": "some-model", "slot_id": "s1", "text": bad_raw}]},
                                {"s1": ("change", "coupling")})
    record = parsed.actor_records[0].to_dict()
    assert record["status"] == "parse_failure"
    assert record["raw_text"] == bad_raw
    assert record["parts"] == ["change", "coupling"] and parsed.findings == []


# ── Test 7: save/load roundtrip ───────────────────────────────────────────────

def test_commit_attempt_roundtrip_with_actor_records(tmp_path):
    """triad_raw_results and scope_raw_result survive JSON serialization roundtrip."""
    from ouroboros.review_state import CommitAttemptRecord, _commit_attempt_from_dict
    import dataclasses

    triad = [
        {"model_id": "gpt-5.5", "status": "responded", "raw_text": '["pass"]',
         "parsed_items": [{"item": "code_quality", "verdict": "PASS"}],
         "tokens_in": 100, "tokens_out": 50, "cost_usd": 0.01},
        {"model_id": "gemini-pro", "status": "parse_failure", "raw_text": "not json",
         "parsed_items": [], "tokens_in": 80, "tokens_out": 20, "cost_usd": 0.005},
    ]
    scope = {
        "model_id": "claude-opus-4.6",
        "status": "responded",
        "raw_text": '[{"item":"intent_alignment","verdict":"PASS"}]',
        "prompt_chars": 3000,
        "tokens_in": 500,
        "tokens_out": 200,
        "cost_usd": 0.03,
        "critical_findings": [],
        "advisory_findings": [],
    }

    record = CommitAttemptRecord(
        ts="2026-04-16T10:00:00Z",
        commit_message="test: observability",
        status="blocked",
        block_reason="critical_findings",
        triad_raw_results=triad,
        scope_raw_result=scope,
    )

    # Serialize to dict (as done by asdict for JSON persistence)
    as_dict = dataclasses.asdict(record)
    # Deserialize
    restored = _commit_attempt_from_dict(as_dict)

    assert len(restored.triad_raw_results) == 2
    assert restored.triad_raw_results[0]["model_id"] == "gpt-5.5"
    assert restored.triad_raw_results[0]["status"] == "responded"
    assert restored.triad_raw_results[1]["status"] == "parse_failure"
    assert restored.triad_raw_results[1]["raw_text"] == "not json"

    assert restored.scope_raw_result["model_id"] == "claude-opus-4.6"
    assert restored.scope_raw_result["status"] == "responded"
    assert restored.scope_raw_result["tokens_in"] == 500


def test_commit_attempt_roundtrip_empty_actor_fields(tmp_path):
    """CommitAttemptRecord with no actor data roundtrips cleanly (backward compat)."""
    from ouroboros.review_state import CommitAttemptRecord, _commit_attempt_from_dict
    import dataclasses

    record = CommitAttemptRecord(
        ts="2026-04-16T10:00:00Z",
        commit_message="old format record",
        status="succeeded",
    )
    as_dict = dataclasses.asdict(record)
    # Simulate old format — remove new keys
    as_dict.pop("triad_raw_results", None)
    as_dict.pop("scope_raw_result", None)

    restored = _commit_attempt_from_dict(as_dict)
    assert restored.triad_raw_results == []
    assert restored.scope_raw_result == {}


# ── Test 8: _collect_review_findings returns 4-tuple ─────────────────────────

def test_collect_review_findings_returns_4_tuple(tmp_path):
    """_collect_review_findings must return a 4-tuple (not 3)."""
    from ouroboros.tools.review import _collect_review_findings

    ctx = _make_ctx(tmp_path)
    result = _collect_review_findings(ctx, [])
    assert len(result) == 4, f"Expected 4-tuple, got {len(result)}-tuple"
    critical_fails, advisory_warns, errored_models, triad_raw = result
    assert isinstance(critical_fails, list)
    assert isinstance(advisory_warns, list)
    assert isinstance(errored_models, list)
    assert isinstance(triad_raw, list)


# ── Test 9: ctx._last_triad_raw_results reset per attempt ────────────────────

def test_coupling_history_entry_preserves_status_for_parse_failure():
    """_coupling_history_entry must preserve status='parse_failure' so it is
    not misread as clean PASS on the retry path (empty findings != PASS)."""
    from ouroboros.review_ledger import CouplingOutcome
    from ouroboros.tools.parallel_review import _coupling_history_entry

    # A parse_failure result has no findings (they couldn't be parsed)
    # but must NOT be summarised as "(no findings)" — which would look like PASS
    result = CouplingOutcome(blocked=True, status="parse_failure")
    entry = _coupling_history_entry(result)
    assert entry["status"] == "parse_failure", "status must be preserved in history entry"
    assert entry["summary"] != "(no findings)", (
        "parse_failure with no findings must NOT summarise as '(no findings)' — "
        "that is indistinguishable from a clean PASS"
    )
    assert "parse_failure" in entry["summary"], "summary must expose the status signal"


def test_coupling_history_entry_budget_exceeded_summary():
    """budget_exceeded status must appear in the summary even with no findings."""
    from ouroboros.review_ledger import CouplingOutcome
    from ouroboros.tools.parallel_review import _coupling_history_entry

    result = CouplingOutcome(blocked=False, status="budget_exceeded")
    entry = _coupling_history_entry(result)
    assert entry["status"] == "budget_exceeded"
    assert "budget_exceeded" in entry["summary"]


def test_coupling_history_entry_clean_pass_keeps_no_findings_label():
    """A genuine responded+no-findings result keeps '(no findings)' summary."""
    from ouroboros.review_ledger import CouplingOutcome
    from ouroboros.tools.parallel_review import _coupling_history_entry

    result = CouplingOutcome(verdict="PASS", blocked=False, status="responded")
    entry = _coupling_history_entry(result)
    assert entry["status"] == "responded"
    assert entry["summary"] == "(no findings)"


# ── Scope history round-label rendering (v4.32.0 epistemic-integrity fix) ────

def test_scope_history_section_does_not_label_budget_exceeded_as_passed():
    """The rendered history section must NOT show ``PASSED`` for a
    ``status='budget_exceeded'`` entry even though ``blocked=False``.

    Regression guard for obligation b039eaf4402b (v4.32.0): degraded states
    (budget_exceeded, omitted, parse_failure) were previously rendered as
    ``PASSED`` because the renderer derived the label purely from
    ``blocked=False`` — indistinguishable from a genuine clean PASS.
    """
    from ouroboros.tools.review_brief_coupling import build_coupling_history_section

    history = [{
        "blocked": False,
        "status": "budget_exceeded",
        "summary": "(budget_exceeded)",
        "critical_findings": [],
        "advisory_findings": [],
    }]
    section = build_coupling_history_section(history)
    # Extract just the "Round 1:" label line
    round1_line = next(
        (ln for ln in section.splitlines() if ln.startswith("Round 1:")),
        "",
    )
    assert round1_line, "expected a 'Round 1:' label line"
    assert "PASSED" not in round1_line, (
        f"budget_exceeded must NOT be rendered as PASSED — got: {round1_line!r}"
    )
    assert "BUDGET_EXCEEDED" in round1_line, (
        f"label must surface the status signal — got: {round1_line!r}"
    )


def test_scope_history_section_does_not_label_omitted_as_passed():
    """The rendered history section must NOT show ``PASSED`` for a
    ``status='omitted'`` sentinel entry."""
    from ouroboros.tools.review_brief_coupling import build_coupling_history_section

    history = [{
        "blocked": False,
        "status": "omitted",
        "summary": "earlier scope-review round(s) omitted",
        "critical_findings": [],
        "advisory_findings": [],
    }]
    section = build_coupling_history_section(history)
    round1_line = next(
        (ln for ln in section.splitlines() if ln.startswith("Round 1:")),
        "",
    )
    assert round1_line
    assert "PASSED" not in round1_line, (
        f"omitted must NOT be rendered as PASSED — got: {round1_line!r}"
    )
    assert "OMITTED" in round1_line


def test_scope_history_section_does_not_label_parse_failure_as_passed():
    """The rendered history section must NOT show ``PASSED`` for a
    ``status='parse_failure'`` entry. Although the entry is marked
    ``blocked=True`` by _scope_history_entry for parse_failure results,
    the renderer must also guard against the degenerate case where
    upstream code produced blocked=False + status=parse_failure."""
    from ouroboros.tools.review_brief_coupling import build_coupling_history_section

    # Degenerate-but-guarded case: blocked=False + status=parse_failure
    history = [{
        "blocked": False,
        "status": "parse_failure",
        "summary": "(parse_failure)",
        "critical_findings": [],
        "advisory_findings": [],
    }]
    section = build_coupling_history_section(history)
    round1_line = next(
        (ln for ln in section.splitlines() if ln.startswith("Round 1:")),
        "",
    )
    assert round1_line
    assert "PASSED" not in round1_line, (
        f"parse_failure must NOT be rendered as PASSED — got: {round1_line!r}"
    )
    assert "PARSE_FAILURE" in round1_line


def test_scope_history_section_labels_genuine_pass_as_passed():
    """A genuine ``responded`` + ``blocked=False`` + no findings entry MUST
    still render as ``PASSED`` so the reviewer can distinguish it from
    degraded states."""
    from ouroboros.tools.review_brief_coupling import build_coupling_history_section

    history = [{
        "blocked": False,
        "status": "responded",
        "summary": "(no findings)",
        "critical_findings": [],
        "advisory_findings": [],
    }]
    section = build_coupling_history_section(history)
    round1_line = next(
        (ln for ln in section.splitlines() if ln.startswith("Round 1:")),
        "",
    )
    assert round1_line
    assert "PASSED" in round1_line, (
        f"genuine clean responded round must still render as PASSED — got: {round1_line!r}"
    )


def test_scope_history_section_labels_blocked_as_blocked():
    """A ``blocked=True`` entry must render as ``BLOCKED`` regardless of status."""
    from ouroboros.tools.review_brief_coupling import build_coupling_history_section

    history = [{
        "blocked": True,
        "status": "responded",
        "summary": "Critical: some_item",
        "critical_findings": [{"item": "some_item"}],
        "advisory_findings": [],
    }]
    section = build_coupling_history_section(history)
    round1_line = next(
        (ln for ln in section.splitlines() if ln.startswith("Round 1:")),
        "",
    )
    assert round1_line
    assert "BLOCKED" in round1_line
    assert "PASSED" not in round1_line


# test_scope_round_label_helper_all_paths trimmed in v5.15.x — the three
# behavioural branches (BLOCKED wins / status uppercased / responded+no
# block = PASSED) are exercised end-to-end by the
# test_scope_history_section_does_not_label_{budget_exceeded,omitted,
# parse_failure}_as_passed integration tests above. The direct unit-level
# coverage is redundant.


def _run_unified_review_with_one_fresh_seat(tmp_path, monkeypatch, *, pool: bool):
    """One ``_run_unified_review`` over a ctx that still carries a prior attempt's actor
    record. ``pool=True`` puts one marked api seat (``fresh-model``) in the environment catalog
    so the wave is dispatched to the mocked fan-out; ``pool=False`` leaves the pool empty."""
    from ouroboros.tools import review as review_mod
    from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool

    if pool:
        set_review_pool(monkeypatch, pool_roster(pool_seat("fresh-seat", "fresh-model")))
    ctx = _make_ctx(tmp_path)
    # Pre-seed stale data simulating a prior attempt
    ctx._last_triad_raw_results = [
        {"model_id": "stale-model", "status": "responded", "raw_text": "stale data",
         "parsed_items": [], "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
    ]
    pass_items = json.dumps([
        {"item": "bible_compliance", "verdict": "PASS", "severity": "critical", "reason": "OK"}
    ])
    mock_review_output = json.dumps({
        "results": [
            {"model": "fresh-model", "text": pass_items, "verdict": "PASS",
             "tokens_in": 10, "tokens_out": 5, "cost_estimate": 0.001}
        ]
    })
    fan_out = MagicMock(return_value=mock_review_output)
    # Return a staged diff so function proceeds past the empty-diff guard
    with patch.object(review_mod, "run_cmd", return_value="some diff content"), \
         patch("ouroboros.tools.review_binary_context.capture_staged_diff",
               return_value="some diff content"), \
         patch.object(review_mod, "_handle_multi_model_review", fan_out), \
         patch.object(review_mod, "_load_checklist_section", return_value="## checklist"), \
         patch.object(review_mod, "_preflight_check", return_value=None), \
         patch.object(review_mod, "load_governance_doc", return_value=""), \
         patch("ouroboros.tools.review_helpers.build_touched_file_pack",
               return_value=("(files)", [])):
        review_mod._run_unified_review(ctx, "test commit")
    return ctx, fan_out


def test_last_triad_raw_results_reset_at_start_of_run_unified_review(tmp_path, monkeypatch):
    """ctx._last_triad_raw_results must be reset at start of each _run_unified_review call.

    We verify by pre-seeding stale data then running a review over a pool with one
    seat that completes with mocked LLM output (no findings). The stale data from
    the previous attempt must be gone — replaced by fresh actor records from this run.
    (The pool is the environment catalog's marked rows; with no seat the wave is not
    dispatched at all — the case below.)
    """
    ctx, fan_out = _run_unified_review_with_one_fresh_seat(tmp_path, monkeypatch, pool=True)

    assert fan_out.called, "the one-seat pool must reach the fan-out"
    # After the run, stale model_id must not appear
    model_ids = [r["model_id"] for r in ctx._last_triad_raw_results]
    assert "stale-model" not in model_ids, (
        "Stale triad_raw_results from prior attempt must be cleared at function entry"
    )
    assert "fresh-model" in model_ids, "Fresh actor record from this run must be present"


def test_last_triad_raw_results_reset_precedes_the_empty_pool_exit(tmp_path, monkeypatch):
    """The reset is at function ENTRY, before the pool is even assembled: an empty pool
    ends the run as the typed ``pool_empty`` block without a wave (PR-3 W2), and a prior
    attempt's actor records must not survive into that record either."""
    from ouroboros.tools.review_helpers import REVIEW_POOL_EMPTY_REASON

    ctx, fan_out = _run_unified_review_with_one_fresh_seat(tmp_path, monkeypatch, pool=False)

    assert not fan_out.called, "an empty pool dispatches no wave"
    assert ctx._last_review_block_reason == REVIEW_POOL_EMPTY_REASON
    assert ctx._last_triad_raw_results == [], "stale actor records must not outlive the attempt that made them"


# ── Test 10: parse_failure actors do NOT count toward quorum ──────────────────

def test_parse_failure_does_not_count_toward_quorum(tmp_path):
    """1 responded + 1 parse_failure + 1 error = quorum failure (only 1 usable reviewer).

    Before this fix, errored_models only tracked transport errors so parse_failure
    was counted as a successful quorum participant, silently undermining epistemic
    integrity.
    """
    from ouroboros.tools.review import _collect_review_findings

    ctx = _make_ctx(tmp_path)

    pass_items = json.dumps([
        {"item": "bible_compliance", "verdict": "PASS", "severity": "critical", "reason": "OK"}
    ])
    results = [
        _model_result("model-responded", pass_items),        # 1 good reviewer
        _model_result("model-parse-fail", "not valid json"), # parse_failure — unusable
        _model_result("model-error", "", verdict="ERROR"),   # transport error — unusable
    ]
    critical_fails, advisory_warns, errored_models, triad_raw = _collect_review_findings(ctx, results)

    # Store on ctx as _run_unified_review would
    ctx._last_triad_raw_results = triad_raw

    # Derive quorum the new way: count only responded actors
    successful_reviewers = sum(1 for r in triad_raw if r.get("status") == "responded")
    assert successful_reviewers == 1, (
        f"Only 1 actor responded; parse_failure and error must not count. "
        f"Got successful_reviewers={successful_reviewers}"
    )
    # Confirm statuses are distinct
    statuses = {r["model_id"]: r["status"] for r in triad_raw}
    assert statuses["model-responded"] == "responded"
    assert statuses["model-parse-fail"] == "parse_failure"
    assert statuses["model-error"] == "error"


def test_two_responded_plus_one_parse_failure_meets_quorum(tmp_path):
    """2 responded + 1 parse_failure = quorum met (2 usable reviewers, DEGRADED recorded).

    Also verifies that parse_failure does NOT appear in critical_fails — it must only
    produce an advisory warning so the commit is not blocked when quorum is met.
    """
    from ouroboros.tools.review import _collect_review_findings

    ctx = _make_ctx(tmp_path)
    ctx._review_degraded_reasons = []

    pass_items = json.dumps([
        {"item": "bible_compliance", "verdict": "PASS", "severity": "critical", "reason": "OK"}
    ])
    results = [
        _model_result("model-a", pass_items),
        _model_result("model-b", pass_items),
        _model_result("model-c", "not valid json"),  # parse_failure
    ]
    critical_fails, advisory_warns, errored_models, triad_raw = _collect_review_findings(ctx, results)

    successful = sum(1 for r in triad_raw if r.get("status") == "responded")
    assert successful == 2, "2 responded reviewers should meet the quorum threshold of 2"
    # DEGRADED must be recorded because 1 actor failed
    assert ctx._review_degraded_reasons, "parse_failure with quorum met must still record DEGRADED"
    # parse_failure must NOT be in critical_fails — it should only be an advisory note
    parse_fail_in_critical = any("parse" in f.lower() or "Could not parse" in f for f in critical_fails)
    assert not parse_fail_in_critical, (
        f"parse_failure must not appear in critical_fails when quorum is met. "
        f"critical_fails={critical_fails}"
    )
    # parse_failure must appear as an advisory warning
    parse_fail_in_advisory = any("parse_failure" in w or "Could not parse" in w for w in advisory_warns)
    assert parse_fail_in_advisory, (
        f"parse_failure must produce an advisory warning. advisory_warns={advisory_warns}"
    )


# ── Test 10a: stale forensic fields cleared at commit entrypoint ─────────────

def test_stale_actor_evidence_cleared_at_commit_start(tmp_path):
    """ctx._last_triad_raw_results, _last_scope_raw_result, _review_degraded_reasons
    must be reset at the start of _repo_commit_push so that
    stale data from a prior attempt never bleeds into early-exit paths (e.g.
    fingerprint failure before run_parallel_review runs).
    """
    from ouroboros.tools import git as git_mod

    ctx = _make_ctx(tmp_path)
    # Pre-seed stale forensic data from a previous attempt
    ctx._last_triad_raw_results = [
        {"model_id": "stale-model", "status": "responded", "raw_text": "old text",
         "parsed_items": [], "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
    ]
    ctx._last_scope_raw_result = {"model_id": "stale-scope", "status": "responded", "raw_text": "old"}
    ctx._review_degraded_reasons = ["stale-degraded"]
    ctx._last_triad_models = ["stale-triad-model"]
    ctx._last_scope_model = "stale-scope-model"
    ctx._review_advisory = []
    ctx.last_push_succeeded = False

    # Trigger the commit entrypoint and let it fail early (empty commit_message)
    result = git_mod._repo_commit_push(ctx, commit_message="")

    # Regardless of the outcome, the stale data must have been cleared
    assert ctx._last_triad_raw_results == [], (
        f"_last_triad_raw_results must be reset at commit start; got {ctx._last_triad_raw_results}"
    )
    assert ctx._last_scope_raw_result == {}, (
        f"_last_scope_raw_result must be reset at commit start; got {ctx._last_scope_raw_result}"
    )
    assert ctx._review_degraded_reasons == [], (
        f"_review_degraded_reasons must be reset at commit start; got {ctx._review_degraded_reasons}"
    )
    assert "ERROR" in result or "error" in result.lower()  # empty message triggers error




def test_empty_seat_response_distinct_from_error(tmp_path):
    """An empty seat answer is a parse failure of the seat (status='parse_failure',
    raw_text=''), distinct from a transport failure (status='error') — and never
    a PASS of either part."""
    from ouroboros.triad_review import parse_seat_answers

    parsed = parse_seat_answers({"results": [
        {"model": "test-model", "slot_id": "s1", "text": ""},
        {"model": "test-model", "slot_id": "s2", "verdict": "ERROR", "text": "Error: transport"},
    ]}, {"s1": ("change", "coupling"), "s2": ("change", "coupling")})
    empty, errored = (r.to_dict() for r in parsed.actor_records)
    assert empty["status"] == "parse_failure" and empty["raw_text"] == ""
    assert errored["status"] == "error" and "transport" in errored["raw_text"]
    assert parsed.findings == [] and parsed.responsive_models == []


# ── Test 11: scope_raw_result has parsed_items for shape parity ───────────────

# test_scope_raw_result_has_parsed_items removed in v5.15.x — the test
# replicated the dict-construction logic from parallel_review.py inline
# and asserted on the test's own output. It exercised no production code.
# Real shape parity between scope_raw and triad actor records is enforced
# by the integration tests in test_review_observability that consume
# `_last_scope_raw_result` after a real review run.
