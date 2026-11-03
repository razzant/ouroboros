"""Received outcomes, pending custody and informed author choices at real owners."""
import json
import pytest
from ouroboros.review_ledger import CouplingOutcome
from ouroboros.tools import git
from ouroboros.mutation_attribution import capture_mutation_baseline
from ouroboros.task_results import write_task_result
from ouroboros.review_state import load_state
from ouroboros.tools.scope_review_contract import COUPLING_QUESTION_IDS
from tests.test_git_review_preflight_gate import candidate  # noqa: F401


@pytest.mark.parametrize("basis", ["none", "unrelated_prior", "explicit_prior", "custody_lost"])
def test_commit_finish_requires_received_outcome(candidate, monkeypatch, basis):  # noqa: F811
    ctx = candidate
    monkeypatch.setenv('OUROBOROS_REVIEW_ENFORCEMENT', 'advisory')
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'pro')
    monkeypatch.setenv('OUROBOROS_REVIEW_MAX_CYCLES', 'unlimited')
    ctx.branch_dev = git.run_cmd(['git', 'branch', '--show-current'], cwd=ctx.repo_dir).strip()
    git.run_cmd(['git', 'reset', '--hard', 'HEAD'], cwd=ctx.repo_dir)
    (ctx.repo_dir / 'VERSION').write_text('1.0.0\n', encoding='utf-8')
    git.run_cmd(['git', 'add', 'VERSION'], cwd=ctx.repo_dir)
    git.run_cmd(['git', 'commit', '-m', 'fixture version'], cwd=ctx.repo_dir)
    write_task_result(ctx.drive_root, ctx.task_id, 'running')
    capture_mutation_baseline(ctx.drive_root, ctx.task_id, [{'surface_type':'system_repo','host_root':str(ctx.repo_dir)}])
    (ctx.repo_dir / 'change.py').write_text('value = 2\n', encoding='utf-8')
    monkeypatch.setattr(git, '_run_review_preflight_tests', lambda *_a, **_kw: None)
    monkeypatch.setattr(git, '_post_commit_result', lambda *_a, **_kw: None)
    monkeypatch.setattr(git, '_auto_push', lambda *_a, **_kw: '')
    calls = []
    def reviewer(_ctx, message, **kw):
        from ouroboros.review_dispatch import invoke_review_paid_stamp
        invoke_review_paid_stamp(ctx._review_paid_stamp)
        calls.append(kw["review_binding_fingerprint"])
        if basis in {"unrelated_prior", "explicit_prior"} and len(calls) == 1:
            ctx._last_triad_raw_results = [{"status": "responded", "raw_text": "Address the amount."}]
            ctx._last_review_critical_findings = [{"item": "amount", "severity": "critical"}]
            return "Critical feedback", CouplingOutcome(verdict="PASS", status="responded"), "critical_findings", []
        state = "custody_lost" if basis == "custody_lost" else "in_flight"
        ctx._last_triad_raw_results = [{'slot_id': 'critic', 'status':'pending', 'operation_state':state,'operation_id':'pending-critic','late_result_pending':True}]
        return 'All reviewers still running; no feedback received.', CouplingOutcome(status='pending'), 'infra_failure', []
    monkeypatch.setattr(git, '_run_parallel_review', reviewer)
    head = git.run_cmd(['git','rev-parse','HEAD'], cwd=ctx.repo_dir)
    if basis in {"unrelated_prior", "explicit_prior"}:
        prior = git._repo_commit_push(ctx, "Prior attempt", skip_advisory_review=True)
        assert "Review outcome returned before commit" in prior
        (ctx.repo_dir / "change.py").write_text("value = 3\n", encoding="utf-8")
    first = git._repo_commit_push(ctx, 'Fix amount', skip_advisory_review=True)
    reference = json.loads(first.split('\n',1)[1])['review_reference']
    if basis == "explicit_prior":
        reference = json.loads(prior.split('\n', 1)[1])['review_reference']
    row = load_state(ctx.drive_root).attempts[-1]
    from ouroboros.tools.preflight_review import _handle_review_status
    projected = json.loads(_handle_review_status(ctx))
    assert ("author_disposition" in projected["next_step"]) is (basis == "custody_lost")
    assert ("review_reference" in projected) is (basis == "custody_lost")
    assert row.status == 'reviewing' and row.late_result_pending
    assert not row.critical_findings
    second = git._repo_commit_push(ctx, 'Fix amount', review_reference=reference,
        author_disposition={'disposition':'accepted','rationale':'Proceed without any reviewer response.'})
    changed = git.run_cmd(['git','rev-parse','HEAD'],cwd=ctx.repo_dir) != head
    assert changed is (basis in {"explicit_prior", "custody_lost"}), second
    if basis in {"none", "unrelated_prior"}:
        assert "needs received feedback" in second
        assert "informed continuation needs received feedback" in first
    assert len(calls) == (2 if basis in {"unrelated_prior", "explicit_prior"} else 1)
    retained = next(item for item in load_state(ctx.drive_root).attempts if item.attempt == row.attempt)
    assert retained.triad_raw_results == row.triad_raw_results and retained.late_result_pending


def test_unavailable_triad_requires_author_handback(candidate, monkeypatch):  # noqa: F811
    ctx = candidate
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE','pro')
    monkeypatch.setenv('OUROBOROS_REVIEW_ENFORCEMENT','advisory')
    git._reset_commit_review_state(ctx)
    monkeypatch.setattr(git, '_run_review_preflight_tests', lambda *_a, **_kw: None)
    from ouroboros.tools import review
    monkeypatch.setattr(review, '_handle_multi_model_review', lambda *_a, **_kw: json.dumps({'error':'Review service unavailable.'}))
    def reviewer(_ctx, message, **kw):
        from ouroboros.review_dispatch import invoke_review_paid_stamp
        invoke_review_paid_stamp(ctx._review_paid_stamp)
        error = review._dispatch_unified_review(ctx, message, {
            'blocking_review':False,'prompt':'fixture','models':['critic'],'stable_prefix_len':0,
            'routes':['api_chat'],'session_task':'','target_repo':ctx.repo_dir,'row_plan':{},'retry_key':'fixture'})
        coupling = getattr(ctx, "_last_coupling_result", None) or CouplingOutcome()
        return error, coupling, ctx._last_review_block_reason, list(ctx._review_advisory)
    monkeypatch.setattr(git, '_run_parallel_review', reviewer)
    result = git._run_reviewed_stage_cycle(ctx,'Fix amount',0,skip_advisory_review=True,require_release_tag=False)
    assert result['status'] == 'reviewed', 'Unavailable triad proceeded to commit without first returning its failure for author choice'


from ouroboros.review_records import review_outcome_received


@pytest.mark.parametrize("rows,terminal,expected", [
    ([{"operation_state": "in_flight", "error": "logical wait ended"}], False, False),
    ([{"operation_state": "pending_dispatch", "status": "error"}], True, False),
    ([{"raw_results": [{"operation_state": "in_flight"}]}], True, False),
    ([{"operation_state": "in_flight"}, {"status": "responded"}], False, True),
    ([{"operation_state": "in_flight", "raw_text": "Pending host placeholder", "parsed": [{"item": "host note"}]}], False, False),
    ([{"operation_state": "in_flight"}, {"operation_state": "settled", "ok": True}], False, True),
    ([{"operation_state": "custody_lost", "late_result_pending": True}], False, True),
    ([{"operation_state": "settled", "error": "reviewer unavailable"}], False, True),
    ([], True, True),
    ([], False, False),
])
def test_received_outcome_keeps_live_custody_distinct(rows, terminal, expected):
    import copy
    before = copy.deepcopy(rows)
    assert review_outcome_received(rows, terminal=terminal) is expected
    assert rows == before


@pytest.mark.parametrize("kind", ["clean", "advisory"])
def test_clean_commit_path_and_structured_advice_are_distinct(candidate, monkeypatch, kind):  # noqa: F811
    ctx = candidate
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    git._reset_commit_review_state(ctx)
    monkeypatch.setattr(git, "_run_review_preflight_tests", lambda *a, **kw: None)
    finding = {"item": "observable review advice", "severity": "advisory", "reason": "Check this tradeoff"}
    def reviewer(*args, **kwargs):
        ctx._last_review_advisory_findings = [finding] if kind == "advisory" else []
        ctx._last_triad_raw_results = [{"status": "responded", "raw_text": "[]"}]
        return None, CouplingOutcome(verdict="PASS", status="responded"), "", []
    monkeypatch.setattr(git, "_run_parallel_review", reviewer)
    result = git._run_reviewed_stage_cycle(ctx, "Review changed candidate", 0,
        skip_advisory_pre_review=True, require_release_tag=False)
    # #1741: under Advisory a clean PASS also returns before Git; only its recorded advice differs.
    assert result["status"] == "reviewed"
    row = load_state(ctx.drive_root).attempts[-1]
    assert row.phase == "review_only" and bool(row.advisory_findings or ctx._last_review_advisory_findings) is (kind == "advisory")
    # Every dispatched wave leaves one durable review ledger record the result names.
    assert result["review_reference"]["review_record_id"]


# --- the five aggregate branches of the one wave (§1.7), tabular ------------------


_COUPLING_PASS = [{"item": item, "verdict": "PASS", "severity": "advisory", "reason": f"{item} checked against the diff and its callers"}
                  for item in COUPLING_QUESTION_IDS]
_CRITICAL = {"item": "secrets_check", "verdict": "FAIL", "severity": "critical", "reason": "a key is committed"}


def _two_part(change=(), coupling=_COUPLING_PASS, clean=None):
    change = list(change)
    return json.dumps({"change": change, "change_clean": (not change) if clean is None else clean, "coupling": list(coupling)})


def _seat(slot_id, model, text, **extra):
    return {"slot_id": slot_id, "model": model, "text": text, "verdict": "PASS", **extra}


_PANEL = {"models": ["openai/a", "anthropic/b", "google/c"], "routes": ["api_chat"] * 3, "efforts": ["high"] * 3,
          "session_targets": [""] * 3, "session_profiles": [""] * 3, "slot_ids": ["s1", "s2", "s3"],
          "subagent_ids": [""] * 3, "retrieves": [True, False, False], "use_local": [None] * 3,
          "parts": [("change", "coupling"), ("change",), ("change",)], "brief_shas": ["b1", "", ""]}

_BRANCHES = {
    # (seat results) -> (aggregate, reason, blocked, block_reason)
    "pass": ([_seat("s1", "openai/a", _two_part()), _seat("s2", "anthropic/b", "[]"), _seat("s3", "google/c", "[]")],
             ("PASS", "pass", False, "")),
    "fail_change": ([_seat("s1", "openai/a", _two_part()), _seat("s2", "anthropic/b", json.dumps([_CRITICAL])),
                     _seat("s3", "google/c", "[]")], ("FAIL", "critical_findings", True, "critical_findings")),
    "fail_coupling": ([_seat("s1", "openai/a", _two_part(coupling=[{**_COUPLING_PASS[1], "verdict": "FAIL", "severity": "critical",
                                                                      "reason": "docs/ARCHITECTURE.md still names the old module"},
                                                                     *_COUPLING_PASS[:1], *_COUPLING_PASS[2:]])),
                       _seat("s2", "anthropic/b", "[]"), _seat("s3", "google/c", "[]")],
                      ("FAIL", "critical_findings", True, "critical_findings")),
    "quorum_failed": ([_seat("s1", "openai/a", "", verdict="ERROR"), _seat("s2", "anthropic/b", json.dumps([_CRITICAL])),
                       _seat("s3", "google/c", "", verdict="ERROR")], ("QUORUM_FAILED", "review_quorum", True, "review_quorum")),
    "coupling_not_performed": ([_seat("s1", "openai/a", "[]"), _seat("s2", "anthropic/b", "[]"), _seat("s3", "google/c", "[]")],
                               ("NOT_PERFORMED", "coupling_not_performed", True, "coupling_not_performed")),
    "pending": ([_seat("s1", "openai/a", "", verdict="ERROR", operation_state="in_flight", late_result_pending=True),
                 _seat("s2", "anthropic/b", "[]"), _seat("s3", "google/c", "[]")],
                ("NOT_PERFORMED", "review_late_result_pending", True, "review_late_result_pending")),
}


@pytest.mark.parametrize("branch", sorted(_BRANCHES))
def test_the_gate_decides_each_aggregate_branch_once(candidate, monkeypatch, branch):  # noqa: F811
    """The one wave's five branches (§1.7) through the gate's dispatch, in the order
    the record reduces them: pending → QUORUM_FAILED (before FAIL, as the gate) →
    NOT_PERFORMED by coupling → FAIL → PASS. The same ``reduce_verdict`` decides
    the gate and the durable record."""
    from ouroboros.tools import review

    ctx = candidate
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    git._reset_commit_review_state(ctx)
    results, (aggregate, reason, blocked, block_reason) = _BRANCHES[branch]
    monkeypatch.setattr(review, "_handle_multi_model_review", lambda *a, **kw: json.dumps({"results": results}))
    prepared = {"blocking_review": True, "prompt": "fixture", "models": list(_PANEL["models"]), "stable_prefix_len": 0,
                "routes": list(_PANEL["routes"]), "session_task": "", "target_repo": ctx.repo_dir, "row_plan": dict(_PANEL),
                "retry_key": "fixture", "retrieving_manifests": [{"slot_id": "s1", "sha": {"brief": "b1"}, "governance_manifest": []}]}
    error = review._dispatch_unified_review(ctx, "Fix amount", prepared)
    verdict = ctx._last_review_verdict
    assert verdict["aggregate"] == aggregate and verdict["reason"] == reason, verdict
    assert bool(error) is blocked
    assert ctx._last_review_block_reason == block_reason
    coupling = ctx._last_coupling_result
    if branch == "fail_coupling":
        assert coupling.blocked and coupling.verdict == "FAIL" and coupling.critical_findings
        assert "forgotten_touchpoints" in error and "critical" in error.lower()
    if branch == "pass":
        assert coupling.status == "responded" and coupling.verdict == "PASS" and verdict["per_question"] == {"change": "PASS", "coupling": "PASS"}
    if branch == "quorum_failed":
        assert verdict["per_question"]["change"] == "FAIL", "the finding is recorded, the wave still has no quorum"
        assert "1 of 3" in error
    if branch == "coupling_not_performed":
        assert "Part 2" in error and "asked of: s1" in error
        assert verdict["per_question"] == {"change": "PASS", "coupling": "unanswered"}, "asked of one seat that left it out"
    # The record reduces the same rows to the same aggregate.
    from ouroboros import review_ledger as rl
    rows = rl.rows_from_plan(_PANEL, _PANEL["routes"], ctx._last_triad_raw_results)
    assert rl.reduce_verdict(rows, pending=branch == "pending")["aggregate"] == aggregate


def test_not_dispatched_is_the_first_branch():
    from ouroboros import review_ledger as rl

    refused = rl.reduce_verdict([], dispatch_refusal={"kind": "review_wave_budget_insufficient"})
    assert refused["aggregate"] == "NOT_DISPATCHED" and refused["reason"] == "dispatch_refusal"
    assert rl.reduce_verdict([])["reason"] == "nothing_dispatched"


# --- an uncounted coupling answer is named per seat, never hidden (D2-V5) ----------


_SEATS = (("s1", "openai/a"), ("s2", "anthropic/b"), ("s3", "google/c"))
_RETRIEVING_PANEL = {**_PANEL, "retrieves": [True] * 3, "parts": [("change", "coupling")] * 3, "brief_shas": ["b1", "b2", "b3"]}
_PACKET_PANEL = {**_PANEL, "retrieves": [False] * 3, "parts": [("change",)] * 3, "brief_shas": ["", "", ""]}
_REASONLESS = _two_part(coupling=[{**row, "reason": ""} for row in _COUPLING_PASS])


def _prepared(ctx, panel, *, blocking=True):
    return {"blocking_review": blocking, "prompt": "fixture", "models": list(panel["models"]), "stable_prefix_len": 0,
            "routes": list(panel["routes"]), "session_task": "", "target_repo": ctx.repo_dir, "row_plan": dict(panel),
            "retry_key": "fixture", "retrieving_manifests": [{"slot_id": s, "sha": {"brief": b}, "governance_manifest": []}
                                                             for s, b in zip(panel["slot_ids"], panel["brief_shas"]) if b]}


def _dispatch(ctx, monkeypatch, panel, results, *, blocking=True):
    from ouroboros.tools import review

    git._reset_commit_review_state(ctx)
    monkeypatch.setattr(review, "_handle_multi_model_review", lambda *a, **kw: json.dumps({"results": list(results)}))
    return review._dispatch_unified_review(ctx, "Fix amount", _prepared(ctx, panel, blocking=blocking))


@pytest.mark.parametrize("enforcement", ["blocking", "advisory"])
def test_three_uncountable_coupling_answers_are_named_per_seat(candidate, monkeypatch, enforcement):  # noqa: F811
    """Three retrieving seats each answer Part 2 with a matrix the gate cannot count
    (PASS rows without a reason). NOT_PERFORMED names every seat's own error — in the block
    (blocking) or in the author's advisory list beside the typed per-seat entries
    (advisory) — and does not send the owner to configure a retrieving seat: the
    pool has three. That advice belongs to a panel in which no seat read the work."""
    ctx = candidate
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    blocking = enforcement == "blocking"  # the assembled wave carries the enforcement into the dispatch
    error = _dispatch(ctx, monkeypatch, _RETRIEVING_PANEL, [_seat(s, m, _REASONLESS) for s, m in _SEATS], blocking=blocking)
    assert ctx._last_review_verdict["reason"] == "coupling_not_performed"
    text = error if enforcement == "blocking" else ctx._review_advisory[0]
    assert (error is None) is (enforcement == "advisory")
    assert "Part 2" in text and "asked of: s1, s2, s3" in text
    for seat_id, model in _SEATS:
        line = text.split(f"- {seat_id} ({model}): ", 1)
        assert len(line) == 2 and "missing reason" in line[1].split("\n- ")[0], text
    assert "configure a retrieving reviewer seat" not in text
    typed = ctx._last_review_advisory_findings
    assert [(f["item"], f["model"]) for f in typed] == [("review_coupling_unanswered", m) for _s, m in _SEATS]
    assert all("missing reason" in f["reason"] for f in typed)
    if enforcement == "advisory":
        assert ctx._review_advisory[1:] == typed, "the early branch keeps the wave's typed diagnostics for the author"
    # No seat could read the work: the same branch, and now the advice is the pool.
    error = _dispatch(ctx, monkeypatch, _PACKET_PANEL, [_seat(s, m, "[]") for s, m in _SEATS], blocking=blocking)
    text = error if enforcement == "blocking" else ctx._review_advisory[0]
    assert ctx._last_review_verdict["reason"] == "coupling_not_performed" and "asked of: no seat" in text
    assert "configure a retrieving reviewer seat" in text and ctx._last_review_advisory_findings == []


def test_a_critical_in_an_uncountable_matrix_stays_visible_under_an_aggregate_pass(candidate, monkeypatch):  # noqa: F811
    """One retrieving seat finds a critical coupling FAIL but writes a terse PASS on
    another item; the other two seats are clean. Part 2 reaches PASS on the two
    countable answers (contract: the broken matrix is ``unanswered``), the gate
    passes — and the discarded critical is still in the author's advisory list,
    in the record's advisory findings and on the seat's own ledger row."""
    ctx = candidate
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    broken = [{**_COUPLING_PASS[-1], "verdict": "FAIL", "severity": "critical",
               "reason": "the new helper bypasses the documented invariant on a live path"},
              {**_COUPLING_PASS[0], "reason": ""}, *_COUPLING_PASS[1:-1]]
    results = [_seat("s1", "openai/a", _two_part(coupling=broken)), _seat("s2", "anthropic/b", _two_part()),
               _seat("s3", "google/c", _two_part())]
    error = _dispatch(ctx, monkeypatch, _RETRIEVING_PANEL, results)
    verdict = ctx._last_review_verdict
    assert error is None and verdict["aggregate"] == "PASS" and verdict["per_question"]["coupling"] == "PASS"
    assert verdict["per_row"] == {"s1": "unanswered", "s2": "PASS", "s3": "PASS"}
    assert ctx._last_review_critical_findings == [], "nothing uncountable is counted"
    typed = ctx._last_review_advisory_findings
    by_item = {f["item"]: f for f in typed}
    assert set(by_item) == {"review_coupling_unanswered", "implicit_contracts"}
    discarded = by_item["implicit_contracts"]
    assert discarded["model"] == "openai/a" and discarded["severity"] == "advisory"
    assert "not counted" in discarded["reason"] and "critical FAIL" in discarded["reason"]
    assert "bypasses the documented invariant" in discarded["reason"]
    assert discarded in ctx._review_advisory, "the author's commit result carries it"
    # The seat's own answer keeps the row the gate could not count, into the ledger.
    from ouroboros import review_ledger as rl

    raw = next(r for r in ctx._last_triad_raw_results if r["slot_id"] == "s1")["answers"]["coupling"]
    assert raw["status"] == "unanswered" and raw["findings"] == [] and raw["critical"] == 0
    assert [(d["item"], d["severity"]) for d in raw["discarded"]] == [("implicit_contracts", "critical")]
    row = next(r for r in rl.rows_from_plan(_RETRIEVING_PANEL, _RETRIEVING_PANEL["routes"], ctx._last_triad_raw_results)
               if r["seat_id"] == "s1")
    assert row["answers"]["coupling"]["discarded"] == raw["discarded"] and row["critical_count"] == 0


@pytest.mark.parametrize("unreadable", [False, True], ids=["parsed_but_uncountable", "unreadable_text_control"])
def test_a_discarded_critical_reaches_the_author_when_the_whole_row_is_uncountable(candidate, monkeypatch,  # noqa: F811
                                                                                     unreadable):
    """The seat above also writes ``change: []`` with ``change_clean: false`` — a contradiction, so
    Part 1 is unanswered too and NO part of the row is countable: the record is ``parse_failure``
    (contract: the row stays whole, nothing of it is counted; quorum and the aggregate are the two
    clean seats'). The author still sees everything the seat said: ``review_model_parse_failure``
    for the row, the per-part ``review_<part>_unanswered`` errors, and the discarded critical FAIL
    with its reason — the same diagnostics a responded seat's uncountable part gets. Control: a seat
    whose text carries no object at all has no answers to diagnose and ends at the row-level entry."""
    ctx = candidate
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    broken = [{**_COUPLING_PASS[-1], "verdict": "FAIL", "severity": "critical",
               "reason": "the new helper bypasses the documented invariant on a live path"},
              {**_COUPLING_PASS[0], "reason": ""}, *_COUPLING_PASS[1:-1]]
    s1_text = "I could not finish reading the diff in time." if unreadable else _two_part(coupling=broken, clean=False)
    results = [_seat("s1", "openai/a", s1_text), _seat("s2", "anthropic/b", _two_part()), _seat("s3", "google/c", _two_part())]
    error = _dispatch(ctx, monkeypatch, _RETRIEVING_PANEL, results)
    verdict = ctx._last_review_verdict
    assert error is None and verdict["aggregate"] == "PASS" and verdict["per_question"]["coupling"] == "PASS"
    assert ctx._last_review_critical_findings == [], "nothing uncountable is counted"
    raw = next(r for r in ctx._last_triad_raw_results if r["slot_id"] == "s1")
    assert raw["status"] == "parse_failure"
    typed = ctx._last_review_advisory_findings
    by_item = {f["item"]: f for f in typed}
    assert all(f["model"] == "openai/a" and f["severity"] == "advisory" for f in typed)
    if unreadable:
        assert set(by_item) == {"review_model_parse_failure"} and raw["answers"] == {}
        return
    assert set(by_item) == {"review_model_parse_failure", "review_change_unanswered", "review_coupling_unanswered",
                            "implicit_contracts"}
    assert raw["answers"]["change"]["status"] == raw["answers"]["coupling"]["status"] == "unanswered"
    assert raw["answers"]["change"]["error"] in by_item["review_change_unanswered"]["reason"]
    assert raw["answers"]["coupling"]["error"] in by_item["review_coupling_unanswered"]["reason"]
    discarded = by_item["implicit_contracts"]
    assert "not counted" in discarded["reason"] and "critical FAIL" in discarded["reason"]
    assert "bypasses the documented invariant" in discarded["reason"]
    assert discarded in ctx._review_advisory, "the author's commit result carries it"
