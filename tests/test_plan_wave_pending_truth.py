"""An answer that has not arrived yet is a gap: not a failure and not a verdict.

``plan_task`` returns at the dispatch barrier, so a reviewer slot that has not
answered is stored as an error actor (``ok=False``, ``operation_state=
"pending_dispatch"``) and the wave is held fail-closed as an open ``DEGRADED``
with ``custody_pending``. That stored model is the FLOOR and these tests pin it
unchanged. What they pin as NEW is that every plan renderer reads the typed slot
census first: the owner line, the collect pre-line, the finalization disclosure,
the agent tool result, the advisory-open event and the acceptance exhibit. Every
guard is asserted in both directions — the awaiting wording AND the still-loud
real failure — so removing a guard turns its test red.

New module: ``tests/test_phase4_plan_review_continuity.py`` and
``tests/test_plan_review_w3.py`` sit near their size targets.
"""

from __future__ import annotations

import copy
import json

import pytest

from ouroboros.owner_hurry import force_plan_decision, plan_review_disclosure
from ouroboros.review_records import review_slot_awaiting, review_slot_unresolved
from ouroboros.tools import plan_review as pr
from ouroboros.tools.plan_render import (
    _actor_outcome, _degraded_replay_note, _next_step, _parse_plan_review_control, _render_wave,
)
from ouroboros.tools.plan_review_runtime import (
    plan_no_dispatch_line, plan_pending_actors, plan_wave_has_in_flight,
    plan_wave_line_has_news, plan_wave_progress_line, plan_wave_slot_census,
)
from tests.test_plan_finalization_collection import panel as _panel
from tests.test_plan_review_engine import CLEAN, DECK_SPEC, _call, _control, _finding, _state
from tests.test_plan_review_engine import harness as _engine_harness
from tests.test_plan_review_event_route import _mailbox_entries, _wait_until

harness = _engine_harness  # noqa: F811 - pytest fixture re-export
panel = _panel  # noqa: F811 - pytest fixture re-export

PENDING_ERROR = "Pending dispatch; the physical review operation is in flight (window 21600s)"
FP = "f" * 64


def _pending(slot, state="pending_dispatch", **extra):
    """The error actor ``review_custody`` mints for a slot with no answer yet."""
    return {"slot_id": slot, "model": "m", "route": "agent_session", "ok": False, "failure_code": "",
            "error": PENDING_ERROR, "operation_state": state, "late_result_pending": True,
            "operation_id": f"op-{slot}", **extra}


def _ok(slot):
    return {"slot_id": slot, "model": "m", "route": "api_chat", "ok": True, "error": None,
            "operation_state": "settled", "operation_id": f"op-{slot}"}


def _failed(slot, code="run_failed", error="harness unavailable"):
    return {"slot_id": slot, "model": "m", "route": "agent_session", "ok": False, "failure_code": code,
            "error": error, "operation_state": "late_settled", "operation_id": f"op-{slot}"}


def _skipped(slot):
    return {"slot_id": slot, "model": "m", "route": "agent_session", "ok": False,
            "failure_code": "subscription_window_exhausted", "error": "health_skip[...]: skipped at $0",
            "operation_state": "not_dispatched"}


def _wave(actors, *, pending=True, **extra):
    """A recorded wave shaped like ``synthesize_plan_review_wave`` leaves it."""
    parseable = sum(1 for a in actors if isinstance(a, dict) and a.get("ok"))
    return {
        "cycle_index": 1, "request_fingerprint": FP, "aggregate": "DEGRADED", "closed": False,
        "custody_pending": pending, "actors": actors, "findings": [],
        "counts": {"configured": len(actors), "parseable": parseable, "quorum": 2,
                   "blocking": 0, "note": 0, "need_evidence": 0, "blocking_slots": 0},
        "reasons": [f"slot_unparseable:{a['slot_id']}:{a['error']}" for a in actors
                    if isinstance(a, dict) and not a.get("ok")]
                   + [f"parseable_slots_below_quorum:{parseable}/2"]
                   + (["review_late_result_pending"] if pending else []),
        **extra,
    }


def _line(wave, *, aggregate="DEGRADED", cycles_paid=1, cap=3):
    return plan_wave_progress_line(aggregate, wave["counts"], cycles_paid=cycles_paid, cap=cap, wave=wave)


def _supplement(slot, state="late_settled", cycle=1):
    return {"slot_id": slot, "operation_id": f"op-{slot}", "operation_state": state, "cycle_index": cycle}


# ------------------------------------------------------------------ typed vocabulary


def test_awaiting_is_only_pending_dispatch_and_late_result_pending_never_means_a_planned_wait():
    assert review_slot_awaiting({"operation_state": "pending_dispatch"})
    for state in ("in_flight", "custody_lost"):
        row = {"operation_state": state, "late_result_pending": True}
        assert review_slot_unresolved(row) and not review_slot_awaiting(row)
    for row in ({"operation_state": "settled", "late_result_pending": True}, {"late_result_pending": True},
                {"operation_state": "not_dispatched"}, {"operation_state": "late_settled"}, {}, None, "pending_dispatch"):
        assert not review_slot_awaiting(row) and not review_slot_unresolved(row)


def test_census_puts_every_roster_row_in_exactly_one_typed_class():
    rows = [_ok("a"), _pending("w"), _pending("f", "in_flight"), _pending("l", "custody_lost"),
            _pending("u"), _skipped("k"), _failed("x"),
            {"slot_id": "legacy", "ok": False, "error": "transport died"},  # a pre-typed row is a failure
            {**_ok("stale"), "late_result_pending": True}]  # ok beside pending custody stays fail-closed
    wave = _wave(rows, historical_supplements=[
        _supplement("u"), _supplement("w", cycle=2), _supplement("f", state="in_flight")])
    census = plan_wave_slot_census(wave)
    names = ("answered", "awaiting", "unresolved", "uncollected", "skipped", "failed")
    assert {name: [r["slot_id"] for r in census[name]] for name in names} == {
        "answered": ["a"], "awaiting": ["w"], "unresolved": ["f", "l", "stale"],
        "uncollected": ["u"], "skipped": ["k"], "failed": ["x", "legacy"]}
    assert census["configured"] == len(rows) == sum(len(census[name]) for name in names)
    # ONE definition: the pending classes ARE plan_pending_actors, the same row objects.
    pending = plan_pending_actors(wave)
    assert [id(r) for r in pending] == [id(r) for r in rows if r["slot_id"] in {"w", "f", "l", "stale"}]
    assert {id(r) for r in census["awaiting"] + census["unresolved"]} == {id(r) for r in pending}
    assert wave["actors"] == rows and all(r is w for r, w in zip(rows, wave["actors"]))  # pure


@pytest.mark.parametrize("roster", [None, {}, "rows", 7, [], ["row", 3, None]])
def test_census_of_a_malformed_roster_classifies_nothing_and_never_raises(roster):
    census = plan_wave_slot_census({"actors": roster, "custody_pending": True, "paid": True})
    assert census == {"answered": [], "awaiting": [], "unresolved": [], "uncollected": [],
                      "skipped": [], "failed": [], "configured": 0}
    assert plan_wave_slot_census(None)["configured"] == 0


# ------------------------------------------------------------------ owner progress line


def test_progress_line_states_the_gap_for_each_open_branch():
    assert _line(_wave([_pending("s1"), _pending("s2"), _pending("s3")]), cycles_paid=0) == (
        "📐 Plan review: sent to 3 reviewers, none has answered yet.")
    assert _line(_wave([_ok("s1"), _pending("s2"), _pending("s3")])) == (
        "📐 Plan review so far: 1 of 3 reviewers answered.")
    assert _line(_wave([_ok("s1"), _failed("s2"), _skipped("s3"), _pending("s4")]), cap=None) == (
        "📐 Plan review so far: 1 of 4 reviewers answered, 1 didn't answer, 1 not sent.")
    # Nobody answered, but the roster is not only planned waits: the line never claims all were sent to.
    assert _line(_wave([_pending("s1"), _pending("s2"), _skipped("s3")]), cycles_paid=0) == (
        "📐 Plan review so far: 0 of 3 reviewers answered, 1 not sent.")
    # The raw custody state stays on the Reviews row; the line only counts the unresolved.
    assert _line(_wave([_ok("s1"), _pending("s2", "in_flight"), _pending("s3", "custody_lost")])) == (
        "📐 Plan review: 1 of 3 reviewers answered; 2 unresolved — no verdict.")
    late = _wave([_ok("s1"), _pending("s2"), _pending("s3")], pending=False,
                 historical_supplements=[_supplement("s2"), _supplement("s3", "settled")])
    # A settled slot may have settled as a failure: until collected it is "finished", never "answered".
    assert _line(late) == "📐 Plan review so far: 1 of 3 reviewers answered, 2 finished but not collected yet."
    # A plain in-flight line carries no paid-cycle and no declared-effort tail: the verdict line does.
    assert _line(_wave([_pending("s1")], reviewer_effort="high"), cycles_paid=0) == (
        "📐 Plan review: sent to 1 reviewer, none has answered yet.")
    # A roster of one keeps the singular in every sentence of the family.
    assert _line(_wave([_pending("s1", "in_flight")])) == (
        "📐 Plan review: 0 of 1 reviewer answered; 1 unresolved — no verdict.")
    assert _line(_wave([_ok("s1"), _pending("s2")], reviewer_effort="high")) == (
        "📐 Plan review so far: 1 of 2 reviewers answered.")


def test_an_open_line_carries_no_verdict_no_finding_count_and_no_failure_word_for_a_waiting_slot():
    counts = {"configured": 3, "parseable": 2, "quorum": 2, "blocking": 1, "note": 6, "need_evidence": 1}
    wave = {**_wave([_ok("s1"), _ok("s2"), _pending("s3")]), "counts": counts}
    line = plan_wave_progress_line("DEGRADED", counts, cycles_paid=1, cap=3, wave=wave)
    assert line == "📐 Plan review so far: 2 of 3 reviewers answered."
    open_lines = [line, _line(_wave([_pending("s1"), _pending("s2")]), cycles_paid=0),
                  _line(_wave([_ok("s1"), _pending("s2"), _pending("s3", "custody_lost")], reviewer_effort="high"))]
    for text in open_lines:  # no verdict, no finding count, no machine enum of a planned wait, no verdict-line tail
        for word in ("DEGRADED", "parseable", "untrusted", "blocking", "note", "need_evidence", "slot reasons",
                     "Pending dispatch", "failed", "late result pending", "pending_dispatch", "plan_task",
                     "cycles paid", "declared reviewer effort", "waiting", "run_failed", "in_flight",
                     "custody_lost", "settled", "not dispatched"):
            assert word not in text
        assert "\n" not in text and text.index("answered") < 80  # decisive words lead the row


def test_an_unresolved_slot_is_never_worded_as_waiting_and_a_waiting_slot_never_as_unresolved():
    unresolved = _line(_wave([_ok("s1"), _pending("s2"), _pending("s3", "custody_lost")]))
    # Beside an unresolved slot an awaited one is counted apart — awaited, never unresolved, never "waiting".
    assert unresolved == "📐 Plan review: 1 of 3 reviewers answered, 1 awaited; 1 unresolved — no verdict."
    for wait_word in ("waiting", "so far", "yet", "pending_dispatch", "custody_lost"):  # an exceptional state is never a planned wait
        assert wait_word not in unresolved
    mixed = _line(_wave([_pending("s1"), _failed("s2"), _skipped("s3"), _pending("s4", "in_flight"), _pending("s5")],
                        historical_supplements=[_supplement("s5")]))
    assert mixed == ("📐 Plan review: 0 of 5 reviewers answered, 1 awaited, 1 finished but not collected yet, "
                     "1 didn't answer, 1 not sent; 1 unresolved — no verdict.")
    waiting = _line(_wave([_ok("s1"), _pending("s2")]))
    assert "unresolved" not in waiting and "no verdict" not in waiting and waiting.startswith("📐 Plan review so far: ")


def test_a_real_failure_is_still_counted_while_others_are_awaited_and_its_reason_stays_off_the_line():
    wave = _wave([_pending("s1"), _failed("s2", "run_failed"), _failed("s3", "", "transport died"),
                  _skipped("s4"), _pending("s5", "in_flight"), _pending("s6")],
                 historical_supplements=[_supplement("s6")])
    line = _line(wave)
    assert "2 didn't answer, 1 not sent" in line and "run_failed" not in line and "transport died" not in line


def _settled_family_lines():
    """Every verdict-line shape of a fully collected wave, as (line, expected)."""
    counts = {"parseable": 3, "configured": 3, "blocking": 0, "note": 0, "need_evidence": 0}
    actors = [_ok("s1"), _ok("s2"), _ok("s3")]
    partial = {"configured": 3, "parseable": 1, "note": 4, "blocking": 0, "need_evidence": 0}
    yield (plan_wave_progress_line("GREEN", counts, cycles_paid=2, cap=3),
           "📐 Plan review: all 3 reviewers answered — no findings.")
    yield (plan_wave_progress_line("GREEN", counts, cycles_paid=2, cap=3, wave={"actors": actors, "custody_pending": False}),
           "📐 Plan review: all 3 reviewers answered — no findings.")
    yield (plan_wave_progress_line("REVIEW_REQUIRED", {**counts, "note": 2}, cycles_paid=2, cap=3),
           "📐 Plan review: all 3 reviewers answered — 2 notes, nothing blocking.")
    yield (plan_wave_progress_line("REVIEW_REQUIRED", {**counts, "note": 2, "need_evidence": 1}, cycles_paid=2, cap=3),
           "📐 Plan review: all 3 reviewers answered — 1 ask for evidence, 2 notes, nothing blocking.")
    yield (plan_wave_progress_line("REVISE_PLAN", {**counts, "blocking": 2, "note": 1}, cycles_paid=2, cap=3),
           "📐 Plan review: all 3 reviewers answered — 2 blocking findings, 1 note.")
    # The owner target: a wave whose reviewers died at the vendor still states who answered and what they said.
    yield (_line(_wave([_ok("s1"), _failed("s2", "run_failed"), _failed("s3", "", "transport died")],
                       pending=False, reviewer_effort="high", counts=partial), cap=None),
           "📐 Plan review: 1 of 3 reviewers answered — 4 notes, nothing blocking.")
    yield (_line(_wave([_ok("s1"), _failed("s2", "run_failed"), _skipped("s3")], pending=False, counts=partial)),
           "📐 Plan review: 1 of 3 reviewers answered, 1 not sent — 4 notes, nothing blocking.")
    yield (_line(_wave([_failed("s1"), _failed("s2"), _skipped("s3")], pending=False)),
           "📐 Plan review: none of the 3 reviewers answered, 1 not sent.")
    yield (_line(_wave([_ok("s1")], pending=False, counts={"configured": 1, "parseable": 1, "blocking": 0, "note": 0, "need_evidence": 0})),
           "📐 Plan review: the reviewer answered — no findings.")
    yield (_line(_wave([_failed("s1")], pending=False)), "📐 Plan review: the reviewer didn't answer.")
    # A custody-pending wave whose roster carries no typed custody keeps the verdict form and says a result is owed.
    untyped = {"actors": [{"slot_id": "s1", "ok": False, "error": "transport died"}], "custody_pending": True,
               "reviewer_effort": "high"}
    yield (plan_wave_progress_line("DEGRADED", {**counts, "parseable": 0, "configured": 1, "note": 1}, cycles_paid=1, cap=2, wave=untyped),
           "📐 Plan review: none of the 1 reviewers answered; a reviewer's answer is still on its way.")
    # A wave of typed $0 refusals keeps its own line: how many were not sent, and the earliest reset when known.
    yield (plan_no_dispatch_line(_wave([_skipped("s1")], pending=False)),
           "📐 Plan review: no reviewer could take the plan — 1 not sent.")
    yield (plan_no_dispatch_line({**_wave([_skipped("s1"), _skipped("s2")], pending=False), "earliest_reset": "2030-01-02T00:00:00Z"}),
           "📐 Plan review: no reviewer could take the plan — 2 not sent, earliest window reset 2030-01-02T00:00:00Z.")


def test_settled_waves_render_one_plain_family():
    for line, expected in _settled_family_lines():
        assert line == expected  # the counts are always stated: k of n, all n, none, or the one reviewer


def test_no_verdict_line_carries_an_internal_token():
    """Over every family row the harness produces, no verdict token, arithmetic tail,
    paid-cycle count, per-slot reason or machine enum reaches the owner (both directions:
    ``test_settled_waves_render_one_plain_family`` pins that the counts ARE still named)."""
    for line, _expected in _settled_family_lines():
        assert line.startswith("📐 Plan review: ") and "\n" not in line
        for token in ("DEGRADED", "GREEN", "REVISE_PLAN", "REVIEW_REQUIRED", "parseable", "untrusted", "cycles paid",
                      "slot reasons", "run_failed", "PLAN_REVIEW_", "need_evidence", "transport died",
                      "subscription_window_exhausted", "late result pending", "declared reviewer effort",
                      "not dispatched", "settled", " / "):
            assert token not in line, (token, line)


def test_the_wave_line_has_news_unless_the_roster_is_only_planned_waits():
    assert not plan_wave_line_has_news(_wave([_pending("s1"), _pending("s2"), _pending("s3")]))
    assert not plan_wave_line_has_news(_wave([_pending("s1")]))
    for roster in ([_ok("s1"), _pending("s2")], [_failed("s1"), _pending("s2")], [_skipped("s1"), _pending("s2")],
                   [_pending("s1", "in_flight"), _pending("s2")], [_pending("s1", "custody_lost")],
                   [_ok("s1"), _ok("s2")], [_failed("s1")], []):
        assert plan_wave_line_has_news(_wave(roster)), roster
    assert plan_wave_line_has_news(_wave([_pending("s1"), _pending("s2")], historical_supplements=[_supplement("s2")]))
    for unreadable in (None, {}, {"actors": "rows", "custody_pending": True}):
        assert plan_wave_line_has_news(unreadable)  # a roster the census cannot read is never silently withheld
    # The silent case is exactly the one sentence the dispatch line already said.
    for wave in (_wave([_pending("s1"), _pending("s2")]), _wave([_ok("s1"), _pending("s2")])):
        assert plan_wave_line_has_news(wave) != _line(wave).endswith("none has answered yet.")


# ------------------------------------------------------------------ finalization disclosure


def test_the_disclosure_names_no_verdict_token_while_reviewer_work_is_pending():
    base = {"required": True, "status": "open", "allow": True, "outcome": "DEGRADED",
            "enforcement": "advisory", "reviewer_slots_degraded": True}
    pending = plan_review_disclosure({**base, "custody_pending": True})
    assert "Plan review is still open (reviewer work is running or awaiting collection)" in pending
    assert "DEGRADED" not in pending and "no parseable reviewer quorum" not in pending
    owed = plan_review_disclosure({**base, "review_late_result_pending": True})
    assert "(reviewer work is running or awaiting collection)" in owed and "DEGRADED" not in owed
    assert "a late result is still owed" in owed
    assert "paid" not in owed  # a slot released at the barrier is $0 until its row proves the send
    # A settled panel with no quorum keeps its verdict token and its cause, byte for byte.
    assert plan_review_disclosure(base) == (
        "\n\n⚠️ Plan review is still open (DEGRADED; no parseable reviewer quorum); work proceeded "
        "under the owner-selected advisory enforcement.")


# ------------------------------------------------------------------ agent tool result


def test_actor_rows_word_a_gap_as_a_gap_and_a_failure_as_a_failure():
    assert _actor_outcome(_pending("s1"), "awaiting") == "NO ANSWER YET (pending_dispatch)"
    assert _actor_outcome(_pending("s1"), "uncollected") == "SETTLED — not collected yet"
    lost = {**_pending("s1", "custody_lost"), "error": "no actor record"}
    assert _actor_outcome(lost, "unresolved") == "NO ANSWER — custody_lost: no actor record"
    # No census class (a settled row): the FAILED forms are untouched.
    assert _actor_outcome(_ok("s1")) == "ok"
    assert _actor_outcome({"ok": False, "error": "transport died"}) == "FAILED: transport died"
    typed = {"ok": False, "failure_code": "subscription_window_exhausted", "reset_at": "2026-01-01T00:00:00Z",
             "error": "window spent"}
    assert _actor_outcome(typed) == "FAILED[subscription_window_exhausted] (resets 2026-01-01T00:00:00Z): window spent"
    assert _actor_outcome(_skipped("s1")).startswith("FAILED[subscription_window_exhausted]: health_skip")
    # A row that carries the engine's reported sentence quotes it in place of the error prose.
    words = "Selected model is at capacity. Please try a different model."
    dead = {"ok": False, "failure_code": "run_failed", "model": "codex=gpt-6-astra", "reported_cause": words,
            "error": 'delegated review session run-e23 ended failed: {"nextActions": ["Retry the run"]}'}
    assert _actor_outcome(dead) == f'FAILED[run_failed] — model=codex=gpt-6-astra; reported cause: "{words}"'
    assert _actor_outcome({**dead, "failure_code": "", "reset_at": "2030-01-01T00:00:00Z"}) == (
        f'FAILED[none] (resets 2030-01-01T00:00:00Z) — model=codex=gpt-6-astra; reported cause: "{words}"')
    assert _actor_outcome({**dead, "reported_cause": ""}) == f"FAILED[run_failed]: {dead['error']}"  # no words: today's bytes


@pytest.mark.parametrize("enforcement", ["blocking", "advisory"])
def test_an_awaiting_wave_renders_no_failure_no_verdict_and_keeps_the_control_footer(enforcement):
    wave = _wave([_ok("s1"), _pending("s2"), _failed("s3"), _pending("s4")],
                 historical_supplements=[{**_supplement("s4"), "source_ref": {}}])
    before = copy.deepcopy(wave)
    text = _render_wave(wave, cap=3, cycles_paid=1, enforcement=enforcement)
    assert wave == before, "rendering never rewrites the stored wave (its reasons are dialogue-free truth)"
    assert ("⚠️ REVIEW CUSTODY PENDING: 1 of 4 reviewer(s) have answered, 1 settled but not collected yet; "
            "no reviewer verdict exists yet — "
            "the DEGRADED aggregate below is a placeholder that keeps this wave open.") in text
    assert "received quorum" not in text and "paid reviewer" not in text
    assert "· NO ANSWER YET (pending_dispatch)" in text and "· SETTLED — not collected yet" in text
    assert "· FAILED[run_failed]: harness unavailable" in text  # the real failure stays loud
    assert "FAILED: Pending dispatch" not in text and "window 21600s" not in text
    assert "### Aggregate: no verdict — held open as DEGRADED (open)" in text
    assert ("Reasons: slot_unparseable:s3:harness unavailable, parseable_slots_below_quorum:1/2, "
            "review_late_result_pending, awaiting: s2, not collected yet: s4. Counts: ") in text
    assert _parse_plan_review_control(text) == ("DEGRADED", False)
    assert text.rstrip().endswith('PLAN_REVIEW_CONTROL_JSON: {"outcome":"DEGRADED","closed":false}')


@pytest.mark.parametrize("enforcement", ["blocking", "advisory"])
def test_the_custody_paragraph_states_facts_and_never_a_replay_recipe_or_an_exit(enforcement):
    wave = _wave([_pending("s1"), _pending("s2"), _pending("s3")])
    text = _next_step(wave, enforcement=enforcement, cap=2, cycles_paid=2)
    assert text.startswith("Open: one or more reviewer operations are still in flight. No reviewer verdict exists yet")
    assert "custody reconciliation" in text
    assert ("The host writes ONE message into this task's mailbox when every released slot settles: "
            "wait_task on this task's own id (wait_tasks while children run) returns on it") in text
    assert f"plan_task(review_disposition={{review_fingerprint: '{FP}', items: []}})" in text
    assert "The same $0 call at any earlier moment is legitimate" in text
    assert "never waits and never re-dispatches" in text
    assert "kept in the durable record and do not close this wave" in text
    if enforcement == "blocking":
        assert text.endswith("Blocking enforcement: the review must close before the work starts.")
        assert "Advisory enforcement" not in text
    else:
        assert text.endswith("Advisory enforcement: you may proceed with the review OPEN; the open review "
                             "stays typed in this task's state and result either way, and your own final "
                             "answer is where it is said in words.")
        assert "Blocking enforcement" not in text and "host discloses" not in text
    for forbidden in ("fresh panel", "RELEASED", "schedule_followup", "paid reviewer", "cycle cap is reached",
                      "DEGRADED: parseable reviewer verdicts", "A changed spec may start"):
        assert forbidden not in text


def test_the_custody_paragraph_counts_the_reviewers_that_did_not_answer():
    """One typed count line from the slot census (the same source the gate decision reads),
    with the dead buckets named; absent once every slot answered."""
    mixed = _next_step(_wave([_ok("s1"), _failed("s2"), _skipped("s3"), _pending("s4")]),
                       enforcement="advisory", cap=3, cycles_paid=1)
    assert "Reviewers: 1 of 4 answered; 1 did not answer, 1 not sent. " in mixed
    assert mixed.count("Reviewers:") == 1 and "run_failed" not in mixed
    awaited = _next_step(_wave([_ok("s1"), _pending("s2"), _pending("s3")]), enforcement="blocking", cap=3, cycles_paid=1)
    assert "Reviewers: 1 of 3 answered. " in awaited and "did not answer" not in awaited
    # Other direction: every slot answered (an untyped custody-pending roster) — no count line at all.
    answered = _next_step({**_wave([_ok("s1"), _ok("s2")]), "custody_pending": True},
                          enforcement="advisory", cap=3, cycles_paid=1)
    assert "Reviewers:" not in answered
    assert "Reviewers:" not in _next_step(_wave([_ok("s1"), _failed("s2")], pending=False),
                                          enforcement="advisory", cap=3, cycles_paid=1)  # a settled wave has its own paragraph


def test_an_unresolved_wave_and_a_structurally_unreachable_one_state_their_own_facts():
    lost = _next_step(_wave([_ok("s1"), _pending("s2", "custody_lost")]), enforcement="blocking", cap=3, cycles_paid=1)
    assert lost.startswith("Open: one or more reviewer operations have no recorded answer and their "
                           "custody is unresolved (typed state under Reviewer slots above).")
    assert "still in flight" not in lost
    dead = {**_wave([_skipped("s1"), _skipped("s2"), _pending("s3")]), "quorum_unreachable": True,
            "structurally_dead_slots": ["s1", "s2"], "earliest_reset": "2026-01-01T00:00:00Z"}
    text = _next_step(dead, enforcement="blocking", cap=3, cycles_paid=0)
    assert ("Quorum is STRUCTURALLY unreachable for this wave: slot(s) s1, s2 are window-spent, leaving fewer "
            "live slots than the quorum; earliest recorded reset 2026-01-01T00:00:00Z. ") in text
    # The gate releases finalization for an unreachable quorum whether or not a slot is awaited
    # (task_results.plan_review_gate_projection): the mind is told that fact, without a route.
    assert "finalization is RELEASED even while a slot is awaited" in text and "still in flight" in text
    assert "schedule_followup" not in text and text.rstrip().endswith("implementation still held.")
    advisory = _next_step(dead, enforcement="advisory", cap=3, cycles_paid=0)
    assert "RELEASED" not in advisory and advisory.endswith("your own final answer is where it is said in words.")
    assert "STRUCTURALLY" not in _next_step(_wave([_pending("s1")]), enforcement="blocking", cap=3, cycles_paid=0)


def test_a_settled_degraded_wave_keeps_its_failure_render_and_gains_the_whole_roster_fact():
    wave = _wave([_ok("s1"), _failed("s2"), {"slot_id": "s3", "model": "m", "ok": False, "error": "transport died"}],
                 pending=False)
    text = _render_wave(wave, cap=3, cycles_paid=1, enforcement="blocking")
    assert "REVIEW CUSTODY PENDING" not in text and "no verdict —" not in text
    assert "⚠️ DEGRADED: no parseable reviewer quorum — recorded as an OPEN wave; " in text
    assert "· FAILED[run_failed]: harness unavailable" in text and "· FAILED: transport died" in text
    assert "### Aggregate: DEGRADED (open)" in text
    assert ("Reasons: slot_unparseable:s2:harness unavailable, slot_unparseable:s3:transport died, "
            "parseable_slots_below_quorum:1/2. Counts: ") in text
    assert "DEGRADED: parseable reviewer verdicts 1 of 3 configured slot(s)" in text
    whole = ("every callable slot is asked again as the next paid cycle, slots that already answered "
             "included (health-skipped lanes stay $0 rows); no failed-slots-only path exists")
    empty_epoch = _degraded_replay_note({})
    assert f"an identical envelope re-dispatches a fresh panel ({whole})" in empty_epoch
    assert empty_epoch in text and "WHOLE" not in empty_epoch  # health-skipped lanes are not asked: no "whole roster"
    with_epoch = _degraded_replay_note({"health_epoch": [{"slot": "s1"}]})
    assert "re-dispatches a fresh panel" not in with_epoch and f"re-dispatches: {whole})" in with_epoch
    assert "asked again" not in _degraded_replay_note({}, paid_available=False)  # a spent cap dispatches nothing


# ------------------------------------------------------------------ durable truth


def test_the_advisory_open_event_carries_typed_custody_and_per_slot_state(tmp_path):
    from ouroboros.tools.plan_review_runtime import emit_plan_review_advisory_open

    ctx = type("Ctx", (), {"event_queue": None})()
    for fingerprint, wave in (("a" * 64, _wave([_ok("s1"), _pending("s2"), _pending("s3", "custody_lost")])),
                              ("b" * 64, _wave([_ok("s1"), _failed("s2")], pending=False))):
        emit_plan_review_advisory_open(ctx, tmp_path, task_id="t-event", cycles_paid=0, cap=3,
                                       wave={**wave, "request_fingerprint": fingerprint})
    rows = [json.loads(line) for line in
            (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    open_row, settled_row = [r for r in rows if r.get("type") == "plan_review_advisory_open"]
    assert open_row["aggregate"] == "DEGRADED" and open_row["custody_pending"] is True
    assert [(s["slot_id"], s["ok"], s["operation_state"]) for s in open_row["slots"]] == [
        ("s1", True, "settled"), ("s2", False, "pending_dispatch"), ("s3", False, "custody_lost")]
    assert settled_row["custody_pending"] is False
    assert [s["operation_state"] for s in settled_row["slots"]] == ["settled", "late_settled"]
    assert [s["failure_code"] for s in settled_row["slots"]] == ["", "run_failed"]


@pytest.mark.parametrize("pending", [True, False])
def test_the_acceptance_exhibit_of_an_open_wave_carries_typed_custody(pending, tmp_path):
    import types

    from ouroboros.review_evidence import build_task_acceptance_evidence
    from ouroboros.task_results import STATUS_RUNNING, record_plan_review_wave, write_task_result
    from tests.test_acceptance_claims_wiring import _v2_wave

    root = tmp_path
    write_task_result(root, "acc", STATUS_RUNNING, result="running")
    record_plan_review_wave(root, "acc", {
        **_v2_wave("a" * 64, ["unreviewed claim"], aggregate="DEGRADED", closed=False),
        "custody_pending": pending, "paid": False})
    ctx = types.SimpleNamespace(task_contract={"requirements": "do X"}, task_metadata={},
                                drive_root=str(root), task_id="acc", repo_dir=str(root))
    exhibit = build_task_acceptance_evidence(
        ctx, llm_trace={"tool_calls": []}, drive_root=root, task_id="acc")["plan_claims_exhibit"]
    assert exhibit["binding"] == "not bound: wave open" and exhibit["aggregate"] == "DEGRADED"
    assert exhibit["custody_pending"] is pending


# ------------------------------------------------------------------ the floor, through the real substrate


def test_a_fully_awaiting_wave_keeps_the_fail_closed_floor_and_every_line_tells_the_truth(harness, panel):
    ctx = harness.make_ctx()
    first = _call(ctx)
    assert _wait_until(lambda: sum(e.execute_calls for e in panel.values()) == 3)
    at_dispatch = [line for line in harness.progress if line.startswith("📐")]
    state = _state(harness)
    wave = state["waves"][-1]
    # FLOOR: the stored model and every gate input are exactly what they were.
    assert (wave["aggregate"], wave["closed"], wave["custody_pending"], wave["paid"]) == ("DEGRADED", False, True, False)
    assert wave["counts"]["parseable"] == 0 and wave["counts"]["configured"] == 3 and state["cycles_paid"] == 0
    assert wave["actors_degraded"] == ["s1", "s2", "s3"] and plan_wave_has_in_flight(wave)
    assert [r.split(":")[0] for r in wave["reasons"]] == [
        "slot_unparseable", "slot_unparseable", "slot_unparseable", "parseable_slots_below_quorum",
        "review_late_result_pending"]
    for actor in wave["actors"]:
        assert actor["ok"] is False and actor["error"].startswith("Pending dispatch;")
        assert (actor["operation_state"], actor["late_result_pending"], actor["failure_code"]) == (
            "pending_dispatch", True, "")
    assert _control(first) == {"outcome": "DEGRADED", "closed": False}
    decision = force_plan_decision(ctx, {}, enforcement="blocking")
    assert decision["allow"] is False and decision["custody_pending"] and decision["reviewer_slots_degraded"]
    refused = pr._handle_plan_task(ctx, review_disposition={
        "review_fingerprint": wave["request_fingerprint"], "items": [], "author_action": "finish",
        "author_disposition": {"disposition": "accepted", "rationale": "Proceed without the reviewers."}})
    assert "reviewers are still running" in refused
    assert not (_state(harness)["current_attempt"] or {}).get("author_subject")
    # TRUTH: a paid dispatch has its own line, and a fresh roster of planned waits adds no wave-state
    # line under it (the dispatch line and the per-slot started rows already said it); the text names no failure.
    assert at_dispatch == ["📐 Plan review: sending the plan to 3 reviewers (round 1 of 2, blocking)…"]
    assert harness.progress[0] == at_dispatch[0]
    assert first.count("· NO ANSWER YET (pending_dispatch)") == 3 and "FAILED" not in first
    assert "0 of 3 reviewer(s) have answered" in first and "awaiting: s1, s2, s3." in first
    # A $0 collection dispatches nothing, so it never reads as a plan being sent — and a collection
    # always prints the wave state, the same roster of planned waits included.
    mark = len(harness.progress)
    collect = {"review_disposition": {"review_fingerprint": wave["request_fingerprint"], "items": []}}
    early = pr._handle_plan_task(ctx, **collect)
    assert [line for line in harness.progress[mark:] if line.startswith("📐")] == [
        "📐 Plan review: checking for reviewer answers…",
        "📐 Plan review: sent to 3 reviewers, none has answered yet."]
    assert _control(early) == {"outcome": "DEGRADED", "closed": False}
    panel["s1"].release.set()
    assert _wait_until(lambda: any("m/a answered" in line for line in harness.progress))
    mark = len(harness.progress)
    partial = pr._handle_plan_task(ctx, **collect)
    emitted = [line for line in harness.progress[mark:] if line.startswith("📐")]
    assert emitted == ["📐 Plan review: checking for reviewer answers…",
                       "📐 Plan review so far: 1 of 3 reviewers answered."]
    assert "1 of 3 reviewer(s) have answered" in partial and partial.count("NO ANSWER YET") == 2
    assert _control(partial) == {"outcome": "DEGRADED", "closed": False}
    assert sum(e.execute_calls for e in panel.values()) == 3  # nothing was re-sent
    # FLOOR: two clean answers meet the arithmetic quorum, and the wave still may not read GREEN
    # while the third slot can land a blocker — it stays the open DEGRADED placeholder.
    panel["s2"].release.set()
    assert _wait_until(lambda: any("m/b answered" in line for line in harness.progress))
    quorum = pr._handle_plan_task(ctx, **collect)
    held = _state(harness)["waves"][-1]
    assert (held["aggregate"], held["closed"], held["custody_pending"]) == ("DEGRADED", False, True)
    assert held["counts"]["parseable"] == 2 and _control(quorum) == {"outcome": "DEGRADED", "closed": False}
    assert force_plan_decision(ctx, {}, enforcement="blocking")["allow"] is False
    assert "📐 Plan review so far: 2 of 3 reviewers answered." in harness.progress
    assert "### Aggregate: no verdict — held open as DEGRADED (open)" in quorum and "GREEN" not in quorum
    for executor in panel.values():
        executor.release.set()
    assert _wait_until(lambda: len(_mailbox_entries(ctx.drive_root, ctx.task_id)) == 1)
    mark = len(harness.progress)
    final = pr._handle_plan_task(ctx, **collect)
    assert _control(final) == {"outcome": "GREEN", "closed": True}
    assert [line for line in harness.progress[mark:] if line.startswith("📐")] == [
        "📐 Plan review: checking for reviewer answers…",
        "📐 Plan review: all 3 reviewers answered — no findings."]
    # ONE family: every 📐 line of the whole card opens with the same words.
    assert all(line.startswith("📐 Plan review") for line in harness.progress if line.startswith("📐"))


def test_a_fresh_dispatch_with_an_immediate_refusal_prints_the_wave_line_and_keeps_the_floor(harness, panel, monkeypatch):
    import ouroboros.tools.plan_review_runtime as runtime

    monkeypatch.setattr(runtime, "plan_panel_health_snapshot", lambda _slots: {
        "s3": {"failure_code": "subscription_window_exhausted", "reset_at": "2030-01-02T00:00:00+00:00"}})
    ctx = harness.make_ctx()
    first = _call(ctx)
    assert _wait_until(lambda: sum(e.execute_calls for e in panel.values()) == 2)
    wave = _state(harness)["waves"][-1]
    # FLOOR: withholding or printing a line never touches the stored wave or the control footer.
    assert (wave["aggregate"], wave["closed"], wave["custody_pending"]) == ("DEGRADED", False, True)
    assert [a["operation_state"] for a in wave["actors"]] == ["pending_dispatch", "pending_dispatch", "not_dispatched"]
    assert wave["actors_degraded"] == ["s1", "s2", "s3"] and _control(first) == {"outcome": "DEGRADED", "closed": False}
    # A lane refused at $0 is news the dispatch line could not carry in full: the wave line is printed at once.
    assert [line for line in harness.progress if line.startswith("📐")] == [
        "📐 Plan review: sending the plan to 2 reviewers (round 1 of 2, blocking); 1 lane skipped at $0…",
        "📐 Plan review so far: 0 of 3 reviewers answered, 1 not sent."]


def test_a_dispatch_that_reaches_no_lane_never_says_the_plan_is_being_sent(harness, panel, monkeypatch):
    import ouroboros.tools.plan_review_runtime as runtime

    monkeypatch.setattr(runtime, "plan_panel_health_snapshot", lambda _slots: {
        slot: {"failure_code": "subscription_window_exhausted", "reset_at": "2030-01-02T00:00:00+00:00"}
        for slot in ("s1", "s2", "s3")})
    _call(harness.make_ctx())
    assert sum(e.execute_calls for e in panel.values()) == 0
    first = [line for line in harness.progress if line.startswith("📐")][0]
    assert first == "📐 Plan review: no reviewer lane can take the plan (round 1 of 2, blocking); 3 lanes skipped at $0."
    assert "sending" not in first


def test_every_owner_line_of_the_organ_opens_with_the_one_prefix(harness, monkeypatch):
    harness.install({"s1": json.dumps([_finding("n1", "note")]), "s2": CLEAN, "s3": CLEAN})
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "unlimited")
    system_target = {**DECK_SPEC, "affected_paths": [str(harness.system / "ouroboros" / "loop.py")]}
    _call(harness.make_ctx(task_id="task-c"), spec=system_target)
    assert harness.progress == [  # no cap prints a bare round; a constitutional plan says so inside the parentheses
        "📐 Plan review: sending the plan to 3 reviewers (round 1, blocking, constitutional)…",
        "📐 Plan review: all 3 reviewers answered — 1 note, nothing blocking."]
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    harness.progress.clear()
    ctx = harness.make_ctx()
    _call(ctx)
    pr._handle_plan_task(ctx, review_disposition={
        "review_fingerprint": _state(harness)["waves"][-1]["request_fingerprint"],
        "items": [{"finding_id": "s1:n1", "decision": "accept", "rationale": "will do"}]})
    _call(ctx, spec={**DECK_SPEC, "in_scope": ["a 6-slide deck"]})  # a revised envelope at the spent cap
    assert harness.progress == [
        "📐 Plan review: sending the plan to 3 reviewers (round 1 of 1, blocking)…",
        "📐 Plan review: all 3 reviewers answered — 1 note, nothing blocking.",
        "📐 Plan review: findings answered — review closed.",
        "📐 Plan review: no review rounds left — 1 of 1 used (blocking)."]


def test_an_identical_envelope_over_an_uncollected_wave_says_it_collects_and_sends_nothing(harness, panel):
    ctx = harness.make_ctx()
    _call(ctx)
    assert _wait_until(lambda: sum(e.execute_calls for e in panel.values()) == 3)
    assert harness.progress[0].startswith("📐 Plan review: sending the plan to 3 reviewers")  # a paid dispatch
    for executor in panel.values():
        executor.release.set()
    assert _wait_until(lambda: len(_mailbox_entries(ctx.drive_root, ctx.task_id)) == 1)
    mark = len(harness.progress)
    resumed = _call(ctx)  # the identical envelope only reconciles the recorded wave
    lines = [line for line in harness.progress[mark:] if line.startswith("📐")]
    assert lines == ["📐 Plan review: checking for reviewer answers…",
                     "📐 Plan review: all 3 reviewers answered — no findings."]
    assert _control(resumed) == {"outcome": "GREEN", "closed": True}
    assert sum(e.execute_calls for e in panel.values()) == 3  # nothing was re-sent


def test_cyber_pro_is_told_the_custody_facts_of_an_awaiting_wave_and_keeps_its_own_words_otherwise(monkeypatch):
    from ouroboros.tools import plan_render

    monkeypatch.setattr(plan_render, "review_enforcement_blocks", lambda _mode: False)  # Cyber Pro: nothing blocks
    awaited = _next_step(_wave([_ok("s1"), _pending("s2")]), enforcement="advisory", cap=3, cycles_paid=1)
    assert awaited.startswith("Open: one or more reviewer operations are still in flight. No reviewer verdict exists yet")
    assert "never waits and never re-dispatches" in awaited and "wait_task" in awaited
    assert awaited.endswith("Cyber Pro: Ouroboros decides whether and how to continue; "
                            "continuation does not close the wave or create a PASS.")
    assert "recorded verdict" not in awaited and "Advisory enforcement" not in awaited and "Blocking" not in awaited
    settled = _next_step(_wave([_ok("s1"), _failed("s2")], pending=False), enforcement="advisory", cap=3, cycles_paid=1)
    assert settled.startswith("Cyber Pro: Ouroboros decides whether and how to continue. The recorded verdict")


def test_a_free_historical_read_never_promises_a_collection():
    wave = _wave([_ok("s1"), _pending("s2")], pending=False, historical_supplements=[_supplement("s2")])
    live = _render_wave(wave, cap=3, cycles_paid=1, enforcement="advisory")
    assert "· SETTLED — not collected yet" in live and "not collected yet: s2" in live
    read = _render_wave(wave, cap=3, cycles_paid=1, enforcement="advisory", historical_feedback=[])
    assert "· SETTLED LATE — its answer is under Historical feedback below" in read and "settled late: s2" in read
    assert "not collected yet" not in read
