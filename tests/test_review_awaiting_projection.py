"""A reviewer slot released at the dispatch barrier is a gap on the shared review
substrate: never a transport failure, a malformed answer or a verdict.

Two halves, both directions each. The WORDS about an awaited slot change (the
per-actor projection, the panel fold, the aggregate's reasons).
The ARITHMETIC does not: aggregate, quorum and the per-actor gate fields equal the
literal values pinned below for every input. Canonical unfinished workers remain
pending after a drain expires; lost custody, settled failures and refusals remain typed. A slot
is awaited only while it carries no answer: a row that carries one is judged by its
answer exactly as before, words included, whatever its custody state says.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros.review_seat_outcome import aggregate_outcomes, finalize_seat_outcome
from ouroboros.review_projection import (
    AWAITING_PROJECTION,
    _review_actor_projection,
    compact_review_projection,
)
from ouroboros.review_records import (
    ReviewActorRecord,
    review_outcome_received,
    review_slot_awaiting,
)

PASS_TEXT = json.dumps({
    "verdict": "PASS", "outcome_tier": "solved", "summary": "done",
    "criteria_used": [{"criterion": "works", "status": "supported", "evidence_refs": ["tool:1"]}],
    "findings": [],
})
FAIL_TEXT = json.dumps({
    "verdict": "FAIL", "outcome_tier": "best_effort", "summary": "broken", "completion_coach": "fix x",
    "criteria_used": [{"criterion": "works", "status": "rejected"}],
    "findings": [{"severity": "critical", "item": "x is broken", "recommendation": "fix x"}],
})
DEGRADED_TEXT = json.dumps({"verdict": "DEGRADED", "summary": "cannot judge", "findings": []})
PENDING_ERROR = "Pending dispatch; the physical review operation is in flight (window 21600s)"
TIMEOUT_ERROR = "Timeout after 1800s; physical review operation remains in flight"
CUSTODY_ERROR = "Exact review custody is unavailable; refusing a second paid dispatch"
REFUSAL_ERROR = "preflight_oversize: assembled acceptance prompt exceeds this slot's cap"
RUN_FAILED_ERROR = "delegated review session ended failed: harness_unavailable"
AWAITING_REASON = "No answer recorded: the host returned at the dispatch barrier before this reviewer answered."


def row(shape: str, slot: str) -> ReviewActorRecord:
    """One actor per constructor shape of the review substrate and custody."""
    def make(**fields):
        return ReviewActorRecord(slot_id=slot, model="m/" + slot, **fields)

    op = "op-" + slot
    pending = dict(status="error", operation_id=op, operation_state="pending_dispatch",
                   late_result_pending=True, error=PENDING_ERROR)
    shapes = {
        "pass": lambda: make(status="ok", raw_text=PASS_TEXT, operation_id=op),
        "fail": lambda: make(status="ok", raw_text=FAIL_TEXT, operation_id=op),
        "degraded": lambda: make(status="ok", raw_text=DEGRADED_TEXT, operation_id=op),
        "pending": lambda: make(**pending),
        "timeout": lambda: make(status="error", operation_id=op, operation_state="in_flight",
                                late_result_pending=True, error=TIMEOUT_ERROR),
        "custody_lost": lambda: make(status="error", operation_state="custody_lost",
                                     late_result_pending=True, error=CUSTODY_ERROR),
        "not_dispatched": lambda: make(status="not_dispatched", operation_id=op,
                                       operation_state="not_dispatched", error=REFUSAL_ERROR),
        "run_failed": lambda: make(status="error", operation_id=op, operation_state="settled",
                                   transport_status="provider_transport_error",
                                   failure_code="run_failed", error=RUN_FAILED_ERROR),
        "empty": lambda: make(status="empty", raw_text="", operation_id=op),
        "malformed": lambda: make(status="ok", raw_text="I think it is fine.", operation_id=op),
        # What a frozen commit reservation would look like as a substrate actor.
        "reservation": lambda: make(status="error", error="", operation_id=op,
                                    operation_state="in_flight", late_result_pending=True),
        "late_settled_pass": lambda: make(status="ok", raw_text=PASS_TEXT, operation_id=op,
                                          operation_state="late_settled"),
        # No constructor mints the shapes below: a pending-dispatch row that carries an answer.
        "held_raw_pass": lambda: make(**pending, raw_text=PASS_TEXT),
        "held_raw_fail": lambda: make(**pending, raw_text=FAIL_TEXT),
        "held_raw_prose": lambda: make(**pending, raw_text="I think it is fine."),
        "ok_pending": lambda: make(status="ok", raw_text=PASS_TEXT, operation_id=op,
                                   operation_state="pending_dispatch", late_result_pending=True),
    }
    return shapes[shape]()


POLICIES = {
    "fail_closed": ("plan_review", {"min_successful_slots": 2, "fail_closed_on_errors": True}),
    "open": ("skill_review", {"min_successful_slots": 2}),
    "quorum3": ("plan_review", {"min_successful_slots": 3, "fail_closed_on_errors": True}),
    "quorum0": ("scope_review", {"min_successful_slots": 0}),
    "quorum_negative": ("skill_review", {"min_successful_slots": -2}),
    "quorum9": ("scope_review", {"min_successful_slots": 9}),
}

# The gate-visible row per shape and the aggregate per (case, policy), captured by running
# the sprint base commit on these fixtures: (signal, semantic_verdict, enforcement_impact,
# quorum_contribution). An awaited slot changes words only, so every value here still holds.
BASE_ROW = {
    **{shape: ("PASS", "PASS", "supports_pass", True) for shape in (
        "pass", "late_settled_pass", "held_raw_pass", "ok_pending")},
    "fail": ("FAIL", "FAIL", "veto", True),
    "held_raw_fail": ("FAIL", "FAIL", "veto", True),
    "degraded": ("DEGRADED", "DEGRADED", "abstains", False),
    **{shape: ("DEGRADED", "", "abstains", False) for shape in (
        "pending", "timeout", "custody_lost", "not_dispatched", "run_failed", "empty",
        "malformed", "reservation", "held_raw_prose")},
}
P, F, D = "PASS", "FAIL", "DEGRADED"
BASE_AGGREGATE = {  # columns follow POLICIES
    ("pass", "pass", "pass"): (P, P, P, P, P, D),
    ("pass", "pending", "pending"): (D, D, D, P, P, D),
    ("pass", "pass", "pending"): (D, P, D, P, P, D),
    ("fail", "pending", "pending"): (F, F, F, F, F, F),
    ("pending", "pending", "pending"): (D, D, D, D, D, D),
    ("pending", "not_dispatched", "not_dispatched"): (D, D, D, D, D, D),
    ("run_failed", "pending", "pending"): (D, D, D, D, D, D),
    ("degraded", "pending", "pending"): (D, D, D, D, D, D),
    ("timeout", "pending", "pending"): (D, D, D, D, D, D),
    ("custody_lost", "pending", "pending"): (D, D, D, D, D, D),
    ("reservation", "reservation", "reservation"): (D, D, D, D, D, D),
    ("pass", "custody_lost", "custody_lost"): (D, D, D, P, P, D),
    ("pass", "pass", "custody_lost"): (D, P, D, P, P, D),
    ("timeout", "timeout", "timeout"): (D, D, D, D, D, D),
    ("pass", "pass", "timeout"): (D, P, D, P, P, D),
    ("not_dispatched", "not_dispatched", "not_dispatched"): (D, D, D, D, D, D),
    ("pass", "run_failed", "run_failed"): (D, D, D, P, P, D),
    ("pass", "pass", "empty"): (D, P, D, P, P, D),
    ("pass", "pass", "malformed"): (P, P, D, P, P, D),
    ("late_settled_pass", "late_settled_pass", "late_settled_pass"): (P, P, P, P, P, D),
    ("late_settled_pass", "late_settled_pass", "pending"): (D, P, D, P, P, D),
    (): (D, D, D, D, D, D),
    # A pending-dispatch row that carries an answer votes exactly as the base votes it.
    ("pass", "held_raw_pass", "pending"): (D, P, D, P, P, D),
    ("pass", "held_raw_pass", "pass"): (D, P, D, P, P, D),
    ("pass", "held_raw_pass", "run_failed"): (D, P, D, P, P, D),
    ("pass", "pass", "held_raw_fail"): (F, F, F, F, F, F),
    ("pass", "pass", "held_raw_prose"): (D, P, D, P, P, D),
    ("pass", "ok_pending", "pending"): (D, P, D, P, P, D),
    ("ok_pending", "pass", "pass"): (P, P, P, P, P, D),
}
INPUT_FIELDS = ("status", "operation_state", "late_result_pending", "error", "failure_code",
                "operation_id", "raw_text")


def aggregate(shapes, policy="open"):
    surface, settings = POLICIES[policy]
    actors = [row(shape, f"s{index + 1}") for index, shape in enumerate(shapes)]
    slots = [SimpleNamespace(slot_id=actor.slot_id, role_hint="") for actor in actors]
    # These historical object assessments exercise the compatibility projections;
    # live array/two-part surfaces are covered by test_review_seat_outcome.
    request = SimpleNamespace(surface="task_acceptance", policy=dict(settings), resolved_wave_id="", task_id="test")
    for actor, slot in zip(actors, slots):
        finalize_seat_outcome(request, slot, actor)
    result = aggregate_outcomes(request, slots, actors)
    return result, actors


def panel_of(shapes, policy="open", **run_fields):
    surface, settings = POLICIES[policy]
    result, actors = aggregate(shapes, policy)
    run = {"request": {"surface": surface, "policy": dict(settings)}, "authority": "host_root",
           "actors": [dataclasses.asdict(actor) for actor in actors], **result, **run_fields}
    return compact_review_projection([run])["panels"][0]


# ── the per-actor projection ────────────────────────────────────────────────

def test_an_awaited_row_projects_a_gap_and_keeps_its_custody_identity():
    actor = row("pending", "s1")
    truth = _review_actor_projection(actor, "task_acceptance")
    assert truth["transport_status"] == truth["parse_status"] == AWAITING_PROJECTION == "awaiting"
    assert truth["semantic_verdict"] == "" and truth["quorum_contribution"] is False
    assert truth["reason"] == AWAITING_REASON
    assert "Pending dispatch" not in truth["reason"] and "window" not in truth["reason"]
    assert "yet" not in truth["reason"]
    assert (truth["operation_state"], truth["late_result_pending"], truth["operation_id"]) == (
        "pending_dispatch", True, "op-s1")
    # The stored row is the floor: the projection never rewrites it.
    assert (actor.status, actor.error, actor.failure_code) == ("error", PENDING_ERROR, "")


@pytest.mark.parametrize("shape, transport, parse, reason", [
    ("timeout", "timeout", "malformed", TIMEOUT_ERROR),
    ("custody_lost", "provider_transport_error", "malformed", CUSTODY_ERROR),
    ("run_failed", "provider_transport_error", "malformed", RUN_FAILED_ERROR),
    ("not_dispatched", "not_dispatched", "malformed", REFUSAL_ERROR),
    ("reservation", "provider_transport_error", "malformed", "Reviewer response was malformed or absent."),
    ("empty", "success", "malformed", "Reviewer response was malformed or absent."),
])
def test_a_row_that_is_not_awaited_keeps_its_failure_words(shape, transport, parse, reason):
    truth = _review_actor_projection(row(shape, "s1"), "task_acceptance")
    assert (truth["transport_status"], truth["parse_status"], truth["reason"]) == (transport, parse, reason)
    assert AWAITING_PROJECTION not in (truth["transport_status"], truth["parse_status"])


def test_the_typed_state_outranks_words_stored_by_an_earlier_projection():
    stored = {"slot_id": "t", "model": "m", "status": "error", "error": PENDING_ERROR,
              "transport_status": "provider_transport_error", "parse_status": "malformed",
              "reason": PENDING_ERROR, "operation_state": "pending_dispatch", "late_result_pending": True}
    truth = _review_actor_projection(stored, "task_acceptance")
    assert (truth["transport_status"], truth["parse_status"], truth["reason"]) == (
        "awaiting", "awaiting", AWAITING_REASON)
    # The derived word never outlives the wait: a settled answer is projected afresh.
    settled = {"slot_id": "t", "model": "m", "status": "ok", "transport_status": "awaiting",
               "parse_status": "awaiting", "operation_state": "late_settled",
               "raw_text": PASS_TEXT, "parsed": json.loads(PASS_TEXT)}
    truth = _review_actor_projection(settled, "task_acceptance")
    assert (truth["transport_status"], truth["parse_status"], truth["semantic_verdict"]) == (
        "success", "valid", "PASS")
    # ... and a wait that ended in lost custody gets its alarm back.
    lost = {**stored, "transport_status": "awaiting", "parse_status": "awaiting", "reason": "",
            "error": CUSTODY_ERROR, "operation_state": "custody_lost"}
    truth = _review_actor_projection(lost, "task_acceptance")
    assert (truth["transport_status"], truth["parse_status"], truth["reason"]) == (
        "provider_transport_error", "malformed", CUSTODY_ERROR)


# ── arithmetic: identical to the base for every input ────────────────────────

@pytest.mark.parametrize("policy", list(POLICIES))
@pytest.mark.parametrize("shapes", list(BASE_AGGREGATE), ids=lambda shapes: "+".join(shapes) or "empty")
def test_aggregate_quorum_and_gate_fields_equal_the_base(shapes, policy):
    before = [dataclasses.asdict(row(shape, f"s{index + 1}")) for index, shape in enumerate(shapes)]
    result, actors = aggregate(shapes, policy)
    expected = BASE_AGGREGATE[shapes][list(POLICIES).index(policy)]
    assert result["aggregate_signal"] == expected
    assert result["degraded"] is (expected == "DEGRADED")
    for shape, actor, stored in zip(shapes, actors, before):
        assert (actor.signal, actor.semantic_verdict, actor.enforcement_impact,
                actor.quorum_contribution) == BASE_ROW[shape], shape
        assert {key: getattr(actor, key) for key in INPUT_FIELDS} == {key: stored[key] for key in INPUT_FIELDS}
    panel = panel_of(shapes, policy)
    assert panel["aggregate_signal"] == expected
    assert panel["quorum"] == {
        "required": max(1, int(POLICIES[policy][1]["min_successful_slots"] or 1)),
        "contributed": sum(BASE_ROW[shape][3] for shape in shapes), "configured": len(shapes)}


def test_an_awaited_slot_holds_the_fail_closed_place_without_becoming_a_pass():
    # Both directions of the hold: a fail-closed surface stays DEGRADED, a surface that
    # does not fail closed keeps its quorum PASS, and a reviewer FAIL stays a FAIL.
    assert aggregate(("pass", "pass", "pending"), "fail_closed")[0]["aggregate_signal"] == "DEGRADED"
    assert aggregate(("pass", "pass", "pending"), "open")[0]["aggregate_signal"] == "PASS"
    assert aggregate(("pass", "pass", "pass"), "fail_closed")[0]["aggregate_signal"] == "PASS"
    assert aggregate(("fail", "pending", "pending"), "fail_closed")[0]["aggregate_signal"] == "FAIL"
    assert aggregate(("pass", "pass", "pending"), "quorum3")[0]["aggregate_signal"] == "DEGRADED"


# ── the aggregate's reasons ─────────────────────────────────────────────────

def test_awaited_slots_are_reported_once_and_never_as_a_fault():
    assert aggregate(("pass", "pending", "pending"))[0]["degraded_reasons"] == [
        "awaiting 2 of 3 reviewer slot(s): s2, s3 — no verdict"]
    assert aggregate(("pending", "pending", "pending"))[0]["degraded_reasons"] == [
        "awaiting 3 of 3 reviewer slot(s): s1, s2, s3 — no verdict"]
    # A quorum that is already met, or a veto, is not "no verdict".
    assert aggregate(("pass", "pass", "pending"))[0]["degraded_reasons"] == [
        "awaiting 1 of 3 reviewer slot(s): s3"]
    assert aggregate(("pass", "pass", "pending"), "fail_closed")[0]["degraded_reasons"] == [
        "awaiting 1 of 3 reviewer slot(s): s3 — no verdict"]
    assert aggregate(("fail", "pending", "pending"))[0]["degraded_reasons"] == [
        "awaiting 2 of 3 reviewer slot(s): s2, s3"]


@pytest.mark.parametrize("shapes, real", [
    (("run_failed", "pending", "pending"), [f"s1:{RUN_FAILED_ERROR}", "s1:degraded"]),
    (("custody_lost", "pending", "pending"), [f"s1:{CUSTODY_ERROR}", "s1:degraded"]),
    (("degraded", "pending", "pending"), ["s1:degraded"]),
])
def test_a_real_failure_beside_a_wait_stays_named(shapes, real):
    assert aggregate(shapes)[0]["degraded_reasons"] == [
        "awaiting 2 of 3 reviewer slot(s): s2, s3 — no verdict", *real]


def test_a_refusal_beside_a_wait_keeps_both_of_its_entries():
    assert aggregate(("pending", "not_dispatched", "not_dispatched"))[0]["degraded_reasons"] == [
        "awaiting 1 of 3 reviewer slot(s): s1 — no verdict",
        f"s2:{REFUSAL_ERROR}", f"s3:{REFUSAL_ERROR}", "s2:degraded", "s3:degraded"]


@pytest.mark.parametrize("shapes, reasons", [
    (("pass", "run_failed", "run_failed"),
     [f"s2:{RUN_FAILED_ERROR}", f"s3:{RUN_FAILED_ERROR}", "s2:degraded", "s3:degraded"]),
    (("not_dispatched",) * 3,
     [f"s1:{REFUSAL_ERROR}", f"s2:{REFUSAL_ERROR}", f"s3:{REFUSAL_ERROR}",
      "s1:degraded", "s2:degraded", "s3:degraded"]),
    (("timeout",) * 3, ["awaiting 3 of 3 reviewer slot(s): s1, s2, s3 — no verdict"]),
    (("pass", "custody_lost", "custody_lost"),
     [f"s2:{CUSTODY_ERROR}", f"s3:{CUSTODY_ERROR}", "s2:degraded", "s3:degraded"]),
    (("reservation",) * 3, ["awaiting 3 of 3 reviewer slot(s): s1, s2, s3 — no verdict"]),
    (("pass", "pass", "empty"), ["s3:empty", "s3:degraded"]),
    (("pass", "pass", "malformed"), ["s3:degraded"]),
    ((), ["quorum_not_met: pass_count=0 < min_successful=2"]),
    (("pass", "pass", "pass"), []),
])
def test_reasons_without_an_awaited_slot_are_the_base_lists(shapes, reasons):
    assert aggregate(shapes)[0]["degraded_reasons"] == reasons


# ── one decision point: awaited only while the row carries no answer ─────────

def test_the_hold_is_decided_from_the_row_as_a_mapping(monkeypatch):
    actor = row("pending", "s1")
    assert review_slot_awaiting(actor) is False  # the typed predicate never matches a dataclass
    assert review_slot_awaiting(dataclasses.asdict(actor)) is True
    import ouroboros.review_projection as projection

    seen = []
    monkeypatch.setattr(projection, "review_slot_awaiting",
                        lambda value: seen.append(type(value)) or review_slot_awaiting(value))
    result, actors = aggregate(("pass", "pending", "run_failed"))
    assert seen == [dict, dict, dict]  # the projection asks about every row, always as a mapping
    assert result["degraded_reasons"] == [
        "awaiting 1 of 3 reviewer slot(s): s2 — no verdict", f"s3:{RUN_FAILED_ERROR}", "s3:degraded"]
    # The aggregation and the stamped row agree with the projection, never with a second reading.
    assert [actor.transport_status for actor in actors] == ["success", "awaiting", "provider_transport_error"]


@pytest.mark.parametrize("shape, words, vote", [
    ("held_raw_pass", ("provider_transport_error", "valid", "PASS", "solved", "done"), True),
    ("held_raw_fail", ("provider_transport_error", "valid", "FAIL", "best_effort", "broken"), True),
    ("held_raw_prose", ("provider_transport_error", "malformed", "", "", PENDING_ERROR), False),
    ("ok_pending", ("success", "valid", "PASS", "solved", "done"), True),
])
def test_a_pending_row_that_carries_an_answer_is_judged_by_its_answer(shape, words, vote):
    result, actors = aggregate(("pass", "pass", shape))
    actor, stored = actors[-1], dataclasses.asdict(actors[-1])
    truth = _review_actor_projection(stored, "task_acceptance")
    assert (truth["transport_status"], truth["parse_status"], truth["semantic_verdict"],
            truth["outcome_tier"], truth["reason"]) == words
    assert actor.quorum_contribution is vote
    # Its reasons are the base's: a participation fault by name, never an awaited slot.
    assert result["degraded_reasons"] == {
        "held_raw_pass": [f"s3:{PENDING_ERROR}"], "held_raw_fail": [f"s3:{PENDING_ERROR}"],
        "held_raw_prose": [f"s3:{PENDING_ERROR}", "s3:degraded"], "ok_pending": [],
    }[shape]
    panel = panel_of(("pass", "pass", shape))
    assert "awaiting" not in json.dumps(panel)
    assert (panel["transport_status"], panel["coverage"]["transport_success"]) == (
        ("success", 3) if shape == "ok_pending" else ("partial", 2))


def test_a_stored_parsed_answer_on_a_pending_row_projects_as_before():
    stale = {"slot_id": "s1", "model": "m", "status": "error", "error": PENDING_ERROR,
             "operation_state": "pending_dispatch", "late_result_pending": True,
             "parsed": {**json.loads(PASS_TEXT), "dialogue_status": "continue_actionable"}}
    truth = _review_actor_projection(stale, "task_acceptance")
    assert {key: truth[key] for key in ("transport_status", "parse_status", "semantic_verdict", "outcome_tier",
                                        "dialogue_status", "coverage", "reason", "findings")} == {
        "transport_status": "provider_transport_error", "parse_status": "valid", "semantic_verdict": "PASS",
        "outcome_tier": "solved", "dialogue_status": "continue_actionable",
        "coverage": {"criteria_total": 1, "findings": 0}, "reason": "done", "findings": []}
    # Without the answer the same row is a gap.
    silent = {key: value for key, value in stale.items() if key != "parsed"}
    truth = _review_actor_projection(silent, "task_acceptance")
    assert (truth["transport_status"], truth["parse_status"], truth["semantic_verdict"]) == ("awaiting", "awaiting", "")


# ── the panel ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("shapes, transport, parse", [
    (("pending", "pending", "pending"), "awaiting", "awaiting"),
    (("pass", "pass", "pending"), "awaiting", "awaiting"),
    (("fail", "pending", "pending"), "awaiting", "awaiting"),
    (("degraded", "pending", "pending"), "awaiting", "awaiting"),
    # A real failure beside a wait keeps its own word.
    (("run_failed", "pending", "pending"), "provider_transport_error", "malformed"),
    (("custody_lost", "pending", "pending"), "provider_transport_error", "malformed"),
    (("timeout", "pending", "pending"), "awaiting", "awaiting"),
    (("pending", "not_dispatched", "not_dispatched"), "not_dispatched", "malformed"),
    (("pass", "run_failed", "pending"), "partial", "malformed"),
    # No awaited slot: the base fold, literally.
    (("pass", "pass", "pass"), "success", "valid"),
    (("pass", "pass", "timeout"), "awaiting", "awaiting"),
    (("pass", "pass", "malformed"), "success", "malformed"),
    (("timeout",) * 3, "awaiting", "awaiting"),
    (("not_dispatched",) * 3, "not_dispatched", "malformed"),
    (("reservation",) * 3, "awaiting", "awaiting"),
    ((), "provider_transport_error", "malformed"),
])
def test_panel_words_follow_the_collected_slots(shapes, transport, parse):
    panel = panel_of(shapes)
    assert (panel["transport_status"], panel["parse_status"]) == (transport, parse)
    answered = sum(shape in {"pass", "fail", "degraded", "malformed"} for shape in shapes)
    assert panel["coverage"]["transport_success"] == answered  # counts stay over every slot


def test_a_settled_panel_projects_the_base_reason_and_an_explicit_one_still_wins():
    failed = panel_of(("pass", "run_failed", "run_failed"))
    assert failed["reason"] == (f"s2:{RUN_FAILED_ERROR}; s3:{RUN_FAILED_ERROR}; s2:degraded; s3:degraded")
    explicit = panel_of(("pass", "run_failed", "run_failed"), reason="recorded by the host",
                        transport_status="timeout", parse_status="valid")
    assert (explicit["reason"], explicit["transport_status"], explicit["parse_status"]) == (
        "recorded by the host", "timeout", "valid")


def test_a_run_recorded_with_failure_words_about_a_wait_reprojects_as_awaiting():
    noise = [f"s1:{RUN_FAILED_ERROR}", f"s2:{PENDING_ERROR}", f"s3:{PENDING_ERROR}",
             "s1:degraded", "s2:degraded", "s3:degraded"]
    panel = panel_of(("run_failed", "pending", "pending"), degraded_reasons=noise,
                     reason="; ".join(noise), transport_status="provider_transport_error",
                     parse_status="malformed")
    assert panel["reason"] == ("awaiting 2 of 3 reviewer slot(s): s2, s3 — no verdict; "
                               f"s1:{RUN_FAILED_ERROR}; s1:degraded")
    waiting = panel_of(("pending", "pending", "pending"), reason=f"s1:{PENDING_ERROR}",
                       degraded_reasons=[f"s1:{PENDING_ERROR}", "s1:degraded"],
                       transport_status="provider_transport_error", parse_status="malformed")
    assert (waiting["transport_status"], waiting["parse_status"]) == ("awaiting", "awaiting")
    assert waiting["reason"] == "awaiting 3 of 3 reviewer slot(s): s1, s2, s3 — no verdict"
    assert waiting["aggregate_signal"] == "DEGRADED" and "Pending dispatch" not in waiting["reason"]


def test_a_superseded_awaited_panel_keeps_its_words_and_its_flag():
    panel = panel_of(("pass", "pending", "pending"), superseded_by_revision=True)
    assert panel["superseded"] is True
    assert (panel["transport_status"], panel["aggregate_signal"]) == ("awaiting", "DEGRADED")


# ── end to end through the substrate ────────────────────────────────────────

def test_a_released_run_reads_as_awaiting_and_then_as_its_answers(tmp_path, monkeypatch):
    import ouroboros.review_custody as custody
    from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request

    release, settled, entered = threading.Event(), threading.Semaphore(0), threading.Semaphore(0)
    original_settle = custody._settle_review_attempt

    def settle(*args, **kwargs):
        try:
            return original_settle(*args, **kwargs)
        finally:
            settled.release()

    monkeypatch.setattr(custody, "_settle_review_attempt", settle)

    class HeldModel:
        def chat(self, **kwargs):
            entered.release()
            assert release.wait(10), "fixture did not release its model"
            return {"content": "[]\nNO_FINDINGS"}, {"prompt_tokens": 5, "completion_tokens": 2}

    class Unused:
        def chat(self, **kwargs):
            pytest.fail("the settled wave was bought again")

    ctx = SimpleNamespace(task_id="awaiting-root", task_attempt=1, drive_root=tmp_path,
                          budget_drive_root=tmp_path, task_metadata={}, pending_events=[], event_queue=None)
    request = ReviewRequest(surface="skill_review", task_id=ctx.task_id, goal="goal", subject="result",
                            evidence={"requirement": "exact"}, retry_key="awaiting-subject",
                            policy={"min_successful_slots": 2}, drain_deadline=time.monotonic())
    slots = [ReviewSlot(slot_id=f"s{index}", model=f"model/{index}", effort="high", timeout_sec=20)
             for index in (1, 2, 3)]
    try:
        first = run_review_request(request, slots=slots, drive_root=tmp_path, usage_ctx=ctx, llm=HeldModel())
        assert all(entered.acquire(timeout=5) for _ in slots)
        # FLOOR: the stored rows and every gate reading of them are untouched.
        for actor in first.actors:
            assert actor["status"] == "error" and actor["error"].startswith("Pending dispatch;")
            assert (actor["operation_state"], actor["late_result_pending"], actor["failure_code"]) == (
                "pending_dispatch", True, "")
            assert (actor["signal"], actor["quorum_contribution"], actor["enforcement_impact"]) == (
                "DEGRADED", False, "abstains")
        assert (first.aggregate_signal, first.degraded) == ("DEGRADED", True)
        assert not review_outcome_received(first.actors)
        # The words about those rows.
        assert {(actor["transport_status"], actor["parse_status"]) for actor in first.actors} == {
            ("awaiting", "awaiting")}
        assert first.degraded_reasons == ["awaiting 3 of 3 reviewer slot(s): s1, s2, s3 — no verdict"]
        frozen = json.loads(json.dumps(dataclasses.asdict(first)))
        panel = compact_review_projection([{**frozen, "authority": "host_root"}])["panels"][0]
        assert (panel["transport_status"], panel["parse_status"], panel["aggregate_signal"]) == (
            "awaiting", "awaiting", "DEGRADED")
        assert panel["reason"] == first.degraded_reasons[0]
        assert panel["coverage"]["transport_success"] == 0
        release.set()
        assert all(settled.acquire(timeout=5) for _ in slots)
        # The same wave rejoined after its reviewers answered collects the same paid operations.
        result = run_review_request(request, slots=slots, drive_root=tmp_path, usage_ctx=ctx, llm=Unused())
        assert not any(actor["operation_state"] in {"pending_dispatch", "in_flight"} for actor in result.actors)
        assert (result.aggregate_signal, result.degraded_reasons) == ("PASS", [])
        assert [actor["operation_id"] for actor in result.actors] == [
            actor["operation_id"] for actor in first.actors]
        done = compact_review_projection([{**dataclasses.asdict(result), "authority": "host_root"}])["panels"][0]
        assert (done["transport_status"], done["parse_status"]) == ("success", "valid")
        assert "awaiting" not in json.dumps(done)
    finally:
        release.set()
