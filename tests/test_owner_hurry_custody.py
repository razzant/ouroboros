"""The plan-review disclosure over a LEGACY (v2) wave series states what is open NOW.

The live gate reads an in-flight legacy wave through ``plan_record.import_legacy_v2``: its
open blocking findings are objections of their seats, its unanswered reviewers are one
pending ask ("a result is still owed"), and nothing else is "open" — a settled DEGRADED wave
with no finding holds nothing and says nothing (silence never holds). The suffix names the
rail that forced the finalization and never a past tense; an absent rail reason renders as
absence, never as the internal token ``task_rail``.

New module rather than tests/test_owner_hurry_s3.py: that file is 935 lines and
these cases would carry it past the 1000-line target.
"""

from __future__ import annotations

import pathlib
import tempfile
import types

from ouroboros.owner_hurry import plan_gate_decision, plan_review_disclosure
from ouroboros.task_results import write_task_result

_FINGERPRINTS = {"REVISE_PLAN": "a" * 64, "REVIEW_REQUIRED": "b" * 64, "DEGRADED": "c" * 64}


def _wave(aggregate: str, **fields) -> dict:
    return {
        "request_fingerprint": _FINGERPRINTS[aggregate], "aggregate": aggregate,
        "closed": False, "spec": {}, "findings": [], "paid": True, **fields,
    }


def _blocking(seat: str = "s1") -> dict:
    return {"finding_id": f"{seat}:f1", "class": "blocking", "slot": seat, "summary": "the budget line is missing"}


def _ctx_over(waves: list, current: str) -> types.SimpleNamespace:
    """A context whose durable authority is exactly the given wave series."""
    root = pathlib.Path(tempfile.mkdtemp())
    write_task_result(
        root, "hurry-custody", "running",
        plan_review_state={
            "schema_version": 2, "series_id": "series-1", "cycles_paid": len(waves),
            "need_evidence_seen": [],
            "current_attempt": {
                "fingerprint": _FINGERPRINTS[current], "status": "open", "reason": "",
            },
            "waves": waves, "waves_omitted": 0,
        },
    )
    return types.SimpleNamespace(
        task_id="hurry-custody", budget_drive_root=str(root), drive_root=str(root),
        repo_dir=root, system_repo_dir=root,
        task_metadata={"force_plan": True},
    )


def test_a_three_wave_series_discloses_the_open_state_never_a_past_tense() -> None:
    """REVISE_PLAN then REVIEW_REQUIRED then DEGRADED: the LAST wave is the one the importer
    reads. Its open blocking finding is an objection the rail form names in the present tense;
    a clean settled series is nothing open, so the rail discloses nothing about the plan."""
    ctx = _ctx_over(
        [_wave("REVISE_PLAN"), _wave("REVIEW_REQUIRED"), _wave("DEGRADED", findings=[_blocking("s2")])], "DEGRADED",
    )
    decision = plan_gate_decision(ctx, {}, hard_rail="round_limit", enforcement="blocking")

    disclosure = plan_review_disclosure(decision, "round_limit")

    assert decision["status"] == "rail_degraded" and decision["allow"] is True and decision["source"] == "legacy_v2"
    assert "Blocking plan review is open (1 open objection: s2/s2:f1" in disclosure
    assert "remained" not in disclosure and "DEGRADED" not in disclosure
    assert "`round_limit`" in disclosure

    settled = plan_gate_decision(
        _ctx_over([_wave("REVISE_PLAN"), _wave("REVIEW_REQUIRED"), _wave("DEGRADED")], "DEGRADED"), {},
        hard_rail="round_limit", enforcement="blocking",
    )
    assert settled["status"] == "rail_degraded" and settled["open_objections"] == [] and settled["pending_asks"] == []
    assert plan_review_disclosure(settled, "round_limit") == ""


def test_a_late_result_still_owed_reaches_the_owner_line() -> None:
    """plan_review_runtime kept a wave open and DEGRADED precisely because a paid slot could
    still settle; the importer carries that as ONE pending ask the owner line names."""
    ctx = _ctx_over(
        [_wave("DEGRADED", custody_pending=True, reasons=["review_late_result_pending"])],
        "DEGRADED",
    )
    decision = plan_gate_decision(ctx, {}, hard_rail="round_limit", enforcement="blocking")

    assert decision["pending_asks"] == ["legacy wave " + "c" * 12]
    disclosure = plan_review_disclosure(decision, "round_limit")
    assert "answers still on their way for ask(s) legacy wave cccccccccccc" in disclosure
    assert "`round_limit`" in disclosure

    quiet = plan_gate_decision(
        _ctx_over([_wave("DEGRADED")], "DEGRADED"), {},
        hard_rail="round_limit", enforcement="blocking",
    )
    assert quiet["pending_asks"] == []
    assert "still on their way" not in plan_review_disclosure(quiet, "round_limit")


def test_advisory_enforcement_also_states_what_is_open() -> None:
    """Advisory never holds: an open objection of the legacy wave is disclosed as open NOW;
    a settled wave with no finding discloses nothing."""
    ctx = _ctx_over([_wave("DEGRADED", findings=[_blocking()])], "DEGRADED")
    decision = plan_gate_decision(ctx, {}, enforcement="advisory")

    disclosure = plan_review_disclosure(decision)

    assert decision["allow"] is True and decision["status"] == "objections_open"
    assert "Plan review: 1 open objection remains open — s1/s1:f1" in disclosure
    assert "remained" not in disclosure and "DEGRADED" not in disclosure
    assert "advisory enforcement" in disclosure

    quiet = plan_gate_decision(_ctx_over([_wave("DEGRADED")], "DEGRADED"), {}, enforcement="advisory")
    assert quiet["allow"] is True and quiet["status"] == "clear" and plan_review_disclosure(quiet) == ""


def test_a_rail_with_no_recorded_reason_renders_absence_not_an_internal_token() -> None:
    """The owner-visible suffix quoted ``task_rail`` in backticks when the rail had no
    recorded reason, which reads as a named cause and is not one."""
    railed = {
        "required": True, "enforcement": "blocking", "status": "rail_degraded", "allow": True, "reason": "",
        "open_objections": [{"seat_id": "s1", "id": "s1:f1", "text": "the budget line is missing", "holds": True,
                             "ask_id": "a1", "revision": 1}],
        "holding": ["s1:s1:f1"], "pending_asks": [],
    }

    without_reason = plan_review_disclosure(railed)
    assert "task_rail" not in without_reason
    assert "a task-wide rail required best-effort finalization." in without_reason

    with_reason = plan_review_disclosure({**railed, "reason": "budget_exhausted"})
    assert "the task-wide rail `budget_exhausted` required best-effort" in with_reason

    forced = plan_review_disclosure(railed, "round_limit")
    assert "the task-wide rail `round_limit` required best-effort" in forced
