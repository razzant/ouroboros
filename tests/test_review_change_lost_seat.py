"""A rerun whose one owed seat lost custody is decided by the seats that answered (owner 17.10, Q3-2)."""
from __future__ import annotations

import ouroboros.review_substrate as substrate
from ouroboros.tools.review_change import run_review_change
from tests import _contributor_packet_shared as shared
from tests.test_review_change_end_to_end import _foreign_project, _pending_then_answering_seat


def test_a_rerun_whose_owed_seat_lost_custody_settles_by_the_seats_that_answered(tmp_path, monkeypatch):
    """A round whose one owed seat can no longer answer (its custody is lost: outcome unknown, never re-sent)
    is decided by the seats that did answer on the rerun: the SAME record settles to PASS with the lost seat
    failed, instead of staying pending for a re-issue that can never change it (owner 17.10, Q3-2)."""
    from ouroboros.tools import review_change_custody

    ctx, project = _foreign_project(tmp_path, monkeypatch)
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")  # where the wave's pending block is raised
    (project / "app.py").write_text("VALUE = 2  # staged\n", encoding="utf-8")
    shared.git(project, "add", "app.py")
    sends: list[dict] = []
    answer, _starts = _pending_then_answering_seat("t2", "invocation-t2-lost")
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", shared.golden_physical_seam(sends, answer=answer))
    ask = dict(subject="index", goal="Bump", scope="app.py")
    first = run_review_change(ctx, **ask)
    assert first["state"] == "pending", first

    collect = review_change_custody.collect_recorded_commit_actor

    def lose_t2(ctx_, surface, slot_id, retry_key):
        actor = collect(ctx_, surface, slot_id, retry_key)
        if slot_id != "t2":
            return actor
        # This replacement simulates a lost producer, so it cannot also retain
        # the original producer's canonical responded outcome (S1).
        actor = {key: value for key, value in actor.items() if key != "outcome"}
        return {**actor, "status": "error", "raw_text": "", "operation_state": "custody_lost",
                "late_result_pending": True, "failure_code": "review_custody_lost",
                "error": "A start was requested but no run was ever bound to it."}

    monkeypatch.setattr(review_change_custody, "collect_recorded_commit_actor", lose_t2)
    rerun = run_review_change(ctx, **ask)
    assert (rerun["state"], rerun["aggregate"]) == ("settled", "PASS"), rerun
    assert rerun["record_id"] == first["record_id"]
    seats = {row["seat_id"]: row for row in rerun["rows"]}
    assert seats["t2"]["operation_state"] == "custody_lost" and seats["t2"]["status"] == "error"
    assert rerun["quorum"]["responded"] == 2 and rerun["quorum"]["required"] == 2
