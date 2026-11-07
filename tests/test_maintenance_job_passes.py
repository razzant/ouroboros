"""A retirer pass is a fact in one shape: what it released and what it holds, by typed class, since when.

``maintenance_jobs.record_pass`` writes the pass as its job's row only when the pass is evidence (something
released, an error, a hold set that differs from the row's, the job's first pass, the first pass after one
threw); an identical second pass writes nothing (the rest law), a rested pass leaves the row alone, a partial
pass keeps the row's holds, a class held across passes keeps its first ``since``, and the event row beside the
job row carries the same facts. Isolated roots only.
"""
from __future__ import annotations

import json

from ouroboros import maintenance_jobs, store_import
from ouroboros.maintenance_jobs import Holds, pass_report, record_pass


def _events(root):
    path = root / "logs" / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def test_holds_count_by_class_and_keep_the_first_example():
    holds = Holds()
    holds.note("task_live", "/d/one")
    holds.note("task_live", "/d/two")
    holds.note("owner_keep", "x" * 500)
    assert holds.rows() == [{"class": "owner_keep", "count": 1, "example": "x" * 160},
                            {"class": "task_live", "count": 2, "example": "/d/one"}]
    assert len(holds) == 3
    assert pass_report(["a", "b"], holds, ["boom"]) == {"released": 2, "holds": holds.rows(), "errors": ["boom"]}
    assert pass_report(0, [{"class": "run_open", "count": 1, "example": ""}]) == {
        "released": 0, "holds": [{"class": "run_open", "count": 1, "example": ""}], "errors": []}


def test_a_scratch_hold_example_from_a_marker_is_bounded_like_every_other():
    """A producer's marker line can be any length; the example reaching the job row (and the per-turn storage fact)
    is cut to the engine's bound, as ``Holds.note`` cuts, while a short one stays whole."""
    from ouroboros.session_scratch import aggregate_holds

    long, short = "/var/tmp/obs-1: " + "retained path " * 80, "/var/tmp/obs-2: kept for inspection"
    rows = aggregate_holds([("producer_retained", long, None), ("seat_open", short, None)])
    assert [row["example"] for row in rows] == [long[:maintenance_jobs.HOLD_EXAMPLE_CHARS], short]


def test_a_first_hold_set_is_recorded_and_an_identical_second_pass_writes_nothing(tmp_path):
    store_import.ensure_current_store(tmp_path)
    holds = Holds()
    holds.note("task_live", "task-1")
    first = record_pass(tmp_path, "copy_retirement", pass_report(0, holds), event="subagent_worktree_prune")
    assert first is not None and first["released"] == 0 and first["held"] == 1
    assert first["holds"] == [{"class": "task_live", "count": 1, "example": "task-1", "since": first["pass_at"]}]
    assert [row["type"] for row in _events(tmp_path)] == ["subagent_worktree_prune"]
    before = maintenance_jobs.job(tmp_path, "copy_retirement")

    assert record_pass(tmp_path, "copy_retirement", pass_report(0, holds), event="subagent_worktree_prune") is None
    assert maintenance_jobs.job(tmp_path, "copy_retirement") == before, "the rest law: an unchanged pass writes nothing"
    assert len(_events(tmp_path)) == 1


def test_since_carries_over_while_a_class_stays_held_and_restarts_once_it_went(tmp_path):
    store_import.ensure_current_store(tmp_path)
    held = Holds()
    held.note("owner_keep", "k")
    first = record_pass(tmp_path, "review_checkout_prune", pass_report(0, held))
    both = Holds()
    both.note("owner_keep", "k")
    both.note("liveness_unknown", "u")
    second = record_pass(tmp_path, "review_checkout_prune", pass_report(0, both))
    assert second is not None, "a new class is a changed hold set"
    since = {row["class"]: row["since"] for row in second["holds"]}
    assert since["owner_keep"] == first["pass_at"] and since["liveness_unknown"] == second["pass_at"]
    gone = record_pass(tmp_path, "review_checkout_prune", pass_report(0, Holds()))
    assert gone is not None and gone["holds"] == [] and gone["held"] == 0, "every hold released is a change too"
    back = Holds()
    back.note("owner_keep", "k")
    again = record_pass(tmp_path, "review_checkout_prune", pass_report(0, back))
    assert again["holds"][0]["since"] == again["pass_at"] != first["pass_at"] or again["pass_at"] == first["pass_at"]


def test_a_release_or_an_error_is_evidence_and_a_rested_pass_is_not(tmp_path):
    store_import.ensure_current_store(tmp_path)
    clean = record_pass(tmp_path, "session_scratch_release", pass_report(0, Holds()))
    assert clean is not None and clean["state"] == "done" and clean["holds"] == [], "a tidy plane's first pass is a fact, never 'not run'"
    assert record_pass(tmp_path, "session_scratch_release", pass_report(0, Holds())) is None, "nothing kept, nothing freed"
    released = record_pass(tmp_path, "session_scratch_release", pass_report(["/tmp/obs-1"], Holds()), event="session_scratch_release")
    assert released["released"] == 1 and released["holds"] == []
    event, = _events(tmp_path)
    assert event["released"] == 1 and event["report"]["released"] == 1 and event["job"] == "session_scratch_release"
    kept_list = record_pass(tmp_path, "session_scratch_release", {**pass_report(0, Holds()), "released": ["/tmp/obs-2"]},
                            event="session_scratch_release")
    assert kept_list["released"] == 1 and _events(tmp_path)[-1]["report"]["released"] == ["/tmp/obs-2"], \
        "a retirer's own report keeps what it names; the row keeps the count"
    failed = record_pass(tmp_path, "session_scratch_release", pass_report(0, Holds(), ["note: OSError"]))
    assert failed["errors"] == ["note: OSError"]
    rested = {**pass_report(0, Holds()), "rested": True}
    assert record_pass(tmp_path, "session_scratch_release", rested) is None
    assert maintenance_jobs.job(tmp_path, "session_scratch_release") == failed, "a rested pass leaves the row as it stands"


def test_a_partial_pass_records_what_it_released_under_the_holds_of_the_last_whole_pass(tmp_path):
    store_import.ensure_current_store(tmp_path)
    seats_only = Holds()
    seats_only.note("seat_busy", "seat-2")
    opening = record_pass(tmp_path, "review_checkout_prune", {**pass_report(0, seats_only), "partial": True})
    assert [row["class"] for row in opening["holds"]] == ["seat_busy"], "a first part has no row to keep: what it judged stands"
    whole = Holds()
    whole.note("review_custody_active", "checkout-1")
    whole.note("owner_keep", "seat-1")
    first = record_pass(tmp_path, "review_checkout_prune", pass_report(0, whole))
    partial = record_pass(tmp_path, "review_checkout_prune", {**pass_report(1, seats_only), "partial": True})
    assert partial["released"] == 1 and partial["holds"] == first["holds"], "a part of the plane replaces no hold"
    assert record_pass(tmp_path, "review_checkout_prune", {**pass_report(0, seats_only), "partial": True}) is None
    again = record_pass(tmp_path, "review_checkout_prune", pass_report(0, seats_only))
    assert [row["class"] for row in again["holds"]] == ["seat_busy"], "a whole pass's holds replace the row's"


def test_a_pass_that_threw_is_a_fact_until_a_pass_is_recorded_again(tmp_path):
    store_import.ensure_current_store(tmp_path)
    held = Holds()
    held.note("task_live", "t")
    record_pass(tmp_path, "copy_retirement", pass_report(0, held))
    maintenance_jobs.record_threw(tmp_path, "copy_retirement", OSError("disk"))
    assert maintenance_jobs.job(tmp_path, "copy_retirement")["last_error"]["class"] == "OSError"
    assert record_pass(tmp_path, "copy_retirement", {**pass_report(0, held), "rested": True}) is None
    assert "last_error" in maintenance_jobs.job(tmp_path, "copy_retirement"), "a pass that rested proves no return"
    back = record_pass(tmp_path, "copy_retirement", pass_report(0, held))
    assert back is not None and "last_error" not in back and "last_error" not in maintenance_jobs.job(tmp_path, "copy_retirement")
    assert record_pass(tmp_path, "copy_retirement", pass_report(0, held)) is None, "then the rest law again"
