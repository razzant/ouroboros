"""Ouroboros sees what holds its rooms: queue holds, a Pause by its declared author, and live work whose
own root has ended (helpers, retained runs) — beside the unchanged tell_task address book."""
import pytest

from ouroboros.live_works import live_works_present
from ouroboros.peer_roster import independent_roots, render_roster_note, unlisted_live_work
from ouroboros.task_results import write_task_result
from ouroboros.utils import utc_now_iso
from tests._shared import write_queue_document


def _snapshot(root, *, running=(), pending=()):
    stamp = utc_now_iso()
    write_queue_document(root / "state/queue_snapshot.json", {"ts": stamp, "running": list(running), "pending": list(pending)})
    write_queue_document(root / "state/direct_roots.json", {"ts": stamp, "incomplete": False, "roots": []})


def test_a_writer_held_continue_reads_as_held_for_the_earlier_attempt(tmp_path):
    write_task_result(tmp_path, "c", "scheduled", title="Continue: Release")
    held = {"reason": "continuation_writer_unsettled", "held_at": "2026-10-17T11:00:00Z"}
    _snapshot(tmp_path, pending=[{"id": "c", "attempt": 1, "task": {
        "id": "c", "chat_id": 42, "_budget_pause_hold": held,
        "metadata": {"continuation": {"predecessor_task_id": "p"}}}}])
    row = independent_roots(tmp_path)["roots"][0]
    assert row["waiting"] == [{"kind": "hold", "reason": "earlier_work_unconfirmed",
                               "text": "Continue starts when the earlier attempt's work is confirmed finished",
                               "since": "2026-10-17T11:00:00Z", "tasks": ["p"], "release": "none"}]
    assert "held (reason=earlier_work_unconfirmed" in render_roster_note(independent_roots(tmp_path))
    # A hold the owner already selected explains nothing any more.
    _snapshot(tmp_path, pending=[{"id": "c", "attempt": 1, "task": {
        "id": "c", "chat_id": 42, "_resume_selection": {"selected_by": "owner"}}}])
    assert "waiting" not in independent_roots(tmp_path)["roots"][0]


@pytest.mark.parametrize("declared, by", [
    ({"by": {"actor": "owner", "text": "", "at": "2026-10-13T12:40:00Z"}}, "owner"),
    ({"by": {"actor": "ouroboros", "text": "pausing for the update", "at": "2026-10-13T12:40:00Z"}},
     "Ouroboros: pausing for the update"),
    ({"actor": "mind", "actor_task_id": "m1"}, "Ouroboros"),  # a fence written before the declaration existed
    ({}, ""),  # requested_by=owner alone is the door a press came through: it names nobody
])
def test_an_owner_pause_names_the_author_its_fence_declared(tmp_path, declared, by):
    write_task_result(tmp_path, "t", "running", title="Work", owner_pause={
        "state": "paused", "requested_by": "owner", "requested_at": "2026-10-13T12:40:00Z",
        "paused_at": "2026-10-13T12:50:00Z", "fence_id": "f1", **declared})
    _snapshot(tmp_path, running=[{"id": "t", "attempt": 1, "task": {"id": "t", "chat_id": 1}}])
    [wait] = independent_roots(tmp_path)["roots"][0]["waiting"]
    # Dated by the press, as the owner's card dates it: a settling fence restamps paused_at.
    assert {k: v for k, v in wait.items() if k not in {"text", "reason", "release"}} == {"kind": "owner_pause", "state": "paused", "since": "2026-10-13T12:40:00Z",
                    **({"by": by} if by else {})}
    note = render_roster_note(independent_roots(tmp_path))
    assert "owner Pause (" in note and "release=resume" in note
    if by:
        assert f"by={by}" in note
    assert wait["release"] == "resume"
    write_task_result(tmp_path, "t", "running", owner_pause={"state": "released", "requested_by": "owner",
                                                            "fence_id": "f1"})
    assert "waiting" not in independent_roots(tmp_path)["roots"][0]


def test_live_work_without_a_listed_root_is_visible_but_never_addressable(tmp_path, monkeypatch):
    write_task_result(tmp_path, "p", "failed", suggested_name="Release 7.7", chat_id=42)
    _snapshot(tmp_path, running=[{"id": f"h{index}", "task": {"id": f"h{index}", "parent_task_id": "p",
                                                               "root_task_id": "p"}} for index in range(3)])
    rows = unlisted_live_work(tmp_path, set())
    assert rows == [{"work_id": "p", "title": "Release 7.7", "room": {"chat_id": 42, "project_id": ""},
                     "root_status": "failed", "helpers_running": 3, "helpers_waiting": 0, "tails": [
                         {"code": "helpers_running", "text": "3 helpers still working", "count": 3,
                          "ids": ["h0", "h1", "h2"], "kind": "helper", "state": "live"}]}]
    assert unlisted_live_work(tmp_path, {"p"}) == []
    present = live_works_present(tmp_path)
    assert present["works"] == [] and present["other_live_work"][0]["work_id"] == "p"
    assert "ordinary hygiene" in present["note"]
    # The address book is unchanged: the ended root is not a tell_task target.
    assert independent_roots(tmp_path)["roots"] == []
    assert present["incomplete"] is False
    from ouroboros import task_results
    from ouroboros.store import StoreBusy
    def busy(*_args, **_kwargs):
        raise StoreBusy("ended root busy")
    with monkeypatch.context() as patch:
        patch.setattr(task_results, "load_task_result", busy)
        partial = live_works_present(tmp_path)
        assert partial["incomplete"] is True
        [unreadable] = partial["other_live_work"]
        assert unreadable["work_id"] == "p" and unreadable["helpers_running"] == 3
        assert unreadable["unavailable"] == {"task_result": "StoreBusy"}
        assert unreadable["root_status"] == "unknown"
    recovered = live_works_present(tmp_path)
    assert recovered["incomplete"] is False
    assert recovered["other_live_work"] == present["other_live_work"] and recovered["works"] == present["works"]


def test_runs_still_working_under_a_settled_root_are_its_typed_tails(tmp_path):
    """Through the real custody store, typed as the owner's card types them: a run whose own task answered is
    live, a review is the review tail, a run whose own task failed is unconfirmed; a settled run custody still
    holds and a run the engine already finished run nothing. The ended root stays in other_live_work."""
    from ouroboros import delegate_custody as custody

    write_task_result(tmp_path, "p", "done", title="Done work")
    write_task_result(tmp_path, "q", "done", title="Done earlier")
    write_task_result(tmp_path, "p-h", "failed", parent_task_id="p", root_task_id="p")
    _snapshot(tmp_path)
    assert custody.emit(tmp_path, custody.STARTED, {"run_id": "r1", "task_id": "p-h", "root_task_id": "p"})
    assert custody.emit(tmp_path, custody.STARTED, {"run_id": "r2", "task_id": "p"})
    assert custody.emit(tmp_path, custody.STARTED, {"run_id": "rv", "task_id": "p", "source": "review_substrate"})
    assert custody.emit(tmp_path, custody.STARTED, {"run_id": "done", "task_id": "p"})
    assert custody.emit(tmp_path, custody.TERMINAL_OBSERVED, {
        "run_id": "done", "task_id": "p", "state": "succeeded", "observation_seq": 1,
        "source_ref": {"sha256": "0" * 64}})
    assert custody.emit(tmp_path, custody.STARTED, {"run_id": "kept", "task_id": "q", "project_id": "x",
                                                     "project_owned": True})
    assert custody.emit(tmp_path, custody.SETTLED, {"run_id": "kept", "task_id": "q", "state": "succeeded"})
    [row] = unlisted_live_work(tmp_path, set())
    assert (row["work_id"], row["helpers_running"], row["helpers_waiting"]) == ("p", 0, 0)
    assert [(tail["kind"], tail["state"], tail["text"], tail["ids"]) for tail in row["tails"]] == [
        ("run", "live", "1 external run still working", ["r2"]), ("review", "live", "Review work finishing", ["rv"]),
        ("run", "unknown", "1 external run may still be running", ["r1"])]
    assert live_works_present(tmp_path)["other_live_work"] == [row]


def _paused_root(root, *, fence=None, helpers=()):
    """A listed root parked as the exact pause of the owner's Pause, with its tree's helpers in the queue."""
    write_task_result(root, "t", "scheduled", title="Paused work", chat_id=1, reason_code="budget_paused",
                      budget_pause={"pause_id": "p-t", "state": "paused", "task_attempt": 1, "reason": "owner",
                                    "rail": "owner_pause", "paused_at": 1790409600.0},
                      owner_pause={"state": "paused", "fence_id": "f1", "requested_at": "2026-10-20T00:43:00Z",
                                   "by": {"actor": "owner", "text": "", "at": "2026-10-20T00:43:00Z"},
                                   **(fence or {})})
    _snapshot(root, pending=[{"id": "t", "attempt": 1, "task": {"id": "t", "chat_id": 1, "_attempt": 1}}],
              running=[{"id": helper, "task": {"id": helper, "parent_task_id": "t", "root_task_id": "t"}}
                       for helper in helpers])


def test_a_paused_root_says_what_of_its_tree_still_runs_as_the_owners_card_does(tmp_path, monkeypatch):
    """Parked under the owner's press: Ouroboros reads the review its Pause lets finish and a run nobody can
    confirm beside the wait, never as "resume now"; the owner's card types them by the same rule."""
    from ouroboros import delegate_custody as custody
    from ouroboros.gateway import work_status
    from ouroboros.peer_roster import roster_fingerprint

    _paused_root(tmp_path)
    write_task_result(tmp_path, "h", "failed", parent_task_id="t", root_task_id="t")
    assert custody.emit(tmp_path, custody.STARTED, {"run_id": "rv", "task_id": "t", "root_task_id": "t",
                                                     "source": "review_substrate"})
    assert custody.emit(tmp_path, custody.STARTED, {"run_id": "lost", "task_id": "h", "root_task_id": "t"})
    roster = independent_roots(tmp_path)
    [row] = roster["roots"]
    assert row["waiting"][0]["kind"] == "owner_pause" and row["waiting"][0]["by"] == "owner"
    assert [(tail["kind"], tail["state"], tail["text"], tail["ids"]) for tail in row["tails"]] == [
        ("review", "live", "Review work finishing", ["rv"]), ("run", "unknown", "1 external run may still be running", ["lost"])]
    assert "\n  tails: Review work finishing; 1 external run may still be running" in render_roster_note(roster)
    [work] = live_works_present(tmp_path)["works"]
    assert work["tails"] == row["tails"]
    # The same tree reads the same whether its root is listed or not.
    assert unlisted_live_work(tmp_path, set())[0]["tails"] == row["tails"]
    # A tail that ends changes the roster (stale catalogue pages refuse); with none left, no tails line.
    assert custody.emit(tmp_path, custody.SETTLED, {"run_id": "lost", "task_id": "h", "state": "cancelled"})
    settled = independent_roots(tmp_path)
    assert roster_fingerprint(settled) != roster_fingerprint(roster)
    assert [tail["kind"] for tail in settled["roots"][0]["tails"]] == ["review"]
    assert custody.emit(tmp_path, custody.SETTLED, {"run_id": "rv", "task_id": "t", "state": "succeeded"})
    quiet = independent_roots(tmp_path)
    assert "tails" not in quiet["roots"][0] and "tails:" not in render_roster_note(quiet)
    # One typing for both readers: the roster follows the card's rule, never a copy of its own.
    assert custody.emit(tmp_path, custody.STARTED, {"run_id": "r9", "task_id": "t", "root_task_id": "t"})
    assert [tail["state"] for tail in independent_roots(tmp_path)["roots"][0]["tails"]] == ["live"]
    monkeypatch.setattr(work_status, "run_tail", lambda source, owner, **facts: ("run", "unknown"))
    assert [tail["state"] for tail in independent_roots(tmp_path)["roots"][0]["tails"]] == ["unknown"]


def test_a_settled_pause_lists_its_finishing_reviews_without_a_custody_row(tmp_path):
    _paused_root(tmp_path, fence={"finishing_reviews": ["t"]})
    [row] = independent_roots(tmp_path)["roots"]
    assert row["tails"] == [{"code": "review_work_finishing", "text": "Review work finishing", "kind": "review",
                             "state": "live"}]
    _paused_root(tmp_path, fence={"finishing_reviews": []})
    assert "tails" not in independent_roots(tmp_path)["roots"][0]


@pytest.mark.parametrize("fence, working", [
    ({"state": "requested", "parked_members": ["h1"]}, ["h2"]),  # still settling: h2 finishes its step
    ({"state": "paused", "parked_members": ["h1", "h2"]}, []),  # settled: the Pause parked them all
    ({"state": "released", "parked_members": ["h1", "h2"]}, ["h1", "h2"]),  # a released fence parks nobody
])
def test_helpers_the_pause_parked_are_not_still_working(tmp_path, fence, working):
    _paused_root(tmp_path, fence=fence, helpers=["h1", "h2"])
    [row] = independent_roots(tmp_path)["roots"]
    assert [tail["ids"] for tail in row.get("tails") or ()] == ([working] if working else [])


def test_a_running_root_says_no_tails_its_own_status_already_says_it_runs(tmp_path):
    from ouroboros import delegate_custody as custody

    write_task_result(tmp_path, "r", "running", title="Busy", chat_id=1)
    _snapshot(tmp_path, running=[{"id": "r", "attempt": 1, "task": {"id": "r", "chat_id": 1}},
                                 {"id": "h", "task": {"id": "h", "parent_task_id": "r", "root_task_id": "r"}}])
    assert custody.emit(tmp_path, custody.STARTED, {"run_id": "r1", "task_id": "r"})
    [row] = independent_roots(tmp_path)["roots"]
    assert "waiting" not in row and "tails" not in row
    assert unlisted_live_work(tmp_path, {"r"}) == []  # its tree is listed: nothing moves into other_live_work


def test_an_unreadable_custody_leaves_the_roster_incomplete(tmp_path, monkeypatch):
    """Runs the roster could not read are a gap it discloses, never a quiet tree."""
    from ouroboros.gateway import work_projection

    _paused_root(tmp_path)
    assert independent_roots(tmp_path)["incomplete"] is False

    def unreadable(drive_root):
        raise OSError("custody store unreadable")

    monkeypatch.setattr(work_projection, "_open_runs", unreadable)
    assert independent_roots(tmp_path)["incomplete"] is True
