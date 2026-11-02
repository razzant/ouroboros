"""Changed roster rows stay factual, compact and recoverable after compaction."""
from copy import deepcopy
from types import SimpleNamespace

from ouroboros import peer_roster


def _roster(count=18):
    return {"roots": [{"task_id": f"peer-{i:02}", "title": f"Independent task {i}",
                        "project_id": "project", "chat_id": i, "status": "running"}
                       for i in range(count)], "incomplete": False}


def test_changed_row_does_not_copy_seventeen_unchanged_roots(monkeypatch, tmp_path):
    roster = _roster()
    monkeypatch.setattr(peer_roster, "independent_roots", lambda root: deepcopy(roster))
    ctx = SimpleNamespace(task_id="observer", task_metadata={})
    messages = [{"role": "user", "content": "Owner request stays exact."}]
    assert peer_roster.maybe_append_roster_note(ctx, messages, tmp_path)
    before = deepcopy(messages)
    roster["roots"][7]["waiting"] = [{"kind": "owner", "since": "2026-10-09T13:00:00Z", "quiz_id": "q1"}]
    assert peer_roster.maybe_append_roster_note(ctx, messages, tmp_path)
    assert messages[:-1] == before
    delta = messages[-1]["content"]
    assert "peer-07" in delta and "quiz_id=q1" in delta
    assert all(f"peer-{i:02}" not in delta for i in range(18) if i != 7)
    assert len(delta) < len(before[-1]["content"])
    assert not peer_roster.maybe_append_roster_note(ctx, messages, tmp_path)
    roster["roots"][2]["status"] = "pending"
    roster["roots"][3]["status"] = "pending"
    assert peer_roster.maybe_append_roster_note(ctx, messages, tmp_path)
    pair_delta = messages[-1]["content"]
    assert "peer-02" in pair_delta and "peer-03" in pair_delta
    assert all(f"peer-{i:02}" not in pair_delta for i in range(18) if i not in (2, 3))
    assert pair_delta.count("[INDEPENDENT_ROOTS]") == 1
    assert len(pair_delta) < len(before[-1]["content"])
    del roster["roots"][7]["waiting"]
    assert peer_roster.maybe_append_roster_note(ctx, messages, tmp_path)
    assert "peer-07" in messages[-1]["content"] and "quiz_id=q1" not in messages[-1]["content"]
    assert "absent facts are no longer reported" in messages[-1]["content"]


def test_missing_base_or_middle_update_restores_complete_snapshot(monkeypatch, tmp_path):
    roster = _roster()
    monkeypatch.setattr(peer_roster, "independent_roots", lambda root: deepcopy(roster))
    ctx = SimpleNamespace(task_id="observer", task_metadata={})
    messages = []
    assert peer_roster.maybe_append_roster_note(ctx, messages, tmp_path)
    roster["roots"][0]["status"] = "pending"
    assert peer_roster.maybe_append_roster_note(ctx, messages, tmp_path)
    roster["roots"][1]["status"] = "pending"
    assert peer_roster.maybe_append_roster_note(ctx, messages, tmp_path)
    del messages[1]  # A compaction may retain the full base and latest delta only.
    preserved = deepcopy(messages)
    assert peer_roster.maybe_append_roster_note(ctx, messages, tmp_path)
    assert messages[:-1] == preserved
    assert messages[-1]["content"] == peer_roster.render_roster_note(roster, exclude="observer")
    messages[:] = [{"role": "assistant", "content": "Previous work understood."}]
    assert peer_roster.maybe_append_roster_note(ctx, messages, tmp_path)
    assert messages[-1]["content"] == peer_roster.render_roster_note(roster, exclude="observer")


def test_removed_display_row_and_health_change_are_not_completion(monkeypatch, tmp_path):
    roster = _roster()
    monkeypatch.setattr(peer_roster, "independent_roots", lambda root: deepcopy(roster))
    ctx = SimpleNamespace(task_id="observer", task_metadata={})
    messages = []
    assert peer_roster.maybe_append_roster_note(ctx, messages, tmp_path)
    roster["roots"].pop(0)
    roster["incomplete"] = True
    assert peer_roster.maybe_append_roster_note(ctx, messages, tmp_path)
    delta = messages[-1]["content"]
    assert "No longer in the displayed roster (not proof of completion): peer-00" in delta
    assert "projection incomplete" in delta
    assert "peer-01" not in delta


def _peer_world(root, *, waits=True):
    """Two listed roots through the real task store and queue projection; ``peer`` keeps its queue focus
    and, with ``waits``, a stored owner wait. Queue rows carry the start, so it is known either way."""
    from ouroboros.task_results import write_task_result
    from ouroboros.utils import utc_now_iso
    from tests._shared import write_queue_document

    focus = {"text": "Checking the release notes", "author_task_id": "peer", "authored_at": "2026-10-26T10:00:00Z",
             "source_ref": {"reader": "get_task_result", "task_id": "peer"}}
    wait = {"owner_wait": {"state": "waiting", "quiz_id": "q1", "task_attempt": 1, "reason": "owner",
                           "parked_at": "2026-10-26T10:05:00Z"}} if waits else {}
    write_task_result(root, "peer", "running", title="Peer work", chat_id=1, **wait)
    write_task_result(root, "other", "running", title="Other work", chat_id=1)
    stamp = utc_now_iso()
    write_queue_document(root / "state/queue_snapshot.json", {"ts": stamp, "pending": [], "running": [
        {"id": "peer", "attempt": 1, "started_at": 1793000000.0,
         "task": {"id": "peer", "chat_id": 1, "title": "Peer work", "focus": focus}},
        {"id": "other", "attempt": 1, "started_at": 1793000000.0,
         "task": {"id": "other", "chat_id": 1, "title": "Other work"}}]})
    write_queue_document(root / "state/direct_roots.json", {"ts": stamp, "incomplete": False, "roots": []})


def _unreadable_results(monkeypatch, failing):
    """The task store refuses each root id in ``failing`` (changed between reads), as a busy store does."""
    from ouroboros import task_results
    from ouroboros.store import StoreBusy

    real = task_results.load_task_result

    def load(root, task_id, *args, **kwargs):
        if str(task_id) in failing:
            raise StoreBusy("task store busy")
        return real(root, task_id, *args, **kwargs)

    monkeypatch.setattr(task_results, "load_task_result", load)


def test_an_unreadable_task_result_is_named_on_its_row_and_never_reads_as_an_ended_wait(monkeypatch, tmp_path):
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY

    _peer_world(tmp_path)
    failing = set()
    _unreadable_results(monkeypatch, failing)
    ctx = SimpleNamespace(task_id="observer", task_metadata={})
    messages = []
    assert peer_roster.maybe_append_roster_note(ctx, messages, tmp_path)
    assert "owner answer (quiz_id=q1, since=2026-10-26T10:05:00Z)" in messages[-1]["content"]
    assert "unavailable" not in messages[-1]["content"]
    failing.add("peer")
    assert peer_roster.maybe_append_roster_note(ctx, messages, tmp_path)
    delta = messages[-1]["content"]
    assert messages[-1][HOST_CONTEXT_KIND_KEY] == peer_roster.ROSTER_UPDATE_NAME
    # The queue's facts stay; the row names the record it could not read and what that leaves unknown.
    assert "- peer · Peer work · chat=1 · running" in delta and "Checking the release notes" in delta
    assert "unavailable at this read: task_result=StoreBusy" in delta
    assert "waits, open questions and settlement are unknown, not ended" in delta
    assert "owner answer" not in delta and "- other ·" not in delta  # no stale wait, no unchanged row
    assert "projection incomplete" in delta
    # The same refusal at the next read changes nothing, so nothing is appended.
    assert not peer_roster.maybe_append_roster_note(ctx, messages, tmp_path)
    # A lost base restores the complete snapshot, the gap still named on its row.
    del messages[:-1]
    assert peer_roster.maybe_append_roster_note(ctx, messages, tmp_path)
    snapshot = messages[-1]["content"]
    assert messages[-1][HOST_CONTEXT_KIND_KEY] == peer_roster.ROSTER_SNAPSHOT_NAME
    assert "unavailable at this read: task_result=StoreBusy" in snapshot and "- other · Other work" in snapshot
    assert "(roster incomplete:" in snapshot
    failing.clear()
    assert peer_roster.maybe_append_roster_note(ctx, messages, tmp_path)
    recovered = messages[-1]["content"]
    assert "owner answer (quiz_id=q1, since=2026-10-26T10:05:00Z)" in recovered and "unavailable" not in recovered


def test_a_gap_that_moves_to_another_row_changes_the_catalogue_snapshot(monkeypatch, tmp_path):
    _peer_world(tmp_path, waits=False)
    failing = {"peer"}
    _unreadable_results(monkeypatch, failing)
    first = peer_roster.live_root_catalogue(tmp_path, limit=1)
    assert first["observation"]["incomplete"] is True and first["next"]
    unavailable = {row["task_id"]: row.get("unavailable") for row in peer_roster.independent_roots(tmp_path)["roots"]}
    assert unavailable == {"peer": {"task_result": "StoreBusy"}, "other": None}
    assert peer_roster.live_root_catalogue(tmp_path, limit=1)["snapshot"] == first["snapshot"]  # same refusal: no churn
    failing.clear()
    failing.add("other")
    moved = peer_roster.live_root_catalogue(tmp_path, limit=1)
    assert moved["observation"]["incomplete"] is True and moved["snapshot"] != first["snapshot"]
    stale = peer_roster.live_root_catalogue(tmp_path, limit=1, offset=1, snapshot=first["snapshot"])
    assert stale["error"]["code"] == "LIVE_ROOTS_SNAPSHOT_CHANGED" and stale["roots"] == []
