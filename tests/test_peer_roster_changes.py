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
