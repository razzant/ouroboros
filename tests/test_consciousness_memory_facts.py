"""What a wake observes of memory and what it may write there.

A wake sees the main view (open conversation, the earlier memory, drafts, refusals) and, as
events, only the journal records written after the accepted sequence, under their own header.
Standing inventory is never an event, so a wake on an installation with a journal can still
find nothing new. Without a journal the observation is the base's, byte for byte. Observe keeps
``chronicle_write`` (sealing is internal memory work) while starting work stays withheld.
"""
from __future__ import annotations

import json

import pytest

from ouroboros import chat_chain
from ouroboros import consciousness_authority as ca
from ouroboros import consciousness_wake as wake
from ouroboros import memory_inventory as mi
from ouroboros.chronicle_store import ChronicleStore
from tests import _memory_inventory_shared as shared

NOW = 1_900_000_000.0  # well after the fixture's rows
NOTHING_NEW = "\n- wake cause: scheduled heartbeat; no event reason is recorded for this wake"
HELPER = {"kind": "helper", "writer": "fallback_page", "attribution": "helper draft, not lived"}
WAKE_MIND = {"kind": "mind", "task_id": "wake-1", "focus": {"role": "consciousness", "task_id": "wake-1"}}
MEMORY_HEADER = "Recorded in memory after the accepted sequence (source dates do not order publication):"
RESULTS_HEADER = "Recorded in task results with no chat row in this window (by their own stamps):"


def observe(root, boundary=None):
    return wake.observe_wake(root, boundary=boundary, since=NOW - 1, now=NOW, reason="heartbeat")


def bind(root, observation, task_id):
    task = {"id": task_id}
    boundary = wake.bind_wake_observation(root, task, observation, lambda text: text)
    assert boundary is not None and "source_error" not in task["metadata"][wake.WAKE_OBSERVATION_KEY]
    return boundary


def page(root, room, first, last, author, text):
    from ouroboros.tools.chronicle import page_covers

    addresses = {pos: address for address, _row, pos in chat_chain.iter_rows(root)}
    covers = page_covers(root, room, from_addr=addresses[first], to_addr=addresses[last])["covers"]
    result = ChronicleStore(root).publish_page(room_id=room, text=text, covers=covers, author=author)
    assert result.ok, result
    return result.record


def task_result(root, task_id):
    """A terminal no chat row announces: it is listed under the task-results header."""
    stamp = wake._iso(NOW - 30)
    row = {"task_id": task_id, "status": "completed", "ts": stamp, "updated_at": stamp, "_schema_version": 1,
           "accounted_upper_bound_usd": 0.5, "description": f"do {task_id}", "metadata": {}, "_is_direct_chat": False}
    (root / "task_results").mkdir(exist_ok=True)
    (root / "task_results" / f"{task_id}.json").write_text(json.dumps(row), encoding="utf-8")


def test_records_after_the_accepted_sequence_are_lines_under_their_own_header(tmp_path):
    shared.world(tmp_path)
    accepted = bind(tmp_path, observe(tmp_path), "wake-baseline")
    sealed = page(tmp_path, "1", 11, 12, shared.MIND, "The mind's page text")
    draft = page(tmp_path, "1", 15, 16, HELPER, "The helper's draft text")
    store = ChronicleStore(tmp_path)
    correction = store.correct(sealed["id"], "The mind's corrected text", shared.MIND)
    decision = store.decide(draft["id"], False, WAKE_MIND, "Not what happened")
    assert correction.ok and decision.ok
    task_result(tmp_path, "t-late")
    observation = observe(tmp_path, accepted)
    assert observation.counts() == {"task_terminal": 1, "memory_change": 4}
    assert "1 task terminal, 4 memory changes" in observation.composition()
    lines = observation.full_text().split("\n")
    memory = lines.index(MEMORY_HEADER)
    assert lines.index(RESULTS_HEADER) < memory  # the task terminal stays under its own header
    listed = lines[memory + 1:memory + 5]
    assert [line.split()[1:3] for line in listed] == [["page", sealed["id"]], ["page", "draft"],
                                                      ["correction", correction.record["id"]],
                                                      ["decision", decision.record["id"]]]
    assert listed[0].endswith(f"memory_read(node_id='{sealed['id']}')") and "by mind (root, task t1)" in listed[0]
    assert "seals 2 rows" in listed[0] and f"draft {draft['id']} in room 1, by helper (fallback_page)" in listed[1]
    assert listed[2].endswith(f"target memory_read(node_id='{sealed['id']}')")
    assert "rejects the draft" in listed[3] and listed[3].endswith(f"target memory_read(node_id='{draft['id']}')")
    assert "by mind (consciousness, task wake-1)" in listed[3]
    for text in ("The mind's page text", "The helper's draft text", "corrected text", "Not what happened"):
        assert text not in observation.full_text()  # a line names and addresses a record; the wake reads it
    assert "memory records: sequence " in observation.full_text()
    # Until a wake accepts this observation the boundary stays, and the same changes come again.
    assert [event for event in observe(tmp_path, accepted).events if event[0] == "memory_change"] == [
        event for event in observation.events if event[0] == "memory_change"]
    assert observe(tmp_path, bind(tmp_path, observation, "wake-2")).counts() == {}


def test_standing_inventory_is_never_an_event(tmp_path):
    rooms = shared.world(tmp_path)
    store = ChronicleStore(tmp_path)
    # The import (legacy sections, the activation receipt) is in the journal, but not a change to observe.
    assert {record["kind"] for record in store.records()} >= {"legacy", "activation"}
    imported = observe(tmp_path, {"memory": {"sequence": 0, "record_id": None}})
    assert imported.counts().get("memory_change") is None
    # An old draft, open rows, unfolded periods and a refusal receipt all stand before the accepted wake.
    old = page(tmp_path, str(rooms["alpha"]), 13, 13, HELPER, "An old draft")
    assert store.publish([], scan_state={"fallback_refusals": {"legacy-b01-r1": {"kind": "context_overflow"}}}).ok
    accepted = bind(tmp_path, observe(tmp_path), "wake-baseline")
    assert mi.open_rows_by_room(tmp_path)["1"] and mi.legacy_progress(mi.legacy_units(store, tmp_path))["pending"]
    assert any(unit.refusal for unit in mi.legacy_units(store, tmp_path))
    quiet = observe(tmp_path, accepted)
    assert quiet.events == () and quiet.composition() == "no events" and quiet.full_text() == NOTHING_NEW
    assert old["id"] not in quiet.full_text()
    # A receipt is scan state, not a record: still nothing new.
    assert store.publish([], scan_state={"fallback_refusals": {"legacy-b01-r1": {"kind": "invalid"}}}).ok
    assert observe(tmp_path, accepted).full_text() == NOTHING_NEW
    # The other direction: one new page is one event.
    new = page(tmp_path, "1", 18, 19, shared.MIND, "A new page")
    after = observe(tmp_path, accepted)
    assert after.counts() == {"memory_change": 1} and f"page {new['id']} in room 1" in after.events[0][2]


def test_a_released_mark_is_a_change_that_names_who_released_it(tmp_path):
    """A helper keeps its right to release any mark, the mind's global one included, and the
    next wake sees that release with its author. No release, no line; the mind's own release of a
    room mark is the same kind of line (one rule for every focus)."""
    from ouroboros.tools.chronicle import _memory_mark
    from ouroboros.tools.registry import ToolContext

    shared.world(tmp_path)
    store = ChronicleStore(tmp_path)
    answer = store.mark({"kind": "task", "task_id": "t-answer"}, "The owner answered: fold the old memory",
                        WAKE_MIND, room_id="1", scope="global").record
    kept = store.mark({"kind": "task", "task_id": "t-kept"}, "Keep this in view", WAKE_MIND, room_id="1").record
    accepted = bind(tmp_path, observe(tmp_path), "wake-baseline")
    assert observe(tmp_path, accepted).counts() == {}  # the marks stand before the accepted wake
    child = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="kid-1", current_chat_id=1,
                        task_metadata={"delegation_role": "subagent", "parent_task_id": "t1"})
    released = json.loads(_memory_mark(child, release_id=answer["id"], reason="Looked settled to me"))
    assert released["ok"] and released["operation"] == "mark_release"  # the helper's right is unchanged
    release = next(record for record in store.records() if record["kind"] == "mark_release")
    after_child = observe(tmp_path, accepted)
    assert after_child.counts() == {"memory_change": 1}
    line = after_child.events[0][2]
    assert line.startswith(f"- mark_release {release['id']} global, by mind (child, task kid-1): releases its target mark")
    assert line.endswith(f"target memory_read(node_id='{answer['id']}')")
    assert "Looked settled" not in line and "fold the old memory" not in line  # the wake reads the record
    assert store.release_mark(kept["id"], WAKE_MIND, "Done with it").ok
    lines = [text for kind, _offset, text in observe(tmp_path, accepted).events if kind == "memory_change"]
    assert lines[0] == line and lines[1].startswith("- mark_release ")
    assert " in room 1, by mind (consciousness, task wake-1): releases its target mark" in lines[1]
    assert lines[1].endswith(f"target memory_read(node_id='{kept['id']}')")


def _without_memory(monkeypatch):
    monkeypatch.setattr(mi, "memory_changes", lambda _root, boundary, _gaps: ([], boundary, None))


def test_without_a_journal_the_observation_is_the_bases(tmp_path, monkeypatch):
    log = tmp_path / "logs" / "chat.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text(json.dumps({"ts": wake._iso(NOW), "direction": "in", "chat_id": 1, "text": "hello there",
                               "source": "web", "client_message_id": "m", "ingress_accepted": True}) + "\n",
                   encoding="utf-8")
    task_result(tmp_path, "t-done")
    first = observe(tmp_path)
    with monkeypatch.context() as patch:
        _without_memory(patch)
        base = observe(tmp_path)
    assert first.events == base.events and first.window == base.window
    assert first.full_text() == base.full_text() and "memory" not in first.full_text().lower()
    assert {key: value for key, value in first.boundary.items() if key != "memory"} == base.boundary
    assert not (tmp_path / "memory").exists()
    accepted = bind(tmp_path, first, "wake-1")
    assert observe(tmp_path, accepted).full_text() == NOTHING_NEW
    with monkeypatch.context() as patch:
        _without_memory(patch)
        assert observe(tmp_path, accepted).full_text() == NOTHING_NEW
    assert not (tmp_path / "memory").exists()


# --- what a wake may write ----------------------------------------------------------------------


@pytest.mark.parametrize("level", ca.LEVELS)
def test_every_level_keeps_chronicle_write(level):
    assert "chronicle_write" not in ca.disabled_tools_for(level)
    assert ca.observe_argument_refusal({"initiator": "consciousness", "consciousness_autonomy": level},
                                       "chronicle_write", {"kind": "page"}) == ""


def test_observe_seals_a_page_and_still_cannot_start_work(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    assert {"promote_chat_to_task", "run_command"} <= set(ca.disabled_tools_for("observe"))
    shared.world(tmp_path)
    accepted = bind(tmp_path, observe(tmp_path), "wake-baseline")
    task = ca.apply_consciousness_authority({"id": "wake-1", "metadata": {
        "initiator": "consciousness", "usage_category": "consciousness", "consciousness_autonomy": "observe",
        "model_role": "consciousness"}})
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="wake-1", is_direct_chat=True,
                                     current_chat_id=1, task_metadata=task["metadata"]))
    addresses = {pos: chat_chain.format_address(address) for address, _row, pos in chat_chain.iter_rows(tmp_path)}
    written = json.loads(registry.execute("chronicle_write", {
        "kind": "page", "room_id": "1", "text": "The wake sealed this stretch",
        "covers": {"from": addresses[10], "to": addresses[12]}}))
    assert written["ok"], written
    assert ChronicleStore(tmp_path).get(written["node_id"])["author"]["kind"] == "mind"
    for name, args in (("promote_chat_to_task", {"text": "start this"}), ("run_command", {"cmd": "touch x.py"})):
        result = registry.execute(name, args)
        assert "RESOURCE_CONSTRAINT_BLOCKED" in result and name in result, (name, result[:300])
    assert not (tmp_path / "x.py").exists()
    # The next wake observes what this one sealed.
    following = observe(tmp_path, accepted)
    assert following.counts() == {"memory_change": 1} and written["node_id"] in following.events[0][2]
