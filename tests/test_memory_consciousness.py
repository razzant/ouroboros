"""A wake's view of the memory journal: the accepted sequence, exact reads, disclosed gaps.

``observe_wake`` lists the journal records written after the sequence the last ACCEPTED wake
observed (``memory_inventory.memory_changes``), one line of text with the exact read of each
record and of its target. The boundary moves only when a wake binds and is accepted; an empty
install gets no memory directory; a read failure or a replaced anchor is a disclosed gap.
"""
from __future__ import annotations

from ouroboros import consciousness_wake as wake
from ouroboros.chronicle_store import ChronicleStore

NOW = 1_900_000_000.0
MIND = {"kind": "mind", "task_id": "t-mind", "focus": {"role": "root", "task_id": "t-mind"}}
HELPER = {"kind": "helper", "writer": "fallback_page", "route": "actual-helper"}


def observe(root, boundary=None):
    return wake.observe_wake(root, boundary=boundary, since=NOW - 1, now=NOW, reason="heartbeat")


def bind(root, observation, task_id):
    task = {"id": task_id}
    boundary = wake.bind_wake_observation(root, task, observation, lambda text: text)
    assert boundary is not None and "source_error" not in task["metadata"][wake.WAKE_OBSERVATION_KEY]
    return boundary


def note(store, key):
    result = store.publish([{"id": key, "kind": "note", "room_id": "7", "task_id": "t-mind",
                             "text": "Exact original meaning " + key, "author": MIND}])
    assert result.ok, result
    return result.record


def draft(store, key):
    result = store.publish_page(room_id="7", text="Helper interpretation " + key, record_id=key, author=HELPER,
                                covers={"rows": [key * 4], "stream_span": [0, 0]})
    assert result.ok, result
    return result.record


def test_an_empty_install_accepts_zero_without_creating_memory_storage(tmp_path):
    observation = observe(tmp_path)
    assert observation.boundary["memory"] == {"sequence": 0, "record_id": None}
    assert not (tmp_path / "memory").exists()
    assert not observation.events and "memory records" not in observation.full_text()
    accepted = bind(tmp_path, observation, "wake-empty")
    assert accepted["memory"] == {"sequence": 0, "record_id": None}
    assert observe(tmp_path, accepted).events == () and not (tmp_path / "memory").exists()


def test_an_existing_journal_starts_a_baseline_without_reading_its_past(tmp_path, monkeypatch):
    store = ChronicleStore(tmp_path)
    old = note(store, "old")
    monkeypatch.setattr(ChronicleStore, "records",
                        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("full corpus read")))
    first = observe(tmp_path)
    assert first.events == ()
    assert first.boundary["memory"] == {"sequence": old["sequence"], "record_id": "old"}
    text = first.full_text()
    assert "first observed sequence, earlier changes not inventoried" in text
    assert "memory_read(node_id='old')" in text
    assert "Exact original meaning" not in text


def test_the_accepted_source_roundtrip_reports_only_new_records_and_their_targets(tmp_path):
    accepted = bind(tmp_path, observe(tmp_path), "wake-baseline")
    store = ChronicleStore(tmp_path)
    original = note(store, "original")
    correction = store.correct(original["id"], "The mind's corrected meaning", MIND)
    page = draft(store, "d")
    decision = store.decide(page["id"], False, MIND, "Incorrect interpretation")
    assert correction.ok and decision.ok
    observation = observe(tmp_path, accepted)
    assert observation.counts() == {"memory_change": 4}
    lines = [line for _kind, _offset, line in observation.events]
    assert [line.split()[1] for line in lines] == ["note", "correction", "page", "decision"]
    assert lines[1].endswith(f"target memory_read(node_id='{original['id']}')")
    assert lines[2].startswith("- page draft d in room 7, by helper (fallback_page): seals 1 row")
    assert "rejects the draft" in lines[3] and lines[3].endswith("target memory_read(node_id='d')")
    for line, record in zip(lines, (original, correction.record, page, decision.record)):
        assert f"memory_read(node_id='{record['id']}')" in line
    text = observation.full_text()
    assert "Exact original meaning" not in text and "corrected meaning" not in text
    assert "Recorded in task results" not in text
    assert "Recorded in memory after the accepted sequence (source dates do not order publication):" in text
    # An observation the wake did not accept never moves the caller's boundary.
    assert observe(tmp_path, accepted).events == observation.events
    # This write races binding, after the observation's captured upper bound.
    late = note(store, "appended-after-observation")
    accepted = bind(tmp_path, observation, "wake-with-memory")
    assert accepted["memory"] == {"sequence": decision.record["sequence"], "record_id": decision.record["id"]}
    following = observe(tmp_path, accepted)
    assert following.counts() == {"memory_change": 1} and late["id"] in following.events[0][2]
    last = bind(tmp_path, following, "wake-next")
    assert observe(tmp_path, last).events == ()
    # The memory position is part of the bound source: an accepted boundary that disagrees with it is not trusted.
    tampered = observe(tmp_path, {**last, "memory": {**last["memory"], "sequence": 1}})
    assert "unreadable_transition_source: ValueError" in tampered.gaps and tampered.boundary is None


def test_a_memory_read_failure_is_a_gap_and_keeps_the_accepted_sequence(tmp_path, monkeypatch):
    store = ChronicleStore(tmp_path)
    note(store, "old")
    accepted = bind(tmp_path, observe(tmp_path), "wake-baseline")
    note(store, "new")
    with monkeypatch.context() as fault:
        fault.setattr(ChronicleStore, "observation_snapshot",
                      lambda *_a, **_kw: (_ for _ in ()).throw(OSError("locked")))
        blind = observe(tmp_path, accepted)
    assert blind.boundary["memory"] == accepted["memory"]
    assert "memory changes unreadable: OSError; accepted sequence retained" in blind.gaps
    assert "memory changes unreadable" in blind.full_text()
    resumed = observe(tmp_path, bind(tmp_path, blind, "wake-blind"))
    assert resumed.counts() == {"memory_change": 1} and " new in room 7" in resumed.events[0][2]


def test_a_replaced_memory_anchor_never_consumes_history(tmp_path):
    store = ChronicleStore(tmp_path)
    note(store, "old")
    first = observe(tmp_path)
    first.boundary["memory"]["record_id"] = "different-source"
    broken = observe(tmp_path, first.boundary)
    assert broken.boundary["memory"] == first.boundary["memory"]
    assert "memory changes unreadable: ValueError; accepted sequence retained" in broken.gaps
    assert not broken.events
    # A missing journal under an accepted non-zero sequence is the same gap, not a fresh start.
    (tmp_path / "elsewhere").mkdir()
    missing = observe(tmp_path / "elsewhere", {"memory": {"sequence": 3, "record_id": "x"}})
    assert missing.boundary["memory"] == {"sequence": 3, "record_id": "x"}
    assert "memory changes unreadable: ValueError; accepted sequence retained" in missing.gaps
    assert not (tmp_path / "elsewhere" / "memory").exists()
