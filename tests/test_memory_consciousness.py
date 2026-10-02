"""Memory change awareness uses the accepted wake boundary and exact source tools."""
import json

from ouroboros import consciousness_wake as wake
from ouroboros.chronicle_store import ChronicleStore


NOW = 1_900_000_000.0


def observe(root, boundary=None):
    return wake.observe_wake(root, boundary=boundary, since=NOW - 1, now=NOW, reason="heartbeat")


def bind(root, observation, task_id):
    task = {"id": task_id}
    boundary = wake.bind_wake_observation(root, task, observation, lambda text: text)
    assert boundary is not None and "source_error" not in task["metadata"][wake.WAKE_OBSERVATION_KEY]
    return boundary


def episode(store, key):
    return store.publish([{"id": key, "kind": "episode", "room_id": "old-room",
                           "ts": "2000-01-01T00:00:00Z", "text": "Exact original meaning " + key,
                           "author": {"kind": "mind", "task_id": "original-mind"}}])[0]


def test_empty_install_accepts_zero_without_creating_memory_storage(tmp_path):
    observation = observe(tmp_path)
    assert observation.boundary["memory"] == {"sequence": 0, "record_id": None}
    assert not (tmp_path / "memory").exists()
    assert not observation.events and "memory records:" not in observation.full_text()


def test_existing_corpus_bootstraps_index_only_with_explicit_historical_read(tmp_path, monkeypatch):
    store = ChronicleStore(tmp_path)
    old = episode(store, "old")
    monkeypatch.setattr(ChronicleStore, "records", lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("full corpus read")))
    first = observe(tmp_path)
    assert first.events == ()
    assert first.boundary["memory"]["record_id"] == old["id"]
    assert "first observed sequence, earlier changes not inventoried" in first.full_text()
    assert "memory_read(node_id=old)" in first.full_text()
    assert "Exact original meaning" not in first.full_text()


def test_accepted_source_roundtrip_reports_only_new_old_stamped_records_and_originals(tmp_path):
    accepted = bind(tmp_path, observe(tmp_path), "wake-baseline")
    store = ChronicleStore(tmp_path)
    original = episode(store, "original")
    revised = store.revise(original["id"], "Helper interpretation", {"kind": "helper", "route": "actual-helper"})
    decision = store.decide_revision(revised["id"], False, {"kind": "mind"}, "Incorrect interpretation")
    observation = observe(tmp_path, accepted)
    assert observation.counts() == {"memory_change": 3}
    facts = [json.loads(line.removeprefix("- memory change: ")) for _, _, line in observation.events]
    assert [fact["kind"] for fact in facts] == ["episode", "revision", "revision_decision"]
    assert facts[1]["author"] == {"kind": "helper", "route": "actual-helper"}
    assert facts[1]["read_original"]["arguments"]["node_id"] == original["id"]
    assert facts[2]["read_original"]["arguments"]["node_id"] == revised["id"]
    assert facts[2]["accepted"] is False and "Exact original meaning" not in observation.full_text()
    from types import SimpleNamespace
    from ouroboros.tools.chronicle import _memory_read
    ctx = SimpleNamespace(drive_root=tmp_path, task_id="mind-reading-changes")
    source = json.loads(_memory_read(ctx, **facts[1]["read_original"]["arguments"]))
    correction = json.loads(_memory_read(ctx, **facts[1]["read"]["arguments"]))
    assert source["original"]["text"] == original["text"]
    assert correction["original"]["text"] == "Helper interpretation"

    assert "Recorded in task results" not in observation.full_text()
    assert "Recorded in memory" in observation.full_text()
    # A rejected/unaccepted wake never moves the caller's boundary.
    assert observe(tmp_path, accepted).events == observation.events
    # This write races binding, after the observation's captured upper bound.
    late = episode(store, "appended-after-observation")
    accepted = bind(tmp_path, observation, "wake-with-memory")
    assert accepted["memory"]["record_id"] == decision["id"]
    following = observe(tmp_path, accepted)
    assert following.counts() == {"memory_change": 1} and late["id"] in following.events[0][2]
    assert observe(tmp_path, bind(tmp_path, following, "wake-next")).events == ()


def test_memory_read_failure_discloses_gap_and_preserves_accepted_sequence(tmp_path, monkeypatch):
    store = ChronicleStore(tmp_path)
    episode(store, "old")
    accepted = bind(tmp_path, observe(tmp_path), "wake-baseline")
    episode(store, "new")
    with monkeypatch.context() as fault:
        fault.setattr(ChronicleStore, "observation_snapshot", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("locked")))
        blind = observe(tmp_path, accepted)
    assert blind.boundary["memory"] == accepted["memory"]
    assert "memory changes unreadable: OSError; accepted sequence retained" in blind.gaps
    resumed = observe(tmp_path, bind(tmp_path, blind, "wake-blind"))
    assert resumed.counts() == {"memory_change": 1} and '"id": "new"' in resumed.events[0][2]


def test_mismatched_memory_anchor_never_consumes_replaced_history(tmp_path):
    store = ChronicleStore(tmp_path)
    episode(store, "old")
    first = observe(tmp_path)
    first.boundary["memory"]["record_id"] = "different-source"
    broken = observe(tmp_path, first.boundary)
    assert broken.boundary["memory"] == first.boundary["memory"]
    assert "memory changes unreadable: ValueError; accepted sequence retained" in broken.gaps
    assert not broken.events
