"""Chronicle durability, attribution and non-destructive legacy migration."""
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from ouroboros.chronicle_store import ChronicleStore


MIND = {"kind": "mind", "task_id": "t1"}
LIGHT = {"kind": "helper", "model": "configured-light"}


def test_revision_keeps_original_and_exposes_helper_without_approval(tmp_path):
    store = ChronicleStore(tmp_path)
    original = store.append_episode("chat:1", "I chose X", [{"id": "raw-1"}], MIND)
    correction = store.revise(original["id"], "The mind chose Y", LIGHT)
    current = store.room_records("chat:1")[0]
    assert current["text"] == "I chose X"
    assert current["current_text"] == "The mind chose Y"
    assert current["current_author"] == LIGHT
    store.decide_revision(correction["id"], True, MIND, "Checked the original")
    assert store.room_records("chat:1")[0]["current_author"] == LIGHT
    store.decide_revision(correction["id"], False, MIND, "Correction misunderstood the source")
    assert store.room_records("chat:1")[0]["current_text"] == "I chose X"
    assert store.get(correction["id"])["text"] == "The mind chose Y"


def test_frontier_and_scan_publish_atomically_and_replay_after_index_loss(tmp_path):
    store = ChronicleStore(tmp_path)
    original = store.append_episode("r", "first", [], MIND, frontier={"offset": 4}, expected_frontier={})
    store.publish([{"id": "next", "kind": "episode", "room_id": "r", "text": "next"}],
                  expected_frontiers={"r": {"offset": 4}}, frontiers={"r": {"offset": 8}},
                  scan_state={"last_offset": 8})
    with pytest.raises(ValueError, match="frontier changed"):
        store.publish([{"id": "loser", "kind": "episode"}], expected_frontiers={"r": {}})
    store.index_path.unlink()
    assert store.get(original["id"])["text"] == "first"
    assert store.get("loser") is None
    assert store.room_state("r")["frontier"] == {"offset": 8}
    assert store.scan_state() == {"last_offset": 8}


def test_failed_index_commit_replays_logged_transaction(tmp_path, monkeypatch):
    store = ChronicleStore(tmp_path)
    store.records()
    original = store._project
    def fail(db, tx):
        raise OSError("simulated interruption after durable append")
    monkeypatch.setattr(store, "_project", fail)
    with pytest.raises(OSError):
        store.publish([{"id": "durable", "kind": "episode", "text": "kept"}], scan_state={"offset": 7})
    monkeypatch.setattr(store, "_project", original)
    assert store.get("durable")["text"] == "kept"
    assert store.scan_state() == {"offset": 7}


def test_duplicate_publication_replays_and_collision_is_refused(tmp_path):
    store = ChronicleStore(tmp_path)
    record = {"id": "stable", "kind": "episode", "text": "once"}
    first = store.publish([record])
    before = store.log_path.read_bytes()
    assert store.publish([record]) == first
    assert store.log_path.read_bytes() == before
    with pytest.raises(ValueError, match="collision"):
        store.publish([{**record, "text": "not once"}])
    assert store.get("stable")["text"] == "once"


def test_concurrent_writers_and_index_corruption_preserve_records(tmp_path):
    def write(n):
        return ChronicleStore(tmp_path).append_episode("r", str(n), [], MIND)
    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(write, range(16)))
    store = ChronicleStore(tmp_path)
    assert len(store.room_records("r")) == 16
    store.index_path.write_bytes(b"not a sqlite database")
    assert {r["id"] for r in store.room_records("r")} == {r["id"] for r in records}


def test_marks_release_is_explicit_and_room_projection_does_not_change_sources(tmp_path):
    store = ChronicleStore(tmp_path)
    source = {"kind": "raw_chat", "id": "original-main-row", "chat_id": 1}
    mark = store.mark(source, "Keep this decision", MIND, room_id="r", scope="global", quote="exact")
    assert store.active_marks("other")[0]["target_ref"] == source
    with pytest.raises(ValueError):
        store.release_mark(mark["id"], MIND, "")
    store.release_mark(mark["id"], MIND, "Superseded by owner's later choice")
    assert store.active_marks("r") == []
    assert store.get(mark["id"])["target_ref"] == source


def test_import_unfolds_era_preserves_source_bytes_and_does_not_duplicate_view(tmp_path):
    from ouroboros.consolidator import retain_memory_source
    source = [{"type": "summary", "content": "old detail", "range": "2026-01-01", "rooms": [
        {"room_id": "chat:1", "label": "Main", "content": "old detail"}]}]
    ref = retain_memory_source(SimpleNamespace(drive_root=tmp_path, task_id="legacy"),
                               "blocks", json.dumps(source).encode(), "json")
    era = {"type": "era", "content": "whole life", "source_ref": ref, "rooms": [
        {"room_id": "chat:1", "label": "Main", "content": "whole life"}]}
    memory = tmp_path / "memory"
    memory.mkdir(exist_ok=True)
    files = {"dialogue_blocks.json": json.dumps([era]), "dialogue_meta.json": '{"last_offset":17}',
             "dialogue_summary.md": "whole life"}
    for name, text in files.items():
        (memory / name).write_text(text, encoding="utf-8")
    store = ChronicleStore(tmp_path)
    receipt = store.import_legacy()
    assert receipt["kind"] == "activation"
    assert store.scan_state() == {"last_offset": 17}
    assert [r["text"] for r in store.room_records("chat:1")] == ["whole life"]
    roots = store.room_records("chat:1")
    assert store.get(roots[0]["metadata"]["children"][0])["text"] == "old detail"
    assert not store.room_records("legacy")
    store.publish([], scan_state={"last_offset": 25})
    assert store.import_legacy() == receipt
    assert store.scan_state() == {"last_offset": 25}
    assert {name: (memory / name).read_text(encoding="utf-8") for name in files} == files


def test_import_preserves_flat_and_activates_with_source_bound_corruption_gap(tmp_path):
    memory = tmp_path / "memory"
    memory.mkdir()
    path = memory / "dialogue_blocks.json"
    path.write_text("{broken", encoding="utf-8")
    (memory / "dialogue_summary.md").write_text("Only surviving biography", encoding="utf-8")
    store = ChronicleStore(tmp_path)
    assert store.import_legacy()["kind"] == "activation"
    assert path.read_text(encoding="utf-8") == "{broken"
    assert store.room_records("legacy")[0]["text"] == "Only surviving biography"
    gap = store.room_records("legacy")[1]
    assert gap["metadata"]["coverage"] == "unknown" and gap["source_refs"]


def test_hot_lookup_reads_only_unindexed_log_suffix(tmp_path, monkeypatch):
    store = ChronicleStore(tmp_path)
    store.append_episode("r", "hello", [], MIND, record_id="x")
    original = store._project
    def fail_on_replay(db, tx):
        pytest.fail("hot lookup replayed an already indexed transaction")
    monkeypatch.setattr(store, "_project", fail_on_replay)
    assert store.get("x")["text"] == "hello"
    monkeypatch.setattr(store, "_project", original)


def test_batch_identity_collision_cannot_poison_log(tmp_path):
    store = ChronicleStore(tmp_path)
    with pytest.raises(ValueError, match="collision"):
        store.publish([{"id": "same", "kind": "episode", "text": "a"},
                       {"id": "same", "kind": "episode", "text": "b"}])
    assert not store.log_path.exists()
    assert store.records() == []


def test_import_missing_era_source_keeps_available_biography(tmp_path):
    memory = tmp_path / "memory"
    memory.mkdir()
    era = {"type": "era", "content": "Only surviving era", "source_ref": {
        "kind": "task_source", "root": "artifact_store", "task_id": "missing",
        "path": "_sources/context_checkpoints/missing.json", "size": 1, "sha256": "missing"}}
    (memory / "dialogue_blocks.json").write_text(json.dumps([era]), encoding="utf-8")
    store = ChronicleStore(tmp_path)
    store.import_legacy()
    row = store.room_records("legacy")[0]
    assert row["text"] == "Only surviving era"
    assert row["metadata"]["source_gap"]


def test_rejected_later_correction_exposes_decision_and_prior_interpretation(tmp_path):
    store = ChronicleStore(tmp_path)
    base = store.append_episode("r", "original", [], MIND)
    first = store.revise(base["id"], "correction 1", LIGHT)
    second = store.revise(base["id"], "correction 2", LIGHT)
    store.decide_revision(second["id"], False, MIND, "Not source-grounded")
    row = store.room_records("r")[0]
    assert row["correction"]["id"] == first["id"]
    assert row["current_text"] == "correction 1"
    assert row["revisions"][-1]["decision"]["accepted"] is False


def test_import_waits_for_legacy_writer_publication(tmp_path):
    import os
    from ouroboros.platform_layer import file_lock_exclusive_nb, file_unlock
    memory = tmp_path / "memory"
    memory.mkdir()
    path = memory / "dialogue_blocks.json"
    path.write_text("[]", encoding="utf-8")
    fd = os.open(str(memory / ".consolidation.lock"), os.O_CREAT | os.O_WRONLY, 0o644)
    file_lock_exclusive_nb(fd)
    store = ChronicleStore(tmp_path)
    try:
        assert store.import_legacy()["kind"] == "import_pending"
        assert store.activation() is None
        path.write_text(json.dumps([{"content": "writer just finished"}]), encoding="utf-8")
    finally:
        file_unlock(fd)
        os.close(fd)
    store.import_legacy()
    assert store.room_records("legacy")[0]["text"] == "writer just finished"


def test_source_identity_uses_original_fields_and_canonical_key_order():
    from ouroboros.chronicle_store import source_row_id
    row = {"text": "Привет", "chat_id": 1, "task_id": "t"}
    assert source_row_id(row) == source_row_id(dict(reversed(list(row.items()))))
    assert source_row_id(row) != source_row_id({**row, "text": "Привет!"})


@pytest.mark.serial
def test_two_processes_publish_without_losing_frontiers(tmp_path):
    import subprocess
    import sys
    script = """
import sys
from ouroboros.chronicle_store import ChronicleStore
store = ChronicleStore(sys.argv[1])
room = sys.argv[2]
for i in range(8):
    store.append_episode(room, str(i), [], {'kind':'mind'}, frontier={'offset':i+1})
"""
    children = [subprocess.Popen([sys.executable, "-c", script, str(tmp_path), room],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                for room in ("a", "b")]
    try:
        for child in children:
            stdout, stderr = child.communicate(timeout=30)
            assert child.returncode == 0, (stdout, stderr)
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait()
    store = ChronicleStore(tmp_path)
    assert len(store.room_records("a")) == len(store.room_records("b")) == 8
    assert store.room_state("a")["frontier"] == store.room_state("b")["frontier"] == {"offset": 8}


def test_pending_corrections_use_real_published_receipts(tmp_path):
    store = ChronicleStore(tmp_path)
    plain = store.append_episode("r", "unbound", [], MIND)
    bound = store.append_episode("r", "bound", [{"id": "raw"}], MIND)
    store.revise(bound["id"], "my manual update", MIND)
    assert [r["id"] for r in store.pending_episodes()] == [bound["id"]]
    store.revise(bound["id"], "source checked", LIGHT, record_id_override="correction:" + bound["id"],
                 metadata={"auto_correction": True})
    assert store.pending_episodes() == []
    assert store.get(plain["id"])["text"] == "unbound"


def test_torn_transaction_keeps_prior_memory_gap_and_allows_new_publication(tmp_path):
    store = ChronicleStore(tmp_path)
    first = store.append_episode("1", "Completed memory", [], MIND, frontier={"offset": 3})
    original_prefix = store.log_path.read_bytes()
    torn = b'{"kind":"transaction","records":[{"id":"lost","kind":"episode"'
    with store.log_path.open("ab") as stream:
        stream.write(torn)
    assert store.get(first["id"])["text"] == "Completed memory"
    gap = store.room_cover("legacy")[0]
    assert gap["kind"] == "gap"
    source = gap["source_refs"][0]
    assert store.log_path.read_bytes()[source["start_byte"]:source["end_byte"]] == torn
    assert store.room_state("1")["frontier"] == {"offset": 3}
    assert store.get("lost") is None
    second = store.append_episode("1", "New memory after interruption", [], MIND, frontier={"offset": 4})
    assert store.log_path.read_bytes().startswith(original_prefix + torn)
    assert b"\n" in store.log_path.read_bytes()[source["end_byte"]:]
    before = store.room_cover("legacy")
    store.index_path.unlink()
    assert store.get(first["id"])["text"] == "Completed memory"
    assert store.get(second["id"])["text"] == "New memory after interruption"
    assert store.room_cover("legacy") == before
    assert store.room_state("1")["frontier"] == {"offset": 4}
    assert store.log_path.read_bytes().startswith(original_prefix + torn)


def test_invalid_complete_transaction_cannot_publish_its_first_record_or_frontier(tmp_path):
    store = ChronicleStore(tmp_path)
    store.append_episode("1", "Committed", [], MIND, frontier={"offset": 5})
    invalid = {"kind": "transaction", "records": [{"id": "phantom", "kind": "episode", "room_id": "1", "text": "never committed"}, {}],
               "frontiers": {"1": {"offset": 999}}}
    with store.log_path.open("ab") as stream:
        stream.write((json.dumps(invalid) + "\n").encode())
    assert store.get("phantom") is None
    assert store.room_state("1")["frontier"] == {"offset": 5}
    assert store.room_records("legacy")[0]["kind"] == "gap"
