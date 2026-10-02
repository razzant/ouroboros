"""Ordinary coarsest views read selected bodies, not the lifetime corpus."""
from ouroboros.chronicle_store import ChronicleStore


def _records(count=1000):
    return [{"id": f"e{n}", "kind": "episode", "room_id": "1", "text": f"event {n} " + "x" * 2000,
             "author": {"kind": "mind", "task_id": f"task-{n}"}, "metadata": {"task_ids": [f"task-{n}"]}}
            for n in range(count)]


def _digests(count=1000):
    return [{"id": f"d{n}", "kind": "digest", "room_id": "1", "text": f"Meaning of interval {n}",
             "author": {"kind": "helper"}, "metadata": {"covers_record_ids": [f"e{k}" for k in range(n, n + 100)]}}
            for n in range(0, count, 100)]


def test_thousand_episode_cover_never_decodes_hidden_bodies_on_hot_or_dirty_read(tmp_path, monkeypatch):
    from ouroboros.chronicle_view import _cuts
    store = ChronicleStore(tmp_path)
    store.publish(_records() + _digests())
    expected = [row["id"] for row in _cuts(store.room_records("1"))[-1]]
    read_ids = []
    original = store._record_at
    def counted(db, record_id):
        read_ids.append(record_id)
        return original(db, record_id)
    monkeypatch.setattr(store, "_record_at", counted)
    assert store.room_ids() == ["1"]
    assert read_ids == []
    assert [row["id"] for row in store.room_cover("1")] == expected
    assert read_ids == expected and len(read_ids) == 10
    read_ids.clear()
    assert len(store.room_cover("1")) == 10
    assert read_ids == expected
    # A new uncovered event must not force re-decoding the old thousand bodies.
    store.append_episode("1", "New meaningful event", [], {"kind": "mind"}, record_id="new")
    read_ids.clear()
    cover = store.room_cover("1")
    assert len(cover) == len(read_ids) == 11
    assert not any(record_id.startswith("e") for record_id in read_ids)
    assert {row["id"] for row in cover} == set(expected) | {"new"}


def test_stale_digest_expands_only_changed_branch_and_reject_restores_cover(tmp_path):
    store = ChronicleStore(tmp_path)
    store.publish(_records(200) + _digests(200))
    assert {r["id"] for r in store.room_cover("1")} == {"d0", "d100"}
    revision = store.revise("e5", "Owner reversed that choice", {"kind": "helper"})
    expanded = store.room_cover("1")
    ids = {row["id"] for row in expanded}
    assert "d0" not in ids and "d100" in ids
    assert len(ids) == 101
    corrected = next(row for row in expanded if row["id"] == "e5")
    assert corrected["current_text"] == "Owner reversed that choice"
    store.decide_revision(revision["id"], False, {"kind": "mind"}, "Correction not grounded")
    assert {row["id"] for row in store.room_cover("1")} == {"d0", "d100"}
    before = store.room_cover("1")
    store.index_path.unlink()
    assert store.room_cover("1") == before


def test_task_lookup_survives_new_room_binding_without_moving_storage(tmp_path, monkeypatch):
    store = ChronicleStore(tmp_path)
    store.publish(_records() + _digests())
    read_ids = []
    original = store._record_at
    def counted(db, record_id):
        read_ids.append(record_id)
        return original(db, record_id)
    monkeypatch.setattr(store, "_record_at", counted)
    rows = store.records_for_tasks(["task-997"])
    assert read_ids == ["e997"]
    assert rows[0]["room_id"] == "1"
    assert rows[0]["metadata"]["task_ids"] == ["task-997"]
    assert store.records_for_tasks(["nonexistent"]) == []


def test_nested_digest_cover_uses_exact_revision_dependencies(tmp_path):
    store = ChronicleStore(tmp_path)
    store.publish(_records(200) + _digests(200) + [{
        "id": "whole", "kind": "digest", "room_id": "1", "text": "The complete arc",
        "author": {"kind": "helper"}, "metadata": {"covers_record_ids": ["d0", "d100"]}}])
    assert [r["id"] for r in store.room_cover("1")] == ["whole"]
    corrected = store.revise("e5", "Important correction", {"kind": "mind"})
    assert "whole" not in {r["id"] for r in store.room_cover("1")}
    store.decide_revision(corrected["id"], False, {"kind": "mind"}, "Superseded revision")
    assert [r["id"] for r in store.room_cover("1")] == ["whole"]
