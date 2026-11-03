"""Legacy catch-up is one addressed compatibility step, not normal journal replay."""

import json
from types import SimpleNamespace

import pytest

from ouroboros import store, store_hot_facts, swarm_rollup, task_results, task_store
from ouroboros.swarm_receipt_sources import tree_inputs
from ouroboros.task_finalization import build_swarm_efficiency
from ouroboros.utils import append_jsonl
from tests._store_shared import seed_result


def event(identity, lane, child):
    return {"type": "swarm_fanout", "event_id": identity, "parent_task_id": "parent",
            "task_ids": [child], "requested_count": 1, "fanout_interval_sec": 0.5,
            "requested_model_lane": lane}


@pytest.mark.parametrize("prior", ["missing", "v1"])
def test_finalization_catches_up_legacy_once_and_new_event_ids_are_not_doubled(tmp_path, monkeypatch, prior):
    seed_result(tmp_path, "parent", {"status": "running"})
    path = tmp_path / "logs/events.jsonl"
    append_jsonl(path, {"type": "swarm_fanout", "parent_task_id": "parent",
                        "task_ids": ["old"], "requested_count": 1,
                        "inter_wave_latency_sec": 1.5, "requested_model_lane": "legacy"})
    if prior == "v1":
        task_store.patch_document(tmp_path, "parent", lambda row: {**row, "swarm_fanout_rollup": {
            "version": 1, "source_root": str(tmp_path.resolve()), "through": path.stat().st_size,
            "fanout_count": 1, "requested_count": 1, "fanout_interval_sec_total": 1.5,
            "child_ids": ["old"], "lanes_requested": ["legacy"]}})
    # Reverse-lexical ids make UUID sorting visibly wrong, including a retry.
    for facts in (event("swarm:z-first", "first", "one"), event("swarm:a-second", "second", "two")):
        swarm_rollup.record_swarm_fanout(tmp_path, facts)
        append_jsonl(path, facts)
    swarm_rollup.record_swarm_fanout(tmp_path, event("swarm:z-first", "first", "one"))
    append_jsonl(path, event("swarm:z-first", "first", "one"))
    result = build_swarm_efficiency(SimpleNamespace(drive_root=tmp_path), {
        "id": "parent", "metadata": {"subagents_mode": "swarm"}})
    assert result is not None
    assert result["fanout_count"] == result["requested_count"] == result["subagent_count"] == 3
    assert result["fanout_interval_sec_total"] == 2.5
    assert result["lanes_requested"] == ["legacy", "first", "second"]
    assert task_results.load_task_result(tmp_path, "parent")["swarm_fanout_rollup"]["version"] == 2
    # A new unrelated suffix, a corrupt log, or no log cannot affect v2 consumers.
    path.write_text("broken\n", encoding="utf-8")
    monkeypatch.setattr(swarm_rollup, "JsonlChainSnapshot", lambda *_: pytest.fail("v2 reopened history"))
    assert build_swarm_efficiency(SimpleNamespace(drive_root=tmp_path), {
        "id": "parent", "metadata": {"subagents_mode": "swarm"}}) == result


def test_rollup_first_read_repairs_legacy_before_receipt_snapshot(tmp_path, monkeypatch):
    seed_result(tmp_path, "parent", {"status": "running"})
    append_jsonl(tmp_path / "logs/events.jsonl", event("old", "legacy", "missing-child"))
    seen = []
    snapshot = swarm_rollup.JsonlChainSnapshot

    def outside_transaction(path):
        # A writer can acquire this store while history is read: receipt took no
        # journal snapshot under its task/custody read transaction.
        with store.open_store(tmp_path, write=True) as txn:
            txn.set_meta("fixture_repair_probe", True)
        seen.append(path)
        return snapshot(path)

    monkeypatch.setattr(swarm_rollup, "JsonlChainSnapshot", outside_transaction)
    assert "fanout_history_unavailable" in tree_inputs(tmp_path, "parent")["coverage"]["gaps"]
    assert seen == []
    assert swarm_rollup.load_swarm_rollup(tmp_path, "parent")["child_ids"] == ["missing-child"]
    first = tree_inputs(tmp_path, "parent")
    assert len(seen) == 1
    assert "fanout_history_unavailable" not in first["coverage"]["gaps"]
    assert "missing_child_results" in first["coverage"]["gaps"]
    monkeypatch.setattr(swarm_rollup, "JsonlChainSnapshot", lambda *_: pytest.fail("receipt reopened v2 history"))
    assert "missing_child_results" in tree_inputs(tmp_path, "parent")["coverage"]["gaps"]


@pytest.mark.parametrize("damage", ["absent", "torn", "corrupt", "unreadable"])
def test_automatic_legacy_failure_keeps_unknown_without_publishing(tmp_path, monkeypatch, damage):
    seed_result(tmp_path, "parent", {"status": "running"})
    path = tmp_path / "logs/events.jsonl"
    if damage != "absent":
        append_jsonl(path, event("old", "legacy", "child"))
        if damage == "torn":
            with path.open("ab") as handle:
                handle.write(b'{"type":"swarm_fanout"')
        elif damage == "corrupt":
            with path.open("ab") as handle:
                handle.write(b'broken\n')
        else:
            monkeypatch.setattr(swarm_rollup, "JsonlChainSnapshot", lambda *_: (_ for _ in ()).throw(OSError("denied")))
    assert build_swarm_efficiency(SimpleNamespace(drive_root=tmp_path), {"id": "parent"}) is None
    assert "swarm_fanout_rollup" not in task_results.load_task_result(tmp_path, "parent")
    assert "fanout_history_unavailable" in tree_inputs(tmp_path, "parent")["coverage"]["gaps"]
    assert "swarm_fanout_rollup" not in task_results.load_task_result(tmp_path, "parent")


def test_automatic_repair_race_stays_unknown_then_next_observation_catches_up(tmp_path, monkeypatch):
    seed_result(tmp_path, "parent", {"status": "running"})
    append_jsonl(tmp_path / "logs/events.jsonl", event("old", "legacy", "old-child"))
    patch = task_store.patch_document

    def racing(*args, **kwargs):
        swarm_rollup.record_swarm_fanout(tmp_path, event("new", "new", "new-child"))
        return patch(*args, **kwargs)

    with monkeypatch.context() as race:
        race.setattr(task_store, "patch_document", racing)
        assert build_swarm_efficiency(SimpleNamespace(drive_root=tmp_path), {"id": "parent"}) is None
    assert "swarm_fanout_rollup" not in task_results.load_task_result(tmp_path, "parent")
    assert swarm_rollup.load_swarm_rollup(tmp_path, "parent")["fanout_count"] == 2


def test_reverse_id_insertion_order_survives_retry_copyback_and_export_import(tmp_path):
    from ouroboros import store_import
    from ouroboros.post_task_checkpoint import project_replica_task_result_fields

    root, replica, imported = (tmp_path / name for name in ("root", "replica", "imported"))
    for drive in (root, replica):
        task_results.write_task_result(drive, "parent", "running")
    first, second = event("swarm:z-first", "first", "one"), event("swarm:a-second", "second", "two")
    swarm_rollup.record_swarm_fanout(root, first)
    swarm_rollup.record_swarm_fanout(replica, second)
    swarm_rollup.record_swarm_fanout(root, first)
    incoming = task_results.load_task_result(replica, "parent")
    for _ in range(2):
        task_results.write_task_result(root, "parent", "completed", _field_projector=lambda current, _: {
            **project_replica_task_result_fields(current, incoming), "status": "completed"})
        assert swarm_rollup.load_swarm_rollup(root, "parent")["lanes_requested"] == ["first", "second"]
    with store.open_store(root, write=True, service=True) as txn:
        txn.set_meta("phase", store.EXPORTING)
    store_import.export_facts(root)
    exported = json.loads((root / "task_results/parent.json").read_text(encoding="utf-8"))
    assert list(exported["swarm_fanouts"]) == ["swarm:z-first", "swarm:a-second"]
    (imported / "task_results").mkdir(parents=True)
    (imported / "task_results/parent.json").write_text(json.dumps(exported), encoding="utf-8")
    store_import.ensure_current_store(imported)
    assert swarm_rollup.load_swarm_rollup(imported, "parent")["lanes_requested"] == ["first", "second"]


def test_complete_old_writer_roundtrip_invalidates_list_stable(tmp_path, monkeypatch):
    from tests.test_task_list_selection import client_for

    task_results.write_task_result(tmp_path, "old", "completed", result="answer")
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.row("old", "list_stable") == 1
    derive = store_hot_facts.task_columns
    with monkeypatch.context() as old:
        old.setattr(store_hot_facts, "task_columns", lambda doc: {
            key: value for key, value in derive(doc).items() if key != "list_stable"})
        # Execute the real body UPSERT + typed REPLACE + handoffs/card/obligations
        # writer, with the pre-feature column producer, rather than isolated SQL.
        task_results.write_task_result(tmp_path, "old", "completed", result="new answer",
            workspace_root=str(tmp_path / "workspace"), artifact_status="pending")
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.row("old", "list_stable") is None
        assert txn.load("old")["result"] == "new answer"
    with client_for(tmp_path) as client:
        rows = client.get("/api/tasks?status=running&limit=1").json()["tasks"]
        assert [row["task_id"] for row in rows] == ["old"]
    assert store_hot_facts.run_hot_facts(tmp_path)["state"] == "done"
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.row("old", "list_stable") == 0


def test_real_761a_cursor_roundtrip_does_not_double_count_addressed_prefix(tmp_path, monkeypatch):
    from pathlib import Path
    import runpy

    old_load = runpy.run_path(str(Path(__file__).with_name("fixtures") / "swarm_761a_load.py"))["load_swarm_rollup"]
    seed_result(tmp_path, "parent", {"status": "running"})
    path = tmp_path / "logs/events.jsonl"
    append_jsonl(path, {"type": "swarm_fanout", "parent_task_id": "parent",
                        "task_ids": ["old"], "requested_count": 1, "requested_model_lane": "legacy"})
    assert old_load(tmp_path, "parent")["fanout_count"] == 1
    addressed = event("swarm:z-first", "first", "one")
    swarm_rollup.record_swarm_fanout(tmp_path, addressed)
    append_jsonl(path, addressed)
    # The actual old loader saves a cursor/base that already includes this id;
    # it does not know the separate, still-retained row-backed field.
    assert old_load(tmp_path, "parent")["fanout_count"] == 2
    before = task_results.load_task_result(tmp_path, "parent")
    assert before["swarm_fanout_rollup"].get("version") != 2
    assert list(before["swarm_fanouts"]) == [addressed["event_id"]]
    caught_up = swarm_rollup.load_swarm_rollup(tmp_path, "parent")
    assert caught_up["fanout_count"] == caught_up["requested_count"] == 2
    assert caught_up["lanes_requested"] == ["legacy", "first"]
    monkeypatch.setattr(swarm_rollup, "JsonlChainSnapshot", lambda *_: pytest.fail("v2 read the prefix again"))
    assert swarm_rollup.load_swarm_rollup(tmp_path, "parent") == caught_up


@pytest.mark.parametrize("damage", ["shortened", "torn", "corrupt", "unreadable", "conflicting_id"])
def test_legacy_prefix_overlap_requires_readable_exact_evidence(tmp_path, monkeypatch, damage):
    from pathlib import Path
    import runpy

    old_load = runpy.run_path(str(Path(__file__).with_name("fixtures") / "swarm_761a_load.py"))["load_swarm_rollup"]
    seed_result(tmp_path, "parent", {"status": "running"})
    path = tmp_path / "logs/events.jsonl"
    addressed = event("overlap", "first", "one")
    swarm_rollup.record_swarm_fanout(tmp_path, addressed)
    append_jsonl(path, addressed)
    assert old_load(tmp_path, "parent")["fanout_count"] == 1
    before = task_results.load_task_result(tmp_path, "parent")["swarm_fanout_rollup"]
    if damage == "shortened":
        path.write_text("{}\n", encoding="utf-8")
    elif damage == "torn":
        text = path.read_text(encoding="utf-8")
        path.write_text(text[:-1] + " ", encoding="utf-8")  # Same cursor, no complete line.
    elif damage == "corrupt":
        text = path.read_text(encoding="utf-8")
        path.write_text("!" + text[1:], encoding="utf-8")  # Same complete-line length/cursor.
    elif damage == "unreadable":
        monkeypatch.setattr(swarm_rollup, "JsonlChainSnapshot", lambda *_: (_ for _ in ()).throw(OSError("denied")))
    else:
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace('"first"', '"other"'), encoding="utf-8")  # Same id, different facts.
    assert build_swarm_efficiency(SimpleNamespace(drive_root=tmp_path), {"id": "parent"}) is None
    assert task_results.load_task_result(tmp_path, "parent")["swarm_fanout_rollup"] == before
