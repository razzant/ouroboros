"""Persisted fan-out summaries retain legacy, rotated and interrupted writes."""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ouroboros import store, swarm_rollup, task_results, task_status, task_store
from ouroboros.finalization_timing import finalization_child_scope
from ouroboros.task_finalization import build_swarm_efficiency
from ouroboros.task_status import find_child_tasks
from ouroboros.tools.control_scheduling import _emit_swarm_fanout
from ouroboros.utils import append_jsonl
from tests._store_shared import seed_result


@pytest.fixture
def decode_events(monkeypatch):
    decode = Mock(wraps=json.loads)
    monkeypatch.setattr(swarm_rollup, "json", SimpleNamespace(loads=decode))
    return decode


def emit(root, child="child"):
    _emit_swarm_fanout(SimpleNamespace(drive_logs=lambda: root / "logs"), parent_task_id="parent",
                      root_task_id="parent", depth=1, task_group_id="group", task_ids=[child],
                      role="researcher", requested_model_lane="auto", objective="fixture", emitted_live=True)


def build(root):
    return build_swarm_efficiency(SimpleNamespace(drive_root=root), {"id": "parent"})


@pytest.mark.scale
def test_new_parent_skips_history_and_retains_failed_log_publication(tmp_path, monkeypatch, decode_events):
    from ouroboros.tools import control_scheduling

    for index in range(256):
        append_jsonl(tmp_path / "logs/events.jsonl", {"type": "unrelated", "index": index})
    task_results.write_task_result(tmp_path, "parent", "running", future_field={"kept": True})
    emit(tmp_path)
    assert swarm_rollup.load_swarm_rollup(tmp_path, "parent")["fanout_count"] == 1
    with monkeypatch.context() as broken:
        broken.setattr(control_scheduling, "append_jsonl", lambda *a, **kw: (_ for _ in ()).throw(OSError("fixture full")))
        emit(tmp_path, "second")
    decode_events.reset_mock()
    assert build(tmp_path)["subagent_count"] == 2
    assert decode_events.call_count == 0
    assert task_results.load_task_result(tmp_path, "parent")["future_field"] == {"kept": True}
    decode_events.reset_mock()
    assert build(tmp_path)["fanout_count"] == 2
    assert decode_events.call_count == 0


@pytest.mark.scale
def test_legacy_repairs_on_first_read_and_rotation_does_not_affect_producer_facts(tmp_path, decode_events):
    seed_result(tmp_path, "parent", {"_schema_version": 1, "task_id": "parent", "status": "running"})
    for index in range(64):
        append_jsonl(tmp_path / "logs/events.jsonl", {"type": "unrelated", "index": index})
    append_jsonl(tmp_path / "logs/events.jsonl", {"type": "swarm_fanout", "parent_task_id": "parent",
                                                "task_ids": ["first"], "inter_wave_latency_sec": 2.5})
    assert build(tmp_path)["fanout_interval_sec_total"] == 2.5
    assert decode_events.call_count == 65
    archive = tmp_path / "archive/events_20261010T000000.jsonl"
    archive.parent.mkdir()
    (tmp_path / "logs/events.jsonl").rename(archive)
    emit(tmp_path, "second")
    decode_events.reset_mock()
    assert build(tmp_path)["subagent_count"] == 2
    assert decode_events.call_count == 0


def test_incomplete_or_shortened_history_never_publishes_partial_counts(tmp_path):
    seed_result(tmp_path, "parent", {"status": "running"})
    path = tmp_path / "logs/events.jsonl"
    append_jsonl(path, {"type": "swarm_fanout", "parent_task_id": "parent", "task_ids": ["first"]})
    end = path.stat().st_size
    task_store.patch_document(tmp_path, "parent", lambda row: {**row, "swarm_fanout_rollup": {
        "source_root": str(tmp_path.resolve()), "through": end, "fanout_count": 1, "child_ids": ["first"]}})
    with path.open("ab") as stream:
        stream.write(b'{"type":"swarm_fanout","parent_task_id":"parent","task_ids":["next"]}')
    before = task_results.load_task_result(tmp_path, "parent")["swarm_fanout_rollup"]
    with pytest.raises(ValueError, match="incomplete"):
        swarm_rollup.repair_swarm_rollup(tmp_path, "parent")
    assert task_results.load_task_result(tmp_path, "parent")["swarm_fanout_rollup"] == before
    with path.open("ab") as stream:
        stream.write(b"\n")
    assert swarm_rollup.repair_swarm_rollup(tmp_path, "parent")["fanout_count"] == 2
    path.write_text("{}\n", encoding="utf-8")
    # Once repaired, ordinary reads no longer depend on rotated/retired diagnostics.
    assert swarm_rollup.load_swarm_rollup(tmp_path, "parent")["fanout_count"] == 2
    task_store.patch_document(tmp_path, "parent", lambda row: {**row, "swarm_fanout_rollup": before})
    with pytest.raises(ValueError, match="shortened"):
        swarm_rollup.repair_swarm_rollup(tmp_path, "parent")
    assert task_results.load_task_result(tmp_path, "parent")["swarm_fanout_rollup"] == before


def test_malformed_history_does_not_become_zero_swarm(tmp_path):
    seed_result(tmp_path, "parent", {"status": "running"})
    path = tmp_path / "logs/events.jsonl"
    path.parent.mkdir(exist_ok=True)
    path.write_text("broken\n", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        swarm_rollup.repair_swarm_rollup(tmp_path, "parent")
    assert build_swarm_efficiency(SimpleNamespace(drive_root=tmp_path), {
        "id": "parent", "metadata": {"subagents_mode": "swarm"}}) is None
    path.write_text("", encoding="utf-8")
    assert swarm_rollup.repair_swarm_rollup(tmp_path, "parent")["child_ids"] == []


def test_child_set_is_shared_within_pass_and_refreshed_afterwards(tmp_path, monkeypatch):
    for task_id, parent in (("child", "parent"), ("grandchild", "child")):
        task_results.write_task_result(tmp_path, task_id, "completed", delegation_role="subagent",
                                       parent_task_id=parent, root_task_id="parent")
    select = Mock(wraps=task_status.select_lineage)
    monkeypatch.setattr(task_status, "select_lineage", select)
    query = {"parent_task_id": "parent", "root_task_id": "parent", "materialize_artifacts": False}
    with finalization_child_scope():
        direct = find_child_tasks(tmp_path, **query, scope="direct")
        assert [row["task_id"] for row in direct] == ["child"]
        direct[0]["status"] = "mutated"
        with finalization_child_scope():
            subtree = find_child_tasks(tmp_path, **query, exclude_task_id="grandchild")
            assert len(subtree) == 1 and subtree[0]["status"] == "completed"
        assert select.call_count == 1
        task_results.write_task_result(tmp_path, "later", "completed", delegation_role="subagent",
                                       parent_task_id="parent", root_task_id="parent")
        assert len(find_child_tasks(tmp_path, **query)) == 2
    assert len(find_child_tasks(tmp_path, **query)) == 3
    assert select.call_count == 2


def test_child_cache_isolates_query_roots_modes_and_failed_passes(tmp_path, monkeypatch):
    from ouroboros import task_status

    calls = []
    def select(root, parent, root_id, *, direct_only, queue_members=()):
        calls.append(root)
        return []
    monkeypatch.setattr(task_status, "select_lineage", select)
    with pytest.raises(RuntimeError), finalization_child_scope():
        find_child_tasks(tmp_path, parent_task_id="parent")
        raise RuntimeError("fixture ended")
    with finalization_child_scope():
        for kwargs in ({}, {"parent_task_id": "other"}, {"materialize_artifacts": False}):
            find_child_tasks(tmp_path, **kwargs)
    assert len(calls) == 4


def test_repair_racing_emission_publishes_nothing_and_retry_counts_once(tmp_path, monkeypatch):
    seed_result(tmp_path, "parent", {"status": "running"})
    append_jsonl(tmp_path / "logs/events.jsonl", {"type": "swarm_fanout", "parent_task_id": "parent",
                                                "task_ids": ["first"], "requested_count": 1})
    patch = task_store.patch_document
    def race(*args, **kwargs):
        emit(tmp_path, "second")
        task_results.write_task_result(tmp_path, "parent", "completed")
        return patch(*args, **kwargs)
    with monkeypatch.context() as racing:
        racing.setattr(task_store, "patch_document", race)
        with pytest.raises(ValueError, match="changed during repair"):
            swarm_rollup.repair_swarm_rollup(tmp_path, "parent")
    row = task_results.load_task_result(tmp_path, "parent")
    assert row["status"] == "completed" and "swarm_fanout_rollup" not in row
    facts = swarm_rollup.repair_swarm_rollup(tmp_path, "parent")
    assert facts["fanout_count"] == facts["requested_count"] == 2
    assert facts["child_ids"] == ["first", "second"]


def test_unreadable_journal_does_not_block_admission_and_foreign_cursor_needs_repair(tmp_path, monkeypatch):
    with monkeypatch.context() as broken:
        broken.setattr(swarm_rollup, "JsonlChainSnapshot", lambda *a, **kw: (_ for _ in ()).throw(OSError("fixture denied")))
        row = task_results.write_task_result(tmp_path, "parent", "running")
    assert row["swarm_fanout_rollup"]["version"] == 2
    emit(tmp_path)
    row = task_results.load_task_result(tmp_path, "parent")
    row["swarm_fanout_rollup"] = {"source_root": "different-fixture-root", "through": 10_000_000}
    seed_result(tmp_path, "parent", row)
    assert swarm_rollup.load_swarm_rollup(tmp_path, "parent")["fanout_count"] == 1


@pytest.mark.scale
def test_emission_and_finalization_ignore_growing_foreign_suffix(tmp_path, decode_events, monkeypatch):
    task_results.write_task_result(tmp_path, "parent", "running")
    emit(tmp_path)
    measurements = []
    for count in (20, 200):
        for index in range(count):
            append_jsonl(tmp_path / "logs/events.jsonl", {"type": "unrelated", "task_id": "foreign", "i": index})
        decode_events.reset_mock()
        # Both real consumers, twice with an independently growing foreign suffix.
        before = build(tmp_path)
        assert before["fanout_count"] == (1 if count == 20 else 2)
        emit(tmp_path, str(count))
        after = build(tmp_path)
        assert after["fanout_count"] == before["fanout_count"] + 1
        assert after["subagent_count"] == before["subagent_count"] + 1
        assert decode_events.call_count == 0
        measurements.append({"foreign_events": count, "event_decodes": decode_events.call_count,
                             "fanouts": after["fanout_count"]})
    with monkeypatch.context() as blocked:
        blocked.setattr(swarm_rollup, "JsonlChainSnapshot", lambda *_: pytest.fail("normal consumer opened journal"))
        emit(tmp_path, "last")
        assert build(tmp_path)["fanout_count"] == 4
    print("SWARM_ADDRESSED_READS", json.dumps(measurements, sort_keys=True))


def test_concurrent_emissions_retry_and_stale_body_cannot_lose_facts(tmp_path):
    task_results.write_task_result(tmp_path, "parent", "running")
    stale = task_results.load_task_result(tmp_path, "parent")
    events = [{"type": "swarm_fanout", "event_id": f"swarm:{index}", "parent_task_id": "parent",
               "task_ids": [str(index), "shared"], "requested_count": 2,
               "fanout_interval_sec": 0.5, "requested_model_lane": "auto"} for index in range(20)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda event: swarm_rollup.record_swarm_fanout(tmp_path, event), events + events))
    # A frozen finalization/copy-back document cannot remove independently published telemetry.
    seed_result(tmp_path, "parent", {**stale, "status": "completed"})
    facts = swarm_rollup.load_swarm_rollup(tmp_path, "parent")
    assert facts["fanout_count"] == 20 and facts["requested_count"] == 40
    assert facts["fanout_interval_sec_total"] == 10 and facts["lanes_requested"] == ["auto"]
    assert facts["child_ids"] == sorted([str(index) for index in range(20)] + ["shared"])
    with pytest.raises(ValueError, match="different facts"):
        swarm_rollup.record_swarm_fanout(tmp_path, {**events[0], "requested_count": 9})


def test_failed_recording_rolls_back_and_is_best_effort_for_emitter(tmp_path, monkeypatch):
    task_results.write_task_result(tmp_path, "parent", "running")
    touch = store.Txn.touch
    def fail_after_event_insert(*_):
        raise OSError("fixture died before commit")
    with monkeypatch.context() as broken:
        broken.setattr(store.Txn, "touch", fail_after_event_insert)
        emit(tmp_path)  # telemetry failure does not become a launch failure
    assert swarm_rollup.load_swarm_rollup(tmp_path, "parent")["fanout_count"] == 0
    diagnostic = json.loads((tmp_path / "logs/events.jsonl").read_text())
    assert diagnostic["type"] == "swarm_fanout" and diagnostic["task_ids"] == ["child"]
    assert store.Txn.touch is touch
    emit(tmp_path)
    assert swarm_rollup.load_swarm_rollup(tmp_path, "parent")["fanout_count"] == 1


def test_absent_parent_and_unreadable_store_never_prove_zero(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="history is absent"):
        swarm_rollup.load_swarm_rollup(tmp_path, "absent")
    assert not task_store.exists(tmp_path, "absent")
    task_results.write_task_result(tmp_path, "parent", "running")
    def unreadable(*_a, **_kw):
        raise store.StoreError("fixture unreadable")
    monkeypatch.setattr(task_results, "load_task_result", unreadable)
    assert build(tmp_path) is None


@pytest.mark.parametrize("boundary", ["before_commit", "after_commit"])
@pytest.mark.serial
def test_process_death_at_publication_boundary_keeps_exact_committed_facts(tmp_path, boundary):
    task_results.write_task_result(tmp_path, "parent", "running")
    event = {"type": "swarm_fanout", "event_id": "swarm:crash", "parent_task_id": "parent",
             "task_ids": ["child"], "requested_count": 1, "fanout_interval_sec": 1.5,
             "requested_model_lane": "fixture"}
    script = """
import json, os, sys
from ouroboros import store, swarm_rollup
if sys.argv[2] == 'before_commit':
    store.Txn.touch = lambda *a: os._exit(73)
swarm_rollup.record_swarm_fanout(sys.argv[1], json.loads(sys.argv[3]))
os._exit(74)  # committed source, before any diagnostic append
"""
    crashed = subprocess.run([sys.executable, "-c", script, str(tmp_path), boundary, json.dumps(event)],
                             capture_output=True, text=True, timeout=30)
    assert crashed.returncode == (73 if boundary == "before_commit" else 74), crashed.stderr
    assert swarm_rollup.load_swarm_rollup(tmp_path, "parent")["fanout_count"] == (boundary == "after_commit")
    # Retry the original retained event, after SQLite rolled back/joined the dead writer.
    swarm_rollup.record_swarm_fanout(tmp_path, event)
    facts = swarm_rollup.load_swarm_rollup(tmp_path, "parent")
    assert facts["fanout_count"] == facts["requested_count"] == 1
    assert facts["fanout_interval_sec_total"] == 1.5 and facts["lanes_requested"] == ["fixture"]
    assert not (tmp_path / "logs/events.jsonl").exists()


def test_copyback_old_writer_export_import_preserve_addressed_emissions(tmp_path, monkeypatch):
    from ouroboros import store_import
    from ouroboros.post_task_checkpoint import project_replica_task_result_fields

    root, child, imported = tmp_path / "root", tmp_path / "child", tmp_path / "imported"
    for drive in (root, child):
        task_results.write_task_result(drive, "parent", "running")
    emit(root, "canonical-child")
    emit(child, "execution-child")
    replica = task_results.load_task_result(child, "parent")
    task_results.write_task_result(root, "parent", "completed", _field_projector=lambda current, _: {
        **project_replica_task_result_fields(current, replica), "status": "completed"})
    expected = swarm_rollup.load_swarm_rollup(root, "parent")
    assert expected["fanout_count"] == 2
    assert expected["child_ids"] == ["canonical-child", "execution-child"]
    # A pre-feature writer projects unknown row kinds but does not own/deletes them.
    with monkeypatch.context() as old_writer:
        old_writer.setattr(store, "ROW_BACKED_FIELDS", tuple(key for key in store.ROW_BACKED_FIELDS
                                                           if key != "swarm_fanouts"))
        task_results.write_task_result(root, "parent", "completed", note="older writer round-trip")
    assert swarm_rollup.load_swarm_rollup(root, "parent") == expected
    with store.open_store(root, write=True, service=True) as txn:
        txn.set_meta("phase", store.EXPORTING)
    store_import.export_facts(root)
    exported = (root / "task_results/parent.json").read_text(encoding="utf-8")
    assert len(json.loads(exported)["swarm_fanouts"]) == 2
    (imported / "task_results").mkdir(parents=True)
    (imported / "task_results/parent.json").write_text(exported, encoding="utf-8")
    store_import.ensure_current_store(imported)
    assert swarm_rollup.load_swarm_rollup(imported, "parent") == expected


def test_stale_copyback_cannot_replace_repaired_legacy_base(tmp_path):
    from ouroboros.post_task_checkpoint import project_replica_task_result_fields

    seed_result(tmp_path, "parent", {"status": "running"})
    append_jsonl(tmp_path / "logs/events.jsonl", {"type": "swarm_fanout", "parent_task_id": "parent",
        "task_ids": ["legacy-child"], "requested_count": 1, "requested_model_lane": "legacy"})
    swarm_rollup.repair_swarm_rollup(tmp_path, "parent")
    stale = {"status": "completed", "swarm_fanout_rollup": swarm_rollup.initial_swarm_rollup(tmp_path)}
    task_results.write_task_result(tmp_path, "parent", "completed", _field_projector=lambda current, _: {
        **project_replica_task_result_fields(current, stale), "status": "completed"})
    emit(tmp_path, "new-child")
    facts = swarm_rollup.load_swarm_rollup(tmp_path, "parent")
    assert facts["fanout_count"] == facts["requested_count"] == 2
    assert facts["child_ids"] == ["legacy-child", "new-child"]
    assert facts["lanes_requested"] == ["legacy", "auto"]


def test_real_producer_preserves_requested_counts_intervals_children_and_lane_order(tmp_path, monkeypatch):
    from ouroboros.tools import control_scheduling

    task_results.write_task_result(tmp_path, "parent", "running")
    clock = iter((100.0, 101.25, 103.0))
    monkeypatch.setattr(control_scheduling, "time", SimpleNamespace(time=lambda: next(clock)))
    ctx = SimpleNamespace(drive_logs=lambda: tmp_path / "logs")
    for lane, children in (("first", ["a", "b"]), ("second", ["b", "c"]), ("first", ["a"])):
        _emit_swarm_fanout(ctx, parent_task_id="parent", root_task_id="parent", depth=1,
                          task_group_id="group", task_ids=children, role="researcher",
                          requested_model_lane=lane, objective="fixture", emitted_live=True)
    assert swarm_rollup.load_swarm_rollup(tmp_path, "parent") == {
        "version": 2, "fanout_count": 3, "requested_count": 5, "fanout_interval_sec_total": 3.0,
        "child_ids": ["a", "b", "c"], "lanes_requested": ["first", "second"]}
    result = build_swarm_efficiency(SimpleNamespace(drive_root=tmp_path), {
        "id": "parent", "metadata": {"subagents_mode": "swarm"}})
    assert {key: result[key] for key in ("subagent_count", "fanout_count", "requested_count",
                                        "fanout_interval_sec_total", "lanes_requested", "intent_source")} == {
        "subagent_count": 3, "fanout_count": 3, "requested_count": 5, "fanout_interval_sec_total": 3.0,
        "lanes_requested": ["first", "second"], "intent_source": "swarm"}


def test_legacy_missing_journal_is_unavailable_even_with_child_rows(tmp_path):
    seed_result(tmp_path, "parent", {"status": "running"})
    task_results.write_task_result(tmp_path, "child", "completed", parent_task_id="parent", root_task_id="parent")
    assert build(tmp_path) is None
    with pytest.raises(ValueError, match="history is absent"):
        swarm_rollup.repair_swarm_rollup(tmp_path, "parent")
