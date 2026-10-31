"""Real old producer/writer transitions and public finalization/copy-back consumers."""

import json
from types import SimpleNamespace

import pytest

from ouroboros import store_import, swarm_rollup, task_results, task_store
from ouroboros.headless import copy_child_task_result
from ouroboros.swarm_receipt_sources import tree_inputs
from ouroboros.task_finalization import build_swarm_efficiency
from ouroboros.tools import control_scheduling
from ouroboros.utils import append_jsonl
from tests._store_shared import seed_result
from tests.fixtures import swarm_761a_load, swarm_761a_writer
from tests.test_store_import import reconcile_legacy


def emit(root, child, *, old=False, monkeypatch):
    producer = swarm_761a_writer if old else control_scheduling
    with monkeypatch.context() as patch:
        if old:
            patch.setattr(swarm_rollup, "load_swarm_rollup", swarm_761a_load.load_swarm_rollup)
        patch.setattr(producer, "time", SimpleNamespace(time=lambda: 101.5))
        producer._emit_swarm_fanout(
            SimpleNamespace(drive_logs=lambda: root / "logs", _last_fanout_ts=100.0),
            parent_task_id="parent", root_task_id="parent", depth=1, task_group_id="",
            task_ids=[child], role="researcher", requested_model_lane=child,
            objective="portable legacy transition", emitted_live=True)


def legacy(root, child="legacy"):
    seed_result(root, "parent", {"status": "running"})
    append_jsonl(root / "logs/events.jsonl", {
        "type": "swarm_fanout", "parent_task_id": "parent", "task_ids": [child],
        "requested_count": 1, "fanout_interval_sec": 0.5, "requested_model_lane": "legacy"})


def row(root):
    return task_results.load_task_result(root, "parent", strict=True)


@pytest.mark.parametrize("addressed_before_repair", [False, True])
@pytest.mark.parametrize("duplicate_diagnostic", [False, True])
def test_frozen_producer_and_writer_after_repair_count_known_emissions_once(
        tmp_path, monkeypatch, addressed_before_repair, duplicate_diagnostic):
    legacy(tmp_path)
    if addressed_before_repair:
        emit(tmp_path, "addressed", monkeypatch=monkeypatch)
    swarm_rollup.load_swarm_rollup(tmp_path, "parent")
    witness = row(tmp_path)["swarm_fanout_basis"]
    if not addressed_before_repair:
        emit(tmp_path, "addressed", monkeypatch=monkeypatch)
    if duplicate_diagnostic:
        append_jsonl(tmp_path / "logs/events.jsonl", next(iter(row(tmp_path)["swarm_fanouts"].values())))
    emit(tmp_path, "old-runtime", old=True, monkeypatch=monkeypatch)
    stored = row(tmp_path)
    # The frozen writer really advanced the stored legacy base, preserving the
    # unknown v2 tag/witness. A current-writer guard must not hide this trigger.
    expected_old = 2 + int(not addressed_before_repair) + int(duplicate_diagnostic)
    assert stored["swarm_fanout_rollup"]["version"] == 2
    assert stored["swarm_fanout_rollup"]["fanout_count"] == expected_old
    assert stored["swarm_fanout_basis"] == witness
    facts = swarm_rollup.load_swarm_rollup(tmp_path, "parent")
    assert facts["fanout_count"] == facts["requested_count"] == 3
    assert facts["fanout_interval_sec_total"] == 3.5
    assert facts["child_ids"] == ["addressed", "legacy", "old-runtime"]
    monkeypatch.setattr(swarm_rollup, "JsonlChainSnapshot", lambda *_: pytest.fail("repaired read opened history"))
    assert swarm_rollup.load_swarm_rollup(tmp_path, "parent") == facts


@pytest.mark.parametrize("damage", ["absent", "shortened", "torn", "unreadable"])
def test_invalidated_witness_with_lost_history_is_unknown(tmp_path, monkeypatch, damage):
    legacy(tmp_path)
    swarm_rollup.load_swarm_rollup(tmp_path, "parent")
    emit(tmp_path, "addressed", monkeypatch=monkeypatch)
    emit(tmp_path, "old-runtime", old=True, monkeypatch=monkeypatch)
    before = row(tmp_path)
    path = tmp_path / "logs/events.jsonl"
    if damage == "absent":
        path.unlink()
    elif damage == "shortened":
        path.write_text("{}\n", encoding="utf-8")
    elif damage == "torn":
        with path.open("ab") as handle:
            handle.write(b'{"type":')
    else:
        monkeypatch.setattr(swarm_rollup, "JsonlChainSnapshot", lambda *_: (_ for _ in ()).throw(OSError("denied")))
    assert build_swarm_efficiency(SimpleNamespace(drive_root=tmp_path), {
        "id": "parent", "metadata": {"subagents_mode": "swarm"}}) is None
    coverage = tree_inputs(tmp_path, "parent")["coverage"]
    assert coverage["tree"] == "unknown" and "fanout_history_unavailable" in coverage["gaps"]
    after = row(tmp_path)
    receipt = after.pop("swarm_receipt")
    assert receipt["state"] == "provisional" and "fanout_history_unavailable" in receipt["coverage"]["gaps"]
    assert after == before  # no changed base, witness, emission or lifecycle fact


@pytest.mark.parametrize("observed_at", ["replica", "canonical", "addressed"])
def test_finalization_then_public_copyback_preserves_expected_children(tmp_path, monkeypatch, observed_at):
    canonical, replica = tmp_path / "canonical", tmp_path / "replica"
    for root in (canonical, replica):
        seed_result(root, "parent", {"status": "running"})
        append_jsonl(root / "logs/events.jsonl", {"type": "unrelated"})
    observed = canonical if observed_at == "canonical" else replica
    if observed_at == "addressed":
        emit(observed, "missing-child", monkeypatch=monkeypatch)
    else:
        append_jsonl(observed / "logs/events.jsonl", {"type": "swarm_fanout",
            "parent_task_id": "parent", "task_ids": ["missing-child"], "requested_count": 1})
    task = {"id": "parent", "root_task_id": "parent", "child_drive_root": str(replica),
            "budget_drive_root": str(canonical), "metadata": {"subagents_mode": "swarm"}}
    summary = build_swarm_efficiency(SimpleNamespace(drive_root=replica), task)
    assert row(canonical)["swarm_fanout_rollup"]["version"] == 2
    task_results.write_task_result(replica, "parent", "completed", result="child answer", swarm_efficiency=summary)
    for _ in range(2):
        assert copy_child_task_result(canonical, task)["result"] == "child answer"
        after = build_swarm_efficiency(SimpleNamespace(drive_root=canonical), task)
        assert after["fanout_count"] == after["requested_count"] == 1
        coverage = tree_inputs(canonical, "parent")["coverage"]
        assert coverage["tree"] == "unknown" and "missing_child_results" in coverage["gaps"]
        assert coverage["causal_edges"] == coverage["history_before_instrumentation"] == "unknown"


def export(root):
    store_import.export_facts(root)
    return json.loads((root / "task_results/parent.json").read_text(encoding="utf-8"))


def test_import_reconcile_and_stale_empty_copy_keep_legacy_source(tmp_path):
    origin, target = tmp_path / "origin", tmp_path / "target"
    legacy(origin, "missing-child")
    swarm_rollup.load_swarm_rollup(origin, "parent")
    transferred = export(origin)
    (target / "task_results").mkdir(parents=True)
    path = target / "task_results/parent.json"
    path.write_text(json.dumps(transferred), encoding="utf-8")
    store_import.ensure_current_store(target)
    assert swarm_rollup.load_swarm_rollup(target, "parent")["fanout_count"] == 1
    assert row(target)["swarm_fanout_basis"] == transferred["swarm_fanout_basis"]
    stale = {**transferred, "swarm_fanout_rollup": swarm_rollup.initial_swarm_rollup(target)}
    stale.pop("swarm_fanout_basis")
    export(target)
    path.write_text(json.dumps(stale), encoding="utf-8")
    assert reconcile_legacy(target)["changed"] == 1
    assert swarm_rollup.load_swarm_rollup(target, "parent")["fanout_count"] == 1
    assert "missing_child_results" in tree_inputs(target, "parent")["coverage"]["gaps"]


def test_independent_nonempty_sources_keep_evidence_and_unknown_through_reconcile(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    for root, child in ((left, "one"), (right, "two")):
        legacy(root, child)
        swarm_rollup.load_swarm_rollup(root, "parent")
    incoming = export(right)
    before = row(left)["swarm_fanout_rollup"]
    export(left)
    (left / "task_results/parent.json").write_text(json.dumps(incoming), encoding="utf-8")
    assert reconcile_legacy(left)["changed"] == 1
    states = row(left)["swarm_fanout_basis"]["conflicts"]
    assert {state["base"]["source_root"] for state in states} == {str(left), str(right)}
    assert any(state["base"] == before for state in states)
    assert build_swarm_efficiency(SimpleNamespace(drive_root=left), {"id": "parent"}) is None
    assert "fanout_history_unavailable" in tree_inputs(left, "parent")["coverage"]["gaps"]
    task_store.patch_document(left, "parent", lambda _: incoming)
    assert len(row(left)["swarm_fanout_basis"]["conflicts"]) == 2


def test_downgrade_at_imported_root_cannot_replace_foreign_legacy_history_with_local_zero(tmp_path, monkeypatch):
    origin, target = tmp_path / "origin", tmp_path / "target"
    legacy(origin, "missing-child")
    swarm_rollup.load_swarm_rollup(origin, "parent")
    task_store.patch_document(target, "parent", lambda _: row(origin))
    emit(target, "local-child", old=True, monkeypatch=monkeypatch)
    assert row(target)["swarm_fanout_rollup"]["source_root"] == str(target)
    assert row(target)["swarm_fanout_basis"]["base"]["source_root"] == str(origin)
    assert build_swarm_efficiency(SimpleNamespace(drive_root=target), {"id": "parent"}) is None
    assert "fanout_history_unavailable" in tree_inputs(target, "parent")["coverage"]["gaps"]


def test_unconverted_foreign_legacy_copyback_cannot_inherit_canonical_empty_witness(tmp_path):
    origin, target = tmp_path / "origin", tmp_path / "target"
    legacy(origin, "missing-child")
    swarm_761a_load.load_swarm_rollup(origin, "parent")
    task_results.write_task_result(target, "parent", "running")
    append_jsonl(target / "logs/events.jsonl", {"type": "unrelated"})
    task_results.write_task_result(origin, "parent", "completed", result="legacy answer")
    assert copy_child_task_result(target, {"id": "parent", "child_drive_root": str(origin)})["result"] == "legacy answer"
    assert build_swarm_efficiency(SimpleNamespace(drive_root=target), {"id": "parent"}) is None
    assert "fanout_history_unavailable" in tree_inputs(target, "parent")["coverage"]["gaps"]


def test_unwitnessed_v2_tag_is_not_inclusion_evidence(tmp_path):
    legacy(tmp_path)
    old = swarm_761a_load.load_swarm_rollup(tmp_path, "parent")
    task_store.patch_document(tmp_path, "parent", lambda current: {
        **current, "swarm_fanout_rollup": {**old, "version": 2}})
    assert build_swarm_efficiency(SimpleNamespace(drive_root=tmp_path), {"id": "parent"}) is None
    assert "fanout_history_unavailable" in tree_inputs(tmp_path, "parent")["coverage"]["gaps"]


def test_unwitnessed_old_prefix_with_duplicate_addressed_logs_cannot_claim_exact_count(tmp_path, monkeypatch):
    legacy(tmp_path)
    emit(tmp_path, "addressed", monkeypatch=monkeypatch)
    append_jsonl(tmp_path / "logs/events.jsonl", next(iter(row(tmp_path)["swarm_fanouts"].values())))
    emit(tmp_path, "old-runtime", old=True, monkeypatch=monkeypatch)
    assert row(tmp_path)["swarm_fanout_rollup"]["fanout_count"] == 4
    assert build_swarm_efficiency(SimpleNamespace(drive_root=tmp_path), {"id": "parent"}) is None
    assert "fanout_history_unavailable" in tree_inputs(tmp_path, "parent")["coverage"]["gaps"]


def test_unreadable_legacy_basis_never_blocks_lifecycle_or_claims_zero(tmp_path):
    seed_result(tmp_path, "parent", {"status": "running", "swarm_fanout_rollup": "unreadable"})
    append_jsonl(tmp_path / "logs/events.jsonl", {"type": "unrelated"})
    result = task_results.write_task_result(tmp_path, "parent", "completed", result="answer",
                                           swarm_fanout_rollup=swarm_rollup.initial_swarm_rollup(tmp_path))
    assert result["status"] == "completed" and result["result"] == "answer"
    assert build_swarm_efficiency(SimpleNamespace(drive_root=tmp_path), {"id": "parent"}) is None
    assert "fanout_history_unavailable" in tree_inputs(tmp_path, "parent")["coverage"]["gaps"]
