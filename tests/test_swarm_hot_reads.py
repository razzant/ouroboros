"""Writer work stays addressed; compatibility observations do not invent task rows."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ouroboros import store, swarm_rollup, task_store
from ouroboros.swarm_receipt import publish
from ouroboros.swarm_receipt_sources import tree_inputs
from ouroboros.task_finalization import build_swarm_efficiency
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.utils import append_jsonl
from tests._store_shared import seed_result


@pytest.mark.parametrize("damaged", [False, True])
def test_child_receipt_writes_leave_legacy_catchup_to_an_explicit_read(tmp_path, monkeypatch, damaged):
    log = tmp_path / "logs/events.jsonl"
    append_jsonl(log, {"type": "swarm_fanout", "parent_task_id": "root", "task_ids": ["child"],
                       "requested_count": 1})
    legacy = {"source_root": str(tmp_path.resolve()), "through": log.stat().st_size,
              "fanout_count": 1, "requested_count": 1, "child_ids": ["child"], "lanes_requested": []}
    seed_result(tmp_path, "root", {"status": "completed", "swarm_fanout_rollup": legacy,
                                  "swarm_receipt": {"revision": 1}})
    seed_result(tmp_path, "child", {"status": "completed", "parent_task_id": "root", "root_task_id": "root"})
    with log.open("a") as handle:
        handle.write((json.dumps({"type": "foreign", "pad": "y" * 400}) + "\n") * 2000)
        if damaged:
            handle.write('{"type": "torn"\n')
    append_jsonl(log, {"type": "swarm_fanout", "parent_task_id": "root", "task_ids": ["missing-child"],
                       "requested_count": 1})
    decode = Mock(wraps=json.loads)
    snapshot = Mock(wraps=swarm_rollup.JsonlChainSnapshot)
    monkeypatch.setattr(swarm_rollup, "json", SimpleNamespace(loads=decode))
    monkeypatch.setattr(swarm_rollup, "JsonlChainSnapshot", snapshot)
    for attempt in range(3):
        write_task_result(tmp_path, "child", "completed", late_note=str(attempt))
        assert decode.call_count == snapshot.call_count == 0
        root = load_task_result(tmp_path, "root", strict=True)
        assert root["swarm_fanout_rollup"] == legacy
        receipt = root["swarm_receipt"]
        assert receipt["state"] == "provisional"
        assert receipt["coverage"]["tree"] == "unknown"
        assert "fanout_history_unavailable" in receipt["coverage"]["gaps"]
    if damaged:
        with pytest.raises(ValueError):
            swarm_rollup.load_swarm_rollup(tmp_path, "root")
        assert load_task_result(tmp_path, "root")["swarm_fanout_rollup"] == legacy
    else:
        facts = swarm_rollup.load_swarm_rollup(tmp_path, "root")
        assert facts["fanout_count"] == facts["requested_count"] == 2
        assert facts["child_ids"] == ["child", "missing-child"]
    assert decode.call_count == 2001 and snapshot.call_count == 1
    decode.reset_mock()
    snapshot.reset_mock()
    receipt = publish(tmp_path, "root")
    assert decode.call_count == snapshot.call_count == 0
    expected_gap = "fanout_history_unavailable" if damaged else "missing_child_results"
    assert expected_gap in receipt["coverage"]["gaps"]


@pytest.mark.parametrize("history", ["event", "empty", "absent", "torn", "unreadable"])
def test_missing_parent_observes_available_history_without_creating_result(tmp_path, monkeypatch, history):
    log = tmp_path / "logs/events.jsonl"
    if history != "absent":
        log.parent.mkdir()
        log.write_text(('{"type":"swarm_fanout","parent_task_id":"root","task_ids":["child"]}\n'
                        if history in {"event", "unreadable"} else '{"type":' if history == "torn" else ""))
    if history == "unreadable":
        monkeypatch.setattr(swarm_rollup, "JsonlChainSnapshot", Mock(side_effect=OSError("denied")))
    result = build_swarm_efficiency(SimpleNamespace(drive_root=tmp_path), {
        "id": "root", "metadata": {"subagents_mode": "swarm"}})
    if history == "event":
        assert result["subagent_count"] == result["fanout_count"] == 1
    elif history == "empty":
        assert result == {"intent_source": "swarm", "planned": None, "observed_started": 0,
                          "status": "no_fanout_observed"}
    else:
        assert result is None
    assert load_task_result(tmp_path, "root", strict=True) is None
    coverage = tree_inputs(tmp_path, "root")["coverage"]
    assert coverage["tree"] == "unknown"
    assert {"root_result_absent", "fanout_history_unavailable"} <= set(coverage["gaps"])


@pytest.mark.parametrize("admitted", [False, True])
def test_ordinary_update_reuses_the_current_body_for_basis_selection(tmp_path, monkeypatch, admitted):
    if admitted:
        write_task_result(tmp_path, "root", "running", result="body:" + "x" * 100_000)
    else:
        seed_result(tmp_path, "root", {"status": "running", "result": "body:" + "x" * 100_000})
    queries = []
    execute = store._StoreConnection.execute
    decode = Mock(wraps=swarm_rollup.loads)

    def query(conn, sql, *args, **kwargs):
        if sql.startswith("SELECT body FROM task_bodies WHERE task_id="):
            queries.append(sql)
        return execute(conn, sql, *args, **kwargs)

    monkeypatch.setattr(store._StoreConnection, "execute", query)
    monkeypatch.setattr(swarm_rollup, "loads", decode)
    task_store.patch_document(tmp_path, "root", lambda current: {**current, "late_note": "kept"})
    assert len(queries) == 1
    assert decode.call_count == 0
    assert load_task_result(tmp_path, "root")["late_note"] == "kept"


def test_direct_store_selects_its_current_basis_again_after_each_write(tmp_path):
    """Direct upserts may carry a stale or absent base; they have no held current document."""
    write_task_result(tmp_path, "root", "running")
    stale = load_task_result(tmp_path, "root")
    base = {**stale["swarm_fanout_rollup"], "fanout_count": 1, "requested_count": 1, "child_ids": ["child"]}
    with store.open_store(tmp_path, write=True) as txn:
        txn.store_task({**stale, "swarm_fanout_rollup": base, "swarm_fanout_basis": {"base": base}})
        txn.store_task(dict(stale))
        txn.store_task({"task_id": "root", "status": "running"})
        assert txn.load("root")["swarm_fanout_rollup"] == base
    assert swarm_rollup.load_swarm_rollup(tmp_path, "root")["fanout_count"] == 1


@pytest.mark.parametrize("failed", ["record", "log"])
def test_fanout_record_and_diagnostic_fail_independently(tmp_path, monkeypatch, failed):
    from ouroboros.tools import control_scheduling

    write_task_result(tmp_path, "root", "running")
    if failed == "record":
        monkeypatch.setattr(swarm_rollup, "record_swarm_fanout", Mock(side_effect=OSError("record refused")))
    else:
        monkeypatch.setattr(control_scheduling, "append_jsonl", Mock(side_effect=OSError("log refused")))
    control_scheduling._emit_swarm_fanout(
        SimpleNamespace(drive_logs=lambda: tmp_path / "logs"), parent_task_id="root", root_task_id="root",
        depth=1, task_group_id="", task_ids=["child"], role="researcher", requested_model_lane="auto",
        objective="Independent emission channels", emitted_live=True)
    if failed == "record":
        event = json.loads((tmp_path / "logs/events.jsonl").read_text())
        assert event["task_ids"] == ["child"] and event["event_id"].startswith("swarm:")
    else:
        facts = swarm_rollup.load_swarm_rollup(tmp_path, "root")
        assert facts["fanout_count"] == 1 and facts["child_ids"] == ["child"]


def test_unavailable_fanout_still_publishes_other_known_receipt_facts(tmp_path):
    seed_result(tmp_path, "root", {"status": "completed", "result": "retained answer"})
    seed_result(tmp_path, "child", {"status": "completed", "root_task_id": "root", "parent_task_id": "root"})
    assert build_swarm_efficiency(SimpleNamespace(drive_root=tmp_path), {"id": "root"}) is None
    row = load_task_result(tmp_path, "root")
    assert row["result"] == "retained answer" and "swarm_fanout_rollup" not in row
    receipt = row["swarm_receipt"]
    assert receipt["state"] == "provisional"
    assert "fanout_history_unavailable" in receipt["coverage"]["gaps"]
    assert {actor["task_id"] for actor in receipt["actor_intervals"]} == {"root", "child"}
    assert not (tmp_path / "logs/events.jsonl").exists()
    assert not (tmp_path / "state/usage.sqlite").exists()
    assert build_swarm_efficiency(SimpleNamespace(drive_root=tmp_path), {"id": "root"}) is None


def test_public_list_and_detail_keep_summary_but_not_swarm_storage_evidence(tmp_path):
    from ouroboros.gateway import tasks

    unknown = {"swarm_fanouts": "owner's nested field", "swarm_fanout_basis": ["also owner data"]}
    write_task_result(tmp_path, "root", "completed", result="answer", owner_extension=unknown,
                      swarm_efficiency={"subagent_count": 1})
    swarm_rollup.record_swarm_fanout(tmp_path, {"event_id": "swarm:one", "type": "swarm_fanout",
        "parent_task_id": "root", "task_ids": ["child"], "requested_count": 1})
    publish(tmp_path, "root")
    stored = load_task_result(tmp_path, "root")
    listed = tasks._tasks_list_payload(tmp_path, set(), 50, False)["tasks"][0]
    request = SimpleNamespace(path_params={"task_id": "root"},
                              app=SimpleNamespace(state=SimpleNamespace(drive_root=tmp_path)))
    detail = json.loads(tasks._task_get_response(request).body)
    for public in (listed, detail):
        assert "swarm_fanouts" not in public and "swarm_fanout_basis" not in public
        assert public["owner_extension"] == unknown
        assert public["swarm_efficiency"] == stored["swarm_efficiency"]
        assert public["swarm_receipt"] == stored["swarm_receipt"]
        assert public["result"] == "answer"
    assert load_task_result(tmp_path, "root") == stored
    assert swarm_rollup.load_swarm_rollup(tmp_path, "root")["fanout_count"] == 1


@pytest.mark.parametrize("children", [42, ["known-child", {"damaged": "child"}]])
def test_unreadable_legacy_children_keep_receipt_unknown(tmp_path, children):
    seed_result(tmp_path, "root", {"status": "completed", "swarm_fanout_rollup": {"child_ids": children}})
    receipt = publish(tmp_path, "root")
    assert receipt["state"] == "provisional"
    assert "fanout_history_unavailable" in receipt["coverage"]["gaps"]
    if isinstance(children, list):
        assert "missing_child_results" in receipt["coverage"]["gaps"]
