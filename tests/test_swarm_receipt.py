"""Tree measurements never infer a critical path from overlapping ancestry."""
from types import SimpleNamespace

import pytest

from ouroboros import swarm_receipt as receipt
from ouroboros.task_results import load_task_result, write_task_result


def actor(name, start, end, **fields):
    return {"id": name, "start": start, "end": end, "terminal": end is not None, **fields}


def test_parallel_and_serial_intervals_measure_overlap_without_policy():
    parallel = receipt.fold_tree("root", [actor("a", 0, 10), actor("b", 0, 10)])
    serial = receipt.fold_tree("root", [actor("a", 0, 10), actor("b", 10, 20)])
    assert parallel["parallelism"]["mean_observed_parallelism"] == 2
    assert parallel["parallelism"]["overlap_sec"] == 10
    assert serial["parallelism"]["mean_observed_parallelism"] == 1
    assert serial["parallelism"]["overlap_sec"] == 0
    assert parallel["longest_causal_path"]["seconds"] is None
    assert serial["longest_causal_path"]["seconds"] is None  # serialization is not a causal edge


def test_explicit_nonoverlapping_edges_and_missing_or_ancestry_edges():
    actors = [actor("a", 0, 5), actor("b", 5, 12)]
    observed = [{"from": "a", "to": "b", "observed": True}]
    complete = {"tree": "complete", "causal_edges": "complete"}
    assert receipt.fold_tree("root", actors, edges=observed, coverage=complete)["longest_causal_path"]["seconds"] == 12
    assert receipt.fold_tree("root", actors, edges=observed)["longest_causal_path"]["seconds"] is None
    nested = [actor("a", 0, 12), actor("b", 3, 9, parent_task_id="a")]
    path = receipt.fold_tree("root", nested, edges=observed, coverage=complete)["longest_causal_path"]
    assert path["seconds"] is None and path["observed_seconds"] == 12
    assert len(path["rejected_edges"]) == 1


def test_live_child_root_end_and_unknown_interval_stay_provisional():
    coverage = {"tree": "complete"}
    actors = [actor("root", 0, 10), actor("child", 3, None)]
    provisional = receipt.fold_tree("root", actors, coverage=coverage)
    assert provisional["state"] == "provisional"
    assert provisional["coverage"]["intervals_observed"] == 1
    actors[1] = actor("child", 3, 15)
    assert receipt.fold_tree("root", actors, coverage=coverage)["state"] == "closed"
    assert receipt.fold_tree("root", actors)["state"] == "provisional"


@pytest.mark.parametrize("resolution", ["assigned", "refused"])
def test_unresolved_start_keeps_receipt_provisional_until_custody_resolves(tmp_path, resolution):
    from ouroboros import delegate_custody as custody

    write_task_result(tmp_path, "root", "completed")
    assert custody.record_start_requested(tmp_path, task_id="root", root_task_id="root",
        invocation_id="pending-one", request={"prompt": "already submitted"})
    pending = receipt.publish(tmp_path, "root")
    actor = next(a for a in pending["actor_intervals"] if a["kind"] == "delegated_start")
    assert pending["state"] == "provisional" and actor["invocation_id"] == "pending-one"
    assert actor["terminal"] is False and actor["state"] == "pending"
    assert actor["start"] is actor["end"] is actor["accounted_upper_bound_usd"] is None
    if resolution == "refused":
        assert custody.emit(tmp_path, custody.START_FAILED,
            {"task_id": "root", "invocation_id": "pending-one", "definite": True})
    else:
        run = custody.RunCustody(run_id="run-one", task_id="root", root_task_id="root", invocation_id="pending-one")
        assert custody.record_started(tmp_path, run)
        assigned = receipt.publish(tmp_path, "root")
        assert assigned["state"] == "provisional"
        assert {a["id"] for a in assigned["actor_intervals"]} == {"task:root", "run:run-one"}
        assert custody.emit(tmp_path, custody.TERMINAL_OBSERVED, {"run_id": run.run_id, "task_id": "root",
            "state": "succeeded", "observation_seq": 1, "source_ref": {"sha256": "a" * 64}})
        assert receipt.publish(tmp_path, "root")["state"] == "provisional"
        assert custody.emit(tmp_path, custody.SETTLED, {"run_id": run.run_id, "task_id": "root", "state": "succeeded"})
    closed = receipt.publish(tmp_path, "root")
    assert closed["state"] == "closed" and closed["revision"] > pending["revision"]
    assert not any(a["kind"] == "delegated_start" for a in closed["actor_intervals"])


def test_sibling_lock_waits_until_the_new_revision_is_published(tmp_path):
    import hashlib
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
    from ouroboros.task_store import patch_document

    write_task_result(tmp_path, "root", "completed")
    before = receipt.publish(tmp_path, "root")
    lock_path = tmp_path / "state" / ("swarm_receipt." + hashlib.sha256(b"root").hexdigest() + ".lock")
    lock = acquire_exclusive_file_lock(lock_path, owner_aware_stale=True)
    assert lock is not None
    patch_document(tmp_path, "root", lambda row: {**row, "swarm_comparison": "A late observation."})
    entered, done = threading.Event(), threading.Event()

    def sibling():
        entered.set()
        try:
            return receipt.publish(tmp_path, "root")
        finally:
            done.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(sibling)
        try:
            assert entered.wait(10)
            assert not done.wait(2.1), "contending publication must survive the old acquisition window"
        finally:
            release_exclusive_file_lock(lock_path, lock)
        after = future.result(timeout=10)
    assert after["revision"] > before["revision"]
    assert after["mind_comparison"]["value"] == "A late observation."
    assert not lock_path.exists()


def test_cost_identity_account_context_and_mind_claim_are_separate():
    usage = [{"attempt_id": "api", "task_id": "root", "kind": "attempt", "cost_usd": None,
              "physical_context": {"capacity_total_tokens": 100, "target_total_tokens": 50}, "prompt_tokens": 43},
             {"attempt_id": "session", "task_id": "child", "kind": "subscription_session",
              "cost_usd": 0, "cost_final": True, "credential_profile_id": "account-a", "subscription_route": "codex"}]
    result = receipt.fold_tree("root", [], usage=usage + [usage[1]], coverage={"usage": "complete"},
                               mind_comparison="I would choose solo next time.")
    assert len(result["cost"]["attempts"]) == 2 and result["cost"]["total_usd"] is None
    assert result["cost"]["unknown_attempts"] == 1
    assert result["cost"]["attempts"][1]["account"]["credential_profile_id"] == "account-a"
    assert [p["task_id"] for p in result["supervisor_context_pressure"]] == ["root"]
    assert result["mind_comparison"]["author"] == "mind"
    usage[0].update(cost_usd=0.25, cost_final=True)
    result = receipt.fold_tree("root", [], usage=usage, coverage={"usage": "complete"})
    assert result["cost"]["total_usd"] == 0.25
    assert result["mind_comparison"]["state"] == "not_recorded"


def test_supervisor_pressure_does_not_relabel_an_executor_context():
    usage = [{"attempt_id": "api", "task_id": "child", "kind": "attempt", "physical_context": {"target_total_tokens": 50}}]
    child = actor("task:child", 0, 10, task_id="child")
    assert receipt.fold_tree("root", [child], usage=usage)["supervisor_context_pressure"] == []
    child["supervisor_route"] = {"model": "explicit-supervisor"}
    assert receipt.fold_tree("root", [child], usage=usage)["supervisor_context_pressure"][0]["task_id"] == "child"


def test_terminal_child_publication_revises_receipt_and_replay_does_not(tmp_path):
    write_task_result(tmp_path, "root", "running", started_at="2026-10-22T00:00:00Z")
    write_task_result(tmp_path, "child", "running", root_task_id="root", parent_task_id="root",
                      started_at="2026-10-22T00:01:00Z")
    receipt.publish(tmp_path, "root")
    write_task_result(tmp_path, "root", "completed", _terminal_observed=True)
    first = load_task_result(tmp_path, "root")["swarm_receipt"]
    assert first["state"] == "provisional"
    write_task_result(tmp_path, "child", "completed", _terminal_observed=True)
    closed = load_task_result(tmp_path, "root")["swarm_receipt"]
    assert closed["state"] == "closed" and closed["revision"] > first["revision"]
    write_task_result(tmp_path, "child", "completed")
    assert load_task_result(tmp_path, "root")["swarm_receipt"] == closed


def test_late_ledger_publication_revises_existing_receipt_after_transaction(tmp_path):
    from ouroboros.usage_accounting import record_subscription_session

    write_task_result(tmp_path, "root", "completed")
    receipt.publish(tmp_path, "root")
    before = load_task_result(tmp_path, "root")["swarm_receipt"]
    args = {"drive_root": tmp_path, "route": "codex", "task_id": "root", "root_task_id": "root",
            "credential_profile_id": "paid-profile", "spend_usd": 1.25}
    attempt_id = record_subscription_session("run-one", **args)
    after = load_task_result(tmp_path, "root")["swarm_receipt"]
    assert after["revision"] > before["revision"]
    assert after["cost"]["total_usd"] == 1.25
    assert after["cost"]["attempts"][0]["attempt_id"] == attempt_id
    record_subscription_session("run-one", **args)
    assert load_task_result(tmp_path, "root")["swarm_receipt"] == after


def test_late_settlement_replaces_unknown_cost_of_same_attempt(tmp_path):
    from ouroboros.usage_store import hold

    write_task_result(tmp_path, "root", "completed")
    receipt.publish(tmp_path, "root")
    previous = None
    for state in ("reserved", "dispatched", "unresolved", "settled"):
        fields = {"attempt_id": "late-api", "kind": "attempt", "state": state,
                  "task_id": "root", "root_task_id": "root", "provider": "test", "model": "test/model"}
        if state == "settled":
            fields.update(cost_usd=0.4, cost_final=True, settle_reason="late_receipt")
        with hold(tmp_path) as txn:
            previous = txn.write(fields, previous)
        result = load_task_result(tmp_path, "root")["swarm_receipt"]
        assert len(result["cost"]["attempts"]) == 1
        assert result["cost"]["total_usd"] == (0.4 if state == "settled" else None)
    assert receipt.publish(tmp_path, "root") == result


def test_sleep_selection_and_actual_wake_produce_only_observed_edges(tmp_path, monkeypatch):
    from ouroboros import model_sleep, swarm_receipt_sources
    from ouroboros import store
    from ouroboros.subagent_metrics import set_name

    ctx = SimpleNamespace(task_id="root", drive_root=tmp_path, budget_drive_root=None,
                          _model_sleep={"sleep_id": "sleep-one", "tasks": ["child"]})
    model_sleep.begin(ctx, started_at=4)
    ctx._model_sleep = None  # owner_wait clears the request before announcing the warm resume
    monkeypatch.setattr("ouroboros.swarm_wait_facts.utc_now_iso", lambda: "1970-01-01T00:00:10Z")
    model_sleep.announce_resumed(ctx, "task:child:completed")
    with store.open_store(tmp_path, write=False) as txn:
        waits = list(txn.set_rows(set_name("swarm_waits", "root")).values())
    assert waits[0]["selection"]["tasks"] == ["child"]
    assert waits[0]["woke_by"] == "task:child:completed"
    actors = {"task:root": actor("task:root", 0, 12), "task:child": actor("task:child", 4, 10)}
    edges = swarm_receipt_sources._wait_edges(actors, waits)
    assert any(e["from"] == "task:child" for e in edges)
    assert receipt.fold_tree("root", list(actors.values()), edges=edges)["longest_causal_path"]["observed_seconds"] == 8
    waits[0]["selection"]["tasks"] = []
    actors = {"task:root": actor("task:root", 0, 12), "task:child": actor("task:child", 4, 10)}
    assert not any(e["from"] == "task:child" for e in swarm_receipt_sources._wait_edges(actors, waits))


def test_ordinary_task_writes_do_not_scan_tree_until_receipt_exists(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.swarm_receipt_sources.tree_inputs", lambda *a, **kw: pytest.fail("tree scan"))
    write_task_result(tmp_path, "root", "running")
    write_task_result(tmp_path, "root", "completed")


@pytest.mark.parametrize("case", ["selected", "unselected", "pending", "additional"])
def test_review_detach_retains_the_canonical_waking_actor(tmp_path, monkeypatch, case):
    from ouroboros import review_pause, store, swarm_receipt_sources
    from ouroboros.subagent_metrics import set_name
    from tests.test_review_wake_selection import _entry, _pin

    entry = _entry("pending" if case == "pending" else "responded", additional=case == "additional")
    selected = {"reviewers": [] if case == "unselected" else [_pin()]}
    binding = {"task_id": "root", "drive_root": tmp_path, "selection": selected}
    monkeypatch.setattr(review_pause, "author_paused", lambda _op: False)
    monkeypatch.setattr("ouroboros.swarm_wait_facts.utc_now_iso", lambda: "1970-01-01T00:00:10Z")
    with review_pause.receptive_author(binding):
        binding["_swarm_wait_started"] = 4
        assert review_pause.author_detaches(None, [entry]) is (case == "selected")
    with store.open_store(tmp_path, write=False) as txn:
        waits = list(txn.set_rows(set_name("swarm_waits", "root")).values())
    assert waits[0]["woke_by"] == ("review_result" if case == "selected" else "whole_panel_returned")
    actors = {"task:root": actor("task:root", 0, 12), "review:op:seat": actor("review:op:seat", 4, 10)}
    edges = swarm_receipt_sources._wait_edges(actors, waits)
    assert any(edge["from"] == "review:op:seat" for edge in edges) is (case == "selected")


def test_only_tree_end_initializes_the_tree_receipt(tmp_path):
    from ouroboros.task_finalization import build_swarm_efficiency

    write_task_result(tmp_path, "root", "running")
    write_task_result(tmp_path, "child", "running", root_task_id="root", parent_task_id="root")
    env = SimpleNamespace(drive_root=tmp_path)
    build_swarm_efficiency(env, {"id": "child", "root_task_id": "root", "parent_task_id": "root"})
    assert "swarm_receipt" not in load_task_result(tmp_path, "child")
    assert "swarm_receipt" not in load_task_result(tmp_path, "root")
    build_swarm_efficiency(env, {"id": "root"})
    assert load_task_result(tmp_path, "root")["swarm_receipt"]["state"] == "provisional"


def test_receipt_source_failure_does_not_erase_observed_fanout(tmp_path, monkeypatch):
    from tests.test_swarm_rollup import emit, build

    write_task_result(tmp_path, "parent", "running")
    emit(tmp_path)
    monkeypatch.setattr("ouroboros.swarm_receipt_sources.tree_inputs",
                        lambda *a, **kw: (_ for _ in ()).throw(OSError("unavailable source")))
    assert build(tmp_path)["subagent_count"] == 1


def test_full_task_handoff_discloses_receipt_only_when_recorded():
    import json
    from ouroboros.tools.control_task_results import _subtask_outcome_summary

    measured = receipt.fold_tree("root", [actor("root", 0, 2)])
    assert json.loads(_subtask_outcome_summary({"swarm_receipt": measured}))["swarm_receipt"] == measured
    assert "swarm_receipt" not in json.loads(_subtask_outcome_summary({}))


@pytest.mark.parametrize("task_id,expected", [("root", True), ("system:provider_test", False)])
def test_accounting_refresh_reads_only_task_result_identities(tmp_path, monkeypatch, task_id, expected):
    """The hook asks the store for two typed fields of a task identity; it never parses a
    task result and never asks about a system accounting id."""
    from ouroboros import swarm_receipt, task_results, task_store

    lookups, publishes = [], []
    monkeypatch.setattr(task_results, "load_task_result",
                        lambda root, identity, **kwargs: pytest.fail("a receipt refresh parsed a task result"))
    monkeypatch.setattr(task_store, "receipt_anchor",
                        lambda root, identity: lookups.append(identity) or (identity, True))
    monkeypatch.setattr(swarm_receipt, "publish", lambda root, identity: publishes.append(identity))
    swarm_receipt.refresh_for_task(tmp_path, task_id)
    assert lookups == publishes == ([task_id] if expected else [])


def test_refresh_follows_the_stored_root_and_skips_roots_without_a_receipt(tmp_path, monkeypatch):
    from ouroboros import swarm_receipt
    from ouroboros.task_results import write_task_result

    write_task_result(tmp_path, "root-a", "running")
    write_task_result(tmp_path, "child-a", "completed", root_task_id="root-a")
    publishes = []
    monkeypatch.setattr(swarm_receipt, "publish", lambda root, identity: publishes.append(identity))
    swarm_receipt.refresh_for_task(tmp_path, "child-a")
    assert publishes == []  # the root holds no receipt yet: nothing to revise
    from ouroboros.task_store import patch_document
    patch_document(tmp_path, "root-a", lambda current: {**current, "swarm_receipt": {"revision": 1}})
    swarm_receipt.refresh_for_task(tmp_path, "child-a")
    swarm_receipt.refresh_for_task(tmp_path, {"task_id": "child-a", "root_task_id": "root-a"})
    assert publishes == ["root-a", "root-a"]
    assert swarm_receipt.refresh_for_task(tmp_path, "absent-task") is None and publishes == ["root-a", "root-a"]


def test_receipt_anchor_resolves_the_root_and_the_receipt_without_a_body_parse(tmp_path):
    """Typed root, legacy metadata root (NULL or empty typed column), the task itself; JSON
    null and an absent key both mean no receipt, any other value means one; a root header
    without its body triggers nothing; an absent task is None."""
    from ouroboros import store, task_store

    with store.open_store(tmp_path, write=True) as txn:
        txn.store_task({"task_id": "root", "status": "running", "swarm_receipt": {"revision": 1}})
        txn.store_task({"task_id": "child", "status": "completed", "root_task_id": "root"})
        txn.store_task({"task_id": "legacy", "status": "completed", "metadata": {"root_task_id": "root"}})
        txn.conn.execute("UPDATE tasks SET root_task_id=NULL WHERE task_id='legacy'")
        txn.store_task({"task_id": "self", "status": "running"})
        txn.conn.execute("UPDATE tasks SET root_task_id=NULL WHERE task_id='self'")
    assert task_store.receipt_anchor(tmp_path, "child") == ("root", True)
    assert task_store.receipt_anchor(tmp_path, "legacy") == ("root", True)
    assert task_store.receipt_anchor(tmp_path, "self") == ("self", False)
    assert task_store.receipt_anchor(tmp_path, "missing") is None
    with store.open_store(tmp_path, write=True) as txn:
        txn.conn.execute("UPDATE tasks SET root_task_id='' WHERE task_id='legacy'")
    assert task_store.receipt_anchor(tmp_path, "legacy") == ("root", True)
    for value, expected in ((None, False), ({}, True), (False, True), ({"revision": 2}, True)):
        with store.open_store(tmp_path, write=True) as txn:
            txn.store_task({"task_id": "root", "status": "running", "swarm_receipt": value})
        assert task_store.receipt_anchor(tmp_path, "child") == ("root", expected), value
    with store.open_store(tmp_path, write=True) as txn:
        txn.conn.execute("DELETE FROM task_bodies WHERE task_id='root'")
    assert task_store.receipt_anchor(tmp_path, "child") == ("root", False)
    assert task_store.receipt_anchor(tmp_path, "root") is None
