"""CUT1 proof: compare the live row reducer with the import fold on the same chain."""
from __future__ import annotations

from contextlib import closing
import json
import sqlite3

from ouroboros import delegate_custody as custody
from ouroboros import delegate_custody_current as current
from ouroboros import delegate_invocation as inv
from ouroboros import store, store_import


def _chain():
    rows = []

    def add(kind, **facts):
        rows.append({"type": kind, "event_id": f"event-{len(rows)}",
                     "ts": f"2026-10-11T00:00:{len(rows):02d}Z", **facts})

    add(custody.START_REQUESTED, invocation_id="inv-a", task_id="t1", root_task_id="t1",
        billing_group={"billing_group_id": "group-a", "billing_group_limit_usd": 5})
    add(custody.STARTED, run_id="a", invocation_id="inv-a", task_id="t1", root_task_id="t1",
        project_id="project", project_owned=True, delegated=True,
        resource_ref={"workspace_kind": "directory", "strategy": "copy"})
    add(custody.CONTAINMENT_FAULT, run_id="a", task_id="t1", detail="synthetic fault")
    add(custody.CONTAINMENT_RESOLVED, run_id="a", task_id="t1")
    add(custody.OUTPUT_SPILLED, run_id="a", task_id="t1", staged=True, artifact="out.txt",
        full_content=True, sha256="0" * 64)
    add(custody.OUTPUT_CONSUMED, run_id="a", task_id="t1", sha256="0" * 64, reader_task_id="t1")
    add(custody.SETTLED, run_id="a", task_id="t1", cost_usd=1.25, spend_disclosed=True,
        model="test-model", access_profile="test-profile", source="delegated_subagent")
    add(custody.COLLECTION_ATTEMPTED, run_id="a", task_id="t1",
        outcome={"inputs": {"terminal_sha": "terminal"}, "generation": "body", "retry": "transient"})
    add(custody.NAMED_EVIDENCE_RETAINED, run_id="a", task_id="t1",
        outcome={"directory": "evidence-generation", "status": "partial", "inputs": {"witness": "source"}})
    add(custody.CAPTURE_JOB_CHANGED, run_id="a", task_id="t1", expected_id="",
        outcome={"id": "capture-job", "state": "pending", "binding": "snapshot"})
    add(custody.STARTED, run_id="a", invocation_id="inv-a", task_id="t1", root_task_id="t1")
    add(custody.CAPTURE_JOB_CHANGED, run_id="a", task_id="t1", expected_id="older-job",
        outcome={"id": "stale", "state": "ready"})
    add(custody.PATCH_DISPOSED, run_id="a", task_id="t1")
    add(custody.START_REQUESTED, invocation_id="inv-b", task_id="t2", root_task_id="t1",
        billing_group={"billing_group_id": "group-b", "billing_group_limit_usd": 5})
    add(custody.START_FAILED, invocation_id="inv-b", task_id="t2", definite=True)
    add(custody.START_REQUESTED, invocation_id="inv-b", task_id="t2", root_task_id="t1")
    add(custody.START_REQUESTED, invocation_id="inv-c", task_id="t3", root_task_id="t1")
    add(custody.STARTED, run_id="d", task_id="t4", root_task_id="t1",
        project_id="project", project_owned=True, delegated=True)
    add(custody.PROJECT_RETIRED, run_id="d", project_id="project")
    add(custody.SETTLED, run_id="d", task_id="t4")
    add(custody.SETTLED, run_id="unknown-settled", task_id="t9")
    add(custody.CONTAINMENT_FAULT, run_id="unknown-held", task_id="t9")
    add(current._NUDGE, task_id="t1")
    add("delegate_run_start_blocked", task_id="t5", source="delegated_subagent")
    add("delegate_run_patch_verdict", task_id="t1", run_id="a", verdict="accepted")
    add(custody.CLOSED_ABSENT, run_id="a", task_id="t1")
    # A repeated outbox append replays the chain row, not the live transaction.
    rows.extend([dict(rows[6]), dict(rows[-2])])
    return rows


def _normalise(value):
    return json.loads(json.dumps(value, sort_keys=True))


def _reduced(rows):
    with closing(sqlite3.connect(":memory:", isolation_level=None)) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(store._SCHEMA)
        txn = store.Txn(conn)
        for row in custody._unique_rows(rows):
            current._reduce(txn, row, current._loaded(txn, row), {})
        return _normalise({
            "runs": {row["run_id"]: (json.loads(row["facts"]), row["state"]) for row in
                     conn.execute("SELECT run_id, facts, state FROM custody_runs")},
            "open": txn.set_rows("custody_open"), "refused": txn.set_rows("custody_refused"),
            "marks": txn.set_rows("custody_marks"),
            "invocations": txn.set_rows(inv.INVOCATIONS), "operations": txn.set_rows(inv.OPERATIONS),
            "event_ordinals": txn.set_rows(inv.EVENTS), "ordinal": txn.meta(inv.ORDINAL),
            "cost_revisions": {tid: facts["cost_revision"] for tid, facts in txn.set_rows("delegated_runs").items()
                               if facts.get("cost_revision")},
        }), _normalise(txn.set_rows("delegated_runs"))


def test_required_chain_matches_at_every_prefix_and_at_import(tmp_path):
    rows = _chain()
    requests = [row for row in rows if row["type"] == custody.START_REQUESTED and row.get("invocation_id") == "inv-b"]
    assert len({row["event_id"] for row in requests}) == 2
    for end in range(1, len(rows) + 1):
        live, _ = _reduced(rows[:end])
        folded = _normalise(current.fold(rows[:end]))
        # Live publication debt and import recovery have different lifetimes:
        # closed history needs no new publication. Custody facts remain identical.
        live_cost, folded_cost = live.pop("cost_revisions"), folded.pop("cost_revisions")
        assert folded_cost.items() <= live_cost.items()
        assert live == folded, f"prefix {end}"
    assert set(live["open"]) == {"invocation:inv-c"}
    assert set(live["refused"]) == {"inv-b"}
    assert not live["runs"]["a"][0]["project_owned"]
    assert not live["runs"]["d"][0]["project_owned"]
    assert live["runs"]["a"][0]["collection_attempt"]["generation"] == "body"
    assert live["runs"]["a"][0]["named_evidence"]["status"] == "partial"
    assert live["runs"]["a"][0]["capture_job"]["id"] == "capture-job"
    assert len(live["marks"]["t1"]["patch_verdicts"]) == 1
    path = custody.event_log_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    imported, receipts = store_import._custody_rows(tmp_path, [
        (task, {"delegated_runs_unreconciled": True}) for task in ("t1", "t4", "t9")])
    assert imported.pop("chain_offset") > 0
    assert imported.pop("cost_revisions") == folded_cost
    assert _normalise(imported) == live
    expected = _reduced(rows)[1]
    for tid in list(expected):
        if tid not in folded_cost:
            expected[tid].pop("cost_revision", None)
        if not expected[tid].get("closed_runs") and not expected[tid].get("cost_revision"):
            del expected[tid]
    assert _normalise(receipts["delegated_runs"]) == expected


def test_start_after_repeated_refused_request_preserves_equivalence():
    """A late start keeps the original request and billing group through refusal/retry."""
    rows = [row for row in _chain() if row.get("invocation_id") == "inv-b"]
    rows.append({"type": custody.STARTED, "event_id": "late-start", "run_id": "b",
                 "invocation_id": "inv-b", "task_id": "t2", "root_task_id": "t1"})
    live, imported = _reduced(rows)[0], _normalise(current.fold(rows))
    assert live["runs"]["b"][0]["request_event"] == rows[0]
    assert live["runs"]["b"][0]["billing_group"] == rows[0]["billing_group"]
    assert live == imported
