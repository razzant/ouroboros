"""Conditional result reads retain current facts without recording a seen state."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from ouroboros.task_results import write_task_result
from ouroboros.task_status import load_effective_task_result
from ouroboros.tools import control_task_results as results
from ouroboros.tools.join_ledger import _child_result_sha256
from tests.wait_handoff_support import full_wait_payload

BODY = "Complete child analysis.\n" * 1200
TRACE = "Exact trace summary.\n" * 400


def _ctx(drive):
    return SimpleNamespace(
        drive_root=drive, budget_drive_root=drive, task_id="parent1",
        task_attempt=1, task_metadata={"root_task_id": "parent1", "budget_drive_root": str(drive)},
        _loop_mailbox_seen_ids=set(),
    )


def _child(drive, task_id="child1", **changes):
    values = dict(
        result=BODY, trace_summary=TRACE, parent_task_id="parent1",
        root_task_id="parent1", delegation_role="subagent",
    )
    values.update(changes)
    write_task_result(drive, task_id, "completed", **values)
    return load_effective_task_result(drive, task_id)


def _digest(drive, task_id="child1"):
    return _child_result_sha256(load_effective_task_result(drive, task_id))


@pytest.mark.parametrize("surface", ["get", "wait", "batch"])
def test_condition_is_explicit_and_full_read_survives_a_new_context(tmp_path, surface):
    _child(tmp_path)
    known = _digest(tmp_path)

    def read(ctx, condition=False):
        if surface == "batch":
            kwargs = {"known_result_sha256_by_task": {"child1": known}} if condition else {}
            return results._wait_for_tasks(ctx, ["child1"], timeout_sec=0, **kwargs)
        handler = results._get_task_result if surface == "get" else results._wait_for_task
        kwargs = {"known_result_sha256": known} if condition else {}
        if surface == "wait":
            kwargs["timeout_sec"] = 0
        return handler(ctx, "child1", **kwargs)

    ctx = _ctx(tmp_path)
    before = (tmp_path / "task_results" / "child1.json").read_bytes()
    first = read(ctx)
    unchanged = read(ctx, True)
    assert BODY in (first if surface == "get" else full_wait_payload(ctx, first)["tasks"]["child1"]["result"])
    assert known in first and known in unchanged
    assert "result_unchanged" in unchanged
    assert "Complete child analysis." not in unchanged
    assert "Exact trace summary." not in unchanged
    full_chars = len(first) if surface == "get" else len(json.dumps(full_wait_payload(ctx, first)))
    assert len(unchanged) < full_chars / 5
    # A result hash is neither "seen" nor proof of a current in-context copy.
    def result_view(text):
        return text if surface == "get" else full_wait_payload(ctx, text)["tasks"]
    assert result_view(read(ctx)) == result_view(first)
    assert result_view(read(_ctx(tmp_path))) == result_view(first)
    assert (tmp_path / "task_results" / "child1.json").read_bytes() == before
    assert not (tmp_path / "state" / "task_trees").exists()


@pytest.mark.parametrize("field,new_value", [
    ("result", "A revised complete answer."),
    ("trace_summary", "A revised trace."),
    ("status", "failed"),
    ("artifact_status", "failed"),
    ("artifacts", [{"name": "report.txt", "sha256": "b" * 64}]),
    ("terminal_host_notice", "Unresolved delegated execution remains."),
])
def test_semantic_change_returns_full_single_and_batch(tmp_path, monkeypatch, field, new_value):
    data = dict(task_id="child1", status="completed", result=BODY, trace_summary=TRACE)
    original = _child_result_sha256(data)
    data[field] = new_value
    monkeypatch.setattr(results, "load_effective_task_result", lambda *_args: copy.deepcopy(data))
    monkeypatch.setattr(results, "_unminted_wait_ids", lambda *_args: [])
    monkeypatch.setattr(results, "wait_for_effective_tasks", lambda *_args, **_kwargs: {
        "tasks": {"child1": copy.deepcopy(data)}, "all_terminal": True, "elapsed_sec": 0,
    })
    single = results._get_task_result(_ctx(tmp_path), "child1", known_result_sha256=original)
    batch = full_wait_payload(_ctx(tmp_path), results._wait_for_tasks(
        _ctx(tmp_path), ["child1"], timeout_sec=0,
        known_result_sha256_by_task={"child1": original},
    ))["tasks"]["child1"]
    assert "result_unchanged" not in single
    assert "result_unchanged" not in batch
    assert str(data["result"]) in single
    assert batch["result"] == data["result"]


def test_batch_omits_only_matched_children_and_keeps_unknown_truth(tmp_path):
    _child(tmp_path)
    _child(tmp_path, "child2", result="Other complete result")
    old = _digest(tmp_path)
    result = json.loads(results._wait_for_tasks(
        _ctx(tmp_path), ["child1", "child2", "unknown"], timeout_sec=0,
        known_result_sha256_by_task={"child1": old, "unknown": _child_result_sha256({})},
    ))
    assert result["tasks"]["child1"]["result_unchanged"] is True
    assert "result" not in result["tasks"]["child1"]
    assert result["tasks"]["child2"]["result"] == "Other complete result"
    assert result["tasks"]["unknown"].get("result_unchanged") is not True
    source = result["tasks"]["child1"]["result_source"]
    full = results._get_task_result(_ctx(tmp_path), **source["arguments"])
    assert BODY in full


def test_accounting_and_current_handoff_facts_do_not_duplicate_body(tmp_path, monkeypatch):
    data = dict(task_id="child1", status="completed", result=BODY, trace_summary=TRACE,
                delegate_terminal_reconciliation={"audit_status": "pending", "open_run_ids": ["run-a"]})
    original = _child_result_sha256(data)
    data.update(
        cost_usd=None, cost_final=False,
        capability_delta={"reduced": True, "legacy_note": "Source unavailable now"},
        verification_ledger={"schema_version": 1, "summary": {"entry_count": 1, "failed": 1}},
    )
    assert _child_result_sha256(data) == original
    monkeypatch.setattr(results, "load_effective_task_result", lambda *_args: copy.deepcopy(data))
    monkeypatch.setattr(results, "_unminted_wait_ids", lambda *_args: [])
    monkeypatch.setattr(results, "wait_for_effective_tasks", lambda *_args, **_kwargs: {
        "tasks": {"child1": copy.deepcopy(data)}, "all_terminal": True, "elapsed_sec": 0,
    })
    shown = results._get_task_result(_ctx(tmp_path), "child1", known_result_sha256=original)
    assert "cost=unknown" in shown
    assert "Source unavailable now" in shown and "run-a" in shown
    assert '"entry_count": 1' in shown
    assert BODY not in shown
    batch = full_wait_payload(_ctx(tmp_path), results._wait_for_tasks(
        _ctx(tmp_path), ["child1"], timeout_sec=0,
        known_result_sha256_by_task={"child1": original},
    ))["tasks"]["child1"]
    assert batch["accounted_upper_bound_usd"] is None and batch["cost_final"] is False
    assert batch["capability_delta"] == data["capability_delta"]


def test_source_or_authority_request_is_not_suppressed_by_a_known_result(tmp_path, monkeypatch):
    _child(tmp_path)
    import ouroboros.agent_startup_checks as checks
    import ouroboros.task_finalization as finalization

    monkeypatch.setattr(checks, "task_result_authority_projection", lambda *_args, **_kwargs: {"current": True})
    monkeypatch.setattr(finalization, "completion_source_projection", lambda *_args: {"text": "exact source"})
    result = json.loads(results._get_task_result(
        _ctx(tmp_path), "child1", include_authority=True, include_completion_source=True,
        known_result_sha256=_digest(tmp_path),
    ))
    assert result["authority"] == {"current": True}
    assert result["completion_source"]["text"] == "exact source"
    assert "result_unchanged" not in result


@pytest.mark.parametrize("known", ["", "stale", None, 17, {"child1": "bad"}])
def test_nonmatching_condition_never_hides_result(tmp_path, known):
    _child(tmp_path)
    assert BODY in results._get_task_result(_ctx(tmp_path), "child1", known_result_sha256=known)


def test_unchanged_wait_still_delivers_parent_mailbox_without_ack(tmp_path):
    from ouroboros.owner_mailbox import acknowledged_task_message_ids, write_owner_message

    _child(tmp_path)
    known = _digest(tmp_path)
    assert write_owner_message(tmp_path, "Keep the complete original.", "parent1", msg_id="owner-followup")
    ctx = _ctx(tmp_path)
    single = results._wait_for_task(ctx, "child1", timeout_sec=0, known_result_sha256=known)
    assert "unread message" in single and "result_unchanged" in single
    assert acknowledged_task_message_ids(tmp_path, "parent1", attempt_key=1) == set()
    batch = json.loads(results._wait_for_tasks(
        _ctx(tmp_path), ["child1"], timeout_sec=0,
        known_result_sha256_by_task={"child1": known},
    ))
    assert batch["early_return"]["reason"] == "owner_mailbox_pending"
    assert batch["tasks"]["child1"]["result_unchanged"] is True


def test_public_schemas_offer_the_condition_without_changing_required_args():
    from ouroboros.tools.control import get_tools

    tools = {entry.name: entry for entry in get_tools()}
    for name in ("get_task_result", "wait_task", "wait_tasks"):
        schema = tools[name].schema["parameters"]
        field = "known_result_sha256_by_task" if name == "wait_tasks" else "known_result_sha256"
        assert field in schema["properties"]
        assert field not in schema["required"]


def test_duplicate_reference_and_new_receipts_remain_visible_when_body_matches(tmp_path, monkeypatch):
    import ouroboros.outcomes as outcomes

    data = dict(task_id="child1", status="rejected_duplicate", result="Duplicate answer",
                trace_summary="trace", duplicate_of="original")
    known = _child_result_sha256(data)
    monkeypatch.setattr(results, "load_effective_task_result", lambda *_args: data)
    monkeypatch.setattr(outcomes, "read_verification_receipts_from_roots", lambda *_args: [
        {"status": "failed", "check": "a newly observed verification", "matched": False},
    ])
    result = results._get_task_result(_ctx(tmp_path), "child1", known_result_sha256=known)
    assert '"duplicate_of": "original"' in result
    assert "a newly observed verification" in result
    assert '"matched": false' in result
    assert "Duplicate answer" not in result
