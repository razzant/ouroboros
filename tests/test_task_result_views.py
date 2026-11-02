"""Short result-view consumer proofs."""
from __future__ import annotations

import hashlib
import json
import pytest
from types import SimpleNamespace

from ouroboros.task_results import write_task_result
from ouroboros.tools import control_task_results as results


def _ctx(root):
    return SimpleNamespace(
        drive_root=root, budget_drive_root=root, task_id="parent", task_attempt=1,
        task_metadata={"budget_drive_root": str(root), "root_task_id": "parent"},
        _loop_mailbox_seen_ids=set(), owner_wait_callback=lambda *_a, **_kw: None,
    )


def _write(root, **changes):
    row = dict(
        result="Exact authored answer.", terminal_origin="model_final",
        parent_task_id="parent", root_task_id="parent", delegation_role="subagent",
        accounted_upper_bound_usd=1.25, cost_final=False,
        capability_delta={"reduced": True, "legacy_note": "network unavailable"},
        delegate_terminal_reconciliation={"audit_status": "complete", "open_run_ids": ["run-a"]},
        retry_lineage=[{"task_id": "older", "status": "failed", "terminal_origin": "host_notice"}],
    )
    row.update(changes)
    write_task_result(root, "child", "completed", **row)


def test_omitted_and_explicit_full_are_byte_identical(tmp_path):
    _write(tmp_path)
    ctx = _ctx(tmp_path)
    full = results._get_task_result(ctx, "child")
    assert full == results._get_task_result(ctx, "child", view="full")
    authority = results._get_task_result(ctx, "child", include_authority=True)
    assert authority == results._get_task_result(
        ctx, "child", include_authority=True, view="full",
    )
    answer = results._get_task_result(ctx, "child", view="answer")
    assert len(answer.encode("utf-8")) < len(full.encode("utf-8"))
    assert json.loads(answer)["answer"] == "Exact authored answer."


def test_answer_view_keeps_core_handoff_facts_and_uses_child_result_identity(tmp_path):
    _write(tmp_path, terminal_host_notice="Host retained a non-clean terminal limitation.")
    ctx = _ctx(tmp_path)
    first = json.loads(results._get_task_result(ctx, "child", view="answer"))
    answer_sha = hashlib.sha256("Exact authored answer.".encode()).hexdigest()
    result_sha = first["child_result_sha256"]
    assert first["answer"] == "Exact authored answer."
    assert first["answer_sha256"] == answer_sha
    assert first["terminal_origin"] == "model_final"
    assert first["outcome_axes"]
    assert first["nonclean_terminal_host_notice"]
    assert first["capability_delta"]["reduced"] is True

    unchanged = json.loads(results._get_task_result(
        ctx, "child", view="answer", known_result_sha256=result_sha,
    ))
    assert unchanged["result_unchanged"] is True and "answer" not in unchanged
    assert unchanged["result_source"]["immutable"] is False


def test_summary_and_evidence_refresh_when_result_hash_matches(tmp_path, monkeypatch):
    ledger = {
        "schema_version": 1,
        "summary": {"entry_count": 2, "passed": 1, "failed": 1},
        "entries": [{"large": "not copied"}, {"large": "not copied"}],
    }
    bundle = {
        "schema_version": 1, "status": "ready",
        "artifacts": [{
            "kind": "user_file", "name": f"answer-{index}.txt",
            "path": f"/private/full/path/answer-{index}.txt", "status": "ready",
            "sha256": f"{index:064x}", "size": index + 1, "errors": [],
        } for index in range(12)],
        "errors": [],
    }
    _write(tmp_path, verification_ledger=ledger, artifact_bundle=bundle)
    monkeypatch.setattr(
        "ouroboros.outcomes.read_verification_receipts_from_roots",
        lambda *_a, **_kw: [{
            "status": "pass", "matched": True, "criterion_id": "unit-tests",
            "contract_kind": "explicit_command", "check": "pytest -q",
        }],
    )
    ctx = _ctx(tmp_path)
    answer_sha = hashlib.sha256("Exact authored answer.".encode()).hexdigest()
    result_sha = json.loads(results._get_task_result(ctx, "child", view="answer"))["child_result_sha256"]
    summary = json.loads(results._get_task_result(
        ctx, "child", view="summary", known_result_sha256=result_sha,
    ))
    assert summary["known_result_matches"] is True
    assert summary["cost"]["accounted_upper_bound_usd"] == 1.25
    assert summary["delegated_custody"]["open_run_ids"] == ["run-a"]
    assert summary["retry"]["lineage"][0]["task_id"] == "older"
    assert summary["answer_source"]["arguments"] == {"task_id": "child", "view": "answer"}

    # Same authored bytes, changed dynamic custody/cost: no unchanged shortcut.
    _write(tmp_path, accounted_upper_bound_usd=2.5, verification_ledger=ledger,
           artifact_bundle=bundle,
           delegate_terminal_reconciliation={"audit_status": "pending", "open_run_ids": ["run-b"]})
    refreshed = json.loads(results._get_task_result(
        ctx, "child", view="summary", known_result_sha256=result_sha,
    ))
    assert refreshed["answer_sha256"] == answer_sha
    assert refreshed["cost"]["accounted_upper_bound_usd"] == 2.5
    assert refreshed["delegated_custody"]["open_run_ids"] == ["run-b"]

    # Evidence is a layered view: it keeps the same fresh dynamic summary and
    # adds current authorized projection recipes without copying stored rows.
    _write(tmp_path, accounted_upper_bound_usd=3.5, verification_ledger=ledger,
           artifact_bundle=bundle,
           delegate_terminal_reconciliation={"audit_status": "pending", "open_run_ids": ["run-c"]})
    evidence = json.loads(results._get_task_result(
        ctx, "child", view="evidence", known_result_sha256=result_sha,
    ))
    encoded = json.dumps(evidence)
    assert evidence["known_result_matches"] is True
    assert evidence["cost"]["accounted_upper_bound_usd"] == 3.5
    assert evidence["delegated_custody"]["open_run_ids"] == ["run-c"]
    assert evidence["retry"]["lineage"][0]["task_id"] == "older"
    assert evidence["verification_ledger"] == {
        "schema_version": 1,
        "summary": {"entry_count": 2, "passed": 1, "failed": 1},
        "entry_count": 2,
    }
    assert evidence["verification_receipts"][0]["criterion_id"] == "unit-tests"
    assert evidence["verification_receipts_omitted"] == 0
    artifact_facts = evidence["artifact_facts"]
    assert artifact_facts["schema_version"] == 1 and artifact_facts["status"] == "ready"
    assert artifact_facts["artifact_count"] == 12 and artifact_facts["artifacts_omitted"] == 2
    assert len(artifact_facts["artifacts"]) == 10
    assert artifact_facts["artifacts"][0] == {
        "kind": "user_file", "status": "ready", "sha256": "0" * 64,
        "size": 1, "name": "answer-0.txt",
    }
    assert all(ref["kind"] == "current_reread_recipe" and ref["immutable"] is False
               for ref in evidence["projection_refs"])
    assert "/private/full/path" not in encoded
    assert "trace_refs" not in encoded and "manifest_ref" not in encoded and "EOF" not in encoded


def test_short_view_rejects_source_selector_before_io(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    monkeypatch.setattr(results, "load_effective_task_result",
                        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("source I/O reached")))
    text = results._get_task_result(ctx, "child", view="answer", include_authority=True)
    assert "TOOL_ARG_ERROR" in text and "incompatible with authority/source selectors" in text
    ranged = results._get_task_result(ctx, "child", view="summary", source_start_char=0)
    assert "TOOL_ARG_ERROR" in ranged and "incompatible with authority/source selectors" in ranged


def test_presence_evidence_recipes_retain_the_authorized_scope(tmp_path, monkeypatch):
    _write(tmp_path)
    ctx = _ctx(tmp_path)
    import ouroboros.presence_authority as presence

    monkeypatch.setattr(presence, "presence_caller_binding", lambda _ctx: "binding")
    monkeypatch.setattr(presence, "presence_work_refusal", lambda *_a, **_kw: "")
    monkeypatch.setattr(presence, "presence_effective_refusal", lambda *_a, **_kw: "")
    evidence = json.loads(results._get_task_result(
        ctx, "child", view="evidence", presence_scope="own_binding",
    ))
    assert evidence["projection_refs"]
    assert all(ref["arguments"]["presence_scope"] == "own_binding"
               for ref in evidence["projection_refs"])
    summary = json.loads(results._get_task_result(
        ctx, "child", view="summary", presence_scope="own_binding",
    ))
    assert summary["answer_source"]["arguments"]["presence_scope"] == "own_binding"


def test_settled_wait_task_returns_requested_short_view(tmp_path):
    _write(tmp_path)
    text = results._wait_for_task(_ctx(tmp_path), "child", timeout_sec=0, view="answer")
    assert "Task wait completed" in text
    payload = json.loads(text.split("\n\n", 1)[1])
    assert payload["view"] == "answer" and payload["answer"] == "Exact authored answer."


@pytest.mark.parametrize("view", ["answer", "summary", "evidence"])
@pytest.mark.parametrize("unchanged", [False, True])
def test_every_short_view_keeps_current_handoff_disclosures(tmp_path, view, unchanged):
    _write(tmp_path, cancel_origin={"actor": "owner"}, retired_tool_invocations=[{"tool": "send_file"}],
           unread_mailbox={"rows": [json.dumps({"text": "owner correction"})], "read_complete": True},
           effort_requested="high", effort_level="low", effort_source="pin")
    ctx = _ctx(tmp_path)
    identity = json.loads(results._get_task_result(ctx, "child", view="answer"))["child_result_sha256"]
    value = json.loads(results._get_task_result(ctx, "child", view=view,
                                              known_result_sha256=identity if unchanged else ""))
    assert value["cancel_origin"] == {"actor": "owner"}
    assert "external effects remain unknown" in value["interrupted_tool_calls"]["warning"]
    assert value["interrupted_tool_calls"]["invocations"] == [{"tool": "send_file"}]
    assert "owner correction" in value["unread_mail_notice"]
    assert value["effort"] == {"requested": "high", "level": "low", "source": "pin"}
    if view == "answer" and unchanged:
        assert value["result_unchanged"] and "answer" not in value
    answer_hash = value["answer_sha256"]
    answer = json.loads(results._get_task_result(ctx, "child", view="answer", known_result_sha256=answer_hash))
    assert "answer" in answer and not answer.get("result_unchanged")
