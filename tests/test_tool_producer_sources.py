"""Host notes stay visible without corrupting data retained for a program."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros import artifacts
from ouroboros import loop_tool_execution as execution
from ouroboros.tools import extension_dispatch
from ouroboros.tools.core import _read_file
from ouroboros.tools.tool_context import ToolContext
from ouroboros.tools.tool_result import ToolResult, _compose_execute_result_result
from tests._tool_result_delivery_shared import measured_fit


WARNING = "⚠️ SAFETY_WARNING: inspect the returned data before acting."
PAYLOAD = json.dumps({"ok": False, "body": "Unicode: 雪\n\n---\nSAFETY_WARNING is data", "rows": [None, True]})


@pytest.mark.parametrize("route", ["builtin", "extension", "mcp"])
@pytest.mark.parametrize("warning", ["", WARNING])
def test_dispatch_preserves_payload_and_failure_without_parsing_notes(monkeypatch, route, warning):
    base = ToolResult(status="error", code="TOOL_REPORTED_FAILURE", text=PAYLOAD)
    if route == "builtin":
        result = _compose_execute_result_result("fixture", base, "", warning)
    elif route == "extension":
        result = extension_dispatch._extension_completion(PAYLOAD, warning)
    else:
        monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, warning))
        monkeypatch.setattr("ouroboros.mcp_client._call_mcp_tool_result", lambda *a, **k: base)
        result = extension_dispatch._dispatch_mcp_tool_result(SimpleNamespace(), "mcp_demo", {})
    assert result.status == "error" and result.code == "TOOL_REPORTED_FAILURE"
    assert result.text == PAYLOAD + ("\n\n" + warning if warning else "")
    assert result.producer_text == (PAYLOAD if warning else None)
    assert result.host_annotations == ((warning,) if warning else ())
    assert json.loads(result.producer_text or result.text)["ok"] is False


def test_nested_notes_and_generation_stamp_keep_original_payload(monkeypatch):
    result = extension_dispatch._extension_completion(PAYLOAD, WARNING)
    result = _compose_execute_result_result("fixture", result, "route note", "")
    monkeypatch.setattr(extension_dispatch, "_dispatch_extension_tool_untagged", lambda *a: result)
    stamped = extension_dispatch._dispatch_extension_tool_result(
        SimpleNamespace(), "ext_demo", {"extension_generation": "generation", "content_hash": "hash"}, {},
    )
    assert stamped.producer_text == PAYLOAD
    assert stamped.host_annotations == (WARNING, "route note")
    assert stamped.meta["extension_generation"] == "generation"
    assert stamped.text == PAYLOAD + "\n\n" + WARNING + "\n\nroute note"


@pytest.mark.parametrize("actor", ["presence", "project", "local_readonly_subagent"])
@pytest.mark.parametrize("large", [False, True])
def test_loop_retains_parseable_source_and_annotated_review_source(tmp_path, monkeypatch, actor, large):
    repo = tmp_path / "repo"
    repo.mkdir()
    drive = tmp_path / actor / "data"
    (drive / "logs").mkdir(parents=True)
    ctx = ToolContext(
        repo_dir=repo, drive_root=drive, task_id="producer-source",
        budget_drive_root=tmp_path / "canonical",
        workspace_root=tmp_path / "project" if actor != "presence" else None,
        workspace_mode="external" if actor != "presence" else "",
        task_depth=1 if actor == "local_readonly_subagent" else 0,
    )
    ctx.task_metadata = {"task_id": ctx.task_id}
    if actor == "presence":
        ctx.task_metadata.update(_presence_turn=True, presence={"profile": {}})
    elif actor == "project":
        ctx.task_metadata["project_id"] = "project-demo"
    else:
        ctx.task_constraint = {"mode": "local_readonly_subagent"}
        ctx.task_metadata["parent_task_id"] = "parent"
    payload = json.dumps({"ok": False, "text": "雪" * (20000 if large else 1)}, ensure_ascii=False)
    result = extension_dispatch._extension_completion(payload, WARNING)
    registry = SimpleNamespace(_ctx=ctx, CODE_TOOLS=frozenset(), execute_result=lambda *a: result)
    captured = []
    monkeypatch.setattr(execution, "persist_call", lambda *a, **k: captured.append(k["payload"]) or {})
    row = execution._execute_single_tool(
        registry, {"id": "call-json", "function": {"name": "ext_demo", "arguments": "{}"}},
        drive / "logs", ctx.task_id,
    )
    messages, trace = [], {"tool_calls": []}
    # A measured 4,000-token frame: the large payload cannot be shown whole and is
    # projected head+tail with its exact annotated source; the small one stays whole.
    execution.process_tool_results([row], messages, trace, lambda *a, **k: None, registry,
                                   fit_candidate=measured_fit(window=4_000, reserve=1_000))
    recorded = trace["tool_calls"][0]
    ref = recorded["producer_source_ref"]
    exact = artifacts.read_actor_source_bytes(drive, ctx.task_id, ref)
    assert exact == payload.encode("utf-8")
    assert json.loads(exact)["ok"] is False
    assert "雪" in _read_file(ctx, **ref["read"]["arguments"])
    assert WARNING in messages[0]["content"]
    assert "PRODUCER_RESULT_SOURCE_JSON=" in messages[0]["content"]
    assert recorded["is_error"] is True
    assert captured[0]["producer_result"] == payload
    assert captured[0]["result"] == result.text
    assert captured[0]["host_annotations"] == [WARNING]
    if large:
        full_ref = recorded["result_source_ref"]
        assert artifacts.read_actor_source_bytes(drive, ctx.task_id, full_ref).decode("utf-8") == result.text
        assert full_ref != ref


def test_clean_small_result_keeps_existing_projection(tmp_path):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="clean-source")
    result = extension_dispatch._extension_completion(PAYLOAD, "")
    messages, trace = [], {"tool_calls": []}
    execution.process_tool_results(
        [{"fn_name": "ext_demo", "tool_call_id": "clean", "result": PAYLOAD,
          "tool_result": result, "is_error": True, "tool_args": {}, "args_for_log": {}}],
        messages, trace, lambda *a, **k: None, SimpleNamespace(_ctx=ctx),
    )
    assert messages[0]["content"] == PAYLOAD
    assert "producer_source_ref" not in trace["tool_calls"][0]


def test_missing_source_is_disclosed_without_hiding_warning_or_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "store_actor_source_bytes", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="source-failure")
    result = extension_dispatch._extension_completion(PAYLOAD, WARNING)
    messages, trace = [], {"tool_calls": []}
    execution.process_tool_results(
        [{"fn_name": "ext_demo", "tool_call_id": "failed", "result": result.text,
          "tool_result": result, "is_error": True, "tool_args": {}, "args_for_log": {}}],
        messages, trace, lambda *a, **k: None, SimpleNamespace(_ctx=ctx),
    )
    assert result.text in messages[0]["content"]
    assert "PRODUCER_RESULT_SOURCE_UNAVAILABLE=true" in messages[0]["content"]
    assert trace["tool_calls"][0]["is_error"] is True
    assert trace["tool_calls"][0]["producer_source_ref"] == {}
