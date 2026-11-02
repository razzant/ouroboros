"""Process capability and independent Supervisor evidence on the real dispatch path."""
from __future__ import annotations

import json
import sys

import pytest

from ouroboros import config, safety
from ouroboros.tools.registry import ToolContext, ToolRegistry

pytestmark = pytest.mark.serial


def context(tmp_path, monkeypatch, mode, coverage):
    repo, data, work = (tmp_path / name for name in ("repo", "data", "work"))
    for path in (repo, data, work):
        path.mkdir()
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", mode)
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", coverage)
    monkeypatch.setattr(config, "get_runtime_mode", lambda: mode)
    monkeypatch.setattr(safety, "get_runtime_mode", lambda: mode)
    monkeypatch.setattr(safety, "get_safety_mode", lambda kind="owner": coverage)
    ctx = ToolContext(repo_dir=repo, system_repo_dir=repo, drive_root=data,
                      workspace_root=work, workspace_mode="external", task_id="agency")
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(ctx)
    return ctx, registry, work


@pytest.mark.parametrize("mode", ["light", "advanced", "cyber_pro"])
@pytest.mark.parametrize("coverage", ["full", "light", "off"])
def test_one_process_effect_under_selected_supervisor(tmp_path, monkeypatch, mode, coverage):
    ctx, registry, work = context(tmp_path, monkeypatch, mode, coverage)
    checks = []

    def assessment(name, arguments, messages, actor, binding=None):
        checks.append((name, arguments, binding))
        return False, "Independent DANGEROUS finding: fixture criticism"

    monkeypatch.setattr(safety, "_run_llm_check", assessment)
    script = "from pathlib import Path; p=Path('once.txt'); p.write_text(p.read_text()+'x' if p.exists() else 'x')"
    result = registry.execute_result("run_command", {
        "cmd": [sys.executable, "-c", script], "cwd": str(work), "outputs": ["once.txt"],
    })
    assert len(checks) == (1 if coverage == "full" else 0)
    permitted = coverage != "full" or mode == "cyber_pro"
    assert (work / "once.txt").exists() is permitted
    if permitted:
        assert (work / "once.txt").read_text() == "x"
        assert result.status == "ok", result.text
    else:
        assert result.status == "blocked"
    if mode == "cyber_pro" and coverage == "full":
        assert "SAFETY_ADVICE" in result.text
        events = [json.loads(line) for line in (ctx.drive_logs() / "events.jsonl").read_text().splitlines()]
        advice = next(row for row in events if row["type"] == "safety_advisory")
        assert advice["assessment_allowed"] is False
        assert advice["execution_allowed"] is True
        assert "DANGEROUS" in advice["assessment"]


@pytest.mark.parametrize("script", [
    "print('OUROBOROS_CONTEXT_MODE settings.json OUROBOROS_REVIEW_ENFORCEMENT')",
    "import io; f=io.StringIO(); f.write('data/state/skills/grants.json'); print(f.getvalue())",
    "import ast; print(ast.dump(ast.parse('OUROBOROS_SAFETY_MODE = 1')))",
    "import importlib; print(importlib.import_module('json').dumps({'content':'ok'}))",
    "print(compile('1+1', '<example>', 'eval').co_name)",
    "import sqlite3; uri='file:fixture?mode=memory&cache=shared'; c=sqlite3.connect(uri,uri=True); print(c.execute('select 1').fetchone())",
])
def test_useful_interpreter_operations_are_not_word_denials(tmp_path, monkeypatch, script):
    _ctx, registry, work = context(tmp_path, monkeypatch, "light", "off")
    result = registry.execute_result("run_command", {"cmd": [sys.executable, "-c", script], "cwd": str(work)})
    assert result.status == "ok", result.text
    assert "exit_code=0" in result.text


def test_safety_keeps_full_task_sources_after_transcript_compaction(tmp_path, monkeypatch):
    ctx, _registry, work = context(tmp_path, monkeypatch, "advanced", "full")
    origin = "Original authorization: " + "context " * 1500 + "exact final instruction."
    reply = "Later answer: " + "detail " * 1000 + "permission applies to this repository."
    ctx._owner_directives = [{"source": "initial_user", "content": origin},
                             {"source": "owner_mailbox", "msg_id": "answer-1", "content": reply}]
    ctx.task_contract = {"objective": "Create the requested private repository"}
    from ouroboros.tool_access import build_resolved_resource_binding
    binding = build_resolved_resource_binding(ctx, operation="shell", process_cwd=str(work))
    prompt = safety._build_check_prompt("run_command", {"cmd": ["gh", "repo", "create"]},
        [{"role": "user", "content": "compacted conversation"}], ctx=ctx, resolved_binding=binding)
    assert origin in prompt and reply in prompt
    assert "answer-1" in prompt
    source_json = prompt.split("Host evidence (provenance is not consent):\n", 1)[1]
    facts, _ = json.JSONDecoder().raw_decode(source_json)
    assert facts["subject"]["physical_target"]["target_path"] == repr(work)
    assert "Create the requested private repository" not in prompt  # task_contract is not retained source evidence


@pytest.mark.parametrize("mode", ["advanced", "cyber_pro"])
def test_supervisor_backend_failure_keeps_original_evidence(tmp_path, monkeypatch, mode):
    ctx, _registry, _work = context(tmp_path, monkeypatch, mode, "full")
    def failed(*args, **kwargs):
        raise RuntimeError("fixture backend unavailable")
    monkeypatch.setattr(safety, "_run_llm_check", failed)
    allowed, message = safety.check_safety("mcp_fixture__run", {}, ctx=ctx)
    assert allowed == (mode == "cyber_pro")
    assert "SAFETY_UNAVAILABLE" in message and "backend unavailable" in message
    if mode == "cyber_pro":
        assert "SAFETY_ADVICE" in message


def test_cyber_does_not_swallow_owner_stop(tmp_path, monkeypatch):
    ctx, _registry, _work = context(tmp_path, monkeypatch, "cyber_pro", "full")
    from ouroboros.model_wait import ModelWaitInterrupted
    def stopped(*args, **kwargs):
        raise ModelWaitInterrupted("owner_stopped")
    monkeypatch.setattr(safety, "_run_llm_check", stopped)
    with pytest.raises(ModelWaitInterrupted):
        safety.check_safety("mcp_fixture__run", {}, ctx=ctx)


@pytest.mark.parametrize("mode", ["advanced", "cyber_pro"])
def test_cyber_acting_uses_registered_catalog_without_inventing_tools(tmp_path, monkeypatch, mode):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolEntry
    ctx, registry, work = context(tmp_path, monkeypatch, mode, "off")
    ctx.task_constraint = TaskConstraint(mode="acting_subagent", surface="external_workspace", write_root=str(work))
    calls = []
    entry = ToolEntry(name="fixture_capability", schema={"name": "fixture_capability",
        "description": "One scoped capability", "parameters": {"type": "object", "properties": {}}},
        handler=lambda ctx: calls.append("effect") or "actual effect")
    registry.register(entry)
    visible = "fixture_capability" in registry.available_tools()
    assert visible is (mode == "cyber_pro")
    assert (registry.get_schema_by_name("fixture_capability") is not None) is visible
    result = registry.execute_result("fixture_capability", {})
    assert calls == (["effect"] if visible else [])
    assert result.status == ("ok" if visible else "blocked")
    assert registry.get_schema_by_name("absent_capability") is None
    ctx.task_contract = {"disabled_tools": ["fixture_capability"]}
    assert registry.get_schema_by_name("fixture_capability") is None
    assert registry.execute_result("fixture_capability", {}).status == "blocked"

    assert len(calls) == int(visible)
    ctx.task_contract = {}
    ctx.task_constraint = TaskConstraint(mode="local_readonly_subagent")
    assert registry.get_schema_by_name("fixture_capability") is None
    assert registry.execute_result("fixture_capability", {}).status == "blocked"


def test_cyber_external_write_is_not_vetoed_by_other_runtime_update(tmp_path, monkeypatch):
    ctx, registry, work = context(tmp_path, monkeypatch, "cyber_pro", "off")
    monkeypatch.setattr("supervisor.update_merge.managed_assisted_tx_for", lambda *args: ({}, True))
    result = registry.execute_result("write_file", {
        "root": "active_workspace", "path": "output.txt", "content": "actual external result",
    })
    assert result.status == "ok", result.text
    assert (work / "output.txt").read_text() == "actual external result"
