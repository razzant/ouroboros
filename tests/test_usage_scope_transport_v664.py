"""Focused accounting integration regressions for v6.64.0."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import io
import json
import subprocess
import sys
import types
from pathlib import Path


def _scope(root: Path):
    from ouroboros.usage_accounting import UsageScope

    return UsageScope(
        drive_root=root,
        task_id="child-task",
        root_task_id="root-task",
        parent_task_id="parent-task",
        category="acceptance",
        source="task_acceptance",
        global_limit_usd=0.0,
        root_limit_usd=0.0,
    )


def test_web_search_scope_preserves_bound_root_limits(tmp_path):
    from ouroboros.tools import search
    from ouroboros.usage_accounting import usage_scope

    class Ctx:
        task_id = "wrong-task"
        task_metadata = {
            "root_task_id": "wrong-root",
            "budget_drive_root": "/wrong/root",
        }

    outer = _scope(tmp_path)
    with usage_scope(outer):
        derived = search._accounting_scope(Ctx(), "web_search.openai_responses")

    assert derived.drive_root == tmp_path
    assert derived.task_id == "child-task"
    assert derived.root_task_id == "root-task"
    assert derived.parent_task_id == "parent-task"
    assert derived.global_limit_usd == 0.0
    assert derived.root_limit_usd == 0.0
    assert derived.category == "web_search"
    assert derived.source == "web_search.openai_responses"


def test_generic_llm_attempt_keeps_semantic_scope_source(tmp_path, monkeypatch):
    from ouroboros import llm
    from ouroboros.usage_accounting import UsageScope, release_attempt, reserve_attempt, usage_scope

    monkeypatch.setenv("TOTAL_BUDGET", "10")
    semantic = UsageScope(
        drive_root=tmp_path,
        task_id="review-task",
        root_task_id="root-task",
        category="review",
        source="task_acceptance",
    )
    with usage_scope(semantic):
        request = llm._attempt_request(
            {
                "provider": "openai",
                "usage_model": "openai/gpt-5.2",
                "resolved_model": "gpt-5.2",
            },
            {"model": "gpt-5.2", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
        )
        assert request.source == "task_acceptance"
        reservation = reserve_attempt(request)
        release_attempt(reservation)

    rows = [json.loads(line) for line in (tmp_path / "state" / "usage_attempts.jsonl").read_text().splitlines()]
    reserved = next(row for row in rows if row.get("state") == "reserved")
    assert reserved["source"] == "task_acceptance"
    assert reserved["task_id"] == "review-task"
    assert reserved["root_task_id"] == "root-task"
    assert llm._attempt_request({}, {}).source == "llm.chat"


def test_claude_readonly_parent_serializes_usage_scope(tmp_path, monkeypatch):
    from ouroboros.gateways import claude_code
    from ouroboros.usage_accounting import usage_scope

    captured = {}

    class FakeProcess:
        returncode = 0

        def communicate(self, input=None, timeout=None):
            captured["payload"] = json.loads(input)
            return json.dumps({"success": True, "result_text": "ok"}), ""

    monkeypatch.setattr(claude_code.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    with usage_scope(_scope(tmp_path)):
        result = claude_code._run_readonly_out_of_process(
            prompt="audit",
            cwd=str(tmp_path),
            model="opus[1m]",
            max_turns=1,
            effort="high",
            max_budget_usd=2.5,
        )

    assert result.success
    transported = captured["payload"]["usage_scope"]
    assert transported["drive_root"] == str(tmp_path)
    assert transported["task_id"] == "child-task"
    assert transported["root_task_id"] == "root-task"
    assert transported["global_limit_usd"] == 0.0
    assert transported["root_limit_usd"] == 0.0
    assert captured["payload"]["max_budget_usd"] == 2.5


def test_claude_readonly_public_api_forwards_budget_cap(tmp_path, monkeypatch):
    from ouroboros.gateways import claude_code

    captured = {}

    def fake_out_of_process(**kwargs):
        captured.update(kwargs)
        return claude_code.ClaudeCodeResult(success=True, result_text="ok")

    monkeypatch.delenv("OUROBOROS_CLAUDE_READONLY_CHILD", raising=False)
    monkeypatch.setattr(claude_code, "_run_readonly_out_of_process", fake_out_of_process)

    result = claude_code.run_readonly(
        prompt="audit",
        cwd=str(tmp_path),
        model="opus[1m]",
        max_turns=1,
        effort="high",
        max_budget_usd=2.5,
    )

    assert result.success
    assert captured["max_budget_usd"] == 2.5


def test_claude_readonly_child_restores_usage_scope(tmp_path, monkeypatch, capsys):
    from ouroboros import process_custody
    from ouroboros.gateways import claude_code
    from ouroboros.usage_accounting import current_usage_scope

    seen = {}

    async def fake_readonly(**kwargs):
        seen["scope"] = current_usage_scope()
        seen["kwargs"] = kwargs
        return claude_code.ClaudeCodeResult(success=True, result_text="ok")

    payload = {
        "prompt": "audit",
        "cwd": str(tmp_path),
        "model": "opus[1m]",
        "max_turns": 1,
        "effort": "high",
        "max_budget_usd": 2.5,
        "usage_scope": {
            **vars(_scope(tmp_path)),
            "drive_root": str(tmp_path),
        },
    }
    monkeypatch.setattr(process_custody, "start_parent_lifeline", lambda **kwargs: None)
    monkeypatch.setattr(claude_code, "_run_readonly_async", fake_readonly)
    monkeypatch.setattr(sys, "argv", ["claude_code.py", "--readonly-child"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))

    assert claude_code._main() == 0
    capsys.readouterr()
    assert seen["scope"].task_id == "child-task"
    assert seen["scope"].root_task_id == "root-task"
    assert seen["scope"].root_limit_usd == 0.0
    assert seen["kwargs"]["max_budget_usd"] == 2.5
    assert current_usage_scope() is None


def test_claude_readonly_budget_cap_reaches_sdk_and_ledger(tmp_path, monkeypatch):
    from ouroboros.gateways import claude_code
    from ouroboros.usage_accounting import UsageScope, usage_scope

    captured = {}

    class FakeOptions:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeResultMessage:
        session_id = "readonly-budget"
        total_cost_usd = 0.25
        usage = {"input_tokens": 10, "output_tokens": 5}
        subtype = "success"

    class FakeSDKClient:
        def __init__(self, options=None):
            self.options = options

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def query(self, prompt):
            return None

        async def receive_response(self):
            yield FakeResultMessage()

    monkeypatch.setattr(claude_code, "ClaudeAgentOptions", FakeOptions)
    monkeypatch.setattr(claude_code, "ClaudeSDKClient", FakeSDKClient)
    monkeypatch.setattr(claude_code, "ResultMessage", FakeResultMessage)
    scope = UsageScope(
        drive_root=tmp_path,
        task_id="review-task",
        root_task_id="root-task",
        category="review",
        source="claude_code.readonly",
        global_limit_usd=10.0,
        root_limit_usd=5.0,
    )

    with usage_scope(scope):
        result = asyncio.run(claude_code._run_readonly_async(
            prompt="audit",
            cwd=str(tmp_path),
            model="opus[1m]",
            max_turns=1,
            effort=None,
            max_budget_usd=2.5,
        ))

    assert result.success
    assert captured["max_budget_usd"] == 2.5
    rows = [
        json.loads(line)
        for line in (tmp_path / "state" / "usage_attempts.jsonl").read_text().splitlines()
    ]
    reserved = next(
        row
        for row in rows
        if row.get("kind") == "attempt"
        and row.get("state") == "reserved"
        and row.get("source") == "claude_code.readonly"
    )
    assert reserved["reservation_upper_bound_usd"] == 2.5
    assert reserved["reservation_basis"] == "explicit_upper_bound"


def test_vlm_child_payload_carries_usage_scope(tmp_path, monkeypatch):
    from ouroboros import llm
    from ouroboros.tools import shell, vision
    from ouroboros.usage_accounting import current_usage_scope, usage_scope

    captured = {}

    class FakeLLMClient:
        def vision_query(self, **kwargs):
            captured["restored_scope"] = current_usage_scope()
            return "seen", {}

    def fake_run(cmd, **kwargs):
        captured["script"] = cmd[2]
        captured["payload"] = json.loads(Path(cmd[3]).read_text())
        output = io.StringIO()
        previous_argv = sys.argv
        try:
            sys.argv = ["vlm-child", cmd[3]]
            with contextlib.redirect_stdout(output):
                # A fresh Context simulates the real subprocess: no parent
                # contextvar exists unless the payload restoration works.
                contextvars.Context().run(
                    exec,
                    compile(cmd[2], "<vlm-child>", "exec"),
                    {"__name__": "__vlm_child_test__"},
                )
        finally:
            sys.argv = previous_argv
        return subprocess.CompletedProcess(cmd, 0, stdout=output.getvalue(), stderr="")

    monkeypatch.setattr(llm, "LLMClient", FakeLLMClient)
    monkeypatch.setattr(shell, "_tracked_subprocess_run", fake_run)
    with usage_scope(_scope(tmp_path)):
        text, usage = vision._vision_query_with_timeout(
            object(),
            prompt="inspect",
            images=[],
            model="openai/gpt-5.2",
            timeout=5,
        )

    assert text == "seen"
    assert usage == {}
    transported = captured["payload"]["_usage_scope"]
    assert transported["drive_root"] == str(tmp_path)
    assert transported["task_id"] == "child-task"
    assert transported["root_task_id"] == "root-task"
    assert transported["root_limit_usd"] == 0.0
    assert captured["restored_scope"].task_id == "child-task"
    assert captured["restored_scope"].root_task_id == "root-task"
    assert captured["restored_scope"].root_limit_usd == 0.0


def test_provider_owned_web_search_disables_sdk_retries(monkeypatch):
    from ouroboros import llm, llm_attempt

    captured = {}

    class Completions:
        def create(self, **kwargs):
            return object()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["openai"] = kwargs
            self.chat = types.SimpleNamespace(completions=Completions())

    class Messages:
        def create(self, **kwargs):
            return object()

    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured["anthropic"] = kwargs
            self.messages = Messages()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=FakeAnthropic))
    monkeypatch.setattr(llm_attempt, "execute_physical_attempt", lambda request, send: send())

    llm.openrouter_web_search_server_tool(
        api_key="test",
        model="openai/gpt-5.2",
        query="query",
        search_context_size="medium",
    )
    llm.anthropic_web_search_server_tool(
        api_key="test",
        model="claude-sonnet-4-6",
        query="query",
    )

    assert captured["openai"]["max_retries"] == 0
    assert captured["anthropic"]["max_retries"] == 0


def test_openai_responses_web_search_disables_sdk_retries(tmp_path, monkeypatch):
    from ouroboros.tools import search

    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            response = types.SimpleNamespace(usage=None)
            self.responses = types.SimpleNamespace(create=lambda **request: [
                types.SimpleNamespace(type="response.output_text.delta", delta="answer"),
                types.SimpleNamespace(type="response.completed", response=response),
            ])

    class Ctx:
        task_id = "search-task"
        task_metadata = {"budget_drive_root": str(tmp_path), "root_task_id": "root-task"}

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OUROBOROS_WEBSEARCH_BACKEND", "openai")
    monkeypatch.setattr(search, "reserve_attempt", lambda request: types.SimpleNamespace(attempt_id="attempt"))
    monkeypatch.setattr(search, "mark_dispatched", lambda reservation: None)
    monkeypatch.setattr(search, "settle_attempt", lambda *args, **kwargs: None)

    result = json.loads(search._web_search(Ctx(), "query"))
    assert result["answer"] == "answer"
    assert captured["max_retries"] == 0


def test_local_model_probe_disables_sdk_retries(monkeypatch):
    from ouroboros.local_model import LocalModelManager

    captured = {}

    class Completions:
        def create(self, **kwargs):
            if kwargs.get("tools"):
                message = types.SimpleNamespace(content="", tool_calls=[object()])
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=None)
            message = types.SimpleNamespace(content="hello", tool_calls=[])
            usage = types.SimpleNamespace(completion_tokens=1)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=usage)

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.chat = types.SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    result = LocalModelManager().test_tool_calling()

    assert result["success"] is True
    assert captured["max_retries"] == 0
