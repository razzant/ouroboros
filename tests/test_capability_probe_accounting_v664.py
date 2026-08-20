"""Physical-attempt attribution for generative Capability Evidence probes."""

from __future__ import annotations

import json
import sys
import types


def _probe_client(monkeypatch, *, failure: BaseException | None = None):
    from ouroboros import llm

    calls: list[dict] = []

    class Completions:
        def create(self, **payload):
            calls.append(payload)
            if failure is not None:
                raise failure
            message = types.SimpleNamespace(content="OBOCANARY")
            choice = types.SimpleNamespace(message=message)
            usage = types.SimpleNamespace(prompt_tokens=123)
            return types.SimpleNamespace(choices=[choice], usage=usage)

    remote = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=Completions()),
        with_options=lambda **_kwargs: remote,
    )
    client = llm.LLMClient()
    monkeypatch.setattr(
        client,
        "_resolve_remote_target",
        lambda _model: {
            "provider": "openai",
            "resolved_model": "gpt-test",
            "usage_model": "openai/gpt-test",
            "base_url": "https://example.invalid/v1",
            "api_key": "test",
        },
    )
    monkeypatch.setattr(client, "_get_remote_client", lambda _target: remote)
    return client, calls


def test_unscoped_capability_probe_uses_stable_system_usage_scope(monkeypatch):
    from ouroboros import llm_attempt
    from ouroboros.usage_accounting import current_usage_scope

    client, provider_calls = _probe_client(monkeypatch)
    observed = []

    def fake_execute(request, send):
        observed.append((request, current_usage_scope()))
        return send()

    monkeypatch.setattr(llm_attempt, "execute_physical_attempt", fake_execute)

    result = client.probe_oversized_context("openai/gpt-test", "oversized")

    assert result["ok"] is True
    assert len(provider_calls) == 1
    assert len(observed) == 1
    request, scope = observed[0]
    assert request.source == "capability_probe"
    assert scope.task_id == "system:capability_probe"
    assert scope.root_task_id == "system:capability_probe"
    assert scope.category == "capability_probe"
    assert scope.source == "capability_probe"
    assert current_usage_scope() is None


def test_task_bound_capability_probe_inherits_task_and_budget_scope(monkeypatch, tmp_path):
    from ouroboros import llm_attempt
    from ouroboros.usage_accounting import UsageScope, current_usage_scope, usage_scope

    client, provider_calls = _probe_client(monkeypatch, failure=RuntimeError("HTTP 400 context overflow"))
    observed = []

    def fake_execute(request, send):
        observed.append((request, current_usage_scope()))
        return send()

    monkeypatch.setattr(llm_attempt, "execute_physical_attempt", fake_execute)
    task_scope = UsageScope(
        drive_root=tmp_path,
        task_id="probe-task",
        root_task_id="probe-root",
        parent_task_id="probe-parent",
        category="planning",
        source="planning_panel",
        global_limit_usd=17.0,
        root_limit_usd=3.0,
    )

    with usage_scope(task_scope):
        result = client.probe_oversized_context("openai/gpt-test", "oversized")

    assert result["ok"] is False
    assert len(provider_calls) == 1
    assert len(observed) == 1
    request, scope = observed[0]
    assert request.source == "capability_probe"
    assert scope == task_scope
    assert scope.task_id == "probe-task"
    assert scope.root_task_id == "probe-root"
    assert scope.parent_task_id == "probe-parent"
    assert scope.global_limit_usd == 17.0
    assert scope.root_limit_usd == 3.0
    assert current_usage_scope() is None


def test_local_model_capability_checks_are_accounted_and_inherit_task_scope(
    monkeypatch,
    tmp_path,
):
    from ouroboros.local_model import LocalModelManager
    from ouroboros.usage_accounting import UsageScope, current_usage_scope, usage_scope

    provider_calls = []

    class Completions:
        def create(self, **payload):
            provider_calls.append(payload)
            if payload.get("tools"):
                message = types.SimpleNamespace(content="", tool_calls=[object()])
            else:
                message = types.SimpleNamespace(content="hello", tool_calls=[])
            usage = types.SimpleNamespace(completion_tokens=1)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=message)],
                usage=usage,
            )

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = types.SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "10")

    manager = LocalModelManager()
    system_result = manager.test_tool_calling()

    assert system_result["success"] is True
    assert len(provider_calls) == 2

    task_scope = UsageScope(
        drive_root=tmp_path,
        task_id="local-probe-task",
        root_task_id="local-probe-root",
        parent_task_id="local-probe-parent",
        category="task",
        source="main_loop",
        global_limit_usd=11.0,
        root_limit_usd=2.0,
    )
    provider_calls.clear()
    with usage_scope(task_scope):
        task_result = manager.test_tool_calling()

    assert task_result["success"] is True
    assert len(provider_calls) == 2
    assert current_usage_scope() is None

    rows = [
        json.loads(line)
        for line in (tmp_path / "state" / "usage_attempts.jsonl").read_text().splitlines()
    ]
    settled = [row for row in rows if row.get("state") == "settled"]
    assert len(settled) == 4
    for row in settled[:2]:
        assert row["task_id"] == "system:capability_probe"
        assert row["root_task_id"] == "system:capability_probe"
        assert row["category"] == "capability_probe"
        assert row["source"] == "capability_probe.local_model"
        assert row["cost_usd"] == 0.0
        assert row["cost_final"] is True
    for row in settled[2:]:
        assert row["task_id"] == "local-probe-task"
        assert row["root_task_id"] == "local-probe-root"
        assert row["parent_task_id"] == "local-probe-parent"
        assert row["category"] == "task"
        assert row["source"] == "capability_probe.local_model"
