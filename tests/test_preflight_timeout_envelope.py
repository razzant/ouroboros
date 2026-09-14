"""Standalone preflight covers the test producer and existing review envelope."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('override,expected_tests', [(None, 1800), ('3600', 3600), ('5400', 5400), ('invalid', 1800)])
@pytest.mark.parametrize('task_ceiling,transport', [(21600, 2700), (300, 7200)])
def test_preflight_outer_uses_resolved_test_budget_and_existing_bounds(
    monkeypatch, override, expected_tests, task_ceiling, transport,
):
    from ouroboros import config, preflight_runner
    from ouroboros.tools import claude_advisory_review as advisory
    from ouroboros import loop_tool_execution as execution

    if override is None:
        monkeypatch.delenv('OUROBOROS_PREFLIGHT_TIMEOUT_SEC', raising=False)
    else:
        monkeypatch.setenv('OUROBOROS_PREFLIGHT_TIMEOUT_SEC', override)
    monkeypatch.setattr(config, 'get_task_abs_ceiling_sec', lambda: task_ceiling)
    monkeypatch.setattr(config, 'get_llm_transport_read_timeout_sec', lambda: transport)
    monkeypatch.setattr(advisory, 'get_finalization_grace_sec', lambda: 30)
    monkeypatch.setattr(execution, 'load_settings', lambda: {'OUROBOROS_TOOL_TIMEOUT_SEC': 600})
    expected = expected_tests + max(task_ceiling, transport + 30) + 30
    entries = {entry.name: entry for entry in advisory.get_tools()}
    tools = SimpleNamespace(get_timeout=lambda name: entries[name].timeout_sec)
    assert preflight_runner._resolve_preflight_timeout() == expected_tests
    for name in ('preflight_review', 'advisory_review'):
        assert entries[name].timeout_sec == expected
        assert execution._get_tool_timeout(tools, name) == expected
        assert name not in execution.REVIEWED_MUTATIVE_TOOLS


@pytest.mark.parametrize('name', ['preflight_review', 'advisory_review'])
def test_loop_waiter_keeps_the_derived_envelope_until_preflight_settles(tmp_path, monkeypatch, name):
    """The real tool wrapper uses the registered envelope, without waiting hours."""
    from ouroboros import config, preflight_runner
    from ouroboros.tools import claude_advisory_review as advisory
    from ouroboros import loop_tool_execution as execution
    from ouroboros.tools.tool_result import ToolResult

    monkeypatch.setattr(preflight_runner, '_resolve_preflight_timeout', lambda: 3600)
    monkeypatch.setattr(config, 'get_task_abs_ceiling_sec', lambda: 21600)
    monkeypatch.setattr(config, 'get_llm_transport_read_timeout_sec', lambda: 2700)
    monkeypatch.setattr(advisory, 'get_finalization_grace_sec', lambda: 30)
    monkeypatch.setattr(execution, 'load_settings', lambda: {})
    entries = {entry.name: entry for entry in advisory.get_tools()}
    waits, events = [], []

    def wait(future, timeout):
        waits.append(timeout)
        assert timeout >= 3600 + 21600 + 30
        return future.result(timeout=2)

    monkeypatch.setattr(execution, 'future_result', wait)
    tools = SimpleNamespace(
        CODE_TOOLS=set(),
        _ctx=SimpleNamespace(event_queue=SimpleNamespace(put_nowait=events.append), task_metadata={}),
        get_timeout=lambda tool: entries[tool].timeout_sec,
        execute_result=lambda *args: ToolResult(status='ok', code='OK', text='tests and critic settled'),
    )
    result = execution._execute_with_timeout(
        tools, {'id': 'preflight-call', 'function': {'name': name, 'arguments': '{}'}},
        tmp_path, execution._get_tool_timeout(tools, name), task_id='fixture',
    )
    assert result['result'] == 'tests and critic settled'
    assert waits == [25230]
    payloads = [event.get('data') or {} for event in events]
    assert not any(event.get('type') == 'tool_call_timeout' for event in payloads)


def test_commit_reviewed_keeps_its_existing_terminal_wait_classification():
    from ouroboros.tool_capabilities import REVIEWED_MUTATIVE_TOOLS

    assert REVIEWED_MUTATIVE_TOOLS == {'commit_reviewed', 'vcs_commit_reviewed'}
