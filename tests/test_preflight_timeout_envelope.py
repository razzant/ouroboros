"""The preflight names settle under review_change's envelope: one action, one bound.

No tests run inside ``preflight_review`` any more (``commit_reviewed`` runs them), so
the old test-budget term left the envelope with the pipeline it belonged to.
"""
from types import SimpleNamespace

import pytest


def _patch_bounds(monkeypatch, *, task_ceiling, transport):
    from ouroboros.tools import review_change

    monkeypatch.setattr(review_change, 'get_task_abs_ceiling_sec', lambda: task_ceiling)
    monkeypatch.setattr(review_change, 'get_llm_transport_read_timeout_sec', lambda: transport)
    monkeypatch.setattr(review_change, 'get_finalization_grace_sec', lambda: 30)


@pytest.mark.parametrize('task_ceiling,transport', [(21600, 2700), (300, 7200)])
def test_preflight_names_take_review_changes_envelope(monkeypatch, task_ceiling, transport):
    from ouroboros import loop_tool_execution as execution
    from ouroboros.tools import preflight_review, review_change

    _patch_bounds(monkeypatch, task_ceiling=task_ceiling, transport=transport)
    monkeypatch.setattr(execution, 'load_settings', lambda: {'OUROBOROS_TOOL_TIMEOUT_SEC': 600})
    expected = max(task_ceiling, transport + 30) + 30
    (review_entry,) = review_change.get_tools()
    entries = {entry.name: entry for entry in preflight_review.get_tools()}
    tools = SimpleNamespace(get_timeout=lambda name: entries[name].timeout_sec)
    assert review_entry.timeout_sec == expected
    for name in ('preflight_review', 'advisory_review'):
        assert entries[name].timeout_sec == expected
        assert execution._get_tool_timeout(tools, name) == expected
        assert name not in execution.REVIEWED_MUTATIVE_TOOLS


@pytest.mark.parametrize('name', ['preflight_review', 'advisory_review'])
def test_loop_waiter_keeps_the_envelope_until_the_look_settles(tmp_path, monkeypatch, name):
    """The real tool wrapper waits on the registered envelope, without waiting hours."""
    from ouroboros import loop_tool_execution as execution
    from ouroboros.tools import preflight_review
    from ouroboros.tools.tool_result import ToolResult

    _patch_bounds(monkeypatch, task_ceiling=21600, transport=2700)
    monkeypatch.setattr(execution, 'load_settings', lambda: {})
    entries = {entry.name: entry for entry in preflight_review.get_tools()}
    waits, events = [], []

    def wait(future, timeout):
        waits.append(timeout)
        return future.result(timeout=2)

    monkeypatch.setattr(execution, 'future_result', wait)
    tools = SimpleNamespace(
        CODE_TOOLS=set(),
        _ctx=SimpleNamespace(event_queue=SimpleNamespace(put_nowait=events.append), task_metadata={}),
        get_timeout=lambda tool: entries[tool].timeout_sec,
        execute_result=lambda *args: ToolResult(status='ok', code='OK', text='the look settled'),
    )
    result = execution._execute_with_timeout(
        tools, {'id': 'preflight-call', 'function': {'name': name, 'arguments': '{}'}},
        tmp_path, execution._get_tool_timeout(tools, name), task_id='fixture',
    )
    assert result['result'] == 'the look settled'
    assert waits == [21630]
    payloads = [event.get('data') or {} for event in events]
    assert not any(event.get('type') == 'tool_call_timeout' for event in payloads)


def test_commit_reviewed_keeps_its_existing_terminal_wait_classification():
    from ouroboros.tool_capabilities import REVIEWED_MUTATIVE_TOOLS

    assert REVIEWED_MUTATIVE_TOOLS == {'commit_reviewed', 'vcs_commit_reviewed'}
