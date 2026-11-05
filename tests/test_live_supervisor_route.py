"""A speaking parent delegates cognition to the room's ordinary text route."""
import pytest

from tests.test_available_subagents_runtime import _session_row, _settings
from tests.test_supervisor_route import cognition_row
from tests.test_supervisor_route_flow import root_context, schedule


@pytest.mark.parametrize('call', [False, True])
@pytest.mark.parametrize('explicit', [False, True])
def test_call_delegation_keeps_speech_parent_and_text_supervisor(tmp_path, monkeypatch, call, explicit):
    from ouroboros.subagent_runtime import subagent_supervision_fact_line

    settings = {**_settings(_session_row(), cognition_row()), 'OUROBOROS_MODEL': 'openai::room-text'}
    ctx = root_context(tmp_path, settings, monkeypatch)
    if call:
        ctx.task_metadata['execution_transport'] = {'kind': 'live_conversation', 'model': 'gpt-realtime-2.1'}
        ctx.task_metadata['model'] = ctx.active_model = 'openai::gpt-realtime-2.1'
    parent_model = ctx.active_model
    result = schedule(ctx, **({'supervisor': 'cognition'} if explicit else {}))
    assert result.startswith('Subagent request queued'), result
    event = next(e for e in ctx.pending_events if e['type'] == 'schedule_subagent')
    route = event['parent_cognitive_route']
    expected = 'claudexor::codex=gpt-5.6-sol' if explicit else 'openai::room-text' if call else parent_model
    assert route['model'] == expected, route
    assert route['chosen_by'] == ('delegator' if explicit else 'inherited')
    assert ctx.active_model == parent_model, 'Delegation must not retarget the talking parent.'
    fact = subagent_supervision_fact_line(ctx=ctx, catalog={'rows': [{'route_class': 'Agent session'}]})
    assert ('openai::room-text' if call else parent_model) in fact
    if call:
        assert 'gpt-realtime' not in fact


def test_preparation_fact_uses_text_cognition_before_a_call_host_exists(monkeypatch):
    from ouroboros.subagent_runtime import current_cognitive_route_and_billing
    monkeypatch.setenv('OUROBOROS_MODEL', 'openai::room-text')
    task = {'model': 'openai::gpt-realtime-2.1',
            'metadata': {'execution_transport': {'kind': 'live_conversation'}}}
    assert current_cognitive_route_and_billing(task)[0] == 'openai::room-text'
