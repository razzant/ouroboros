"""A call owes the target's Low-start line at its first adopted connection only, once per Low episode."""
from types import SimpleNamespace

import pytest

from ouroboros.live_context_view import record_session_view
from tests.test_memory_low_start import _plan, _room, sent as sent


@pytest.mark.parametrize('room,expected', [
    (_room(), 1), (_room('route'), 0), (_room('held'), 0),
    (_room(zero_need=1_100_000), 0), (None, 0),
])
def test_adopted_call_reports_a_memory_lowered_start_once(tmp_path, sent, room, expected):
    plan = _plan(room)
    ctx = SimpleNamespace(drive_root=tmp_path, task_metadata={}, task_id='call-low', task_attempt=2,
                          context_fit_plan=plan, active_context_mode='low', emit_progress_fn=None)
    host = SimpleNamespace(ctx=ctx, plan=plan, messages=[{'role': 'system', 'content': 'Core.'}],
                           schemas=[], event_queue=None, _session_opened=False)
    evidence = {'fit': {'measurement': {}}, 'work_room': {}, 'wire_sha256': 'first-wire'}
    record_session_view(host, evidence, 'first-manifest')
    assert len(sent) == expected and host._session_opened
    if expected:
        root, event = sent[0]
        assert root == tmp_path and event['chat_id'] == 7 and event['task_id'] == ''
        assert event['delivery_id'] == 'low_start:call-low:2'
        assert event['system_type'] == 'low_start_notice' and 'started in Low' in event['text']
    # Even if later fits are crowded, renewals never invent another start notice.
    host.plan = ctx.context_fit_plan = _plan(_room())
    for n in range(3):
        record_session_view(host, evidence, f'renewed-manifest-{n}')
    assert len(sent) == expected
    # Later calls: Low while an episode holds adds nothing, one that holds Max closes it, Low again opens a new one.
    for task, room in (('call-b', _room()), ('call-c', _room('held')), ('call-d', _room())):
        later = SimpleNamespace(**{**vars(host), 'plan': _plan(room), '_session_opened': False,
                                   'ctx': SimpleNamespace(**{**vars(ctx), 'task_id': task})})
        record_session_view(later, evidence, f'{task}-manifest')
    assert [event['delivery_id'] for _root, event in sent] == ['low_start:call-low:2'] * expected + (
        [] if expected else ['low_start:call-b:2']) + ['low_start:call-d:2']


@pytest.mark.parametrize('manifest, expected', [
    ({'path': 'calls/first.json', 'chronicle_lead_rule': 1}, 1), ({'path': 'calls/first.json'}, None),
    ('first-manifest', None), (None, None)])
def test_an_adopted_call_reads_the_lead_fact_only_from_a_send_seal_reference(tmp_path, sent, manifest, expected):
    """The send seal's manifest reference carries the delivered lead fact; an opaque or older reference names none
    and is kept as the call's physical source reference, never a reason to fail the adopted connection."""
    plan = _plan(_room('held'))
    ctx = SimpleNamespace(drive_root=tmp_path, task_metadata={}, task_id='call-lead', task_attempt=1,
                          context_fit_plan=plan, active_context_mode='max', emit_progress_fn=None)
    host = SimpleNamespace(ctx=ctx, plan=plan, messages=[{'role': 'system', 'content': 'Core.'}],
                           schemas=[], event_queue=None, _session_opened=False)
    record_session_view(host, {'fit': {'measurement': {}}, 'work_room': {}, 'wire_sha256': 'wire'}, manifest)
    observation = ctx._last_context_observation
    assert observation['chronicle_lead_rule'] == expected and observation['physical_source_ref'] == manifest
